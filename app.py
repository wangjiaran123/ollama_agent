#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
from flask_socketio import SocketIO, emit
import requests
import openai
from werkzeug.utils import secure_filename
import aiohttp
from sandbox_executor import sandbox_executor, SandboxResult

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 配置
class Config:
    def __init__(self):
        self.UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
        self.ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'mp3', 'wav', 'mp4', 'avi', 'mov'}
        self.MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

config = Config()

# 确保上传目录存在
os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)

class ModelManager:
    """模型管理器 - 支持本地Ollama和外部API"""
    
    def __init__(self):
        self.OLLAMA_BASE_URL = "http://localhost:11434"
        self.ollama_models = []
        self.external_api_key = None
        self.external_api_base = None
        self.current_model = None
        self.load_config()
    
    def load_config(self):
        """加载配置"""
        config_file = os.path.join(os.path.dirname(__file__), 'config.json')
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.external_api_key = data.get('external_api_key')
                    self.external_api_base = data.get('external_api_base')
                    self.ollama_models = data.get('ollama_models', [])
                    self.current_model = data.get('current_model')
            except Exception as e:
                logger.error(f"加载配置文件失败: {e}")
    
    def save_config(self):
        """保存配置"""
        config_file = os.path.join(os.path.dirname(__file__), 'config.json')
        data = {
            'external_api_key': self.external_api_key,
            'external_api_base': self.external_api_base,
            'ollama_models': self.ollama_models,
            'current_model': self.current_model
        }
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")
    
    def get_ollama_models(self) -> List[Dict]:
        """获取Ollama模型列表"""
        try:
            response = requests.get(f"{self.OLLAMA_BASE_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                models = [{'name': model['name'], 'size': model.get('size', 0)} 
                         for model in models_data.get('models', [])]
                return models
            return []
        except Exception as e:
            logger.error(f"获取Ollama模型失败: {e}")
            return []
    
    def test_ollama_connection(self) -> bool:
        """测试Ollama连接"""
        try:
            response = requests.get(f"{self.OLLAMA_BASE_URL}/api/tags", timeout=3)
            return response.status_code == 200
        except:
            return False
    
    def is_vision_model(self, model_name: str) -> bool:
        """检查是否为视觉模型"""
        vision_models = ['qwen2.5-vl', 'qwen2.5vl', 'moondream', 'llava-llama3', 'llava-v1.6']
        return any(vision in model_name.lower() for vision in vision_models)
    
    def is_audio_model(self, model_name: str) -> bool:
        """检查是否为音频模型"""
        audio_models = ['whisper', 'wav2vec2', 'speech']
        return any(audio in model_name.lower() for audio in audio_models)

class ChatManager:
    """对话管理器"""
    
    def __init__(self):
        self.conversations = {}  # {session_id: [{role, content, timestamp}]}
        self.thinking_enabled = True
    
    def add_message(self, session_id: str, role: str, content: str, metadata: Dict = None):
        """添加消息到对话"""
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self.conversations[session_id].append(message)
    
    def get_conversation(self, session_id: str) -> List[Dict]:
        """获取对话历史"""
        return self.conversations.get(session_id, [])
    
    def clear_conversation(self, session_id: str):
        """清空对话"""
        self.conversations[session_id] = []
    
    def set_thinking_enabled(self, enabled: bool):
        """设置思考链开关"""
        self.thinking_enabled = enabled

class OllamaAgent:
    """Ollama Agent 核心类"""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.chat_manager = ChatManager()
    
    async def generate_response(self, session_id: str, message: str, model_name: str = None, 
                              stream: bool = True, thinking: bool = True) -> Dict:
        """生成回复"""
        try:
            if not model_name:
                model_name = self.model_manager.current_model or "llama3.1"
            
            # 添加用户消息到对话历史
            self.chat_manager.add_message(session_id, "user", message)
            
            # 获取对话历史
            conversation = self.chat_manager.get_conversation(session_id)
            
            # 构建消息格式
            messages = []
            for msg in conversation[-10:]:  # 只保留最近10条消息
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # 检查是否为视觉模型
            is_vision = self.model_manager.is_vision_model(model_name)
            is_audio = self.model_manager.is_audio_model(model_name)
            
            # 根据模型类型选择调用方式
            if is_vision:
                return await self._call_vision_model(messages, model_name, stream, thinking)
            elif is_audio:
                return await self._call_audio_model(messages, model_name, stream, thinking)
            else:
                return await self._call_text_model(messages, model_name, stream, thinking)
                
        except Exception as e:
            logger.error(f"生成回复失败: {e}")
            return {"error": str(e)}
    
    async def _call_text_model(self, messages: List[Dict], model: str, stream: bool, thinking: bool) -> Dict:
        """调用文本模型"""
        try:
            # 尝试本地Ollama
            if self.model_manager.test_ollama_connection():
                return await self._call_ollama(messages, model, stream, thinking)
            # 尝试外部API
            elif self.model_manager.external_api_key:
                return await self._call_external_api(messages, model, stream, thinking)
            else:
                return {"error": "请配置Ollama服务或外部API"}
        except Exception as e:
            return {"error": f"文本模型调用失败: {str(e)}"}
    
    async def _call_ollama(self, messages: List[Dict], model: str, stream: bool, thinking: bool) -> Dict:
        """调用本地Ollama"""
        try:
            url = f"{self.model_manager.OLLAMA_BASE_URL}/api/chat"
            payload = {
                "model": model,
                "messages": messages,
                "stream": stream,
                "options": {"temperature": 0.7, "top_p": 0.9}
            }
            
            if not thinking:
                payload["options"]["stop"] = ["<think>", "</think>"]
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        if stream:
                            return self._stream_ollama_response(response)
                        else:
                            result = await response.json()
                            return {"content": result.get("message", {}).get("content", "")}
                    else:
                        return {"error": f"Ollama API错误: {response.status}"}
        except Exception as e:
            return {"error": f"Ollama调用失败: {str(e)}"}
    
    def _stream_ollama_response(self, response):
        """处理Ollama流式响应"""
        async def stream_generator():
            async for line in response.content:
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        if 'message' in data:
                            yield data['message']['content']
                        if data.get('done', False):
                            break
                    except json.JSONDecodeError:
                        continue
        return stream_generator()
    
    async def _collect_stream_response(self, response_generator):
        """收集流式响应到列表"""
        chunks = []
        async for chunk in response_generator:
            chunks.append(chunk)
        return chunks
    
    async def _call_external_api(self, messages: List[Dict], model: str, stream: bool, thinking: bool) -> Dict:
        """调用外部API"""
        try:
            client = openai.OpenAI(
                api_key=self.model_manager.external_api_key,
                base_url=self.model_manager.external_api_base
            )
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=stream,
                temperature=0.7,
                top_p=0.9
            )
            
            if stream:
                return self._stream_external_response(response)
            else:
                return {"content": response.choices[0].message.content}
        except Exception as e:
            return {"error": f"外部API调用失败: {str(e)}"}
    
    def _stream_external_response(self, response):
        """处理外部API流式响应"""
        def stream_generator():
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        return stream_generator()
    
    async def _call_vision_model(self, messages: List[Dict], model: str, stream: bool, thinking: bool) -> Dict:
        """调用视觉模型"""
        try:
            # 对于视觉模型，我们需要特殊处理图像数据
            if self.model_manager.test_ollama_connection():
                return await self._call_ollama_vision(messages, model, stream, thinking)
            else:
                return {"error": "Ollama服务不可用，无法使用视觉模型"}
        except Exception as e:
            logger.error(f"视觉模型调用失败: {e}")
            return {"error": f"视觉模型调用失败: {str(e)}"}
    
    async def _call_ollama_vision(self, messages: List[Dict], model: str, stream: bool, thinking: bool) -> Dict:
        """调用Ollama视觉模型"""
        try:
            # 检查消息中是否包含图像数据
            image_data = None
            text_content = ""
            
            # 提取文本内容和图像数据
            for message in messages:
                if message["role"] == "user":
                    content = message["content"]
                    # 这里假设前端会将图像数据以特定格式发送
                    # 例如: "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
                    if isinstance(content, str) and content.startswith("data:image"):
                        image_data = content
                    else:
                        text_content = content
            
            # 构建载荷
            if image_data:
                # 对于qwen2.5-vl等需要使用/api/generate端点的模型
                # 需要将图像数据放在images字段中
                payload = {
                    "model": model,
                    "prompt": text_content,
                    "stream": stream,
                    "images": [image_data.split(",", 1)[1]]  # 提取base64数据部分
                }
                url = f"{self.model_manager.OLLAMA_BASE_URL}/api/generate"
            else:
                # 没有图像数据时，使用常规的/api/chat端点
                payload = {
                    "model": model,
                    "messages": messages,
                    "stream": stream
                }
                url = f"{self.model_manager.OLLAMA_BASE_URL}/api/chat"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=30) as response:
                    if response.status == 200:
                        if stream:
                            return await self._handle_stream_response(response, image_data is not None)
                        else:
                            result = await response.json()
                            if image_data:
                                return {"content": result.get("response", "")}
                            else:
                                return {"content": result.get("message", {}).get("content", "")}
                    else:
                        return {"error": f"Ollama视觉API错误: {response.status}"}
        except Exception as e:
            logger.error(f"Ollama视觉模型调用失败: {e}")
            return {"error": str(e)}
    
    async def _handle_stream_response(self, response, is_vision: bool = False) -> Dict:
        """处理流式响应"""
        async def stream_generator():
            async for line in response.content:
                if line:
                    try:
                        decoded_line = line.decode('utf-8').strip()
                        if decoded_line.startswith('data:'):
                            data = json.loads(decoded_line[5:])
                            if is_vision:
                                content = data.get('response', '')
                            else:
                                content = data.get('message', {}).get('content', '')
                            if content:
                                yield f"data: {json.dumps({'content': content})}\n\n"
                    except Exception as e:
                        logger.error(f"流式响应处理错误: {e}")
                        continue
        
        return {"stream": stream_generator()}
    
    async def _call_audio_model(self, messages: List[Dict], model: str, stream: bool, thinking: bool) -> Dict:
        """调用音频模型"""
        return await self._call_text_model(messages, model, stream, thinking)

# Flask应用
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
socketio = SocketIO(app, cors_allowed_origins="*")

# 全局实例
agent = OllamaAgent()

# WebSocket事件处理
@socketio.on('connect')
def handle_connect():
    """客户端连接"""
    emit('connected', {'data': 'Connected to Ollama Agent'})

@socketio.on('disconnect')
def handle_disconnect():
    """客户端断开连接"""
    print('Client disconnected')

@socketio.on('send_message')
def handle_message(data):
    """处理消息 - 修复版本"""
    session_id = data.get('session_id', 'default')
    message = data.get('message', '')
    model = data.get('model')
    thinking = data.get('thinking', True)
    image_data = data.get('image')  # 获取图像数据
    
    # 捕获客户端会话ID
    client_sid = request.sid
    
    # 使用SocketIO的background_task来正确处理request context
    try:
        # 立即发送开始消息，指定room为client_sid
        emit('message_start', {'model': model}, room=client_sid)
        
        def generate_response():
            # 在后台线程中创建应用上下文
            with app.app_context():
                try:
                    # 检查是否为视觉模型
                    is_vision = agent.model_manager.is_vision_model(model)
                    
                    if is_vision and image_data:
                        # 处理视觉模型调用
                        # 构建包含图像数据的消息
                        messages = [{"role": "user", "content": message}]
                        
                        # 对于视觉模型，需要使用专门的端点和格式
                        if "qwen2.5-vl" in model.lower():
                            # qwen2.5-vl系列模型使用/api/generate端点
                            url = f"{agent.model_manager.OLLAMA_BASE_URL}/api/generate"
                            payload = {
                                "model": model,
                                "prompt": message,
                                "stream": False,
                                "images": [image_data.split(",", 1)[1]]  # 提取base64数据部分
                            }
                        else:
                            # 其他视觉模型使用/api/chat端点
                            url = f"{agent.model_manager.OLLAMA_BASE_URL}/api/chat"
                            # 将图像数据添加到消息中
                            messages[0]["content"] = f"data:image/jpeg;base64,{image_data.split(',', 1)[1]} {message}"
                            payload = {
                                "model": model,
                                "messages": messages,
                                "stream": False,
                                "options": {"temperature": 0.7, "top_p": 0.9}
                            }
                        
                        response = requests.post(url, json=payload, timeout=30)
                    else:
                        # 处理普通文本模型调用
                        url = f"{agent.model_manager.OLLAMA_BASE_URL}/api/chat"
                        payload = {
                            "model": model,
                            "messages": [{"role": "user", "content": message}],
                            "stream": False,
                            "options": {"temperature": 0.7, "top_p": 0.9}
                        }
                        
                        response = requests.post(url, json=payload, timeout=30)
                    
                    if response.status_code == 200:
                        result = response.json()
                        if is_vision and "qwen2.5-vl" in model.lower():
                            content = result.get("response", "")
                        else:
                            content = result.get("message", {}).get("content", "")
                        socketio.emit('message_response', {'response': content}, room=client_sid)
                    else:
                        # 处理API调用失败的情况
                        if is_vision and image_data:
                            error_msg = f'视觉模型调用失败，API返回状态码: {response.status_code}。请检查模型是否正确加载或Ollama服务是否正常运行。'
                        else:
                            error_msg = f'Ollama API错误: {response.status_code}'
                        socketio.emit('error', {'error': error_msg}, room=client_sid)
                        
                except Exception as e:
                    # 检查是否为视觉模型调用失败
                    is_vision = agent.model_manager.is_vision_model(model)
                    if is_vision and image_data:
                        error_msg = f'视觉模型调用失败，请检查模型是否正确加载或图片格式是否支持。错误详情: {str(e)}'
                    else:
                        error_msg = f'处理失败: {str(e)}'
                    socketio.emit('error', {'error': error_msg}, room=client_sid)
        
        # 使用SocketIO的background_task，它会自动处理request context
        socketio.start_background_task(generate_response)
        
    except Exception as e:
        emit('error', {'error': f'消息处理失败: {str(e)}'}, room=client_sid)

# HTTP路由
@app.route('/')
def index():
    """主页"""
    return send_from_directory(os.path.join(os.path.dirname(__file__), 'static'), 'index.html')

@app.route('/api/models')
def get_models():
    """获取模型列表"""
    try:
        ollama_models = agent.model_manager.get_ollama_models()
        
        # 添加视觉和音频标识
        for model in ollama_models:
            model['is_vision'] = agent.model_manager.is_vision_model(model['name'])
            model['is_audio'] = agent.model_manager.is_audio_model(model['name'])
        
        return jsonify({
            'success': True,
            'models': ollama_models,
            'ollama_connected': agent.model_manager.test_ollama_connection()
        })
    except Exception as e:
        logger.error(f"获取模型列表失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'models': [],
            'ollama_connected': False
        })

@app.route('/api/config', methods=['GET', 'POST'])
def config_api():
    """配置API"""
    if request.method == 'GET':
        return jsonify({
            'external_api_key': agent.model_manager.external_api_key,
            'external_api_base': agent.model_manager.external_api_base,
            'current_model': agent.model_manager.current_model,
            'thinking_enabled': agent.chat_manager.thinking_enabled
        })
    else:
        try:
            data = request.json
            agent.model_manager.external_api_key = data.get('external_api_key')
            agent.model_manager.external_api_base = data.get('external_api_base')
            agent.model_manager.current_model = data.get('current_model')
            agent.chat_manager.set_thinking_enabled(data.get('thinking_enabled', True))
            
            agent.model_manager.save_config()
            
            return jsonify({'success': True})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})

@app.route('/api/chat/history/<session_id>')
def get_chat_history(session_id):
    """获取聊天历史"""
    history = agent.chat_manager.get_conversation(session_id)
    return jsonify({'success': True, 'history': history})

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """文件上传"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': '没有文件'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': '没有选择文件'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(config.UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'path': filepath,
            'size': os.path.getsize(filepath)
        })
    
    return jsonify({'success': False, 'error': '文件类型不支持'})

def allowed_file(filename):
    """检查文件类型"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in config.ALLOWED_EXTENSIONS

@app.route('/api/files')
def get_files():
    """获取已上传文件列表"""
    files = []
    for filename in os.listdir(config.UPLOAD_FOLDER):
        filepath = os.path.join(config.UPLOAD_FOLDER, filename)
        if os.path.isfile(filepath):
            files.append({
                'filename': filename,
                'path': filepath,
                'size': os.path.getsize(filepath)
            })
    return jsonify({'success': True, 'files': files})

@app.route('/api/test-ollama', methods=['POST'])
def test_ollama():
    """测试Ollama连接"""
    connected = agent.model_manager.test_ollama_connection()
    return jsonify({
        'success': True,
        'connected': connected,
        'message': 'Ollama连接正常' if connected else 'Ollama连接失败'
    })

@app.route('/api/code/execute', methods=['POST'])
def execute_code():
    """执行代码"""
    data = request.json
    code = data.get('code', '')
    session_id = data.get('session_id', 'default')
    timeout = data.get('timeout', 30)
    tags = data.get('tags', None)

    if not code.strip():
        return jsonify({'success': False, 'error': '代码不能为空'})

    result = sandbox_executor.execute(code, session_id, timeout, tags)
    return jsonify({'success': True, 'result': result.to_dict()})

@app.route('/api/code/output/<exec_id>/<filename>')
def get_code_output(exec_id, filename):
    """获取代码执行输出文件"""
    filepath = sandbox_executor.get_output_file(exec_id, filename)
    if filepath:
        return send_file(filepath)
    return jsonify({'success': False, 'error': '文件不存在'}), 404

@socketio.on('code_execute')
def handle_code_execute(data):
    """处理代码执行请求（流式）"""
    session_id = data.get('session_id', 'default')
    code = data.get('code', '')
    timeout = data.get('timeout', 30)
    client_sid = request.sid

    def run_code():
        with app.app_context():
            try:
                socketio.emit('code_execution_start', {}, room=client_sid)
                result = sandbox_executor.execute(code, session_id, timeout)
                socketio.emit('code_execution_result', {'result': result.to_dict()}, room=client_sid)
            except Exception as e:
                socketio.emit('code_execution_error', {'error': str(e)}, room=client_sid)

    socketio.start_background_task(run_code)

@socketio.on('code_stop')
def handle_code_stop(data):
    """停止代码执行"""
    session_id = data.get('session_id', 'default')
    sandbox_executor.stop_execution(session_id)
    emit('code_execution_stopped', {}, room=request.sid)

if __name__ == '__main__':
    try:
        print("启动 Ollama Agent...")
        print("Web界面: http://localhost:5000")
        print("WebSocket: ws://localhost:5000")
        
        # 检查静态文件
        static_path = os.path.join(os.path.dirname(__file__), 'static', 'index.html')
        if os.path.exists(static_path):
            print(f"静态文件存在: {static_path}")
        else:
            print(f"静态文件不存在: {static_path}")
        
        # 启动应用
        socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
        
    except Exception as e:
        print(f"启动失败: {e}")
        import traceback
        traceback.print_exc()