import json
import requests
from typing import Optional, List, Dict, Iterator, Set
import os
import base64
import re
from pathlib import Path

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class FilePermissionManager:
    """文件访问权限管理器"""
    
    def __init__(self):
        """初始化权限管理器"""
        self.allowed_dirs: Set[str] = set()
        self.allowed_files: Set[str] = set()
        self.enabled = False  # 默认关闭权限检查
    
    def allow_directory(self, directory: str):
        """允许访问的目录"""
        abs_path = os.path.abspath(directory)
        self.allowed_dirs.add(abs_path)
        self.enabled = True
    
    def allow_file(self, file_path: str):
        """允许访问的文件"""
        abs_path = os.path.abspath(file_path)
        self.allowed_files.add(abs_path)
        self.enabled = True
    
    def check_permission(self, path: str) -> bool:
        """
        检查路径是否有访问权限
        
        Args:
            path: 要检查的路径
            
        Returns:
            是否有权限访问
        """
        if not self.enabled:
            return True  # 如果未启用权限管理，允许所有访问
        
        abs_path = os.path.abspath(path)
        
        # 检查文件是否在允许列表中
        if abs_path in self.allowed_files:
            return True
        
        # 检查是否在允许的目录中
        for allowed_dir in self.allowed_dirs:
            if abs_path.startswith(allowed_dir):
                return True
        
        return False
    
    def get_allowed_paths(self) -> Dict[str, List[str]]:
        """获取所有允许的路径"""
        return {
            "directories": list(self.allowed_dirs),
            "files": list(self.allowed_files)
        }


class OllamaAgent:
    """用于调用本地 Ollama 模型的 Agent 类"""
    
    # 视觉相关问题关键词
    VISION_KEYWORDS = [
        "看到了什么", "看到了", "看到了吗", "看到",
        "图片", "图像", "照片", "图像中", "图片中", 
        "这张图片", "这个图片", "这张图", "这个图",
        "图片展示", "图像展示", "展示什么", "显示什么",
        "描述图片", "描述图像", "图片内容", "图像内容",
        "摄像头", "相机", "拍照", "拍摄", "画面",
        "视觉", "看到的内容", "眼前", "前面"
    ]
    
    # 需要使用 /api/generate 端点的模型
    GENERATE_ONLY_MODELS = [
        "qwen2.5-vl",
        "qwen2.5:vl",
        "qwen2.5:7b-vl",
        "qwen2.5:13b-vl",
        "qwen2.5:32b-vl",
        "qwen2.5:72b-vl",
        "qwen2.5:7b-vl-instruct",
        "qwen2.5:13b-vl-instruct",
        "qwen2.5:32b-vl-instruct",
        "qwen2.5:72b-vl-instruct"
    ]
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "DeepSeek-R1:7b",
        vision_model: Optional[str] = None,
        timeout: int = 120,
        allowed_dirs: Optional[List[str]] = None,
        allowed_files: Optional[List[str]] = None
    ):
        """
        初始化 Ollama Agent
        
        Args:
            base_url: Ollama API 的基础 URL
            model: 普通对话使用的模型名称
            vision_model: 视觉模型名称（如 qwen2.5-vl），如果为None则自动检测
            timeout: 请求超时时间（秒）
            allowed_dirs: 允许访问的目录列表
            allowed_files: 允许访问的文件列表
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.vision_model = vision_model
        self.timeout = timeout
        self.conversation_history: List[Dict[str, str]] = []
        self.permission_manager = FilePermissionManager()
        
        # 设置允许的目录和文件
        if allowed_dirs:
            for dir_path in allowed_dirs:
                self.permission_manager.allow_directory(dir_path)
        if allowed_files:
            for file_path in allowed_files:
                self.permission_manager.allow_file(file_path)
    
    def is_generate_only_model(self, model_name: str) -> bool:
        """
        检查模型是否需要使用 /api/generate 端点
        
        Args:
            model_name: 模型名称
            
        Returns:
            是否需要使用 /api/generate 端点
        """
        model_lower = model_name.lower()
        return any(generate_model in model_lower for generate_model in self.GENERATE_ONLY_MODELS)
    
    def _check_connection(self) -> bool:
        """检查与 Ollama 服务的连接"""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def list_models(self) -> List[str]:
        """获取可用的模型列表"""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            return models
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"无法获取模型列表: {e}")
    
    def find_vision_model(self) -> Optional[str]:
        """
        自动查找可用的视觉模型
        
        Returns:
            找到的视觉模型名称，如果没有则返回None
        """
        try:
            models = self.list_models()
            # 常见的视觉模型关键词
            vision_keywords = ['vl', 'vision', 'qwen2.5vl', 'llava', 'bakllava', 'moondream']
            
            # 优先使用用户指定的视觉模型
            if self.vision_model and self.vision_model in models:
                return self.vision_model
            
            # 自动查找包含视觉关键词的模型
            for model in models:
                model_lower = model.lower()
                if any(keyword in model_lower for keyword in vision_keywords):
                    return model
            
            return None
        except Exception:
            return None
    
    def is_vision_query(self, message: str) -> bool:
        """
        判断用户输入是否需要视觉模型
        
        Args:
            message: 用户消息
            
        Returns:
            是否需要使用视觉模型
        """
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in self.VISION_KEYWORDS)
    
    def set_model(self, model: str):
        """设置要使用的模型"""
        self.model = model
        print(f"已切换到模型: {model}")
    
    def encode_image(self, image_path: str) -> Optional[str]:
        """
        将图片编码为base64字符串
        
        Args:
            image_path: 图片文件路径
            
        Returns:
            base64编码的图片字符串，失败返回None
        """
        try:
            # 检查权限
            if not self.permission_manager.check_permission(image_path):
                print(f"错误: 没有权限访问文件 {image_path}")
                return None
            
            if not os.path.exists(image_path):
                print(f"错误: 文件不存在 {image_path}")
                return None
            
            with open(image_path, 'rb') as image_file:
                image_data = image_file.read()
                base64_image = base64.b64encode(image_data).decode('utf-8')
                
                # 根据文件扩展名确定MIME类型
                ext = Path(image_path).suffix.lower()
                mime_types = {
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg',
                    '.png': 'image/png',
                    '.gif': 'image/gif',
                    '.webp': 'image/webp'
                }
                mime_type = mime_types.get(ext, 'image/jpeg')
                
                return f"data:{mime_type};base64,{base64_image}"
        except Exception as e:
            print(f"编码图片失败: {e}")
            return None
    
    def capture_camera(self, camera_index: int = 0, save_path: Optional[str] = None) -> Optional[str]:
        """
        从摄像头捕获一张图片
        
        Args:
            camera_index: 摄像头索引，默认为0
            save_path: 保存路径，如果为None则不保存
            
        Returns:
            如果save_path不为None，返回保存的图片路径；否则返回临时图片路径
        """
        if not CV2_AVAILABLE:
            print("错误: OpenCV (cv2) 未安装，无法使用摄像头功能")
            print("请安装: pip install opencv-python")
            return None
        
        try:
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                print(f"错误: 无法打开摄像头 {camera_index}")
                return None
            
            print("正在从摄像头捕获图片...（按任意键继续）")
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                print("错误: 无法从摄像头读取图片")
                return None
            
            # 确定保存路径
            if save_path is None:
                save_path = "camera_capture.jpg"
            
            # 确保目录存在
            save_dir = os.path.dirname(save_path) if os.path.dirname(save_path) else "."
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # 保存图片
            cv2.imwrite(save_path, frame)
            print(f"图片已保存到: {save_path}")
            
            return save_path
        except Exception as e:
            print(f"摄像头捕获失败: {e}")
            return None
    
    def read_image_file(self, file_path: str) -> Optional[str]:
        """
        读取图片文件并检查权限
        
        Args:
            file_path: 图片文件路径
            
        Returns:
            如果成功，返回base64编码的图片；否则返回None
        """
        return self.encode_image(file_path)
    
    def chat(
        self,
        message: str,
        stream: bool = True,
        context: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None
    ) -> Optional[str]:
        """
        发送消息并获取回复
        
        Args:
            message: 用户消息
            stream: 是否使用流式响应
            context: 自定义对话上下文（如果为 None，则使用内部历史记录）
            system_prompt: 系统提示词
        
        Returns:
            如果 stream=False，返回完整的回复文本
            如果 stream=True，返回 None（需要通过 chat_stream 方法使用）
        """
        if stream:
            print("使用 chat_stream 方法获取流式响应")
            return None
        
        # 准备消息列表
        messages = []
        
        # 添加系统提示词
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # 添加对话历史或自定义上下文
        if context is None:
            messages.extend(self.conversation_history)
        else:
            messages.extend(context)
        
        # 添加当前消息
        messages.append({"role": "user", "content": message})
        
        # 构建请求载荷
        payload = {
            "model": selected_model,
            "messages": messages,
            "stream": True
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            
            # 获取回复
            assistant_message = result.get('message', {}).get('content', '')
            
            # 更新对话历史
            self.conversation_history.append({"role": "user", "content": message})
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            
            return assistant_message
            
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"请求失败: {e}")
    
    def chat_stream(
        self,
        message: str,
        image: Optional[str] = None,
        context: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        force_vision: bool = False
    ) -> Iterator[str]:
        """
        发送消息并获取流式回复
        
        Args:
            message: 用户消息
            image: base64编码的图片数据，或者图片文件路径（会被自动编码）
            context: 自定义对话上下文（如果为 None，则使用内部历史记录）
            system_prompt: 系统提示词
            force_vision: 强制使用视觉模型
        
        Yields:
            回复文本的片段
        """
        # 确定使用的模型
        use_vision = force_vision or self.is_vision_query(message) or image is not None
        selected_model = self.model
        
        if use_vision:
            vision_model = self.find_vision_model()
            if vision_model:
                selected_model = vision_model
                print(f"[切换到视觉模型: {vision_model}]")
            else:
                print("[警告: 未找到视觉模型，使用普通模型]")
                if image:
                    print("[警告: 当前模型可能无法处理图片]")
        
        # 检查是否是需要使用 /api/generate 端点的模型（如 qwen2.5-vl）
        if self.is_generate_only_model(selected_model) and (use_vision or image):
            # 对于 qwen2.5-vl 等模型，使用专门的视觉聊天方法
            print(f"[使用 /api/generate 端点处理视觉模型: {selected_model}]")
            yield from self.generate_stream_with_vision(message, image, model=selected_model, **{})
            return
        
        # 处理图片输入
        image_data = None
        if image:
            # 如果是文件路径，先编码
            if os.path.exists(image):
                image_data = self.encode_image(image)
                if not image_data:
                    print(f"[错误: 图片编码失败，文件路径: {image}]")
                    print("[将只使用文本进行回复]")
            else:
                # 假设已经是base64编码的数据
                image_data = image
        
        # 准备消息列表
        messages = []
        
        # 添加系统提示词
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # 添加对话历史或自定义上下文
        if context is None:
            messages.extend(self.conversation_history)
        else:
            messages.extend(context)
        
        # 添加当前消息
        messages.append({"role": "user", "content": message})
        
        # 构建请求载荷
        payload = {
            "model": selected_model,
            "messages": messages,
            "stream": True
        }
        
        # 对于 qwen2.5-vl 模型，使用 images 字段传递图片数据
        if image_data:
            # 提取 base64 数据（去掉 data:image/...;base64, 前缀）
            if image_data.startswith('data:'):
                base64_data = image_data.split(',', 1)[1]
            else:
                base64_data = image_data
            
            # 使用 images 字段传递图片
            payload["images"] = [base64_data]
            print(f"[已添加图片数据，base64长度: {len(base64_data)} 字符]")
        elif image:
            print("[警告: 提供了图片参数但未能成功处理图片数据]")
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        content = data.get('message', {}).get('content', '')
                        if content:
                            full_response += content
                            yield content
                    except json.JSONDecodeError:
                        continue
            
            # 更新对话历史（不包含图片数据）
            self.conversation_history.append({"role": "user", "content": message})
            self.conversation_history.append({"role": "assistant", "content": full_response})
            
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"流式请求失败: {e}")
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []
        print("对话历史已清空")
    
    def get_history(self) -> List[Dict[str, str]]:
        """获取对话历史"""
        return self.conversation_history.copy()
    
    def generate(
        self,
        prompt: str,
        stream: bool = False,
        **kwargs
    ) -> Optional[str]:
        """
        直接生成文本（不使用对话格式）
        
        Args:
            prompt: 输入提示词
            stream: 是否使用流式响应
            **kwargs: 其他参数（如 temperature, top_p 等）
        
        Returns:
            生成的文本
        """
        if stream:
            print("流式生成请使用 generate_stream 方法")
            return None
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            **kwargs
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            return result.get('response', '')
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"生成请求失败: {e}")
    
    def generate_stream_with_vision(
        self,
        prompt: str,
        image: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> Iterator[str]:
        """
        使用 /api/generate 端点进行流式生成，支持视觉输入（用于 qwen2.5-vl 等模型）
        
        Args:
            prompt: 输入提示词
            image: base64编码的图片数据，或者图片文件路径（会被自动编码）
            model: 使用的模型名称，如果为None则使用self.model
            **kwargs: 其他参数
        
        Yields:
            生成的文本片段
        """
        # 使用指定的模型或默认模型
        selected_model = model if model else self.model
        
        # 处理图片输入
        image_data = None
        if image:
            # 如果是文件路径，先编码
            if os.path.exists(image):
                image_data = self.encode_image(image)
                if not image_data:
                    print(f"[错误: 图片编码失败，文件路径: {image}]")
                    print("[将只使用文本进行回复]")
            else:
                # 假设已经是base64编码的数据
                image_data = image
        
        # 构建请求载荷
        payload = {
            "model": selected_model,
            "prompt": prompt,
            "stream": True,
            **kwargs
        }
        
        # 对于 qwen2.5-vl 模型，使用 images 字段传递图片数据
        if image_data:
            # 提取 base64 数据（去掉 data:image/...;base64, 前缀）
            if image_data.startswith('data:'):
                base64_data = image_data.split(',', 1)[1]
            else:
                base64_data = image_data
            
            # 使用 images 字段传递图片
            payload["images"] = [base64_data]
            print(f"[已添加图片数据，base64长度: {len(base64_data)} 字符]")
        elif image:
            print("[警告: 提供了图片参数但未能成功处理图片数据]")
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            full_response = ""
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        content = data.get('response', '')
                        if content:
                            full_response += content
                            yield content
                    except json.JSONDecodeError:
                        continue
            
            # 更新对话历史
            self.conversation_history.append({"role": "user", "content": f"[包含图片] {prompt}"})
            self.conversation_history.append({"role": "assistant", "content": full_response})
            
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"流式生成请求失败: {e}")
    
    def generate_file_description(self, file_path: str) -> Optional[str]:
        """
        使用语言模型生成文件描述
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件描述文本
        """
        try:
            # 检查权限
            if not self.permission_manager.check_permission(file_path):
                print(f"错误: 没有权限访问文件 {file_path}")
                return None
            
            if not os.path.exists(file_path):
                print(f"错误: 文件不存在 {file_path}")
                return None
            
            # 获取文件信息
            file_name = os.path.basename(file_path)
            file_ext = os.path.splitext(file_name)[1].lower()
            
            # 根据文件类型构建不同的提示词
            if file_ext in ['.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm']:
                # 视频文件提示词
                prompt = f"请为以下视频文件生成一个简短描述，不超过20个字：\n文件名：{file_name}\n文件类型：{file_ext}\n描述："
            elif file_ext == '.gif':
                # GIF文件提示词
                prompt = f"请为以下GIF动图文件生成一个简短描述，不超过20个字：\n文件名：{file_name}\n文件类型：{file_ext}\n描述："
            elif file_ext in ['.doc', '.docx', '.pdf', '.txt', '.rtf']:
                # 文档文件提示词
                prompt = f"请为以下文档文件生成一个简短描述，不超过20个字：\n文件名：{file_name}\n文件类型：{file_ext}\n描述："
            elif file_ext in ['.xls', '.xlsx', '.csv']:
                # 表格文件提示词
                prompt = f"请为以下表格文件生成一个简短描述，不超过20个字：\n文件名：{file_name}\n文件类型：{file_ext}\n描述："
            elif file_ext in ['.ppt', '.pptx']:
                # 演示文稿文件提示词
                prompt = f"请为以下演示文稿文件生成一个简短描述，不超过20个字：\n文件名：{file_name}\n文件类型：{file_ext}\n描述："
            elif file_ext in ['.zip', '.rar', '.7z', '.tar', '.gz']:
                # 压缩文件提示词
                prompt = f"请为以下压缩文件生成一个简短描述，不超过20个字：\n文件名：{file_name}\n文件类型：{file_ext}\n描述："
            elif file_ext in ['.exe', '.msi', '.dmg']:
                # 可执行文件提示词
                prompt = f"请为以下可执行文件生成一个简短描述，不超过20个字：\n文件名：{file_name}\n文件类型：{file_ext}\n描述："
            elif file_ext in ['.mp3', '.wav', '.flac', '.aac', '.ogg']:
                # 音频文件提示词
                prompt = f"请为以下音频文件生成一个简短描述，不超过20个字：\n文件名：{file_name}\n文件类型：{file_ext}\n描述："
            else:
                # 其他文件类型提示词
                prompt = f"请为以下文件生成一个简短描述，不超过20个字：\n文件名：{file_name}\n文件类型：{file_ext}\n描述："
            
            # 使用普通模型生成描述
            response = ""
            for chunk in self.generate_stream(prompt):
                response += chunk
            
            # 清理响应，只保留描述部分
            description = response.strip()
            if "描述：" in description:
                description = description.split("描述：", 1)[1].strip()
            
            return description[:50]  # 限制长度
            
        except Exception as e:
            print(f"生成文件描述失败: {e}")
            return None
    
    def generate_image_description(self, image_path: str) -> Optional[str]:
        """
        使用视觉语言模型生成图片描述
        
        Args:
            image_path: 图片文件路径
            
        Returns:
            图片描述文本
        """
        try:
            # 检查权限
            if not self.permission_manager.check_permission(image_path):
                print(f"错误: 没有权限访问文件 {image_path}")
                return None
            
            if not os.path.exists(image_path):
                print(f"错误: 文件不存在 {image_path}")
                return None
            
            # 确保使用视觉模型
            vision_model = self.find_vision_model()
            if not vision_model:
                print("错误: 未找到可用的视觉模型")
                return None
            
            # 构建提示词
            prompt = "请用简短的语言描述这张图片的内容，不超过20个字"
            
            # 使用视觉模型生成描述
            response = ""
            if self.is_generate_only_model(vision_model):
                # 对于 qwen2.5-vl 等模型，使用专门的视觉聊天方法
                for chunk in self.generate_stream_with_vision(prompt, image=image_path, model=vision_model):
                    response += chunk
            else:
                # 对于其他视觉模型，使用 chat_stream
                for chunk in self.chat_stream(prompt, image=image_path):
                    response += chunk
            
            # 清理响应
            description = response.strip()
            
            return description[:50]  # 限制长度
            
        except Exception as e:
            print(f"生成图片描述失败: {e}")
            return None
    
    def classify_file(self, file_path: str) -> Optional[str]:
        """
        根据文件名和内容对文件进行分类
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件分类名称
        """
        try:
            # 检查权限
            if not self.permission_manager.check_permission(file_path):
                print(f"错误: 没有权限访问文件 {file_path}")
                return None
            
            if not os.path.exists(file_path):
                print(f"错误: 文件不存在 {file_path}")
                return None
            
            # 获取文件信息
            file_name = os.path.basename(file_path).lower()
            file_ext = os.path.splitext(file_name)[1].lower()
            
            # 定义文件类型关键词映射
            file_type_keywords = {
                "截图": ["screenshot", "截图", "screen", "capture", "snip"],
                "设计素材": ["design", "素材", "template", "mockup", "icon", "logo", "ui", "ux"],
                "照片/活动": ["photo", "照片", "img", "picture", "pic", "活动", "event", "trip", "travel", "vacation"],
                "图表与示意图": ["chart", "diagram", "graph", "示意图", "流程图", "结构图", "mindmap"],
                "工作相关视频": ["meeting", "会议", "work", "工作", "presentation", "演示", "tutorial", "教程"],
                "GIF动图": [".gif"]
            }
            
            # 根据文件名关键词判断类型
            for category, keywords in file_type_keywords.items():
                for keyword in keywords:
                    if keyword in file_name:
                        return category
            
            # 如果是图片文件，使用视觉模型进一步分析
            if file_ext in ['.jpg', '.jpeg', '.png', '.webp'] and self.find_vision_model():
                description = self.generate_image_description(file_path)
                if description:
                    # 根据描述内容进一步分类
                    desc_lower = description.lower()
                    if any(word in desc_lower for word in ["截图", "屏幕", "界面", "screen"]):
                        return "截图"
                    elif any(word in desc_lower for word in ["图表", "图", "表", "数据", "chart"]):
                        return "图表与示意图"
                    elif any(word in desc_lower for word in ["设计", "logo", "界面", "design"]):
                        return "设计素材"
                    else:
                        return "照片/活动"
            
            # 如果是视频文件，尝试根据文件名生成描述并分类
            elif file_ext in ['.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm']:
                # 使用语言模型分析文件名，生成更准确的分类
                description = self.generate_file_description(file_path)
                if description:
                    desc_lower = description.lower()
                    if any(word in desc_lower for word in ["会议", "meeting", "工作", "work", "演示", "presentation", "教程", "tutorial"]):
                        return "工作相关视频"
                    elif any(word in desc_lower for word in ["娱乐", "电影", "电视剧", "视频", "video"]):
                        return "其他文件"  # 娱乐视频归类到其他文件
                
                # 如果无法通过描述分类，则根据文件扩展名默认分类
                return "工作相关视频"
            
            # 如果是GIF文件，尝试根据文件名生成描述并分类
            elif file_ext == '.gif':
                # 使用语言模型分析文件名，生成更准确的分类
                description = self.generate_file_description(file_path)
                if description:
                    desc_lower = description.lower()
                    if any(word in desc_lower for word in ["截图", "屏幕", "界面", "screen"]):
                        return "截图"
                    elif any(word in desc_lower for word in ["设计", "logo", "界面", "design"]):
                        return "设计素材"
                    elif any(word in desc_lower for word in ["图表", "图", "表", "数据", "chart"]):
                        return "图表与示意图"
                
                # 如果无法通过描述分类，则默认分类为GIF动图
                return "GIF动图"
            
            # 如果是文档文件，尝试根据文件名生成描述并分类
            elif file_ext in ['.doc', '.docx', '.pdf', '.txt', '.rtf']:
                # 使用语言模型分析文件名，生成更准确的分类
                description = self.generate_file_description(file_path)
                if description:
                    desc_lower = description.lower()
                    if any(word in desc_lower for word in ["合同", "协议", "报告", "方案", "工作", "work"]):
                        return "其他文件"  # 工作文档归类到其他文件
                
                # 默认分类
                return "其他文件"
            
            # 如果是表格文件，尝试根据文件名生成描述并分类
            elif file_ext in ['.xls', '.xlsx', '.csv']:
                # 使用语言模型分析文件名，生成更准确的分类
                description = self.generate_file_description(file_path)
                if description:
                    desc_lower = description.lower()
                    if any(word in desc_lower for word in ["数据", "统计", "报表", "分析"]):
                        return "图表与示意图"  # 数据表格归类到图表与示意图
                
                # 默认分类
                return "图表与示意图"
            
            # 如果是演示文稿文件，尝试根据文件名生成描述并分类
            elif file_ext in ['.ppt', '.pptx']:
                # 使用语言模型分析文件名，生成更准确的分类
                description = self.generate_file_description(file_path)
                if description:
                    desc_lower = description.lower()
                    if any(word in desc_lower for word in ["会议", "meeting", "工作", "work", "演示", "presentation", "教程", "tutorial"]):
                        return "工作相关视频"  # 工作演示归类到工作相关视频
                
                # 默认分类
                return "工作相关视频"
            
            # 如果是音频文件，尝试根据文件名生成描述并分类
            elif file_ext in ['.mp3', '.wav', '.flac', '.aac', '.ogg']:
                # 使用语言模型分析文件名，生成更准确的分类
                description = self.generate_file_description(file_path)
                if description:
                    desc_lower = description.lower()
                    if any(word in desc_lower for word in ["会议", "meeting", "工作", "work", "教程", "tutorial"]):
                        return "工作相关视频"  # 工作音频归类到工作相关视频
                
                # 默认分类
                return "其他文件"
            
            # 如果是压缩文件，尝试根据文件名生成描述并分类
            elif file_ext in ['.zip', '.rar', '.7z', '.tar', '.gz']:
                # 使用语言模型分析文件名，生成更准确的分类
                description = self.generate_file_description(file_path)
                if description:
                    desc_lower = description.lower()
                    if any(word in desc_lower for word in ["素材", "设计", "模板", "图标", "icon"]):
                        return "设计素材"
                    elif any(word in desc_lower for word in ["照片", "图片", "相册"]):
                        return "照片/活动"
                
                # 默认分类
                return "其他文件"
            
            # 如果是可执行文件，尝试根据文件名生成描述并分类
            elif file_ext in ['.exe', '.msi', '.dmg']:
                # 使用语言模型分析文件名，生成更准确的分类
                description = self.generate_file_description(file_path)
                if description:
                    desc_lower = description.lower()
                    if any(word in desc_lower for word in ["工具", "软件", "应用", "程序"]):
                        return "其他文件"  # 工具软件归类到其他文件
                
                # 默认分类
                return "其他文件"
            
            # 根据文件扩展名进行默认分类
            if file_ext in ['.jpg', '.jpeg', '.png', '.webp']:
                return "照片/活动"
            elif file_ext in ['.mp4', '.mov', '.avi', '.mkv']:
                return "工作相关视频"
            elif file_ext == '.gif':
                return "GIF动图"
            else:
                return "其他文件"
                
        except Exception as e:
            print(f"文件分类失败: {e}")
            return None
    
    def scan_directory(self, directory: str, recursive: bool = True) -> List[str]:
        """
        扫描目录中的所有媒体文件
        
        Args:
            directory: 要扫描的目录路径
            recursive: 是否递归扫描子目录
            
        Returns:
            找到的媒体文件路径列表
        """
        try:
            # 检查权限
            if not self.permission_manager.check_permission(directory):
                print(f"错误: 没有权限访问目录 {directory}")
                return []
            
            if not os.path.exists(directory):
                print(f"错误: 目录不存在 {directory}")
                return []
            
            if not os.path.isdir(directory):
                print(f"错误: 路径不是目录 {directory}")
                return []
            
            # 支持的文件扩展名（包含所有常见文件类型）
            supported_extensions = {
                # 图片
                '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp',
                # 视频
                '.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm',
                # 音频
                '.mp3', '.wav', '.flac', '.aac', '.ogg',
                # 文档
                '.doc', '.docx', '.pdf', '.txt', '.rtf',
                # 表格
                '.xls', '.xlsx', '.csv',
                # 演示文稿
                '.ppt', '.pptx',
                # 压缩文件
                '.zip', '.rar', '.7z', '.tar', '.gz',
                # 可执行文件
                '.exe', '.msi', '.dmg'
            }
            
            found_files = []
            
            if recursive:
                # 递归扫描所有子目录
                for root, dirs, files in os.walk(directory):
                    for file in files:
                        file_ext = os.path.splitext(file)[1].lower()
                        if file_ext in supported_extensions:
                            file_path = os.path.join(root, file)
                            found_files.append(file_path)
            else:
                # 只扫描当前目录
                for file in os.listdir(directory):
                    file_path = os.path.join(directory, file)
                    if os.path.isfile(file_path):
                        file_ext = os.path.splitext(file)[1].lower()
                        if file_ext in supported_extensions:
                            found_files.append(file_path)
            
            print(f"在 {directory} 中找到 {len(found_files)} 个文件")
            return found_files
            
        except Exception as e:
            print(f"扫描目录失败: {e}")
            return []
    
    def organize_files(self, source_dir: str, target_dir: str, move_files: bool = False) -> Dict[str, int]:
        """
        整理文件到分类文件夹
        
        Args:
            source_dir: 源目录路径
            target_dir: 目标目录路径
            move_files: 是否移动文件（False表示复制）
            
        Returns:
            整理结果统计
        """
        try:
            # 检查权限
            if not self.permission_manager.check_permission(source_dir):
                print(f"错误: 没有权限访问源目录 {source_dir}")
                return {}
            
            if not self.permission_manager.check_permission(target_dir):
                print(f"错误: 没有权限访问目标目录 {target_dir}")
                return {}
            
            # 扫描源目录中的所有文件
            files_to_organize = self.scan_directory(source_dir, recursive=True)
            
            if not files_to_organize:
                print("没有找到需要整理的文件")
                return {}
            
            # 确保目标目录存在
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
                print(f"创建目标目录: {target_dir}")
            
            # 定义分类文件夹
            category_folders = {
                "截图": "截图",
                "设计素材": "设计素材",
                "照片/活动": "照片和活动",
                "图表与示意图": "图表与示意图",
                "工作相关视频": "工作相关视频",
                "GIF动图": "GIF动图",
                "其他文件": "其他文件"
            }
            
            # 先统计每个类别的文件数量
            category_counts = {category: 0 for category in category_folders.keys()}
            for file_path in files_to_organize:
                category = self.classify_file(file_path)
                if not category:
                    category = "其他文件"
                category_counts[category] += 1
            
            # 只为有文件的类别创建文件夹
            for category, count in category_counts.items():
                if count > 0:
                    folder_path = os.path.join(target_dir, category_folders[category])
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                        print(f"创建分类文件夹: {folder_path} (预计 {count} 个文件)")
            
            # 统计结果
            stats = {category: 0 for category in category_folders.keys()}
            stats["总文件数"] = len(files_to_organize)
            stats["已处理"] = 0
            stats["失败"] = 0
            
            print(f"开始整理 {len(files_to_organize)} 个文件...")
            
            # 处理每个文件
            for i, file_path in enumerate(files_to_organize, 1):
                try:
                    print(f"处理文件 {i}/{len(files_to_organize)}: {os.path.basename(file_path)}")
                    
                    # 分类文件
                    category = self.classify_file(file_path)
                    if not category:
                        category = "其他文件"
                    
                    # 生成新文件名（可选）
                    file_name = os.path.basename(file_path)
                    file_ext = os.path.splitext(file_name)[1]
                    
                    # 根据文件类型生成描述
                    description = None
                    
                    # 如果是图片文件，使用视觉模型生成描述
                    if file_ext.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                        description = self.generate_image_description(file_path)
                    # 如果是视频文件，尝试根据文件名生成描述
                    elif file_ext.lower() in ['.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm']:
                        description = self.generate_file_description(file_path)
                        if not description:
                            # 如果无法生成描述，使用默认描述
                            if category == "工作相关视频":
                                description = "工作相关视频"
                            else:
                                description = "视频文件"
                    # 如果是GIF文件，尝试根据文件名生成描述
                    elif file_ext.lower() == '.gif':
                        description = self.generate_file_description(file_path)
                        if not description:
                            description = "GIF动图"
                    # 如果是文档文件，尝试根据文件名生成描述
                    elif file_ext.lower() in ['.doc', '.docx', '.pdf', '.txt', '.rtf']:
                        description = self.generate_file_description(file_path)
                        if not description:
                            description = "文档文件"
                    # 如果是表格文件，尝试根据文件名生成描述
                    elif file_ext.lower() in ['.xls', '.xlsx', '.csv']:
                        description = self.generate_file_description(file_path)
                        if not description:
                            description = "表格文件"
                    # 如果是演示文稿文件，尝试根据文件名生成描述
                    elif file_ext.lower() in ['.ppt', '.pptx']:
                        description = self.generate_file_description(file_path)
                        if not description:
                            description = "演示文稿"
                    # 如果是音频文件，尝试根据文件名生成描述
                    elif file_ext.lower() in ['.mp3', '.wav', '.flac', '.aac', '.ogg']:
                        description = self.generate_file_description(file_path)
                        if not description:
                            description = "音频文件"
                    # 如果是压缩文件，尝试根据文件名生成描述
                    elif file_ext.lower() in ['.zip', '.rar', '.7z', '.tar', '.gz']:
                        description = self.generate_file_description(file_path)
                        if not description:
                            description = "压缩文件"
                    # 如果是可执行文件，尝试根据文件名生成描述
                    elif file_ext.lower() in ['.exe', '.msi', '.dmg']:
                        description = self.generate_file_description(file_path)
                        if not description:
                            description = "可执行文件"
                    # 其他文件类型
                    else:
                        description = self.generate_file_description(file_path)
                        if not description:
                            description = f"{category}文件"
                    
                    # 如果有描述，使用描述作为文件名
                    if description:
                        # 清理描述，移除不适用于文件名的字符
                        safe_desc = re.sub(r'[\\/*?:"<>|]', "", description)
                        # 限制文件名长度，避免过长
                        if len(safe_desc) > 50:
                            safe_desc = safe_desc[:50]
                        new_name = f"{safe_desc}{file_ext}"
                    else:
                        new_name = file_name
                    
                    # 确定目标路径
                    target_folder = os.path.join(target_dir, category_folders[category])
                    
                    # 确保目标文件夹存在（理论上应该已经存在，但以防万一）
                    os.makedirs(target_folder, exist_ok=True)
                    
                    target_path = os.path.join(target_folder, new_name)
                    
                    # 如果目标文件已存在，添加序号
                    counter = 1
                    original_target = target_path
                    while os.path.exists(target_path):
                        name_part = os.path.splitext(os.path.basename(original_target))[0]
                        ext_part = os.path.splitext(os.path.basename(original_target))[1]
                        target_path = os.path.join(target_folder, f"{name_part}_{counter}{ext_part}")
                        counter += 1
                    
                    # 移动或复制文件
                    if move_files:
                        import shutil
                        shutil.move(file_path, target_path)
                        action = "移动"
                    else:
                        import shutil
                        shutil.copy2(file_path, target_path)
                        action = "复制"
                    
                    print(f"  {action}到: {category_folders[category]}/{os.path.basename(target_path)}")
                    
                    # 更新统计
                    stats[category] += 1
                    stats["已处理"] += 1
                    
                except Exception as e:
                    print(f"  处理失败: {e}")
                    stats["失败"] += 1
            
            # 打印统计结果
            print("\n整理完成！统计结果:")
            print(f"  总文件数: {stats['总文件数']}")
            print(f"  已处理: {stats['已处理']}")
            print(f"  失败: {stats['失败']}")
            print("\n各类型文件数量:")
            for category, count in stats.items():
                if category not in ["总文件数", "已处理", "失败"] and count > 0:
                    print(f"  {category}: {count}")
            
            return stats
            
        except Exception as e:
            print(f"文件整理失败: {e}")
            return {}


def select_model(agent: OllamaAgent) -> str:
    """
    显示可用模型列表并让用户通过数字选择
    
    Returns:
        选中的模型名称
    """
    try:
        models = agent.list_models()
        
        if not models:
            print("警告: 未找到可用的模型")
            print("请使用 'ollama pull <model_name>' 下载模型")
            return None
        
        print("\n可用的模型列表:")
        print("-" * 50)
        for i, model in enumerate(models, 1):
            # 高亮当前使用的模型
            marker = " <-- 当前使用" if model == agent.model else ""
            print(f"  {i}. {model}{marker}")
        print("-" * 50)
        
        while True:
            try:
                choice = input(f"\n请选择模型 (1-{len(models)}, 按 Enter 使用当前模型): ").strip()
                
                if not choice:
                    # 如果当前模型不在列表中，自动使用第一个
                    if agent.model not in models:
                        selected_model = models[0]
                        agent.set_model(selected_model)
                        print(f"当前模型 '{agent.model}' 不可用，已自动切换到: {selected_model}")
                        return selected_model
                    return agent.model
                
                choice_num = int(choice)
                if 1 <= choice_num <= len(models):
                    selected_model = models[choice_num - 1]
                    agent.set_model(selected_model)
                    return selected_model
                else:
                    print(f"无效选择，请输入 1 到 {len(models)} 之间的数字")
            except ValueError:
                print("请输入有效的数字")
            except KeyboardInterrupt:
                print("\n已取消选择")
                return None
                
    except Exception as e:
        print(f"获取模型列表失败: {e}")
        return None


def main():
    """示例用法"""
    print("初始化 Ollama Agent...")
    
    # 可选：设置允许访问的目录和文件
    # agent = OllamaAgent(allowed_dirs=["./images", "./data"], allowed_files=["./config.json"])
    agent = OllamaAgent()
    
    # 检查连接
    if not agent._check_connection():
        print("错误: 无法连接到 Ollama 服务")
        print(f"请确保 Ollama 正在运行在 {agent.base_url}")
        return
    
    print("✓ 已连接到 Ollama 服务")
    
    # 检查是否有视觉模型
    vision_model = agent.find_vision_model()
    if vision_model:
        print(f"✓ 检测到视觉模型: {vision_model}")
    else:
        print("⚠ 未检测到视觉模型，视觉功能将不可用")
        print("  可以运行: ollama pull qwen2.5-vl 下载视觉模型")
    
    # 选择默认模型
    selected_model = select_model(agent)
    if not selected_model:
        print("无法选择模型，程序退出")
        return
    
    print(f"\n当前默认模型: {selected_model}")
    
    # 交互式对话
    print("\n=== 开始对话 ===")
    print("命令提示:")
    print("  - 输入 'quit' 或 'exit' 退出")
    print("  - 输入 'clear' 清空历史")
    print("  - 输入 'model' 或 'switch' 切换模型")
    print("  - 输入 'list' 查看可用模型")
    print("  - 输入 'camera' 或 '摄像头' 从摄像头捕获图片并分析")
    print("  - 输入 'image <路径>' 分析指定图片文件")
    print("  - 输入 'permission' 查看/设置文件访问权限")
    print("  - 输入 'organize' 或 '整理' 启动文件夹管理助手")
    print("  - 询问视觉相关问题（如'你看到了什么？'）会自动使用视觉模型")
    print("-" * 50)
    
    while True:
        user_input = input("\n你: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', '退出']:
            print("再见！")
            break
        
        if user_input.lower() in ['clear', '清空']:
            agent.clear_history()
            continue
        
        if user_input.lower() in ['model', 'switch', '切换', '切换模型']:
            selected_model = select_model(agent)
            if selected_model:
                print(f"已切换到模型: {selected_model}")
            continue
        
        if user_input.lower() in ['list', '列表', '模型列表']:
            try:
                models = agent.list_models()
                print("\n可用模型:")
                for i, model in enumerate(models, 1):
                    marker = " <-- 当前默认" if model == agent.model else ""
                    vision_marker = " (视觉)" if vision_model and model == vision_model else ""
                    print(f"  {i}. {model}{marker}{vision_marker}")
            except Exception as e:
                print(f"获取模型列表失败: {e}")
            continue
        
        # 摄像头捕获
        if user_input.lower() in ['camera', '摄像头', '拍照']:
            image_path = agent.capture_camera()
            if image_path:
                prompt = input("请输入对这张图片的问题（直接回车使用默认）: ").strip()
                if not prompt:
                    prompt = "请描述这张图片中看到的内容"
                
                try:
                    # 确保使用视觉模型
                    vision_model = agent.find_vision_model()
                    if vision_model:
                        print(f"\n[切换到视觉模型: {vision_model}]")
                        print("\n助手: ", end="", flush=True)
                        for chunk in agent.chat_stream(prompt, image=image_path):
                            print(chunk, end="", flush=True)
                        print()
                    else:
                        print("\n[错误: 未找到视觉模型]")
                        print("请确保已安装视觉模型，如 qwen2.5-vl")
                except Exception as e:
                    print(f"\n错误: {e}")
            continue
        
        # 图片文件分析
        if user_input.lower().startswith('image ') or user_input.lower().startswith('图片 '):
            parts = user_input.split(' ', 1)
            if len(parts) < 2:
                print("用法: image <图片路径>")
                continue
            
            image_path = parts[1].strip().strip('"').strip("'")
            if not os.path.exists(image_path):
                print(f"错误: 文件不存在 {image_path}")
                continue
            
            prompt = input("请输入对这张图片的问题（直接回车使用默认）: ").strip()
            if not prompt:
                prompt = "请描述这张图片中看到的内容"
            
            try:
                print("\n助手: ", end="", flush=True)
                for chunk in agent.chat_stream(prompt, image=image_path):
                    print(chunk, end="", flush=True)
                print()
            except Exception as e:
                print(f"\n错误: {e}")
            continue
        
        # 文件权限管理
        if user_input.lower() in ['permission', '权限', 'perm']:
            print("\n文件访问权限管理:")
            paths = agent.permission_manager.get_allowed_paths()
            if not paths['directories'] and not paths['files']:
                print("  当前未设置任何权限限制（允许访问所有文件）")
            else:
                print("  允许的目录:")
                for d in paths['directories']:
                    print(f"    - {d}")
                print("  允许的文件:")
                for f in paths['files']:
                    print(f"    - {f}")
            
            print("\n提示: 在代码中初始化时使用 allowed_dirs 和 allowed_files 参数设置权限")
            continue
        
        # 文件夹管理助手
        if user_input.lower() in ['organize', '整理', '文件整理']:
            print("\n=== 文件夹管理助手 ===")
            
            # 获取源目录
            while True:
                source_dir = input("请输入要整理的源文件夹路径（按Enter使用当前目录）: ").strip()
                if not source_dir:
                    source_dir = "."
                
                if not os.path.exists(source_dir):
                    print(f"错误: 目录不存在 {source_dir}")
                    continue
                
                if not os.path.isdir(source_dir):
                    print(f"错误: 路径不是目录 {source_dir}")
                    continue
                
                # 添加到允许访问的目录
                agent.permission_manager.allow_directory(source_dir)
                break
            
            # 获取目标目录
            while True:
                target_dir = input("请输入整理后文件的保存路径（按Enter在源目录下创建'整理后的文件'文件夹）: ").strip()
                if not target_dir:
                    target_dir = os.path.join(source_dir, "整理后的文件")
                
                # 如果目标目录不存在，询问是否创建
                if not os.path.exists(target_dir):
                    create = input(f"目标目录 {target_dir} 不存在，是否创建？(y/n): ").strip().lower()
                    if create in ['y', 'yes', '是']:
                        os.makedirs(target_dir)
                        print(f"已创建目录: {target_dir}")
                    else:
                        continue
                
                # 添加到允许访问的目录
                agent.permission_manager.allow_directory(target_dir)
                break
            
            # 询问是移动还是复制文件
            move_files = False
            while True:
                action = input("请选择操作方式 - 移动文件会从原位置删除，复制文件会保留原文件 (move/copy): ").strip().lower()
                if action in ['move', '移动', 'm']:
                    move_files = True
                    break
                elif action in ['copy', '复制', 'c']:
                    move_files = False
                    break
                else:
                    print("请输入 'move' 或 'copy'")
            
            # 确认操作
            print(f"\n即将执行文件整理:")
            print(f"  源目录: {source_dir}")
            print(f"  目标目录: {target_dir}")
            print(f"  操作方式: {'移动' if move_files else '复制'}")
            
            confirm = input("\n确认执行？(y/n): ").strip().lower()
            if confirm not in ['y', 'yes', '是']:
                print("已取消文件整理")
                continue
            
            # 执行文件整理
            try:
                stats = agent.organize_files(source_dir, target_dir, move_files)
                if stats:
                    print(f"\n文件整理完成！共处理 {stats.get('已处理', 0)} 个文件")
                else:
                    print("\n文件整理失败")
            except Exception as e:
                print(f"\n文件整理过程中出错: {e}")
            
            continue
        
        # 普通对话（自动判断是否需要视觉模型）
        try:
            # 检查是否是视觉相关问题
            if agent.is_vision_query(user_input):
                print("\n[提示: 检测到视觉相关问题]")
                if vision_model and CV2_AVAILABLE:
                    print(f"[将使用视觉模型: {vision_model} 并自动捕获摄像头图片]")
                    
                    # 自动捕获摄像头图片
                    image_path = agent.capture_camera()
                    if image_path:
                        try:
                            print(f"\n[切换到视觉模型: {vision_model}]")
                            print("\n助手: ", end="", flush=True)
                            for chunk in agent.chat_stream(user_input, image=image_path):
                                print(chunk, end="", flush=True)
                            print()
                        except Exception as e:
                            print(f"\n错误: {e}")
                    else:
                        print("\n[错误: 无法捕获摄像头图片]")
                        print("将使用普通模型回答")
                        print("\n助手: ", end="", flush=True)
                        for chunk in agent.chat_stream(user_input):
                            print(chunk, end="", flush=True)
                        print()
                else:
                    if not vision_model:
                        print("[警告: 未找到视觉模型]")
                    if not CV2_AVAILABLE:
                        print("[警告: OpenCV未安装，无法使用摄像头功能]")
                    
                    print("将使用普通模型回答")
                    print("\n助手: ", end="", flush=True)
                    for chunk in agent.chat_stream(user_input):
                        print(chunk, end="", flush=True)
                    print()
            else:
                # 使用流式响应（自动切换模型）
                print("\n助手: ", end="", flush=True)
                for chunk in agent.chat_stream(user_input):
                    print(chunk, end="", flush=True)
                print()  # 换行
        except Exception as e:
            print(f"\n错误: {e}")


if __name__ == "__main__":
    main()



