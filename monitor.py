#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ollama Agent 命令行监控工具
实时显示系统状态、连接信息、模型状态等
"""

import os
import sys
import time
import json
import requests
import psutil
from datetime import datetime
from typing import Dict, List

class OllamaAgentMonitor:
    def __init__(self):
        self.base_url = "http://localhost:5000"
        self.running = True
        
    def clear_screen(self):
        """清屏"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def get_system_info(self) -> Dict:
        """获取系统信息"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory': psutil.virtual_memory(),
            'disk': psutil.disk_usage('/'),
            'network': psutil.net_io_counters(),
            'processes': len(psutil.pids())
        }
    
    def get_ollama_status(self) -> Dict:
        """获取Ollama状态"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=3)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return {
                    'connected': True,
                    'models_count': len(models),
                    'models': [model['name'] for model in models]
                }
        except:
            pass
        
        return {
            'connected': False,
            'models_count': 0,
            'models': []
        }
    
    def get_agent_status(self) -> Dict:
        """获取Agent状态"""
        try:
            response = requests.get(f"{self.base_url}/api/models", timeout=3)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        
        return {'success': False, 'error': 'Agent服务不可用'}
    
    def format_bytes(self, bytes_val: int) -> str:
        """格式化字节数"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_val < 1024.0:
                return f"{bytes_val:.1f} {unit}"
            bytes_val /= 1024.0
        return f"{bytes_val:.1f} PB"
    
    def format_duration(self, seconds: float) -> str:
        """格式化时间"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    def display_header(self):
        """显示头部信息"""
        print("=" * 80)
        print("🤖 Ollama Agent 监控系统".center(80))
        print("=" * 80)
        print(f"⏰ 监控时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🌐 Agent地址: {self.base_url}")
        print(f"🔗 Ollama地址: http://localhost:11434")
        print("-" * 80)
    
    def display_system_status(self, system_info: Dict):
        """显示系统状态"""
        print("📊 系统状态:")
        print(f"   CPU使用率: {system_info['cpu_percent']:.1f}%")
        
        memory = system_info['memory']
        print(f"   内存使用: {self.format_bytes(memory.used)} / {self.format_bytes(memory.total)} ({memory.percent:.1f}%)")
        
        disk = system_info['disk']
        print(f"   磁盘使用: {self.format_bytes(disk.used)} / {self.format_bytes(disk.total)} ({disk.percent:.1f}%)")
        
        print(f"   活跃进程: {system_info['processes']}")
        print()
    
    def display_ollama_status(self, ollama_status: Dict):
        """显示Ollama状态"""
        print("🦙 Ollama状态:")
        if ollama_status['connected']:
            print("   ✅ 连接正常")
            print(f"   📦 可用模型: {ollama_status['models_count']} 个")
            
            if ollama_status['models']:
                print("   📋 模型列表:")
                for i, model in enumerate(ollama_status['models'][:5], 1):  # 只显示前5个
                    print(f"      {i}. {model}")
                if len(ollama_status['models']) > 5:
                    print(f"      ... 还有 {len(ollama_status['models']) - 5} 个模型")
        else:
            print("   ❌ 连接失败")
            print("   💡 请确保Ollama服务正在运行")
        print()
    
    def display_agent_status(self, agent_status: Dict):
        """显示Agent状态"""
        print("🤖 Agent状态:")
        if agent_status.get('success'):
            print("   ✅ Agent服务正常")
            print(f"   📡 连接状态: {'正常' if agent_status.get('ollama_connected') else '异常'}")
            
            models = agent_status.get('models', [])
            print(f"   🧠 检测到模型: {len(models)} 个")
            
            # 统计模型类型
            vision_count = sum(1 for m in models if m.get('is_vision'))
            audio_count = sum(1 for m in models if m.get('is_audio'))
            text_count = len(models) - vision_count - audio_count
            
            if text_count > 0:
                print(f"      📝 文本模型: {text_count}")
            if vision_count > 0:
                print(f"      👁️ 视觉模型: {vision_count}")
            if audio_count > 0:
                print(f"      🎵 音频模型: {audio_count}")
        else:
            print("   ❌ Agent服务异常")
            print(f"   💡 错误信息: {agent_status.get('error', '未知错误')}")
        print()
    
    def display_network_info(self, system_info: Dict):
        """显示网络信息"""
        network = system_info['network']
        print("🌐 网络状态:")
        print(f"   📤 发送: {self.format_bytes(network.bytes_sent)}")
        print(f"   📥 接收: {self.format_bytes(network.bytes_recv)}")
        print(f"   📦 数据包发送: {network.packets_sent}")
        print(f"   📦 数据包接收: {network.packets_recv}")
        print()
    
    def display_commands(self):
        """显示可用命令"""
        print("⌨️  快捷键:")
        print("   q - 退出监控")
        print("   r - 刷新状态")
        print("   c - 清屏")
        print("   h - 显示帮助")
        print("-" * 80)
    
    def run(self):
        """运行监控"""
        print("🚀 启动Ollama Agent监控...")
        time.sleep(1)
        
        while self.running:
            try:
                self.clear_screen()
                self.display_header()
                
                # 获取状态信息
                system_info = self.get_system_info()
                ollama_status = self.get_ollama_status()
                agent_status = self.get_agent_status()
                
                # 显示状态
                self.display_system_status(system_info)
                self.display_ollama_status(ollama_status)
                self.display_agent_status(agent_status)
                self.display_network_info(system_info)
                self.display_commands()
                
                # 等待用户输入
                print("按键操作 (自动刷新: 5秒)...")
                start_time = time.time()
                
                # 使用跨平台的方式检查输入
                try:
                    import select
                    # 在Windows上，select.select()不能用于stdin，所以我们使用更简单的方法
                    if os.name == 'nt':
                        # Windows系统：使用简单的时间等待
                        time.sleep(5)
                        # 在Windows上我们无法非阻塞读取输入，所以跳过输入检查
                        key = ''
                    else:
                        # Unix/Linux系统：使用select
                        if select.select([sys.stdin], [], [], 5) == ([sys.stdin], [], []):
                            key = sys.stdin.readline().strip().lower()
                        else:
                            key = ''
                    
                    if key == 'q':
                        print("\n👋 退出监控...")
                        break
                    elif key == 'r':
                        continue  # 立即刷新
                    elif key == 'c':
                        self.clear_screen()
                        continue
                    elif key == 'h':
                        self.show_help()
                        input("\n按回车键继续...")
                        continue
                except (OSError, ValueError) as e:
                    # 忽略套接字错误，继续监控
                    print(f"\n⚠️  输入检查错误: {e}")
                    time.sleep(5)
                
            except KeyboardInterrupt:
                print("\n\n👋 退出监控...")
                break
            except Exception as e:
                print(f"\n❌ 监控错误: {e}")
                time.sleep(2)
        
        self.running = False
    
    def show_help(self):
        """显示帮助信息"""
        self.clear_screen()
        print("=" * 60)
        print("🤖 Ollama Agent 监控系统帮助".center(60))
        print("=" * 60)
        print()
        print("📋 功能说明:")
        print("   • 实时监控系统资源使用情况")
        print("   • 监控Ollama服务连接状态")
        print("   • 显示可用模型列表和类型")
        print("   • 监控Agent服务状态")
        print("   • 显示网络IO统计")
        print()
        print("⌨️  快捷键:")
        print("   q - 退出监控系统")
        print("   r - 手动刷新状态")
        print("   c - 清屏")
        print("   h - 显示此帮助")
        print()
        print("📊 状态指示:")
        print("   ✅ 绿色 - 状态正常")
        print("   ❌ 红色 - 状态异常")
        print("   🟡 黄色 - 警告状态")
        print()
        print("🔧 故障排除:")
        print("   • 如果Ollama连接失败，请检查服务是否启动")
        print("   • 如果Agent服务异常，请检查端口5000是否被占用")
        print("   • 内存使用率过高可能影响模型运行性能")
        print()

if __name__ == "__main__":
    try:
        monitor = OllamaAgentMonitor()
        monitor.run()
    except ImportError as e:
        print(f"❌ 缺少依赖包: {e}")
        print("请运行: pip install psutil requests")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)