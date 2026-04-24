#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ollama Agent 启动脚本
"""

import sys
import os

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    # 导入应用
    from app import app, socketio
    
    if __name__ == '__main__':
        print("🚀 启动 Ollama Agent...")
        print(f"📁 工作目录: {current_dir}")
        print(f"📡 Web界面: http://localhost:5000")
        print(f"🔗 WebSocket: ws://localhost:5000")
        socketio.run(app, host='0.0.0.0', port=5000, debug=True)
        
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请运行 python fix_and_start.py 来修复依赖")
    sys.exit(1)
except Exception as e:
    print(f"❌ 启动失败: {e}")
    sys.exit(1)