#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ollama Agent 启动脚本
"""

import os
import sys
import subprocess
import time
import requests

def check_ollama():
    """检查Ollama服务是否运行"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        return response.status_code == 200
    except:
        return False

def main():
    """主函数"""
    print("Ollama Agent 启动脚本")
    print("=" * 50)
    
    # 检查Ollama服务
    print("检查Ollama服务...")
    if not check_ollama():
        print("错误: Ollama服务未运行，请先启动Ollama服务")
        print("可以在终端中运行: ollama serve")
        return
    
    print("Ollama服务运行正常")
    
    # 启动应用
    print("启动Flask-SocketIO应用...")
    try:
        subprocess.run([sys.executable, "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n应用已停止")
    except Exception as e:
        print(f"启动失败: {e}")

if __name__ == "__main__":
    main()