#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
超神量子系统 - 统一启动入口
此文件是系统的唯一入口点，永远不会改变
"""

import os
import sys
import subprocess
import argparse

# 获取脚本所在目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='超神量子系统 - 统一启动入口')
    
    parser.add_argument('--mode', '-m', default='desktop', 
                      choices=['desktop', 'console', 'server', 'cockpit'],
                      help='运行模式 (桌面GUI/控制台/服务器/驾驶舱)')
    
    parser.add_argument('--debug', '-d', action='store_true',
                      help='启用调试模式')
    
    parser.add_argument('--config', '-c', type=str, default=os.path.join(SCRIPT_DIR, 'config/system_config.json'),
                      help='配置文件路径 (默认: config/system_config.json)')
    
    return parser.parse_args()

def main():
    """主函数 - 启动系统"""
    args = parse_args()
    
    # 构建启动命令
    launcher_path = os.path.join(SCRIPT_DIR, "launch_quantum_core.py")
    
    # 检查启动器是否存在
    if not os.path.exists(launcher_path):
        print(f"错误: 启动器脚本不存在: {launcher_path}")
        return 1
        
    cmd = [sys.executable, launcher_path]
    
    # 添加模式参数
    cmd.extend(["--mode", args.mode])
    
    # 添加调试标志（如果启用）
    if args.debug:
        cmd.append("--debug")
    
    # 添加配置文件（如果指定）
    if args.config:
        cmd.extend(["--config", args.config])
    
    # 提示用户系统正在启动
    mode_names = {
        'desktop': '桌面模式',
        'console': '控制台模式',
        'server': '服务器模式',
        'cockpit': '驾驶舱模式'
    }
    
    print(f"正在启动超神量子系统 - {mode_names.get(args.mode, args.mode)}...")
    
    try:
        # 执行启动命令
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        print("用户中断，系统退出")
        return 0
    except Exception as e:
        print(f"启动失败: {e}")
        return 1

if __name__ == "__main__":
    # 切换到脚本所在目录，确保相对路径正确
    os.chdir(SCRIPT_DIR)
    sys.exit(main()) 