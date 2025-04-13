#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
量子交易系统统一启动入口

这是量子交易系统的标准化启动脚本。
请始终使用此脚本启动系统，而不是直接使用其他脚本。
"""

import os
import sys
import logging
from datetime import datetime

# 配置简单日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("QuantumTrading.Launcher")

def main():
    """启动量子交易系统"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 检查是否存在入口守护程序
    entry_guard_path = os.path.join(current_dir, "quantum_system_entry.py")
    if not os.path.exists(entry_guard_path):
        logger.error("错误: 系统入口守护程序不存在!")
        logger.error(f"请确保文件 '{entry_guard_path}' 存在")
        return 1
    
    # 将所有参数传递给入口守护程序
    args = sys.argv[1:] if len(sys.argv) > 1 else []
    
    # 启动入口守护程序
    logger.info("正在启动量子交易系统...")
    
    # 构建命令并执行
    cmd = [sys.executable, entry_guard_path] + args
    
    # 使用os.execv替换当前进程，而不是创建子进程
    # 这样可以确保用户看到的是守护程序的输出，而不是这个脚本的输出
    os.execv(sys.executable, cmd)
    
    # 这行代码永远不会执行，因为os.execv会替换当前进程
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n用户中断，正在退出...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"启动失败: {str(e)}")
        sys.exit(1) 