#!/usr/bin/env python3
"""
超神系统启动重定向脚本

此脚本已不再是主要启动入口
请使用 launch_supergod.py 启动超神系统
"""

import os
import sys
import logging
import subprocess

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger("QuantumTradingSystem")


def main():
    """重定向到官方启动入口"""
    logger.info("=" * 80)
    logger.info("超神系统提示：请使用官方启动入口")
    logger.info("请使用 'python launch_supergod.py' 启动超神系统")
    logger.info("=" * 80)
    
    # 检查launch_supergod.py是否存在
    if not os.path.exists("launch_supergod.py"):
        logger.error("错误：找不到官方启动脚本 'launch_supergod.py'")
        logger.error("请确保您的超神系统安装完整")
        return 1
    
    # 询问用户是否自动启动
    print("\n是否要切换到官方启动脚本？(y/n): ", end="")
    choice = input().strip().lower()
    
    if choice == 'y' or choice == 'yes':
        logger.info("正在启动官方超神系统...")
        
        # 准备命令行参数（去掉当前脚本名称）
        args = sys.argv[1:] if len(sys.argv) > 1 else []
        cmd = [sys.executable, "launch_supergod.py"] + args
        
        # 启动官方脚本并等待其完成
        try:
            subprocess.call(cmd)
        except Exception as e:
            logger.error(f"启动失败: {str(e)}")
            return 1
    else:
        logger.info("您选择了不自动启动")
        logger.info("请记住，此脚本已不再维护，建议使用官方入口:")
        logger.info("python launch_supergod.py [选项]")

    return 0

if __name__ == "__main__":
    sys.exit(main()) 