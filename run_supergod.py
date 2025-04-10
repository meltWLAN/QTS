#!/usr/bin/env python3
"""
超神量子共生系统 - 统一驾驶舱入口
整合所有功能的单一启动入口

说明：这是超神量子共生系统的唯一官方桌面版启动入口文件。
所有其他入口已被移除，以简化使用体验。

使用方法：
1. 终端模式：python run_supergod.py --mode terminal
2. 驾驶舱模式：python run_supergod.py --mode cockpit (默认模式)
3. 桌面模式：python run_supergod.py --mode desktop

可选参数：
--tushare-token TOKEN：设置Tushare API的token
--debug：启用调试日志
--help：显示帮助信息
"""

import os
import sys
import time
import logging
import argparse
import traceback
from datetime import datetime
import signal
import platform
from PyQt5.QtWidgets import QApplication
from SuperQuantumNetwork.supergod_cockpit import SupergodCockpit

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("supergod.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SupergodLauncher")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="超神量子共生系统启动器")
    parser.add_argument('--mode', choices=['terminal', 'cockpit', 'desktop'],
                      default='cockpit', help='运行模式')
    parser.add_argument('--tushare-token', help='Tushare API token')
    parser.add_argument('--debug', action='store_true', help='启用调试日志')
    return parser.parse_args()

def setup_environment():
    """设置运行环境"""
    # 设置工作目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # 添加模块搜索路径
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def start_cockpit():
    """启动驾驶舱模式"""
    try:
        logger.info("正在启动超神驾驶舱...")
        app = QApplication(sys.argv)
        cockpit = SupergodCockpit()
        cockpit.show()
        logger.info("驾驶舱启动成功")
        return app.exec_()
    except Exception as e:
        logger.error(f"启动驾驶舱失败: {str(e)}")
        logger.error(traceback.format_exc())
        return 1

def start_desktop():
    """启动桌面模式"""
    try:
        logger.info("正在启动超神桌面版...")
        # 尝试导入桌面版主程序
        try:
            from SuperQuantumNetwork.supergod_desktop import SupergodDesktop
            
            # 创建应用程序实例
            app = QApplication(sys.argv)
            
            # 设置应用程序样式
            app.setStyle('Fusion')
            
            # 创建主窗口
            window = SupergodDesktop()
            window.show()
            
            logger.info("桌面版启动成功")
            
            # 运行应用程序
            return app.exec_()
        except ImportError:
            # 尝试备份路径
            logger.warning("从主路径导入桌面版失败，尝试备份路径...")
            from SuperQuantumNetwork.backup_current.supergod_desktop import SupergodDesktop
            
            # 创建应用程序实例
            app = QApplication(sys.argv)
            
            # 设置应用程序样式
            app.setStyle('Fusion')
            
            # 创建主窗口
            window = SupergodDesktop()
            window.show()
            
            logger.info("桌面版启动成功 (从备份路径)")
            
            # 运行应用程序
            return app.exec_()
    except Exception as e:
        logger.error(f"启动桌面版失败: {str(e)}")
        logger.error(traceback.format_exc())
        return 1

def main():
    """主函数"""
    try:
        # 解析命令行参数
        args = parse_arguments()
        
        # 设置日志级别
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            
        # 设置环境
        setup_environment()
        
        # 显示启动信息
        logger.info("=" * 50)
        logger.info("超神量子共生系统 - 启动中")
        logger.info(f"运行模式: {args.mode}")
        logger.info(f"系统信息: {platform.platform()}")
        logger.info(f"Python版本: {platform.python_version()}")
        logger.info("=" * 50)
        
        # 根据模式启动系统
        if args.mode == 'cockpit':
            return start_cockpit()
        elif args.mode == 'desktop':
            return start_desktop()
        elif args.mode == 'terminal':
            logger.info("终端模式未实现，请使用驾驶舱或桌面模式")
            return 1
        else:
            logger.error(f"不支持的运行模式: {args.mode}")
            return 1
            
    except Exception as e:
        logger.error(f"系统启动失败: {str(e)}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())