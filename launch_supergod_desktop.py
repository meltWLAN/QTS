#!/usr/bin/env python3
"""
超神量子共生系统 - 桌面版快速启动器
简化版桌面启动脚本
"""

import sys
import os
import logging
from datetime import datetime

# 设置日志
log_file = f"supergod_desktop_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SupergodDesktopLauncher")

def main():
    """主函数，负责启动桌面版系统"""
    logger.info("=" * 50)
    logger.info("超神量子共生系统 - 桌面版启动器")
    logger.info("=" * 50)
    
    # 添加必要的路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    backup_dir = os.path.join(current_dir, "SuperQuantumNetwork", "backup_current")
    
    # 检查备份目录是否存在
    if not os.path.exists(backup_dir):
        logger.error(f"备份目录不存在: {backup_dir}")
        return 1
    
    # 将备份目录添加到sys.path
    sys.path.append(backup_dir)
    
    try:
        # 切换到备份目录
        os.chdir(backup_dir)
        logger.info(f"切换到目录: {backup_dir}")
        
        # 导入必要的模块
        try:
            from PyQt5.QtWidgets import QApplication
            from supergod_desktop import SupergodDesktop
            
            logger.info("成功导入桌面版模块")
        except ImportError as e:
            logger.error(f"导入桌面版模块失败: {str(e)}")
            return 1
            
        # 创建并启动应用
        logger.info("正在启动桌面版应用...")
        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        
        # 创建主窗口
        window = SupergodDesktop()
        window.show()
        
        logger.info("桌面版应用启动成功")
        
        # 运行应用程序
        return app.exec_()
    except Exception as e:
        logger.error(f"启动桌面版失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 