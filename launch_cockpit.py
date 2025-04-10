#!/usr/bin/env python3
"""
超神量子共生系统 - 驾驶舱模式直接启动器
"""

import sys
import os
import logging
from datetime import datetime
from PyQt5.QtWidgets import QApplication
import traceback

# 设置日志
log_file = f"supergod_cockpit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CockpitLauncher")

def main():
    """启动驾驶舱"""
    try:
        # 显示启动信息
        banner = """
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                                                                       ║
    ║                  超神量子共生系统 - 驾驶舱模式                       ║
    ║                                                                       ║
    ║                 SUPERGOD QUANTUM SYMBIOTIC SYSTEM                     ║
    ║                         COCKPIT EDITION                               ║
    ║                                                                       ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """
        print(banner)
        
        logger.info("启动超神量子共生系统驾驶舱模式...")
        
        # 创建应用程序
        app = QApplication(sys.argv)
        app.setStyle("Fusion")  # 使用Fusion风格
        
        # 导入SupergodCockpit类
        from supergod_cockpit import SupergodCockpit
        
        # 创建驾驶舱实例
        cockpit = SupergodCockpit()
        
        # 添加必要的初始化代码
        if hasattr(cockpit, 'set_data_connector'):
            try:
                # 尝试初始化数据连接器
                from tushare_data_connector import TushareConnector
                data_connector = TushareConnector(token="0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10")
                if data_connector:
                    logger.info("数据连接器初始化成功")
                    cockpit.set_data_connector(data_connector)
            except Exception as e:
                logger.warning(f"数据连接器初始化失败: {str(e)}")
        
        # 显示驾驶舱
        cockpit.show()
        logger.info("驾驶舱启动成功")
        
        # 开始事件循环
        sys.exit(app.exec_())
        
    except Exception as e:
        logger.error(f"启动驾驶舱失败: {str(e)}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    main() 