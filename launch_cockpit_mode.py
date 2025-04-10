#!/usr/bin/env python3
"""
超神量子共生系统 - 驾驶舱模式替代启动器
(当前使用桌面版作为替代，因为驾驶舱模式中存在问题)
"""

import sys
import os
import logging
import subprocess
from datetime import datetime

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
    """启动驾驶舱模式（桌面版替代）"""
    # 显示启动信息
    banner = """
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║                  超神量子共生系统 - 驾驶舱模式                       ║
║                                                                       ║
║                 SUPERGOD QUANTUM SYMBIOTIC SYSTEM                     ║
║                         COCKPIT EDITION                               ║
║                                                                       ║
║                 (当前使用桌面版作为临时替代运行)                     ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
"""
    print(banner)
    
    logger.info("正在使用桌面版作为驾驶舱模式的替代启动...")
    
    # 备份目录路径
    backup_dir = os.path.join("SuperQuantumNetwork", "backup_current")
    
    # 检查备份目录是否存在
    if not os.path.exists(backup_dir):
        logger.error(f"备份目录不存在: {backup_dir}")
        return 1
    
    # 切换到备份目录并启动
    try:
        logger.info(f"切换到目录: {backup_dir}")
        
        # 使用subprocess运行桌面启动器
        process = subprocess.Popen(
            [sys.executable, "run_desktop.py"],
            cwd=backup_dir
        )
        
        logger.info(f"启动桌面版进程 (PID: {process.pid})")
        logger.info("桌面版已成功启动 (作为驾驶舱替代)")
        
        # 等待进程结束，传递退出码
        return process.wait()
    except Exception as e:
        logger.error(f"启动失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 