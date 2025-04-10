#!/usr/bin/env python3
"""
超神量子共生系统 - 驾驶舱模式唯一入口
固定使用 SuperQuantumNetwork/backup_current/run_supergod.py --mode cockpit 作为启动入口
"""

import os
import sys
import subprocess
import logging
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
logger = logging.getLogger("SupergodCockpitLauncher")

def main():
    """驾驶舱模式唯一入口"""
    # 显示启动信息
    banner = """
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║                  超神量子共生系统 - 驾驶舱模式                       ║
║                                                                       ║
║                 SUPERGOD QUANTUM SYMBIOTIC SYSTEM                     ║
║                         COCKPIT EDITION                               ║
║                                                                       ║
║           实时数据 · 增强分析 · 量子扩展 · 智能交互                  ║
║                   集成一体 · 尽在掌握                                 ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
"""
    print(banner)
    
    logger.info("正在启动超神量子共生系统驾驶舱模式...")
    
    # 备份目录路径
    backup_dir = os.path.join(os.getcwd(), "SuperQuantumNetwork", "backup_current")
    
    # 检查备份目录是否存在
    if not os.path.exists(backup_dir):
        logger.error(f"备份目录不存在: {backup_dir}")
        return 1
    
    # 显示启动命令
    logger.info("执行启动命令: cd SuperQuantumNetwork/backup_current && python run_supergod.py --mode cockpit")
    
    # 切换到备份目录并执行命令
    try:
        # 保存当前目录
        original_dir = os.getcwd()
        
        # 切换到备份目录
        os.chdir(backup_dir)
        logger.info(f"切换到目录: {backup_dir}")
        
        # 构建命令
        cmd = [sys.executable, "run_supergod.py", "--mode", "cockpit"]
        
        # 执行命令
        logger.info(f"执行命令: {' '.join(cmd)}")
        process = subprocess.run(cmd)
        
        # 返回原目录
        os.chdir(original_dir)
        
        # 返回进程退出码
        return process.returncode
    
    except Exception as e:
        logger.error(f"启动驾驶舱模式失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main()) 