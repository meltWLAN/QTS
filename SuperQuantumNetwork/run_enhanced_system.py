#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - 重定向脚本

此脚本已被废弃，请使用统一的官方入口: launch_supergod.py
"""

import os
import sys
import logging
import subprocess

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("enhanced_quantum_trading.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("SuperGodQuantumSystem")

def main():
    """重定向到官方启动入口"""
    logger.info("\n" + "="*80)
    logger.info("超神系统更新通知")
    logger.info("此脚本(run_enhanced_system.py)已被废弃")
    logger.info("超神系统现在使用统一的启动入口: launch_supergod.py")
    logger.info("="*80 + "\n")
    
    # 检查launch_supergod.py是否存在
    if not os.path.exists("launch_supergod.py"):
        logger.error("错误：找不到官方启动脚本 'launch_supergod.py'")
        logger.error("请确保您的超神系统安装完整")
        return 1
    
    # 询问用户是否跳转到官方脚本
    print("\n您想使用官方启动脚本吗？(y/n): ", end="")
    choice = input().strip().lower()
    
    if choice == 'y' or choice == 'yes':
        logger.info("正在启动官方超神系统...")
        
        # 设置启动参数 - 保留原有参数并添加超神模式
        args = sys.argv[1:] if len(sys.argv) > 1 else []
        
        # 确保包含超神模式参数
        if '--activate-field' not in args:
            args.append('--activate-field')
            
        if '--consciousness-boost' not in args:
            args.append('--consciousness-boost')
            
        cmd = [sys.executable, "launch_supergod.py"] + args
        
        # 启动官方脚本
        try:
            logger.info(f"执行命令: {' '.join(cmd)}")
            subprocess.call(cmd)
        except Exception as e:
            logger.error(f"启动失败: {str(e)}")
            return 1
    else:
        logger.info("您选择不启动官方脚本")
        logger.info("请注意，此脚本已不再维护，建议使用:")
        logger.info("python launch_supergod.py --activate-field --consciousness-boost")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 