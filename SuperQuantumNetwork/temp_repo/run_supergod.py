#!/usr/bin/env python3
"""
超神量子共生系统 - 启动器
高级量子金融分析平台启动脚本
"""

import os
import sys
import logging
import importlib
import subprocess
import time

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("supergod_startup.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SupergodLauncher")

# 检查依赖
REQUIRED_PACKAGES = [
    'PyQt5', 'numpy', 'pandas', 'matplotlib', 
    'seaborn', 'scikit-learn', 'statsmodels'
]

# 检查所有依赖是否已安装
def check_dependencies():
    """检查所有必要的依赖是否已安装"""
    missing = []
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
            logger.info(f"已找到: {package}")
        except ImportError:
            missing.append(package)
            logger.warning(f"缺少依赖: {package}")
    
    # 询问是否安装缺少的依赖
    if missing:
        print("缺少以下必要依赖:")
        for pkg in missing:
            print(f"  - {pkg}")
        
        answer = input("是否自动安装这些依赖? (y/n): ")
        if answer.lower() == 'y':
            try:
                # 使用pip安装
                subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
                logger.info("依赖已安装")
                return True
            except subprocess.CalledProcessError:
                logger.error("安装依赖失败")
                return False
        else:
            logger.warning("用户选择不安装依赖")
            return False
    
    return True

def show_banner():
    """显示超神系统启动横幅"""
    banner = """
  _____                             _____              _   
 / ____|                           / ____|            | |  
| (___  _   _ _ __   ___ _ __ __ _| |  __  ___   __| | |_ 
 \___ \| | | | '_ \ / _ \ '__/ _` | | |_ |/ _ \ / _` | __|
 ____) | |_| | |_) |  __/ | | (_| | |__| | (_) | (_| | |_ 
|_____/ \__,_| .__/ \___|_|  \__, |\_____|\___/ \__,_|\__|
             | |              __/ |                       
             |_|             |___/                         
                          
        超神量子共生系统 - 高级量子金融分析平台
    """
    print(banner)
    print("\n初始化中...\n")

def check_system_files():
    """检查必要的系统文件是否存在"""
    required_files = [
        'supergod_cockpit.py',
        'china_market_core.py',
        'quantum_dimension_enhancer.py',
        'chaos_theory_framework.py'
    ]
    
    missing = [f for f in required_files if not os.path.exists(f)]
    
    if missing:
        logger.error("缺少必要的系统文件:")
        for file in missing:
            logger.error(f"  - {file}")
        return False
    
    logger.info("系统文件检查完成，所有必要文件存在")
    return True

def launch_system():
    """启动超神系统"""
    
    # 确定启动模式
    if os.path.exists('supergod_cockpit.py'):
        logger.info("发现高级驾驶舱，正在启动...")
        # 延迟一秒以显示启动信息
        time.sleep(1)
        print("正在启动超神量子共生系统 - 全息驾驶舱...")
        try:
            os.execl(sys.executable, sys.executable, 'supergod_cockpit.py')
        except Exception as e:
            logger.error(f"启动驾驶舱失败: {str(e)}")
            # 回退到桌面版
            print("驾驶舱启动失败，尝试启动桌面版...")
            try:
                os.execl(sys.executable, sys.executable, 'supergod_desktop.py')
            except Exception as e:
                logger.error(f"启动桌面版失败: {str(e)}")
                return False
    elif os.path.exists('supergod_desktop.py'):
        logger.info("发现桌面版，正在启动...")
        # 延迟一秒以显示启动信息
        time.sleep(1)
        print("正在启动超神量子共生系统 - 桌面版...")
        try:
            os.execl(sys.executable, sys.executable, 'supergod_desktop.py')
        except Exception as e:
            logger.error(f"启动桌面版失败: {str(e)}")
            return False
    else:
        logger.error("未找到可启动的系统界面")
        print("错误: 未找到可启动的系统界面，请确保supergod_cockpit.py或supergod_desktop.py文件存在")
        return False
    
    return True

def main():
    """主函数"""
    show_banner()
    
    # 检查依赖
    if not check_dependencies():
        print("无法继续，缺少必要依赖")
        return
    
    # 检查系统文件
    if not check_system_files():
        print("无法继续，缺少必要系统文件")
        return
    
    # 启动系统
    if not launch_system():
        print("系统启动失败，请检查日志了解详细信息")

if __name__ == "__main__":
    main() 