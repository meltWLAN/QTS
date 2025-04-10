#!/usr/bin/env python3
"""
超神量子系统豪华桌面应用启动脚本
"""

import sys
import os
import logging
import subprocess
import traceback
import importlib.util

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("supergod_desktop_launcher.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("SupergodLauncher")

def check_dependencies():
    """检查所需依赖包"""
    required_packages = [
        'PyQt5',
        'numpy',
        'pandas',
        'matplotlib',
        'qt_material',
        'qdarkstyle',
        'jieba'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def check_core_modules():
    """检查核心模块是否存在"""
    core_modules = [
        'china_market_core.py',
        'policy_analyzer.py',
        'sector_rotation_tracker.py',
        'quantum_desktop_app.py',
        'quantum_ui_components.py'
    ]
    
    missing_modules = []
    
    for module in core_modules:
        if not os.path.exists(module):
            missing_modules.append(module)
    
    return missing_modules

def check_resource_directories():
    """检查资源目录是否存在"""
    resource_dirs = [
        'gui/resources',
        'resources'
    ]
    
    for dir_path in resource_dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"创建资源目录: {dir_path}")
    
    # 检查图标文件
    icon_paths = [
        'gui/resources/icon.png',
        'resources/icon.png',
        'icon.png'
    ]
    
    icon_exists = False
    for path in icon_paths:
        if os.path.exists(path):
            icon_exists = True
            break
    
    if not icon_exists:
        logger.warning("未找到图标文件，应用将使用默认图标")

def install_missing_packages(packages):
    """安装缺失的依赖包"""
    if not packages:
        return True
    
    logger.info(f"准备安装以下缺失的依赖包: {', '.join(packages)}")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
        logger.info("所有依赖包安装成功")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"安装依赖包失败: {str(e)}")
        return False

def launch_desktop_app():
    """启动超神系统豪华桌面应用"""
    try:
        # 确保当前目录是项目根目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        # 检查quantum_desktop_app.py文件是否存在并可导入
        app_path = os.path.join(script_dir, "quantum_desktop_app.py")
        if not os.path.exists(app_path):
            raise ImportError(f"找不到主应用文件: {app_path}")
        
        # 动态导入主模块
        spec = importlib.util.spec_from_file_location("quantum_desktop_app", app_path)
        if spec is None:
            raise ImportError(f"无法加载模块规范: {app_path}")
            
        quantum_desktop_app = importlib.util.module_from_spec(spec)
        if spec.loader is None:
            raise ImportError(f"加载器为空: {app_path}")
            
        spec.loader.exec_module(quantum_desktop_app)
        
        # 启动应用
        logger.info("正在启动超神量子系统豪华桌面应用...")
        return quantum_desktop_app.main()
    except Exception as e:
        error_message = f"启动应用失败: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_message)
        print(f"错误: {error_message}")
        return 1

def main():
    """主函数"""
    logger.info("超神量子系统豪华桌面应用启动器开始运行")
    
    # 检查资源目录
    check_resource_directories()
    
    # 检查核心模块
    missing_modules = check_core_modules()
    if missing_modules:
        logger.error(f"缺少核心模块文件: {', '.join(missing_modules)}")
        print(f"错误: 缺少核心模块文件: {', '.join(missing_modules)}")
        print("请确保所有必要的核心模块文件都存在于程序目录中。")
        return 1
        
    # 检查依赖
    missing_packages = check_dependencies()
    
    # 安装缺失的依赖
    if missing_packages:
        logger.warning(f"检测到缺失的依赖包: {', '.join(missing_packages)}")
        print(f"正在安装以下依赖包: {', '.join(missing_packages)}")
        if not install_missing_packages(missing_packages):
            logger.error("无法安装缺失的依赖包，请手动安装后重试")
            print("错误: 无法安装缺失的依赖包，请手动安装以下包后重试:")
            for package in missing_packages:
                print(f"  - {package}")
            return 1
    
    # 启动应用
    return launch_desktop_app()

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"启动器发生未处理异常: {str(e)}\n{traceback.format_exc()}")
        print(f"启动器发生未处理异常: {str(e)}")
        sys.exit(1) 