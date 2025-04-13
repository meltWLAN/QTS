#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
量子系统 - 简易启动脚本
提供一键启动量子系统的功能，让用户无需了解复杂的参数即可使用
"""

import os
import sys
import logging
import time
from PyQt5.QtWidgets import QApplication, QSplashScreen, QMessageBox
from PyQt5.QtGui import QPixmap, QFont, QColor
from PyQt5.QtCore import Qt, QTimer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("quantum_easy.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("QuantumEasy")

def show_splash_screen():
    """显示启动画面"""
    # 创建启动画面
    splash_pix = QPixmap(500, 300)
    splash_pix.fill(QColor(40, 44, 52))  # 深蓝色背景
    
    # 创建启动画面
    splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
    splash.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
    
    # 设置启动画面的字体
    font = splash.font()
    font.setPointSize(14)
    font.setBold(True)
    splash.setFont(font)
    
    # 显示启动画面
    splash.show()
    
    # 更新启动信息
    splash.showMessage("初始化量子系统...", Qt.AlignBottom | Qt.AlignCenter, Qt.white)
    QApplication.processEvents()
    
    time.sleep(0.5)
    splash.showMessage("加载量子后端...", Qt.AlignBottom | Qt.AlignCenter, Qt.white)
    QApplication.processEvents()
    
    time.sleep(0.5)
    splash.showMessage("初始化量子电路设计器...", Qt.AlignBottom | Qt.AlignCenter, Qt.white)
    QApplication.processEvents()
    
    time.sleep(0.5)
    splash.showMessage("连接量子市场分析器...", Qt.AlignBottom | Qt.AlignCenter, Qt.white)
    QApplication.processEvents()
    
    time.sleep(0.5)
    splash.showMessage("准备超级量子系统...", Qt.AlignBottom | Qt.AlignCenter, Qt.white)
    QApplication.processEvents()
    
    time.sleep(0.5)
    
    return splash

def main():
    """主函数"""
    try:
        # 设置工作目录
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        # 创建应用程序
        app = QApplication(sys.argv)
        
        # 显示启动画面
        splash = show_splash_screen()
        
        # 导入量子桌面系统
        from quantum_desktop.main import QuantumDesktopApp, MainWindow
        
        # 创建量子桌面应用
        quantum_app = QuantumDesktopApp()
        
        # 创建主窗口 - 不传递config参数
        main_window = MainWindow(quantum_app.system_manager)
        main_window.setWindowTitle("超级量子系统 - 简易版")
        
        # 设置窗口大小和位置
        main_window.resize(1200, 800)
        main_window.move(100, 100)
        
        # 创建一个延迟定时器，以便在应用程序启动后执行其他操作
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(lambda: splash.finish(main_window))
        timer.start(1000)  # 1秒后关闭启动画面
        
        # 显示主窗口
        main_window.show()
        
        # 运行应用程序
        return app.exec_()
    except Exception as e:
        # 记录错误
        logger.error(f"启动量子系统时出错: {str(e)}")
        
        # 显示错误消息
        QMessageBox.critical(None, "错误", f"启动量子系统时出错: {str(e)}")
        
        return 1

if __name__ == "__main__":
    sys.exit(main()) 