#!/usr/bin/env python3
"""
超神量子共生系统 - 桌面版启动器
"""

import sys
import os
from PyQt5.QtWidgets import QApplication
from supergod_desktop import SupergodDesktop

def main():
    """启动桌面版"""
    # 创建应用程序实例
    app = QApplication(sys.argv)
    
    # 设置应用程序样式
    app.setStyle('Fusion')
    
    # 创建主窗口
    window = SupergodDesktop()
    window.show()
    
    # 运行应用程序
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 