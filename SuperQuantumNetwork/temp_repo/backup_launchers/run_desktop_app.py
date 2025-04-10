#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - 桌面客户端启动脚本
"""

import sys
import os
import logging
import traceback

# 确保脚本可以在任何目录下运行
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

try:
    # 检查是否需要跳过启动画面
    no_splash = "--no-splash" in sys.argv
    
    # 如果有--no-splash参数，则从sys.argv中移除，以免影响其他代码
    if no_splash and "--no-splash" in sys.argv:
        sys.argv.remove("--no-splash")
    
    from gui_app import main, main_without_splash
    
    # 运行应用，根据参数决定是否显示启动画面
    if no_splash:
        sys.exit(main_without_splash())
    else:
        sys.exit(main())
    
except Exception as e:
    # 显示错误消息
    error_message = f"启动失败: {str(e)}\n{traceback.format_exc()}"
    
    # 记录到日志
    logging.error(error_message)
    
    # 尝试显示GUI错误消息
    try:
        from PyQt5.QtWidgets import QApplication, QMessageBox
        app = QApplication(sys.argv)
        QMessageBox.critical(None, "启动错误", error_message)
    except:
        # 如果无法显示GUI错误，打印到控制台
        print("启动错误:", file=sys.stderr)
        print(error_message, file=sys.stderr)
    
    sys.exit(1) 