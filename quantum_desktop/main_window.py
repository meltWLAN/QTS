#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
量子桌面应用程序 - 主窗口
"""

import os
import sys
import logging
import json
from datetime import datetime
from PyQt5.QtWidgets import QAction, QMessageBox
from PyQt5.QtCore import Qt

# 设置日志
logger = logging.getLogger("QuantumDesktop")

class MainWindow:
    def _create_menus(self):
        """创建菜单"""
        # 主菜单栏
        self.menu_bar = self.menuBar()
        
        # 系统菜单
        system_menu = self.menu_bar.addMenu("系统")
        
        # 启动系统
        start_action = QAction("启动系统", self)
        start_action.triggered.connect(self._start_system)
        system_menu.addAction(start_action)
        
        # 停止系统
        stop_action = QAction("停止系统", self)
        stop_action.triggered.connect(self._stop_system)
        system_menu.addAction(stop_action)
        
        system_menu.addSeparator()
        
        # 量子选股验证
        validate_action = QAction("量子选股验证", self)
        validate_action.triggered.connect(self._launch_validation_tool)
        system_menu.addAction(validate_action)
        
        system_menu.addSeparator()
        
        # 退出
        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        system_menu.addAction(exit_action)
        
        # 视图菜单
        view_menu = self.menu_bar.addMenu("视图")
        
        # 切换暗色/亮色主题
        toggle_theme = QAction("切换主题", self)
        toggle_theme.triggered.connect(self._toggle_theme)
        view_menu.addAction(toggle_theme)
        
        # 帮助菜单
        help_menu = self.menu_bar.addMenu("帮助")
        
        # 关于
        about_action = QAction("关于", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _launch_validation_tool(self):
        """启动量子选股验证工具"""
        try:
            # 检查验证工具是否存在
            script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                       "quantum_desktop", "scripts", "validate_stock_finder.py")
            
            if not os.path.exists(script_path):
                QMessageBox.critical(self, "错误", "无法找到验证工具脚本")
                return
                
            # 启动验证工具
            # 从当前模块导入
            from quantum_desktop.scripts.validate_stock_finder import ValidationWindow
            
            # 创建验证窗口
            self.validation_window = ValidationWindow()
            self.validation_window.show()
            
            # 消息
            self.statusBar().showMessage("已启动量子选股验证工具", 3000)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动验证工具时出错: {str(e)}")
            logger.error(f"启动验证工具时出错: {str(e)}") 