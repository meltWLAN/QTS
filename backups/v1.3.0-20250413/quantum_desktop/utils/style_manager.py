"""
样式管理器 - 管理应用程序主题和样式
"""

import os
import logging
from typing import Dict, Any
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPalette, QColor, QFont
from PyQt5.QtCore import Qt

logger = logging.getLogger("StyleManager")

class StyleManager:
    """样式管理器类"""
    
    def __init__(self):
        self.themes = {
            'dark': self._get_dark_theme(),
            'light': self._get_light_theme()
        }
        
    def apply_theme(self, app, theme_name='dark'):
        """应用主题"""
        if theme_name not in self.themes:
            logger.warning(f"主题 {theme_name} 不存在，使用默认主题")
            theme_name = 'dark'
            
        app.setStyleSheet(self.themes[theme_name])
        logger.info(f"已应用主题: {theme_name}")
        
    def _get_dark_theme(self):
        """获取暗色主题样式"""
        return """
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                color: #ffffff;
                background-color: #1e1e1e;
            }
            QTabWidget::pane {
                border: 1px solid #3e3e3e;
                background-color: #1e1e1e;
            }
            QTabBar::tab {
                background-color: #2d2d2d;
                color: #ffffff;
                padding: 8px 20px;
                border: 1px solid #3e3e3e;
            }
            QTabBar::tab:selected {
                background-color: #3e3e3e;
            }
            QPushButton {
                background-color: #2d2d2d;
                color: #ffffff;
                border: 1px solid #3e3e3e;
                padding: 5px 15px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #3e3e3e;
            }
            QLabel {
                color: #ffffff;
            }
            QStatusBar {
                background-color: #2d2d2d;
                color: #ffffff;
            }
        """
        
    def _get_light_theme(self):
        """获取亮色主题样式"""
        return """
            QMainWindow {
                background-color: #ffffff;
            }
            QWidget {
                color: #000000;
                background-color: #ffffff;
            }
            QTabWidget::pane {
                border: 1px solid #d0d0d0;
                background-color: #ffffff;
            }
            QTabBar::tab {
                background-color: #f0f0f0;
                color: #000000;
                padding: 8px 20px;
                border: 1px solid #d0d0d0;
            }
            QTabBar::tab:selected {
                background-color: #ffffff;
            }
            QPushButton {
                background-color: #f0f0f0;
                color: #000000;
                border: 1px solid #d0d0d0;
                padding: 5px 15px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
            QLabel {
                color: #000000;
            }
            QStatusBar {
                background-color: #f0f0f0;
                color: #000000;
            }
        """
        
    def get_current_theme(self) -> str:
        """获取当前主题名称"""
        return 'dark'  # Assuming 'dark' is the only theme available
        
    def get_available_themes(self) -> list:
        """获取可用主题列表"""
        return list(self.themes.keys())
        
    def _create_dark_theme(self) -> Dict[str, Any]:
        """创建深色主题"""
        palette = QPalette()
        
        # 设置窗口背景色
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, Qt.white)
        
        # 设置按钮颜色
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, Qt.white)
        
        # 设置链接颜色
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.LinkVisited, QColor(100, 100, 200))
        
        # 设置输入框颜色
        palette.setColor(QPalette.Base, QColor(35, 35, 35))
        palette.setColor(QPalette.AlternateBase, QColor(45, 45, 45))
        palette.setColor(QPalette.Text, Qt.white)
        
        # 设置工具提示颜色
        palette.setColor(QPalette.ToolTipBase, QColor(25, 25, 25))
        palette.setColor(QPalette.ToolTipText, Qt.white)
        
        # 设置高亮颜色
        palette.setColor(QPalette.Highlight, QColor(65, 105, 225))  # 蓝紫色
        palette.setColor(QPalette.HighlightedText, Qt.white)
        
        # 设置字体
        font = QFont("Segoe UI", 9)
        
        # 设置样式表
        stylesheet = """
        QMainWindow {
            background-color: #353535;
        }
        
        QTabWidget::pane {
            border: 1px solid #444;
            top: -1px;
            background: #353535;
        }
        
        QTabBar::tab {
            background: #353535;
            border: 1px solid #444;
            padding: 8px 12px;
            color: #fff;
        }
        
        QTabBar::tab:selected {
            background: #4527a0;
            border-bottom-color: #4527a0;
        }
        
        QTabBar::tab:!selected {
            margin-top: 3px;
        }
        
        QPushButton {
            background-color: #4527a0;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 6px 12px;
            outline: none;
        }
        
        QPushButton:hover {
            background-color: #5e35b1;
        }
        
        QPushButton:pressed {
            background-color: #3f1f89;
        }
        
        QPushButton:disabled {
            background-color: #444;
            color: #999;
        }
        
        QLineEdit {
            border: 1px solid #444;
            border-radius: 4px;
            padding: 4px;
            background: #252525;
            color: white;
        }
        
        QComboBox {
            border: 1px solid #444;
            border-radius: 4px;
            padding: 4px;
            background: #252525;
            color: white;
        }
        
        QMenuBar {
            background-color: #252525;
            color: white;
        }
        
        QMenuBar::item {
            background: transparent;
        }
        
        QMenuBar::item:selected {
            background: #4527a0;
        }
        
        QMenu {
            background-color: #252525;
            color: white;
            border: 1px solid #444;
        }
        
        QMenu::item:selected {
            background-color: #4527a0;
        }
        
        QToolBar {
            background: #252525;
            spacing: 6px;
            border-bottom: 1px solid #444;
        }
        
        QStatusBar {
            background: #252525;
            color: white;
        }
        
        QScrollBar:vertical {
            border: none;
            background: #333;
            width: 10px;
            margin: 0px;
        }
        
        QScrollBar::handle:vertical {
            background: #555;
            min-height: 20px;
            border-radius: 5px;
        }
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            border: none;
            background: none;
        }
        
        QTreeView, QListView {
            background-color: #252525;
            border: 1px solid #444;
            color: white;
        }
        
        QTreeView::item:selected, QListView::item:selected {
            background-color: #4527a0;
        }
        """
        
        return {
            'palette': palette,
            'font': font,
            'stylesheet': stylesheet
        }
        
    def _create_light_theme(self) -> Dict[str, Any]:
        """创建浅色主题"""
        palette = QPalette()
        
        # 设置窗口背景色
        palette.setColor(QPalette.Window, QColor(240, 240, 240))
        palette.setColor(QPalette.WindowText, Qt.black)
        
        # 设置按钮颜色
        palette.setColor(QPalette.Button, QColor(240, 240, 240))
        palette.setColor(QPalette.ButtonText, Qt.black)
        
        # 设置链接颜色
        palette.setColor(QPalette.Link, QColor(0, 0, 255))
        palette.setColor(QPalette.LinkVisited, QColor(80, 0, 255))
        
        # 设置输入框颜色
        palette.setColor(QPalette.Base, QColor(255, 255, 255))
        palette.setColor(QPalette.AlternateBase, QColor(233, 231, 245))
        palette.setColor(QPalette.Text, Qt.black)
        
        # 设置工具提示颜色
        palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
        palette.setColor(QPalette.ToolTipText, Qt.black)
        
        # 设置高亮颜色
        palette.setColor(QPalette.Highlight, QColor(102, 58, 183))  # 紫色
        palette.setColor(QPalette.HighlightedText, Qt.white)
        
        # 设置字体
        font = QFont("Segoe UI", 9)
        
        # 设置样式表
        stylesheet = """
        QMainWindow {
            background-color: #f0f0f0;
        }
        
        QTabWidget::pane {
            border: 1px solid #cccccc;
            top: -1px;
            background: #f0f0f0;
        }
        
        QTabBar::tab {
            background: #f0f0f0;
            border: 1px solid #cccccc;
            padding: 8px 12px;
            color: #333;
        }
        
        QTabBar::tab:selected {
            background: #673ab7;
            color: white;
            border-bottom-color: #673ab7;
        }
        
        QTabBar::tab:!selected {
            margin-top: 3px;
        }
        
        QPushButton {
            background-color: #673ab7;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 6px 12px;
            outline: none;
        }
        
        QPushButton:hover {
            background-color: #7e57c2;
        }
        
        QPushButton:pressed {
            background-color: #5e35b1;
        }
        
        QPushButton:disabled {
            background-color: #cccccc;
            color: #666666;
        }
        
        QLineEdit {
            border: 1px solid #cccccc;
            border-radius: 4px;
            padding: 4px;
            background: white;
            color: #333;
        }
        
        QComboBox {
            border: 1px solid #cccccc;
            border-radius: 4px;
            padding: 4px;
            background: white;
            color: #333;
        }
        
        QMenuBar {
            background-color: #f5f5f5;
            color: #333;
        }
        
        QMenuBar::item {
            background: transparent;
        }
        
        QMenuBar::item:selected {
            background: #673ab7;
            color: white;
        }
        
        QMenu {
            background-color: #f5f5f5;
            color: #333;
            border: 1px solid #cccccc;
        }
        
        QMenu::item:selected {
            background-color: #673ab7;
            color: white;
        }
        
        QToolBar {
            background: #f5f5f5;
            spacing: 6px;
            border-bottom: 1px solid #cccccc;
        }
        
        QStatusBar {
            background: #f5f5f5;
            color: #333;
        }
        
        QScrollBar:vertical {
            border: none;
            background: #e0e0e0;
            width: 10px;
            margin: 0px;
        }
        
        QScrollBar::handle:vertical {
            background: #bbbbbb;
            min-height: 20px;
            border-radius: 5px;
        }
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            border: none;
            background: none;
        }
        
        QTreeView, QListView {
            background-color: white;
            border: 1px solid #cccccc;
            color: #333;
        }
        
        QTreeView::item:selected, QListView::item:selected {
            background-color: #673ab7;
            color: white;
        }
        """
        
        return {
            'palette': palette,
            'font': font,
            'stylesheet': stylesheet
        }
        
    def _create_quantum_theme(self) -> Dict[str, Any]:
        """创建量子主题（特殊定制主题）"""
        palette = QPalette()
        
        # 设置窗口背景色 - 深蓝色
        palette.setColor(QPalette.Window, QColor(30, 30, 60))
        palette.setColor(QPalette.WindowText, QColor(220, 220, 255))
        
        # 设置按钮颜色
        palette.setColor(QPalette.Button, QColor(60, 60, 100))
        palette.setColor(QPalette.ButtonText, QColor(220, 220, 255))
        
        # 设置链接颜色
        palette.setColor(QPalette.Link, QColor(110, 160, 255))
        palette.setColor(QPalette.LinkVisited, QColor(160, 140, 255))
        
        # 设置输入框颜色
        palette.setColor(QPalette.Base, QColor(40, 40, 80))
        palette.setColor(QPalette.AlternateBase, QColor(50, 50, 90))
        palette.setColor(QPalette.Text, QColor(220, 220, 255))
        
        # 设置工具提示颜色
        palette.setColor(QPalette.ToolTipBase, QColor(30, 30, 60))
        palette.setColor(QPalette.ToolTipText, QColor(220, 220, 255))
        
        # 设置高亮颜色
        palette.setColor(QPalette.Highlight, QColor(100, 120, 240))
        palette.setColor(QPalette.HighlightedText, Qt.white)
        
        # 设置字体
        font = QFont("Segoe UI", 9)
        
        # 设置样式表
        stylesheet = """
        QMainWindow {
            background-color: #1e1e3c;
        }
        
        QTabWidget::pane {
            border: 1px solid #3c3c64;
            top: -1px;
            background: #1e1e3c;
        }
        
        QTabBar::tab {
            background: #3c3c64;
            border: 1px solid #3c3c64;
            padding: 8px 12px;
            color: #dcdcff;
        }
        
        QTabBar::tab:selected {
            background: #5050a0;
            border-bottom-color: #5050a0;
        }
        
        QTabBar::tab:!selected {
            margin-top: 3px;
        }
        
        QPushButton {
            background-color: #5050a0;
            color: #dcdcff;
            border: none;
            border-radius: 4px;
            padding: 6px 12px;
            outline: none;
        }
        
        QPushButton:hover {
            background-color: #6060b0;
        }
        
        QPushButton:pressed {
            background-color: #404090;
        }
        
        QPushButton:disabled {
            background-color: #2a2a4a;
            color: #7a7aa0;
        }
        
        QLineEdit {
            border: 1px solid #3c3c64;
            border-radius: 4px;
            padding: 4px;
            background: #2a2a4a;
            color: #dcdcff;
        }
        
        QComboBox {
            border: 1px solid #3c3c64;
            border-radius: 4px;
            padding: 4px;
            background: #2a2a4a;
            color: #dcdcff;
        }
        
        QMenuBar {
            background-color: #2a2a4a;
            color: #dcdcff;
        }
        
        QMenuBar::item {
            background: transparent;
        }
        
        QMenuBar::item:selected {
            background: #5050a0;
        }
        
        QMenu {
            background-color: #2a2a4a;
            color: #dcdcff;
            border: 1px solid #3c3c64;
        }
        
        QMenu::item:selected {
            background-color: #5050a0;
        }
        
        QToolBar {
            background: #2a2a4a;
            spacing: 6px;
            border-bottom: 1px solid #3c3c64;
        }
        
        QStatusBar {
            background: #2a2a4a;
            color: #dcdcff;
        }
        
        QScrollBar:vertical {
            border: none;
            background: #2a2a4a;
            width: 10px;
            margin: 0px;
        }
        
        QScrollBar::handle:vertical {
            background: #5050a0;
            min-height: 20px;
            border-radius: 5px;
        }
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            border: none;
            background: none;
        }
        
        QTreeView, QListView {
            background-color: #2a2a4a;
            border: 1px solid #3c3c64;
            color: #dcdcff;
        }
        
        QTreeView::item:selected, QListView::item:selected {
            background-color: #5050a0;
        }
        """
        
        return {
            'palette': palette,
            'font': font,
            'stylesheet': stylesheet
        }
        
    def _create_system_theme(self) -> Dict[str, Any]:
        """创建系统默认主题"""
        # 使用应用程序默认调色板
        palette = QPalette()
        font = QFont()
        
        return {
            'palette': palette,
            'font': font,
            'stylesheet': ""
        } 