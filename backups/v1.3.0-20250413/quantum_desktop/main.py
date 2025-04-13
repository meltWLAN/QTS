#!/usr/bin/env python3
"""
超神量子核心 - 桌面版
高性能量子计算与市场分析系统
"""

import sys
import os
import logging
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QMainWindow, QSplashScreen, QDesktopWidget
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QIcon, QFont, QFontDatabase
import json
import platform

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)

# 配置日志
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file = os.path.join(log_dir, f"quantum_desktop_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("QuantumDesktop")

# 导入自定义模块
from ui.main_window import MainWindow
from core.system_manager import SystemManager
from utils.config_manager import ConfigManager
from utils.style_manager import StyleManager

class QuantumDesktopApp:
    """超神量子核心桌面应用程序"""
    
    def __init__(self):
        """初始化应用程序"""
        # 创建应用实例
        self.app = QApplication(sys.argv)
        self.app.setApplicationName("量子交易系统")
        self.app.setOrganizationName("量子科技")
        self.app.setOrganizationDomain("quantum-tech.org")
        
        # 加载配置
        self.config = self.load_config()
        
        # 设置样式
        self.set_application_style()
        
        # 创建主窗口
        self.main_window = None
        self.splash = None
        
        # 初始化了系统管理器
        self.system_manager = None
        
        # 日志
        self.logger = logging.getLogger("quantum_desktop.main")
        self.logger.info("量子桌面应用程序初始化")
        
        # 显示启动画面
        self.show_splash_screen()
        
        # 初始化系统管理器 - 默认启用自动启动
        self.system_manager = SystemManager(auto_start=True)
        
        # 创建主窗口
        self.main_window = MainWindow(self.system_manager)
        
        # 定时器用于模拟启动时间
        self.timer = QTimer()
        self.timer.timeout.connect(self.initialize_app)
        self.timer.start(2000)  # 2秒后初始化
        
        logger.info("应用程序初始化完成")
        
    def load_config(self):
        """加载系统配置"""
        try:
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config/system_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    logger.info(f"已加载系统配置: {config_path}")
                    return config
            else:
                logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
                config = {
                    "system": {"log_level": "INFO"},
                    "market_data": {"use_real_data": True},
                    "quantum": {"max_qubits": 20, "evolution_level": 1}
                }
                return config
        except Exception as e:
            logger.error(f"加载配置失败: {str(e)}")
            config = {
                "system": {"log_level": "INFO"},
                "market_data": {"use_real_data": True},
                "quantum": {"max_qubits": 20, "evolution_level": 1}
            }
            return config
        
    def set_application_style(self):
        """设置应用程序样式"""
        # 加载配置
        self.config_manager = ConfigManager()
        
        # 设置样式
        self.style_manager = StyleManager()
        self.style_manager.apply_theme(self.app, 'dark')
        
    def show_splash_screen(self):
        """显示启动画面"""
        # 创建资源目录
        resource_dir = os.path.join(os.path.dirname(__file__), "resources")
        if not os.path.exists(resource_dir):
            os.makedirs(resource_dir)
            
        # 创建默认启动画面路径
        splash_path = os.path.join(resource_dir, "splash.png")
        
        # 如果启动画面不存在，使用纯色背景
        if not os.path.exists(splash_path):
            pixmap = QPixmap(600, 400)
            pixmap.fill(Qt.darkBlue)
        else:
            pixmap = QPixmap(splash_path)
        
        self.splash = QSplashScreen(pixmap)
        self.splash.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.splash.show()
        self.app.processEvents()
        
        # 显示初始化消息
        self.splash.showMessage("正在初始化核心组件...", Qt.AlignBottom | Qt.AlignCenter, Qt.white)
        
    def initialize_app(self):
        """初始化应用程序"""
        # 停止定时器
        self.timer.stop()
        
        # 显示主窗口
        self.main_window.show()
        
        # 居中显示
        frame_geometry = self.main_window.frameGeometry()
        screen_center = QDesktopWidget().availableGeometry().center()
        frame_geometry.moveCenter(screen_center)
        self.main_window.move(frame_geometry.topLeft())
        
        # 关闭启动画面
        self.splash.finish(self.main_window)
        
        logger.info("应用程序启动完成")
        
    def run(self):
        """运行应用程序"""
        return self.app.exec_()

    def update_config(self, new_config):
        """更新配置
        
        Args:
            new_config: 新配置字典
        """
        if not new_config:
            return

        # 递归更新配置
        def update_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d:
                    d[k] = update_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        self.config = update_dict(self.config, new_config)
        self.logger.info(f"配置已更新: {self.config}")
        
        # 如果主窗口已创建，更新设置
        if self.main_window:
            self.main_window.apply_settings(self.config)

def main():
    """主函数"""
    app = QuantumDesktopApp()
    return app.run()

if __name__ == "__main__":
    sys.exit(main()) 