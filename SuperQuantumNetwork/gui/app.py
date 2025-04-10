#!/usr/bin/env python3
"""
超神系统 GUI 应用
集成量子共生网络和高维统一场，提供超预测交易界面
"""

import os
import sys
import logging
import traceback
from PyQt5.QtWidgets import QApplication, QSplashScreen, QMessageBox, QMainWindow, QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap

from gui.views.main_window import SuperTradingMainWindow as MainWindow

logger = logging.getLogger("SuperGodApp")

class SuperGodApp:
    """超神系统应用类
    
    集成所有系统组件，管理主窗口
    """
    
    def __init__(self, controllers=None, quantum_components=None, symbiotic_core=None):
        """初始化超神系统应用
        
        Args:
            controllers: 控制器字典
            quantum_components: 量子组件字典
            symbiotic_core: 量子共生核心
        """
        self.logger = logging.getLogger("SuperGodApp")
        self.logger.info("初始化超神系统应用...")
        
        # 存储组件引用
        self.controllers = controllers or {}
        self.quantum_components = quantum_components or {}
        self.symbiotic_core = symbiotic_core
        
        # 显示启动屏
        self._show_splash_screen()
        
        # 初始化市场共生意识
        self._initialize_market_consciousness()
        
        # 初始化交易信号生成器
        self._initialize_signal_generator()
        
        # 连接控制器
        self._connect_controllers()
        
        # 创建主窗口
        self.main_window = self._create_main_window()
        
        # 隐藏启动屏
        if hasattr(self, 'splash') and self.splash:
            self.splash.finish(self.main_window)
        
        # 应用状态
        self.app_state = {
            "initialized": True,
            "quantum_connected": self.symbiotic_core is not None,
            "cosmic_active": self.symbiotic_core and hasattr(self.symbiotic_core, 'field_state') and 
                            self.symbiotic_core.field_state.get("active", False),
            "controllers_ready": all(c is not None for c in self.controllers.values())
        }
        
        self.logger.info("超神系统应用初始化完成")
    
    def _show_splash_screen(self):
        """显示启动屏"""
        try:
            splash_path = os.path.join(os.path.dirname(__file__), "assets", "splash.png")
            if not os.path.exists(splash_path):
                # 如果没有启动屏图片，则跳过
                self.splash = None
                return
                
            splash_pixmap = QPixmap(splash_path)
            self.splash = QSplashScreen(splash_pixmap, Qt.WindowStaysOnTopHint)
            self.splash.show()
            self.splash.showMessage("启动超神系统...", Qt.AlignBottom | Qt.AlignCenter, Qt.white)
            
            # 处理事件循环，确保显示启动屏
            QApplication.processEvents()
        except Exception as e:
            self.logger.warning(f"显示启动屏失败: {str(e)}")
            self.splash = None
    
    def _initialize_market_consciousness(self):
        """初始化市场共生意识"""
        self.logger.info("初始化市场共生意识...")
        
        try:
            from market_symbiosis.market_consciousness import get_market_consciousness
            
            # 创建市场共生意识
            self.market_consciousness = get_market_consciousness(self.symbiotic_core)
            
            # 启动市场共生意识
            self.market_consciousness.start()
            
            # 更新启动屏
            if hasattr(self, 'splash') and self.splash:
                self.splash.showMessage("市场共生意识已连接...", Qt.AlignBottom | Qt.AlignCenter, Qt.white)
                QApplication.processEvents()
                
            self.logger.info("市场共生意识已初始化")
            
        except Exception as e:
            self.logger.error(f"初始化市场共生意识失败: {str(e)}")
            self.market_consciousness = None
    
    def _initialize_signal_generator(self):
        """初始化交易信号生成器"""
        self.logger.info("初始化交易信号生成器...")
        
        try:
            from trading_signals.quantum_signal_generator import QuantumSignalGenerator
            
            # 创建信号生成器
            self.signal_generator = QuantumSignalGenerator(
                symbiotic_core=self.symbiotic_core,
                market_consciousness=self.market_consciousness
            )
            
            # 启动信号生成器
            self.signal_generator.start()
            
            # 更新启动屏
            if hasattr(self, 'splash') and self.splash:
                self.splash.showMessage("量子交易信号生成器已连接...", Qt.AlignBottom | Qt.AlignCenter, Qt.white)
                QApplication.processEvents()
                
            self.logger.info("交易信号生成器已初始化")
            
            # 连接到交易控制器
            if 'trading' in self.controllers:
                self.controllers['trading'].set_signal_generator(self.signal_generator)
                
        except Exception as e:
            self.logger.error(f"初始化交易信号生成器失败: {str(e)}")
            self.signal_generator = None
    
    def _connect_controllers(self):
        """连接控制器"""
        self.logger.info("连接控制器...")
        
        try:
            # 连接控制器之间的关系
            if 'data' in self.controllers and 'trading' in self.controllers:
                # 数据控制器 -> 交易控制器
                self.controllers['trading'].set_data_controller(self.controllers['data'])
                
            if 'portfolio' in self.controllers and 'trading' in self.controllers:
                # 投资组合控制器 -> 交易控制器
                self.controllers['trading'].set_portfolio_controller(self.controllers['portfolio'])
                
            if 'data' in self.controllers and 'portfolio' in self.controllers:
                # 数据控制器 -> 投资组合控制器
                self.controllers['portfolio'].set_data_controller(self.controllers['data'])
                
            # 连接市场共生意识
            if self.market_consciousness:
                if 'trading' in self.controllers:
                    # 注册交易控制器为观察者
                    self.market_consciousness.register_observer(self.controllers['trading'])
                    
                if 'portfolio' in self.controllers:
                    # 注册投资组合控制器为观察者
                    self.market_consciousness.register_observer(self.controllers['portfolio'])
                    
            # 更新启动屏
            if hasattr(self, 'splash') and self.splash:
                self.splash.showMessage("控制器连接完成，准备启动界面...", Qt.AlignBottom | Qt.AlignCenter, Qt.white)
                QApplication.processEvents()
                
            self.logger.info("控制器连接完成")
            
        except Exception as e:
            self.logger.error(f"连接控制器失败: {str(e)}")
            traceback.print_exc()
    
    def _create_main_window(self):
        """创建主窗口
        
        Returns:
            MainWindow: 创建的主窗口实例
        """
        self.logger.info("创建主窗口...")
        
        try:
            # 启动窗口更新
            if hasattr(self, 'splash') and self.splash:
                self.splash.showMessage("创建主界面...", Qt.AlignBottom | Qt.AlignCenter, Qt.white)
                QApplication.processEvents()
            
            # 从controllers字典中获取所需的控制器
            data_controller = self.controllers.get('data')
            trading_controller = self.controllers.get('trading')
            
            if not data_controller or not trading_controller:
                self.logger.error("缺少必要的控制器: data_controller或trading_controller")
                raise ValueError("缺少必要的控制器")
            
            # 创建主窗口
            main_window = MainWindow(
                data_controller=data_controller,
                trading_controller=trading_controller
            )
            
            # 设置窗口标题
            main_window.setWindowTitle("超神系统 - 量子共生高维统一场交易平台")
            
            # 最大化窗口
            main_window.showMaximized()
            
            self.logger.info("主窗口创建完成")
            return main_window
            
        except Exception as e:
            self.logger.error(f"创建主窗口失败: {str(e)}")
            traceback.print_exc()
            
            # 显示错误消息
            error_msg = QMessageBox()
            error_msg.setIcon(QMessageBox.Critical)
            error_msg.setWindowTitle("启动错误")
            error_msg.setText("创建主窗口时发生错误")
            error_msg.setDetailedText(str(e))
            error_msg.exec_()
            
            # 创建一个最小化的主窗口
            try:
                # 尝试创建一个简单版本的窗口
                minimal_window = QMainWindow()
                minimal_window.setWindowTitle("超神系统 - 紧急模式")
                minimal_window.resize(800, 600)
                
                central_widget = QWidget()
                minimal_window.setCentralWidget(central_widget)
                
                layout = QVBoxLayout(central_widget)
                
                error_label = QLabel("GUI加载失败，请检查系统日志")
                error_label.setStyleSheet("font-size: 18px; color: red;")
                layout.addWidget(error_label)
                
                detail_label = QLabel(str(e))
                detail_label.setWordWrap(True)
                layout.addWidget(detail_label)
                
                return minimal_window
            except:
                # 如果还是失败，返回一个空白窗口
                self.logger.critical("无法创建任何类型的窗口，系统异常")
                return QMainWindow()
    
    def shutdown(self):
        """关闭应用"""
        self.logger.info("关闭超神系统应用...")
        
        # 停止市场共生意识
        if hasattr(self, 'market_consciousness') and self.market_consciousness:
            self.market_consciousness.stop()
            
        # 停止信号生成器
        if hasattr(self, 'signal_generator') and self.signal_generator:
            self.signal_generator.stop()
            
        # 关闭高维统一场
        if self.symbiotic_core and hasattr(self.symbiotic_core, 'deactivate_field'):
            self.symbiotic_core.deactivate_field()
            
        self.logger.info("超神系统应用已关闭")


def main():
    """主函数"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 创建QApplication实例
    app = QApplication(sys.argv)
    
    # 创建超神应用
    supergod_app = SuperGodApp()
    
    # 显示主窗口
    supergod_app.main_window.show()
    
    # 执行应用
    exit_code = app.exec_()
    
    # 关闭应用
    supergod_app.shutdown()
    
    return exit_code

if __name__ == "__main__":
    sys.exit(main()) 