#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - 主窗口
实现系统的主界面，整合各个功能模块
"""

from PyQt5.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QStatusBar, QAction, QToolBar, QSplitter, QMessageBox, QFrame, QProgressBar, QPushButton, QDialog, QGroupBox, QLineEdit,
    QListWidget, QListWidgetItem, QGridLayout, QFileDialog
)
from PyQt5.QtCore import Qt, QSize, QMetaObject, Q_ARG, pyqtSignal
from PyQt5.QtGui import QIcon, QFont, QPixmap, QMovie
import qtawesome as qta
import logging
import socket
import threading
import sys
import numpy as np
import os
import json
from datetime import datetime

# 设置日志
logger = logging.getLogger("SuperTradingMainWindow")

# 导入量子预测视图
try:
    from gui.views.quantum_prediction_view import QuantumPredictionView
except ImportError as e:
    logger.error(f"导入量子预测视图失败: {str(e)}")

# 导入市场视图
try:
    from gui.views.market_view import RealTimeMarketView
    MARKET_VIEW_AVAILABLE = True
except ImportError as e:
    MARKET_VIEW_AVAILABLE = False
    logger.error(f"导入市场视图失败: {str(e)}")

# 导入量子意识视图和宇宙视图
from gui.views.consciousness_view import ConsciousnessView
from gui.controllers.consciousness_controller import ConsciousnessController
from gui.views.cosmic_view import CosmicView
from gui.controllers.cosmic_controller import CosmicController

class SuperTradingMainWindow(QMainWindow):
    """超级交易系统主窗口"""
    
    # 类变量，保存单例实例
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """获取实例，实现单例模式"""
        return cls._instance
    
    def __init__(self, data_controller, trading_controller, parent=None):
        """初始化主窗口"""
        super().__init__(parent)
        
        # 保存实例引用，实现单例模式
        SuperTradingMainWindow._instance = self
        
        # 保存控制器引用
        self.data_controller = data_controller
        self.trading_controller = trading_controller
        
        # 设置logger
        self.logger = logger
        
        # 设置基本属性
        self.setWindowTitle("超神量子共生网络交易系统 v1.0.0")
        self.resize(1280, 800)
        
        # 初始化控制器
        self.init_controllers()
        
        # 设置UI
        self._setup_ui()
        
        # 显示欢迎消息
        self.statusBar().showMessage("系统准备就绪")
        
        # 启动监听线程，处理来自其他实例的激活请求
        self._start_activation_listener()
        
        # 连接信号和槽
        self.connect_signals_slots()
        
        # 设置初始标签页
        self.init_tabs()
        
        # 记录日志
        self.logger.info("主窗口初始化完成")
    
    def init_controllers(self):
        """初始化所有控制器"""
        # 初始化量子意识控制器
        self.consciousness_controller = ConsciousnessController(self)
        
        # 初始化宇宙控制器
        self.cosmic_controller = CosmicController(self)
    
    def _start_activation_listener(self):
        """启动激活监听线程"""
        def listen_for_activation():
            # 使用全局套接字
            try:
                # 尝试获取gui_app模块
                gui_app = None
                try:
                    if 'gui_app' in sys.modules:
                        gui_app = sys.modules['gui_app']
                except Exception as e:
                    logging.warning(f"获取gui_app模块失败: {str(e)}")
                
                # 尝试获取套接字
                sock = None
                if gui_app:
                    sock = getattr(gui_app, '_socket_lock', None)
                
                if sock:
                    while True:
                        # 接受连接
                        client, addr = sock.accept()
                        # 获取数据
                        data = client.recv(1024)
                        client.close()
                        # 如果收到激活命令，则激活窗口
                        if data == b'ACTIVATE':
                            # 使用信号/槽机制在主线程中激活窗口
                            QMetaObject.invokeMethod(self, "_activate_window", 
                                                   Qt.QueuedConnection)
                else:
                    logging.warning("获取激活套接字失败，激活监听功能不可用")
            except Exception as e:
                logging.error(f"激活监听线程出错: {str(e)}")
        
        # 创建并启动线程
        thread = threading.Thread(target=listen_for_activation, daemon=True)
        thread.start()
    
    def _activate_window(self):
        """激活窗口（在主线程中调用）"""
        # 激活窗口，放到前台
        self.setWindowState(self.windowState() & ~Qt.WindowMinimized | Qt.WindowActive)
        self.activateWindow()
        self.raise_()
        self.show()
        
        # 显示一个消息
        self.statusBar().showMessage("超神系统已被激活", 3000)
    
    def _setup_ui(self):
        """设置UI"""
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 创建主选项卡部件
        self.main_tab = QTabWidget()
        self.main_tab.setTabPosition(QTabWidget.North)
        self.main_tab.setDocumentMode(True)
        main_layout.addWidget(self.main_tab)
        
        # 创建顶部搜索栏
        self._setup_search_bar()
        
        # 创建并添加市场视图选项卡
        if MARKET_VIEW_AVAILABLE:
            try:
                self.market_view = RealTimeMarketView(self.data_controller)
                self.main_tab.addTab(self.market_view, qta.icon('fa5s.chart-line'), "市场")
                logger.info("成功加载实时市场视图")
            except Exception as e:
                logger.error(f"加载实时市场视图失败: {str(e)}")
                self._add_simplified_market_view()
        else:
            self._add_simplified_market_view()
        
        # 创建并添加交易视图选项卡 - 使用超神级视图
        try:
            self.trading_view = self._create_trading_view()
            self.main_tab.addTab(self.trading_view, qta.icon('fa5s.exchange-alt'), "交易")
            if self.trading_controller:
                self.trading_view.controller = self.trading_controller
                self.trading_controller.view = self.trading_view
            self.logger.info("成功加载超神交易视图")
        except Exception as e:
            self.logger.error(f"加载超神交易视图失败: {str(e)}")
            # 创建一个备用视图，带有重新加载按钮
            trading_tab = QWidget()
            trading_layout = QVBoxLayout(trading_tab)
            
            title_label = QLabel("超神交易系统")
            title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #7700ff;")
            title_label.setAlignment(Qt.AlignCenter)
            trading_layout.addWidget(title_label)
            
            error_label = QLabel(f"交易视图加载失败: {str(e)}")
            error_label.setStyleSheet("color: red; margin: 20px;")
            error_label.setAlignment(Qt.AlignCenter)
            trading_layout.addWidget(error_label)
            
            # 添加重新加载按钮
            reload_btn = QPushButton("强制加载超神模块")
            reload_btn.clicked.connect(self._try_reload_trading_view)
            trading_layout.addWidget(reload_btn)
            
            self.main_tab.addTab(trading_tab, qta.icon('fa5s.exchange-alt'), "交易")
        
        # 创建并添加量子网络视图选项卡 - 使用超神级视图
        try:
            from gui.views.quantum_view import SuperQuantumNetworkView
    
            self.quantum_view = SuperQuantumNetworkView()
            self.main_tab.addTab(self.quantum_view, qta.icon('fa5s.atom'), "超神量子网络")
            logging.info("成功加载超神量子网络视图")
        except ImportError as e:
            logging.error(f"缺少量子网络视图依赖: {str(e)}")
            # 创建一个简单的备用量子网络视图
            quantum_tab = QWidget()
            quantum_layout = QVBoxLayout(quantum_tab)
            
            title_label = QLabel("超神量子网络")
            title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: cyan;")
            title_label.setAlignment(Qt.AlignCenter)
            quantum_layout.addWidget(title_label)
            
            error_label = QLabel(f"量子网络视图加载失败: 缺少必要依赖\n{str(e)}")
            error_label.setStyleSheet("color: red; margin: 20px;")
            error_label.setAlignment(Qt.AlignCenter)
            quantum_layout.addWidget(error_label)
            
            # 添加重新加载按钮
            reload_btn = QPushButton("强制加载量子网络")
            reload_btn.clicked.connect(lambda: QMessageBox.information(self, "提示", "请先安装必要依赖"))
            quantum_layout.addWidget(reload_btn)
            
            self.main_tab.addTab(quantum_tab, qta.icon('fa5s.atom'), "超神量子网络")
        
        # 创建并添加投资组合视图选项卡 - 使用超神级视图
        try:
            self.portfolio_view = self._create_portfolio_view()
            self.main_tab.addTab(self.portfolio_view, qta.icon('fa5s.chart-pie'), "投资组合")
            self.logger.info("成功加载超神投资组合视图")
        except Exception as e:
            self.logger.error(f"加载超神投资组合视图失败: {str(e)}")
            # 创建一个备用视图，带有重新加载按钮
            portfolio_tab = QWidget()
            portfolio_layout = QVBoxLayout(portfolio_tab)
            
            title_label = QLabel("超神投资组合管理")
            title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #7700ff;")
            title_label.setAlignment(Qt.AlignCenter)
            portfolio_layout.addWidget(title_label)
            
            error_label = QLabel(f"投资组合视图加载失败: {str(e)}")
            error_label.setStyleSheet("color: red; margin: 20px;")
            error_label.setAlignment(Qt.AlignCenter)
            portfolio_layout.addWidget(error_label)
            
            # 添加重新加载按钮
            reload_btn = QPushButton("强制加载超神模块")
            reload_btn.clicked.connect(self._try_reload_portfolio_view)
            portfolio_layout.addWidget(reload_btn)
            
            self.main_tab.addTab(portfolio_tab, qta.icon('fa5s.chart-pie'), "投资组合")
        
        # 创建并添加量子预测视图选项卡
        if "QuantumPredictionView" in globals():
            try:
                self.prediction_view = QuantumPredictionView(data_controller=self.data_controller)
                self.main_tab.addTab(self.prediction_view, qta.icon('fa5s.brain'), "量子预测")
                
                # 连接预测器
                if hasattr(self.data_controller, 'predictor'):
                    self.predictor = self.data_controller.predictor
                    self.prediction_view.set_predictor(self.predictor)
                    
                # 存储可视化器引用
                self.prediction_visualizer = self.prediction_view
                
                logger.info("成功加载量子预测视图")
            except Exception as e:
                logger.error(f"加载量子预测视图失败: {str(e)}")
        
        # 创建并添加宇宙意识视图选项卡
        try:
            self.consciousness_view = ConsciousnessView()
            self.main_tab.addTab(self.consciousness_view, qta.icon('fa5s.eye'), "量子意识")
            
            # 连接意识视图与控制器
            self.consciousness_view.controller = self.consciousness_controller
            self.consciousness_controller.view = self.consciousness_view
            
            logger.info("成功加载量子意识视图")
        except Exception as e:
            logger.error(f"加载量子意识视图失败: {str(e)}")
        
        # 创建并添加宇宙共振视图选项卡
        try:
            self.cosmic_view = CosmicView()
            self.main_tab.addTab(self.cosmic_view, qta.icon('fa5s.stream'), "宇宙共振")
            
            # 连接宇宙视图与控制器
            self.cosmic_view.controller = self.cosmic_controller
            self.cosmic_controller.view = self.cosmic_view
            
            logger.info("成功加载宇宙共振视图")
        except Exception as e:
            logger.error(f"加载宇宙共振视图失败: {str(e)}")
            
        # 设置状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # 创建工具栏
        self.toolbar = QToolBar("主工具栏")
        self.toolbar.setIconSize(QSize(20, 20))
        self.addToolBar(self.toolbar)
        
        # 添加设置按钮
        settings_action = QAction(qta.icon('fa5s.cog'), "设置", self)
        settings_action.triggered.connect(self.show_settings_dialog)
        self.toolbar.addAction(settings_action)
        
        # 添加刷新按钮
        refresh_action = QAction(qta.icon('fa5s.sync-alt'), "刷新", self)
        refresh_action.triggered.connect(self.refresh_data)
        self.toolbar.addAction(refresh_action)
    
    def _add_simplified_market_view(self):
        """添加简化版市场视图"""
        market_tab = QWidget()
        market_layout = QVBoxLayout(market_tab)
        
        # 添加标题
        title_label = QLabel("市场视图 - 简化版本")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: cyan;")
        title_label.setAlignment(Qt.AlignCenter)
        market_layout.addWidget(title_label)
        
        # 添加提示
        hint_label = QLabel("提示: 安装完整版依赖后可使用实时市场数据")
        hint_label.setStyleSheet("color: yellow; font-weight: bold; margin: 10px;")
        hint_label.setAlignment(Qt.AlignCenter)
        market_layout.addWidget(hint_label)
        
        # 创建一个简单的市场状态显示
        status_frame = QFrame()
        status_frame.setFrameShape(QFrame.StyledPanel)
        status_frame.setStyleSheet("background-color: #1a1a1a; padding: 20px;")
        status_layout = QVBoxLayout(status_frame)
        
        # 添加几个静态指数
        for index_name, color in [("上证指数", "red"), ("深证成指", "green"), 
                                  ("创业板指", "orange"), ("科创50", "cyan")]:
            index_layout = QHBoxLayout()
            index_layout.addWidget(QLabel(f"{index_name}:"))
            
            # 随机值
            value_label = QLabel(f"{3000 + 1000*np.random.random():.2f}")
            value_label.setStyleSheet(f"color: {color}; font-weight: bold;")
            index_layout.addWidget(value_label)
            
            # 随机涨跌幅
            change = (np.random.random() - 0.5) * 5
            change_label = QLabel(f"{change:+.2f}%")
            change_label.setStyleSheet(f"color: {'red' if change > 0 else 'green'}; font-weight: bold;")
            index_layout.addWidget(change_label)
            
            status_layout.addLayout(index_layout)
        
        market_layout.addWidget(status_frame)
        
        # 添加安装依赖按钮
        install_button = QPushButton("安装完整市场视图依赖")
        install_button.clicked.connect(lambda: self.statusBar().showMessage("请使用pip安装: pyqtgraph numpy pandas", 5000))
        market_layout.addWidget(install_button)
        
        market_layout.addStretch(1)
        self.main_tab.addTab(market_tab, qta.icon('fa5s.chart-line'), "市场")
    
    def show_settings_dialog(self):
        """显示设置对话框"""
        # 创建设置对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("系统设置")
        dialog.setFixedSize(400, 300)
        
        # 创建布局
        layout = QVBoxLayout(dialog)
        
        # 创建选项组
        api_group = QGroupBox("API设置")
        api_layout = QVBoxLayout(api_group)
        
        # Tushare Token
        token_layout = QHBoxLayout()
        token_layout.addWidget(QLabel("Tushare Token:"))
        token_edit = QLineEdit()
        token_edit.setText(self.data_controller.tushare_token)
        token_layout.addWidget(token_edit)
        api_layout.addLayout(token_layout)
        
        # 自动更新
        update_layout = QHBoxLayout()
        update_layout.addWidget(QLabel("自动更新间隔(秒):"))
        update_edit = QLineEdit("60")
        update_layout.addWidget(update_edit)
        api_layout.addLayout(update_layout)
        
        layout.addWidget(api_group)
        
        # 主题设置
        theme_group = QGroupBox("主题设置")
        theme_layout = QVBoxLayout(theme_group)
        
        theme_layout.addWidget(QLabel("当前主题: 超神量子暗"))
        theme_layout.addWidget(QLabel("更多主题即将推出..."))
        
        layout.addWidget(theme_group)
        
        # 按钮
        button_layout = QHBoxLayout()
        ok_button = QPushButton("确定")
        cancel_button = QPushButton("取消")
        
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
        # 连接信号
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)
        
        # 显示对话框
        if dialog.exec_() == QDialog.Accepted:
            # 更新设置
            new_token = token_edit.text().strip()
            if new_token != self.data_controller.tushare_token:
                self.data_controller.tushare_token = new_token
                self.statusBar().showMessage("Token已更新，重启系统后生效", 3000)
    
    def update_market_status(self, status_data):
        """更新市场状态"""
        status = status_data.get("status", "未知")
        
        # 设置状态文本和颜色
        if status == "交易中":
            self.market_status.setText(f"市场: {status}")
            self.market_status.setStyleSheet("color: #00ff00;")
        elif status == "已收盘":
            self.market_status.setText(f"市场: {status}")
            self.market_status.setStyleSheet("color: #ff9900;")
        elif status == "休市":
            self.market_status.setText(f"市场: {status}")
            self.market_status.setStyleSheet("color: #ff3333;")
        else:
            self.market_status.setText(f"市场: {status}")
            self.market_status.setStyleSheet("color: #cccccc;")
    
    def refresh_data(self):
        """刷新数据"""
        self.statusBar().showMessage("正在刷新数据...", 2000)
        
        # 刷新市场视图数据
        if hasattr(self, 'market_view') and hasattr(self.market_view, 'refresh_data'):
            try:
                self.market_view.refresh_data()
            except Exception as e:
                logger.error(f"刷新市场数据失败: {str(e)}")
        
        # 刷新量子网络视图数据
        if hasattr(self, 'quantum_view') and hasattr(self.quantum_view, 'refresh_data'):
            try:
                self.quantum_view.refresh_data()
            except Exception as e:
                logger.error(f"刷新量子网络数据失败: {str(e)}")
        
        # 刷新量子预测视图数据
        if hasattr(self, 'quantum_prediction_view') and hasattr(self.quantum_prediction_view, 'refresh_data'):
            try:
                self.quantum_prediction_view.refresh_data()
            except Exception as e:
                logger.error(f"刷新量子预测数据失败: {str(e)}")
        
        # 刷新量子意识视图数据
        if hasattr(self, 'consciousness_view') and hasattr(self.consciousness_view, 'refresh_data'):
            try:
                self.consciousness_view.refresh_data()
            except Exception as e:
                logger.error(f"刷新量子意识数据失败: {str(e)}")
    
    def connect_signals_slots(self):
        """连接信号和槽"""
        # 连接量子意识相关信号
        self.consciousness_controller.consciousness_updated.connect(
            self.consciousness_view.update_consciousness_state
        )
        self.consciousness_controller.insights_updated.connect(
            self.consciousness_view.update_insights
        )
        self.consciousness_view.update_requested.connect(
            self.consciousness_controller.update_consciousness_state
        )
        
        # 连接宇宙共振相关信号
        self.cosmic_controller.resonance_updated.connect(
            self.cosmic_view.update_resonance_state
        )
        self.cosmic_controller.cosmic_events_updated.connect(
            self.cosmic_view.update_cosmic_events
        )
        self.cosmic_controller.energy_level_updated.connect(
            self.cosmic_view.update_energy_level
        )
        self.cosmic_controller.market_patterns_updated.connect(
            self.cosmic_view.update_market_patterns
        )
        self.cosmic_view.update_requested.connect(
            self.cosmic_controller.update_resonance_state
        )
        self.cosmic_view.analyze_market_requested.connect(
            lambda: self.cosmic_controller.analyze_market_patterns(self.data_controller.get_market_data())
        )
        
        # 在数据控制器中使用量子意识增强预测
        self.data_controller.predictor_enhanced = self.enhance_prediction
    
    def init_tabs(self):
        """初始化标签页"""
        # 设置初始标签页
        self.main_tab.setCurrentIndex(0)
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        # 关闭量子意识控制器
        if hasattr(self, 'consciousness_controller'):
            self.consciousness_controller.shutdown()
        
        # 关闭宇宙控制器
        if hasattr(self, 'cosmic_controller'):
            self.cosmic_controller.shutdown()
        
        super().closeEvent(event)
    
    def enhance_prediction(self, prediction_data):
        """增强预测结果"""
        try:
            self.logger.info("开始增强预测数据")
            if not prediction_data:
                self.logger.warning("预测数据为空，无法增强")
                return prediction_data
                
            # 使用量子预测器增强
            if hasattr(self, 'predictor') and hasattr(self.predictor, 'enhance_prediction'):
                prediction_data = self.predictor.enhance_prediction(prediction_data)
                logger.info("使用量子预测器增强完成")
                
            # 使用量子意识增强
            if hasattr(self, 'consciousness_controller'):
                prediction_data = self.consciousness_controller.enhance_prediction(prediction_data)
                logger.info("使用量子意识增强完成")
            
            # 使用宇宙共振进一步增强
            if hasattr(self, 'cosmic_engine'):
                # 增强预测数据中的数值
                if "predictions" in prediction_data and isinstance(prediction_data["predictions"], list):
                    prediction_data["predictions"] = self.cosmic_engine.apply_cosmic_filter(prediction_data["predictions"])
                
                # 获取宇宙事件并添加到预测中
                try:
                    # 尝试调用generate_cosmic_events方法
                    cosmic_events = self.cosmic_engine.generate_cosmic_events(prediction_data.get('stock_code', ''))
                except AttributeError:
                    # 如果找不到方法，添加方法别名
                    try:
                        if not hasattr(self.cosmic_engine, '_generate_cosmic_events'):
                            self.cosmic_engine._generate_cosmic_events = self.cosmic_engine.generate_cosmic_events
                        cosmic_events = self.cosmic_engine._generate_cosmic_events(prediction_data.get('stock_code', ''))
                    except Exception as e:
                        logger.error(f"调用宇宙事件生成方法失败: {str(e)}")
                        cosmic_events = []
                    
                if cosmic_events:
                    prediction_data["cosmic_events"] = cosmic_events
                
                # 添加宇宙共振信息
                resonance_state = self.cosmic_engine.get_resonance_state()
                if resonance_state:
                    prediction_data["cosmic_resonance"] = {
                        "applied": True,
                        "level": resonance_state.get("resonance_level", 0),
                        "accuracy": resonance_state.get("cosmic_accuracy", 0)
                    }
                
                logger.info("使用宇宙共振引擎增强完成")
            elif hasattr(self, 'cosmic_controller') and hasattr(self.cosmic_controller, 'resonance_engine'):
                # 兼容旧版接口
                # 增强预测数据中的数值
                if "predictions" in prediction_data and isinstance(prediction_data["predictions"], list):
                    prediction_data["predictions"] = self.cosmic_controller.apply_cosmic_filter(prediction_data["predictions"])
                
                # 添加宇宙共振信息
                resonance_state = self.cosmic_controller.resonance_engine.get_resonance_state()
                if resonance_state:
                    prediction_data["cosmic_resonance"] = {
                        "applied": True,
                        "level": resonance_state.get("resonance_level", 0),
                        "accuracy": resonance_state.get("cosmic_accuracy", 0)
                    }
                
                logger.info("使用旧版宇宙共振引擎增强完成")
            
            # 使用增强预测可视化处理 (如果可用)
            if hasattr(self, 'prediction_visualizer') and hasattr(self, 'data_controller'):
                try:
                    # 获取股票代码
                    stock_code = prediction_data.get('stock_code', '')
                    if stock_code:
                        # 生成结果文件名
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")
                        os.makedirs(results_dir, exist_ok=True)
                        
                        json_file = os.path.join(results_dir, f"prediction_{stock_code}_{timestamp}.json")
                        image_file = os.path.join(results_dir, f"prediction_{stock_code}_{timestamp}.png")
                        
                        # 保存JSON和生成可视化
                        self.prediction_visualizer.save_prediction_json(prediction_data, json_file)
                        self.prediction_visualizer.plot_prediction(prediction_data, image_file)
                        
                        # 添加可视化文件路径到预测数据中
                        prediction_data["visualization"] = {
                            "json_file": json_file,
                            "image_file": image_file
                        }
                        
                        logger.info(f"预测可视化已保存: {image_file}")
                except Exception as e:
                    logger.error(f"预测可视化失败: {str(e)}")
            
            return prediction_data
                
        except Exception as e:
            self.logger.error(f"增强预测失败: {str(e)}")
            return prediction_data
            
    def request_stock_prediction(self, stock_code, days=10):
        """请求股票预测"""
        try:
            # 使用增强的预测可视化 (如果可用)
            if hasattr(self, 'prediction_visualizer'):
                prediction = self.prediction_visualizer.predict_stock(stock_code, days)
                
                # 如果预测成功，显示结果
                if prediction:
                    # 保存结果
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")
                    os.makedirs(results_dir, exist_ok=True)
                    
                    json_file = os.path.join(results_dir, f"prediction_{stock_code}_{timestamp}.json")
                    image_file = os.path.join(results_dir, f"prediction_{stock_code}_{timestamp}.png")
                    
                    # 保存JSON和图像
                    self.prediction_visualizer.save_prediction_json(prediction, json_file)
                    self.prediction_visualizer.plot_prediction(prediction, image_file)
                    
                    # 显示预测图像
                    self._display_prediction_image(image_file, prediction)
                    return True
                else:
                    QMessageBox.warning(self, "预测警告", f"无法为股票 {stock_code} 生成预测")
                    return False
            else:
                # 回退到数据控制器的预测方法
                if hasattr(self.data_controller, 'predict_stock'):
                    self.data_controller.predict_stock(stock_code, days, self._on_prediction_result)
                    return True
                else:
                    QMessageBox.warning(self, "预测警告", "预测功能不可用")
                    return False
        except Exception as e:
            logger.error(f"请求股票预测失败: {str(e)}")
            QMessageBox.critical(self, "预测错误", f"预测过程中发生错误: {str(e)}")
            return False
    
    def _on_prediction_result(self, prediction_data):
        """处理预测结果回调"""
        try:
            if not prediction_data:
                QMessageBox.warning(self, "预测警告", "未收到有效的预测结果")
                return
                
            # 增强预测
            enhanced_prediction = self.enhance_prediction(prediction_data)
            
            # 显示预测结果
            self._display_prediction_result(enhanced_prediction)
        except Exception as e:
            logger.error(f"处理预测结果失败: {str(e)}")
            QMessageBox.critical(self, "预测错误", f"处理预测结果时发生错误: {str(e)}")
    
    def _display_prediction_image(self, image_path, prediction_data=None):
        """显示预测图像"""
        try:
            # 切换到预测标签页 (如果有)
            for i in range(self.main_tab.count()):
                if "prediction" in self.main_tab.tabText(i).lower() or "预测" in self.main_tab.tabText(i):
                    self.main_tab.setCurrentIndex(i)
                    break
            
            # 如果有预测视图则使用它
            if hasattr(self, 'prediction_view') and hasattr(self.prediction_view, 'display_prediction_image'):
                self.prediction_view.display_prediction_image(image_path)
            elif hasattr(self, 'quantum_view') and hasattr(self.quantum_view, 'display_prediction_image'):
                self.quantum_view.display_prediction_image(image_path)
            else:
                # 回退方案：显示一个弹窗
                dialog = QDialog(self)
                dialog.setWindowTitle("股票预测结果")
                dialog.setMinimumSize(900, 700)
                
                layout = QVBoxLayout(dialog)
                
                # 显示图像
                image_label = QLabel()
                pixmap = QPixmap(image_path)
                image_label.setPixmap(pixmap.scaled(850, 600, Qt.KeepAspectRatio))
                layout.addWidget(image_label)
                
                # 如果有预测数据，添加一些详情
                if prediction_data:
                    details_box = QGroupBox("预测详情")
                    details_layout = QGridLayout(details_box)
                    
                    # 添加股票名称和代码
                    stock_code = prediction_data.get('stock_code', 'Unknown')
                    stock_name = prediction_data.get('stock_name', 'Unknown')
                    details_layout.addWidget(QLabel(f"股票: {stock_name} ({stock_code})"), 0, 0)
                    
                    # 添加预测日期范围
                    start_date = prediction_data.get('start_date', 'Unknown')
                    end_date = prediction_data.get('end_date', 'Unknown')
                    details_layout.addWidget(QLabel(f"预测范围: {start_date} 至 {end_date}"), 0, 1)
                    
                    # 添加预测置信度
                    confidence = prediction_data.get('confidence', 0)
                    details_layout.addWidget(QLabel(f"预测置信度: {confidence:.2f}%"), 1, 0)
                    
                    # 添加市场洞察
                    insights = prediction_data.get('market_insights', [])
                    if insights:
                        insight_text = "市场洞察:\n" + "\n".join([f"- {insight}" for insight in insights[:3]])
                        insight_label = QLabel(insight_text)
                        insight_label.setWordWrap(True)
                        details_layout.addWidget(insight_label, 2, 0, 1, 2)
                    
                    layout.addWidget(details_box)
                
                # 添加按钮
                buttons_layout = QHBoxLayout()
                save_button = QPushButton("保存图像")
                save_button.clicked.connect(lambda: self._save_prediction_image(image_path))
                close_button = QPushButton("关闭")
                close_button.clicked.connect(dialog.accept)
                
                buttons_layout.addWidget(save_button)
                buttons_layout.addWidget(close_button)
                layout.addLayout(buttons_layout)
                
                dialog.setLayout(layout)
                dialog.exec_()
        except Exception as e:
            logger.error(f"显示预测图像失败: {str(e)}")
            QMessageBox.warning(self, "显示警告", f"无法显示预测图像: {str(e)}")
    
    def _save_prediction_image(self, source_path):
        """保存预测图像到用户指定位置"""
        try:
            # 打开文件对话框
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存预测图像", 
                os.path.expanduser("~/Desktop/prediction.png"),
                "图像文件 (*.png *.jpg);;所有文件 (*)"
            )
            
            if not file_path:
                return
                
            # 复制文件
            import shutil
            shutil.copy2(source_path, file_path)
            
            QMessageBox.information(self, "保存成功", f"预测图像已保存至: {file_path}")
        except Exception as e:
            logger.error(f"保存预测图像失败: {str(e)}")
            QMessageBox.warning(self, "保存失败", f"无法保存预测图像: {str(e)}")
            
    def _display_prediction_result(self, prediction_data):
        """显示预测结果"""
        try:
            # 如果有可视化图像，则显示它
            visualization = prediction_data.get('visualization', {})
            image_file = visualization.get('image_file')
            
            if image_file and os.path.exists(image_file):
                self._display_prediction_image(image_file, prediction_data)
            else:
                # 没有图像，显示文本结果
                stock_code = prediction_data.get('stock_code', 'Unknown')
                stock_name = prediction_data.get('stock_name', 'Unknown')
                
                message = f"股票 {stock_name} ({stock_code}) 预测结果:\n\n"
                
                predictions = prediction_data.get('predictions', [])
                if predictions:
                    message += "价格预测:\n"
                    for i, pred in enumerate(predictions[:5]):  # 只显示前5个预测
                        date = pred.get('date', f'Day {i+1}')
                        price = pred.get('price', 0)
                        change = pred.get('change_percent', 0)
                        message += f"{date}: {price:.2f} ({'+' if change >= 0 else ''}{change:.2f}%)\n"
                
                insights = prediction_data.get('market_insights', [])
                if insights:
                    message += "\n市场洞察:\n"
                    for insight in insights[:3]:  # 只显示前3个洞察
                        message += f"- {insight}\n"
                
                QMessageBox.information(self, "预测结果", message)
        except Exception as e:
            logger.error(f"显示预测结果失败: {str(e)}")
            QMessageBox.warning(self, "显示警告", f"无法显示预测结果: {str(e)}")
    
    def _setup_search_bar(self):
        """设置顶部搜索栏"""
        try:
            # 创建搜索栏容器
            search_container = QWidget()
            search_layout = QHBoxLayout(search_container)
            search_layout.setContentsMargins(10, 5, 10, 5)
            
            # 创建搜索框标签
            search_label = QLabel("搜索股票:")
            search_layout.addWidget(search_label)
            
            # 创建搜索输入框
            self.search_input = QLineEdit()
            self.search_input.setPlaceholderText("输入股票代码、名称或拼音...")
            self.search_input.setMinimumWidth(250)
            search_layout.addWidget(self.search_input)
            
            # 创建搜索按钮
            search_button = QPushButton("搜索")
            search_button.clicked.connect(lambda: self._handle_search(self.search_input.text()))
            search_layout.addWidget(search_button)
            
            # 添加弹性空间
            search_layout.addStretch(1)
            
            # 将搜索栏添加到主布局
            if hasattr(self, 'centralWidget') and hasattr(self.centralWidget(), 'layout'):
                self.centralWidget().layout().insertWidget(0, search_container)
            
            # 创建搜索结果列表（初始隐藏）
            self.search_results_list = QListWidget()
            self.search_results_list.setMaximumHeight(300)
            self.search_results_list.setVisible(False)
            self.search_results_list.itemClicked.connect(self._on_search_result_selected)
            
            if hasattr(self, 'centralWidget') and hasattr(self.centralWidget(), 'layout'):
                self.centralWidget().layout().insertWidget(1, self.search_results_list)
                
            logger.info("搜索栏设置完成")
        except Exception as e:
            logger.error(f"设置搜索栏失败: {str(e)}")
    
    def _handle_search(self, query):
        """处理搜索请求"""
        try:
            if not query or len(query) < 2:
                self.search_results_list.clear()
                self.search_results_list.setVisible(False)
                return
                
            # 使用高级搜索组件 (如果可用)
            if hasattr(self, 'stock_searcher'):
                results = self.stock_searcher.find_stock(query, max_results=10)
                self.update_search_results(results)
            else:
                # 回退到基本搜索
                self.data_controller.search_stocks(query, self.update_search_results)
        except Exception as e:
            logger.error(f"处理搜索请求失败: {str(e)}")
    
    def update_search_results(self, results):
        """更新搜索结果列表"""
        try:
            self.search_results_list.clear()
            
            if not results:
                self.search_results_list.addItem("未找到匹配结果")
                self.search_results_list.setVisible(True)
                return
                
            for stock in results:
                # 创建列表项
                item = QListWidgetItem()
                
                # 获取股票数据
                name = stock.get('name', 'Unknown')
                code = stock.get('ts_code', stock.get('code', 'Unknown'))
                industry = stock.get('industry', '')
                match_score = stock.get('match_score', 100)
                
                # 设置显示文本
                if match_score < 100:
                    display_text = f"{name} ({code}) - {industry} [匹配度: {match_score}%]"
                else:
                    display_text = f"{name} ({code}) - {industry}"
                
                item.setText(display_text)
                
                # 保存股票数据
                item.setData(Qt.UserRole, stock)
                
                # 添加到列表
                self.search_results_list.addItem(item)
                
            self.search_results_list.setVisible(True)
            logger.info(f"更新搜索结果: {len(results)} 个结果")
        except Exception as e:
            logger.error(f"更新搜索结果失败: {str(e)}")
    
    def _on_search_result_selected(self, item):
        """处理搜索结果选择"""
        try:
            # 获取股票数据
            stock = item.data(Qt.UserRole)
            if not stock:
                return
                
            # 显示股票详情
            self._display_stock_details(stock)
            
            # 隐藏搜索结果列表
            self.search_results_list.setVisible(False)
        except Exception as e:
            logger.error(f"处理搜索结果选择失败: {str(e)}")
    
    def _display_stock_details(self, stock):
        """显示股票详情"""
        try:
            # 切换到市场标签页
            for i in range(self.main_tab.count()):
                if self.main_tab.tabText(i) == "市场" or "market" in self.main_tab.tabText(i).lower():
                    self.main_tab.setCurrentIndex(i)
                    break
            
            # 更新市场视图显示该股票
            if MARKET_VIEW_AVAILABLE and hasattr(self, 'market_view'):
                if hasattr(self.market_view, 'display_stock'):
                    self.market_view.display_stock(stock.get('ts_code'))
                elif hasattr(self.market_view, 'select_stock'):
                    self.market_view.select_stock(stock.get('ts_code'))
                else:
                    logger.warning("市场视图没有display_stock或select_stock方法")
                    
                    # 回退方案: 弹窗显示
                    code = stock.get('ts_code', stock.get('code', ''))
                    name = stock.get('name', '')
                    QMessageBox.information(self, "股票信息", f"选择了股票: {name} ({code})")
            else:
                # 回退方案: 弹窗显示
                code = stock.get('ts_code', stock.get('code', ''))
                name = stock.get('name', '')
                QMessageBox.information(self, "股票信息", f"选择了股票: {name} ({code})")
        except Exception as e:
            logger.error(f"显示股票详情失败: {str(e)}")
            
    def _create_portfolio_view(self, force_load=False):
        """创建超神级投资组合视图"""
        try:
            from gui.views.portfolio_view import PortfolioView
            
            # 创建投资组合视图
            portfolio_view = PortfolioView()
            
            # 启用全部超神特性
            portfolio_view.enable_super_god_features = True
            
            # 连接控制器
            if hasattr(self, 'portfolio_controller') and self.portfolio_controller:
                portfolio_view.controller = self.portfolio_controller
                self.portfolio_controller.view = portfolio_view
                
                try:
                    # 初始化投资组合数据
                    account_data = self.portfolio_controller.get_account_data()
                    allocation_data = self.portfolio_controller.get_allocation_data()
                    risk_metrics = self.portfolio_controller.get_risk_metrics()
                    performance_data = self.portfolio_controller.get_performance_data()
                    
                    # 使用投资组合数据初始化视图
                    portfolio_view.initialize_with_data({
                        "account_data": account_data,
                        "allocation_data": allocation_data,
                        "risk_metrics": risk_metrics,
                        "performance_data": performance_data
                    })
                    
                    self.logger.info("投资组合视图已成功初始化数据")
                except Exception as e:
                    self.logger.error(f"初始化投资组合数据失败: {str(e)}")
                    if not force_load:
                        raise
            
            return portfolio_view
            
        except Exception as e:
            self.logger.error(f"创建超神投资组合视图失败: {str(e)}")
            if force_load:
                # 尝试创建一个简单的视图
                from gui.views.portfolio_view import SimplePortfolioView
                return SimplePortfolioView()
            else:
                raise
    
    def _try_reload_portfolio_view(self):
        """尝试重新加载超神投资组合视图"""
        try:
            self.logger.info("尝试强制重新加载超神投资组合视图...")
            # 获取当前的选项卡索引
            current_index = self.main_tab.currentIndex()
            portfolio_index = -1
            
            # 找到投资组合选项卡的索引
            for i in range(self.main_tab.count()):
                if self.main_tab.tabText(i) == "投资组合":
                    portfolio_index = i
                    break
            
            if portfolio_index == -1:
                raise ValueError("无法找到投资组合选项卡")
            
            # 创建新的投资组合视图
            self.portfolio_view = self._create_portfolio_view(force_load=True)
            
            # 替换旧的选项卡
            self.main_tab.removeTab(portfolio_index)
            self.main_tab.insertTab(portfolio_index, self.portfolio_view, qta.icon('fa5s.chart-pie'), "投资组合")
            
            # 如果原来选中的是投资组合选项卡，则重新选中它
            if current_index == portfolio_index:
                self.main_tab.setCurrentIndex(portfolio_index)
            
            # 连接控制器
            if self.portfolio_controller:
                self.portfolio_view.controller = self.portfolio_controller
                self.portfolio_controller.view = self.portfolio_view
                self.portfolio_controller.update_view()  # 刷新视图数据
            
            self.logger.info("成功重新加载超神投资组合视图")
            QMessageBox.information(self, "成功", "超神投资组合视图已成功加载！")
        except Exception as e:
            self.logger.error(f"强制重新加载超神投资组合视图失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"无法加载超神投资组合视图:\n{str(e)}")
    
    def _create_trading_view(self, force_load=False):
        """创建超神级交易视图"""
        try:
            from gui.views.trading_view import TradingView
            
            # 创建交易视图
            trading_view = TradingView()
            
            # 启用全部超神特性
            trading_view.enable_super_god_features = True
            
            # 连接控制器
            if hasattr(self, 'trading_controller') and self.trading_controller:
                trading_view.controller = self.trading_controller
                self.trading_controller.view = trading_view
                
                try:
                    # 初始化交易数据
                    orders = self.trading_controller.get_orders()
                    positions = self.trading_controller.get_positions()
                    
                    # 使用交易数据初始化视图
                    trading_view.initialize_with_data({
                        "orders": orders,
                        "positions": positions
                    })
                    
                    # 加载量子交易信号
                    quantum_signals = self.trading_controller.get_quantum_signals()
                    trading_view.update_quantum_signals(quantum_signals)
                    
                    self.logger.info("交易视图已成功初始化数据")
                except Exception as e:
                    self.logger.error(f"初始化交易数据失败: {str(e)}")
                    if not force_load:
                        raise
            
            return trading_view
            
        except Exception as e:
            self.logger.error(f"创建超神交易视图失败: {str(e)}")
            if force_load:
                # 尝试创建一个简单的视图
                from gui.views.trading_view import SimpleTradingView
                return SimpleTradingView()
            else:
                raise
    
    def _try_reload_trading_view(self):
        """尝试重新加载超神交易视图"""
        try:
            self.logger.info("尝试强制重新加载超神交易视图...")
            # 获取当前的交易选项卡索引
            current_index = self.main_tab.currentIndex()
            trading_index = -1
            
            # 找到交易选项卡的索引
            for i in range(self.main_tab.count()):
                if self.main_tab.tabText(i) == "交易":
                    trading_index = i
                    break
            
            if trading_index == -1:
                raise ValueError("无法找到交易选项卡")
            
            # 创建新的交易视图
            self.trading_view = self._create_trading_view(force_load=True)
            
            # 替换旧的选项卡
            self.main_tab.removeTab(trading_index)
            self.main_tab.insertTab(trading_index, self.trading_view, qta.icon('fa5s.exchange-alt'), "交易")
            
            # 如果原来选中的是交易选项卡，则重新选中它
            if current_index == trading_index:
                self.main_tab.setCurrentIndex(trading_index)
            
            # 连接控制器
            if self.trading_controller:
                self.trading_view.controller = self.trading_controller
                self.trading_controller.view = self.trading_view
                self.trading_controller.update_view()  # 刷新视图数据
            
            self.logger.info("成功重新加载超神交易视图")
            QMessageBox.information(self, "成功", "超神交易视图已成功加载！")
        except Exception as e:
            self.logger.error(f"强制重新加载超神交易视图失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"无法加载超神交易视图:\n{str(e)}") 