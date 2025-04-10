"""
主窗口模块 - 提供应用程序的主界面
"""

import os
import logging
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                          QTabWidget, QLabel, QPushButton, QToolBar, QAction, 
                          QStatusBar, QDockWidget, QSplitter, QTreeView, 
                          QListWidget, QMenuBar, QMenu, QFrame)
from PyQt5.QtCore import Qt, QSize, pyqtSlot
from PyQt5.QtGui import QIcon, QFont

# 导入面板组件
from .panels.dashboard_panel import DashboardPanel
from .panels.quantum_circuit_panel import QuantumCircuitPanel
from .panels.market_analysis_panel import MarketAnalysisPanel
from .panels.strategy_backtest_panel import StrategyBacktestPanel
from .panels.quantum_stock_finder_panel import QuantumStockFinderPanel
from .panels.quantum_validation_panel import QuantumValidationPanel
from .widgets.component_status_widget import ComponentStatusWidget

logger = logging.getLogger("QuantumDesktop.MainWindow")

class MainWindow(QMainWindow):
    """主窗口类"""
    
    def __init__(self, system_manager):
        super().__init__()
        
        self.system_manager = system_manager
        
        # 设置窗口属性
        self.setWindowTitle("超神量子核心系统")
        self.setMinimumSize(1200, 800)
        
        # 初始化UI
        self._init_ui()
        
        logger.info("主窗口创建完成")
        
    def _init_ui(self):
        """初始化用户界面"""
        # 设置中央部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # 主布局
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # 创建菜单栏
        self._create_menu_bar()
        
        # 创建工具栏
        self._create_toolbar()
        
        # 创建内容区域
        self._create_content_area()
        
        # 创建状态栏
        self._create_status_bar()
        
    def _create_menu_bar(self):
        """创建菜单栏"""
        # 文件菜单
        file_menu = self.menuBar().addMenu("文件")
        
        new_action = QAction("新建项目", self)
        new_action.setShortcut("Ctrl+N")
        file_menu.addAction(new_action)
        
        open_action = QAction("打开项目", self)
        open_action.setShortcut("Ctrl+O")
        file_menu.addAction(open_action)
        
        save_action = QAction("保存项目", self)
        save_action.setShortcut("Ctrl+S")
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("退出", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 系统菜单
        system_menu = self.menuBar().addMenu("系统")
        
        start_action = QAction("启动核心系统", self)
        start_action.triggered.connect(self._on_start_system)
        system_menu.addAction(start_action)
        
        stop_action = QAction("停止核心系统", self)
        stop_action.triggered.connect(self._on_stop_system)
        system_menu.addAction(stop_action)
        
        system_menu.addSeparator()
        
        settings_action = QAction("系统设置", self)
        system_menu.addAction(settings_action)
        
        # 工具菜单
        tools_menu = self.menuBar().addMenu("工具")
        
        data_import_action = QAction("数据导入", self)
        tools_menu.addAction(data_import_action)
        
        backtest_action = QAction("策略回测", self)
        backtest_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(3))
        tools_menu.addAction(backtest_action)
        
        optimization_action = QAction("策略优化", self)
        tools_menu.addAction(optimization_action)
        
        tools_menu.addSeparator()
        
        stock_finder_action = QAction("超神量子选股", self)
        stock_finder_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(4))
        tools_menu.addAction(stock_finder_action)
        
        stock_validation_action = QAction("量子选股验证", self)
        stock_validation_action.triggered.connect(lambda: self.tab_widget.setCurrentIndex(5))
        tools_menu.addAction(stock_validation_action)
        
        # 视图菜单
        view_menu = self.menuBar().addMenu("视图")
        
        # 帮助菜单
        help_menu = self.menuBar().addMenu("帮助")
        
        about_action = QAction("关于", self)
        help_menu.addAction(about_action)
        
        docs_action = QAction("文档", self)
        help_menu.addAction(docs_action)
        
    def _create_toolbar(self):
        """创建工具栏"""
        # 主工具栏
        self.main_toolbar = QToolBar("主工具栏")
        self.main_toolbar.setIconSize(QSize(24, 24))
        self.main_toolbar.setMovable(False)
        self.addToolBar(self.main_toolbar)
        
        # 启动按钮
        self.start_button = QPushButton("启动系统")
        self.start_button.clicked.connect(self._on_start_system)
        self.main_toolbar.addWidget(self.start_button)
        
        # 停止按钮
        self.stop_button = QPushButton("停止系统")
        self.stop_button.clicked.connect(self._on_stop_system)
        self.stop_button.setEnabled(False)
        self.main_toolbar.addWidget(self.stop_button)
        
        self.main_toolbar.addSeparator()
        
        # 电路设计按钮
        self.circuit_button = QPushButton("电路设计")
        self.circuit_button.clicked.connect(lambda: self.tab_widget.setCurrentIndex(1))
        self.main_toolbar.addWidget(self.circuit_button)
        
        # 市场分析按钮
        self.analysis_button = QPushButton("市场分析")
        self.analysis_button.clicked.connect(lambda: self.tab_widget.setCurrentIndex(2))
        self.main_toolbar.addWidget(self.analysis_button)
        
        # 策略回测按钮
        self.backtest_button = QPushButton("策略回测")
        self.backtest_button.clicked.connect(lambda: self.tab_widget.setCurrentIndex(3))
        self.main_toolbar.addWidget(self.backtest_button)
        
        # 超神量子选股按钮
        self.stock_finder_button = QPushButton("超神选股")
        self.stock_finder_button.setStyleSheet("background-color: #4a148c; color: white; font-weight: bold;")
        self.stock_finder_button.clicked.connect(lambda: self.tab_widget.setCurrentIndex(4))
        self.main_toolbar.addWidget(self.stock_finder_button)
        
        # 量子选股验证按钮
        self.validation_button = QPushButton("选股验证")
        self.validation_button.setStyleSheet("background-color: #00796b; color: white; font-weight: bold;")
        self.validation_button.clicked.connect(lambda: self.tab_widget.setCurrentIndex(5))
        self.main_toolbar.addWidget(self.validation_button)
        
    def _create_content_area(self):
        """创建内容区域"""
        # 分割器，左侧组件状态，右侧内容
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.main_splitter)
        
        # 左侧面板
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        
        # 组件状态小部件
        self.component_status = ComponentStatusWidget(self.system_manager)
        self.left_layout.addWidget(self.component_status)
        
        # 项目文件浏览器
        file_label = QLabel("项目文件")
        file_label.setAlignment(Qt.AlignCenter)
        self.left_layout.addWidget(file_label)
        
        self.file_tree = QTreeView()
        self.left_layout.addWidget(self.file_tree)
        
        # 添加左侧面板到分割器
        self.main_splitter.addWidget(self.left_panel)
        
        # 右侧标签页
        self.tab_widget = QTabWidget()
        
        # 仪表盘面板
        self.dashboard_panel = DashboardPanel(self.system_manager)
        self.tab_widget.addTab(self.dashboard_panel, "仪表盘")
        
        # 量子电路面板
        self.quantum_circuit_panel = QuantumCircuitPanel(self.system_manager)
        self.tab_widget.addTab(self.quantum_circuit_panel, "量子电路")
        
        # 市场分析面板
        self.market_analysis_panel = MarketAnalysisPanel(self.system_manager)
        self.tab_widget.addTab(self.market_analysis_panel, "市场分析")
        
        # 策略回测面板
        self.strategy_backtest_panel = StrategyBacktestPanel(self.system_manager)
        self.tab_widget.addTab(self.strategy_backtest_panel, "策略回测")
        
        # 超神量子选股面板
        self.quantum_stock_finder_panel = QuantumStockFinderPanel(self.system_manager)
        self.tab_widget.addTab(self.quantum_stock_finder_panel, "超神量子选股")
        
        # 量子选股验证面板
        self.quantum_validation_panel = QuantumValidationPanel(self.system_manager)
        self.tab_widget.addTab(self.quantum_validation_panel, "量子选股验证")
        
        # 添加标签页到分割器
        self.main_splitter.addWidget(self.tab_widget)
        
        # 设置分割比例
        self.main_splitter.setSizes([200, 1000])
        
    def _create_status_bar(self):
        """创建状态栏"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # 系统状态标签
        self.system_status_label = QLabel("系统状态：未启动")
        self.status_bar.addWidget(self.system_status_label)
        
        # 内存使用标签
        self.memory_label = QLabel("内存使用：0 MB")
        self.status_bar.addPermanentWidget(self.memory_label)
        
        # 处理器使用标签
        self.cpu_label = QLabel("CPU使用：0%")
        self.status_bar.addPermanentWidget(self.cpu_label)
        
    @pyqtSlot()
    def _on_start_system(self):
        """启动系统事件"""
        try:
            logger.info("正在启动超神量子核心系统...")
            
            # 禁用启动按钮
            self.start_button.setEnabled(False)
            
            # 获取系统管理器并启动系统
            if self.system_manager.start_system():
                # 启动成功
                logger.info("系统启动成功")
                
                # 更新UI状态
                self.stop_button.setEnabled(True)
                
                # 通知各面板系统已启动
                self.dashboard_panel.on_system_started()
                self.quantum_circuit_panel.on_system_started()
                self.market_analysis_panel.on_system_started()
                self.strategy_backtest_panel.on_system_started()
                self.quantum_stock_finder_panel.on_system_started()
                self.quantum_validation_panel.on_system_started()
                
            else:
                # 启动失败
                logger.error("系统启动失败")
                self.start_button.setEnabled(True)
                
        except Exception as e:
            logger.error(f"启动系统时出错: {str(e)}")
            self.start_button.setEnabled(True)
            
    @pyqtSlot()
    def _on_stop_system(self):
        """停止系统事件"""
        try:
            logger.info("正在停止超神量子核心系统...")
            
            # 禁用停止按钮
            self.stop_button.setEnabled(False)
            
            # 获取系统管理器并停止系统
            if self.system_manager.stop_system():
                # 停止成功
                logger.info("系统停止成功")
                
                # 更新UI状态
                self.start_button.setEnabled(True)
                
                # 通知各面板系统已停止
                self.dashboard_panel.on_system_stopped()
                self.quantum_circuit_panel.on_system_stopped()
                self.market_analysis_panel.on_system_stopped()
                self.strategy_backtest_panel.on_system_stopped()
                self.quantum_stock_finder_panel.on_system_stopped()
                self.quantum_validation_panel.on_system_stopped()
                
            else:
                # 停止失败
                logger.error("系统停止失败")
                self.stop_button.setEnabled(True)
                
        except Exception as e:
            logger.error(f"停止系统时出错: {str(e)}")
            self.stop_button.setEnabled(True)
        
    def closeEvent(self, event):
        """窗口关闭事件处理"""
        logger.info("关闭应用程序")
        
        # 停止系统
        if self.system_manager.is_system_running():
            self.system_manager.stop_system()
        
        event.accept() 