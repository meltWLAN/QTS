#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - 桌面客户端
高级交易系统的现代化桌面界面
"""

import sys
import os
import logging
import traceback
import socket
import threading
from datetime import datetime, timedelta
from PyQt5.QtWidgets import QApplication, QMainWindow, QSplashScreen, QMessageBox, QVBoxLayout, QWidget, QTabWidget, QPushButton
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QIcon, QFont
import qdarkstyle
from qt_material import apply_stylesheet

# 导入自定义模块
from gui.views.main_window import SuperTradingMainWindow
from gui.controllers.data_controller import DataController
from gui.controllers.trading_controller import TradingController

# 导入新增功能模块
from advanced_stock_search import StockCodeConverter, AdvancedStockSearch
from market_dimension_analysis import MarketDimensionAnalyzer
import super_prediction_fix as spf

# 导入全局获取器函数以确保使用最新的超神核心模块
from quantum_symbiotic_network.quantum_prediction import get_predictor
from quantum_symbiotic_network.cosmic_resonance import get_engine

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("quantum_trading_gui.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("SuperTradingGUI")

# 全局应用实例和主窗口实例，防止被垃圾回收
_app = None
_main_window = None
_socket_lock = None

# 单例机制 - 确保只有一个应用实例在运行
def ensure_single_instance():
    """确保只有一个应用实例在运行"""
    try:
        # 创建一个套接字
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 设置SO_REUSEADDR选项
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # 尝试绑定到本地端口
        sock.bind(('localhost', 37337))
        # 开始监听
        sock.listen(5)
        # 保存套接字引用，防止被垃圾回收
        global _socket_lock
        _socket_lock = sock
        return True
    except socket.error:
        # 端口已被占用，说明已经有一个实例在运行
        return False


class DataLoadingThread(QThread):
    """数据加载线程"""
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    
    def __init__(self, data_controller, trading_controller):
        super().__init__()
        self.data_controller = data_controller
        self.trading_controller = trading_controller
        
    def run(self):
        try:
            # 初始化数据控制器
            self.progress_signal.emit(10, "初始化数据源...")
            self.data_controller.initialize()
            
            # 加载初始数据
            self.progress_signal.emit(30, "加载市场数据...")
            data = self.data_controller.load_initial_data()
            
            # 初始化交易控制器
            self.progress_signal.emit(50, "初始化交易系统...")
            self.trading_controller.initialize_mock_positions()
            self.trading_controller.initialize_mock_orders()
            
            # 获取账户和订单数据
            self.progress_signal.emit(70, "获取账户数据...")
            data["positions"] = self.trading_controller.get_position_list()
            data["orders"] = self.trading_controller.get_order_list()
            
            # 初始化量子网络
            self.progress_signal.emit(85, "初始化量子共生网络...")
            
            # 初始化高级功能模块
            self.progress_signal.emit(90, "初始化超神高级功能...")
            # 高级股票搜索和市场分析初始化在主窗口中进行
            
            # 发送完成信号
            self.progress_signal.emit(100, "加载完成!")
            self.finished_signal.emit(data)
            
        except Exception as e:
            error_message = f"数据加载失败: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_message)
            self.error_signal.emit(error_message)


# 市场分析线程
class MarketAnalysisThread(QThread):
    """市场分析线程"""
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    
    def __init__(self, market_analyzer):
        super().__init__()
        self.market_analyzer = market_analyzer
        
    def run(self):
        try:
            self.progress_signal.emit(0, "正在启动市场多维度分析...")
            
            # 分析热点板块
            self.progress_signal.emit(20, "分析热点板块...")
            self.market_analyzer.analyze_hot_sectors(period='7d', top_count=15)
            
            # 分析板块轮动
            self.progress_signal.emit(40, "分析板块轮动...")
            self.market_analyzer.analyze_sector_rotation(period='90d')
            
            # 分析板块关联性
            self.progress_signal.emit(60, "分析板块关联性...")
            self.market_analyzer.analyze_sector_correlation(period='30d')
            
            # 分析资金流向
            self.progress_signal.emit(80, "分析资金流向...")
            self.market_analyzer.analyze_capital_flow(period='30d')
            
            # 完成分析
            self.progress_signal.emit(100, "市场分析完成!")
            self.finished_signal.emit(self.market_analyzer.latest_analysis)
            
        except Exception as e:
            error_message = f"市场分析失败: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_message)
            self.error_signal.emit(error_message)


def load_stylesheet(theme="dark_teal"):
    """加载应用样式表"""
    try:
        return apply_stylesheet(QApplication.instance(), theme=theme)
    except:
        return qdarkstyle.load_stylesheet_pyqt5()


def initialize_application():
    """初始化应用实例"""
    global _app
    
    # 检查是否已有实例
    if QApplication.instance():
        _app = QApplication.instance()
    else:
        _app = QApplication(sys.argv)
    
    # 设置应用基本属性
    _app.setApplicationName("超神量子共生网络交易系统")
    _app.setOrganizationName("QuantumSymbioticTeam")
    _app.setApplicationVersion("0.3.0")  # 升级版本号
    
    # 设置应用图标
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gui/resources/icon.png")
    if os.path.exists(icon_path):
        _app.setWindowIcon(QIcon(icon_path))
    
    # 配置应用样式
    try:
        apply_stylesheet(_app, theme="dark_cyan")
    except Exception as e:
        logger.warning(f"应用Material样式失败: {str(e)}")
        _app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
    
    logger.info("应用初始化完成")
    return _app


def initialize_controllers():
    """初始化控制器"""
    # 创建控制器
    data_controller = DataController()
    trading_controller = TradingController()
    
    logger.info("控制器初始化完成")
    return data_controller, trading_controller


def initialize_main_window(data_controller, trading_controller):
    """初始化主窗口"""
    global _main_window
    
    # 创建主窗口
    _main_window = SuperTradingMainWindow(data_controller, trading_controller)
    
    # 初始化高级功能
    _initialize_advanced_features(_main_window, data_controller)
    
    logger.info("主窗口初始化完成")
    return _main_window


def _initialize_advanced_features(main_window, data_controller):
    """初始化高级功能"""
    try:
        # 初始化超神核心引擎 - 确保使用全局单例实例
        logger.info("初始化超神核心引擎...")
        predictor = get_predictor(data_controller.tushare_token, force_new=True)
        cosmic_engine = get_engine()
        
        # 设置超神级参数
        predictor.set_quantum_params(
            coherence=0.95,
            superposition=0.92,
            entanglement=0.90
        )
        logger.info("超神量子参数已设置为最佳值")
        
        # 启动宇宙共振引擎
        cosmic_engine.start()
        logger.info("宇宙共振引擎已启动")
        
        # 初始化高级股票搜索
        main_window.stock_searcher = AdvancedStockSearch(data_controller.tushare_source)
        logger.info("高级股票搜索初始化完成")
        
        # 初始化市场分析器
        main_window.market_analyzer = MarketDimensionAnalyzer(data_controller.tushare_source)
        logger.info("市场维度分析器初始化完成")
        
        # 初始化量子预测可视化
        main_window.prediction_visualizer = spf
        logger.info("增强预测可视化初始化完成")
        
        # 连接超神核心引擎到主窗口
        main_window.quantum_predictor = predictor
        main_window.cosmic_engine = cosmic_engine
        logger.info("超神核心引擎与主窗口连接完成")
        
        # 增强搜索功能
        _enhance_search_functionality(main_window)
        
        # 添加市场分析标签页
        _add_market_analysis_tab(main_window)
        
        # 设置自动市场分析
        _setup_scheduled_market_analysis(main_window)
        
    except Exception as e:
        logger.error(f"初始化高级功能失败: {str(e)}")
        traceback.print_exc()


def _enhance_search_functionality(main_window):
    """增强搜索功能"""
    try:
        # 连接高级搜索功能到搜索框
        if hasattr(main_window, 'search_input'):
            main_window.search_input.textChanged.disconnect()
            main_window.search_input.textChanged.connect(
                lambda text: _handle_advanced_search(main_window, text)
            )
            logger.info("高级搜索功能集成完成")
    except Exception as e:
        logger.error(f"增强搜索功能失败: {str(e)}")


def _handle_advanced_search(main_window, text):
    """处理高级股票搜索"""
    if len(text) < 2:  # 至少输入两个字符才开始搜索
        return
        
    try:
        # 使用高级搜索组件
        results = main_window.stock_searcher.find_stock(text, max_results=10)
        
        # 更新搜索结果列表
        if hasattr(main_window, 'update_search_results'):
            main_window.update_search_results(results)
    except Exception as e:
        logger.error(f"高级搜索处理失败: {str(e)}")


def _add_market_analysis_tab(main_window):
    """添加市场分析标签页"""
    try:
        # 创建市场分析标签页
        if hasattr(main_window, 'main_tabs'):
            market_analysis_tab = QWidget()
            main_window.main_tabs.addTab(market_analysis_tab, "市场分析")
            
            # 设置市场分析UI
            _setup_market_analysis_ui(main_window, market_analysis_tab)
            logger.info("市场分析标签页添加完成")
    except Exception as e:
        logger.error(f"添加市场分析标签页失败: {str(e)}")


def _setup_market_analysis_ui(main_window, tab):
    """设置市场分析界面"""
    try:
        layout = QVBoxLayout(tab)
        
        # 创建子标签页
        tabs = QTabWidget()
        
        # 热点板块标签页
        hot_sectors_tab = QWidget()
        tabs.addTab(hot_sectors_tab, "热点板块")
        main_window.hot_sectors_tab = hot_sectors_tab
        
        # 板块轮动标签页
        rotations_tab = QWidget()
        tabs.addTab(rotations_tab, "板块轮动")
        main_window.rotations_tab = rotations_tab
        
        # 板块关联性标签页
        correlations_tab = QWidget()
        tabs.addTab(correlations_tab, "板块关联性")
        main_window.correlations_tab = correlations_tab
        
        # 资金流向标签页
        fund_flow_tab = QWidget()
        tabs.addTab(fund_flow_tab, "资金流向")
        main_window.fund_flow_tab = fund_flow_tab
        
        # 市场结构标签页
        market_structure_tab = QWidget()
        tabs.addTab(market_structure_tab, "市场结构")
        main_window.market_structure_tab = market_structure_tab
        
        # 添加刷新按钮
        refresh_btn = QPushButton("刷新市场分析")
        refresh_btn.clicked.connect(lambda: _refresh_market_analysis(main_window))
        
        # 添加到布局
        layout.addWidget(tabs)
        layout.addWidget(refresh_btn)
        
        # 保存引用
        main_window.market_analysis_tabs = tabs
    except Exception as e:
        logger.error(f"设置市场分析UI失败: {str(e)}")


def _refresh_market_analysis(main_window):
    """刷新市场分析"""
    try:
        # 启动市场分析线程
        analysis_thread = MarketAnalysisThread(main_window.market_analyzer)
        main_window.analysis_thread = analysis_thread
        
        # 连接信号
        analysis_thread.progress_signal.connect(
            lambda progress, message: main_window.statusBar().showMessage(f"市场分析: {message} ({progress}%)")
        )
        analysis_thread.finished_signal.connect(
            lambda results: _handle_market_analysis_results(main_window, results)
        )
        analysis_thread.error_signal.connect(
            lambda msg: QMessageBox.warning(main_window, "分析警告", msg)
        )
        
        # 启动线程
        analysis_thread.start()
        logger.info("市场分析刷新已启动")
    except Exception as e:
        logger.error(f"刷新市场分析失败: {str(e)}")
        QMessageBox.critical(main_window, "错误", f"刷新市场分析失败: {str(e)}")


def _handle_market_analysis_results(main_window, results):
    """处理市场分析结果"""
    try:
        # 更新状态栏
        main_window.statusBar().showMessage("市场分析完成!", 3000)
        
        # 这里仅作为示例，实际实现需要根据具体UI设计
        # 更新热点板块
        if 'hot_sectors' in results:
            logger.info(f"热点板块分析结果: {len(results['hot_sectors'].get('hot_sectors', []))} 个板块")
        
        # 更新板块轮动
        if 'sector_rotations' in results:
            logger.info(f"板块轮动分析结果: {len(results['sector_rotations'].get('rotations', []))} 次轮动")
        
        # 更新板块关联性
        if 'sector_correlations' in results:
            logger.info(f"板块关联性分析结果: {len(results['sector_correlations'].get('key_correlations', []))} 个关键关联")
        
        # 更新资金流向
        if 'capital_flow' in results:
            logger.info("资金流向分析结果已更新")
        
        # 更新市场结构
        if 'market_structure' in results:
            logger.info("市场结构分析结果已更新")
            
        # TODO: 实际UI更新，根据具体实现添加
    except Exception as e:
        logger.error(f"处理市场分析结果失败: {str(e)}")


def _setup_scheduled_market_analysis(main_window):
    """设置计划任务市场分析"""
    try:
        # 每天收盘后运行市场分析
        now = datetime.now()
        market_close = datetime(now.year, now.month, now.day, 15, 30)  # 15:30 收盘时间
        
        # 如果已经过了收盘时间，设置为明天
        if now > market_close:
            market_close += timedelta(days=1)
        
        # 计算等待时间
        wait_seconds = (market_close - now).total_seconds()
        
        # 设置定时器
        QTimer.singleShot(int(wait_seconds * 1000), lambda: _run_daily_analysis(main_window))
        logger.info(f"计划任务市场分析已设置，将在 {market_close.strftime('%Y-%m-%d %H:%M:%S')} 执行")
    except Exception as e:
        logger.error(f"设置计划任务市场分析失败: {str(e)}")


def _run_daily_analysis(main_window):
    """运行每日分析"""
    try:
        # 在后台线程中运行
        threading.Thread(target=lambda: _daily_analysis_task(main_window), daemon=True).start()
        
        # 设置下一次运行
        QTimer.singleShot(24 * 60 * 60 * 1000, lambda: _run_daily_analysis(main_window))
    except Exception as e:
        logger.error(f"运行每日分析失败: {str(e)}")


def _daily_analysis_task(main_window):
    """每日分析任务"""
    try:
        # 运行市场分析
        results = main_window.market_analyzer.run_full_analysis()
        
        # 通知用户
        main_window.statusBar().showMessage("每日市场分析已完成", 5000)
        logger.info("每日市场分析任务完成")
    except Exception as e:
        logger.error(f"每日分析任务失败: {str(e)}")


def start_data_loading(main_window, data_controller, trading_controller):
    """启动数据加载线程"""
    loading_thread = DataLoadingThread(data_controller, trading_controller)
    # 保存线程引用到主窗口，防止被垃圾回收
    main_window.loading_thread = loading_thread
    
    # 连接信号
    loading_thread.progress_signal.connect(lambda progress, message: main_window.statusBar().showMessage(f"加载中: {message} ({progress}%)"))
    loading_thread.finished_signal.connect(lambda data: on_data_loaded(main_window, data))
    loading_thread.error_signal.connect(lambda msg: QMessageBox.critical(main_window, "错误", msg))
    
    # 启动线程
    loading_thread.start()
    logger.info("数据加载线程已启动")


def on_data_loaded(main_window, data):
    """数据加载完成的回调函数"""
    # 更新市场状态
    if hasattr(main_window, 'update_market_status') and 'market_status' in data:
        main_window.update_market_status(data['market_status'])
    
    # 更新市场视图
    if hasattr(main_window, 'market_view') and hasattr(main_window.market_view, 'refresh_data'):
        try:
            main_window.market_view.refresh_data()
        except Exception as e:
            logger.error(f"刷新市场视图失败: {str(e)}")
    
    # 显示成功消息
    main_window.statusBar().showMessage("数据加载完成!", 3000)
    
    # 开始自动市场分析
    QTimer.singleShot(1000, lambda: _refresh_market_analysis(main_window))
    
    # 尝试激活窗口，确保显示在前台
    main_window._activate_window()
    
    logger.info("数据加载完成回调函数执行完毕")


def show_main_window(splash, main_window, data_controller, trading_controller):
    """显示主窗口并启动数据加载线程"""
    # 如果有启动画面，关闭它
    if splash:
        splash.finish(main_window)
    
    # 显示主窗口
    main_window.showMaximized()
    
    # 启动数据加载线程
    start_data_loading(main_window, data_controller, trading_controller)


def create_app_and_controllers():
    """创建应用和控制器"""
    app = initialize_application()
    data_controller, trading_controller = initialize_controllers()
    main_window = initialize_main_window(data_controller, trading_controller)
    return app, data_controller, trading_controller, main_window


def main():
    """应用主函数"""
    # 确保只有一个应用实例在运行
    if not ensure_single_instance():
        print("超神系统已经在运行！")
        # 尝试唤醒已有的实例
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(('localhost', 37337))
            sock.send(b'ACTIVATE')
            sock.close()
        except:
            pass
        return 0
    
    # 创建应用和控制器
    app, data_controller, trading_controller, main_window = create_app_and_controllers()
    
    # 显示启动画面
    splash_pix = QPixmap("gui/resources/splash.png")
    if not splash_pix.isNull():
        splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
        splash.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        splash.show()
        app.processEvents()
    else:
        splash = None
    
    # 延迟显示主窗口
    QTimer.singleShot(1000, lambda: show_main_window(splash, main_window, data_controller, trading_controller))
    
    # 执行应用
    return app.exec_()


def main_without_splash():
    """应用主函数 - 不显示启动画面"""
    # 确保只有一个应用实例在运行
    if not ensure_single_instance():
        print("超神系统已经在运行！")
        return 0
    
    # 创建应用和控制器
    app, data_controller, trading_controller, main_window = create_app_and_controllers()
    
    # 直接显示主窗口，无需启动画面
    show_main_window(None, main_window, data_controller, trading_controller)
    
    # 执行应用
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main()) 