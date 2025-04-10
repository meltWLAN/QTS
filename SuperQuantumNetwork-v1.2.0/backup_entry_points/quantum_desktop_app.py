#!/usr/bin/env python3
"""
超神量子系统 - 豪华桌面应用
专为中国市场分析设计的高级可视化平台
融合量子视觉特效和先进分析功能
"""

import sys
import os
import logging
import traceback
from datetime import datetime
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QSplashScreen, QMessageBox, 
                            QTabWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, 
                            QWidget, QFrame, QGridLayout, QScrollArea, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QPropertyAnimation, QSize, QEasingCurve
from PyQt5.QtGui import QPixmap, QIcon, QFont, QColor, QPainter, QPen, QPainterPath, QLinearGradient
from PyQt5.QtChart import QChart, QChartView, QLineSeries, QValueAxis, QSplineSeries, QScatterSeries, QAreaSeries
import qdarkstyle
from qt_material import apply_stylesheet

# 导入自定义UI组件
from quantum_ui_components import (QuantumLogo, QuantumLoadingBar, QuantumCard, 
                                 QuantumChart, QuantumHeatmap, QuantumRadarChart, 
                                 QuantumInfoPanel)

# 导入超神系统核心模块
try:
    from china_market_core import ChinaMarketCore
    from policy_analyzer import PolicyAnalyzer
    from sector_rotation_tracker import SectorRotationTracker
    CORE_MODULES_AVAILABLE = True
except ImportError as e:
    CORE_MODULES_AVAILABLE = False
    logging.error(f"无法导入核心模块: {str(e)}")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("supergod_desktop.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("SupergodDesktop")

class DataLoadingThread(QThread):
    """数据加载线程"""
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.market_core = None
        self.policy_analyzer = None
        self.sector_tracker = None
        
    def run(self):
        try:
            if not CORE_MODULES_AVAILABLE:
                self.error_signal.emit("核心模块导入失败，请确认安装了所有必要的模块")
                return
                
            # 阶段1: 初始化核心模块
            self.progress_signal.emit(10, "初始化中国市场分析核心...")
            self.market_core = ChinaMarketCore()
            
            self.progress_signal.emit(20, "初始化政策分析器...")
            self.policy_analyzer = PolicyAnalyzer()
            
            self.progress_signal.emit(30, "初始化板块轮动跟踪器...")
            self.sector_tracker = SectorRotationTracker()
            
            # 阶段2: 注册组件
            self.progress_signal.emit(40, "注册组件并集成系统...")
            self.market_core.register_policy_analyzer(self.policy_analyzer)
            self.market_core.register_sector_rotation_tracker(self.sector_tracker)
            
            # 阶段3: 加载示例数据
            self.progress_signal.emit(50, "生成示例市场数据...")
            # 这里调用测试系统中的数据生成逻辑
            try:
                import test_supergod_system as tester
                market_data = tester.generate_sample_market_data()
                policy_news = tester.generate_sample_policy_news(10)
                sectors = ['银行', '房地产', '医药生物', '食品饮料', '电子', '计算机',
                          '有色金属', '钢铁', '军工', '汽车', '家电', '电力设备']
                sector_data = tester.generate_sample_sector_data(sectors)
            except ImportError as e:
                logger.warning(f"未找到测试模块: {str(e)}，使用空数据")
                # 创建空数据以避免错误
                market_data = pd.DataFrame()
                policy_news = []
                sector_data = {}
            
            # 阶段4: 初始数据分析
            self.progress_signal.emit(60, "更新市场数据...")
            self.market_core.update_market_data(market_data)
            
            self.progress_signal.emit(70, "处理政策新闻...")
            for news in policy_news:
                self.policy_analyzer.add_policy_news(news)
                
            self.progress_signal.emit(80, "更新板块数据...")
            self.sector_tracker.update_sector_data(sector_data)
            
            # 阶段5: 执行综合分析
            self.progress_signal.emit(90, "执行综合市场分析...")
            market_analysis = self.market_core.analyze_market(market_data)
            
            # 激活高级市场预警系统并重新分析
            self.market_core.activate_advanced_warning_system(True)
            enhanced_analysis = self.market_core.analyze_market(market_data)
            
            # 如果有高级分析结果，使用增强版结果
            if enhanced_analysis and 'error' not in enhanced_analysis:
                market_analysis = enhanced_analysis
            
            sector_analysis = self.sector_tracker.analyze()
            
            # 完成并返回结果
            results = {
                "market_core": self.market_core,
                "policy_analyzer": self.policy_analyzer,
                "sector_tracker": self.sector_tracker,
                "market_data": market_data,
                "policy_news": policy_news,
                "sector_data": sector_data,
            }
            
            # 安全获取市场周期
            try:
                if hasattr(self.market_core, 'market_state') and 'current_cycle' in self.market_core.market_state:
                    cycle = self.market_core.market_state['current_cycle']
                    if hasattr(cycle, 'name'):
                        results["market_cycle"] = cycle.name
                    else:
                        results["market_cycle"] = str(cycle)
                else:
                    results["market_cycle"] = "UNKNOWN"
            except Exception as e:
                logger.warning(f"获取市场周期失败: {str(e)}")
                results["market_cycle"] = "UNKNOWN"
            
            # 安全获取其他市场状态数据
            try:
                results["cycle_confidence"] = self.market_core.market_state.get('cycle_confidence', 0.0)
                results["market_sentiment"] = self.market_core.market_state.get('market_sentiment', 0.0)
                results["policy_direction"] = self.market_core.market_state.get('policy_direction', 0.0)
            except Exception as e:
                logger.warning(f"获取市场状态数据失败: {str(e)}")
                results["cycle_confidence"] = 0.0
                results["market_sentiment"] = 0.0
                results["policy_direction"] = 0.0
            
            # 安全获取板块轮动数据
            try:
                rotation_state = self.sector_tracker.get_rotation_state()
                results["sector_rotation"] = rotation_state.get('rotation_direction', 'none')
            except Exception as e:
                logger.warning(f"获取板块轮动方向失败: {str(e)}")
                results["sector_rotation"] = "none"
                
            # 安全获取分析结果数据
            try:
                results["leading_sectors"] = sector_analysis.get('leading_sectors', [])
                results["lagging_sectors"] = sector_analysis.get('lagging_sectors', [])
            except Exception as e:
                logger.warning(f"获取板块数据失败: {str(e)}")
                results["leading_sectors"] = []
                results["lagging_sectors"] = []
            
            results["anomalies"] = []  # 使用空列表，后续可以修改为正确的方法
            results["recommendations"] = []  # 使用空列表，后续可以修改为正确的方法
            results["market_analysis"] = market_analysis
            results["sector_analysis"] = sector_analysis
            
            self.progress_signal.emit(100, "加载完成!")
            self.finished_signal.emit(results)
            
        except Exception as e:
            error_message = f"数据加载失败: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_message)
            self.error_signal.emit(error_message)

class QuantumSplashScreen(QSplashScreen):
    """量子风格的启动屏幕"""
    def __init__(self):
        pixmap = QPixmap(500, 500)
        pixmap.fill(Qt.transparent)
        super().__init__(pixmap)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # 创建UI
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(50, 50, 50, 50)
        self.layout.setAlignment(Qt.AlignCenter)
        
        # Logo
        self.logo = QuantumLogo(self)
        self.layout.addWidget(self.logo, alignment=Qt.AlignCenter)
        
        # 标题标签
        self.title_label = QLabel("超神量子共生系统", self)
        self.title_label.setStyleSheet("""
            font-family: '微软雅黑';
            font-size: 24px;
            color: #40a0ff;
            font-weight: bold;
        """)
        self.layout.addWidget(self.title_label, alignment=Qt.AlignCenter)
        
        # 副标题
        self.subtitle_label = QLabel("豪华版桌面系统", self)
        self.subtitle_label.setStyleSheet("""
            font-family: '微软雅黑';
            font-size: 16px;
            color: #80c0ff;
        """)
        self.layout.addWidget(self.subtitle_label, alignment=Qt.AlignCenter)
        
        # 间隔
        self.layout.addSpacing(20)
        
        # 加载进度条
        self.progress_bar = QuantumLoadingBar(self)
        self.layout.addWidget(self.progress_bar)
        
        # 状态标签
        self.status_label = QLabel("正在初始化系统...", self)
        self.status_label.setStyleSheet("""
            font-family: '微软雅黑';
            font-size: 14px;
            color: #a0a0c0;
        """)
        self.layout.addWidget(self.status_label, alignment=Qt.AlignCenter)
        
        # 显示版本号
        self.version_label = QLabel("v1.1.0", self)
        self.version_label.setStyleSheet("""
            font-family: '微软雅黑';
            font-size: 12px;
            color: #606080;
        """)
        self.layout.addWidget(self.version_label, alignment=Qt.AlignCenter)
    
    def update_progress(self, progress, message):
        """更新进度与消息"""
        self.progress_bar.set_progress(progress)
        self.status_label.setText(message)
        QApplication.processEvents()

class SupergodMainWindow(QMainWindow):
    """超神系统主窗口"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("超神量子共生系统 - 豪华桌面版")
        
        # 修复资源路径问题
        icon_path = "gui/resources/icon.png"
        if not os.path.exists(icon_path):
            # 尝试在其他可能的位置查找图标
            logger.warning(f"未找到图标: {icon_path}，尝试在其他位置查找")
            for possible_path in ["resources/icon.png", "icon.png", "gui/icon.png"]:
                if os.path.exists(possible_path):
                    icon_path = possible_path
                    break
            else:
                icon_path = None
                logger.warning("未找到有效的图标文件")
        
        if icon_path:
            self.setWindowIcon(QIcon(icon_path))
        
        self.resize(1280, 800)
        
        # 应用样式
        try:
            apply_stylesheet(QApplication.instance(), theme="dark_cyan")
        except Exception as e:
            logger.warning(f"应用Material样式失败: {str(e)}")
            QApplication.instance().setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        
        # 设置中心部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # 创建顶部标题栏
        self.create_header()
        
        # 创建标签页容器
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #30304D;
                background: #12122A;
                border-radius: 4px;
            }
            QTabBar::tab {
                background: #1A1A35;
                color: #7f95c0;
                padding: 8px 16px;
                border: 1px solid #30304D;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                min-width: 120px;
            }
            QTabBar::tab:selected {
                background: #12122A;
                color: #40a0ff;
                border-bottom: 2px solid #40a0ff;
            }
            QTabBar::tab:hover:!selected {
                background: #202040;
            }
        """)
        self.main_layout.addWidget(self.tab_widget)
        
        # 添加基本标签页
        self.setup_dashboard_tab()
        self.setup_market_analysis_tab()
        self.setup_policy_analysis_tab()
        self.setup_sector_rotation_tab()
        self.setup_simulation_tab()

        # 存储数据和分析结果
        self.data = {}
        self.market_core = None
        self.policy_analyzer = None
        self.sector_tracker = None
    
    def create_header(self):
        """创建顶部标题栏"""
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                          stop:0 #101025, stop:1 #202045);
                border-radius: 4px;
                border-bottom: 1px solid #303050;
            }
        """)
        header_frame.setFixedHeight(70)
        
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(15, 5, 15, 5)
        
        # Logo
        logo = QuantumLogo()
        header_layout.addWidget(logo)
        
        # 标题
        title_layout = QVBoxLayout()
        title_label = QLabel("超神量子共生系统")
        title_label.setStyleSheet("""
            font-family: '微软雅黑';
            font-size: 22px;
            color: #40a0ff;
            font-weight: bold;
        """)
        
        subtitle_label = QLabel("中国A股市场量子分析平台")
        subtitle_label.setStyleSheet("""
            font-family: '微软雅黑';
            font-size: 14px;
            color: #80c0ff;
        """)
        
        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)
        header_layout.addLayout(title_layout)
        
        # 添加伸展因子
        header_layout.addStretch()
        
        # 当前时间
        self.time_label = QLabel()
        self.time_label.setStyleSheet("""
            font-family: '微软雅黑';
            font-size: 14px;
            color: #a0a0c0;
        """)
        self.update_time()
        
        # 创建定时器更新时间
        timer = QTimer(self)
        timer.timeout.connect(self.update_time)
        timer.start(1000)  # 每秒更新一次
        
        header_layout.addWidget(self.time_label)
        
        self.main_layout.addWidget(header_frame)
    
    def update_time(self):
        """更新时间显示"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.setText(current_time)
    
    def setup_dashboard_tab(self):
        """设置仪表盘标签页"""
        dashboard_tab = QWidget()
        dashboard_layout = QVBoxLayout(dashboard_tab)
        
        # 仪表盘内容将在加载数据后填充
        self.dashboard_content = QWidget()
        dashboard_layout.addWidget(self.dashboard_content)
        
        self.tab_widget.addTab(dashboard_tab, "量子仪表盘")
    
    def setup_market_analysis_tab(self):
        """设置市场分析标签页"""
        market_tab = QWidget()
        self.tab_widget.addTab(market_tab, "市场分析")
    
    def setup_policy_analysis_tab(self):
        """设置政策分析标签页"""
        policy_tab = QWidget()
        self.tab_widget.addTab(policy_tab, "政策分析")
    
    def setup_sector_rotation_tab(self):
        """设置板块轮动标签页"""
        sector_tab = QWidget()
        self.tab_widget.addTab(sector_tab, "板块轮动")
    
    def setup_simulation_tab(self):
        """设置模拟交易标签页"""
        simulation_tab = QWidget()
        self.tab_widget.addTab(simulation_tab, "量子模拟")
        
    def update_dashboard(self, data):
        """更新仪表盘内容"""
        # 清除旧内容
        if self.dashboard_content.layout():
            # 清除旧布局中的所有小部件
            while self.dashboard_content.layout().count():
                item = self.dashboard_content.layout().takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
            # 删除旧布局
            QWidget().setLayout(self.dashboard_content.layout())
        
        # 创建新布局
        dashboard_layout = QGridLayout(self.dashboard_content)
        dashboard_layout.setContentsMargins(10, 10, 10, 10)
        dashboard_layout.setSpacing(15)
        
        # 使用量子卡片组件显示信息
        market_cycle_card = QuantumCard(
            "当前市场周期", 
            data["market_cycle"],
            f"置信度: {data['cycle_confidence']:.2f}",
            bg_color=(25, 35, 60)
        )
        dashboard_layout.addWidget(market_cycle_card, 0, 0)
        
        sentiment_card = QuantumCard(
            "市场情绪", 
            f"{data['market_sentiment']:.2f}",
            self.get_sentiment_description(data['market_sentiment']),
            bg_color=(30, 40, 70)
        )
        dashboard_layout.addWidget(sentiment_card, 0, 1)
        
        policy_card = QuantumCard(
            "政策方向", 
            f"{data['policy_direction']:.2f}",
            self.get_policy_description(data['policy_direction']),
            bg_color=(35, 45, 80)
        )
        dashboard_layout.addWidget(policy_card, 0, 2)
        
        rotation_card = QuantumCard(
            "板块轮动", 
            data['sector_rotation'],
            "活跃度: " + str(self.sector_tracker.get_rotation_state()['rotation_strength']),
            bg_color=(40, 50, 90)
        )
        dashboard_layout.addWidget(rotation_card, 0, 3)
        
        # 在dashboard_container中添加预警信号卡片
        self.warning_card = QuantumCard(
            title="市场预警信号",
            value="无预警",
            description="系统监测正常",
            bg_color=(50, 30, 60),
            parent=self.dashboard_content
        )
        
        # 添加市场结构分析卡片
        self.structure_card = QuantumCard(
            title="市场结构分析",
            value="随机游走",
            description="H=0.5, D=1.5, E=0",
            bg_color=(30, 50, 70),
            parent=self.dashboard_content
        )
        
        # 添加到布局
        dashboard_layout.addWidget(self.warning_card, 1, 2)
        dashboard_layout.addWidget(self.structure_card, 1, 3)
        
        # 中部内容 - 左侧推荐
        recommendations_panel = QuantumInfoPanel("投资建议")
        
        # 添加推荐内容
        recommendations = data.get("recommendations", [])
        if recommendations:
            for i, rec in enumerate(recommendations):
                recommendations_panel.add_item(f"{i+1}. {rec}")
        else:
            recommendations_panel.add_item("暂无推荐", color="#808090")
        
        recommendations_panel.add_spacer()
        dashboard_layout.addWidget(recommendations_panel, 1, 0, 2, 2)
        
        # 中部内容 - 右侧板块热度
        sectors_panel = QuantumInfoPanel("板块表现")
        
        # 板块表现
        leading_sectors = data.get("leading_sectors", [])
        if leading_sectors:
            sectors_panel.add_item("领先板块:", color="#70dd90")
            
            # 检查数据格式，适应可能的多种格式
            for sector_item in leading_sectors:
                if isinstance(sector_item, dict):
                    # 如果是字典格式
                    sector_name = sector_item.get('name', 'Unknown')
                    sector_perf = sector_item.get('short_term_return', 0.0)
                    sectors_panel.add_item(f"{sector_name}: {sector_perf:.2%}", color="#90e0a0", indent=1)
                elif isinstance(sector_item, (list, tuple)) and len(sector_item) >= 2:
                    # 如果是二元组格式
                    sector_name, sector_perf = sector_item[0], sector_item[1]
                    sectors_panel.add_item(f"{sector_name}: {sector_perf:.2%}", color="#90e0a0", indent=1)
                else:
                    # 其他格式，直接显示
                    sectors_panel.add_item(f"{sector_item}", color="#90e0a0", indent=1)
        
        lagging_sectors = data.get("lagging_sectors", [])
        if lagging_sectors:
            sectors_panel.add_item("滞后板块:", color="#dd7070")
            
            # 检查数据格式，适应可能的多种格式
            for sector_item in lagging_sectors:
                if isinstance(sector_item, dict):
                    # 如果是字典格式
                    sector_name = sector_item.get('name', 'Unknown')
                    sector_perf = sector_item.get('short_term_return', 0.0)
                    sectors_panel.add_item(f"{sector_name}: {sector_perf:.2%}", color="#e09090", indent=1)
                elif isinstance(sector_item, (list, tuple)) and len(sector_item) >= 2:
                    # 如果是二元组格式
                    sector_name, sector_perf = sector_item[0], sector_item[1]
                    sectors_panel.add_item(f"{sector_name}: {sector_perf:.2%}", color="#e09090", indent=1)
                else:
                    # 其他格式，直接显示
                    sectors_panel.add_item(f"{sector_item}", color="#e09090", indent=1)
        
        sectors_panel.add_spacer()
        dashboard_layout.addWidget(sectors_panel, 1, 2, 2, 2)
        
        # 底部 - 异常检测
        anomalies_panel = QuantumInfoPanel("异常检测")
        
        # 添加异常内容
        anomalies = data.get("anomalies", [])
        if anomalies:
            for anomaly in anomalies:
                anomalies_panel.add_item(anomaly, color="#e0b0d0")
        else:
            anomalies_panel.add_item("未检测到异常", color="#808090")
        
        anomalies_panel.add_spacer()
        dashboard_layout.addWidget(anomalies_panel, 3, 0, 1, 4)
        
        # 更新预警信号
        if hasattr(self, 'market_analysis') and self.market_analysis:
            warning_signals = self.market_analysis.get('warning_signals', [])
            
            if warning_signals:
                # 获取最重要的预警
                top_warning = max(warning_signals, key=lambda w: w.get('confidence', 0))
                warning_type = top_warning.get('type', 'unknown')
                warning_desc = top_warning.get('description', '未知预警')
                warning_conf = top_warning.get('confidence', 0)
                
                # 设置预警颜色
                if warning_conf > 0.8:
                    color = "#ff4040"  # 高风险红色
                elif warning_conf > 0.6:
                    color = "#ff8000"  # 中风险橙色
                else:
                    color = "#ffff00"  # 低风险黄色
                
                # 更新预警卡片
                self.warning_card.update_value(
                    f"{self._translate_warning_type(warning_type)}", 
                    f"{warning_desc} ({warning_conf:.2f})"
                )
                self.warning_card.setStyleSheet(f"QLabel#value_label {{ color: {color}; }}")
            else:
                # 无预警
                self.warning_card.update_value("无预警", "系统监测正常")
                self.warning_card.setStyleSheet("QLabel#value_label { color: #40ff40; }")
            
            # 更新市场结构分析
            market_structure = self.market_analysis.get('market_structure', {})
            if market_structure:
                # 获取市场结构信息
                regime = market_structure.get('regime', 'unknown')
                hurst = market_structure.get('hurst_exponent', 0.5)
                fractal = market_structure.get('fractal_dimension', 1.5)
                entropy = market_structure.get('entropy', 0)
                stability = market_structure.get('stability', 0.5)
                
                # 更新市场结构卡片
                self.structure_card.update_value(
                    f"{self._translate_regime(regime)}", 
                    f"H={hurst:.2f}, D={fractal:.2f}, 稳定性={stability:.2f}"
                )
                
                # 根据稳定性设置颜色
                if stability < 0.3:
                    self.structure_card.setStyleSheet("QLabel#value_label { color: #ff6060; }")
                elif stability < 0.6:
                    self.structure_card.setStyleSheet("QLabel#value_label { color: #ffff60; }")
                else:
                    self.structure_card.setStyleSheet("QLabel#value_label { color: #60ff60; }")
        
    def get_sentiment_description(self, sentiment):
        """获取情绪描述"""
        if sentiment > 0.6:
            return "市场情绪乐观"
        elif sentiment > 0.4:
            return "市场情绪中性"
        else:
            return "市场情绪谨慎"
    
    def get_policy_description(self, policy):
        """获取政策描述"""
        if policy > 0.6:
            return "政策偏紧"
        elif policy > 0.4:
            return "政策中性"
        elif policy > 0.2:
            return "政策偏松"
        else:
            return "政策宽松"

    def _translate_warning_type(self, warning_type):
        """翻译预警类型"""
        translations = {
            'turning_point': '拐点预警',
            'chaos_warning': '混沌预警',
            'ml_anomaly': '异常行为',
            'market_structure': '结构变化'
        }
        return translations.get(warning_type, warning_type)
    
    def _translate_regime(self, regime):
        """翻译市场结构状态"""
        translations = {
            'mean_reverting': '均值回归',
            'trending': '趋势性市场',
            'random_walk': '随机游走'
        }
        return translations.get(regime, regime)

def main():
    """应用入口函数"""
    try:
        # 初始化应用
        app = QApplication(sys.argv)
        app.setApplicationName("超神量子共生系统")
        
        # 修复图标路径问题
        icon_path = "gui/resources/icon.png"
        if not os.path.exists(icon_path):
            for possible_path in ["resources/icon.png", "icon.png", "gui/icon.png"]:
                if os.path.exists(possible_path):
                    icon_path = possible_path
                    break
            else:
                icon_path = None
                
        if icon_path:
            app.setWindowIcon(QIcon(icon_path))
        
        # 检查是否已导入核心模块
        if not CORE_MODULES_AVAILABLE:
            QMessageBox.warning(None, "模块导入警告", 
                               "无法导入超神系统核心模块。应用将以演示模式运行，部分功能可能不可用。")
        
        # 创建并显示启动画面
        splash = QuantumSplashScreen()
        splash.show()
        
        # 创建数据加载线程
        loading_thread = DataLoadingThread()
        loading_thread.progress_signal.connect(splash.update_progress)
        
        # 创建主窗口(但不显示)
        main_window = SupergodMainWindow()
        
        def on_data_loaded(data):
            """数据加载完成的回调"""
            main_window.data = data
            main_window.market_core = data["market_core"]
            main_window.policy_analyzer = data["policy_analyzer"]
            main_window.sector_tracker = data["sector_tracker"]
            
            # 更新仪表盘
            main_window.update_dashboard(data)
            
            # 延迟关闭启动画面并显示主窗口
            QTimer.singleShot(1000, lambda: show_main_window(splash, main_window))
        
        def on_loading_error(error):
            """数据加载错误的回调"""
            QMessageBox.critical(None, "加载错误", f"无法加载超神系统数据:\n{error}")
            sys.exit(1)
        
        # 连接信号
        loading_thread.finished_signal.connect(on_data_loaded)
        loading_thread.error_signal.connect(on_loading_error)
        
        # 开始加载数据
        loading_thread.start()
        
        # 执行应用
        return app.exec_()
    
    except Exception as e:
        logger.error(f"应用启动失败: {str(e)}\n{traceback.format_exc()}")
        QMessageBox.critical(None, "启动错误", f"超神系统启动失败:\n{str(e)}")
        return 1

def show_main_window(splash, main_window):
    """显示主窗口并关闭启动画面"""
    main_window.show()
    splash.finish(main_window)

if __name__ == "__main__":
    sys.exit(main()) 