#!/usr/bin/env python3
"""
超神量子共生系统 - 桌面版
高级量子金融分析平台桌面应用
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import traceback
from typing import Dict, List, Tuple, Any, Optional
import random

# PyQt5导入
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QTabWidget, QSplitter,
                            QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
                            QComboBox, QLineEdit, QTextEdit, QFrame, QGroupBox, 
                            QCheckBox, QRadioButton, QFileDialog, QMessageBox,
                            QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar,
                            QSlider, QSpinBox, QDoubleSpinBox, QScrollArea, QStatusBar,
                            QToolBar, QAction, QMenu, QSystemTrayIcon, QStyle, QDialog,
                            QTextBrowser)
from PyQt5.QtGui import (QIcon, QPixmap, QFont, QColor, QPalette, QBrush, QMovie,
                        QFontDatabase, QCursor, QKeySequence, QTextCursor)
from PyQt5.QtCore import (Qt, QSize, QThread, pyqtSignal, pyqtSlot, QTimer, QSettings,
                        QUrl, QRect, QPropertyAnimation, QEasingCurve, QByteArray)

# 图表绘制
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import mplfinance as mpf
import seaborn as sns

# 导入超神系统组件
try:
    from china_market_core import ChinaMarketCore
    from policy_analyzer import PolicyAnalyzer
    from sector_rotation_tracker import SectorRotationTracker
    from supergod_enhancer import get_enhancer
    from chaos_theory_framework import get_chaos_analyzer
    from quantum_dimension_enhancer import get_dimension_enhancer
    from quantum_engine import QuantumEngine  # 添加正确的导入

    # 添加DataConnector类
    class DataConnector:
        """数据连接器基类"""
        def __init__(self):
            self.logger = logging.getLogger("DataConnector")
            
        def get_market_data(self, symbol, start_date=None, end_date=None):
            """获取市场数据"""
            raise NotImplementedError("子类必须实现此方法")
            
        def get_sector_data(self):
            """获取板块数据"""
            raise NotImplementedError("子类必须实现此方法")
            
        def close(self):
            """关闭连接"""
            pass

except ImportError as e:
    print(f"导入超神组件失败: {str(e)}")
    print("确保所有超神系统组件已安装")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("supergod_desktop.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SupergodDesktop")

# 主题色彩定义
class SupergodColors:
    """超神系统色彩方案"""
    PRIMARY_DARK = "#1a1a2e"    # 主背景色 (深蓝黑)
    SECONDARY_DARK = "#16213e"  # 次背景色 (深蓝)
    ACCENT_DARK = "#0f3460"     # 强调色 (中蓝)
    HIGHLIGHT = "#e94560"       # 高亮色 (红色)
    POSITIVE = "#4cd97b"        # 上涨/正向 (绿色)
    NEGATIVE = "#e94560"        # 下跌/负向 (红色)
    NEUTRAL = "#7c83fd"         # 中性 (紫色)
    TEXT_PRIMARY = "#ffffff"    # 主文本色 (白色)
    TEXT_SECONDARY = "#b0b0b0"  # 次要文本 (灰色)
    GRID_LINES = "#2a2a4a"      # 网格线 (深灰蓝)
    PANEL_BG = "#1e1e30"        # 面板背景 (深蓝灰)
    CODE_BG = "#2d2d44"         # 代码背景 (深紫)

# 添加MarketAnalyzer类
class MarketAnalyzer:
    """市场分析器"""
    def __init__(self):
        self.logger = logging.getLogger("MarketAnalyzer")
        
    def analyze_market(self, data):
        """分析市场数据"""
        try:
            results = {
                'trend': self._analyze_trend(data),
                'momentum': self._analyze_momentum(data),
                'volatility': self._analyze_volatility(data),
                'volume': self._analyze_volume(data)
            }
            return results
        except Exception as e:
            self.logger.error(f"市场分析失败: {str(e)}")
            return None
            
    def _analyze_trend(self, data):
        """分析趋势"""
        if 'close' not in data.columns:
            return None
        try:
            # 计算移动平均线
            ma5 = data['close'].rolling(window=5).mean()
            ma20 = data['close'].rolling(window=20).mean()
            
            # 判断趋势
            if ma5.iloc[-1] > ma20.iloc[-1]:
                return 'upward'
            elif ma5.iloc[-1] < ma20.iloc[-1]:
                return 'downward'
            else:
                return 'sideways'
        except Exception:
            return None
            
    def _analyze_momentum(self, data):
        """分析动量"""
        if 'close' not in data.columns:
            return None
        try:
            # 计算动量
            momentum = data['close'].pct_change(5)
            return momentum.iloc[-1]
        except Exception:
            return None
            
    def _analyze_volatility(self, data):
        """分析波动率"""
        if 'close' not in data.columns:
            return None
        try:
            # 计算波动率
            returns = data['close'].pct_change()
            volatility = returns.std() * np.sqrt(252)  # 年化波动率
            return volatility
        except Exception:
            return None
            
    def _analyze_volume(self, data):
        """分析成交量"""
        if 'volume' not in data.columns:
            return None
        try:
            # 计算成交量变化
            volume_ma5 = data['volume'].rolling(window=5).mean()
            volume_ma20 = data['volume'].rolling(window=20).mean()
            
            # 判断成交量趋势
            if volume_ma5.iloc[-1] > volume_ma20.iloc[-1]:
                return 'increasing'
            elif volume_ma5.iloc[-1] < volume_ma20.iloc[-1]:
                return 'decreasing'
            else:
                return 'stable'
        except Exception:
            return None
            
    def calibrate(self):
        """校准分析器"""
        self.logger.info("校准市场分析器")
        # 实现校准逻辑
        
    def adjust_sensitivity(self):
        """调整灵敏度"""
        self.logger.info("调整市场分析器灵敏度")
        # 实现灵敏度调整逻辑

# 分析线程类
class AnalysisThread(QThread):
    """后台分析线程，避免UI阻塞"""
    # 信号定义
    analysis_progress = pyqtSignal(int, str)
    analysis_complete = pyqtSignal(dict)
    analysis_error = pyqtSignal(str)
    
    def __init__(self, market_data, modules, config=None):
        super().__init__()
        self.market_data = market_data
        self.modules = modules
        self.config = config or {}
        self.results = {}
        
    def run(self):
        """运行分析任务"""
        try:
            market_core, policy_analyzer, sector_tracker = self.modules.get('core', (None, None, None))
            enhancer = self.modules.get('enhancer')
            chaos_analyzer = self.modules.get('chaos')
            dimension_enhancer = self.modules.get('dimensions')
            
            # 进度报告
            self.analysis_progress.emit(0, "开始分析...")
            
            # 市场分析
            if market_core:
                self.analysis_progress.emit(10, "执行市场核心分析...")
                market_analysis = market_core.analyze_market(self.market_data)
                self.results['market_analysis'] = market_analysis
            
            # 政策分析
            if policy_analyzer:
                self.analysis_progress.emit(25, "执行政策分析...")
                policy_analysis = policy_analyzer.analyze_policy_environment()
                self.results['policy_analysis'] = policy_analysis
            
            # 板块轮动分析
            if sector_tracker:
                self.analysis_progress.emit(40, "执行板块轮动分析...")
                sector_analysis = sector_tracker.analyze_rotation()
                self.results['sector_analysis'] = sector_analysis
            
            # 混沌理论分析
            if chaos_analyzer:
                self.analysis_progress.emit(60, "执行混沌理论分析...")
                if 'close' in self.market_data.columns:
                    chaos_results = chaos_analyzer.analyze(self.market_data['close'].values)
                    self.results['chaos_analysis'] = chaos_results
                    
                    # 生成混沌吸引子图表
                    chaos_analyzer.plot_phase_space("market_chaos_attractor.png")
            
            # 量子维度分析
            if dimension_enhancer:
                self.analysis_progress.emit(75, "执行量子维度分析...")
                dimensions_data = dimension_enhancer.enhance_dimensions(self.market_data)
                dimension_state = dimension_enhancer.get_dimension_state()
                
                self.results['quantum_dimensions'] = {
                    'data': dimensions_data,
                    'state': dimension_state
                }
            
            # 增强
            if enhancer and self.results:
                self.analysis_progress.emit(90, "应用超神增强...")
                enhanced_results = enhancer.enhance(self.market_data, self.results)
                self.results = enhanced_results
            
            # 生成预测
            self.analysis_progress.emit(95, "生成预测...")
            predictions = self._generate_predictions()
            self.results['predictions'] = predictions
            
            # 完成
            self.analysis_progress.emit(100, "分析完成")
            self.analysis_complete.emit(self.results)
            
        except Exception as e:
            logger.error(f"分析过程发生错误: {str(e)}")
            logger.error(traceback.format_exc())
            self.analysis_error.emit(f"分析错误: {str(e)}")
    
    def _generate_predictions(self):
        """根据分析结果生成预测"""
        predictions = {
            'short_term': {'direction': None, 'confidence': 0.0, 'time_frame': '1-3天'},
            'medium_term': {'direction': None, 'confidence': 0.0, 'time_frame': '1-2周'},
            'long_term': {'direction': None, 'confidence': 0.0, 'time_frame': '1-3月'},
            'market_state': {'current_phase': None, 'next_phase': None},
            'critical_points': [],
            'anomalies': []
        }
        
        # 从不同分析结果中整合预测
        # 市场分析
        market_analysis = self.results.get('market_analysis', {})
        if market_analysis:
            if 'current_cycle' in market_analysis:
                predictions['market_state']['current_phase'] = str(market_analysis['current_cycle'])
                
            # 简单市场周期预测
            if 'market_sentiment' in market_analysis:
                sentiment = market_analysis.get('market_sentiment', 0)
                if sentiment > 0.3:
                    predictions['short_term']['direction'] = 'bullish'
                elif sentiment < -0.3:
                    predictions['short_term']['direction'] = 'bearish'
                else:
                    predictions['short_term']['direction'] = 'sideways'
                    
                predictions['short_term']['confidence'] = min(0.9, abs(sentiment) + 0.3)
        
        # 混沌分析
        chaos_analysis = self.results.get('chaos_analysis', {})
        if chaos_analysis:
            # 从混沌分析提取状态和稳定性
            if 'market_regime' in chaos_analysis:
                regime = chaos_analysis['market_regime']
                
                # 根据混沌分析预测方向
                if regime in ['trending', 'complex_trending']:
                    predictions['medium_term']['direction'] = 'bullish'
                elif regime in ['mean_reverting', 'chaotic_reverting']:
                    predictions['medium_term']['direction'] = 'bearish'
                elif regime in ['edge_of_chaos']:
                    predictions['medium_term']['direction'] = 'volatile'
                else:
                    predictions['medium_term']['direction'] = 'sideways'
                
                # 设置置信度
                predictions['medium_term']['confidence'] = max(0.3, min(0.9, chaos_analysis.get('stability', 0.5)))
                
                # 提取临界点
                if 'critical_points' in chaos_analysis and chaos_analysis['critical_points']:
                    predictions['critical_points'] = chaos_analysis['critical_points']
        
        # 量子维度分析
        quantum_dimensions = self.results.get('quantum_dimensions', {})
        if quantum_dimensions and 'state' in quantum_dimensions:
            dimension_state = quantum_dimensions['state']
            
            # 使用量子维度状态调整预测
            if 'energy_potential' in dimension_state:
                energy_potential = dimension_state['energy_potential']['value']
                
                # 使用能量势能预测长期方向
                if energy_potential > 0.7:
                    predictions['long_term']['direction'] = 'bullish'
                    predictions['long_term']['confidence'] = min(0.8, energy_potential)
                elif energy_potential < 0.3:
                    predictions['long_term']['direction'] = 'bearish'
                    predictions['long_term']['confidence'] = min(0.8, 1 - energy_potential)
                else:
                    predictions['long_term']['direction'] = 'sideways'
                    predictions['long_term']['confidence'] = 0.5
            
            # 时间相干性影响
            if 'temporal_coherence' in dimension_state:
                temporal_coherence = dimension_state['temporal_coherence']['value']
                if temporal_coherence < 0.3 and 'next_phase' not in predictions['market_state']:
                    predictions['market_state']['transition_probability'] = 1 - temporal_coherence
            
            # 混沌度影响异常检测
            if 'chaos_degree' in dimension_state:
                chaos_degree = dimension_state['chaos_degree']['value']
                if chaos_degree > 0.75:
                    predictions['anomalies'].append({
                        'type': 'high_chaos',
                        'severity': chaos_degree,
                        'description': '市场混沌度异常高，可能出现剧烈波动'
                    })
        
        return predictions

# 数据加载线程
class DataLoaderThread(QThread):
    """后台数据加载线程"""
    data_loaded = pyqtSignal(pd.DataFrame)
    loading_progress = pyqtSignal(int, str)
    loading_error = pyqtSignal(str)
    
    def __init__(self, data_source, params=None):
        super().__init__()
        self.data_source = data_source
        self.params = params or {}
        
    def run(self):
        """运行数据加载任务"""
        try:
            self.loading_progress.emit(10, "开始加载数据...")
            
            # 根据数据源类型加载
            if self.data_source == 'demo':
                self.loading_progress.emit(30, "生成演示数据...")
                data = self._generate_demo_data()
                
            elif self.data_source == 'csv':
                self.loading_progress.emit(30, "从CSV加载数据...")
                file_path = self.params.get('file_path')
                if not file_path or not os.path.exists(file_path):
                    raise ValueError("CSV文件路径无效")
                data = pd.read_csv(file_path)
                
            elif self.data_source == 'api':
                self.loading_progress.emit(30, "从API加载数据...")
                # 这里可以添加实际的API数据获取代码
                symbol = self.params.get('symbol', '000001.SH')
                start_date = self.params.get('start_date')
                end_date = self.params.get('end_date')
                
                # 示例，实际环境需要连接真实API
                data = self._generate_demo_data()
                
            else:
                raise ValueError(f"不支持的数据源: {self.data_source}")
            
            # 数据预处理
            self.loading_progress.emit(70, "处理数据...")
            data = self._preprocess_data(data)
            
            # 完成
            self.loading_progress.emit(100, "数据加载完成")
            self.data_loaded.emit(data)
            
        except Exception as e:
            logger.error(f"数据加载错误: {str(e)}")
            logger.error(traceback.format_exc())
            self.loading_error.emit(f"数据加载错误: {str(e)}")
    
    def _generate_demo_data(self):
        """生成演示数据"""
        # 生成日期序列
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # 创建价格序列 - 添加趋势、循环和噪声
        base = 3000
        trend = np.linspace(0, 0.15, len(dates))
        cycle1 = 0.05 * np.sin(np.linspace(0, 4*np.pi, len(dates)))
        cycle2 = 0.03 * np.sin(np.linspace(0, 12*np.pi, len(dates)))
        noise = np.random.normal(0, 0.01, len(dates))
        
        # 计算每日涨跌幅
        changes = np.diff(np.concatenate([[0], trend])) + cycle1 + cycle2 + noise
        
        # 计算价格
        prices = base * np.cumprod(1 + changes)
        
        # 生成成交量
        volume_base = 1e9
        volume_trend = np.linspace(0, 0.3, len(dates))
        volume_cycle = 0.2 * np.sin(np.linspace(0, 6*np.pi, len(dates)))
        volume_noise = np.random.normal(0, 0.15, len(dates))
        volumes = volume_base * (1 + volume_trend + volume_cycle + volume_noise)
        volumes = np.abs(volumes)  # 确保成交量为正
        
        # 创建数据框
        data = pd.DataFrame({
            'date': dates,
            'open': prices * (1 - 0.005 * np.random.random(len(dates))),
            'high': prices * (1 + 0.01 * np.random.random(len(dates))),
            'low': prices * (1 - 0.01 * np.random.random(len(dates))),
            'close': prices,
            'volume': volumes,
            'turnover_rate': 0.5 * (1 + 0.5 * np.random.random(len(dates)))
        })
        
        return data
    
    def _preprocess_data(self, data):
        """预处理数据"""
        # 确保日期列正确
        if 'date' in data.columns:
            if not pd.api.types.is_datetime64_any_dtype(data['date']):
                data['date'] = pd.to_datetime(data['date'])
            data.set_index('date', inplace=True)
            
        # 确保必要列存在
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in data.columns:
                logger.warning(f"数据缺少必要列: {col}，使用估算值")
                if col == 'volume':
                    data[col] = 1e9 * np.random.random(len(data))
                else:
                    # 如果close存在，用close估算其他价格列
                    if 'close' in data.columns:
                        base = data['close'].values
                        if col == 'open':
                            data[col] = base * (1 - 0.005 * np.random.random(len(base)))
                        elif col == 'high':
                            data[col] = base * (1 + 0.01 * np.random.random(len(base)))
                        elif col == 'low':
                            data[col] = base * (1 - 0.01 * np.random.random(len(base)))
                    else:
                        # 如果连close都没有，那就随机生成
                        data[col] = 3000 * (1 + 0.1 * np.random.random(len(data)))
        
        # 添加额外的技术指标列
        self._add_technical_indicators(data)
        
        return data
    
    def _add_technical_indicators(self, data):
        """添加技术指标"""
        # 移动平均线
        data['ma5'] = data['close'].rolling(5).mean()
        data['ma10'] = data['close'].rolling(10).mean()
        data['ma20'] = data['close'].rolling(20).mean()
        data['ma60'] = data['close'].rolling(60).mean()
        
        # 成交量移动平均
        data['volume_ma5'] = data['volume'].rolling(5).mean()
        
        # 波动率 (20日标准差)
        data['volatility'] = data['close'].pct_change().rolling(20).std()
        
        # MACD
        exp12 = data['close'].ewm(span=12, adjust=False).mean()
        exp26 = data['close'].ewm(span=26, adjust=False).mean()
        data['macd'] = exp12 - exp26
        data['signal'] = data['macd'].ewm(span=9, adjust=False).mean()
        data['histogram'] = data['macd'] - data['signal']
        
        return data 

class SupergodDesktop(QMainWindow):
    """超神量子共生系统桌面应用主窗口类"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("超神量子共生系统 - 桌面版")
        self.resize(1200, 800)
        self.config_file = os.path.join(os.path.expanduser('~'), '.supergod_config.json')
        self.config = self._load_default_config()
        self.market_data = None
        self.analysis_results = {}
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(lambda: self.log("自动刷新功能尚未实现"))
        self.data_connector = None  # 添加数据连接器属性
        self.enhancement_modules = None  # 添加增强模块属性
        
        # 模块初始化
        self.modules = {}
        
        # 初始化UI
        self.initUI()
        
        # 记录初始化日志
        logger.info("超神量子共生系统 - 桌面版已加载")
        logger.info(f"已加载系统配置: {self.config_file}")
        logger.info(f"欢迎使用超神量子共生系统 - 桌面版")
        logger.info(f"系统版本: 1.0.0")
        logger.info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"输入'help'查看可用命令")
        
        # 加载配置和显示欢迎信息
        self._load_config()
        self.show_welcome_message()
    
    def set_data_connector(self, data_connector):
        """
        设置数据连接器，用于从外部传入数据源
        
        参数:
            data_connector: 数据连接器实例，如TushareDataConnector或AKShareDataConnector
        """
        self.data_connector = data_connector
        self.log(f"已设置数据连接器: {data_connector.__class__.__name__}")
        
        # 尝试从数据连接器获取初始数据
        try:
            if hasattr(self.data_connector, 'get_market_data'):
                symbol = "000001.SH"  # 默认使用上证指数
                self.log(f"正在从数据连接器获取市场数据: {symbol}")
                data = self.data_connector.get_market_data(symbol)
                if data is not None and not data.empty:
                    self.market_data = data
                    self.log(f"成功获取市场数据，共{len(data)}条记录")
                    # 更新UI显示
                    self.update_market_display()
                    # 启用分析按钮
                    self.run_analysis_btn.setEnabled(True)
        except Exception as e:
            self.log(f"从数据连接器获取数据失败: {str(e)}", level="error")
    
    def set_enhancement_modules(self, enhancement_modules):
        """
        设置增强模块，用于从外部传入增强功能
        
        参数:
            enhancement_modules: 增强模块字典，包含各种增强功能模块
        """
        self.enhancement_modules = enhancement_modules
        if enhancement_modules:
            module_names = ", ".join(enhancement_modules.keys())
            self.log(f"已设置增强模块: {module_names}")
            
            # 将增强模块添加到分析模块中
            for name, module in enhancement_modules.items():
                self.modules[name] = module
                
            # 如果已经有市场数据，则运行分析
            if self.market_data is not None and not self.market_data.empty:
                try:
                    self.log("执行增强分析...")
                    self.run_analysis()
                except Exception as e:
                    self.log(f"执行增强分析失败: {str(e)}", level="error")
        else:
            self.log("未提供增强模块")
    
    def update_market_display(self):
        """更新市场数据显示"""
        # 这里实现更新UI显示市场数据的逻辑
        try:
            if self.market_data is not None and not self.market_data.empty:
                # 更新最后一次更新时间
                self.log(f"最后更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                # 如果已实现相关UI组件，可以更新它们
                # 例如：self.market_data_table.setData(self.market_data)
        except Exception as e:
            self.log(f"更新市场数据显示失败: {str(e)}")
            logger.error(f"更新市场数据显示失败: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _load_default_config(self):
        """加载默认配置"""
        return {
            'auto_refresh': False,
            'refresh_interval': 60,  # 秒
            'debug_mode': False,
            'chart_style': 'dark',
            'show_welcome': True,
            'default_data_source': 'demo'
        }
    
    def initUI(self):
        """初始化用户界面"""
        self.setWindowTitle("超神量子共生系统 - 桌面版")
        self.setGeometry(100, 100, 1200, 800)

        # 创建主窗口布局
        self.main_layout = QVBoxLayout()
        
        # 创建进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setValue(0)
        self.progress_bar.hide()  # 初始隐藏
        
        # 创建运行分析按钮
        self.run_analysis_btn = QPushButton("运行分析")
        self.run_analysis_btn.clicked.connect(self.run_analysis)
        self.run_analysis_btn.setEnabled(False)  # 初始禁用，直到数据加载完成

        # 创建标签页
        self.tab_widget = QTabWidget()
        
        # 创建各个标签页的内容
        self.market_tab = QWidget()
        self.analysis_tab = QWidget()
        self.results_tab = QWidget()
        self.prediction_tab = QWidget()
        self.command_tab = QWidget()
        
        # 创建市场数据表格
        self.market_table = QTableWidget()
        self.market_table.setColumnCount(7)
        self.market_table.setHorizontalHeaderLabels(["日期", "开盘", "最高", "最低", "收盘", "成交量", "涨跌幅"])
        
        # 创建结果显示文本框
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        
        # 创建预测显示文本框
        self.prediction_text = QTextEdit()
        self.prediction_text.setReadOnly(True)
        
        # 设置标签页内容
        market_layout = QVBoxLayout()
        market_layout.addWidget(self.market_table)
        self.market_tab.setLayout(market_layout)
        
        analysis_layout = QVBoxLayout()
        analysis_layout.addWidget(self.run_analysis_btn)
        self.analysis_tab.setLayout(analysis_layout)
        
        results_layout = QVBoxLayout()
        results_layout.addWidget(self.results_text)
        self.results_tab.setLayout(results_layout)
        
        prediction_layout = QVBoxLayout()
        prediction_layout.addWidget(self.prediction_text)
        self.prediction_tab.setLayout(prediction_layout)
        
        # 添加标签页
        self.tab_widget.addTab(self.market_tab, "市场数据")
        self.tab_widget.addTab(self.analysis_tab, "分析")
        self.tab_widget.addTab(self.results_tab, "结果")
        self.tab_widget.addTab(self.prediction_tab, "预测")
        self.tab_widget.addTab(self.command_tab, "命令行")

        # 将标签页添加到主窗口布局
        self.main_layout.addWidget(self.tab_widget)
        self.main_layout.addWidget(self.progress_bar)

        # 创建主窗口
        self.central_widget = QWidget()
        self.central_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.central_widget)

        # 初始化变量
        self.market_data = None
        self.analysis_results = None
        self.modules = {}
        self.config = {}

        # 创建数据加载线程
        self.loader_thread = DataLoaderThread('demo')
        self.loader_thread.data_loaded.connect(self.on_data_loaded)
        self.loader_thread.loading_error.connect(self.on_loading_error)

        # 创建分析线程
        self.analysis_thread = AnalysisThread(self.market_data, self.modules, self.config)
        self.analysis_thread.analysis_progress.connect(self.update_analysis_progress)
        self.analysis_thread.analysis_complete.connect(self.on_analysis_complete)
        self.analysis_thread.analysis_error.connect(self.on_analysis_error)

        # 创建命令行输入框
        self.cmd_input = QLineEdit()
        self.cmd_input.setPlaceholderText("输入命令")
        self.cmd_input.returnPressed.connect(self.execute_command)

        # 创建日志显示文本框
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: black; color: white;")

        # 创建命令行标签
        self.cmd_label = QLabel("命令行:")

        # 创建命令行布局
        self.cmd_layout = QVBoxLayout()
        self.cmd_layout.addWidget(self.cmd_label)
        self.cmd_layout.addWidget(self.cmd_input)
        self.cmd_layout.addWidget(self.log_text)

        # 设置命令行标签页布局
        self.command_tab.setLayout(self.cmd_layout)

        # 添加文件菜单
        file_menu = self.menuBar().addMenu('文件')
        
        # 打开文件动作
        open_action = QAction(QIcon.fromTheme('document-open'), '打开...', self)
        open_action.setShortcut('Ctrl+O')
        open_action.setStatusTip('打开数据文件')
        open_action.triggered.connect(self.open_data_file)
        file_menu.addAction(open_action)
        
        # 保存结果动作
        save_action = QAction(QIcon.fromTheme('document-save'), '保存结果...', self)
        save_action.setShortcut('Ctrl+S')
        save_action.setStatusTip('保存分析结果')
        save_action.triggered.connect(self.save_results)
        file_menu.addAction(save_action)
        
        # 生成报告动作
        report_action = QAction(QIcon.fromTheme('document-properties'), '生成分析报告...', self)
        report_action.setShortcut('Ctrl+R')
        report_action.setStatusTip('生成综合市场分析报告')
        report_action.triggered.connect(self.generate_market_report)
        file_menu.addAction(report_action)
        
        # 导出图表动作
        export_action = QAction(QIcon.fromTheme('image-x-generic'), '导出图表...', self)
        export_action.setStatusTip('导出图表为图片')
        export_action.triggered.connect(self.export_chart)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        # 退出动作
        exit_action = QAction(QIcon.fromTheme('application-exit'), '退出', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('退出应用')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 添加分析菜单
        analysis_menu = self.menuBar().addMenu('分析')
        
        # 运行分析动作
        run_analysis_action = QAction(QIcon.fromTheme('system-run'), '运行分析', self)
        run_analysis_action.setShortcut('F5')
        run_analysis_action.setStatusTip('运行市场分析')
        run_analysis_action.triggered.connect(self.run_analysis)
        analysis_menu.addAction(run_analysis_action)
        
        # 刷新数据动作
        refresh_action = QAction(QIcon.fromTheme('view-refresh'), '刷新数据', self)
        refresh_action.setShortcut('F6')
        refresh_action.setStatusTip('刷新市场数据')
        refresh_action.triggered.connect(lambda: self.log("刷新数据功能尚未实现"))
        analysis_menu.addAction(refresh_action)
        
        # 自动刷新动作
        auto_refresh_action = QAction('自动刷新', self)
        auto_refresh_action.setCheckable(True)
        auto_refresh_action.setStatusTip('启用自动刷新')
        auto_refresh_action.triggered.connect(self.toggle_auto_refresh)
        analysis_menu.addAction(auto_refresh_action)

        # 创建工具栏
        toolbar = self.addToolBar('主工具栏')
        toolbar.setIconSize(QSize(24, 24))
        
        # 添加工具栏按钮
        toolbar.addAction(open_action)
        toolbar.addAction(save_action)
        toolbar.addAction(report_action)
        toolbar.addSeparator()
        
        # 添加分析按钮到工具栏
        toolbar.addAction(run_analysis_action)
        toolbar.addAction(refresh_action)
        
        # 创建状态栏
        self.statusBar().showMessage("欢迎使用超神量子共生系统")
        
        # 加载示例数据
        self.log("正在加载演示数据...")
        self.loader_thread.start()

    def on_data_loaded(self, data):
        """数据加载完成回调"""
        self.market_data = data
        self.log(f"数据加载完成，共 {len(data)} 条记录", level="success")
        
        # 更新市场数据表
        self.update_market_table()
        
        # 更新图表
        self.update_chart()
        
        # 启用分析按钮
        self.run_analysis_btn.setEnabled(True)
        
        # 隐藏进度条
        self.progress_bar.hide()
        self.statusBar().showMessage("数据加载完成")
    
    def on_loading_error(self, error_message):
        """数据加载错误回调"""
        self.log(error_message, level="error")
        self.show_error_dialog("数据加载错误", error_message)
        self.progress_bar.hide()
        self.statusBar().showMessage("数据加载失败")
    
    def update_analysis_progress(self, value, message):
        """更新分析进度"""
        self.progress_bar.setValue(value)
        self.statusBar().showMessage(message)
    
    def on_analysis_complete(self, results):
        """分析完成回调"""
        self.analysis_results = results
        self.log("分析完成", level="success")
        
        # 隐藏进度条
        self.progress_bar.hide()
        
        # 更新结果显示
        self.update_results_display()
        
        # 更新预测显示
        if 'predictions' in results:
            self.update_predictions_display()
        
        self.statusBar().showMessage("分析完成")
    
    def on_analysis_error(self, error_message):
        """分析错误回调"""
        self.log(error_message, level="error")
        self.show_error_dialog("分析错误", error_message)
        self.progress_bar.hide()
        self.statusBar().showMessage("分析失败")
    
    def update_results_display(self):
        """更新结果显示"""
        if not self.analysis_results:
            return
            
        # 创建结果文本
        result_text = "===== 分析结果 =====\n\n"
        
        # 添加市场分析结果
        if 'market_analysis' in self.analysis_results:
            market_analysis = self.analysis_results['market_analysis']
            result_text += "【市场分析】\n"
            for key, value in market_analysis.items():
                result_text += f"{key}: {value}\n"
            result_text += "\n"
        
        # 添加混沌理论分析结果
        if 'chaos_analysis' in self.analysis_results:
            chaos_analysis = self.analysis_results['chaos_analysis']
            result_text += "【混沌理论分析】\n"
            for key, value in chaos_analysis.items():
                if key not in ['attractors', 'critical_points', 'fractal_patterns']:  # 排除复杂数据
                    result_text += f"{key}: {value}\n"
            result_text += "\n"
        
        # 添加量子维度分析结果
        if 'quantum_dimensions' in self.analysis_results and 'state' in self.analysis_results['quantum_dimensions']:
            dimension_state = self.analysis_results['quantum_dimensions']['state']
            result_text += "【量子维度分析】\n"
            for dim_name, dim_data in dimension_state.items():
                if isinstance(dim_data, dict) and 'value' in dim_data:
                    result_text += f"{dim_name}: {dim_data['value']:.3f}\n"
            result_text += "\n"
        
        # 设置结果文本
        if hasattr(self, 'results_text'):
            self.results_text.setText(result_text)
        else:
            self.log(result_text)
    
    def update_predictions_display(self):
        """更新预测结果显示"""
        if not self.analysis_results or 'predictions' not in self.analysis_results:
            return
            
        predictions = self.analysis_results['predictions']
        
        # 创建预测文本
        prediction_text = "===== 预测结果 =====\n\n"
        
        # 添加短期预测
        if 'short_term' in predictions:
            short_term = predictions['short_term']
            prediction_text += f"【短期预测】({short_term.get('time_frame', '1-3天')})\n"
            prediction_text += f"方向: {short_term.get('direction', 'unknown')}\n"
            prediction_text += f"置信度: {short_term.get('confidence', 0.0):.2f}\n\n"
        
        # 添加中期预测
        if 'medium_term' in predictions:
            medium_term = predictions['medium_term']
            prediction_text += f"【中期预测】({medium_term.get('time_frame', '1-2周')})\n"
            prediction_text += f"方向: {medium_term.get('direction', 'unknown')}\n"
            prediction_text += f"置信度: {medium_term.get('confidence', 0.0):.2f}\n\n"
        
        # 添加长期预测
        if 'long_term' in predictions:
            long_term = predictions['long_term']
            prediction_text += f"【长期预测】({long_term.get('time_frame', '1-3月')})\n"
            prediction_text += f"方向: {long_term.get('direction', 'unknown')}\n"
            prediction_text += f"置信度: {long_term.get('confidence', 0.0):.2f}\n\n"
        
        # 添加市场状态
        if 'market_state' in predictions:
            market_state = predictions['market_state']
            prediction_text += "【市场状态】\n"
            prediction_text += f"当前阶段: {market_state.get('current_phase', 'unknown')}\n"
            if 'next_phase' in market_state:
                prediction_text += f"下一阶段: {market_state['next_phase']}\n"
            prediction_text += "\n"
        
        # 添加临界点
        if 'critical_points' in predictions and predictions['critical_points']:
            prediction_text += "【临界点】\n"
            for i, point in enumerate(predictions['critical_points']):
                prediction_text += f"{i+1}. {point}\n"
            prediction_text += "\n"
        
        # 添加异常
        if 'anomalies' in predictions and predictions['anomalies']:
            prediction_text += "【异常】\n"
            for i, anomaly in enumerate(predictions['anomalies']):
                prediction_text += f"{i+1}. {anomaly}\n"
            prediction_text += "\n"
        
        # 设置预测文本
        if hasattr(self, 'prediction_text'):
            self.prediction_text.setText(prediction_text)
        else:
            self.log(prediction_text)
    
    def update_chart(self):
        """更新图表"""
        self.log("更新图表，此功能尚未完全实现")
        # 简化版本，仅记录日志
    
    def update_phase_space(self):
        """更新相空间图"""
        self.log("更新相空间图，此功能尚未完全实现")
        # 简化版本，仅记录日志
    
    def update_dimension_visualization(self):
        """更新维度可视化"""
        self.log("更新维度可视化，此功能尚未完全实现")
        # 简化版本，仅记录日志
    
    def save_results(self):
        """保存分析结果"""
        if self.analysis_results is None:
            self.log("没有可保存的分析结果", level="warning")
            return
            
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存分析结果", "supergod_analysis_results.json", "JSON文件 (*.json)", options=options
        )
        
        if file_path:
            try:
                import json
                
                # 转换DataFrame为可序列化格式
                results_copy = {}
                for key, value in self.analysis_results.items():
                    if isinstance(value, pd.DataFrame):
                        results_copy[key] = value.to_dict(orient='records')
                    else:
                        results_copy[key] = value
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(results_copy, f, ensure_ascii=False, indent=4)
                    
                self.log(f"分析结果已保存到 {file_path}", level="success")
                
            except Exception as e:
                error_msg = f"保存结果失败: {str(e)}"
                self.log(error_msg, level="error")
                self.show_error_dialog("保存错误", error_msg)
    
    def export_chart(self):
        """导出图表为图片"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出图表", "supergod_chart.png", "PNG图片 (*.png);;PDF文件 (*.pdf)", options=options
        )
        
        if file_path:
            try:
                # 根据当前标签页导出不同图表
                current_tab = self.tab_widget.currentIndex()
                
                if current_tab == 1:  # 技术图表标签页
                    self.chart_canvas.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                elif current_tab == 2:  # 混沌理论标签页
                    self.phase_canvas.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                elif current_tab == 3:  # 量子维度标签页
                    self.dimension_canvas.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                elif current_tab == 4:  # 预测标签页
                    self.prediction_canvas.fig.savefig(file_path, dpi=300, bbox_inches='tight')
                else:
                    raise ValueError("当前页面没有可导出的图表")
                    
                self.log(f"图表已导出到 {file_path}", level="success")
                
            except Exception as e:
                error_msg = f"导出图表失败: {str(e)}"
                self.log(error_msg, level="error")
                self.show_error_dialog("导出错误", error_msg)
    
    def execute_command(self):
        """执行控制台命令"""
        command = self.cmd_input.text().strip()
        if not command:
            return
            
        self.log(f"执行命令: {command}")
        
        # 清空输入框
        self.cmd_input.clear()
        
        # 解析并执行命令
        parts = command.split()
        cmd = parts[0].lower()
        args = parts[1:]
        
        try:
            if cmd == "help":
                self.show_command_help()
            elif cmd == "status":
                self.show_system_status()
            elif cmd == "clear":
                self.log_text.clear()
                self.log("日志已清空")
            elif cmd == "analyze":
                self.run_analysis()
            elif cmd == "load":
                if args and args[0].lower() == "demo":
                    self.data_source_combo.setCurrentIndex(0)  # Demo数据
                    self.load_data()
                else:
                    self.log("用法: load demo", level="warning")
            elif cmd == "save":
                self.save_results()
            elif cmd == "export":
                self.export_chart()
            elif cmd == "config":
                self.show_config_dialog()
            elif cmd == "exit":
                self.close()
            elif cmd == "report":
                self.generate_market_report()
            else:
                self.log(f"未知命令: {cmd}，输入 'help' 查看可用命令", level="warning")
                
        except Exception as e:
            self.log(f"命令执行错误: {str(e)}", level="error")
    
    def show_command_help(self):
        """显示命令帮助"""
        help_text = """
        可用命令:
        help            - 显示此帮助信息
        status          - 显示系统状态
        clear           - 清空日志
        analyze         - 运行分析
        load demo       - 加载演示数据
        save            - 保存分析结果
        export          - 导出当前图表
        config          - 显示配置对话框
        exit            - 退出应用
        report          - 生成市场分析报告
        """
        self.log(help_text)
    
    def show_system_status(self):
        """显示系统状态"""
        status_text = "系统状态:\n"
        
        # 数据状态
        if self.market_data is not None:
            status_text += f"数据: 已加载 ({len(self.market_data)} 条记录)\n"
        else:
            status_text += "数据: 未加载\n"
            
        # 分析状态
        if self.analysis_results is not None:
            status_text += "分析: 已完成\n"
            
            # 显示分析结果摘要
            if 'market_analysis' in self.analysis_results:
                market_analysis = self.analysis_results['market_analysis']
                if 'current_cycle' in market_analysis:
                    status_text += f"当前市场周期: {market_analysis['current_cycle']}\n"
            
            if 'predictions' in self.analysis_results:
                predictions = self.analysis_results['predictions']
                if 'short_term' in predictions and predictions['short_term']['direction']:
                    status_text += f"短期预测: {predictions['short_term']['direction']}\n"
        else:
            status_text += "分析: 未完成\n"
            
        # 组件状态
        status_text += "已加载组件:\n"
        for module_type, module in self.modules.items():
            if module_type == 'core':
                market_core, policy_analyzer, sector_tracker = module
                if market_core:
                    status_text += "- 市场核心: 已加载\n"
                if policy_analyzer:
                    status_text += "- 政策分析器: 已加载\n"
                if sector_tracker:
                    status_text += "- 板块轮动跟踪器: 已加载\n"
            elif module:
                status_text += f"- {module_type}: 已加载\n"
        
        # 配置状态
        status_text += "配置:\n"
        status_text += f"- 自动刷新: {'已启用' if self.config.get('auto_refresh') else '已禁用'}\n"
        status_text += f"- 调试模式: {'已启用' if self.config.get('debug_mode') else '已禁用'}\n"
        
        self.log(status_text)

    def log(self, message, level="info"):
        """记录日志"""
        # 记录到日志文件
        if level == "info":
            logger.info(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        elif level == "success":
            logger.info(f"[成功] {message}")
        
        # 设置日志颜色
        if level == "info":
            color = "white"
        elif level == "warning":
            color = "yellow"
        elif level == "error":
            color = "red"
        elif level == "success":
            color = "green"
        else:
            color = "white"
            
        # 获取当前时间
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # 格式化日志消息
        formatted_message = f"[{timestamp}] {message}"
        
        # 添加到日志文本框
        if hasattr(self, 'log_text'):
            # 保存当前滚动位置
            scrollbar = self.log_text.verticalScrollBar()
            at_bottom = scrollbar.value() == scrollbar.maximum()
            
            # 设置文本颜色
            self.log_text.setTextColor(QColor(color))
            
            # 追加消息
            self.log_text.append(formatted_message)
            
            # 如果之前在底部，则保持在底部
            if at_bottom:
                scrollbar.setValue(scrollbar.maximum())
                
            # 恢复默认颜色
            self.log_text.setTextColor(QColor("white"))
        else:
            print(formatted_message)
    
    def show_welcome_message(self):
        """显示欢迎信息"""
        # 在日志中显示欢迎信息
        self.log("欢迎使用超神量子共生系统 - 桌面版", level="success")
        self.log(f"系统版本: 1.0.0", level="info")
        self.log(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", level="info")
        self.log("输入'help'查看可用命令", level="info")
        
        # 显示欢迎对话框
        self.show_welcome_dialog()
    
    def show_welcome_dialog(self):
        """显示欢迎对话框"""
        # 创建欢迎对话框
        welcome_dialog = QDialog(self)
        welcome_dialog.setWindowTitle("欢迎使用超神桌面系统")
        welcome_dialog.setMinimumSize(700, 500)
        welcome_dialog.setStyleSheet(f"""
            QDialog {{
                background-color: {SupergodColors.PRIMARY_DARK};
                color: {SupergodColors.TEXT_PRIMARY};
            }}
            QLabel {{
                color: {SupergodColors.TEXT_PRIMARY};
            }}
            QTextBrowser {{
                background-color: {SupergodColors.SECONDARY_DARK};
                color: {SupergodColors.TEXT_PRIMARY};
                border: 1px solid {SupergodColors.ACCENT_DARK};
                border-radius: 5px;
                padding: 10px;
            }}
            QPushButton {{
                background-color: {SupergodColors.ACCENT_DARK};
                color: {SupergodColors.TEXT_PRIMARY};
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
                min-width: 100px;
            }}
            QPushButton:hover {{
                background-color: {SupergodColors.HIGHLIGHT};
            }}
        """)
        
        # 创建布局
        layout = QVBoxLayout(welcome_dialog)
        
        # 添加标题
        title_label = QLabel("超神量子共生系统", welcome_dialog)
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(24)
        title_font.setBold(True)
        title_label.setFont(title_font)
        layout.addWidget(title_label)
        
        # 添加副标题
        subtitle_label = QLabel("高级量子金融分析平台", welcome_dialog)
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_font = QFont()
        subtitle_font.setPointSize(14)
        subtitle_label.setFont(subtitle_font)
        layout.addWidget(subtitle_label)
        
        # 添加分隔线
        separator = QFrame(welcome_dialog)
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet(f"background-color: {SupergodColors.HIGHLIGHT};")
        layout.addWidget(separator)
        
        # 添加欢迎文本
        welcome_text = QTextBrowser(welcome_dialog)
        welcome_text.setOpenExternalLinks(True)
        welcome_text.setHtml(f"""
            <h3>欢迎使用超神桌面系统</h3>
            <p>您现在正在使用的是超神量子共生系统的桌面版本，融合了量子计算、人工智能和混沌理论的先进金融分析平台。</p>
            
            <h4>系统特点：</h4>
            <ul>
                <li><b>量子维度分析</b> - 通过扩展到21个维度的量子共生框架，深入洞察市场本质</li>
                <li><b>混沌理论分析</b> - 识别市场非线性特征、吸引子和分形模式</li>
                <li><b>市场周期识别</b> - 精确定位市场所处的周期阶段和转换点</li>
                <li><b>板块轮动追踪</b> - 分析板块强弱变化和轮动规律</li>
                <li><b>政策环境分析</b> - 评估宏观政策方向和影响</li>
            </ul>
            
            <h4>开始使用：</h4>
            <ol>
                <li>点击 <b>"文件 > 打开..."</b> 加载市场数据，或使用演示数据</li>
                <li>点击 <b>"分析 > 运行分析"</b> 开始量子混沌分析</li>
                <li>在不同标签页查看分析结果、图表和预测</li>
                <li>使用 <b>"文件 > 生成分析报告..."</b> 创建综合报告</li>
            </ol>
            
            <p>高级用户可以使用命令行标签页进行更复杂的操作，输入 <code>help</code> 查看可用命令。</p>
            
            <p style="color: {SupergodColors.HIGHLIGHT};">注意：本系统仅用于学术研究和技术验证，分析结果不构成投资建议。</p>
        """)
        layout.addWidget(welcome_text)
        
        # 添加"不再显示"复选框
        show_again_check = QCheckBox("下次启动时不再显示此对话框", welcome_dialog)
        show_again_check.setStyleSheet(f"color: {SupergodColors.TEXT_SECONDARY};")
        layout.addWidget(show_again_check)
        
        # 添加确定按钮
        button_layout = QHBoxLayout()
        ok_button = QPushButton("开始探索量子市场", welcome_dialog)
        ok_button.clicked.connect(welcome_dialog.accept)
        button_layout.addStretch()
        button_layout.addWidget(ok_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # 设置布局
        welcome_dialog.setLayout(layout)
        
        # 显示对话框
        result = welcome_dialog.exec_()
        
        # 保存"不再显示"设置
        if result == QDialog.Accepted and show_again_check.isChecked():
            self.update_config('show_welcome', False)
    
    def show_error_dialog(self, title, message):
        """显示错误对话框"""
        QMessageBox.critical(self, title, message)
    
    def show_info_dialog(self, title, message):
        """显示信息对话框"""
        QMessageBox.information(self, title, message)
    
    def open_data_file(self):
        """打开数据文件"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开数据文件", "", "CSV文件 (*.csv);;所有文件 (*)", options=options
        )
        
        if file_path:
            self.log(f"选择数据文件: {file_path}")
            # 设置数据源为CSV
            self.data_source_combo.setCurrentIndex(1)  # CSV选项
            # 加载数据
            self.load_data(file_path=file_path)
    
    def load_data(self, file_path=None):
        """加载数据"""
        # 获取数据源类型
        source_index = self.data_source_combo.currentIndex()
        source_types = ['demo', 'csv', 'api']
        source_type = source_types[source_index]
        
        # 准备参数
        params = {}
        if source_type == 'csv' and file_path:
            params['file_path'] = file_path
        elif source_type == 'api':
            # 这里可以添加API参数，例如从UI中获取
            params['symbol'] = '000001.SH'  # 默认指数
            params['start_date'] = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            params['end_date'] = datetime.now().strftime('%Y-%m-%d')
        
        # 启动加载线程
        self.progress_bar.show()
        self.loader_thread = DataLoaderThread(source_type, params)
        self.loader_thread.loading_progress.connect(self.update_loading_progress)
        self.loader_thread.data_loaded.connect(self.on_data_loaded)
        self.loader_thread.loading_error.connect(self.on_loading_error)
        self.loader_thread.start()
        
        self.log(f"开始加载数据 (源: {source_type})")
    
    def update_loading_progress(self, value, message):
        """更新加载进度"""
        self.progress_bar.setValue(value)
        self.statusBar().showMessage(message)
    
    def update_market_table(self):
        """更新市场数据表"""
        if self.market_data is None:
            return
            
        # 清空表格
        self.market_table.setRowCount(0)
        
        # 添加数据
        data = self.market_data.reset_index()  # 将索引转换为列
        
        self.market_table.setRowCount(len(data))
        for i, row in data.iterrows():
            # 限制显示的行数，避免UI卡顿
            if i >= 100:  # 最多显示100行
                break
                
            # 添加数据到表格
            date_item = QTableWidgetItem(str(row['date'].strftime('%Y-%m-%d')))
            open_item = QTableWidgetItem(f"{row['open']:.2f}")
            high_item = QTableWidgetItem(f"{row['high']:.2f}")
            low_item = QTableWidgetItem(f"{row['low']:.2f}")
            close_item = QTableWidgetItem(f"{row['close']:.2f}")
            volume_item = QTableWidgetItem(f"{row['volume']:,.0f}")
            
            # 如果有换手率列，则显示，否则显示N/A
            if 'turnover_rate' in row:
                turnover_item = QTableWidgetItem(f"{row['turnover_rate']:.2f}%")
            else:
                turnover_item = QTableWidgetItem("N/A")
            
            self.market_table.setItem(i, 0, date_item)
            self.market_table.setItem(i, 1, open_item)
            self.market_table.setItem(i, 2, high_item)
            self.market_table.setItem(i, 3, low_item)
            self.market_table.setItem(i, 4, close_item)
            self.market_table.setItem(i, 5, volume_item)
            self.market_table.setItem(i, 6, turnover_item)
        
        self.log(f"市场数据表已更新，显示 {min(len(data), 100)} 条记录")
    
    def run_analysis(self):
        """运行分析"""
        if self.market_data is None:
            self.log("没有数据可分析，请先加载数据", level="warning")
            return
        
        # 直接使用默认的分析类型，不再依赖下拉框
        analysis_type = "全面分析"
        self.log(f"开始执行{analysis_type}...")
        
        # 显示进度条
        self.progress_bar.show()
        self.progress_bar.setValue(0)
        
        # 创建分析线程
        self.analysis_thread = AnalysisThread(self.market_data, self.modules, self.config)
        self.analysis_thread.analysis_progress.connect(self.update_analysis_progress)
        self.analysis_thread.analysis_complete.connect(self.on_analysis_complete)
        self.analysis_thread.analysis_error.connect(self.on_analysis_error)
        self.analysis_thread.start()
    
    def update_market_analysis_display(self):
        """更新市场分析结果显示"""
        market_analysis = self.analysis_results.get('market_analysis', {})
        
        # 更新市场周期
        if 'current_cycle' in market_analysis:
            cycle = market_analysis['current_cycle']
            self.market_cycle_label.setText(f"当前周期: {cycle}")
            
            # 设置周期置信度
            confidence = market_analysis.get('cycle_confidence', 0.0)
            self.cycle_confidence_label.setText(f"周期置信度: {confidence:.2f}")
        
        # 更新市场情绪
        if 'market_sentiment' in market_analysis:
            sentiment = market_analysis['market_sentiment']
            self.sentiment_label.setText(f"当前情绪: {sentiment:.2f}")
            
            # 更新情绪进度条
            sentiment_value = int(sentiment * 100)
            self.sentiment_bar.setValue(sentiment_value)
            
            # 设置情绪进度条颜色
            if sentiment > 0:
                self.sentiment_bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {SupergodColors.POSITIVE}; }}")
            else:
                self.sentiment_bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {SupergodColors.NEGATIVE}; }}")
        
        # 更新异常检测
        if 'anomalies' in market_analysis:
            anomalies = market_analysis['anomalies']
            if anomalies:
                anomalies_text = "检测到以下市场异常:\n\n"
                for i, anomaly in enumerate(anomalies):
                    anomaly_type = anomaly.get('type', 'unknown')
                    anomaly_pos = anomaly.get('position', 'unknown')
                    anomaly_conf = anomaly.get('confidence', 0.0)
                    anomalies_text += f"{i+1}. 类型: {anomaly_type}, 位置: {anomaly_pos}, 置信度: {anomaly_conf:.2f}\n"
            else:
                anomalies_text = "未检测到市场异常"
                
            self.anomalies_text.setText(anomalies_text)
    
    def update_chaos_analysis_display(self):
        """更新混沌理论分析结果显示"""
        chaos_analysis = self.analysis_results.get('chaos_analysis', {})
        
        # 更新市场状态
        if 'market_regime' in chaos_analysis:
            regime = chaos_analysis['market_regime']
            self.market_regime_label.setText(f"混沌状态: {regime}")
            
            # 设置稳定性
            stability = chaos_analysis.get('stability', 0.0)
            self.stability_label.setText(f"稳定性: {stability:.2f}")
        
        # 更新混沌指标
        if 'hurst_exponent' in chaos_analysis:
            hurst = chaos_analysis['hurst_exponent']
            self.hurst_label.setText(f"赫斯特指数: {hurst:.3f}")
        
        if 'fractal_dimension' in chaos_analysis:
            fractal = chaos_analysis['fractal_dimension']
            self.fractal_label.setText(f"分形维度: {fractal:.3f}")
        
        if 'lyapunov_exponent' in chaos_analysis:
            lyapunov = chaos_analysis['lyapunov_exponent']
            self.lyapunov_label.setText(f"莱雅普诺夫指数: {lyapunov:.6f}")
        
        if 'entropy' in chaos_analysis:
            entropy = chaos_analysis['entropy']
            self.entropy_label.setText(f"熵值: {entropy:.3f}")
        
        # 更新临界点检测
        if 'critical_points' in chaos_analysis:
            critical_points = chaos_analysis['critical_points']
            if critical_points:
                critical_text = "检测到以下临界点:\n\n"
                for i, (point, score) in enumerate(critical_points):
                    critical_text += f"{i+1}. 位置: {point}, 临界分数: {score:.2f}\n"
            else:
                critical_text = "未检测到临界点"
                
            self.critical_points_text.setText(critical_text)
        
        # 更新相空间图
        self.update_phase_space()
    
    def update_quantum_dimensions_display(self):
        """更新量子维度分析结果显示"""
        quantum_dimensions = self.analysis_results.get('quantum_dimensions', {})
        
        if 'state' in quantum_dimensions:
            dimension_state = quantum_dimensions['state']
            
            # 更新维度表格
            self.dimensions_table.setRowCount(0)
            
            # 添加基础维度
            base_dims = {k: v for k, v in dimension_state.items() if v.get('type') == 'base'}
            for i, (dim_name, dim_state) in enumerate(base_dims.items()):
                self.dimensions_table.insertRow(i)
                
                # 设置表格项
                name_item = QTableWidgetItem(dim_name)
                value_item = QTableWidgetItem(f"{dim_state['value']:.3f}")
                
                # 设置趋势箭头
                trend = dim_state.get('trend', 0)
                if trend > 0.01:
                    trend_text = "↗"
                elif trend < -0.01:
                    trend_text = "↘"
                else:
                    trend_text = "→"
                trend_item = QTableWidgetItem(trend_text)
                
                weight_item = QTableWidgetItem(f"{dim_state.get('weight', 0):.3f}")
                
                self.dimensions_table.setItem(i, 0, name_item)
                self.dimensions_table.setItem(i, 1, value_item)
                self.dimensions_table.setItem(i, 2, trend_item)
                self.dimensions_table.setItem(i, 3, weight_item)
            
            # 添加扩展维度
            ext_dims = {k: v for k, v in dimension_state.items() if v.get('type') == 'extended'}
            start_row = len(base_dims)
            for i, (dim_name, dim_state) in enumerate(ext_dims.items()):
                row = start_row + i
                self.dimensions_table.insertRow(row)
                
                # 设置表格项
                name_item = QTableWidgetItem(dim_name)
                value_item = QTableWidgetItem(f"{dim_state['value']:.3f}")
                
                # 设置趋势箭头
                trend = dim_state.get('trend', 0)
                if trend > 0.01:
                    trend_text = "↗"
                elif trend < -0.01:
                    trend_text = "↘"
                else:
                    trend_text = "→"
                trend_item = QTableWidgetItem(trend_text)
                
                weight_item = QTableWidgetItem(f"{dim_state.get('weight', 0):.3f}")
                
                self.dimensions_table.setItem(row, 0, name_item)
                self.dimensions_table.setItem(row, 1, value_item)
                self.dimensions_table.setItem(row, 2, trend_item)
                self.dimensions_table.setItem(row, 3, weight_item)
            
            # 更新综合指标
            if 'energy_potential' in dimension_state:
                energy = dimension_state['energy_potential']['value']
                self.energy_label.setText(f"能量势能: {energy:.3f}")
            
            if 'phase_coherence' in dimension_state:
                coherence = dimension_state['phase_coherence']['value']
                self.coherence_label.setText(f"相位相干性: {coherence:.3f}")
            
            if 'temporal_coherence' in dimension_state:
                temporal = dimension_state['temporal_coherence']['value']
                self.temporal_label.setText(f"时间相干性: {temporal:.3f}")
            
            if 'chaos_degree' in dimension_state:
                chaos = dimension_state['chaos_degree']['value']
                self.chaos_degree_label.setText(f"混沌度: {chaos:.3f}")
            
            # 更新维度可视化
            self.update_dimension_visualization()
    
    def show_config_dialog(self):
        """显示配置对话框"""
        # 这里可以实现配置对话框
        self.log("显示配置对话框")
    
    def show_about_dialog(self):
        """显示关于对话框"""
        about_text = """
        超神量子共生系统 - 桌面版
        
        高级量子金融分析平台
        
        版本: 1.0.0
        
        © 2025 超神科技
        """
        
        QMessageBox.about(self, "关于超神桌面系统", about_text)
    
    def show_help(self):
        """显示帮助文档"""
        self.log("显示帮助文档")
    
    def toggle_auto_refresh(self, enabled):
        """切换自动刷新"""
        self.config['auto_refresh'] = enabled
        
        if enabled:
            interval = self.config.get('refresh_interval', 60) * 1000  # 转换为毫秒
            self.refresh_timer.start(interval)
            self.log(f"自动刷新已启用，间隔：{interval//1000}秒")
        else:
            self.refresh_timer.stop()
            self.log("自动刷新已禁用")
        
        self._save_config()
    
    def toggle_debug_mode(self, enabled):
        """切换调试模式"""
        self.config['debug_mode'] = enabled
        
        if enabled:
            logging.getLogger().setLevel(logging.DEBUG)
            self.log("调试模式已启用", level="info")
        else:
            logging.getLogger().setLevel(logging.INFO)
            self.log("调试模式已禁用", level="info")
        
        self._save_config()

    def _load_config(self):
        """加载系统配置"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    import json
                    loaded_config = json.load(f)
                    # 更新配置，但保留默认值作为后备
                    for key, value in loaded_config.items():
                        self.config[key] = value
                self.log(f"已加载系统配置: {self.config_file}", level="info")
            else:
                self.log(f"未找到配置文件，使用默认配置", level="info")
                self._save_config()  # 保存默认配置
        except Exception as e:
            self.log(f"加载配置失败: {str(e)}", level="error")
    
    def _save_config(self):
        """保存系统配置"""
        try:
            with open(self.config_file, 'w') as f:
                import json
                json.dump(self.config, f, indent=4)
            self.log(f"已保存系统配置: {self.config_file}", level="info")
        except Exception as e:
            self.log(f"保存配置失败: {str(e)}", level="error")
    
    def update_config(self, key, value):
        """更新配置项"""
        self.config[key] = value
        self._save_config()
        self.log(f"已更新配置: {key} = {value}", level="info")

    def generate_market_report(self):
        """生成综合市场分析报告"""
        if not self.analysis_results:
            self.show_error_dialog("错误", "没有分析结果可用，请先运行分析")
            return
        
        # 获取文件保存路径
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存市场分析报告", 
            os.path.join(os.getcwd(), "超神市场分析报告.html"),
            "HTML文件 (*.html)")
            
        if not file_path:
            return
            
        try:
            # 创建报告
            report_html = self._generate_report_html()
            
            # 保存到文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(report_html)
                
            self.log(f"已生成市场分析报告: {file_path}", level="success")
            
            # 询问是否打开报告
            reply = QMessageBox.question(
                self, "报告已生成", 
                "市场分析报告已成功生成。是否立即打开?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                
            if reply == QMessageBox.Yes:
                import webbrowser
                webbrowser.open(f"file://{file_path}")
                
        except Exception as e:
            self.log(f"生成报告失败: {str(e)}", level="error")
            self.show_error_dialog("错误", f"生成报告失败: {str(e)}")
    
    def _generate_report_html(self):
        """生成HTML格式的报告内容"""
        # 获取当前时间
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 获取各分析结果
        market_analysis = self.analysis_results.get('market_analysis', {})
        policy_analysis = self.analysis_results.get('policy_analysis', {})
        sector_analysis = self.analysis_results.get('sector_analysis', {})
        chaos_analysis = self.analysis_results.get('chaos_analysis', {})
        quantum_dimensions = self.analysis_results.get('quantum_dimensions', {})
        predictions = self.analysis_results.get('predictions', {})
        
        # 准备HTML报告
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>超神量子共生系统市场分析报告</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 0;
                    color: #333;
                    background-color: #f8f9fa;
                }}
                .container {{
                    width: 90%;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    background-color: #1a1a2e;
                    color: white;
                    padding: 20px;
                    text-align: center;
                    border-radius: 5px 5px 0 0;
                }}
                .section {{
                    background-color: white;
                    margin: 20px 0;
                    padding: 20px;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .section h2 {{
                    color: #1a1a2e;
                    border-bottom: 2px solid #e94560;
                    padding-bottom: 10px;
                }}
                .data-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 15px 0;
                }}
                .data-table th, .data-table td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                .data-table th {{
                    background-color: #1a1a2e;
                    color: white;
                }}
                .data-table tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
                .footer {{
                    background-color: #1a1a2e;
                    color: white;
                    text-align: center;
                    padding: 10px;
                    border-radius: 0 0 5px 5px;
                    font-size: 0.8em;
                }}
                .metric {{
                    display: inline-block;
                    margin: 10px;
                    min-width: 200px;
                }}
                .metric .value {{
                    font-size: 2em;
                    font-weight: bold;
                    color: #1a1a2e;
                }}
                .positive {{
                    color: green;
                }}
                .negative {{
                    color: red;
                }}
                .neutral {{
                    color: #7c83fd;
                }}
                .warning {{
                    background-color: #fff3cd;
                    border-left: 4px solid #ffc107;
                    padding: 10px;
                    margin: 10px 0;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>超神量子共生系统市场分析报告</h1>
                    <p>生成时间: {current_time}</p>
                </div>
                
                <div class="section">
                    <h2>市场概况</h2>
                    <div class="metrics">
                        <div class="metric">
                            <div class="label">当前周期</div>
                            <div class="value">{market_analysis.get('current_cycle', 'N/A')}</div>
                        </div>
                        <div class="metric">
                            <div class="label">周期置信度</div>
                            <div class="value">{market_analysis.get('cycle_confidence', 0):.2f}</div>
                        </div>
                        <div class="metric">
                            <div class="label">市场情绪</div>
                            <div class="value {self._get_sentiment_class(market_analysis.get('market_sentiment', 0))}">
                                {market_analysis.get('market_sentiment', 0):.2f}
                            </div>
                        </div>
                    </div>
                    
                    <h3>市场异常</h3>
                    <div>
        """
        
        # 添加异常检测
        anomalies = market_analysis.get('anomalies', [])
        if anomalies:
            html += "<ul>"
            for anomaly in anomalies:
                html += f"<li><b>{anomaly.get('type', 'unknown')}</b> - 位置: {anomaly.get('position', 'unknown')}, 置信度: {anomaly.get('confidence', 0):.2f}</li>"
            html += "</ul>"
        else:
            html += "<p>未检测到市场异常</p>"
            
        html += """
                    </div>
                </div>
                
                <div class="section">
                    <h2>政策环境分析</h2>
        """
        
        # 添加政策分析
        if policy_analysis:
            html += f"""
                    <div class="metrics">
                        <div class="metric">
                            <div class="label">政策方向</div>
                            <div class="value {self._get_policy_direction_class(policy_analysis.get('policy_direction', 0))}">
                                {policy_analysis.get('policy_direction', 0):.2f}
                            </div>
                        </div>
                        <div class="metric">
                            <div class="label">政策不确定性</div>
                            <div class="value">
                                {policy_analysis.get('policy_uncertainty', 0):.2f}
                            </div>
                        </div>
                    </div>
                    
                    <h3>最新政策动态</h3>
            """
            
            # 添加政策动态
            policy_news = policy_analysis.get('policy_news', [])
            if policy_news:
                html += "<ul>"
                for news in policy_news[:5]:  # 只显示前5条
                    html += f"<li><b>{news.get('date', '')}</b> - {news.get('title', '')}</li>"
                html += "</ul>"
            else:
                html += "<p>无政策动态</p>"
        else:
            html += "<p>无政策分析数据</p>"
            
        html += """
                </div>
                
                <div class="section">
                    <h2>板块轮动分析</h2>
        """
        
        # 添加板块轮动分析
        if sector_analysis:
            html += """
                    <h3>领先板块</h3>
                    <table class="data-table">
                        <tr>
                            <th>板块</th>
                            <th>相对强度</th>
                            <th>趋势</th>
                            <th>热度</th>
                        </tr>
            """
            
            # 添加领先板块
            leading_sectors = sector_analysis.get('leading_sectors', [])
            for sector in leading_sectors:
                sector_name = sector.get('name', '')
                strength = sector.get('relative_strength', 0)
                trend = sector.get('trend', 0)
                heat = sector.get('heat', 0)
                
                trend_class = 'positive' if trend > 0 else 'negative'
                trend_symbol = '↗' if trend > 0 else '↘'
                
                html += f"""
                        <tr>
                            <td>{sector_name}</td>
                            <td>{strength:.2f}</td>
                            <td class="{trend_class}">{trend:.2f} {trend_symbol}</td>
                            <td>{heat:.2f}</td>
                        </tr>
                """
            
            html += """
                    </table>
                    
                    <h3>相对强度热力图</h3>
                    <p>此处应有热力图，已保存为单独图像</p>
            """
        else:
            html += "<p>无板块轮动分析数据</p>"
            
        html += """
                </div>
                
                <div class="section">
                    <h2>混沌理论分析</h2>
        """
        
        # 添加混沌理论分析
        if chaos_analysis:
            html += f"""
                    <div class="metrics">
                        <div class="metric">
                            <div class="label">市场状态</div>
                            <div class="value">
                                {chaos_analysis.get('market_regime', 'unknown')}
                            </div>
                        </div>
                        <div class="metric">
                            <div class="label">稳定性</div>
                            <div class="value">
                                {chaos_analysis.get('stability', 0):.2f}
                            </div>
                        </div>
                        <div class="metric">
                            <div class="label">赫斯特指数</div>
                            <div class="value">
                                {chaos_analysis.get('hurst_exponent', 0):.3f}
                            </div>
                        </div>
                        <div class="metric">
                            <div class="label">莱雅普诺夫指数</div>
                            <div class="value {self._get_lyapunov_class(chaos_analysis.get('lyapunov_exponent', 0))}">
                                {chaos_analysis.get('lyapunov_exponent', 0):.6f}
                            </div>
                        </div>
                        <div class="metric">
                            <div class="label">分形维度</div>
                            <div class="value">
                                {chaos_analysis.get('fractal_dimension', 0):.3f}
                            </div>
                        </div>
                        <div class="metric">
                            <div class="label">熵值</div>
                            <div class="value">
                                {chaos_analysis.get('entropy', 0):.3f}
                            </div>
                        </div>
                    </div>
                    
                    <h3>临界点检测</h3>
            """
            
            # 添加临界点
            critical_points = chaos_analysis.get('critical_points', [])
            if critical_points:
                html += "<ul>"
                for point, score in critical_points:
                    html += f"<li>位置: <b>{point}</b>, 临界分数: <b>{score:.2f}</b></li>"
                html += "</ul>"
                
                html += """
                        <div class="warning">
                            <strong>注意：</strong> 检测到市场临界点，可能预示着重要的市场转折点。
                        </div>
                """
            else:
                html += "<p>未检测到临界点</p>"
        else:
            html += "<p>无混沌理论分析数据</p>"
            
        html += """
                </div>
                
                <div class="section">
                    <h2>量子维度分析</h2>
        """
        
        # 添加量子维度分析
        if quantum_dimensions and 'state' in quantum_dimensions:
            dimension_state = quantum_dimensions['state']
            
            html += """
                    <h3>维度状态</h3>
                    <table class="data-table">
                        <tr>
                            <th>维度</th>
                            <th>当前值</th>
                            <th>趋势</th>
                            <th>权重</th>
                        </tr>
            """
            
            # 添加维度状态
            for dim_name, dim_state in dimension_state.items():
                dim_type = dim_state.get('type', '')
                if dim_type in ['base', 'extended']:
                    value = dim_state.get('value', 0)
                    trend = dim_state.get('trend', 0)
                    weight = dim_state.get('weight', 0)
                    
                    trend_class = 'positive' if trend > 0.01 else ('negative' if trend < -0.01 else 'neutral')
                    trend_symbol = '↗' if trend > 0.01 else ('↘' if trend < -0.01 else '→')
                    
                    html += f"""
                            <tr>
                                <td>{dim_name}</td>
                                <td>{value:.3f}</td>
                                <td class="{trend_class}">{trend:.3f} {trend_symbol}</td>
                                <td>{weight:.3f}</td>
                            </tr>
                    """
            
            html += """
                    </table>
                    
                    <h3>关键综合指标</h3>
            """
            
            # 添加关键综合指标
            key_indicators = ['energy_potential', 'phase_coherence', 'temporal_coherence', 'chaos_degree']
            html += "<div class='metrics'>"
            
            for indicator in key_indicators:
                if indicator in dimension_state:
                    value = dimension_state[indicator]['value']
                    html += f"""
                            <div class="metric">
                                <div class="label">{self._get_dimension_name(indicator)}</div>
                                <div class="value">
                                    {value:.3f}
                                </div>
                            </div>
                    """
            
            html += "</div>"
        else:
            html += "<p>无量子维度分析数据</p>"
            
        html += """
                </div>
                
                <div class="section">
                    <h2>市场预测</h2>
        """
        
        # 添加市场预测
        if predictions:
            # 短期预测
            if 'short_term' in predictions:
                short_term = predictions['short_term']
                direction = short_term.get('direction', 'unknown')
                confidence = short_term.get('confidence', 0)
                time_frame = short_term.get('time_frame', '1-3天')
                
                direction_class = self._get_direction_class(direction)
                direction_text = self._get_direction_text(direction)
                
                html += f"""
                    <h3>短期预测 ({time_frame})</h3>
                    <div class="metrics">
                        <div class="metric">
                            <div class="label">方向</div>
                            <div class="value {direction_class}">
                                {direction_text}
                            </div>
                        </div>
                        <div class="metric">
                            <div class="label">置信度</div>
                            <div class="value">
                                {confidence:.2f}
                            </div>
                        </div>
                    </div>
                """
            
            # 中期预测
            if 'medium_term' in predictions:
                medium_term = predictions['medium_term']
                direction = medium_term.get('direction', 'unknown')
                confidence = medium_term.get('confidence', 0)
                time_frame = medium_term.get('time_frame', '1-2周')
                
                direction_class = self._get_direction_class(direction)
                direction_text = self._get_direction_text(direction)
                
                html += f"""
                    <h3>中期预测 ({time_frame})</h3>
                    <div class="metrics">
                        <div class="metric">
                            <div class="label">方向</div>
                            <div class="value {direction_class}">
                                {direction_text}
                            </div>
                        </div>
                        <div class="metric">
                            <div class="label">置信度</div>
                            <div class="value">
                                {confidence:.2f}
                            </div>
                        </div>
                    </div>
                """
            
            # 长期预测
            if 'long_term' in predictions:
                long_term = predictions['long_term']
                direction = long_term.get('direction', 'unknown')
                confidence = long_term.get('confidence', 0)
                time_frame = long_term.get('time_frame', '1-3月')
                
                direction_class = self._get_direction_class(direction)
                direction_text = self._get_direction_text(direction)
                
                html += f"""
                    <h3>长期预测 ({time_frame})</h3>
                    <div class="metrics">
                        <div class="metric">
                            <div class="label">方向</div>
                            <div class="value {direction_class}">
                                {direction_text}
                            </div>
                        </div>
                        <div class="metric">
                            <div class="label">置信度</div>
                            <div class="value">
                                {confidence:.2f}
                            </div>
                        </div>
                    </div>
                """
            
            # 市场状态转换
            if 'market_state' in predictions:
                market_state = predictions['market_state']
                current_phase = market_state.get('current_phase', 'unknown')
                next_phase = market_state.get('next_phase', '不变')
                transition_prob = market_state.get('transition_probability', 0)
                
                # 预计时间（这里简单估计）
                time_estimate = f"预计 {int(30/transition_prob)} 天" if transition_prob > 0 else "未知"
                
                html += f"""
                    <h3>市场状态转换</h3>
                    <div class="metrics">
                        <div class="metric">
                            <div class="label">当前阶段</div>
                            <div class="value">
                                {current_phase}
                            </div>
                        </div>
                        <div class="metric">
                            <div class="label">下一阶段</div>
                            <div class="value">
                                {next_phase}
                            </div>
                        </div>
                        <div class="metric">
                            <div class="label">转换概率</div>
                            <div class="value">
                                {transition_prob:.2f}
                            </div>
                        </div>
                        <div class="metric">
                            <div class="label">预计时间</div>
                            <div class="value">
                                {time_estimate}
                            </div>
                        </div>
                    </div>
                """
        else:
            html += "<p>无预测数据</p>"
            
        html += """
                </div>
                
                <div class="footer">
                    <p>© 2025 超神科技 - 超神量子共生系统</p>
                    <p>本报告由超神桌面系统自动生成，仅供参考，不构成投资建议</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _get_sentiment_class(self, sentiment):
        """获取情绪对应的CSS类"""
        if sentiment > 0.2:
            return "positive"
        elif sentiment < -0.2:
            return "negative"
        else:
            return "neutral"
    
    def _get_policy_direction_class(self, direction):
        """获取政策方向对应的CSS类"""
        if direction > 0.2:
            return "positive"
        elif direction < -0.2:
            return "negative"
        else:
            return "neutral"
    
    def _get_lyapunov_class(self, lyapunov):
        """获取莱雅普诺夫指数对应的CSS类"""
        if lyapunov > 0:
            return "negative"  # 正值表示混沌，通常是负面的
        else:
            return "positive"  # 负值表示稳定，通常是正面的
    
    def _get_direction_class(self, direction):
        """获取预测方向对应的CSS类"""
        if direction == 'bullish':
            return "positive"
        elif direction == 'bearish':
            return "negative"
        else:
            return "neutral"
    
    def _get_direction_text(self, direction):
        """获取预测方向对应的文本"""
        if direction == 'bullish':
            return "上涨"
        elif direction == 'bearish':
            return "下跌"
        elif direction == 'sideways':
            return "盘整"
        elif direction == 'volatile':
            return "波动"
        else:
            return "未知"
    
    def _get_dimension_name(self, dim_key):
        """获取维度的友好名称"""
        name_map = {
            'energy_potential': '能量势能',
            'phase_coherence': '相位相干性',
            'temporal_coherence': '时间相干性',
            'chaos_degree': '混沌度',
            'fractal': '分形维度',
            'entropy': '熵维度',
            'cycle_resonance': '周期共振',
            'market_depth': '市场深度',
            'volume_pressure': '成交量压力',
            'price_momentum': '价格动量',
            'sentiment': '情绪维度'
        }
        return name_map.get(dim_key, dim_key)

# 主程序入口
def main():
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
    
    # 异常处理装饰器
    def handle_exception(exc_type, exc_value, exc_traceback):
        logger.error("未捕获的异常", exc_info=(exc_type, exc_value, exc_traceback))
        print("发生错误，详情请查看日志文件")
    
    # 设置异常处理器
    sys.excepthook = handle_exception
    
    # 创建应用
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # 使用Fusion风格，以便更好地适配自定义样式
    
    # 创建并显示主窗口
    main_window = SupergodDesktop()
    main_window.show()
    
    # 运行应用
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 