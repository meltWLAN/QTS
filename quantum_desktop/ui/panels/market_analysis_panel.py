"""
市场分析面板 - 展示市场数据分析
"""

import logging
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                         QPushButton, QComboBox, QFrame, QGridLayout, 
                         QTableWidget, QTableWidgetItem, QHeaderView,
                         QSplitter, QTabWidget, QLineEdit)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QColor, QBrush, QPen, QPainter
from PyQt5.QtChart import (QChart, QChartView, QLineSeries, QCandlestickSeries,
                       QCandlestickSet, QValueAxis, QDateTimeAxis)
from datetime import datetime, timedelta

logger = logging.getLogger("QuantumDesktop.MarketAnalysisPanel")

class CandlestickChart(QChartView):
    """K线图表 - 显示股票价格数据"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 创建图表
        self.chart = QChart()
        self.chart.setTitle("市场价格走势")
        self.chart.setAnimationOptions(QChart.SeriesAnimations)
        
        # 设置抗锯齿
        self.setRenderHint(QPainter.Antialiasing)
        
        # 设置图表
        self.setChart(self.chart)
        
        # 创建蜡烛图系列
        self.candle_series = QCandlestickSeries()
        self.candle_series.setIncreasingColor(QColor(46, 204, 113))  # 上涨为绿色
        self.candle_series.setDecreasingColor(QColor(231, 76, 60))   # 下跌为红色
        
        # 添加系列到图表
        self.chart.addSeries(self.candle_series)
        
        # 创建时间轴
        self.axis_x = QDateTimeAxis()
        self.axis_x.setFormat("MM-dd")
        self.axis_x.setTitleText("日期")
        
        # 创建价格轴
        self.axis_y = QValueAxis()
        self.axis_y.setTitleText("价格")
        
        # 添加轴到图表
        self.chart.addAxis(self.axis_x, Qt.AlignBottom)
        self.chart.addAxis(self.axis_y, Qt.AlignLeft)
        
        # 将系列附加到轴
        self.candle_series.attachAxis(self.axis_x)
        self.candle_series.attachAxis(self.axis_y)
        
    def set_data(self, market_data):
        """设置市场数据"""
        # 清除旧数据
        self.candle_series.clear()
        
        if not market_data or 'dates' not in market_data:
            return
            
        # 添加蜡烛图数据
        min_price = float('inf')
        max_price = float('-inf')
        min_time = None
        max_time = None
        
        dates = market_data.get('dates', [])
        opens = market_data.get('open', [])
        highs = market_data.get('high', [])
        lows = market_data.get('low', [])
        closes = market_data.get('close', [])
        
        for i in range(len(dates)):
            try:
                date_obj = datetime.strptime(dates[i], '%Y-%m-%d')
                timestamp = date_obj.timestamp() * 1000  # 毫秒时间戳
                
                open_price = opens[i]
                high_price = highs[i]
                low_price = lows[i]
                close_price = closes[i]
                
                # 创建蜡烛图集
                candle_set = QCandlestickSet(open_price, high_price, low_price, close_price, timestamp)
                self.candle_series.append(candle_set)
                
                # 更新价格范围
                min_price = min(min_price, low_price)
                max_price = max(max_price, high_price)
                
                # 更新时间范围
                if min_time is None or date_obj < min_time:
                    min_time = date_obj
                if max_time is None or date_obj > max_time:
                    max_time = date_obj
                    
            except Exception as e:
                logger.error(f"添加蜡烛图数据时出错: {str(e)}")
                
        # 设置轴范围
        if min_time and max_time:
            self.axis_x.setRange(min_time, max_time + timedelta(days=1))
            
        if min_price < max_price:
            price_range = max_price - min_price
            self.axis_y.setRange(min_price - price_range * 0.05, max_price + price_range * 0.05)

class MarketAnalysisPanel(QWidget):
    """市场分析面板 - 分析市场数据"""
    
    def __init__(self, system_manager, parent=None):
        super().__init__(parent)
        self.system_manager = system_manager
        
        # 市场数据
        self.market_data = {}
        self.analysis_results = {}
        self.current_symbol = None
        
        # 初始化UI
        self._init_ui()
        
        logger.info("市场分析面板初始化完成")
        
    def _init_ui(self):
        """初始化用户界面"""
        # 主布局
        self.main_layout = QVBoxLayout(self)
        
        # 顶部工具栏
        self._create_toolbar()
        
        # 分割器
        self.splitter = QSplitter(Qt.Vertical)
        self.main_layout.addWidget(self.splitter)
        
        # 图表区域
        self.chart_frame = QFrame()
        self.chart_layout = QVBoxLayout(self.chart_frame)
        
        # K线图
        self.candle_chart = CandlestickChart()
        self.chart_layout.addWidget(self.candle_chart)
        
        # 添加图表区域到分割器
        self.splitter.addWidget(self.chart_frame)
        
        # 底部区域 - 标签页
        self.tab_widget = QTabWidget()
        
        # 市场数据标签页
        self.market_tab = QWidget()
        self.market_layout = QVBoxLayout(self.market_tab)
        
        # 市场数据表格
        self.market_table = QTableWidget()
        self.market_table.setColumnCount(6)
        self.market_table.setHorizontalHeaderLabels(["日期", "开盘价", "最高价", "最低价", "收盘价", "成交量"])
        self.market_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.market_layout.addWidget(self.market_table)
        
        # 添加市场数据标签页
        self.tab_widget.addTab(self.market_tab, "市场数据")
        
        # 量子分析标签页
        self.quantum_tab = QWidget()
        self.quantum_layout = QVBoxLayout(self.quantum_tab)
        
        # 分析结果标签
        self.analysis_label = QLabel("量子分析结果")
        self.analysis_label.setAlignment(Qt.AlignCenter)
        self.quantum_layout.addWidget(self.analysis_label)
        
        # 分析表格
        self.analysis_table = QTableWidget()
        self.analysis_table.setColumnCount(3)
        self.analysis_table.setHorizontalHeaderLabels(["维度", "指标", "值"])
        self.analysis_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.quantum_layout.addWidget(self.analysis_table)
        
        # 添加量子分析标签页
        self.tab_widget.addTab(self.quantum_tab, "量子分析")
        
        # 添加标签页到分割器
        self.splitter.addWidget(self.tab_widget)
        
        # 设置分割比例
        self.splitter.setSizes([500, 300])
        
    def _create_toolbar(self):
        """创建工具栏"""
        # 工具栏容器
        self.toolbar_frame = QFrame()
        self.toolbar_frame.setFrameShape(QFrame.StyledPanel)
        self.toolbar_frame.setFrameShadow(QFrame.Raised)
        self.toolbar_layout = QHBoxLayout(self.toolbar_frame)
        self.main_layout.addWidget(self.toolbar_frame)
        
        # 股票选择
        self.symbol_label = QLabel("股票代码:")
        self.toolbar_layout.addWidget(self.symbol_label)
        
        # 将下拉选择改为输入框
        self.symbol_input = QLineEdit()
        self.symbol_input.setPlaceholderText("输入股票代码 如: 600000")
        self.symbol_input.setFixedWidth(150)
        self.toolbar_layout.addWidget(self.symbol_input)
        
        # 搜索按钮
        self.search_button = QPushButton("搜索")
        self.search_button.clicked.connect(self._on_search_clicked)
        self.toolbar_layout.addWidget(self.search_button)
        
        # 添加弹性空间
        self.toolbar_layout.addStretch(1)
        
        # 加载数据按钮
        self.load_button = QPushButton("加载数据")
        self.load_button.clicked.connect(self._load_market_data)
        self.toolbar_layout.addWidget(self.load_button)
        
        # 量子分析按钮
        self.analyze_button = QPushButton("量子分析")
        self.analyze_button.clicked.connect(self._analyze_market_data)
        self.toolbar_layout.addWidget(self.analyze_button)
        
    def _on_search_clicked(self):
        """搜索股票按钮点击处理"""
        symbol = self.symbol_input.text().strip()
        if not symbol:
            logger.warning("股票代码不能为空")
            return
            
        # 添加中国股票代码支持
        self.current_symbol = symbol
        logger.info(f"搜索股票: {symbol}")
        
        # 即使系统未启动，也生成模拟数据用于演示
        self._load_market_data(ignore_system_check=True)
        
    def _load_market_data(self, ignore_system_check=False):
        """加载市场数据"""
        if not self.current_symbol:
            logger.warning("未指定股票代码")
            return
            
        # 检查市场到量子转换器是否已启动
        component_ready = self._check_component('converter')
        if not ignore_system_check and not component_ready:
            logger.warning("converter组件未初始化或未启动")
            return
            
        try:
            # 创建示例市场数据
            self._create_sample_data(self.current_symbol)
            
            # 确保市场数据包含当前股票
            if self.current_symbol not in self.market_data:
                logger.warning(f"无法找到股票数据: {self.current_symbol}")
                return
                
            # 更新UI
            self._update_ui()
            
        except Exception as e:
            logger.error(f"加载市场数据时出错: {str(e)}")
            
    def _analyze_market_data(self):
        """分析市场数据"""
        if not self.current_symbol or not self.market_data:
            logger.warning("没有市场数据可分析")
            return
            
        # 检查组件是否已启动
        component_ready = self._check_component('analyzer')
        if not component_ready:
            logger.warning("analyzer组件未初始化或未启动，将使用模拟分析结果")
            # 生成模拟分析结果
            self._create_sample_analysis()
            self._update_analysis_results()
            return
        
        try:
            # 获取当前股票数据
            symbol_data = self.market_data.get(self.current_symbol, {})
            
            # 调用市场分析器
            analyzer = self.system_manager.get_component('analyzer')
            analysis_results = analyzer.analyze({self.current_symbol: symbol_data})
            
            # 保存结果
            self.analysis_results = analysis_results
            
            # 更新UI
            self._update_analysis_results()
            
        except Exception as e:
            logger.error(f"分析市场数据时出错: {str(e)}")
            self._create_sample_analysis()
            self._update_analysis_results()
            
    def _update_ui(self):
        """更新UI显示"""
        if not self.current_symbol or not self.market_data:
            return
            
        # 获取当前股票数据
        symbol_data = self.market_data.get(self.current_symbol, {})
        
        # 更新K线图
        self.candle_chart.set_data(symbol_data)
        
        # 更新市场数据表格
        self._update_market_table(symbol_data)
        
    def _update_market_table(self, data):
        """更新市场数据表格"""
        self.market_table.setRowCount(0)
        
        if not data or 'dates' not in data:
            return
            
        dates = data.get('dates', [])
        opens = data.get('open', [])
        highs = data.get('high', [])
        lows = data.get('low', [])
        closes = data.get('close', [])
        volumes = data.get('volume', [])
        
        # 设置行数
        self.market_table.setRowCount(len(dates))
        
        # 添加数据
        for i in range(len(dates)):
            # 添加日期
            date_item = QTableWidgetItem(dates[i])
            self.market_table.setItem(i, 0, date_item)
            
            # 添加价格
            self.market_table.setItem(i, 1, QTableWidgetItem(f"{opens[i]:.2f}"))
            self.market_table.setItem(i, 2, QTableWidgetItem(f"{highs[i]:.2f}"))
            self.market_table.setItem(i, 3, QTableWidgetItem(f"{lows[i]:.2f}"))
            
            # 添加收盘价（上涨绿色，下跌红色）
            close_item = QTableWidgetItem(f"{closes[i]:.2f}")
            if i > 0 and closes[i] > closes[i-1]:
                close_item.setForeground(QColor(46, 204, 113))  # 绿色
            elif i > 0 and closes[i] < closes[i-1]:
                close_item.setForeground(QColor(231, 76, 60))   # 红色
            self.market_table.setItem(i, 4, close_item)
            
            # 添加成交量
            volume_item = QTableWidgetItem(f"{volumes[i]:,}")
            self.market_table.setItem(i, 5, volume_item)
            
    def _update_analysis_results(self):
        """更新分析结果"""
        self.analysis_table.setRowCount(0)
        
        if not self.analysis_results or 'combined' not in self.analysis_results:
            return
            
        # 获取综合分析结果
        combined = self.analysis_results.get('combined', {})
        
        # 提取当前股票结果
        symbol_results = combined.get('details', {}).get(self.current_symbol, {})
        if not symbol_results:
            return
            
        # 准备展示的分析结果
        rows = []
        
        # 添加得分
        score = symbol_results.get('score', 0)
        rows.append(("综合", "分数", f"{score:.2f}"))
        
        # 添加各维度结果
        dimensions = symbol_results.get('dimensions', {})
        
        for dim_name, dim_data in dimensions.items():
            if isinstance(dim_data, dict):
                for key, value in dim_data.items():
                    if key == 'trend' and isinstance(value, dict):
                        # 展开趋势信息
                        for trend_key, trend_value in value.items():
                            rows.append((dim_name, f"趋势.{trend_key}", self._format_value(trend_value)))
                    else:
                        rows.append((dim_name, key, self._format_value(value)))
                        
        # 设置行数
        self.analysis_table.setRowCount(len(rows))
        
        # 添加数据
        for i, row_data in enumerate(rows):
            dimension, indicator, value = row_data
            
            # 添加维度
            self.analysis_table.setItem(i, 0, QTableWidgetItem(dimension))
            
            # 添加指标
            self.analysis_table.setItem(i, 1, QTableWidgetItem(indicator))
            
            # 添加值
            value_item = QTableWidgetItem(value)
            
            # 为趋势方向设置颜色
            if indicator == "趋势.direction":
                if value == "up":
                    value_item.setText("上升")
                    value_item.setForeground(QColor(46, 204, 113))  # 绿色
                elif value == "down":
                    value_item.setText("下降")
                    value_item.setForeground(QColor(231, 76, 60))   # 红色
                    
            self.analysis_table.setItem(i, 2, value_item)
            
    def _format_value(self, value):
        """格式化值显示"""
        if isinstance(value, float):
            return f"{value:.4f}"
        elif isinstance(value, int):
            return f"{value:,}"
        else:
            return str(value)
            
    def _check_component(self, name):
        """检查组件是否已启动"""
        component = self.system_manager.get_component(name)
        
        if not component:
            logger.warning(f"{name}组件未初始化")
            return False
            
        if self.system_manager.get_component_status(name).get(name) != 'running':
            logger.warning(f"{name}组件未启动")
            return False
            
        return True
        
    def _create_sample_data(self, symbol=None):
        """创建示例市场数据"""
        # 创建一个简单的价格时间序列
        dates = [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(30)]
        dates.reverse()
        
        # 定义需要处理的股票代码
        if symbol and symbol.strip():
            symbols = [symbol.strip()]
        else:
            symbols = ['600000', '000001', '300059', '688608']
        
        # 初始化市场数据字典（如果不存在）
        if not hasattr(self, 'market_data') or not self.market_data:
            self.market_data = {}
        
        for symbol in symbols:
            # 如果已经有此股票数据，跳过生成
            if symbol in self.market_data:
                continue
                
            # 随机种子，使每个股票有不同的数据
            np.random.seed(hash(symbol) % 10000)
            
            # 生成基础价格趋势
            base_price = np.random.uniform(10, 100)
            trend = np.random.uniform(-0.3, 0.3)
            
            # 生成价格时间序列
            closes = [base_price]
            for i in range(1, len(dates)):
                next_price = closes[-1] * (1 + trend/100 + np.random.normal(0, 0.02))
                closes.append(max(0.01, next_price))  # 确保价格不为负
                
            # 从收盘价生成其他价格数据
            opens = [close * np.random.uniform(0.98, 1.0) for close in closes]
            highs = [max(open_price, close) * np.random.uniform(1.0, 1.03) for open_price, close in zip(opens, closes)]
            lows = [min(open_price, close) * np.random.uniform(0.97, 1.0) for open_price, close in zip(opens, closes)]
            volumes = [int(np.random.uniform(100000, 10000000)) for _ in closes]
            
            # 保存数据
            self.market_data[symbol] = {
                'dates': dates,
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'volume': volumes
            }
        
    def _create_sample_analysis(self):
        """创建示例分析结果"""
        if not self.current_symbol:
            return
            
        # 创建示例分析结果
        self.analysis_results = {
            'combined': {
                'overview': {
                    'total_assets': len(self.market_data),
                    'avg_score': 85.5,
                    'top_performer': self.current_symbol
                },
                'details': {
                    self.current_symbol: {
                        'score': round(np.random.uniform(60, 95), 2),
                        'dimensions': {
                            '技术指标': {
                                'momentum': round(np.random.uniform(-1, 1), 4),
                                'volatility': round(np.random.uniform(0, 0.5), 4),
                                'rsi': round(np.random.uniform(30, 70), 2),
                                'macd': round(np.random.uniform(-2, 2), 4),
                                'trend': {
                                    'strength': round(np.random.uniform(0, 1), 4),
                                    'direction': 'up' if np.random.random() > 0.5 else 'down'
                                }
                            },
                            '基本面': {
                                'pe_ratio': round(np.random.uniform(10, 30), 2),
                                'growth_potential': round(np.random.uniform(0, 1), 4),
                                'stability': round(np.random.uniform(0, 1), 4)
                            },
                            '量子分析': {
                                'quantum_score': round(np.random.uniform(70, 100), 2),
                                'entanglement': round(np.random.uniform(0, 1), 4),
                                'coherence': round(np.random.uniform(0, 1), 4),
                                'prediction': {
                                    'accuracy': round(np.random.uniform(0.6, 0.9), 4),
                                    'confidence': round(np.random.uniform(0.5, 1), 4)
                                }
                            }
                        }
                    }
                }
            }
        }
        
    def on_system_started(self):
        """系统启动时调用"""
        # 设置默认股票
        if not self.symbol_input.text():
            self.symbol_input.setText("600000")
            self.current_symbol = "600000"
            # 加载默认股票数据
            self._load_market_data(ignore_system_check=True)
        elif self.current_symbol:
            # 如果已有股票代码，刷新数据
            self._load_market_data()
        
    def on_system_stopped(self):
        """系统停止时调用"""
        # 保留数据，但清理分析结果
        self.analysis_results = {} 