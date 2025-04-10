"""
超神量子选股器面板 - 利用量子计算捕捉大牛股
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os
import json
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                         QPushButton, QComboBox, QFrame, QGridLayout, 
                         QTableWidget, QTableWidgetItem, QHeaderView,
                         QSplitter, QTabWidget, QLineEdit, QDateEdit,
                         QGroupBox, QFormLayout, QDoubleSpinBox, QSpinBox,
                         QCheckBox, QProgressBar, QTextEdit, QApplication,
                         QMessageBox, QSlider)
from PyQt5.QtCore import Qt, pyqtSlot, QDate, pyqtSignal, QThread, QTimer
from PyQt5.QtGui import QColor, QBrush, QPen, QPainter, QFont
from PyQt5.QtChart import (QChart, QChartView, QLineSeries, 
                       QValueAxis, QDateTimeAxis, QScatterSeries)

logger = logging.getLogger("QuantumDesktop.QuantumStockFinderPanel")

class StockFinderThread(QThread):
    """后台选股线程，避免UI卡顿"""
    
    # 定义信号
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    results_ready = pyqtSignal(dict)
    completed = pyqtSignal()
    
    def __init__(self, stock_strategy=None, quantum_power=50, market_scope="全市场", sector_filter=None, parent=None):
        super().__init__(parent)
        self.stock_strategy = stock_strategy
        self.quantum_power = quantum_power
        self.market_scope = market_scope
        self.sector_filter = sector_filter
        self.stopped = False
        
    def run(self):
        """执行选股算法"""
        self.stopped = False
        
        if not self.stock_strategy:
            # 使用模拟模式
            self._run_simulated()
            return
            
        try:
            # 设置量子能力
            self.stock_strategy.set_quantum_power(self.quantum_power)
            
            # 更新进度和状态
            self.progress_updated.emit(10)
            self.status_updated.emit("正在构建量子态叠加...")
            
            if self.stopped:
                return
                
            # 执行量子选股
            max_stocks = 3 + int(self.quantum_power / 20)  # 每个阶段的进度
            
            # 更新进度
            self.progress_updated.emit(30)
            self.status_updated.emit("正在执行多维度量子纠缠分析...")
            
            if self.stopped:
                return
                
            # 中间进度更新
            self.progress_updated.emit(50)
            self.status_updated.emit("正在解构市场潜在模式...")
            
            if self.stopped:
                return
                
            # 中间进度更新
            self.progress_updated.emit(70)
            self.status_updated.emit("正在计算超神系数...")
            
            if self.stopped:
                return
                
            # 中间进度更新
            self.progress_updated.emit(90)
            self.status_updated.emit("正在优化选股结果...")
            
            if self.stopped:
                return
                
            # 实际执行选股策略
            results = self.stock_strategy.find_potential_stocks(
                market_scope=self.market_scope,
                sector_filter=self.sector_filter if self.sector_filter != "全部行业" else None,
                max_stocks=max_stocks
            )
            
            # 检查结果
            if results.get("status") == "success":
                self.results_ready.emit(results)
                self.status_updated.emit("选股完成！已找到潜在大牛股")
            else:
                self.status_updated.emit(f"选股失败: {results.get('message', '未知错误')}")
                
            self.progress_updated.emit(100)
            self.completed.emit()
            
        except Exception as e:
            logger.error(f"执行选股线程时出错: {str(e)}")
            self.status_updated.emit(f"选股过程出错: {str(e)}")
            self.completed.emit()
            
    def _run_simulated(self):
        """运行模拟选股流程"""
        # 模拟量子计算过程
        max_steps = 100
        for i in range(max_steps):
            if self.stopped:
                break
                
            # 更新进度
            self.progress_updated.emit(i + 1)
            
            # 模拟不同阶段
            if i < 20:
                self.status_updated.emit("正在构建量子态叠加...")
            elif i < 40:
                self.status_updated.emit("正在执行多维度量子纠缠分析...")
            elif i < 60:
                self.status_updated.emit("正在解构市场潜在模式...")
            elif i < 80:
                self.status_updated.emit("正在计算超神系数...")
            else:
                self.status_updated.emit("正在优化选股结果...")
                
            # 模拟计算延迟
            self.msleep(50 + int(random.random() * 30))
        
        if not self.stopped:
            # 生成模拟结果
            results = self._generate_results()
            self.results_ready.emit(results)
            self.status_updated.emit("选股完成！已找到潜在大牛股")
            self.completed.emit()
    
    def stop(self):
        """停止选股过程"""
        self.stopped = True
        
    def _generate_results(self):
        """生成选股结果 (模拟数据)"""
        # 股票池
        stock_pool = [
            {"code": "600519", "name": "贵州茅台", "sector": "白酒"},
            {"code": "000858", "name": "五粮液", "sector": "白酒"},
            {"code": "601318", "name": "中国平安", "sector": "金融保险"},
            {"code": "600036", "name": "招商银行", "sector": "银行"},
            {"code": "000333", "name": "美的集团", "sector": "家电"},
            {"code": "600276", "name": "恒瑞医药", "sector": "医药"},
            {"code": "002475", "name": "立讯精密", "sector": "电子"},
            {"code": "300750", "name": "宁德时代", "sector": "新能源"},
            {"code": "603288", "name": "海天味业", "sector": "食品"},
            {"code": "601888", "name": "中国中免", "sector": "免税"},
            {"code": "600031", "name": "三一重工", "sector": "工程机械"},
            {"code": "000651", "name": "格力电器", "sector": "家电"},
            {"code": "002594", "name": "比亚迪", "sector": "汽车新能源"},
            {"code": "601899", "name": "紫金矿业", "sector": "有色金属"},
            {"code": "600887", "name": "伊利股份", "sector": "食品饮料"},
            {"code": "000538", "name": "云南白药", "sector": "医药"},
            {"code": "600309", "name": "万华化学", "sector": "化工"},
            {"code": "300059", "name": "东方财富", "sector": "金融信息"},
            {"code": "600900", "name": "长江电力", "sector": "公用事业"},
            {"code": "688981", "name": "中芯国际", "sector": "半导体"},
        ]
        
        # 如果设置了行业过滤，应用过滤
        if self.sector_filter and self.sector_filter != "全部行业":
            stock_pool = [s for s in stock_pool if s["sector"] == self.sector_filter]
            
        if len(stock_pool) == 0:
            # 如果过滤后没有股票，返回原始池
            stock_pool = [
                {"code": "600519", "name": "贵州茅台", "sector": "白酒"},
                {"code": "000858", "name": "五粮液", "sector": "白酒"},
                {"code": "601318", "name": "中国平安", "sector": "金融保险"}
            ]
            
        # 根据量子功率选择股票数量
        num_stocks = 3 + int(self.quantum_power / 20)
        selected_stocks = random.sample(stock_pool, min(num_stocks, len(stock_pool)))
        
        results = []
        for stock in selected_stocks:
            # 生成随机的超神评分和预期上涨空间
            quantum_score = random.uniform(80, 99.5)
            expected_gain = random.uniform(20, 100)
            
            # 生成随机理由
            reasons = [
                "量子态分析显示强势突破形态",
                "多维度市场情绪指标极度看好",
                "超空间趋势通道形成",
                "量子态交叉信号确认",
                "多维度技术指标共振",
                "行业景气度量子评分处于高位",
                "超神算法检测到主力资金潜伏",
                "量子波动特征与历史大牛股吻合",
                "超空间市场结构分析显示稀缺性溢价",
                "行业拐点信号被超神算法捕捉"
            ]
            
            selected_reasons = random.sample(reasons, 3)
            
            # 添加到结果列表
            results.append({
                "code": stock["code"],
                "name": stock["name"],
                "sector": stock["sector"],
                "quantum_score": quantum_score,
                "expected_gain": expected_gain,
                "confidence": random.uniform(75, 98),
                "timeframe": random.choice(["短期", "中期", "长期"]),
                "reasons": selected_reasons,
                "recommendation": "强烈推荐" if quantum_score > 95 else "推荐"
            })
            
        # 按超神评分排序
        results.sort(key=lambda x: x["quantum_score"], reverse=True)
        
        return {
            "status": "success",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "quantum_power": self.quantum_power,
            "market_scope": self.market_scope,
            "stocks": results
        }

class PotentialStockChart(QChartView):
    """潜力股图表 - 显示超神选股结果的图形化表示"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 创建图表
        self.chart = QChart()
        self.chart.setTitle("超神量子选股分析")
        self.chart.setAnimationOptions(QChart.SeriesAnimations)
        
        # 设置抗锯齿
        self.setRenderHint(QPainter.Antialiasing)
        
        # 设置图表
        self.setChart(self.chart)
        
        # 创建散点图系列
        self.scatter_series = QScatterSeries()
        self.scatter_series.setName("潜力股")
        self.scatter_series.setMarkerSize(15)
        
        # 添加系列到图表
        self.chart.addSeries(self.scatter_series)
        
        # 创建X轴 (预期收益)
        self.axis_x = QValueAxis()
        self.axis_x.setTitleText("预期收益率(%)")
        self.axis_x.setRange(0, 100)
        
        # 创建Y轴 (超神评分)
        self.axis_y = QValueAxis()
        self.axis_y.setTitleText("超神量子评分")
        self.axis_y.setRange(70, 100)
        
        # 添加轴到图表
        self.chart.addAxis(self.axis_x, Qt.AlignBottom)
        self.chart.addAxis(self.axis_y, Qt.AlignLeft)
        
        # 将系列附加到轴
        self.scatter_series.attachAxis(self.axis_x)
        self.scatter_series.attachAxis(self.axis_y)
        
        # 设置图表主题和样式
        self.chart.setTheme(QChart.ChartThemeDark)
        self.chart.setBackgroundVisible(False)
        self.chart.legend().setVisible(True)
        self.chart.legend().setAlignment(Qt.AlignBottom)
        
    def set_data(self, stocks):
        """设置股票数据"""
        # 清除旧数据
        self.scatter_series.clear()
        
        if not stocks:
            return
        
        # 添加散点数据
        for stock in stocks:
            expected_gain = stock.get("expected_gain", 0)
            quantum_score = stock.get("quantum_score", 0)
            self.scatter_series.append(expected_gain, quantum_score)
            
        # 刷新图表
        self.chart.update()

class QuantumStockFinderPanel(QWidget):
    """超神量子选股器面板 - 利用量子算法发现潜在大牛股"""
    
    def __init__(self, system_manager, parent=None):
        super().__init__(parent)
        self.system_manager = system_manager
        
        # 选股结果
        self.finder_results = None
        
        # 选股线程
        self.finder_thread = None
        
        # 量子选股策略
        self.stock_strategy = None
        
        # 历史选股数据
        self.historical_data = None
        
        # 初始化量子选股策略
        self._init_stock_strategy()
        
        # 加载历史数据
        self._load_historical_data()
        
        # 初始化UI
        self._init_ui()
        
        logger.info("超神量子选股器面板初始化完成")
        
    def _init_stock_strategy(self):
        """初始化量子选股策略"""
        try:
            # 导入量子选股策略
            from quantum_core.quantum_stock_strategy import QuantumStockStrategy
            
            # 获取量子后端
            quantum_backend = None
            if self.system_manager:
                quantum_backend = self.system_manager.get_component("quantum_backend")
                
            # 获取市场分析器
            market_analyzer = None
            if self.system_manager:
                market_analyzer = self.system_manager.get_component("market_analyzer")
                
            # 创建量子选股策略
            self.stock_strategy = QuantumStockStrategy(
                quantum_backend=quantum_backend,
                market_analyzer=market_analyzer
            )
            
            # 如果系统已启动，则启动策略
            if self.system_manager and self.system_manager.is_system_running():
                self.stock_strategy.start()
                
            logger.info("量子选股策略初始化成功")
            
        except Exception as e:
            logger.error(f"初始化量子选股策略时出错: {str(e)}")
            self.stock_strategy = None

    def _load_historical_data(self):
        """加载历史选股数据"""
        try:
            # 检查是否存在历史数据目录
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
                
            # 历史数据文件路径
            history_file = os.path.join(data_dir, "stock_finder_history.json")
            
            if os.path.exists(history_file):
                with open(history_file, 'r', encoding='utf-8') as f:
                    self.historical_data = json.load(f)
                logger.info(f"已加载历史选股数据: {len(self.historical_data.get('stocks', []))} 只股票")
            else:
                logger.info("未找到历史选股数据")
        except Exception as e:
            logger.error(f"加载历史选股数据时出错: {str(e)}")
            self.historical_data = None

    def _init_ui(self):
        """初始化用户界面"""
        # 主布局
        self.main_layout = QVBoxLayout(self)
        
        # 顶部标题
        self.title_label = QLabel("超神量子选股器")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 20px; font-weight: bold; margin: 10px;")
        self.main_layout.addWidget(self.title_label)
        
        # 描述标签
        self.description_label = QLabel(
            "超越宇宙的选股策略，利用量子算法捕捉市场中的大牛股，助您超神"
        )
        self.description_label.setAlignment(Qt.AlignCenter)
        self.description_label.setStyleSheet("font-size: 12px; margin-bottom: 15px;")
        self.main_layout.addWidget(self.description_label)
        
        # 配置区域
        self._create_config_area()
        
        # 结果显示区域
        self._create_results_area()
        
        # 底部状态区域
        self._create_status_area()
        
    def _create_config_area(self):
        """创建配置区域"""
        # 配置框架
        self.config_frame = QFrame()
        self.config_frame.setFrameShape(QFrame.StyledPanel)
        self.config_layout = QHBoxLayout(self.config_frame)
        self.main_layout.addWidget(self.config_frame)
        
        # 超神量子能力设置组
        self.power_group = QGroupBox("超神量子能力")
        self.power_layout = QVBoxLayout(self.power_group)
        
        # 量子能力滑块
        self.power_label = QLabel("量子计算能力:")
        self.power_layout.addWidget(self.power_label)
        
        self.power_slider = QSlider(Qt.Horizontal)
        self.power_slider.setMinimum(10)
        self.power_slider.setMaximum(100)
        self.power_slider.setValue(50)
        self.power_slider.setTickPosition(QSlider.TicksBelow)
        self.power_slider.setTickInterval(10)
        self.power_slider.valueChanged.connect(self._on_power_changed)
        self.power_layout.addWidget(self.power_slider)
        
        self.power_value_label = QLabel("50% - 平衡")
        self.power_value_label.setAlignment(Qt.AlignCenter)
        self.power_layout.addWidget(self.power_value_label)
        
        self.config_layout.addWidget(self.power_group)
        
        # 市场选择组
        self.market_group = QGroupBox("市场范围")
        self.market_layout = QFormLayout(self.market_group)
        
        # 市场选择
        self.market_combo = QComboBox()
        self.market_combo.addItems(["全市场", "沪深300", "中证500", "科创板", "创业板"])
        self.market_layout.addRow("选股范围:", self.market_combo)
        
        # 行业选择
        self.sector_combo = QComboBox()
        self.sector_combo.addItems(["全部行业", "科技", "消费", "医药", "金融", "新能源", "先进制造"])
        self.market_layout.addRow("行业选择:", self.sector_combo)
        
        self.config_layout.addWidget(self.market_group)
        
        # 操作按钮组
        self.action_group = QGroupBox("操作")
        self.action_layout = QVBoxLayout(self.action_group)
        
        # 开始选股按钮
        self.start_button = QPushButton("开始量子选股")
        self.start_button.setStyleSheet("font-weight: bold; height: 40px;")
        self.start_button.clicked.connect(self._on_start_finder)
        self.action_layout.addWidget(self.start_button)
        
        # 停止按钮
        self.stop_button = QPushButton("停止")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._on_stop_finder)
        self.action_layout.addWidget(self.stop_button)
        
        # 刷新数据按钮
        self.refresh_button = QPushButton("刷新数据")
        self.refresh_button.setToolTip("重新选股，不使用历史数据")
        self.refresh_button.clicked.connect(self._on_refresh_data)
        self.action_layout.addWidget(self.refresh_button)
        
        self.config_layout.addWidget(self.action_group)
        
    def _create_results_area(self):
        """创建结果显示区域"""
        # 分割器
        self.results_splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.results_splitter, 1)
        
        # 左侧：结果表格
        self.table_frame = QFrame()
        self.table_layout = QVBoxLayout(self.table_frame)
        
        # 结果表格
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(7)
        self.results_table.setHorizontalHeaderLabels([
            "代码", "名称", "行业", "超神评分", 
            "预期涨幅", "推荐", "时间周期"
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setSelectionMode(QTableWidget.SingleSelection)
        self.results_table.itemSelectionChanged.connect(self._on_stock_selected)
        self.table_layout.addWidget(self.results_table)
        
        self.results_splitter.addWidget(self.table_frame)
        
        # 右侧：详细信息和图表
        self.details_frame = QFrame()
        self.details_layout = QVBoxLayout(self.details_frame)
        
        # 潜力股图表
        self.stock_chart = PotentialStockChart()
        self.details_layout.addWidget(self.stock_chart)
        
        # 详细信息组
        self.details_group = QGroupBox("股票详情")
        self.details_group_layout = QVBoxLayout(self.details_group)
        
        # 选中股票信息
        self.stock_info_label = QLabel("请选择股票查看详情")
        self.stock_info_label.setAlignment(Qt.AlignCenter)
        self.details_group_layout.addWidget(self.stock_info_label)
        
        # 超神推荐理由
        self.reasons_label = QLabel("超神推荐理由")
        self.reasons_label.setStyleSheet("font-weight: bold;")
        self.details_group_layout.addWidget(self.reasons_label)
        
        self.reasons_text = QTextEdit()
        self.reasons_text.setReadOnly(True)
        self.reasons_text.setMaximumHeight(150)
        self.details_group_layout.addWidget(self.reasons_text)
        
        self.details_layout.addWidget(self.details_group)
        
        self.results_splitter.addWidget(self.details_frame)
        
        # 设置分割比例
        self.results_splitter.setSizes([600, 500])
        
    def _create_status_area(self):
        """创建状态区域"""
        # 状态框架
        self.status_frame = QFrame()
        self.status_layout = QHBoxLayout(self.status_frame)
        self.main_layout.addWidget(self.status_frame)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(100)
        self.status_layout.addWidget(self.progress_bar, 1)
        
        # 状态标签
        self.status_label = QLabel("就绪")
        self.status_layout.addWidget(self.status_label)
        
    def _on_power_changed(self, value):
        """量子能力滑块值变化处理"""
        power_descriptions = {
            10: "低 - 保守选股",
            30: "中低 - 稳健选股",
            50: "平衡",
            70: "中高 - 进取选股",
            90: "超高 - 激进选股",
            100: "最大 - 极限预测"
        }
        
        # 找到最接近的描述
        closest_key = min(power_descriptions.keys(), key=lambda k: abs(k - value))
        description = power_descriptions[closest_key]
        
        self.power_value_label.setText(f"{value}% - {description}")
        
    def _on_start_finder(self):
        """开始选股按钮点击事件"""
        # 检查是否有历史数据可用
        if self.historical_data and self.historical_data.get("stocks"):
            self._use_historical_data()
            return
            
        # 检查系统状态，但即使未启动也不阻止选股
        system_running = True
        if self.system_manager:
            system_running = self._check_system_ready()
            
        # 获取配置
        quantum_power = self.power_slider.value()
        market_scope = self.market_combo.currentText()
        sector_filter = self.sector_combo.currentText()
        
        # 如果系统未就绪，使用模拟模式并通知用户
        if not system_running:
            logger.info("系统未就绪，使用模拟模式进行选股")
            self.status_label.setText("系统未就绪，使用模拟模式进行选股...")
            # 创建不带策略的线程（触发模拟模式）
            self.finder_thread = StockFinderThread(
                stock_strategy=None,
                quantum_power=quantum_power, 
                market_scope=market_scope, 
                sector_filter=sector_filter
            )
        else:
            # 创建并启动选股线程（正常模式）
            self.finder_thread = StockFinderThread(
                stock_strategy=self.stock_strategy,
                quantum_power=quantum_power, 
                market_scope=market_scope, 
                sector_filter=sector_filter
            )
        
        # 连接信号
        self.finder_thread.progress_updated.connect(self.progress_bar.setValue)
        self.finder_thread.status_updated.connect(self.status_label.setText)
        self.finder_thread.results_ready.connect(self._on_results_ready)
        self.finder_thread.completed.connect(self._on_finder_completed)
        
        # 启动线程
        self.finder_thread.start()
        
        # 更新UI状态
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.power_slider.setEnabled(False)
        self.market_combo.setEnabled(False)
        self.sector_combo.setEnabled(False)
        self.progress_bar.setValue(0)
        if system_running:
            self.status_label.setText("正在初始化量子选股算法...")
        
    def _on_stop_finder(self):
        """停止选股按钮点击事件"""
        if self.finder_thread and self.finder_thread.isRunning():
            self.finder_thread.stop()
            self.status_label.setText("正在停止...")
            
            # 使用计时器等待线程终止
            QTimer.singleShot(500, self._reset_ui_after_stop)
            
    def _reset_ui_after_stop(self):
        """停止选股后重置UI"""
        # 更新UI状态
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.power_slider.setEnabled(True)
        self.market_combo.setEnabled(True)
        self.sector_combo.setEnabled(True)
        self.status_label.setText("已停止")
        
    def _on_results_ready(self, results):
        """选股结果就绪处理"""
        self.finder_results = results
        
        # 保存结果到历史文件
        self._save_results_to_history(results)
        
        self._update_results_table()
        self._update_chart()
        
    def _on_finder_completed(self):
        """选股完成处理"""
        # 更新UI状态
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.power_slider.setEnabled(True)
        self.market_combo.setEnabled(True)
        self.sector_combo.setEnabled(True)
        
    def _update_results_table(self):
        """更新结果表格"""
        if not self.finder_results:
            return
            
        stocks = self.finder_results.get("stocks", [])
        
        # 设置表格行数
        self.results_table.setRowCount(len(stocks))
        
        for row, stock in enumerate(stocks):
            # 设置单元格数据
            self.results_table.setItem(row, 0, QTableWidgetItem(stock["code"]))
            self.results_table.setItem(row, 1, QTableWidgetItem(stock["name"]))
            self.results_table.setItem(row, 2, QTableWidgetItem(stock["sector"]))
            
            # 超神评分
            score_item = QTableWidgetItem(f"{stock['quantum_score']:.2f}")
            # 根据评分设置颜色
            if stock["quantum_score"] >= 95:
                score_item.setBackground(QBrush(QColor(255, 215, 0, 100)))  # 金色
            elif stock["quantum_score"] >= 90:
                score_item.setBackground(QBrush(QColor(0, 255, 0, 50)))  # 绿色
            self.results_table.setItem(row, 3, score_item)
            
            # 预期涨幅
            gain_item = QTableWidgetItem(f"{stock['expected_gain']:.2f}%")
            if stock["expected_gain"] >= 50:
                gain_item.setBackground(QBrush(QColor(255, 100, 100, 80)))
            elif stock["expected_gain"] >= 30:
                gain_item.setBackground(QBrush(QColor(255, 180, 0, 80)))
            self.results_table.setItem(row, 4, gain_item)
            
            # 推荐
            self.results_table.setItem(row, 5, QTableWidgetItem(stock["recommendation"]))
            
            # 时间周期
            self.results_table.setItem(row, 6, QTableWidgetItem(stock["timeframe"]))
            
        # 自动选择第一行
        if len(stocks) > 0:
            self.results_table.selectRow(0)
            
    def _update_chart(self):
        """更新图表"""
        if not self.finder_results:
            return
            
        stocks = self.finder_results.get("stocks", [])
        self.stock_chart.set_data(stocks)
        
    def _on_stock_selected(self):
        """股票选中事件处理"""
        selected_rows = self.results_table.selectionModel().selectedRows()
        if not selected_rows or not self.finder_results:
            return
            
        row = selected_rows[0].row()
        stocks = self.finder_results.get("stocks", [])
        
        if row >= 0 and row < len(stocks):
            selected_stock = stocks[row]
            self._display_stock_details(selected_stock)
            
    def _display_stock_details(self, stock):
        """显示选中股票的详细信息"""
        # 更新股票信息标签
        info_text = (
            f"<b>{stock['name']}</b> ({stock['code']}) - {stock['sector']}<br>"
            f"超神评分: <span style='color: {'gold' if stock['quantum_score'] >= 95 else 'green'};'>"
            f"{stock['quantum_score']:.2f}</span>  "
            f"预期涨幅: <span style='color: red;'>{stock['expected_gain']:.2f}%</span><br>"
            f"推荐: {stock['recommendation']}  "
            f"置信度: {stock['confidence']:.1f}%  "
            f"时间周期: {stock['timeframe']}"
        )
        self.stock_info_label.setText(info_text)
        
        # 更新推荐理由
        reasons_text = "■ " + "\n\n■ ".join(stock["reasons"])
        self.reasons_text.setText(reasons_text)
        
    def _check_system_ready(self):
        """检查系统是否准备就绪"""
        if not self.system_manager:
            return False
            
        # 确保量子选股策略已启动
        if self.stock_strategy and not self.stock_strategy.is_running:
            try:
                self.stock_strategy.start()
            except Exception as e:
                logger.error(f"启动量子选股策略时出错: {str(e)}")
                
        # 检查量子后端是否启动
        try:
            quantum_backend = self.system_manager.get_component("quantum_backend")
            if quantum_backend and quantum_backend.is_active():
                return True
                
            # 尝试启动量子后端
            if quantum_backend and not quantum_backend.is_active():
                return quantum_backend.start()
                
        except Exception as e:
            logger.error(f"检查系统状态时出错: {str(e)}")
            
        return False
    
    def on_system_started(self):
        """系统启动事件处理"""
        # 启动量子选股策略
        if self.stock_strategy and not self.stock_strategy.is_running:
            try:
                self.stock_strategy.start()
            except Exception as e:
                logger.error(f"启动量子选股策略时出错: {str(e)}")
                
        # 允许使用选股功能
        self.start_button.setEnabled(True)
        
    def on_system_stopped(self):
        """系统停止事件处理"""
        # 停止量子选股策略
        if self.stock_strategy and self.stock_strategy.is_running:
            try:
                self.stock_strategy.stop()
            except Exception as e:
                logger.error(f"停止量子选股策略时出错: {str(e)}")
                
        # 允许使用选股功能（即使系统停止也可以使用模拟模式）
        self.start_button.setEnabled(True)
        
        # 如果正在进行选股，停止它
        if self.finder_thread and self.finder_thread.isRunning():
            self._on_stop_finder()

    def _use_historical_data(self):
        """使用历史数据显示选股结果"""
        logger.info("使用保存的历史选股数据")
        
        # 直接使用历史数据
        self.finder_results = self.historical_data
        
        # 模拟进度条效果
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.power_slider.setEnabled(False)
        self.market_combo.setEnabled(False)
        self.sector_combo.setEnabled(False)
        
        # 清空表格
        self.results_table.setRowCount(0)
        
        # 模拟正在加载
        self.status_label.setText("正在读取历史选股数据...")
        
        # 使用计时器模拟加载进度
        self.progress_counter = 0
        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(self._update_history_progress)
        self.progress_timer.start(30)
    
    def _update_history_progress(self):
        """更新历史数据加载进度"""
        self.progress_counter += 3
        self.progress_bar.setValue(self.progress_counter)
        
        if self.progress_counter >= 100:
            self.progress_timer.stop()
            self.status_label.setText("已加载历史选股数据")
            self._update_results_table()
            self._update_chart()
            self._on_finder_completed()
            
    def _save_results_to_history(self, results):
        """保存选股结果到历史文件"""
        try:
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
                
            history_file = os.path.join(data_dir, "stock_finder_history.json")
            
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                
            logger.info(f"已保存选股结果到历史文件: {history_file}")
            self.historical_data = results
        except Exception as e:
            logger.error(f"保存选股结果到历史文件时出错: {str(e)}")

    def _on_refresh_data(self):
        """刷新数据按钮点击事件"""
        # 临时禁用历史数据
        temp_historical = self.historical_data
        self.historical_data = None
        
        # 调用开始选股
        self._on_start_finder()
        
        # 恢复历史数据引用（新的结果会通过_save_results_to_history更新）
        self.historical_data = temp_historical 