"""
策略回测面板 - 量子策略回测与评估
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                         QPushButton, QComboBox, QFrame, QGridLayout, 
                         QTableWidget, QTableWidgetItem, QHeaderView,
                         QSplitter, QTabWidget, QLineEdit, QDateEdit,
                         QGroupBox, QFormLayout, QDoubleSpinBox, QSpinBox,
                         QCheckBox, QProgressBar, QTextEdit, QApplication)
from PyQt5.QtCore import Qt, pyqtSlot, QDate, pyqtSignal
from PyQt5.QtGui import QColor, QBrush, QPen, QPainter
from PyQt5.QtChart import (QChart, QChartView, QLineSeries, 
                       QValueAxis, QDateTimeAxis, QBarSeries, QBarSet)

logger = logging.getLogger("QuantumDesktop.StrategyBacktestPanel")

class PerformanceChart(QChartView):
    """回测性能图表 - 显示策略回测结果"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 创建图表
        self.chart = QChart()
        self.chart.setTitle("策略回测结果")
        self.chart.setAnimationOptions(QChart.SeriesAnimations)
        
        # 设置抗锯齿
        self.setRenderHint(QPainter.Antialiasing)
        
        # 设置图表
        self.setChart(self.chart)
        
        # 创建策略与基准的系列
        self.strategy_series = QLineSeries()
        self.strategy_series.setName("策略收益")
        
        self.benchmark_series = QLineSeries()
        self.benchmark_series.setName("基准收益")
        
        # 添加系列到图表
        self.chart.addSeries(self.strategy_series)
        self.chart.addSeries(self.benchmark_series)
        
        # 创建时间轴
        self.axis_x = QDateTimeAxis()
        self.axis_x.setFormat("MM-dd")
        self.axis_x.setTitleText("日期")
        
        # 创建价格轴
        self.axis_y = QValueAxis()
        self.axis_y.setTitleText("收益率(%)")
        
        # 添加轴到图表
        self.chart.addAxis(self.axis_x, Qt.AlignBottom)
        self.chart.addAxis(self.axis_y, Qt.AlignLeft)
        
        # 将系列附加到轴
        self.strategy_series.attachAxis(self.axis_x)
        self.strategy_series.attachAxis(self.axis_y)
        self.benchmark_series.attachAxis(self.axis_x)
        self.benchmark_series.attachAxis(self.axis_y)
        
        # 默认显示图例
        self.chart.legend().setVisible(True)
        self.chart.legend().setAlignment(Qt.AlignBottom)
        
    def set_data(self, result_data):
        """设置回测结果数据"""
        # 清除旧数据
        self.strategy_series.clear()
        self.benchmark_series.clear()
        
        if not result_data or 'dates' not in result_data:
            return
            
        dates = result_data.get('dates', [])
        strategy_returns = result_data.get('strategy_returns', [])
        benchmark_returns = result_data.get('benchmark_returns', [])
        
        min_value = float('inf')
        max_value = float('-inf')
        min_date = None
        max_date = None
        
        # 添加策略收益数据
        for i in range(len(dates)):
            try:
                date_obj = datetime.strptime(dates[i], '%Y-%m-%d')
                timestamp = date_obj.timestamp() * 1000  # 毫秒时间戳
                
                strategy_value = strategy_returns[i] * 100  # 转换为百分比
                benchmark_value = benchmark_returns[i] * 100  # 转换为百分比
                
                self.strategy_series.append(timestamp, strategy_value)
                self.benchmark_series.append(timestamp, benchmark_value)
                
                # 更新数值范围
                min_value = min(min_value, strategy_value, benchmark_value)
                max_value = max(max_value, strategy_value, benchmark_value)
                
                # 更新时间范围
                if min_date is None or date_obj < min_date:
                    min_date = date_obj
                if max_date is None or date_obj > max_date:
                    max_date = date_obj
                    
            except Exception as e:
                logger.error(f"添加回测图表数据时出错: {str(e)}")
                
        # 设置轴范围
        if min_date and max_date:
            self.axis_x.setRange(min_date, max_date + timedelta(days=1))
            
        if min_value < max_value:
            value_range = max_value - min_value
            self.axis_y.setRange(min_value - value_range * 0.05, max_value + value_range * 0.05)
            
class StrategyBacktestPanel(QWidget):
    """策略回测面板 - 使用量子策略进行回测"""
    
    # 定义信号
    backtest_started = pyqtSignal()
    backtest_finished = pyqtSignal(dict)
    backtest_error = pyqtSignal(str)
    
    def __init__(self, system_manager, parent=None):
        super().__init__(parent)
        self.system_manager = system_manager
        
        # 回测数据
        self.backtest_results = {}
        self.stock_data = {}
        
        # 初始化UI
        self._init_ui()
        
        logger.info("策略回测面板初始化完成")
        
    def _init_ui(self):
        """初始化用户界面"""
        # 主布局
        self.main_layout = QVBoxLayout(self)
        
        # 顶部配置区域
        self.config_frame = QFrame()
        self.config_layout = QHBoxLayout(self.config_frame)
        self.main_layout.addWidget(self.config_frame)
        
        # 股票选择组
        self.stock_group = QGroupBox("股票选择")
        self.stock_layout = QFormLayout(self.stock_group)
        
        # 股票代码输入
        self.stock_input = QLineEdit()
        self.stock_input.setPlaceholderText("输入股票代码 如: 600000")
        self.stock_layout.addRow("股票代码:", self.stock_input)
        
        # 基准指数输入
        self.benchmark_input = QLineEdit()
        self.benchmark_input.setPlaceholderText("基准指数 如: 000300")
        self.benchmark_input.setText("000300")  # 默认沪深300
        self.stock_layout.addRow("基准指数:", self.benchmark_input)
        
        self.config_layout.addWidget(self.stock_group)
        
        # 回测参数组
        self.params_group = QGroupBox("回测参数")
        self.params_layout = QFormLayout(self.params_group)
        
        # 开始日期
        self.start_date = QDateEdit()
        self.start_date.setDisplayFormat("yyyy-MM-dd")
        self.start_date.setDate(QDate.currentDate().addYears(-1))
        self.params_layout.addRow("开始日期:", self.start_date)
        
        # 结束日期
        self.end_date = QDateEdit()
        self.end_date.setDisplayFormat("yyyy-MM-dd")
        self.end_date.setDate(QDate.currentDate())
        self.params_layout.addRow("结束日期:", self.end_date)
        
        # 初始资金
        self.initial_capital = QDoubleSpinBox()
        self.initial_capital.setRange(10000, 10000000)
        self.initial_capital.setValue(1000000)
        self.initial_capital.setSingleStep(10000)
        self.initial_capital.setPrefix("¥ ")
        self.params_layout.addRow("初始资金:", self.initial_capital)
        
        self.config_layout.addWidget(self.params_group)
        
        # 策略参数组
        self.strategy_group = QGroupBox("量子策略参数")
        self.strategy_layout = QFormLayout(self.strategy_group)
        
        # 止损比例
        self.stop_loss = QDoubleSpinBox()
        self.stop_loss.setRange(0.01, 0.2)
        self.stop_loss.setValue(0.05)
        self.stop_loss.setSingleStep(0.01)
        self.stop_loss.setSuffix(" (5%)")
        self.strategy_layout.addRow("止损比例:", self.stop_loss)
        
        # 止盈比例
        self.take_profit = QDoubleSpinBox()
        self.take_profit.setRange(0.05, 0.5)
        self.take_profit.setValue(0.2)
        self.take_profit.setSingleStep(0.01)
        self.take_profit.setSuffix(" (20%)")
        self.strategy_layout.addRow("止盈比例:", self.take_profit)
        
        # 回看周期
        self.lookback_period = QSpinBox()
        self.lookback_period.setRange(5, 60)
        self.lookback_period.setValue(20)
        self.lookback_period.setSuffix(" 天")
        self.strategy_layout.addRow("回看周期:", self.lookback_period)
        
        # 信号阈值
        self.signal_threshold = QDoubleSpinBox()
        self.signal_threshold.setRange(0.1, 0.9)
        self.signal_threshold.setValue(0.6)
        self.signal_threshold.setSingleStep(0.05)
        self.strategy_layout.addRow("信号阈值:", self.signal_threshold)
        
        self.config_layout.addWidget(self.strategy_group)
        
        # 按钮面板
        self.buttons_frame = QFrame()
        self.buttons_layout = QVBoxLayout(self.buttons_frame)
        
        # 回测按钮
        self.backtest_button = QPushButton("开始回测")
        self.backtest_button.setStyleSheet("background-color: #4CAF50; color: white;")
        self.backtest_button.clicked.connect(self._run_backtest)
        self.buttons_layout.addWidget(self.backtest_button)
        
        # 优化按钮
        self.optimize_button = QPushButton("策略优化")
        self.optimize_button.setStyleSheet("background-color: #2196F3; color: white;")
        self.optimize_button.clicked.connect(self._optimize_strategy)
        self.buttons_layout.addWidget(self.optimize_button)
        
        # 导出按钮
        self.export_button = QPushButton("导出结果")
        self.export_button.clicked.connect(self._export_results)
        self.buttons_layout.addWidget(self.export_button)
        
        self.config_layout.addWidget(self.buttons_frame)
        
        # 中部进度条
        self.progress_frame = QFrame()
        self.progress_layout = QHBoxLayout(self.progress_frame)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("就绪")
        self.progress_layout.addWidget(self.status_label)
        
        self.main_layout.addWidget(self.progress_frame)
        
        # 分割器 - 将界面分为上下两部分
        self.results_splitter = QSplitter(Qt.Vertical)
        self.main_layout.addWidget(self.results_splitter, 1)  # 加1表示可伸展
        
        # 结果图表区域
        self.chart_frame = QFrame()
        self.chart_layout = QVBoxLayout(self.chart_frame)
        
        # 绩效图表
        self.performance_chart = PerformanceChart()
        self.chart_layout.addWidget(self.performance_chart)
        
        self.results_splitter.addWidget(self.chart_frame)
        
        # 底部标签页
        self.results_tabs = QTabWidget()
        
        # 回测结果标签页
        self.results_tab = QWidget()
        self.results_layout = QVBoxLayout(self.results_tab)
        
        # 回测摘要
        self.summary_frame = QFrame()
        self.summary_frame.setFrameShape(QFrame.StyledPanel)
        self.summary_layout = QGridLayout(self.summary_frame)
        
        # 创建结果指标标签
        metrics = [
            ("最终资金:", "final_equity", "¥ {:,.2f}"),
            ("总收益率:", "total_return", "{:.2%}"),
            ("年化收益率:", "annual_return", "{:.2%}"),
            ("夏普比率:", "sharpe_ratio", "{:.2f}"),
            ("最大回撤:", "max_drawdown", "{:.2%}"),
            ("胜率:", "win_rate", "{:.2%}"),
            ("盈亏比:", "profit_factor", "{:.2f}"),
            ("交易次数:", "trade_count", "{:d}"),
        ]
        
        row, col = 0, 0
        self.metric_labels = {}
        for i, (title, key, fmt) in enumerate(metrics):
            title_label = QLabel(title)
            title_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            
            value_label = QLabel("--")
            value_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            
            self.metric_labels[key] = (value_label, fmt)
            
            self.summary_layout.addWidget(title_label, row, col*2)
            self.summary_layout.addWidget(value_label, row, col*2+1)
            
            col += 1
            if col >= 4:  # 每行4个指标
                col = 0
                row += 1
        
        self.results_layout.addWidget(self.summary_frame)
        
        # 交易明细表格
        self.trades_table = QTableWidget()
        self.trades_table.setColumnCount(7)
        self.trades_table.setHorizontalHeaderLabels(["开仓日期", "开仓价", "平仓日期", "平仓价", "方向", "持仓天数", "收益率"])
        self.trades_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_layout.addWidget(self.trades_table)
        
        self.results_tabs.addTab(self.results_tab, "回测结果")
        
        # 日志标签页
        self.log_tab = QWidget()
        self.log_layout = QVBoxLayout(self.log_tab)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_layout.addWidget(self.log_text)
        
        self.results_tabs.addTab(self.log_tab, "回测日志")
        
        self.results_splitter.addWidget(self.results_tabs)
        
        # 设置分割比例
        self.results_splitter.setSizes([400, 400])
        
    def _run_backtest(self):
        """运行回测"""
        # 获取回测参数
        stock_code = self.stock_input.text().strip()
        if not stock_code:
            self._log("错误: 请输入股票代码")
            return
            
        try:
            # 更新界面状态
            self.backtest_button.setEnabled(False)
            self.optimize_button.setEnabled(False)
            self.export_button.setEnabled(False)
            self.progress_bar.setValue(0)
            self.status_label.setText("正在回测...")
            
            # 记录参数
            self._log(f"开始回测 股票: {stock_code}")
            self._log(f"日期范围: {self.start_date.text()} 至 {self.end_date.text()}")
            self._log(f"初始资金: {self.initial_capital.value():.2f}")
            
            # 这里需要连接到实际的回测引擎
            # 但在演示版本中，我们创建模拟数据
            self._create_sample_results()
            
            # 更新进度
            for i in range(1, 101):
                # 在实际应用中，这里会根据回测进度更新
                self.progress_bar.setValue(i)
                # 模拟进度 - 在实际应用中移除这行
                QApplication.processEvents()  # 保持界面响应
                
            # 完成回测
            self.status_label.setText("回测完成")
            self._log("回测完成")
            
            # 显示结果
            self._update_results()
            
        except Exception as e:
            logger.error(f"回测过程中出错: {str(e)}")
            self._log(f"错误: {str(e)}")
            self.status_label.setText("回测出错")
        finally:
            # 恢复按钮状态
            self.backtest_button.setEnabled(True)
            self.optimize_button.setEnabled(True)
            self.export_button.setEnabled(True)
            
    def _optimize_strategy(self):
        """优化策略参数"""
        # 在实际应用中连接到策略优化引擎
        self._log("开始优化策略参数...")
        # 这里可以添加参数网格搜索或遗传算法优化策略参数
        
    def _export_results(self):
        """导出回测结果"""
        # 在实际应用中实现导出功能
        self._log("导出回测结果...")
        
    def _update_results(self):
        """更新回测结果显示"""
        if not self.backtest_results:
            return
            
        # 更新绩效图表
        self.performance_chart.set_data(self.backtest_results)
        
        # 更新结果摘要
        metrics = self.backtest_results.get('metrics', {})
        for key, (label, fmt) in self.metric_labels.items():
            if key in metrics:
                label.setText(fmt.format(metrics[key]))
            else:
                label.setText("--")
                
        # 更新交易明细表格
        trades = self.backtest_results.get('trades', [])
        self.trades_table.setRowCount(len(trades))
        
        for i, trade in enumerate(trades):
            # 开仓日期
            self.trades_table.setItem(i, 0, QTableWidgetItem(trade.get('entry_date', '')))
            
            # 开仓价
            entry_price = QTableWidgetItem(f"{trade.get('entry_price', 0):.2f}")
            self.trades_table.setItem(i, 1, entry_price)
            
            # 平仓日期
            self.trades_table.setItem(i, 2, QTableWidgetItem(trade.get('exit_date', '')))
            
            # 平仓价
            exit_price = QTableWidgetItem(f"{trade.get('exit_price', 0):.2f}")
            self.trades_table.setItem(i, 3, exit_price)
            
            # 方向
            direction = "做多" if trade.get('direction', '') == 'long' else "做空"
            self.trades_table.setItem(i, 4, QTableWidgetItem(direction))
            
            # 持仓天数
            hold_days = QTableWidgetItem(f"{trade.get('hold_days', 0)}")
            self.trades_table.setItem(i, 5, hold_days)
            
            # 收益率
            profit = trade.get('profit', 0)
            profit_item = QTableWidgetItem(f"{profit:.2%}")
            if profit > 0:
                profit_item.setForeground(QBrush(QColor('#4CAF50')))
            elif profit < 0:
                profit_item.setForeground(QBrush(QColor('#F44336')))
            self.trades_table.setItem(i, 6, profit_item)
            
    def _log(self, message):
        """添加日志"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.log_text.append(f"[{timestamp}] {message}")
        
    def _create_sample_results(self):
        """创建示例回测结果数据（仅用于演示）"""
        # 模拟回测日期
        start_date = self.start_date.date().toPyDate()
        end_date = self.end_date.date().toPyDate()
        
        dates = []
        current_date = start_date
        while current_date <= end_date:
            if current_date.weekday() < 5:  # 只包括工作日
                dates.append(current_date.strftime('%Y-%m-%d'))
            current_date += timedelta(days=1)
            
        # 模拟回测数据
        np.random.seed(42)  # 固定随机数种子，确保结果可重现
        
        # 模拟策略收益
        strategy_daily_returns = np.random.normal(0.001, 0.01, len(dates))  # 均值0.1%，标准差1%
        benchmark_daily_returns = np.random.normal(0.0005, 0.008, len(dates))  # 均值0.05%，标准差0.8%
        
        # 计算累积收益
        strategy_returns = []
        benchmark_returns = []
        
        strategy_equity = 1.0
        benchmark_equity = 1.0
        
        for sr, br in zip(strategy_daily_returns, benchmark_daily_returns):
            strategy_equity *= (1 + sr)
            benchmark_equity *= (1 + br)
            
            strategy_returns.append(strategy_equity - 1)
            benchmark_returns.append(benchmark_equity - 1)
            
        # 模拟交易记录
        num_trades = np.random.randint(15, 30)  # 随机15-30笔交易
        trades = []
        
        available_dates = [datetime.strptime(d, '%Y-%m-%d') for d in dates]
        
        for _ in range(num_trades):
            # 随机选择开仓日期
            entry_idx = np.random.randint(0, len(available_dates) - 10)
            entry_date = available_dates[entry_idx]
            
            # 随机选择平仓日期（在开仓之后）
            exit_idx = np.random.randint(entry_idx + 3, min(entry_idx + 15, len(available_dates)))
            exit_date = available_dates[exit_idx]
            
            # 随机价格
            entry_price = np.random.uniform(10, 50)
            
            # 随机方向
            direction = 'long' if np.random.random() > 0.2 else 'short'  # 80%做多
            
            # 根据方向计算盈亏
            if direction == 'long':
                # 多头收益 = 出场价 / 入场价 - 1
                profit_factor = np.random.normal(1.05, 0.2)  # 平均5%收益
                exit_price = entry_price * profit_factor
                profit = profit_factor - 1
            else:
                # 空头收益 = 1 - 出场价 / 入场价
                profit_factor = np.random.normal(0.95, 0.2)  # 平均5%收益
                exit_price = entry_price * profit_factor
                profit = 1 - profit_factor
                
            # 计算持仓天数
            hold_days = (exit_date - entry_date).days
            
            trades.append({
                'entry_date': entry_date.strftime('%Y-%m-%d'),
                'entry_price': entry_price,
                'exit_date': exit_date.strftime('%Y-%m-%d'),
                'exit_price': exit_price,
                'direction': direction,
                'hold_days': hold_days,
                'profit': profit
            })
            
        # 计算绩效指标
        total_return = strategy_returns[-1]
        days_count = len(dates)
        trading_days_per_year = 252
        years = days_count / trading_days_per_year
        
        annual_return = (1 + total_return) ** (1 / years) - 1
        
        # 计算最大回撤
        max_drawdown = 0
        peak = 0
        
        for value in strategy_returns:
            peak = max(peak, value)
            drawdown = (peak - value) / (1 + peak)
            max_drawdown = max(max_drawdown, drawdown)
            
        # 计算夏普比率
        daily_excess_returns = strategy_daily_returns - 0.0001  # 减去无风险利率(日)
        sharpe_ratio = np.mean(daily_excess_returns) / np.std(daily_excess_returns) * np.sqrt(252)
        
        # 计算胜率
        winning_trades = sum(1 for trade in trades if trade['profit'] > 0)
        win_rate = winning_trades / len(trades) if trades else 0
        
        # 计算盈亏比
        if trades:
            total_profits = sum(trade['profit'] for trade in trades if trade['profit'] > 0)
            total_losses = sum(abs(trade['profit']) for trade in trades if trade['profit'] < 0)
            profit_factor = total_profits / total_losses if total_losses > 0 else float('inf')
        else:
            profit_factor = 0
        
        # 组合回测结果
        self.backtest_results = {
            'dates': dates,
            'strategy_returns': strategy_returns,
            'benchmark_returns': benchmark_returns,
            'trades': trades,
            'metrics': {
                'final_equity': self.initial_capital.value() * (1 + total_return),
                'total_return': total_return,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'trade_count': len(trades)
            }
        }
        
    def on_system_started(self):
        """系统启动时调用"""
        logger.info("回测面板：系统已启动")
        
    def on_system_stopped(self):
        """系统停止时调用"""
        logger.info("回测面板：系统已停止") 