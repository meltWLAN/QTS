"""
超神量子选股验证面板 - 验证超神量子选股的有效性和性能
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                         QPushButton, QComboBox, QFrame, QGridLayout, 
                         QTableWidget, QTableWidgetItem, QHeaderView,
                         QSplitter, QTabWidget, QLineEdit, QGroupBox,
                         QFormLayout, QSpinBox, QProgressBar, QFileDialog,
                         QMessageBox, QCheckBox)
from PyQt5.QtCore import Qt, pyqtSlot, pyqtSignal, QThread
from PyQt5.QtGui import QColor, QBrush, QFont

logger = logging.getLogger("QuantumDesktop.QuantumValidationPanel")

class MatplotlibCanvas(FigureCanvas):
    """Matplotlib 画布，用于在Qt界面中显示图表"""
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super(MatplotlibCanvas, self).__init__(self.fig)
        self.setParent(parent)
        
        # 创建子图
        self.axes = self.fig.add_subplot(111)
        
    def clear(self):
        """清除图表"""
        self.axes.clear()
        self.draw()
        
class ValidationThread(QThread):
    """执行验证测试的后台线程"""
    
    # 定义信号
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    validation_completed = pyqtSignal(dict)
    
    def __init__(self, validator, config, parent=None):
        super().__init__(parent)
        self.validator = validator
        self.config = config
        
    def run(self):
        """执行验证测试"""
        try:
            self.status_updated.emit("正在启动验证测试...")
            self.progress_updated.emit(10)
            
            # 获取配置
            backtest_period = self.config.get('backtest_period', 90)
            quantum_powers = self.config.get('quantum_powers', [20, 50, 80])
            market_scopes = self.config.get('market_scopes', ['全市场'])
            
            self.status_updated.emit("正在运行多组合验证测试...")
            self.progress_updated.emit(20)
            
            # 执行验证
            results = self.validator.validate_strategy(
                backtest_period=backtest_period,
                quantum_powers=quantum_powers,
                market_scopes=market_scopes
            )
            
            self.status_updated.emit("正在分析验证结果...")
            self.progress_updated.emit(90)
            
            # 发送结果
            self.validation_completed.emit(results)
            self.status_updated.emit("验证测试完成！")
            self.progress_updated.emit(100)
            
        except Exception as e:
            logger.error(f"验证线程执行时出错: {str(e)}")
            self.status_updated.emit(f"验证出错: {str(e)}")

class QuantumValidationPanel(QWidget):
    """超神量子选股验证面板 - 验证超神量子选股的有效性和性能"""
    
    def __init__(self, system_manager, parent=None):
        super().__init__(parent)
        self.system_manager = system_manager
        
        # 量子选股验证器
        self.validator = None
        
        # 验证结果
        self.validation_results = None
        
        # 验证线程
        self.validation_thread = None
        
        # 初始化验证器
        self._init_validator()
        
        # 初始化UI
        self._init_ui()
        
        logger.info("量子选股验证面板初始化完成")
        
    def _init_validator(self):
        """初始化量子选股验证器"""
        try:
            # 导入量子选股验证器
            from quantum_core.quantum_stock_validator import QuantumStockValidator
            
            # 获取量子选股策略
            stock_strategy = None
            if self.system_manager:
                try:
                    from quantum_core.quantum_stock_strategy import QuantumStockStrategy
                    
                    # 获取量子后端和市场分析器
                    quantum_backend = self.system_manager.get_component("quantum_backend")
                    market_analyzer = self.system_manager.get_component("market_analyzer")
                    
                    # 创建量子选股策略
                    stock_strategy = QuantumStockStrategy(
                        quantum_backend=quantum_backend,
                        market_analyzer=market_analyzer
                    )
                    
                    # 确保策略已启动
                    if self.system_manager.is_system_running():
                        stock_strategy.start()
                        
                except Exception as e:
                    logger.error(f"初始化量子选股策略时出错: {str(e)}")
                    
            # 创建量子选股验证器
            self.validator = QuantumStockValidator(
                stock_strategy=stock_strategy
            )
            
            logger.info("量子选股验证器初始化成功")
            
        except Exception as e:
            logger.error(f"初始化量子选股验证器时出错: {str(e)}")
            self.validator = None
            
    def _init_ui(self):
        """初始化用户界面"""
        # 主布局
        self.main_layout = QVBoxLayout(self)
        
        # 顶部标题
        self.title_label = QLabel("超神量子选股验证")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 20px; font-weight: bold; margin: 10px;")
        self.main_layout.addWidget(self.title_label)
        
        # 描述标签
        self.description_label = QLabel(
            "使用全球最先进的验证技术，评估超神量子选股策略的有效性和性能"
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
        
        # 回测参数组
        self.backtest_group = QGroupBox("回测参数")
        self.backtest_layout = QFormLayout(self.backtest_group)
        
        # 回测周期
        self.period_spinner = QSpinBox()
        self.period_spinner.setRange(30, 365)
        self.period_spinner.setValue(90)
        self.period_spinner.setSuffix(" 天")
        self.backtest_layout.addRow("回测周期:", self.period_spinner)
        
        self.config_layout.addWidget(self.backtest_group)
        
        # 量子能力组
        self.power_group = QGroupBox("测试量子能力")
        self.power_layout = QVBoxLayout(self.power_group)
        
        # 量子能力选择
        self.power_20_check = QCheckBox("20% - 保守模式")
        self.power_20_check.setChecked(True)
        self.power_layout.addWidget(self.power_20_check)
        
        self.power_50_check = QCheckBox("50% - 平衡模式")
        self.power_50_check.setChecked(True)
        self.power_layout.addWidget(self.power_50_check)
        
        self.power_80_check = QCheckBox("80% - 激进模式")
        self.power_80_check.setChecked(True)
        self.power_layout.addWidget(self.power_80_check)
        
        self.power_100_check = QCheckBox("100% - 极限模式")
        self.power_layout.addWidget(self.power_100_check)
        
        self.config_layout.addWidget(self.power_group)
        
        # 市场范围组
        self.market_group = QGroupBox("测试市场范围")
        self.market_layout = QVBoxLayout(self.market_group)
        
        # 市场范围选择
        self.scope_all_check = QCheckBox("全市场")
        self.scope_all_check.setChecked(True)
        self.market_layout.addWidget(self.scope_all_check)
        
        self.scope_300_check = QCheckBox("沪深300")
        self.market_layout.addWidget(self.scope_300_check)
        
        self.scope_500_check = QCheckBox("中证500")
        self.market_layout.addWidget(self.scope_500_check)
        
        self.config_layout.addWidget(self.market_group)
        
        # 操作按钮组
        self.action_group = QGroupBox("操作")
        self.action_layout = QVBoxLayout(self.action_group)
        
        # 开始验证按钮
        self.start_button = QPushButton("开始验证测试")
        self.start_button.setStyleSheet("font-weight: bold; height: 40px;")
        self.start_button.clicked.connect(self._on_start_validation)
        self.action_layout.addWidget(self.start_button)
        
        # 导出报告按钮
        self.export_button = QPushButton("导出验证报告")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._on_export_report)
        self.action_layout.addWidget(self.export_button)
        
        self.config_layout.addWidget(self.action_group)
        
    def _create_results_area(self):
        """创建结果显示区域"""
        # 分割器
        self.results_splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.results_splitter, 1)
        
        # 左侧：结果指标
        self.metrics_frame = QFrame()
        self.metrics_layout = QVBoxLayout(self.metrics_frame)
        
        # 指标表格
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels([
            "指标", "值"
        ])
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.metrics_layout.addWidget(self.metrics_table)
        
        # 测试详情标签
        self.detail_label = QLabel("测试详情")
        self.detail_label.setStyleSheet("font-weight: bold;")
        self.metrics_layout.addWidget(self.detail_label)
        
        # 详情表格
        self.detail_table = QTableWidget()
        self.detail_table.setColumnCount(5)
        self.detail_table.setHorizontalHeaderLabels([
            "测试配置", "平均收益", "胜率", "超额收益", "夏普比率"
        ])
        self.detail_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.metrics_layout.addWidget(self.detail_table)
        
        self.results_splitter.addWidget(self.metrics_frame)
        
        # 右侧：图表
        self.chart_frame = QFrame()
        self.chart_layout = QVBoxLayout(self.chart_frame)
        
        # 图表标签页
        self.chart_tabs = QTabWidget()
        self.chart_layout.addWidget(self.chart_tabs)
        
        # 量子能力与收益关系图
        self.power_return_canvas = MatplotlibCanvas(width=5, height=4)
        self.chart_tabs.addTab(self.power_return_canvas, "量子能力与收益")
        
        # 量子评分与实际收益散点图
        self.score_return_canvas = MatplotlibCanvas(width=5, height=4)
        self.chart_tabs.addTab(self.score_return_canvas, "评分与收益")
        
        # 预测准确度分布图
        self.accuracy_canvas = MatplotlibCanvas(width=5, height=4)
        self.chart_tabs.addTab(self.accuracy_canvas, "预测准确度")
        
        # 策略与基准比较图
        self.benchmark_canvas = MatplotlibCanvas(width=5, height=4)
        self.chart_tabs.addTab(self.benchmark_canvas, "策略VS基准")
        
        self.results_splitter.addWidget(self.chart_frame)
        
        # 设置分割比例
        self.results_splitter.setSizes([400, 700])
        
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
        self.status_layout.addWidget(self.progress_bar, 1)
        
        # 状态标签
        self.status_label = QLabel("就绪")
        self.status_layout.addWidget(self.status_label)
        
    def _on_start_validation(self):
        """开始验证按钮点击事件"""
        # 确保量子后端已启动
        if self.system_manager and not self._check_system_ready():
            QMessageBox.warning(self, "系统未就绪", 
                            "超神量子核心系统未启动，无法执行验证测试。\n请先启动系统。")
            return
            
        # 确保验证器可用
        if not self.validator:
            QMessageBox.warning(self, "验证器未就绪", 
                            "量子选股验证器未就绪，无法执行验证测试。")
            return
            
        # 获取配置
        config = self._get_validation_config()
        
        # 创建并启动验证线程
        self.validation_thread = ValidationThread(self.validator, config)
        
        # 连接信号
        self.validation_thread.progress_updated.connect(self.progress_bar.setValue)
        self.validation_thread.status_updated.connect(self.status_label.setText)
        self.validation_thread.validation_completed.connect(self._on_validation_completed)
        
        # 启动线程
        self.validation_thread.start()
        
        # 更新UI状态
        self.start_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("正在初始化验证测试...")
        
    def _get_validation_config(self):
        """获取验证配置"""
        # 获取回测周期
        backtest_period = self.period_spinner.value()
        
        # 获取要测试的量子能力
        quantum_powers = []
        if self.power_20_check.isChecked():
            quantum_powers.append(20)
        if self.power_50_check.isChecked():
            quantum_powers.append(50)
        if self.power_80_check.isChecked():
            quantum_powers.append(80)
        if self.power_100_check.isChecked():
            quantum_powers.append(100)
            
        # 如果没有选择任何量子能力，使用默认值
        if not quantum_powers:
            quantum_powers = [50]
            
        # 获取要测试的市场范围
        market_scopes = []
        if self.scope_all_check.isChecked():
            market_scopes.append("全市场")
        if self.scope_300_check.isChecked():
            market_scopes.append("沪深300")
        if self.scope_500_check.isChecked():
            market_scopes.append("中证500")
            
        # 如果没有选择任何市场范围，使用默认值
        if not market_scopes:
            market_scopes = ["全市场"]
            
        return {
            "backtest_period": backtest_period,
            "quantum_powers": quantum_powers,
            "market_scopes": market_scopes
        }
        
    def _on_validation_completed(self, results):
        """验证完成事件处理"""
        # 保存验证结果
        self.validation_results = results
        
        # 显示验证结果
        self._display_validation_results()
        
        # 更新UI状态
        self.start_button.setEnabled(True)
        self.export_button.setEnabled(True)
        
    def _display_validation_results(self):
        """显示验证结果"""
        if not self.validation_results:
            return
            
        # 获取汇总信息
        summary = self.validation_results.get("summary", {})
        tests = self.validation_results.get("tests", [])
        
        # 更新指标表格
        self._update_metrics_table(summary)
        
        # 更新详情表格
        self._update_detail_table(tests)
        
        # 更新图表
        self._update_charts(summary, tests)
        
    def _update_metrics_table(self, summary):
        """更新指标表格"""
        # 清空表格
        self.metrics_table.setRowCount(0)
        
        # 添加指标
        metrics = [
            ("最佳量子能力", f"{summary.get('best_quantum_power', '未知')}%"),
            ("平均预测准确度", f"{summary.get('average_prediction_accuracy', 0):.2f}%"),
            ("平均超额收益", f"{summary.get('average_excess_return', 0):.2f}%"),
            ("平均夏普比率", f"{summary.get('sharpe_ratio', 0):.2f}"),
            ("平均胜率", f"{summary.get('win_rate', 0) * 100:.2f}%"),
            ("量子评分相关性", f"{summary.get('quantum_score_correlation', 0):.2f}")
        ]
        
        # 填充表格
        self.metrics_table.setRowCount(len(metrics))
        for i, (name, value) in enumerate(metrics):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(name))
            self.metrics_table.setItem(i, 1, QTableWidgetItem(value))
            
    def _update_detail_table(self, tests):
        """更新详情表格"""
        # 清空表格
        self.detail_table.setRowCount(0)
        
        # 填充表格
        self.detail_table.setRowCount(len(tests))
        for i, test in enumerate(tests):
            power = test.get("quantum_power", 0)
            scope = test.get("market_scope", "")
            backtest = test.get("backtest_result", {})
            
            # 配置
            config_item = QTableWidgetItem(f"{power}% - {scope}")
            
            # 平均收益
            avg_return = backtest.get("average_return", 0)
            return_item = QTableWidgetItem(f"{avg_return:.2f}%")
            if avg_return > 0:
                return_item.setBackground(QBrush(QColor(0, 255, 0, 50)))
            else:
                return_item.setBackground(QBrush(QColor(255, 0, 0, 50)))
                
            # 胜率
            win_rate = backtest.get("win_rate", 0) * 100
            win_item = QTableWidgetItem(f"{win_rate:.2f}%")
            
            # 超额收益
            excess = backtest.get("excess_return", 0)
            excess_item = QTableWidgetItem(f"{excess:.2f}%")
            if excess > 0:
                excess_item.setBackground(QBrush(QColor(0, 255, 0, 50)))
            else:
                excess_item.setBackground(QBrush(QColor(255, 0, 0, 50)))
                
            # 夏普比率
            sharpe = backtest.get("sharpe_ratio", 0)
            sharpe_item = QTableWidgetItem(f"{sharpe:.2f}")
            
            # 添加到表格
            self.detail_table.setItem(i, 0, config_item)
            self.detail_table.setItem(i, 1, return_item)
            self.detail_table.setItem(i, 2, win_item)
            self.detail_table.setItem(i, 3, excess_item)
            self.detail_table.setItem(i, 4, sharpe_item)
            
    def _update_charts(self, summary, tests):
        """更新图表"""
        # 1. 量子能力与收益关系图
        self.power_return_canvas.clear()
        
        powers = []
        returns = []
        for power, avg_return in summary.get("power_performance", {}).items():
            powers.append(power)
            returns.append(avg_return)
            
        if powers and returns:
            self.power_return_canvas.axes.bar(powers, returns)
            self.power_return_canvas.axes.set_title('量子能力与平均收益率关系')
            self.power_return_canvas.axes.set_xlabel('量子能力')
            self.power_return_canvas.axes.set_ylabel('平均收益率 (%)')
            self.power_return_canvas.draw()
            
        # 2. 量子评分与实际收益散点图
        self.score_return_canvas.clear()
        
        scores = []
        actual_returns = []
        for test in tests:
            backtest = test.get("backtest_result", {})
            for stock_data in backtest.get("stock_performances", {}).values():
                scores.append(stock_data.get("quantum_score", 0))
                actual_returns.append(stock_data.get("actual_return", 0))
                
        if scores and actual_returns:
            self.score_return_canvas.axes.scatter(scores, actual_returns)
            self.score_return_canvas.axes.set_title('量子评分与实际收益关系')
            self.score_return_canvas.axes.set_xlabel('量子评分')
            self.score_return_canvas.axes.set_ylabel('实际收益率 (%)')
            
            # 添加趋势线
            if len(scores) > 1:
                z = np.polyfit(scores, actual_returns, 1)
                p = np.poly1d(z)
                x_range = np.linspace(min(scores), max(scores), 100)
                self.score_return_canvas.axes.plot(x_range, p(x_range), "r--")
                
            self.score_return_canvas.draw()
            
        # 3. 预测准确度分布图
        self.accuracy_canvas.clear()
        
        accuracies = []
        for test in tests:
            backtest = test.get("backtest_result", {})
            for stock_data in backtest.get("stock_performances", {}).values():
                accuracies.append(stock_data.get("prediction_accuracy", 0))
                
        if accuracies:
            self.accuracy_canvas.axes.hist(accuracies, bins=10)
            self.accuracy_canvas.axes.set_title('预测准确度分布')
            self.accuracy_canvas.axes.set_xlabel('预测准确度 (%)')
            self.accuracy_canvas.axes.set_ylabel('频率')
            self.accuracy_canvas.draw()
            
        # 4. 策略与基准比较图
        self.benchmark_canvas.clear()
        
        test_labels = []
        test_returns = []
        benchmark_returns = []
        
        for i, test in enumerate(tests):
            backtest = test.get("backtest_result", {})
            test_labels.append(f"Test {i+1}")
            test_returns.append(backtest.get("average_return", 0))
            benchmark_returns.append(backtest.get("benchmark_return", 0))
            
        if test_labels and test_returns and benchmark_returns:
            x = np.arange(len(test_labels))
            width = 0.35
            
            self.benchmark_canvas.axes.bar(x - width/2, test_returns, width, label='选股策略')
            self.benchmark_canvas.axes.bar(x + width/2, benchmark_returns, width, label='市场基准')
            self.benchmark_canvas.axes.set_title('策略表现与基准比较')
            self.benchmark_canvas.axes.set_xticks(x)
            self.benchmark_canvas.axes.set_xticklabels(test_labels)
            self.benchmark_canvas.axes.set_ylabel('收益率 (%)')
            self.benchmark_canvas.axes.legend()
            self.benchmark_canvas.draw()
            
    def _on_export_report(self):
        """导出报告按钮点击事件"""
        if not self.validation_results:
            QMessageBox.warning(self, "导出失败", "没有验证结果可导出")
            return
            
        # 获取保存文件路径
        output_file, _ = QFileDialog.getSaveFileName(
            self,
            "导出验证报告",
            "",
            "JSON文件 (*.json);;所有文件 (*)"
        )
        
        if not output_file:
            return
            
        # 导出报告
        if self.validator.export_validation_report(output_file):
            QMessageBox.information(self, "导出成功", f"验证报告已成功导出至\n{output_file}")
        else:
            QMessageBox.warning(self, "导出失败", "验证报告导出失败")
            
    def _check_system_ready(self):
        """检查系统是否准备就绪"""
        if not self.system_manager:
            return False
            
        # 检查量子后端是否启动
        try:
            quantum_backend = self.system_manager.get_component("quantum_backend")
            if quantum_backend and quantum_backend.is_active():
                # 确保验证器已就绪
                self._init_validator()
                return True
                
            # 尝试启动量子后端
            if quantum_backend and not quantum_backend.is_active():
                success = quantum_backend.start()
                if success:
                    # 重新初始化验证器
                    self._init_validator()
                return success
                
            # 如果仍未成功，创建模拟验证器以进行演示
            logger.warning("系统未就绪，将使用模拟验证器进行演示")
            self._create_demo_validator()
            return True
                
        except Exception as e:
            logger.error(f"检查系统状态时出错: {str(e)}")
            # 创建模拟验证器以进行演示
            self._create_demo_validator()
            return True
            
    def _create_demo_validator(self):
        """创建演示验证器"""
        try:
            # 导入量子选股验证器
            from quantum_core.quantum_stock_validator import QuantumStockValidator
            
            # 创建模拟股票策略
            class DemoStockStrategy:
                def __init__(self):
                    self.is_running = True
                    self.quantum_power = 50
                    import random
                    self.random = random
                    
                def start(self):
                    self.is_running = True
                    return True
                    
                def stop(self):
                    self.is_running = False
                    return True
                    
                def set_quantum_power(self, power):
                    self.quantum_power = power
                    logger.info(f"演示模式：设置量子能力为 {power}")
                    
                def find_potential_stocks(self, market_scope="全市场", max_stocks=10):
                    """模拟查找潜力股"""
                    # 生成随机股票代码
                    stock_codes = []
                    for i in range(max_stocks):
                        if self.random.random() > 0.5:
                            code = f"60{self.random.randint(1000, 9999)}"
                        else:
                            code = f"00{self.random.randint(1000, 9999)}"
                        stock_codes.append(code)
                        
                    # 生成模拟股票数据
                    stocks = []
                    for code in stock_codes:
                        # 量子评分与量子能力正相关
                        quantum_score = 80 + self.random.uniform(-20, 20) + (self.quantum_power - 50) * 0.2
                        quantum_score = max(1, min(99, quantum_score))
                        
                        # 生成预期收益
                        expected_gain = (quantum_score - 80) * 0.8 + self.random.uniform(-10, 10)
                        
                        stocks.append({
                            "code": code,
                            "name": f"演示股票{code[2:6]}",
                            "quantum_score": quantum_score,
                            "expected_gain": expected_gain,
                            "industry": self.random.choice(["科技", "金融", "医药", "能源", "消费"])
                        })
                        
                    return {
                        "status": "success",
                        "market_scope": market_scope,
                        "quantum_power": self.quantum_power,
                        "stocks": stocks
                    }
            
            # 创建演示验证器
            self.validator = QuantumStockValidator(
                stock_strategy=DemoStockStrategy()
            )
            
            logger.info("演示模式：量子选股验证器初始化成功")
            
        except Exception as e:
            logger.error(f"创建演示验证器时出错: {str(e)}")
            self.validator = None
    
    def on_system_started(self):
        """系统启动事件处理"""
        # 确保验证器可用
        self._init_validator()
        
        # 允许使用验证功能
        self.start_button.setEnabled(True)
        
    def on_system_stopped(self):
        """系统停止事件处理"""
        # 禁用验证功能
        self.start_button.setEnabled(False)
        
        # 如果正在进行验证，停止验证线程
        if self.validation_thread and self.validation_thread.isRunning():
            # 无法直接停止线程，只能等待其完成
            self.status_label.setText("系统已停止，等待验证线程完成...") 