#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - 量子预测视图
展示股票预测和市场洞察的高级界面
"""

import os
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QComboBox, QLineEdit, QTableWidget, QTableWidgetItem, 
    QHeaderView, QSplitter, QFrame, QGridLayout, QTabWidget,
    QProgressBar, QSlider, QGroupBox, QSizePolicy, QScrollArea
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QColor, QPixmap, QIcon

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False    # 正确显示负号
except:
    pass


class MatplotlibCanvas(FigureCanvas):
    """Matplotlib画布"""
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        """初始化画布"""
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        
        FigureCanvas.setSizePolicy(self, 
                                  QSizePolicy.Expanding, 
                                  QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
    
    def clear(self):
        """清除图形"""
        self.axes.clear()
        self.draw()


class QuantumPredictionControl(QWidget):
    """量子预测控制面板"""
    
    # 信号定义
    prediction_requested = pyqtSignal(str, int)
    market_insights_requested = pyqtSignal()
    quantum_params_changed = pyqtSignal(float, float, float)
    
    def __init__(self, parent=None):
        """初始化控制面板"""
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        """设置UI"""
        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # 股票选择区域
        stock_select_group = QGroupBox("股票预测设置")
        stock_select_layout = QGridLayout(stock_select_group)
        
        # 股票代码输入
        self.stock_label = QLabel("股票代码:")
        self.stock_input = QLineEdit()
        self.stock_input.setPlaceholderText("输入股票代码 如:000001")
        stock_select_layout.addWidget(self.stock_label, 0, 0)
        stock_select_layout.addWidget(self.stock_input, 0, 1, 1, 2)
        
        # 预测天数选择
        self.days_label = QLabel("预测天数:")
        self.days_combo = QComboBox()
        for days in [5, 10, 15, 20, 30]:
            self.days_combo.addItem(f"{days}天", days)
        stock_select_layout.addWidget(self.days_label, 1, 0)
        stock_select_layout.addWidget(self.days_combo, 1, 1)
        
        # 预测按钮
        self.predict_button = QPushButton("开始预测")
        self.predict_button.setStyleSheet("background-color: #4CAF50; color: white;")
        self.predict_button.clicked.connect(self.request_prediction)
        stock_select_layout.addWidget(self.predict_button, 1, 2)
        
        # 市场洞察按钮
        self.market_insights_button = QPushButton("生成市场洞察")
        self.market_insights_button.setStyleSheet("background-color: #2196F3; color: white;")
        self.market_insights_button.clicked.connect(self.request_market_insights)
        stock_select_layout.addWidget(self.market_insights_button, 2, 0, 1, 3)
        
        main_layout.addWidget(stock_select_group)
        
        # 量子参数控制区域
        quantum_param_group = QGroupBox("量子参数调节")
        quantum_param_layout = QGridLayout(quantum_param_group)
        
        # 相干性参数
        self.coherence_label = QLabel("量子相干性:")
        self.coherence_slider = QSlider(Qt.Horizontal)
        self.coherence_slider.setRange(0, 100)
        self.coherence_slider.setValue(70)  # 默认0.7
        self.coherence_value = QLabel("0.70")
        self.coherence_slider.valueChanged.connect(self.update_coherence_value)
        quantum_param_layout.addWidget(self.coherence_label, 0, 0)
        quantum_param_layout.addWidget(self.coherence_slider, 0, 1)
        quantum_param_layout.addWidget(self.coherence_value, 0, 2)
        
        # 量子叠加态参数
        self.superposition_label = QLabel("量子叠加态:")
        self.superposition_slider = QSlider(Qt.Horizontal)
        self.superposition_slider.setRange(0, 100)
        self.superposition_slider.setValue(50)  # 默认0.5
        self.superposition_value = QLabel("0.50")
        self.superposition_slider.valueChanged.connect(self.update_superposition_value)
        quantum_param_layout.addWidget(self.superposition_label, 1, 0)
        quantum_param_layout.addWidget(self.superposition_slider, 1, 1)
        quantum_param_layout.addWidget(self.superposition_value, 1, 2)
        
        # 量子纠缠参数
        self.entanglement_label = QLabel("量子纠缠:")
        self.entanglement_slider = QSlider(Qt.Horizontal)
        self.entanglement_slider.setRange(0, 100)
        self.entanglement_slider.setValue(80)  # 默认0.8
        self.entanglement_value = QLabel("0.80")
        self.entanglement_slider.valueChanged.connect(self.update_entanglement_value)
        quantum_param_layout.addWidget(self.entanglement_label, 2, 0)
        quantum_param_layout.addWidget(self.entanglement_slider, 2, 1)
        quantum_param_layout.addWidget(self.entanglement_value, 2, 2)
        
        # 应用量子参数按钮
        self.apply_quantum_button = QPushButton("应用量子参数")
        self.apply_quantum_button.clicked.connect(self.apply_quantum_params)
        quantum_param_layout.addWidget(self.apply_quantum_button, 3, 0, 1, 3)
        
        main_layout.addWidget(quantum_param_group)
        
        # 预测说明
        explanation_text = (
            "量子预测模型说明:\n"
            "- 相干性: 影响预测的平滑程度，值越高预测越平滑\n"
            "- 叠加态: 影响模型复杂度，值越高模型越复杂\n"
            "- 纠缠: 影响特征关联强度，值越高特征关联越强\n"
            "\n"
            "调整参数后点击「应用量子参数」使其生效"
        )
        self.explanation_label = QLabel(explanation_text)
        self.explanation_label.setWordWrap(True)
        self.explanation_label.setStyleSheet("color: #666; font-size: 11px;")
        main_layout.addWidget(self.explanation_label)
        
        # 弹性空间
        main_layout.addStretch()
    
    def update_coherence_value(self, value):
        """更新相干性参数值"""
        self.coherence_value.setText(f"{value/100:.2f}")
    
    def update_superposition_value(self, value):
        """更新叠加态参数值"""
        self.superposition_value.setText(f"{value/100:.2f}")
    
    def update_entanglement_value(self, value):
        """更新纠缠参数值"""
        self.entanglement_value.setText(f"{value/100:.2f}")
    
    def request_prediction(self):
        """请求股票预测"""
        stock_code = self.stock_input.text().strip()
        if not stock_code:
            return
        
        days = self.days_combo.currentData()
        self.prediction_requested.emit(stock_code, days)
    
    def request_market_insights(self):
        """请求市场洞察"""
        self.market_insights_requested.emit()
    
    def apply_quantum_params(self):
        """应用量子参数"""
        coherence = self.coherence_slider.value() / 100.0
        superposition = self.superposition_slider.value() / 100.0
        entanglement = self.entanglement_slider.value() / 100.0
        
        self.quantum_params_changed.emit(coherence, superposition, entanglement)


class StockPredictionView(QWidget):
    """股票预测视图"""
    
    def __init__(self, parent=None):
        """初始化股票预测视图"""
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        """设置UI"""
        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # 预测结果标签
        self.title_label = QLabel("股票预测结果")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        main_layout.addWidget(self.title_label)
        
        # 预测图表
        self.canvas = MatplotlibCanvas(self, width=8, height=5)
        main_layout.addWidget(self.canvas)
        
        # 预测详情
        self.details_frame = QFrame()
        self.details_frame.setFrameShape(QFrame.StyledPanel)
        self.details_frame.setStyleSheet("background-color: #f8f9fa;")
        
        details_layout = QVBoxLayout(self.details_frame)
        
        # 当前价格和风险评估
        price_risk_layout = QHBoxLayout()
        
        self.current_price_label = QLabel("当前价格: --")
        self.current_price_label.setStyleSheet("font-weight: bold;")
        price_risk_layout.addWidget(self.current_price_label)
        
        price_risk_layout.addStretch()
        
        self.risk_level_label = QLabel("风险等级: --")
        price_risk_layout.addWidget(self.risk_level_label)
        
        details_layout.addLayout(price_risk_layout)
        
        # 预测表格
        self.prediction_table = QTableWidget()
        self.prediction_table.setColumnCount(3)
        self.prediction_table.setHorizontalHeaderLabels(["日期", "预测价格", "变化率"])
        self.prediction_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        details_layout.addWidget(self.prediction_table)
        
        # 交易建议
        self.recommendation_label = QLabel("交易建议: --")
        self.recommendation_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        details_layout.addWidget(self.recommendation_label)
        
        self.reason_label = QLabel("建议理由: --")
        details_layout.addWidget(self.reason_label)
        
        # 预测精度
        accuracy_layout = QGridLayout()
        
        accuracy_layout.addWidget(QLabel("预测精度指标:"), 0, 0)
        self.rmse_label = QLabel("RMSE: --")
        accuracy_layout.addWidget(self.rmse_label, 0, 1)
        self.mae_label = QLabel("MAE: --")
        accuracy_layout.addWidget(self.mae_label, 0, 2)
        self.r2_label = QLabel("R²: --")
        accuracy_layout.addWidget(self.r2_label, 0, 3)
        
        details_layout.addLayout(accuracy_layout)
        
        main_layout.addWidget(self.details_frame)
    
    def display_prediction(self, stock_code, prediction_data):
        """显示预测结果
        
        Args:
            stock_code: 股票代码
            prediction_data: 预测数据字典
        """
        if not prediction_data:
            return
        
        # 更新标题
        self.title_label.setText(f"{stock_code} 股票预测结果")
        
        # 清除旧图像
        self.canvas.axes.clear()
        
        # 绘制预测曲线
        dates = prediction_data.get("prediction_dates", [])
        prices = prediction_data.get("predicted_prices", [])
        upper_bound = prediction_data.get("upper_bound", [])
        lower_bound = prediction_data.get("lower_bound", [])
        current_price = prediction_data.get("current_price", 0)
        
        # 插入当前价格
        all_dates = ["当前"] + dates
        all_prices = [current_price] + prices
        
        # 绘制预测曲线
        self.canvas.axes.plot(all_dates, all_prices, 'b-o', label='预测价格')
        
        # 绘制置信区间
        if upper_bound and lower_bound:
            upper_with_current = [current_price] + upper_bound
            lower_with_current = [current_price] + lower_bound
            
            self.canvas.axes.fill_between(
                all_dates, lower_with_current, upper_with_current,
                alpha=0.2, color='blue', label='预测区间'
            )
        
        # 设置图表属性
        self.canvas.axes.set_title(f"{stock_code} 价格预测")
        self.canvas.axes.set_xlabel("日期")
        self.canvas.axes.set_ylabel("价格")
        self.canvas.axes.grid(True, linestyle='--', alpha=0.7)
        self.canvas.axes.legend()
        
        # 旋转x轴标签
        self.canvas.axes.tick_params(axis='x', rotation=45)
        
        self.canvas.fig.tight_layout()
        self.canvas.draw()
        
        # 更新详情信息
        self.current_price_label.setText(f"当前价格: ¥{current_price:.2f}")
        
        # 风险等级
        risk_level = prediction_data.get("risk_level", "--")
        if risk_level == "高":
            risk_color = "red"
        elif risk_level == "中":
            risk_color = "orange"
        else:
            risk_color = "green"
        self.risk_level_label.setText(f"风险等级: <span style='color:{risk_color};font-weight:bold;'>{risk_level}</span>")
        
        # 更新表格
        self.prediction_table.setRowCount(len(dates))
        price_changes = prediction_data.get("price_changes", [])
        
        for i, (date, price) in enumerate(zip(dates, prices)):
            # 日期
            self.prediction_table.setItem(i, 0, QTableWidgetItem(date))
            
            # 预测价格
            price_item = QTableWidgetItem(f"{price:.2f}")
            self.prediction_table.setItem(i, 1, price_item)
            
            # 变化率
            if i < len(price_changes):
                change = price_changes[i] * 100
                change_item = QTableWidgetItem(f"{change:+.2f}%")
                if change > 0:
                    change_item.setForeground(QColor("green"))
                elif change < 0:
                    change_item.setForeground(QColor("red"))
                self.prediction_table.setItem(i, 2, change_item)
        
        # 更新交易建议
        recommendation = prediction_data.get("recommendation", "--")
        reason = prediction_data.get("reason", "--")
        
        # 设置建议颜色
        if "强烈买入" in recommendation or "买入" in recommendation:
            recommendation_color = "green"
        elif "卖出" in recommendation:
            recommendation_color = "red"
        else:
            recommendation_color = "#2196F3"  # 蓝色
        
        self.recommendation_label.setText(
            f"交易建议: <span style='color:{recommendation_color};'>{recommendation}</span>"
        )
        self.reason_label.setText(f"建议理由: {reason}")
        
        # 更新精度指标
        accuracy = prediction_data.get("accuracy", {})
        self.rmse_label.setText(f"RMSE: {accuracy.get('rmse', 0):.4f}")
        self.mae_label.setText(f"MAE: {accuracy.get('mae', 0):.4f}")
        self.r2_label.setText(f"R²: {accuracy.get('r2', 0):.4f}")

    def update_prediction_chart(self, stock_code, stock_name, prediction_data):
        """更新预测图表
        
        Args:
            stock_code: 股票代码
            stock_name: 股票名称
            prediction_data: 预测数据，包含 'dates' 和 'predictions' 键
        """
        try:
            self.canvas.fig.clear()
            if not prediction_data or 'predictions' not in prediction_data or not prediction_data['predictions']:
                self.logger.warning(f"股票 {stock_code} 预测数据为空")
                return
                
            # 创建绘图区域
            ax = self.canvas.fig.add_subplot(111)
            
            dates = prediction_data.get('dates', [])
            predictions = prediction_data.get('predictions', [])
            
            # 仅用于显示目的
            if len(dates) != len(predictions):
                self.logger.warning(f"日期和预测数据长度不匹配: {len(dates)} vs {len(predictions)}")
                # 使用索引作为日期备用方案
                dates = [f"Day {i+1}" for i in range(len(predictions))]
            
            # 绘制预测线
            ax.plot(dates, predictions, 'r-', marker='o', linewidth=2, label='预测价格')
            
            # 设置图表格式
            ax.set_title(f"{stock_name}（{stock_code}）价格预测", fontsize=14)
            ax.set_xlabel('日期', fontsize=12)
            ax.set_ylabel('价格 (¥)', fontsize=12)
            
            # 设置x轴标签旋转
            ax.tick_params(axis='x', rotation=45)
            
            # 添加网格和图例
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
            # 突出显示价格增长/下降区域
            if len(predictions) > 1:
                start_price = predictions[0]
                end_price = predictions[-1]
                
                # 添加百分比变化注释
                change_pct = (end_price - start_price) / start_price * 100
                change_color = 'green' if change_pct >= 0 else 'red'
                
                ax.annotate(f'总变化: {change_pct:.2f}%', 
                          xy=(len(dates) - 1, predictions[-1]),
                          xytext=(len(dates) - 2, predictions[-1] * 1.05),
                          color=change_color,
                          fontweight='bold')
                
                # 为图表填充颜色
                if change_pct >= 0:
                    ax.fill_between(dates, predictions, min(predictions) * 0.95, 
                                   alpha=0.2, color='green')
                else:
                    ax.fill_between(dates, predictions, max(predictions) * 1.05, 
                                   alpha=0.2, color='red')
                
            # 调整布局
            self.canvas.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            self.logger.error(f"更新预测图表时出错: {str(e)}")


class MarketInsightsView(QWidget):
    """市场洞察视图"""
    
    def __init__(self, parent=None):
        """初始化市场洞察视图"""
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        """设置UI"""
        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # 主标题
        self.title_label = QLabel("量子市场洞察")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        main_layout.addWidget(self.title_label)
        
        # 市场趋势框
        trend_frame = QFrame()
        trend_frame.setFrameShape(QFrame.StyledPanel)
        trend_frame.setStyleSheet("background-color: #f0f4f8; border-radius: 5px;")
        
        trend_layout = QHBoxLayout(trend_frame)
        
        self.trend_label = QLabel("市场趋势: --")
        self.trend_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        trend_layout.addWidget(self.trend_label)
        
        self.trend_change_label = QLabel("变化率: --")
        trend_layout.addWidget(self.trend_change_label)
        
        self.quantum_confidence_label = QLabel("量子信任度: --")
        self.quantum_confidence_label.setStyleSheet("color: #673AB7;")
        trend_layout.addWidget(self.quantum_confidence_label)
        
        main_layout.addWidget(trend_frame)
        
        # 分割区域为左右两部分
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧: 热门行业
        sectors_widget = QWidget()
        sectors_layout = QVBoxLayout(sectors_widget)
        sectors_layout.setContentsMargins(0, 0, 0, 0)
        
        sectors_title = QLabel("热门行业")
        sectors_title.setStyleSheet("font-weight: bold; margin-bottom: 5px;")
        sectors_layout.addWidget(sectors_title)
        
        self.sectors_table = QTableWidget()
        self.sectors_table.setColumnCount(2)
        self.sectors_table.setHorizontalHeaderLabels(["行业", "预期涨幅"])
        self.sectors_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        sectors_layout.addWidget(self.sectors_table)
        
        splitter.addWidget(sectors_widget)
        
        # 右侧: 推荐股票
        recommendations_widget = QWidget()
        recommendations_layout = QVBoxLayout(recommendations_widget)
        recommendations_layout.setContentsMargins(0, 0, 0, 0)
        
        recommendations_title = QLabel("量子推荐股票")
        recommendations_title.setStyleSheet("font-weight: bold; margin-bottom: 5px;")
        recommendations_layout.addWidget(recommendations_title)
        
        self.recommendations_table = QTableWidget()
        self.recommendations_table.setColumnCount(5)
        self.recommendations_table.setHorizontalHeaderLabels(["排名", "代码", "名称", "预期涨幅", "推荐建议"])
        self.recommendations_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        recommendations_layout.addWidget(self.recommendations_table)
        
        splitter.addWidget(recommendations_widget)
        
        # 设置分割比例
        splitter.setSizes([400, 600])
        
        main_layout.addWidget(splitter)
        
        # 添加说明标签
        note_label = QLabel(
            "注意: 市场洞察基于量子共生算法生成，仅供参考，不构成投资建议。"
            "实际投资决策需考虑个人风险承受能力与市场实际情况。"
        )
        note_label.setWordWrap(True)
        note_label.setStyleSheet("color: #666; font-size: 11px; margin-top: 10px;")
        main_layout.addWidget(note_label)
    
    def display_insights(self, insights_data):
        """显示市场洞察
        
        Args:
            insights_data: 市场洞察数据字典
        """
        if not insights_data:
            return
        
        # 更新市场趋势
        trend = insights_data.get("market_trend", "--")
        avg_change = insights_data.get("avg_market_change", 0) * 100
        
        # 设置趋势颜色
        if "上涨" in trend:
            trend_color = "green"
        elif "下跌" in trend:
            trend_color = "red"
        else:
            trend_color = "#673AB7"  # 紫色
        
        self.trend_label.setText(
            f"市场趋势: <span style='color:{trend_color};'>{trend}</span>"
        )
        
        # 设置变化颜色
        if avg_change > 0:
            change_color = "green"
            change_sign = "+"
        elif avg_change < 0:
            change_color = "red"
            change_sign = ""
        else:
            change_color = "black"
            change_sign = ""
        
        self.trend_change_label.setText(
            f"变化率: <span style='color:{change_color};'>{change_sign}{avg_change:.2f}%</span>"
        )
        
        # 量子信任度
        confidence = insights_data.get("quantum_confidence", 0)
        confidence_percent = min(100, int(confidence * 100))
        self.quantum_confidence_label.setText(f"量子信任度: {confidence_percent}%")
        
        # 更新热门行业
        hot_sectors = insights_data.get("hot_sectors", [])
        self.sectors_table.setRowCount(len(hot_sectors))
        
        for i, sector in enumerate(hot_sectors):
            # 行业名称
            self.sectors_table.setItem(i, 0, QTableWidgetItem(sector.get("name", "--")))
            
            # 预期涨幅
            avg_change = sector.get("avg_change", 0) * 100
            change_item = QTableWidgetItem(f"{avg_change:+.2f}%")
            
            if avg_change > 0:
                change_item.setForeground(QColor("green"))
            elif avg_change < 0:
                change_item.setForeground(QColor("red"))
                
            self.sectors_table.setItem(i, 1, change_item)
        
        # 更新推荐股票
        recommendations = insights_data.get("recommendations", [])
        self.recommendations_table.setRowCount(len(recommendations))
        
        for i, stock in enumerate(recommendations):
            # 排名
            rank_item = QTableWidgetItem(str(stock.get("rank", i+1)))
            self.recommendations_table.setItem(i, 0, rank_item)
            
            # 代码
            code_item = QTableWidgetItem(stock.get("code", "--"))
            self.recommendations_table.setItem(i, 1, code_item)
            
            # 名称
            name_item = QTableWidgetItem(stock.get("name", "--"))
            self.recommendations_table.setItem(i, 2, name_item)
            
            # 预期涨幅
            expected_change = stock.get("expected_change", 0) * 100
            change_item = QTableWidgetItem(f"{expected_change:+.2f}%")
            
            if expected_change > 0:
                change_item.setForeground(QColor("green"))
            elif expected_change < 0:
                change_item.setForeground(QColor("red"))
                
            self.recommendations_table.setItem(i, 3, change_item)
            
            # 推荐建议
            recommendation = stock.get("recommendation", "--")
            reason = stock.get("reason", "")
            
            recommendation_item = QTableWidgetItem(recommendation)
            recommendation_item.setToolTip(reason)
            
            if "买入" in recommendation:
                recommendation_item.setForeground(QColor("green"))
            elif "卖出" in recommendation:
                recommendation_item.setForeground(QColor("red"))
            else:
                recommendation_item.setForeground(QColor("#2196F3"))
                
            self.recommendations_table.setItem(i, 4, recommendation_item)


class RecommendedStocksPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("recommendedStocksPanel")
        self.setup_ui()
        
    def setup_ui(self):
        # 创建主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # 创建顶部标题栏
        header = QHBoxLayout()
        title = QLabel("量子推荐股票")
        title.setStyleSheet("""
            QLabel {
                color: #00ff00;
                font-size: 18px;
                font-weight: bold;
            }
        """)
        
        refresh_btn = QPushButton("刷新推荐")
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a2a2a;
                color: #00ff00;
                border: 1px solid #00ff00;
                padding: 5px 15px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #3a3a3a;
            }
        """)
        refresh_btn.clicked.connect(self.refresh_recommendations)
        
        header.addWidget(title)
        header.addStretch()
        header.addWidget(refresh_btn)
        
        # 创建股票列表滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                border: none;
                background: #2a2a2a;
                width: 8px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #4a4a4a;
                min-height: 20px;
                border-radius: 4px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        # 创建股票列表容器
        stocks_container = QWidget()
        stocks_layout = QVBoxLayout(stocks_container)
        stocks_layout.setSpacing(10)
        
        # 示例股票数据
        demo_stocks = [
            ("贵州茅台", "600519", "+2.45%", 
             "1. 行业龙头地位稳固，市占率持续提升\n"
             "2. 量子态稳定性强，技术指标MACD金叉确认\n"
             "3. 估值处于合理区间，具备配置价值\n"
             "4. 成交量温和放大，筹码趋于集中\n"
             "5. 行业基本面向好，政策面持续利好"),
            
            ("宁德时代", "300750", "+3.67%",
             "1. 全球动力电池龙头，市占率第一\n"
             "2. 研发投入持续加大，技术壁垒高\n"
             "3. 产能释放加速，订单充足\n"
             "4. 下游需求旺盛，新能源车渗透率提升\n"
             "5. 量子态趋势向上，技术面突破确认"),
            
            ("中芯国际", "688981", "+1.23%",
             "1. 半导体制造龙头，国产替代加速\n"
             "2. 先进制程持续突破，已具备28nm量产能力\n"
             "3. 订单充足，产能利用率维持高位\n"
             "4. 政策支持力度大，享受税收优惠\n"
             "5. 行业处于上升周期，景气度持续"),
            
            ("比亚迪", "002594", "+4.12%",
             "1. 新能源汽车龙头，产业链完整\n"
             "2. 技术积累深厚，刀片电池放量\n"
             "3. 市场份额持续提升，品牌力增强\n"
             "4. 海外布局加速，全球化战略推进\n"
             "5. 量子态动能强劲，走势强于大盘"),
            
            ("腾讯控股", "00700", "+1.56%",
             "1. 互联网科技巨头，业务布局全面\n"
             "2. 现金流充沛，财务状况稳健\n"
             "3. 游戏业务稳定，云计算快速增长\n"
             "4. AI布局领先，产业互联网持续发力\n"
             "5. 估值处于历史低位，具备长期投资价值")
        ]
        
        for name, code, change, desc in demo_stocks:
            stock_card = self.create_stock_card(name, code, change, desc)
            stocks_layout.addWidget(stock_card)
        
        scroll.setWidget(stocks_container)
        
        # 添加到主布局
        main_layout.addLayout(header)
        main_layout.addWidget(scroll)
        
    def create_stock_card(self, name, code, change, desc):
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #2a2a2a;
                border: 1px solid #333333;
                border-radius: 8px;
                padding: 15px;
            }
            QFrame:hover {
                border-color: #00ff00;
                background-color: #3a3a3a;
            }
        """)
        
        layout = QVBoxLayout(card)
        layout.setSpacing(10)
        
        # 股票标题行
        header = QHBoxLayout()
        name_label = QLabel(f"{name} ({code})")
        name_label.setStyleSheet("color: #ffffff; font-size: 16px; font-weight: bold;")
        
        change_label = QLabel(change)
        change_color = "#00ff00" if float(change.strip('%')) >= 0 else "#ff0000"
        change_label.setStyleSheet(f"color: {change_color}; font-size: 16px; font-weight: bold;")
        
        header.addWidget(name_label)
        header.addStretch()
        header.addWidget(change_label)
        
        # 推荐理由
        desc_label = QLabel(desc)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("""
            QLabel {
                color: #aaaaaa;
                font-size: 14px;
                line-height: 1.6;
                padding: 5px;
            }
        """)
        
        layout.addLayout(header)
        layout.addWidget(desc_label)
        
        return card
        
    def refresh_recommendations(self):
        # TODO: 实现刷新推荐逻辑
        pass


class StockPredictionPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("stockPredictionPanel")
        self.tushare_connector = TushareDataConnector()
        self.quantum_predictor = QuantumPredictor()
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        
        # 标题
        title = QLabel("量子预测引擎")
        title.setStyleSheet("""
            QLabel {
                color: #00ff00;
                font-size: 18px;
                font-weight: bold;
            }
        """)
        layout.addWidget(title)
        
        # 预测周期选择
        period_layout = QHBoxLayout()
        period_label = QLabel("预测周期:")
        period_label.setStyleSheet("color: #ffffff;")
        self.period_combo = QComboBox()
        self.period_combo.addItems(["短期(7天)", "中期(30天)", "长期(90天)"])
        self.period_combo.setStyleSheet("""
            QComboBox {
                background-color: #2a2a2a;
                color: #ffffff;
                border: 1px solid #444444;
                padding: 5px;
                border-radius: 3px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: url(resources/down_arrow.png);
                width: 12px;
                height: 12px;
            }
        """)
        period_layout.addWidget(period_label)
        period_layout.addWidget(self.period_combo)
        layout.addLayout(period_layout)
        
        # 预测结果显示区域
        self.prediction_area = QScrollArea()
        self.prediction_area.setWidgetResizable(True)
        self.prediction_area.setStyleSheet("""
            QScrollArea {
                border: 1px solid #333333;
                background-color: #1a1a1a;
            }
        """)
        
        prediction_widget = QWidget()
        self.prediction_layout = QVBoxLayout(prediction_widget)
        self.prediction_area.setWidget(prediction_widget)
        layout.addWidget(self.prediction_area)
        
        # 刷新按钮
        refresh_btn = QPushButton("开始预测分析")
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a2a2a;
                color: #00ff00;
                border: 1px solid #00ff00;
                padding: 8px 15px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #3a3a3a;
            }
        """)
        refresh_btn.clicked.connect(self.update_predictions)
        layout.addWidget(refresh_btn)
        
    def update_predictions(self):
        # 清除旧的预测结果
        for i in reversed(range(self.prediction_layout.count())): 
            self.prediction_layout.itemAt(i).widget().deleteLater()
            
        try:
            # 获取实时市场数据
            market_data = self.tushare_connector.get_latest_market_data()
            
            # 获取推荐股票列表
            recommended_stocks = self.quantum_predictor.get_top_recommendations()
            
            for stock in recommended_stocks:
                # 进行量子维度分析
                quantum_analysis = self.quantum_predictor.analyze_quantum_state(stock['code'])
                
                # 多维度预测
                predictions = self.quantum_predictor.predict_multi_dimensional(
                    stock['code'],
                    period=self.period_combo.currentText()
                )
                
                # 创建预测结果卡片
                card = self.create_prediction_card(stock, quantum_analysis, predictions)
                self.prediction_layout.addWidget(card)
                
        except Exception as e:
            self.show_error_message(f"预测更新失败: {str(e)}")
            
    def create_prediction_card(self, stock, quantum_analysis, predictions):
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #2a2a2a;
                border: 1px solid #333333;
                border-radius: 8px;
                padding: 15px;
            }
        """)
        
        layout = QVBoxLayout(card)
        
        # 股票基本信息
        header = QHBoxLayout()
        name_label = QLabel(f"{stock['name']} ({stock['code']})")
        name_label.setStyleSheet("color: #ffffff; font-size: 16px; font-weight: bold;")
        
        current_price = QLabel(f"当前: {stock['current_price']:.2f}")
        current_price.setStyleSheet("color: #00ff00; font-size: 16px;")
        
        header.addWidget(name_label)
        header.addStretch()
        header.addWidget(current_price)
        layout.addLayout(header)
        
        # 量子态分析结果
        quantum_label = QLabel(f"量子态强度: {quantum_analysis['strength']:.2f}")
        quantum_label.setStyleSheet("color: #00ffff; font-size: 14px;")
        layout.addWidget(quantum_label)
        
        # 预测结果
        prediction_grid = QGridLayout()
        periods = ['7天', '30天', '90天']
        for i, period in enumerate(periods):
            period_label = QLabel(period)
            period_label.setStyleSheet("color: #888888;")
            
            price_label = QLabel(f"¥{predictions[period]['price']:.2f}")
            price_label.setStyleSheet("color: #ffffff; font-weight: bold;")
            
            change = predictions[period]['change']
            change_color = "#00ff00" if change >= 0 else "#ff0000"
            change_label = QLabel(f"{change:+.2f}%")
            change_label.setStyleSheet(f"color: {change_color}; font-weight: bold;")
            
            prediction_grid.addWidget(period_label, i, 0)
            prediction_grid.addWidget(price_label, i, 1)
            prediction_grid.addWidget(change_label, i, 2)
        
        layout.addLayout(prediction_grid)
        
        # 预测置信度
        confidence = QProgressBar()
        confidence.setRange(0, 100)
        confidence.setValue(int(predictions['confidence'] * 100))
        confidence.setStyleSheet("""
            QProgressBar {
                border: 1px solid #444444;
                border-radius: 3px;
                text-align: center;
                background-color: #1a1a1a;
            }
            QProgressBar::chunk {
                background-color: #00ff00;
                border-radius: 2px;
            }
        """)
        layout.addWidget(confidence)
        
        return card
        
    def show_error_message(self, message):
        error_label = QLabel(message)
        error_label.setStyleSheet("""
            QLabel {
                color: #ff0000;
                padding: 10px;
                background-color: #2a2a2a;
                border: 1px solid #ff0000;
                border-radius: 5px;
            }
        """)
        self.prediction_layout.addWidget(error_label)


class QuantumPredictionView(QWidget):
    """量子预测视图 - 主视图"""
    
    def __init__(self, data_controller, parent=None):
        """初始化量子预测视图
        
        Args:
            data_controller: 数据控制器
            parent: 父窗口
        """
        super().__init__(parent)
        self.data_controller = data_controller
        
        # 连接信号
        self.data_controller.prediction_ready_signal.connect(self.on_prediction_ready)
        self.data_controller.market_insights_ready_signal.connect(self.on_market_insights_ready)
        
        self.setup_ui()
    
    def setup_ui(self):
        """设置UI"""
        # 主布局
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # 左侧控制面板
        self.control_panel = QuantumPredictionControl()
        self.control_panel.setFixedWidth(300)
        
        # 连接信号
        self.control_panel.prediction_requested.connect(self.request_prediction)
        self.control_panel.market_insights_requested.connect(self.request_market_insights)
        self.control_panel.quantum_params_changed.connect(self.set_quantum_params)
        
        # 右侧内容区域
        self.content_tabs = QTabWidget()
        
        # 添加预测视图标签页
        self.prediction_view = StockPredictionView()
        self.content_tabs.addTab(self.prediction_view, "股票预测")
        
        # 添加市场洞察标签页
        self.insights_view = MarketInsightsView()
        self.content_tabs.addTab(self.insights_view, "市场洞察")
        
        # 分割器
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.control_panel)
        splitter.addWidget(self.content_tabs)
        
        # 设置分割比例
        splitter.setSizes([300, 700])
        
        main_layout.addWidget(splitter)
    
    def request_prediction(self, stock_code, days):
        """请求股票预测
        
        Args:
            stock_code: 股票代码
            days: 预测天数
        """
        # 切换到预测标签页
        self.content_tabs.setCurrentIndex(0)
        
        # 请求预测
        self.data_controller.predict_stock(stock_code, days)
    
    def request_market_insights(self):
        """请求市场洞察"""
        # 切换到市场洞察标签页
        self.content_tabs.setCurrentIndex(1)
        
        # 请求市场洞察
        self.data_controller.generate_market_insights()
    
    def set_quantum_params(self, coherence, superposition, entanglement):
        """设置量子参数
        
        Args:
            coherence: 量子相干性参数
            superposition: 量子叠加态参数
            entanglement: 量子纠缠参数
        """
        self.data_controller.set_quantum_params(
            coherence=coherence,
            superposition=superposition,
            entanglement=entanglement
        )
    
    def on_prediction_ready(self, stock_code, prediction_data):
        """处理预测结果就绪
        
        Args:
            stock_code: 股票代码
            prediction_data: 预测数据
        """
        # 显示预测结果
        self.prediction_view.display_prediction(stock_code, prediction_data)
        
        # 切换到预测标签页
        self.content_tabs.setCurrentIndex(0)
    
    def on_market_insights_ready(self, insights_data):
        """处理市场洞察就绪
        
        Args:
            insights_data: 市场洞察数据
        """
        # 显示市场洞察
        self.insights_view.display_insights(insights_data)
        
        # 切换到市场洞察标签页
        self.content_tabs.setCurrentIndex(1)
    
    def set_predictor(self, predictor):
        """设置量子预测器
        
        Args:
            predictor: 量子预测器实例
        """
        self.predictor = predictor 