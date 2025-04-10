#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - 投资组合视图
显示投资组合信息、资产配置、风险指标和绩效数据
"""

import logging
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QTableWidget, QTableWidgetItem, QFrame, QSplitter,
                            QGridLayout, QGroupBox, QPushButton, QProgressBar,
                            QMessageBox)
from PyQt5.QtCore import Qt, QSize, QTimer
from PyQt5.QtGui import QColor, QPainter, QBrush, QPen, QFont

# 导入自定义组件
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gui.components.charts import PieChart, LineChart

class AccountSummary(QWidget):
    """账户摘要组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger("AccountSummary")
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        
        # 设置标题
        title = QLabel("投资组合概览")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.layout.addWidget(title, 0, 0, 1, 4)
        
        # 添加资产信息
        row = 1
        self._create_metric_row("总资产", "1,000,000", "元", row)
        row += 1
        self._create_metric_row("可用资金", "500,000", "元", row)
        row += 1
        self._create_metric_row("持仓市值", "500,000", "元", row)
        row += 1
        self._create_metric_row("今日收益", "+50,000", "元", row)
        row += 1
        self._create_metric_row("今日收益率", "+5.0", "%", row)
        row += 1
        self._create_metric_row("总收益", "+100,000", "元", row)
        row += 1
        self._create_metric_row("总收益率", "+10.0", "%", row)
        row += 1
        self._create_metric_row("最大回撤", "5.2", "%", row)
        row += 1
        self._create_metric_row("夏普比率", "2.35", "", row)
        row += 1
        
        # 量子评分
        self._create_quantum_score_row("量子优化评分", 0.85, row)
        
        # 设置样式
        self.setStyleSheet("""
            QLabel {
                font-size: 14px;
            }
            QLabel.value {
                font-size: 16px;
                font-weight: bold;
            }
            QLabel.unit {
                font-size: 12px;
                color: #666;
            }
            QLabel.positive {
                color: #00aa00;
            }
            QLabel.negative {
                color: #ff0000;
            }
        """)
    
    def _create_metric_row(self, label_text, value_text, unit_text, row):
        """创建指标行"""
        label = QLabel(label_text)
        self.layout.addWidget(label, row, 0)
        
        value = QLabel(value_text)
        value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        value.setObjectName(f"value_{label_text}")
        value.setProperty("class", "value")
        
        # 设置正负值颜色
        if value_text.startswith("+"):
            value.setProperty("class", "value positive")
        elif value_text.startswith("-"):
            value.setProperty("class", "value negative")
            
        self.layout.addWidget(value, row, 1)
        
        unit = QLabel(unit_text)
        unit.setProperty("class", "unit")
        self.layout.addWidget(unit, row, 2)
    
    def _create_quantum_score_row(self, label_text, score, row):
        """创建量子评分行"""
        label = QLabel(label_text)
        self.layout.addWidget(label, row, 0)
        
        # 进度条显示评分
        progress = QProgressBar()
        progress.setRange(0, 100)
        progress.setValue(int(score * 100))
        progress.setFormat(f"{score:.2f}")
        progress.setAlignment(Qt.AlignCenter)
        progress.setObjectName("quantum_score")
        
        # 根据得分设置颜色
        if score >= 0.8:
            progress.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    text-align: center;
                    font-weight: bold;
                }
                QProgressBar::chunk {
                    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3c00ff, stop:1 #9b6ef3);
                    border-radius: 5px;
                }
            """)
        elif score >= 0.6:
            progress.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    text-align: center;
                    font-weight: bold;
                }
                QProgressBar::chunk {
                    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #007bff, stop:1 #6fc3ff);
                    border-radius: 5px;
                }
            """)
        else:
            progress.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    text-align: center;
                    font-weight: bold;
                }
                QProgressBar::chunk {
                    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ff9500, stop:1 #ffcc00);
                    border-radius: 5px;
                }
            """)
            
        self.layout.addWidget(progress, row, 1, 1, 2)
    
    def update_account_data(self, data):
        """更新账户数据
        
        Args:
            data: 账户数据字典
        """
        try:
            # 更新总资产
            self._update_value("总资产", f"{data.get('total_asset', 0):,.2f}")
            
            # 更新可用资金
            self._update_value("可用资金", f"{data.get('available_cash', 0):,.2f}")
            
            # 更新持仓市值
            self._update_value("持仓市值", f"{data.get('market_value', 0):,.2f}")
            
            # 更新今日收益
            daily_profit = data.get('daily_profit', 0)
            sign = "+" if daily_profit >= 0 else ""
            self._update_value("今日收益", f"{sign}{daily_profit:,.2f}", daily_profit >= 0)
            
            # 更新今日收益率
            daily_profit_pct = data.get('daily_profit_pct', 0) * 100
            sign = "+" if daily_profit_pct >= 0 else ""
            self._update_value("今日收益率", f"{sign}{daily_profit_pct:.2f}", daily_profit_pct >= 0)
            
            # 更新总收益
            total_profit = data.get('total_profit', 0)
            sign = "+" if total_profit >= 0 else ""
            self._update_value("总收益", f"{sign}{total_profit:,.2f}", total_profit >= 0)
            
            # 更新总收益率
            total_profit_pct = data.get('total_profit_pct', 0) * 100
            sign = "+" if total_profit_pct >= 0 else ""
            self._update_value("总收益率", f"{sign}{total_profit_pct:.2f}", total_profit_pct >= 0)
            
            # 更新最大回撤
            max_drawdown = data.get('max_drawdown', 0) * 100
            self._update_value("最大回撤", f"{max_drawdown:.2f}")
            
            # 更新夏普比率
            sharpe = data.get('sharpe', 0)
            self._update_value("夏普比率", f"{sharpe:.2f}")
            
            # 更新量子评分
            quantum_score = data.get('quantum_score', 0)
            progress = self.findChild(QProgressBar, "quantum_score")
            if progress:
                progress.setValue(int(quantum_score * 100))
                progress.setFormat(f"{quantum_score:.2f}")
                
                # 根据得分设置颜色
                if quantum_score >= 0.8:
                    progress.setStyleSheet("""
                        QProgressBar {
                            border: 1px solid #ddd;
                            border-radius: 5px;
                            text-align: center;
                            font-weight: bold;
                        }
                        QProgressBar::chunk {
                            background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3c00ff, stop:1 #9b6ef3);
                            border-radius: 5px;
                        }
                    """)
                elif quantum_score >= 0.6:
                    progress.setStyleSheet("""
                        QProgressBar {
                            border: 1px solid #ddd;
                            border-radius: 5px;
                            text-align: center;
                            font-weight: bold;
                        }
                        QProgressBar::chunk {
                            background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #007bff, stop:1 #6fc3ff);
                            border-radius: 5px;
                        }
                    """)
                else:
                    progress.setStyleSheet("""
                        QProgressBar {
                            border: 1px solid #ddd;
                            border-radius: 5px;
                            text-align: center;
                            font-weight: bold;
                        }
                        QProgressBar::chunk {
                            background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ff9500, stop:1 #ffcc00);
                            border-radius: 5px;
                        }
                    """)
        except Exception as e:
            self.logger.error(f"更新账户数据失败: {str(e)}")
    
    def _update_value(self, label_text, value_text, is_positive=None):
        """更新值标签"""
        value = self.findChild(QLabel, f"value_{label_text}")
        if value:
            value.setText(value_text)
            
            # 设置正负值颜色
            if is_positive is not None:
                if is_positive:
                    value.setStyleSheet("font-size: 16px; font-weight: bold; color: #00aa00;")
                else:
                    value.setStyleSheet("font-size: 16px; font-weight: bold; color: #ff0000;")
            else:
                value.setStyleSheet("font-size: 16px; font-weight: bold;")

class AssetAllocation(QWidget):
    """资产配置组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger("AssetAllocation")
        
        # 初始化数据
        self.allocation_data = []
        
        # 创建布局
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # 设置标题
        title = QLabel("资产配置")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.layout.addWidget(title)
        
        # 创建饼图
        self.pie_chart = PieChart()
        self.layout.addWidget(self.pie_chart)
        
        # 创建配置表格
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["行业", "配置比例", "风险评分", "成长潜力"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        self.layout.addWidget(self.table)
        
        # 设置样式
        self.setStyleSheet("""
            QTableWidget {
                alternate-background-color: #f7f7f7;
                background-color: #ffffff;
            }
            QHeaderView::section {
                background-color: #e0e0e0;
                padding: 5px;
                border: 0px;
                font-weight: bold;
            }
        """)
    
    def update_allocation_data(self, data):
        """更新资产配置数据
        
        Args:
            data: 资产配置数据列表
        """
        try:
            self.allocation_data = data
            
            # 更新饼图
            labels = [item['name'] for item in data]
            values = [item['value'] for item in data]
            colors = [self._get_color(item.get('color', None)) for item in data]
            
            self.pie_chart.update_data(values, labels, colors)
            
            # 更新表格
            self.table.setRowCount(len(data))
            for i, item in enumerate(data):
                # 行业名称
                name = QTableWidgetItem(item['name'])
                self.table.setItem(i, 0, name)
                
                # 配置比例
                value = QTableWidgetItem(f"{item['value']*100:.1f}%")
                value.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(i, 1, value)
                
                # 风险评分
                risk = QTableWidgetItem(f"{item['risk_score']:.2f}")
                risk.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(i, 2, risk)
                
                # 成长潜力
                growth = QTableWidgetItem(f"{item['growth_potential']:.2f}")
                growth.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(i, 3, growth)
                
            self.table.resizeColumnsToContents()
            
        except Exception as e:
            self.logger.error(f"更新资产配置数据失败: {str(e)}")
    
    def _get_color(self, color):
        """获取颜色"""
        if color is None:
            return QColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        elif isinstance(color, tuple) and len(color) >= 3:
            return QColor(color[0], color[1], color[2])
        elif isinstance(color, str):
            return QColor(color)
        else:
            return QColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

class RiskAnalysis(QWidget):
    """风险分析组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger("RiskAnalysis")
        
        # 初始化数据
        self.risk_metrics = {}
        
        # 创建布局
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # 设置标题
        title = QLabel("风险指标分析")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.layout.addWidget(title)
        
        # 创建风险指标网状图
        self.figure = plt.figure(figsize=(6, 5))
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        
        # 创建风险指标列表
        self.risk_grid = QGridLayout()
        risk_widget = QWidget()
        risk_widget.setLayout(self.risk_grid)
        self.layout.addWidget(risk_widget)
        
        # 设置默认风险指标
        self._initialize_risk_metrics()
    
    def _initialize_risk_metrics(self):
        """初始化风险指标"""
        risk_metrics = {
            "overall_risk": 0.65,
            "market_risk": 0.70,
            "specific_risk": 0.60,
            "liquidity_risk": 0.45,
            "correlation_risk": 0.55,
            "dimension_risk": 0.50,
            "quantum_stability": 0.85,
            "adaptive_hedge_ratio": 0.40
        }
        
        # 创建风险指标标签和进度条
        labels = {
            "overall_risk": "整体风险",
            "market_risk": "市场风险",
            "specific_risk": "特质风险",
            "liquidity_risk": "流动性风险",
            "correlation_risk": "相关性风险",
            "dimension_risk": "维度风险",
            "quantum_stability": "量子稳定性",
            "adaptive_hedge_ratio": "自适应对冲比率"
        }
        
        row = 0
        col = 0
        for key, value in risk_metrics.items():
            # 创建标签
            label = QLabel(labels.get(key, key))
            self.risk_grid.addWidget(label, row, col * 2)
            
            # 创建进度条
            progress = QProgressBar()
            progress.setRange(0, 100)
            progress.setValue(int(value * 100))
            progress.setFormat(f"{value:.2f}")
            progress.setAlignment(Qt.AlignCenter)
            progress.setObjectName(f"risk_{key}")
            
            # 设置进度条颜色
            if key in ["quantum_stability"]:
                progress.setStyleSheet("""
                    QProgressBar {
                        border: 1px solid #ddd;
                        border-radius: 5px;
                        text-align: center;
                    }
                    QProgressBar::chunk {
                        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #00aa00, stop:1 #88cc88);
                        border-radius: 5px;
                    }
                """)
            elif key in ["adaptive_hedge_ratio"]:
                progress.setStyleSheet("""
                    QProgressBar {
                        border: 1px solid #ddd;
                        border-radius: 5px;
                        text-align: center;
                    }
                    QProgressBar::chunk {
                        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0088cc, stop:1 #88ccff);
                        border-radius: 5px;
                    }
                """)
            else:
                progress.setStyleSheet("""
                    QProgressBar {
                        border: 1px solid #ddd;
                        border-radius: 5px;
                        text-align: center;
                    }
                    QProgressBar::chunk {
                        background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #cc0000, stop:1 #ff8888);
                        border-radius: 5px;
                    }
                """)
            
            self.risk_grid.addWidget(progress, row, col * 2 + 1)
            
            # 更新行列索引
            col += 1
            if col >= 2:
                col = 0
                row += 1
        
        # 绘制雷达图
        self._plot_radar_chart(risk_metrics)
    
    def update_risk_metrics(self, data):
        """更新风险指标数据
        
        Args:
            data: 风险指标数据字典
        """
        try:
            self.risk_metrics = data
            
            # 更新风险指标进度条
            for key, value in data.items():
                progress = self.findChild(QProgressBar, f"risk_{key}")
                if progress:
                    progress.setValue(int(value * 100))
                    progress.setFormat(f"{value:.2f}")
            
            # 更新雷达图
            self._plot_radar_chart(data)
            
        except Exception as e:
            self.logger.error(f"更新风险指标数据失败: {str(e)}")
    
    def _plot_radar_chart(self, metrics):
        """绘制雷达图
        
        Args:
            metrics: 风险指标数据字典
        """
        try:
            self.figure.clear()
            
            # 准备数据
            categories = {
                "market_risk": "市场风险",
                "specific_risk": "特质风险",
                "liquidity_risk": "流动性风险",
                "correlation_risk": "相关性风险",
                "dimension_risk": "维度风险",
                "quantum_stability": "量子稳定性"
            }
            
            labels = [categories.get(key, key) for key in categories.keys()]
            values = [metrics.get(key, 0) for key in categories.keys()]
            
            # 添加第一个元素到最后，形成闭环
            values_plot = values + [values[0]]
            
            # 计算角度
            angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
            angles += angles[:1]  # 闭合
            
            # 绘制雷达图
            ax = self.figure.add_subplot(111, polar=True)
            ax.fill(angles, values_plot, 'b', alpha=0.1)
            ax.plot(angles, values_plot, 'b', linewidth=2)
            
            # 添加标签
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels)
            
            # 设置y轴范围和网格
            ax.set_yticks([0.2, 0.4, 0.6, 0.8])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8'])
            ax.set_ylim(0, 1)
            
            # 设置标题
            ax.set_title("多维度风险分析", va='bottom')
            
            self.canvas.draw()
            
        except Exception as e:
            self.logger.error(f"绘制雷达图失败: {str(e)}")

class PerformanceAnalysis(QWidget):
    """绩效分析组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger("PerformanceAnalysis")
        
        # 初始化数据
        self.performance_data = {}
        
        # 创建布局
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # 设置标题
        title = QLabel("绩效分析")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.layout.addWidget(title)
        
        # 创建绩效图表
        self.line_chart = LineChart()
        self.layout.addWidget(self.line_chart)
        
        # 创建绩效指标表格
        metrics_layout = QGridLayout()
        metrics_widget = QWidget()
        metrics_widget.setLayout(metrics_layout)
        self.layout.addWidget(metrics_widget)
        
        # 添加关键指标
        row = 0
        metrics_layout.addWidget(QLabel("年化收益率:"), row, 0)
        self.annual_return = QLabel("15.8%")
        self.annual_return.setStyleSheet("font-weight: bold; color: #00aa00;")
        metrics_layout.addWidget(self.annual_return, row, 1)
        
        metrics_layout.addWidget(QLabel("α系数:"), row, 2)
        self.alpha = QLabel("0.052")
        self.alpha.setStyleSheet("font-weight: bold; color: #00aa00;")
        metrics_layout.addWidget(self.alpha, row, 3)
        
        row += 1
        metrics_layout.addWidget(QLabel("β系数:"), row, 0)
        self.beta = QLabel("0.85")
        self.beta.setStyleSheet("font-weight: bold;")
        metrics_layout.addWidget(self.beta, row, 1)
        
        metrics_layout.addWidget(QLabel("Sortino比率:"), row, 2)
        self.sortino = QLabel("1.95")
        self.sortino.setStyleSheet("font-weight: bold; color: #00aa00;")
        metrics_layout.addWidget(self.sortino, row, 3)
        
        row += 1
        metrics_layout.addWidget(QLabel("胜率:"), row, 0)
        self.win_rate = QLabel("65.2%")
        self.win_rate.setStyleSheet("font-weight: bold; color: #00aa00;")
        metrics_layout.addWidget(self.win_rate, row, 1)
        
        metrics_layout.addWidget(QLabel("盈亏比:"), row, 2)
        self.profit_loss_ratio = QLabel("2.50")
        self.profit_loss_ratio.setStyleSheet("font-weight: bold; color: #00aa00;")
        metrics_layout.addWidget(self.profit_loss_ratio, row, 3)
        
        row += 1
        metrics_layout.addWidget(QLabel("量子增强收益:"), row, 0)
        self.quantum_enhancement = QLabel("3.0%")
        self.quantum_enhancement.setStyleSheet("font-weight: bold; color: #7700ff;")
        metrics_layout.addWidget(self.quantum_enhancement, row, 1)
        
        metrics_layout.addWidget(QLabel("维度稳定性:"), row, 2)
        self.dimension_stability = QLabel("0.88")
        self.dimension_stability.setStyleSheet("font-weight: bold; color: #0077ff;")
        metrics_layout.addWidget(self.dimension_stability, row, 3)
    
    def update_performance_data(self, data):
        """更新绩效数据
        
        Args:
            data: 绩效数据字典
        """
        try:
            self.performance_data = data
            
            # 更新折线图
            if 'dates' in data and 'portfolio_values' in data and 'benchmark_values' in data:
                try:
                    dates = data['dates']
                    portfolio_values = data['portfolio_values']
                    benchmark_values = data['benchmark_values']
                    
                    self.line_chart.update_line_chart(
                        dates, 
                        [portfolio_values, benchmark_values], 
                        ["投资组合", "基准"], 
                        ["#3366cc", "#dc3912"]
                    )
                except Exception as e:
                    self.logger.error(f"更新绩效图表失败: {str(e)}")
            
            # 更新绩效指标
            annual_return = data.get('annual_return', 0) * 100
            self.annual_return.setText(f"{annual_return:.1f}%")
            self.annual_return.setStyleSheet(f"font-weight: bold; color: {'#00aa00' if annual_return > 0 else '#ff0000'};")
            
            alpha = data.get('alpha', 0)
            self.alpha.setText(f"{alpha:.3f}")
            self.alpha.setStyleSheet(f"font-weight: bold; color: {'#00aa00' if alpha > 0 else '#ff0000'};")
            
            beta = data.get('beta', 0)
            self.beta.setText(f"{beta:.2f}")
            
            sortino = data.get('sortino', 0)
            self.sortino.setText(f"{sortino:.2f}")
            self.sortino.setStyleSheet(f"font-weight: bold; color: {'#00aa00' if sortino > 0 else '#ff0000'};")
            
            win_rate = data.get('win_rate', 0) * 100
            self.win_rate.setText(f"{win_rate:.1f}%")
            
            profit_loss_ratio = data.get('profit_loss_ratio', 0)
            self.profit_loss_ratio.setText(f"{profit_loss_ratio:.2f}")
            
            quantum_enhancement = data.get('quantum_enhancement', 0) * 100
            self.quantum_enhancement.setText(f"{quantum_enhancement:.1f}%")
            
            dimension_stability = data.get('dimension_stability', 0)
            self.dimension_stability.setText(f"{dimension_stability:.2f}")
            
        except Exception as e:
            self.logger.error(f"更新绩效数据失败: {str(e)}")

class PortfolioView(QWidget):
    """投资组合视图"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger("PortfolioView")
        
        # 创建布局
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # 创建标题
        title = QLabel("超神投资组合管理")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #333;")
        title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(title)
        
        # 创建分割器和子视图
        splitter = QSplitter(Qt.Horizontal)
        self.layout.addWidget(splitter)
        
        # 左侧：账户摘要和资产配置
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_widget.setLayout(left_layout)
        splitter.addWidget(left_widget)
        
        # 创建账户摘要
        self.account_summary = AccountSummary()
        left_layout.addWidget(self.account_summary)
        
        # 创建资产配置
        self.asset_allocation = AssetAllocation()
        left_layout.addWidget(self.asset_allocation)
        
        # 右侧：风险分析和绩效分析
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_widget.setLayout(right_layout)
        splitter.addWidget(right_widget)
        
        # 创建风险分析
        self.risk_analysis = RiskAnalysis()
        right_layout.addWidget(self.risk_analysis)
        
        # 创建绩效分析
        self.performance_analysis = PerformanceAnalysis()
        right_layout.addWidget(self.performance_analysis)
        
        # 设置分割器比例
        splitter.setSizes([400, 600])
        
        # 设置自动更新定时器
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._auto_update)
        self.update_timer.start(5000)  # 5秒刷新一次
    
    def initialize_with_data(self, data):
        """使用初始数据初始化视图
        
        Args:
            data: 包含账户、配置、风险和绩效数据的字典
        """
        try:
            # 更新账户摘要
            if 'account_data' in data:
                self.account_summary.update_account_data(data['account_data'])
            
            # 更新资产配置
            if 'allocation_data' in data:
                self.asset_allocation.update_allocation_data(data['allocation_data'])
            
            # 更新风险分析
            if 'risk_metrics' in data:
                self.risk_analysis.update_risk_metrics(data['risk_metrics'])
            
            # 更新绩效分析
            if 'performance_data' in data:
                self.performance_analysis.update_performance_data(data['performance_data'])
                
        except Exception as e:
            self.logger.error(f"初始化投资组合视图失败: {str(e)}")
    
    def _auto_update(self):
        """自动更新数据"""
        try:
            # 获取控制器
            controller = None
            if hasattr(self, 'controller'):
                controller = self.controller
            
            if controller:
                # 获取最新的投资组合数据
                account_data = controller.get_portfolio_data()
                allocation_data = controller.get_allocation_data()
                risk_metrics = controller.get_risk_metrics()
                performance_data = controller.get_performance_data()
                
                # 更新视图
                self.account_summary.update_account_data(account_data)
                self.asset_allocation.update_allocation_data(allocation_data)
                self.risk_analysis.update_risk_metrics(risk_metrics)
                self.performance_analysis.update_performance_data(performance_data)
        except Exception as e:
            self.logger.debug(f"自动更新失败: {str(e)}")
            # 不做处理，静默失败 

class SimplePortfolioView(QWidget):
    """简化版超神投资组合视图，用于在加载完整版失败时显示"""
    
    def __init__(self):
        super().__init__()
        self.controller = None
        self.enable_super_god_features = True
        self._init_ui()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("简化投资组合视图初始化完成")
    
    def _init_ui(self):
        """初始化界面"""
        main_layout = QVBoxLayout(self)
        
        # 标题标签
        title_label = QLabel("超神投资组合管理系统")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #7700ff; margin: 20px;")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 添加信息区域
        info_frame = QFrame()
        info_frame.setFrameShape(QFrame.StyledPanel)
        info_frame.setStyleSheet("background-color: #1a1a1a; border-radius: 10px; padding: 15px;")
        info_layout = QVBoxLayout(info_frame)
        
        # 状态信息
        status_label = QLabel("简化模式 - 加载高级数据分析组件中...")
        status_label.setStyleSheet("color: #00aaff; font-size: 16px;")
        status_label.setAlignment(Qt.AlignCenter)
        info_layout.addWidget(status_label)
        
        # 添加账户概览部分
        account_frame = QFrame()
        account_frame.setFrameShape(QFrame.StyledPanel)
        account_frame.setStyleSheet("background-color: #222222; border-radius: 8px; padding: 10px;")
        account_layout = QGridLayout(account_frame)
        
        # 添加模拟数据
        labels = [
            ("总资产:", "$1,250,000.00", "color: #00ff88;"),
            ("可用资金:", "$450,000.00", "color: #00aaff;"),
            ("持仓市值:", "$800,000.00", "color: #ff9900;"),
            ("今日收益:", "+$12,500.00", "color: #00ff00;"),
            ("收益率:", "+15.3%", "color: #00ff00;")
        ]
        
        for i, (name, value, style) in enumerate(labels):
            name_label = QLabel(name)
            name_label.setStyleSheet("color: white;")
            account_layout.addWidget(name_label, i, 0)
            
            value_label = QLabel(value)
            value_label.setStyleSheet(style)
            value_label.setAlignment(Qt.AlignRight)
            account_layout.addWidget(value_label, i, 1)
        
        info_layout.addWidget(account_frame)
        
        # 加载中提示
        loading_label = QLabel("正在加载完整的超神投资组合数据...")
        loading_label.setStyleSheet("color: #aaaaaa; margin-top: 20px;")
        loading_label.setAlignment(Qt.AlignCenter)
        info_layout.addWidget(loading_label)
        
        # 刷新按钮
        refresh_btn = QPushButton("刷新数据")
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #5500aa;
                color: white;
                border-radius: 5px;
                padding: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #7700ff;
            }
        """)
        refresh_btn.clicked.connect(self._on_refresh_clicked)
        info_layout.addWidget(refresh_btn)
        
        main_layout.addWidget(info_frame)
        
        # 添加占位符说明
        note_label = QLabel("超神投资组合高级功能正在初始化中，请稍候或切换到其他视图...")
        note_label.setStyleSheet("color: #aaaaaa; margin: 15px;")
        note_label.setAlignment(Qt.AlignCenter)
        note_label.setWordWrap(True)
        main_layout.addWidget(note_label)
        
        main_layout.addStretch(1)
    
    def _on_refresh_clicked(self):
        """刷新按钮点击处理"""
        QMessageBox.information(self, "刷新", "正在尝试加载完整投资组合数据，请稍候...")
        
    def update_view(self):
        """更新视图数据"""
        # 简化版视图不进行实际更新
        pass
        
    def initialize_with_data(self, data):
        """初始化数据"""
        # 简化版视图不进行实际初始化
        pass 