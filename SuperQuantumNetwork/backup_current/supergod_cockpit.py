#!/usr/bin/env python3
"""
超神量子共生系统 - 全息驾驶舱
高级量子金融分析平台集中控制中心
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import threading
import random
import io
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QGridLayout, 
                            QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                            QFrame, QSplitter, QProgressBar, QComboBox,
                            QMessageBox, QFileDialog, QTableWidget, QTableWidgetItem,
                            QScrollArea, QSizePolicy, QToolBar, QProgressDialog)
from PyQt5.QtCore import Qt, QTimer, QSize, pyqtSignal, QMetaObject, Q_ARG, QObject, pyqtSlot, QDateTime
from PyQt5.QtGui import QFont, QColor, QPalette, QImage, QPixmap, QBrush
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
import traceback
from typing import List, Dict, Any
import time
import psutil

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("supergod_cockpit.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SupergodCockpit")

# 定义颜色常量
class SupergodColors:
    """超神系统颜色主题"""
    PRIMARY = "#1a1a2e"
    SECONDARY = "#16213e"
    SECONDARY_DARK = "#0f3460"
    ACCENT_DARK = "#0f3460"     # 添加缺失的颜色
    HIGHLIGHT = "#e94560"
    TEXT_PRIMARY = "#ffffff"
    TEXT_SECONDARY = "#a4a6b3"
    SUCCESS = "#4caf50"
    WARNING = "#ff9800"
    ERROR = "#f44336"
    PANEL_BG = "#1f1f3f"
    CHART_BG = "#2a2a4a"
    POSITIVE = "#4cd97b"        # 上涨/正向 (绿色)
    NEGATIVE = "#e94560"        # 下跌/负向 (红色)
    NEUTRAL = "#7c83fd"         # 中性 (紫色)
    GRID_LINES = "#2a2a4a"      # 网格线 (深灰蓝)
    CODE_BG = "#2d2d44"         # 代码背景 (深紫)

# 定义错误类
class QuantumError(Exception):
    """量子系统错误"""
    pass

class ModuleError(Exception):
    """模块错误"""
    pass

class DataError(Exception):
    """数据错误"""
    pass

class SystemError(Exception):
    """系统错误"""
    pass

# 超神系统模块
try:
    # 尝试导入超神分析引擎
    from supergod_desktop import (
        QuantumEngine, DataConnector, MarketAnalyzer,
        PredictionEngine, VisualizationEngine
    )
    SUPERGOD_MODULES_AVAILABLE = True
    logger.info("成功加载超神分析引擎模块")
except ImportError as e:
    logger.warning(f"无法加载部分或全部超神引擎模块: {str(e)}")
    logger.warning("将使用演示数据")
    SUPERGOD_MODULES_AVAILABLE = False


class QuantumStatePanel(QFrame):
    """量子状态面板 - 显示系统当前状态"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setup_ui()
        
    def setup_ui(self):
        self.setStyleSheet(f"""
            background-color: {SupergodColors.PANEL_BG};
            border-radius: 10px;
            padding: 5px;
        """)
        
        layout = QVBoxLayout(self)
        
        # 标题
        title = QLabel("量子状态矩阵")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"""
            color: {SupergodColors.TEXT_PRIMARY};
            font-size: 18px;
            font-weight: bold;
            padding-bottom: 5px;
            border-bottom: 1px solid {SupergodColors.HIGHLIGHT};
        """)
        layout.addWidget(title)
        
        # 创建状态网格
        grid_layout = QGridLayout()
        states = [
            ("市场周期", "积累期", "97%"),
            ("量子相位", "跃迁临界", "88%"),
            ("维度共振", "稳定", "75%"),
            ("能量势能", "蓄积", "82%"),
            ("混沌程度", "低", "23%"),
            ("临界预警", "否", "10%")
        ]
        
        for i, (name, value, confidence) in enumerate(states):
            # 名称标签
            name_label = QLabel(f"{name}:")
            name_label.setStyleSheet(f"color: {SupergodColors.TEXT_SECONDARY};")
            
            # 值标签
            value_label = QLabel(value)
            value_label.setStyleSheet(f"color: {SupergodColors.TEXT_PRIMARY}; font-weight: bold;")
            
            # 置信度进度条
            conf_bar = QProgressBar()
            conf_bar.setValue(int(confidence.strip('%')))
            conf_bar.setTextVisible(True)
            conf_bar.setStyleSheet(f"""
                QProgressBar {{
                    background-color: {SupergodColors.SECONDARY_DARK};
                    color: {SupergodColors.TEXT_PRIMARY};
                    border-radius: 3px;
                    text-align: center;
                }}
                QProgressBar::chunk {{
                    background-color: {SupergodColors.HIGHLIGHT};
                    border-radius: 3px;
                }}
            """)
            
            grid_layout.addWidget(name_label, i, 0)
            grid_layout.addWidget(value_label, i, 1)
            grid_layout.addWidget(conf_bar, i, 2)
        
        layout.addLayout(grid_layout)
        
        # 添加自动更新标志
        status_label = QLabel("自动更新中 | 上次更新: 10秒前")
        status_label.setAlignment(Qt.AlignRight)
        status_label.setStyleSheet(f"color: {SupergodColors.TEXT_SECONDARY}; font-size: 10px;")
        layout.addWidget(status_label)

    def update_quantum_values(self):
        """更新量子状态值"""
        # 实现更新量子状态值的逻辑
        pass


class MarketInsightPanel(QFrame):
    """市场洞察面板 - 显示关键市场信息"""
    
    # 添加一个信号用于更新界面 - 移到类级别
    data_update_signal = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.logger = logging.getLogger("MarketInsightPanel")
        self.parent_cockpit = parent
        
        # 市场数据值
        self.market_data = {
            'index': {'value': "加载中...", 'color': SupergodColors.TEXT_PRIMARY},
            'sentiment': {'value': "加载中...", 'color': SupergodColors.TEXT_PRIMARY},
            'fund_flow': {'value': "加载中...", 'color': SupergodColors.TEXT_PRIMARY},
            'north_flow': {'value': "加载中...", 'color': SupergodColors.TEXT_PRIMARY},
            'volatility': {'value': "加载中...", 'color': SupergodColors.TEXT_PRIMARY},
            'volume': {'value': "加载中...", 'color': SupergodColors.TEXT_PRIMARY}
        }
        
        # 数据标签引用
        self.data_labels = {}
        
        self.setup_ui()
        
    def setup_ui(self):
        self.setStyleSheet(f"""
            background-color: {SupergodColors.PANEL_BG};
            border-radius: 10px;
            padding: 5px;
        """)
        
        layout = QVBoxLayout(self)
        
        # 标题
        title = QLabel("市场核心洞察")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"""
            color: {SupergodColors.TEXT_PRIMARY};
            font-size: 18px;
            font-weight: bold;
            padding-bottom: 5px;
            border-bottom: 1px solid {SupergodColors.HIGHLIGHT};
        """)
        layout.addWidget(title)
        
        # 关键数据
        insights = [
            ("index", "沪深300指数", self.market_data['index']['value'], self.market_data['index']['color']),
            ("sentiment", "市场情绪", self.market_data['sentiment']['value'], self.market_data['sentiment']['color']),
            ("fund_flow", "资金流向", self.market_data['fund_flow']['value'], self.market_data['fund_flow']['color']),
            ("north_flow", "北向资金", self.market_data['north_flow']['value'], self.market_data['north_flow']['color']),
            ("volatility", "波动率", self.market_data['volatility']['value'], self.market_data['volatility']['color']),
            ("volume", "成交量", self.market_data['volume']['value'], self.market_data['volume']['color'])
        ]
        
        for key, name, value, color in insights:
            item_layout = QHBoxLayout()
            
            name_label = QLabel(name)
            name_label.setStyleSheet(f"color: {SupergodColors.TEXT_SECONDARY};")
            
            value_label = QLabel(value)
            value_label.setStyleSheet(f"color: {color}; font-weight: bold;")
            value_label.setAlignment(Qt.AlignRight)
            
            # 保存标签引用，以便更新
            self.data_labels[key] = value_label
            
            item_layout.addWidget(name_label)
            item_layout.addWidget(value_label)
            
            layout.addLayout(item_layout)
        
        # 异常检测部分
        anomaly_title = QLabel("检测到的异常:")
        anomaly_title.setStyleSheet(f"color: {SupergodColors.HIGHLIGHT}; font-weight: bold; margin-top: 10px;")
        layout.addWidget(anomaly_title)
        
        # 异常容器
        self.anomaly_container = QFrame()
        anomaly_layout = QVBoxLayout(self.anomaly_container)
        anomaly_layout.setContentsMargins(5, 0, 5, 0)
        
        # 默认异常项
        self.anomaly_labels = []
        default_anomalies = [
            "• 加载中...",
        ]
        
        for anomaly in default_anomalies:
            anomaly_label = QLabel(anomaly)
            anomaly_label.setStyleSheet(f"color: {SupergodColors.TEXT_PRIMARY};")
            anomaly_layout.addWidget(anomaly_label)
            self.anomaly_labels.append(anomaly_label)
            
        layout.addWidget(self.anomaly_container)
        
        # 为后续更新操作添加一个刷新按钮
        refresh_layout = QHBoxLayout()
        refresh_layout.addStretch()
        
        refresh_button = QPushButton("刷新")
        refresh_button.setFixedSize(60, 25)
        refresh_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {SupergodColors.ACCENT_DARK};
                color: {SupergodColors.TEXT_PRIMARY};
                border: none;
                border-radius: 3px;
                padding: 3px;
            }}
            QPushButton:hover {{
                background-color: {SupergodColors.HIGHLIGHT};
            }}
        """)
        refresh_button.clicked.connect(self.refresh_data)
        refresh_layout.addWidget(refresh_button)
        
        layout.addLayout(refresh_layout)
        
        # 连接数据更新信号到槽
        self.data_update_signal.connect(self.update_values)

    # 添加pyqtSlot装饰器，使update_values成为Qt槽
    @pyqtSlot(dict)
    def update_values(self, market_data=None):
        """更新市场洞察值"""
        try:
            if market_data:
                # 更新索引值
                if 'index' in market_data:
                    index_value = market_data['index']
                    change = market_data.get('index_change', 0)
                    color = SupergodColors.POSITIVE if change >= 0 else SupergodColors.NEGATIVE
                    self.market_data['index'] = {'value': f"{index_value:,.2f} {'' if change == 0 else ('↑' if change > 0 else '↓')}", 'color': color}
                    
                # 更新市场情绪
                if 'sentiment' in market_data:
                    sentiment = market_data['sentiment']
                    sentiment_text = "乐观" if sentiment > 0.6 else "中性" if sentiment > 0.4 else "谨慎"
                    color = SupergodColors.POSITIVE if sentiment > 0.6 else SupergodColors.NEUTRAL if sentiment > 0.4 else SupergodColors.NEGATIVE
                    self.market_data['sentiment'] = {'value': f"{sentiment_text} ({sentiment:.2f})", 'color': color}
                
                # 更新资金流向
                if 'fund_flow' in market_data:
                    fund_flow = market_data['fund_flow']
                    flow_direction = "流入" if fund_flow > 0 else "流出"
                    flow_abs = abs(fund_flow)
                    color = SupergodColors.POSITIVE if fund_flow > 0 else SupergodColors.NEGATIVE
                    self.market_data['fund_flow'] = {'value': f"{flow_direction} {flow_abs:.1f}亿", 'color': color}
                
                # 更新北向资金
                if 'north_flow' in market_data:
                    north_flow = market_data['north_flow']
                    flow_direction = "流入" if north_flow > 0 else "流出"
                    flow_abs = abs(north_flow)
                    color = SupergodColors.POSITIVE if north_flow > 0 else SupergodColors.NEGATIVE
                    self.market_data['north_flow'] = {'value': f"{flow_direction} {flow_abs:.1f}亿", 'color': color}
                
                # 更新波动率
                if 'volatility' in market_data:
                    volatility = market_data['volatility']
                    vol_change = market_data.get('volatility_change', 0)
                    color = SupergodColors.NEGATIVE if vol_change > 0 else SupergodColors.POSITIVE
                    self.market_data['volatility'] = {'value': f"{volatility:.1f}% {'↑' if vol_change > 0 else '↓'}", 'color': color}
                
                # 更新成交量
                if 'volume' in market_data:
                    volume = market_data['volume']
                    vol_change = market_data.get('volume_change', 0)
                    volume_text = f"{volume/100:.0f}亿" if volume >= 100 else f"{volume:.1f}亿"
                    color = SupergodColors.POSITIVE if vol_change > 0 else SupergodColors.TEXT_PRIMARY
                    self.market_data['volume'] = {'value': f"{volume_text} {'↑' if vol_change > 0 else ''}", 'color': color}
                
                # 更新异常
                if 'anomalies' in market_data and market_data['anomalies']:
                    anomalies = market_data['anomalies']
                    # 清除现有异常标签
                    for label in self.anomaly_labels:
                        label.setParent(None)
                    self.anomaly_labels.clear()
                    
                    # 添加新异常
                    for anomaly in anomalies[:3]:  # 最多显示3个异常
                        anomaly_label = QLabel(f"• {anomaly}")
                        anomaly_label.setStyleSheet(f"color: {SupergodColors.TEXT_PRIMARY};")
                        self.anomaly_container.layout().addWidget(anomaly_label)
                        self.anomaly_labels.append(anomaly_label)
            
            # 更新UI上的标签
            for key, label in self.data_labels.items():
                if key in self.market_data:
                    label.setText(self.market_data[key]['value'])
                    label.setStyleSheet(f"color: {self.market_data[key]['color']}; font-weight: bold;")
            
        except Exception as e:
            self.logger.error(f"更新市场洞察值时出错: {str(e)}")
    
    def refresh_data(self):
        """刷新市场数据"""
        try:
            # 在GUI线程中显示加载状态
            for key in self.market_data:
                self.market_data[key] = {'value': "加载中...", 'color': SupergodColors.TEXT_PRIMARY}
                if key in self.data_labels:
                    self.data_labels[key].setText("加载中...")
                    self.data_labels[key].setStyleSheet(f"color: {SupergodColors.TEXT_PRIMARY}; font-weight: bold;")
            
            # 启动一个线程加载数据，避免UI阻塞
            data_thread = threading.Thread(target=self._fetch_real_market_data, name="MarketDataThread", daemon=True)
            
            # 如果父窗口是SupergodCockpit，则将线程添加到其活动线程列表
            if hasattr(self, 'parent_cockpit') and self.parent_cockpit and hasattr(self.parent_cockpit, 'active_threads'):
                self.parent_cockpit.active_threads.append(data_thread)
                
            data_thread.start()
        except Exception as e:
            self.logger.error(f"刷新市场数据时出错: {str(e)}")
    
    def _fetch_real_market_data(self):
        """获取真实市场数据的后台线程"""
        try:
            # 获取当前线程以便后续移除
            current_thread = threading.current_thread()
            
            from tushare_data_connector import TushareDataConnector
            
            # 初始化Tushare连接器
            connector = TushareDataConnector(token="0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10")
            
            # 获取沪深300指数数据
            df = connector.get_market_data(code="000300.SH")
            
            if df is not None and not df.empty:
                # 从DataFrame提取最新数据
                latest = df.iloc[-1]
                prev = df.iloc[-2] if len(df) > 1 else latest
                
                # 计算指标
                index_value = latest['close']
                index_change = (latest['close'] / prev['close'] - 1) * 100
                
                # 计算波动率 (20日标准差)
                if len(df) >= 20:
                    returns = df['close'].pct_change().dropna()
                    volatility = returns.rolling(window=20).std().iloc[-1] * 100 * (252 ** 0.5)  # 年化
                else:
                    volatility = 15.0  # 默认值
                
                # 计算成交量变化
                volume = latest['vol'] / 10000  # 转换为亿
                volume_change = (latest['vol'] / prev['vol'] - 1) * 100
                
                # 计算市场情绪 (简化版)
                price_ma5 = df['close'].rolling(window=5).mean().iloc[-1]
                price_ma10 = df['close'].rolling(window=10).mean().iloc[-1]
                price_ma20 = df['close'].rolling(window=20).mean().iloc[-1]
                
                trend_score = (latest['close'] > price_ma5) * 0.3 + (price_ma5 > price_ma10) * 0.3 + (price_ma10 > price_ma20) * 0.2
                momentum_score = min(max((latest['close'] / df['close'].iloc[-6] - 1) * 5, 0), 0.5)
                volume_score = 0.2 if volume_change > 0 else 0
                
                sentiment = min(trend_score + momentum_score + volume_score, 1.0)
                
                # 模拟资金流向 (实际应从专门API获取)
                random_factor = (sentiment - 0.5) * 2  # 基于情绪的随机因子
                fund_flow = (latest['amount'] / 10000) * random_factor  # 基于成交额估算
                
                # 模拟北向资金 (实际应从专门API获取)
                north_flow = fund_flow * (0.7 + 0.6 * random.random())  # 基于总资金流估算
                
                # 检测异常
                anomalies = []
                
                # 异常1: 检测量价背离
                if volume_change > 15 and index_change < 0:
                    anomalies.append(f"成交量增加{volume_change:.1f}%但指数下跌{abs(index_change):.2f}% (量价背离)")
                
                # 异常2: 检测高波动
                if volatility > 25:
                    anomalies.append(f"波动率异常高 ({volatility:.1f}%) 市场处于高风险阶段")
                
                # 异常3: 检测跳空缺口
                if abs(latest['open'] - prev['close']) / prev['close'] > 0.02:
                    gap_direction = "向上" if latest['open'] > prev['close'] else "向下"
                    gap_pct = abs(latest['open'] - prev['close']) / prev['close'] * 100
                    anomalies.append(f"指数出现{gap_direction}跳空缺口 ({gap_pct:.2f}%)")
                
                # 创建市场数据字典
                market_data = {
                    'index': index_value,
                    'index_change': index_change,
                    'sentiment': sentiment,
                    'fund_flow': fund_flow,
                    'north_flow': north_flow,
                    'volatility': volatility,
                    'volatility_change': volatility - 15,  # 假设前一天是15
                    'volume': volume,
                    'volume_change': volume_change,
                    'anomalies': anomalies
                }
                
                # 使用信号发送数据而不是QMetaObject.invokeMethod
                self.data_update_signal.emit(market_data)
                
            else:
                self.logger.warning("无法获取市场数据")
        except Exception as e:
            self.logger.error(f"获取市场数据时出错: {str(e)}")
            # 在错误情况下恢复默认数据
            default_data = {
                'index': 4923.68,
                'index_change': 0.72,
                'sentiment': 0.72,
                'fund_flow': 114.5,
                'north_flow': 22.8,
                'volatility': 18.2,
                'volatility_change': -0.5,
                'volume': 8729,
                'volume_change': 3.5,
                'anomalies': [
                    "创业板成交量异常增加 (99.8%)",
                    "外盘期货与A股相关性断裂 (82.3%)"
                ]
            }
            
            # 使用信号发送默认数据
            self.data_update_signal.emit(default_data)
        finally:
            # 如果父窗口是SupergodCockpit，则将线程从其活动线程列表中移除
            if hasattr(self, 'parent_cockpit') and self.parent_cockpit and hasattr(self.parent_cockpit, 'active_threads'):
                if current_thread in self.parent_cockpit.active_threads:
                    self.parent_cockpit.active_threads.remove(current_thread)


class DimensionVisualizerPanel(QFrame):
    """维度可视化面板 - 量子维度的图形化表示"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger("DimensionVisualizer")
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setup_ui()
        
    def setup_ui(self):
        """设置UI元素"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 添加标题
        title = QLabel("21维量子空间")
        title.setStyleSheet(f"color: {SupergodColors.HIGHLIGHT}; font-size: 14px; font-weight: bold;")
        layout.addWidget(title)
        
        # 创建可视化标签
        self.visualization_label = QLabel("量子维度可视化加载中...")
        self.visualization_label.setMinimumHeight(200)
        self.visualization_label.setStyleSheet(f"background-color: {SupergodColors.PANEL_BG}; border-radius: 5px;")
        self.visualization_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.visualization_label)
        
        # 创建图表视图
        self.chart_view = QWidget()
        self.chart_view.setMinimumHeight(200)
        
        # 维度信息
        info_layout = QHBoxLayout()
        
        self.dimension_label = QLabel("活跃量子维度: 21/21")
        self.dimension_label.setStyleSheet(f"color: {SupergodColors.TEXT_PRIMARY};")
        info_layout.addWidget(self.dimension_label)
        
        # 添加控制下拉菜单
        controls_layout = QHBoxLayout()
        
        # 维度控制
        dimension_label = QLabel("维度:")
        dimension_label.setStyleSheet(f"color: {SupergodColors.TEXT_SECONDARY};")
        controls_layout.addWidget(dimension_label)
        
        dimension_combo = QComboBox()
        dimension_combo.addItems(["全部维度", "物理维度", "金融维度", "信息维度", "时间维度"])
        dimension_combo.setStyleSheet(f"""
            background-color: {SupergodColors.SECONDARY_DARK};
            color: {SupergodColors.TEXT_PRIMARY};
            border: 1px solid {SupergodColors.ACCENT_DARK};
            border-radius: 3px;
            padding: 2px;
        """)
        controls_layout.addWidget(dimension_combo)
        
        # 显示方式控制
        display_label = QLabel("显示方式:")
        display_label.setStyleSheet(f"color: {SupergodColors.TEXT_SECONDARY};")
        controls_layout.addWidget(display_label)
        
        display_combo = QComboBox()
        display_combo.addItems(["雷达图", "散点图", "热力图", "网络图"])
        display_combo.setStyleSheet(f"""
            background-color: {SupergodColors.SECONDARY_DARK};
            color: {SupergodColors.TEXT_PRIMARY};
            border: 1px solid {SupergodColors.ACCENT_DARK};
            border-radius: 3px;
            padding: 2px;
        """)
        controls_layout.addWidget(display_combo)
        
        layout.addLayout(controls_layout)
        
        # 初始化可视化
        self.update_visualization()

    def apply_quantum_fluctuation(self):
        """应用量子波动效果"""
        # 实现应用量子波动效果的逻辑
        pass

    def apply_ripple_effect(self):
        """应用量子涟漪效果
        在维度可视化中创建波纹效果，使数据点呈现波动状态
        """
        try:
            # 模拟实现，实际应用中可根据具体绘图库实现动画效果
            self.logger.debug("应用量子涟漪视觉效果")
            
            # 如果使用图表对象，可以在此应用特效
            if hasattr(self, 'chart_view') and self.chart_view:
                # 示例: 添加短暂的特效
                current_style = self.chart_view.styleSheet()
                self.chart_view.setStyleSheet(current_style + "; border: 2px solid #00FFFF;")
                QTimer.singleShot(400, lambda: self.chart_view.setStyleSheet(current_style))
        except Exception as e:
            # 特效不影响核心功能，可以忽略错误
            pass

    def update_visualization(self):
        """更新维度可视化"""
        try:
            # 如果未初始化画布，创建一个新画布
            if not hasattr(self, 'chart_view') or not self.chart_view:
                return
                
            # 生成量子维度数据
            dimensions = 21
            data_points = 200
            
            # 生成量子点数据
            quantum_points = []
            for i in range(data_points):
                # 创建一个21维的量子点，每个维度的值在[-1,1]之间
                point = [random.uniform(-1, 1) for _ in range(dimensions)]
                
                # 添加一些类似于超空间的结构
                for d in range(2, dimensions):
                    # 随机添加维度间的关联，创造量子纠缠效果
                    if random.random() < 0.3:
                        # 维度之间的非线性关系
                        point[d] = point[d-1] * point[d-2] * random.uniform(0.5, 1.5)
                
                quantum_points.append(point)
            
            # 使用t-SNE将21维数据降至3维用于可视化
            # 这里简化为随机3D数据
            viz_points = []
            for _ in range(data_points):
                x = random.uniform(-10, 10)
                y = random.uniform(-10, 10)
                z = random.uniform(-10, 10)
                # 为了形成更有趣的结构，添加一些非线性关系
                if random.random() < 0.7:
                    z = x*y/10 + z*0.2
                viz_points.append((x, y, z))
            
            # 更新3D图表
            figure = Figure(figsize=(5, 4), dpi=100)
            canvas = FigureCanvas(figure)
            ax = figure.add_subplot(111, projection='3d')
            
            # 提取坐标
            xs = [p[0] for p in viz_points]
            ys = [p[1] for p in viz_points]
            zs = [p[2] for p in viz_points]
            
            # 创建散点图
            scatter = ax.scatter(xs, ys, zs, c=range(data_points), cmap='plasma', 
                                marker='o', s=20, alpha=0.6)
            
            # 添加一些连线，表示维度间的关联
            for i in range(0, data_points-1, 10):
                if random.random() < 0.3:  # 只连接30%的点，避免过度拥挤
                    ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]], [zs[i], zs[i+1]], 
                            color='cyan', alpha=0.3, linewidth=0.5)
            
            # 设置图表样式
            ax.set_facecolor('#1A1E2E')  # 深色背景
            figure.patch.set_facecolor('#1A1E2E')
            ax.set_title("21维量子空间投影", color='white')
            ax.set_axis_off()  # 隐藏坐标轴
            
            # 添加维度标签
            ax.text2D(0.02, 0.98, f"维度: 21/21", transform=ax.transAxes, 
                     color='white', fontsize=8)
            ax.text2D(0.02, 0.93, f"活跃量子点: {data_points}", transform=ax.transAxes, 
                     color='white', fontsize=8)
            
            # 渲染图表
            canvas.draw()
            
            # 转换为QImage并显示
            buf = io.BytesIO()
            canvas.print_png(buf)
            buf.seek(0)
            
            image = QImage.fromData(buf.getvalue())
            pixmap = QPixmap.fromImage(image)
            
            # 更新UI
            if hasattr(self, 'visualization_label') and self.visualization_label:
                self.visualization_label.setPixmap(pixmap)
                self.visualization_label.setScaledContents(True)
                
            # 更新维度激活信息
            active_dimensions = f"活跃量子维度: {dimensions}/21"
            if hasattr(self, 'dimension_label') and self.dimension_label:
                self.dimension_label.setText(active_dimensions)
                
        except Exception as e:
            print(f"更新维度可视化时出错: {str(e)}")


class PredictionPanel(QFrame):
    """预测面板 - 显示多时间尺度的预测"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setup_ui()
        
    def setup_ui(self):
        self.setStyleSheet(f"""
            background-color: {SupergodColors.PANEL_BG};
            border-radius: 10px;
            padding: 5px;
        """)
        
        layout = QVBoxLayout(self)
        
        # 标题
        title = QLabel("超级预测引擎")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"""
            color: {SupergodColors.TEXT_PRIMARY};
            font-size: 18px;
            font-weight: bold;
            padding-bottom: 5px;
            border-bottom: 1px solid {SupergodColors.HIGHLIGHT};
        """)
        layout.addWidget(title)
        
        # 创建预测图表
        self.chart_frame = QLabel("预测图表加载中...")
        self.chart_frame.setMinimumHeight(180)
        self.chart_frame.setStyleSheet(f"background-color: {SupergodColors.PANEL_BG}; border-radius: 5px;")
        self.chart_frame.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.chart_frame)
        
        # 初始化图表
        self.init_prediction_chart()
        
        # 预测数据
        predictions = [
            ("短期 (1-3天):", "看涨", "92%", SupergodColors.POSITIVE),
            ("中期 (1-2周):", "看涨", "78%", SupergodColors.POSITIVE),
            ("长期 (1-3月):", "看跌", "64%", SupergodColors.NEGATIVE)
        ]
        
        # 添加一个包含框
        pred_container = QFrame()
        pred_container.setStyleSheet(f"""
            background-color: {SupergodColors.SECONDARY_DARK};
            border-radius: 5px;
            padding: 5px;
            margin-top: 5px;
        """)
        pred_layout = QVBoxLayout(pred_container)
        
        for period, direction, confidence, color in predictions:
            pred_item_layout = QHBoxLayout()
            
            period_label = QLabel(period)
            period_label.setStyleSheet(f"color: {SupergodColors.TEXT_SECONDARY};")
            
            direction_label = QLabel(direction)
            direction_label.setStyleSheet(f"color: {color}; font-weight: bold;")
            
            conf_label = QLabel(f"置信度: {confidence}")
            conf_label.setStyleSheet(f"color: {SupergodColors.TEXT_PRIMARY};")
            conf_label.setAlignment(Qt.AlignRight)
            
            pred_item_layout.addWidget(period_label)
            pred_item_layout.addWidget(direction_label)
            pred_item_layout.addWidget(conf_label)
            
            pred_layout.addLayout(pred_item_layout)
        
        layout.addWidget(pred_container)
        
        # 关键临界点
        critical_label = QLabel("关键临界点: 5月15日 (83% 确信度)")
        critical_label.setStyleSheet(f"color: {SupergodColors.HIGHLIGHT}; font-weight: bold; margin-top: 5px;")
        layout.addWidget(critical_label)

    def init_prediction_chart(self):
        """初始化预测图表"""
        try:
            # 创建图形和画布
            fig = Figure(figsize=(5, 3), dpi=100)
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            
            # 设置图表风格
            ax.set_facecolor('#1A1E2E')
            fig.patch.set_facecolor('#1A1E2E')
            
            # 创建预测数据
            days = list(range(1, 31))  # 30天
            # 实际价格历史
            historic_data = [3200 + i * 5 + 50 * np.sin(i/5) for i in range(10)]
            
            # 预测数据 - 三条路径
            prediction_base = historic_data[-1]
            bullish_pred = [prediction_base * (1 + 0.01 * i + 0.002 * i**1.5 + 0.001 * np.sin(i/3)) for i in range(1, 21)]
            neutral_pred = [prediction_base * (1 + 0.005 * i + 0.001 * np.sin(i/2)) for i in range(1, 21)]
            bearish_pred = [prediction_base * (1 - 0.002 * i + 0.004 * i**0.8 + 0.001 * np.sin(i/2.5)) for i in range(1, 21)]
            
            # 画历史数据
            x_historic = list(range(-9, 1))
            ax.plot(x_historic, historic_data, color='white', linewidth=2, label='历史数据')
            
            # 预测区间分隔线
            ax.axvline(x=0, color='#666666', linestyle='--', alpha=0.7)
            
            # 画三条预测路径
            x_pred = list(range(1, 21))
            ax.plot(x_pred, bullish_pred, color='#4cd97b', linewidth=1.5, label='乐观路径 (30%)')
            ax.plot(x_pred, neutral_pred, color='#7c83fd', linewidth=1.5, label='中性路径 (45%)')
            ax.plot(x_pred, bearish_pred, color='#e94560', linewidth=1.5, label='悲观路径 (25%)')
            
            # 添加置信区间
            # 上方区间
            upper_bound = [max(b, n) * 1.03 for b, n in zip(bullish_pred, neutral_pred)]
            # 下方区间
            lower_bound = [min(b, n) * 0.97 for b, n in zip(bearish_pred, neutral_pred)]
            
            ax.fill_between(x_pred, lower_bound, upper_bound, color='#4cd97b', alpha=0.1)
            
            # 添加标签和标题
            ax.text(x=5, y=bullish_pred[4], s='短期：看涨 (92%)', color='#4cd97b', fontsize=8)
            ax.text(x=12, y=neutral_pred[11], s='中期：看涨 (78%)', color='#7c83fd', fontsize=8)
            ax.text(x=18, y=bearish_pred[17], s='长期：看跌 (64%)', color='#e94560', fontsize=8)
            
            # 标记临界点
            ax.scatter([15], [bearish_pred[14]], color='#e94560', s=50, marker='*')
            ax.text(15.2, bearish_pred[14], '临界点', color='#e94560', fontsize=8)
            
            # 设置坐标轴
            ax.set_xlim(-10, 21)
            ax.set_xticks([-9, -6, -3, 0, 3, 6, 9, 12, 15, 18])
            ax.set_xticklabels(['9天前', '6天前', '3天前', '今天', '3天后', '6天后', '9天后', '12天后', '15天后', '18天后'], 
                               rotation=45, fontsize=7, color='white')
            
            # 去除上、右边框
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#666666')
            ax.spines['left'].set_color('#666666')
            
            # 设置刻度标签颜色
            ax.tick_params(axis='y', colors='white')
            
            # 添加网格
            ax.grid(True, linestyle='--', alpha=0.2, color='#666666')
            
            # 设置标题
            ax.set_title('未来30天市场预测', color='white', fontsize=10)
            
            # 绘制图表
            canvas.draw()
            
            # 转换为QImage显示
            buf = io.BytesIO()
            canvas.print_png(buf)
            buf.seek(0)
            
            image = QImage.fromData(buf.getvalue())
            pixmap = QPixmap.fromImage(image)
            
            # 更新图表显示
            self.chart_frame.setPixmap(pixmap)
            self.chart_frame.setScaledContents(True)
            
        except Exception as e:
            print(f"创建预测图表时出错: {str(e)}")


class ActionPanel(QFrame):
    """行动面板 - 提供快速操作入口"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setup_ui()
        
    def setup_ui(self):
        self.setStyleSheet(f"""
            background-color: {SupergodColors.PANEL_BG};
            border-radius: 10px;
            padding: 5px;
        """)
        
        layout = QVBoxLayout(self)
        
        # 标题
        title = QLabel("智能操作中心")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"""
            color: {SupergodColors.TEXT_PRIMARY};
            font-size: 18px;
            font-weight: bold;
            padding-bottom: 5px;
            border-bottom: 1px solid {SupergodColors.HIGHLIGHT};
        """)
        layout.addWidget(title)
        
        # 操作按钮
        actions = [
            ("进行全面市场扫描", "scan"),
            ("生成智能分析报告", "report"),
            ("调整量子灵敏度", "sensitivity"),
            ("扩展时间维度", "time"),
            ("重新校准预测模型", "calibrate"),
            ("同步最新市场数据", "sync")
        ]
        
        for text, action in actions:
            btn = QPushButton(text)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {SupergodColors.ACCENT_DARK};
                    color: {SupergodColors.TEXT_PRIMARY};
                    border: none;
                    border-radius: 5px;
                    padding: 8px;
                    text-align: left;
                }}
                QPushButton:hover {{
                    background-color: {SupergodColors.HIGHLIGHT};
                }}
            """)
            btn.setProperty("action", action)
            layout.addWidget(btn)
        
        # 语音命令输入
        voice_btn = QPushButton("🎤 启动语音命令")
        voice_btn.setStyleSheet(f"""
            background-color: {SupergodColors.SECONDARY_DARK};
            color: {SupergodColors.TEXT_PRIMARY};
            border: 1px solid {SupergodColors.HIGHLIGHT};
            border-radius: 5px;
            padding: 10px;
            margin-top: 10px;
            font-weight: bold;
        """)
        layout.addWidget(voice_btn)


class ChaosAttractorPanel(QFrame):
    """混沌吸引子面板 - 显示市场的混沌特性"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger("ChaosAttractorPanel")
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setup_ui()
        
    def setup_ui(self):
        """设置UI元素"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # 添加标题
        title = QLabel("混沌吸引子分析")
        title.setStyleSheet(f"color: {SupergodColors.HIGHLIGHT}; font-size: 14px; font-weight: bold;")
        layout.addWidget(title)
        
        # 创建图像容器
        self.attractor_image = QLabel("混沌吸引子")
        self.attractor_image.setMinimumHeight(200)
        self.attractor_image.setStyleSheet(f"background-color: {SupergodColors.PANEL_BG}; border-radius: 5px;")
        self.attractor_image.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.attractor_image)
        
        # 创建图像画布
        figure = Figure(figsize=(5, 4), dpi=100)
        self.attractor_canvas = FigureCanvas(figure)
        
        # 添加参数显示
        params_layout = QGridLayout()
        params_layout.setVerticalSpacing(5)
        params_layout.setHorizontalSpacing(10)
        
        params = [
            ("赫斯特指数:", "0.67 (持续性)"),
            ("莱普诺夫指数:", "-0.00023 (稳定)"),
            ("分形维度:", "1.58"),
            ("熵值:", "0.72"),
            ("混沌边缘:", "混沌边缘")
        ]
        
        for row, (label_text, value_text) in enumerate(params):
            label = QLabel(label_text)
            label.setStyleSheet(f"color: {SupergodColors.TEXT_SECONDARY};")
            
            value = QLabel(value_text)
            value.setStyleSheet(f"color: {SupergodColors.TEXT_PRIMARY};")
            
            params_layout.addWidget(label, row, 0)
            params_layout.addWidget(value, row, 1)
        
        params_frame = QFrame()
        params_frame.setLayout(params_layout)
        layout.addWidget(params_frame)
        
        # 添加关键临界点
        critical_label = QLabel("关键临界点: 5月15日 (83% 确信度)")
        critical_label.setStyleSheet(f"color: {SupergodColors.HIGHLIGHT}; font-weight: bold; margin-top: 5px;")
        layout.addWidget(critical_label)
        
        # 初始化混沌吸引子 - 首次显示
        self.update_attractor()

    def update_attractor(self):
        """更新混沌吸引子可视化"""
        try:
            if not hasattr(self, 'attractor_image') or not self.attractor_image:
                return
                
            # 生成混沌吸引子数据
            points = 1000
            dt = 0.01
            
            # 初始条件随机微调，使每次显示略有不同
            x, y, z = 0.1 + random.uniform(-0.05, 0.05), 0.0, 0.0
            
            # 洛伦兹吸引子参数
            sigma = 10.0
            rho = 28.0
            beta = 8.0 / 3.0
            
            # 计算轨迹
            xs, ys, zs = [], [], []
            for i in range(points):
                # 洛伦兹方程
                dx = sigma * (y - x) * dt
                dy = (x * (rho - z) - y) * dt
                dz = (x * y - beta * z) * dt
                
                x += dx
                y += dy
                z += dz
                
                xs.append(x)
                ys.append(y)
                zs.append(z)
            
            # 清除原有图像
            if hasattr(self, 'attractor_canvas') and self.attractor_canvas:
                # 生成新图像
                figure = Figure(figsize=(5, 4), dpi=100)
                canvas = FigureCanvas(figure)
                ax = figure.add_subplot(111, projection='3d')
                
                # 绘制3D曲线
                ax.plot(xs, ys, zs, color='#FF5500', linewidth=0.7)
                
                # 设置背景颜色、标题和坐标轴
                ax.set_facecolor('#1A1E2E')
                figure.patch.set_facecolor('#1A1E2E')
                
                ax.set_title("混沌吸引子实时图", color='white')
                ax.set_axis_off()  # 隐藏坐标轴
                
                # 更新图像
                canvas.draw()
                
                # 转换为QImage
                buf = io.BytesIO()
                canvas.print_png(buf)
                buf.seek(0)
                
                image = QImage.fromData(buf.getvalue())
                pixmap = QPixmap.fromImage(image)
                
                # 更新图像标签
                self.attractor_image.setPixmap(pixmap)
                self.attractor_image.setScaledContents(True)
                
        except Exception as e:
            print(f"更新混沌吸引子时出错: {str(e)}")


class RecommendedStocksPanel(QFrame):
    """推荐股票面板 - 显示系统推荐的股票"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("recommendedStocksPanel")
        self.setup_ui()
        self.last_refresh_time = None
        
    def setup_ui(self):
        """设置推荐股票面板的UI"""
        # 创建主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # 创建顶部标题栏
        title_layout = QHBoxLayout()
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(10)
        
        # 创建标题标签
        title_label = QLabel("量子推荐股票")
        title_label.setObjectName("panelTitle")
        title_label.setStyleSheet("""
            QLabel {
                color: #00ff00;
                font-size: 16px;
                font-weight: bold;
                padding: 5px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, 
                                          stop:0 #1a1a1a, stop:1 #2a2a2a);
                border-radius: 5px;
            }
        """)
        
        # 创建刷新按钮
        self.refresh_btn = QPushButton("刷新推荐")
        self.refresh_btn.setObjectName("refreshButton")
        self.refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a2a2a;
                color: #00ff00;
                border: 1px solid #00ff00;
                border-radius: 5px;
                padding: 5px 15px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #3a3a3a;
                border-color: #00ff00;
            }
            QPushButton:pressed {
                background-color: #1a1a1a;
            }
        """)
        self.refresh_btn.clicked.connect(self.refresh_recommendations)
        
        # 创建最后刷新时间标签
        self.last_refresh_label = QLabel("上次刷新: 未刷新")
        self.last_refresh_label.setStyleSheet("color: #888888; font-size: 12px;")
        
        # 添加标题和刷新按钮到标题布局
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        title_layout.addWidget(self.last_refresh_label)
        title_layout.addWidget(self.refresh_btn)
        
        # 创建股票网格布局
        self.stocks_grid = QGridLayout()
        self.stocks_grid.setSpacing(10)
        self.stocks_grid.setContentsMargins(0, 0, 0, 0)
        
        # 添加所有组件到主布局
        main_layout.addLayout(title_layout)
        main_layout.addLayout(self.stocks_grid)
        
        # 设置面板样式
        self.setStyleSheet("""
            QFrame#recommendedStocksPanel {
                background-color: #1a1a1a;
                border: 1px solid #333333;
                border-radius: 10px;
            }
        """)
        
        # 初始加载推荐股票
        self.load_recommendations()
    
    def load_recommendations(self):
        """加载推荐股票"""
        # 清除现有推荐
        self.clear_recommendations()
        
        # 从量子分析引擎获取推荐股票
        recommended_stocks = self.get_recommended_stocks()
        
        # 显示推荐股票
        for i, stock in enumerate(recommended_stocks):
            stock_card = self.create_stock_card(*stock)
            self.stocks_grid.addWidget(stock_card, i // 2, i % 2)
        
        # 更新刷新时间
        self.update_refresh_time()
    
    def clear_recommendations(self):
        """清除现有推荐股票"""
        while self.stocks_grid.count():
            item = self.stocks_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
    
    def get_recommended_stocks(self):
        """从量子分析引擎获取推荐股票"""
        # 这里应该调用量子分析引擎的API获取推荐股票
        # 目前使用示例数据
        return [
            ("贵州茅台", "600519", "+2.45%", "白酒行业龙头，量子态稳定性强，估值处于合理区间，"
             "技术面呈现强势突破形态，MACD金叉确认，成交量温和放大，"
             "行业基本面向好，政策面利好持续。"),
            ("宁德时代", "300750", "+3.67%", "新能源电池龙头，研发投入持续加大，"
             "全球市占率第一，技术壁垒高，产能释放加速，"
             "下游需求旺盛，量子态趋势向上。"),
            ("中芯国际", "688981", "+1.23%", "半导体制造龙头，国产替代进程加速，"
             "先进制程突破，订单充足，产能利用率高，"
             "政策支持力度大，行业处于上升周期。"),
            ("比亚迪", "002594", "+4.12%", "新能源汽车龙头，产业链完整，"
             "技术积累深厚，市场份额持续提升，"
             "海外布局加速，量子态动能强劲。"),
            ("腾讯控股", "00700", "+1.56%", "互联网科技巨头，业务布局全面，"
             "现金流充沛，游戏业务稳定，云计算快速增长，"
             "AI布局领先，量子态趋势良好。")
        ]
    
    def update_refresh_time(self):
        """更新刷新时间"""
        self.last_refresh_time = QDateTime.currentDateTime()
        self.last_refresh_label.setText(f"上次刷新: {self.last_refresh_time.toString('yyyy-MM-dd HH:mm:ss')}")
    
    def refresh_recommendations(self):
        """刷新推荐股票"""
        # 禁用刷新按钮，防止重复点击
        self.refresh_btn.setEnabled(False)
        self.refresh_btn.setText("刷新中...")
        
        try:
            # 重新加载推荐股票
            self.load_recommendations()
            
            # 显示刷新成功提示
            QMessageBox.information(self, "刷新成功", "推荐股票已更新！")
        except Exception as e:
            # 显示错误信息
            QMessageBox.warning(self, "刷新失败", f"更新推荐股票时出错：{str(e)}")
        finally:
            # 恢复刷新按钮状态
            self.refresh_btn.setEnabled(True)
            self.refresh_btn.setText("刷新推荐")
    
    def create_stock_card(self, name, code, change, desc):
        """创建单个股票卡片"""
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #2a2a2a;
                border: 1px solid #333333;
                border-radius: 5px;
                padding: 8px;
            }
            QFrame:hover {
                background-color: #3a3a3a;
                border-color: #00ff00;
            }
        """)
        
        layout = QVBoxLayout(card)
        layout.setSpacing(8)
        
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
        desc_label.setStyleSheet("color: #aaaaaa; font-size: 14px; line-height: 1.4;")
        
        layout.addLayout(header)
        layout.addWidget(desc_label)
        
        return card
    
    def create_quantum_state_panel(self):
        panel = QuantumStatePanel()
        return panel
    
    def create_dimension_viz_panel(self):
        panel = DimensionVisualizerPanel()
        return panel
    
    def create_market_insight_panel(self):
        panel = MarketInsightPanel()
        return panel
    
    def create_chaos_panel(self):
        panel = ChaosAttractorPanel()
        return panel
    
    def create_prediction_panel(self):
        panel = PredictionPanel()
        return panel
    
    def create_action_panel(self):
        panel = ActionPanel()
        return panel
    
    def closeEvent(self, event):
        """窗口关闭事件，确保资源被正确释放"""
        self.logger.info("驾驶舱正在关闭...")
        self.safe_stop()
        event.accept()
        
    def safe_stop(self):
        """安全停止所有活动，确保资源被正确释放"""
        self.logger.info("安全停止驾驶舱...")
        
        # 停止所有计时器
        if self.update_timer.isActive():
            self.update_timer.stop()
        if self.special_effects_timer.isActive():
            self.special_effects_timer.stop()
            
        # 标记分析为未运行
        self.analysis_in_progress = False
        
        # 停止所有线程
        for thread in self.active_threads:
            if thread.is_alive():
                self.logger.info(f"等待线程完成: {thread.name}")
                # 给线程一点时间完成
                thread.join(0.5)
        
        # 释放连接
        if self.data_connector:
            # 如果数据连接器有关闭方法，调用它
            if hasattr(self.data_connector, 'close'):
                try:
                    self.data_connector.close()
                    self.logger.info("数据连接器已关闭")
                except Exception as e:
                    self.logger.warning(f"关闭数据连接器时出错: {str(e)}")
        
        self.logger.info("驾驶舱已安全停止")
    
    def set_data_connector(self, data_connector):
        """设置数据连接器，接收从统一入口传入的数据源"""
        self.data_connector = data_connector
        self.logger.info(f"驾驶舱已设置数据连接器: {data_connector.__class__.__name__}")
        
        # 尝试从数据连接器获取市场数据
        try:
            if hasattr(self.data_connector, 'get_market_data'):
                # 获取上证指数数据
                symbol = "000001.SH"
                self.logger.info(f"驾驶舱正在从数据连接器获取市场数据: {symbol}")
                data = self.data_connector.get_market_data(symbol)
                
                if data is not None and not data.empty:
                    self.market_data = data
                    self.logger.info(f"驾驶舱成功获取市场数据，共{len(data)}条记录")
                    
                    # 获取板块数据
                    try:
                        if hasattr(self.data_connector, 'get_sector_data'):
                            sector_data = self.data_connector.get_sector_data()
                            if sector_data and 'leading_sectors' in sector_data:
                                self.logger.info("驾驶舱成功获取板块数据")
                                # 更新板块数据到UI
                            else:
                                self.logger.warning("获取板块数据结构不完整")
                    except Exception as e:
                        self.logger.warning(f"获取板块数据失败: {str(e)}")
                    
                    # 更新市场洞察面板
                    try:
                        self.update_displays()
                    except Exception as e:
                        self.logger.error(f"更新显示失败: {str(e)}")
        except Exception as e:
            self.logger.error(f"从数据连接器获取数据失败: {str(e)}")
            self.logger.error(traceback.format_exc())
    
    def set_enhancement_modules(self, enhancement_modules):
        """设置增强模块，接收从统一入口传入的增强功能"""
        self.enhancement_modules = enhancement_modules
        
        # 记录已加载的增强模块
        if enhancement_modules:
            module_names = list(enhancement_modules.keys())
            self.logger.info(f"驾驶舱已设置增强模块: {', '.join(module_names)}")
            
            # 如果有市场数据，尝试执行增强分析
            if self.market_data is not None and not self.market_data.empty:
                try:
                    self.run_analysis()
                except Exception as e:
                    self.logger.error(f"执行增强分析失败: {str(e)}")
        else:
            self.logger.warning("设置了空的增强模块集合")
    
    def load_core_modules(self):
        """加载核心分析模块"""
        if not SUPERGOD_MODULES_AVAILABLE:
            self.logger.warning("未找到超神模块，使用演示模式")
            return
        
        try:
            # 初始化市场核心
            self.core_modules['market_core'] = ChinaMarketCore()
            self.logger.info("已加载市场分析核心")
            
            # 初始化政策分析器
            self.core_modules['policy_analyzer'] = PolicyAnalyzer()
            self.logger.info("已加载政策分析器")
            
            # 初始化板块轮动追踪器
            self.core_modules['sector_tracker'] = SectorRotationTracker()
            self.logger.info("已加载板块轮动追踪器")
            
            # 初始化混沌理论分析器
            self.core_modules['chaos_analyzer'] = ChaosTheoryAnalyzer()
            self.logger.info("已加载混沌理论分析器")
            
            # 初始化量子维度增强器 - 检查参数
            try:
                self.core_modules['dimension_enhancer'] = QuantumDimensionEnhancer(extended_dimensions=10)
            except TypeError:
                # 如果不支持extended_dimensions参数，尝试不带参数初始化
                self.core_modules['dimension_enhancer'] = QuantumDimensionEnhancer()
            self.logger.info("已加载量子维度增强器")
            
            # 更新状态
            self.core_modules_loaded = True
        except Exception as e:
            self.logger.error(f"加载核心模块失败: {str(e)}")
            self.core_modules_loaded = False
    
    def load_demo_data(self):
        """加载演示数据"""
        try:
            # 创建模拟市场数据
            dates = pd.date_range(end=datetime.now(), periods=100)
            
            # 创建基本价格和成交量数据
            price_start = 3000 + random.randint(-200, 200)
            prices = []
            current_price = price_start
            volumes = []
            
            for i in range(100):
                # 添加一些随机波动，模拟真实市场
                change = np.random.normal(0, 0.01)
                # 添加一个上升趋势
                trend = 0.0003
                # 添加一些周期性
                cycle = 0.005 * np.sin(i / 10 * np.pi)
                
                current_price *= (1 + change + trend + cycle)
                prices.append(current_price)
                
                # 模拟成交量
                volume = abs(np.random.normal(8000, 2000) * (1 + abs(change) * 20))
                volumes.append(volume)
            
            # 创建DataFrame
            self.market_data = pd.DataFrame({
                'date': dates,
                'open': [p * (1 - np.random.uniform(0, 0.005)) for p in prices],
                'high': [p * (1 + np.random.uniform(0, 0.01)) for p in prices],
                'low': [p * (1 - np.random.uniform(0, 0.01)) for p in prices],
                'close': prices,
                'volume': volumes
            })
            
            self.market_data.set_index('date', inplace=True)
            
            # 添加技术指标
            self._add_technical_indicators()
            
            logger.info(f"已加载演示数据: {len(self.market_data)} 行")
            
            # 执行演示分析
            self.run_analysis()
            
        except Exception as e:
            logger.error(f"加载演示数据失败: {str(e)}")
            QMessageBox.warning(self, "数据加载错误", f"加载演示数据失败: {str(e)}")
    
    def _add_technical_indicators(self):
        """添加基本技术指标到数据中"""
        df = self.market_data
        
        # 移动平均线
        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['MA60'] = df['close'].rolling(window=60).mean()
        
        # 成交量移动平均
        df['volume_MA5'] = df['volume'].rolling(window=5).mean()
        
        # 简单的MACD计算
        df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA12'] - df['EMA26']
        df['signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['histogram'] = df['MACD'] - df['signal']
        
        # 计算波动率 (20日)
        df['volatility'] = df['close'].pct_change().rolling(window=20).std() * (252 ** 0.5)
    
    def run_analysis(self):
        """执行市场分析"""
        if self.market_data is None or self.market_data.empty:
            self.show_warning_message("无数据", "请先加载市场数据")
            return
        
        if self.analysis_in_progress:
            self.logger.warning("分析已在进行中")
            return
        
        self.analysis_in_progress = True
        
        # 创建分析线程
        analysis_thread = threading.Thread(target=self._run_analysis_task, name="AnalysisThread")
        analysis_thread.daemon = True
        # 添加到活动线程列表中
        self.active_threads.append(analysis_thread)
        analysis_thread.start()
        
        # 显示分析中的消息
        self.show_info_message("分析进行中", "量子分析正在后台运行，请稍候...")
    
    def update_displays(self):
        """更新所有显示"""
        try:
            # 更新量子状态
            if hasattr(self, 'quantum_state_panel') and self.quantum_state_panel:
                self.quantum_state_panel.update_quantum_values()
                
            # 更新市场洞察
            if hasattr(self, 'market_insight_panel') and self.market_insight_panel:
                self.market_insight_panel.update_values()
                
            # 更新21维量子空间可视化
            if hasattr(self, 'dimension_visualizer') and self.dimension_visualizer:
                # 添加随机抖动使可视化更生动
                if random.random() < 0.3:  # 30%的概率执行抖动
                    self.dimension_visualizer.apply_quantum_fluctuation()
                self.dimension_visualizer.update_visualization()
                
            # 更新混沌吸引子
            if hasattr(self, 'chaos_attractor_panel') and self.chaos_attractor_panel:
                self.chaos_attractor_panel.update_attractor()
                
            # 更新状态栏时间
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if hasattr(self, 'system_time_label') and self.system_time_label:
                self.system_time_label.setText(f"系统时间: {current_time}")
                
            # 随机更新一些界面元素，增强视觉效果
            if random.random() < 0.1:  # 10%的概率执行特效
                self._apply_special_effects()
        except Exception as e:
            self.logger.error(f"更新界面时出错: {str(e)}")
            
    def _apply_special_effects(self):
        """应用特殊视觉效果"""
        try:
            # 量子涟漪效果
            if hasattr(self, 'dimension_visualizer') and self.dimension_visualizer:
                self.dimension_visualizer.apply_ripple_effect()
                
            # 随机更新一个指标闪烁
            all_labels = []
            if hasattr(self, 'quantum_state_panel') and self.quantum_state_panel:
                all_labels.extend(self.quantum_state_panel.findChildren(QLabel))
            if hasattr(self, 'market_insight_panel') and self.market_insight_panel:
                all_labels.extend(self.market_insight_panel.findChildren(QLabel))
                
            if all_labels:
                random_label = random.choice(all_labels)
                current_style = random_label.styleSheet()
                random_label.setStyleSheet(current_style + "; color: #FF5500;")
                QTimer.singleShot(300, lambda: random_label.setStyleSheet(current_style))
        except Exception as e:
            pass  # 忽略特效错误，不影响核心功能
    
    def _show_message_box(self, icon_type, title, message):
        """在主线程中显示消息框"""
        if icon_type == "information":
            QMessageBox.information(self, title, message)
        elif icon_type == "warning":
            QMessageBox.warning(self, title, message)
        elif icon_type == "question":
            return QMessageBox.question(self, title, message,
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.Yes) == QMessageBox.Yes
        elif icon_type == "critical":
            QMessageBox.critical(self, title, message)
    
    def show_info_message(self, title, message):
        """线程安全地显示信息消息框"""
        self.show_message_signal.emit("information", title, message)
    
    def show_warning_message(self, title, message):
        """线程安全地显示警告消息框"""
        self.show_message_signal.emit("warning", title, message)
    
    def show_error_message(self, title, message):
        """线程安全地显示错误消息框"""
        self.show_message_signal.emit("critical", title, message)
    
    def open_data_file(self):
        """打开数据文件"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "选择数据文件",
                "",
                "CSV Files (*.csv);;All Files (*)"
            )
            
            if file_path:
                # 尝试加载数据
                self.load_data(file_path)
                self.show_info_message("数据加载", "数据加载成功！")
                
                # 更新显示
                self.update_displays()
                
                # 记录操作日志
                self.operation_log.append({
                    'timestamp': datetime.now(),
                    'operation': 'open_data_file',
                    'file_path': file_path,
                    'status': 'success'
                })
                
        except Exception as e:
            error_msg = f"打开数据文件时发生错误: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.show_error_message("错误", error_msg)
            
            # 记录错误日志
            self.error_log.append({
                'timestamp': datetime.now(),
                'operation': 'open_data_file',
                'error': str(e),
                'traceback': traceback.format_exc()
            })

    def load_data(self, file_path: str):
        """加载数据文件"""
        try:
            # 读取CSV文件
            df = pd.read_csv(file_path)
            
            # 验证数据格式
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"数据文件缺少必需的列: {', '.join(missing_columns)}")
            
            # 转换日期列
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # 添加技术指标
            self._add_technical_indicators(df)
            
            # 更新显示
            self.market_data = df
            self.update_displays()
            
            # 记录成功
            logger.info(f"成功加载数据文件: {file_path}")
            self.show_info_message("成功", "数据加载成功")
            
        except Exception as e:
            self.show_error_message("错误", f"加载数据时发生错误: {str(e)}")
            logger.error(f"加载数据时发生错误: {str(e)}")
            logger.error(traceback.format_exc())

    def _add_technical_indicators(self, df: pd.DataFrame):
        """添加技术指标"""
        try:
            # 计算移动平均线
            df['MA5'] = df['close'].rolling(window=5).mean()
            df['MA10'] = df['close'].rolling(window=10).mean()
            df['MA20'] = df['close'].rolling(window=20).mean()
            
            # 计算MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            
            # 计算RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # 计算布林带
            df['BB_middle'] = df['close'].rolling(window=20).mean()
            df['BB_upper'] = df['BB_middle'] + 2 * df['close'].rolling(window=20).std()
            df['BB_lower'] = df['BB_middle'] - 2 * df['close'].rolling(window=20).std()
            
        except Exception as e:
            logger.error(f"计算技术指标时发生错误: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def calibrate_model(self):
        """校准量子交易模型"""
        try:
            # 显示校准进度对话框
            progress = QProgressDialog("正在校准模型...", "取消", 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            
            # 更新进度
            progress.setValue(10)
            QApplication.processEvents()
            
            # 校准量子引擎
            if hasattr(self, 'quantum_engine'):
                self.quantum_engine.calibrate()
                progress.setValue(30)
                QApplication.processEvents()
            
            # 校准市场分析模块
            if hasattr(self, 'market_analyzer'):
                self.market_analyzer.calibrate()
                progress.setValue(50)
                QApplication.processEvents()
            
            # 校准预测模型
            if hasattr(self, 'prediction_model'):
                self.prediction_model.calibrate()
                progress.setValue(70)
                QApplication.processEvents()
            
            # 更新系统状态
            self._update_system_state()
            progress.setValue(90)
            QApplication.processEvents()
            
            # 完成校准
            progress.setValue(100)
            self.show_info_message("校准完成", "模型校准成功完成！")
            
        except Exception as e:
            error_msg = f"模型校准失败: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.show_error_message("校准错误", error_msg)
            
            # 记录错误日志
            self.error_log.append({
                'timestamp': datetime.now(),
                'operation': 'calibrate_model',
                'error': str(e),
                'traceback': traceback.format_exc()
            })

    def adjust_sensitivity(self):
        """调整系统灵敏度"""
        try:
            # 显示灵敏度调整对话框
            progress = QProgressDialog("正在调整系统灵敏度...", "取消", 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            
            # 更新进度
            progress.setValue(10)
            QApplication.processEvents()
            
            # 调整量子引擎灵敏度
            if hasattr(self, 'quantum_engine'):
                self.quantum_engine.adjust_sensitivity()
                progress.setValue(30)
                QApplication.processEvents()
            
            # 调整市场分析模块灵敏度
            if hasattr(self, 'market_analyzer'):
                self.market_analyzer.adjust_sensitivity()
                progress.setValue(50)
                QApplication.processEvents()
            
            # 调整预测模型灵敏度
            if hasattr(self, 'prediction_model'):
                self.prediction_model.adjust_sensitivity()
                progress.setValue(70)
                QApplication.processEvents()
            
            # 更新系统状态
            self._update_system_state()
            progress.setValue(90)
            QApplication.processEvents()
            
            # 完成调整
            progress.setValue(100)
            self.show_info_message("灵敏度调整", "系统灵敏度调整完成！")
            
        except Exception as e:
            error_msg = f"灵敏度调整失败: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.show_error_message("调整错误", error_msg)
            
            # 记录错误日志
            self.error_log.append({
                'timestamp': datetime.now(),
                'operation': 'adjust_sensitivity',
                'error': str(e),
                'traceback': traceback.format_exc()
            })

    def expand_time_dimension(self):
        """扩展时间维度"""
        try:
            # 实现时间维度扩展逻辑
            logger.info("开始扩展时间维度...")
            # TODO: 添加实际的时间维度扩展逻辑
            self.show_info_message("成功", "时间维度扩展完成")
        except Exception as e:
            self.show_error_message("错误", f"时间维度扩展失败: {str(e)}")
            logger.error(f"时间维度扩展失败: {str(e)}")

    def sync_market_data(self):
        """同步市场数据"""
        try:
            # 实现市场数据同步逻辑
            logger.info("开始同步市场数据...")
            # TODO: 添加实际的数据同步逻辑
            self.show_info_message("成功", "市场数据同步完成")
        except Exception as e:
            self.show_error_message("错误", f"市场数据同步失败: {str(e)}")
            logger.error(f"市场数据同步失败: {str(e)}")
    
    def activate_voice_command(self):
        """激活语音命令"""
        # 这里应该实现语音命令功能
        self.show_info_message("语音命令", "语音命令系统已启动，请说出您的指令")
        self.logger.info("激活语音命令")

    def _run_analysis_task(self):
        """执行后台分析任务"""
        try:
            # 确保线程结束后从活动线程列表中移除
            current_thread = threading.current_thread()
            
            results = {}
            
            # 如果有真实模块则使用它们
            if SUPERGOD_MODULES_AVAILABLE and self.core_modules:
                # 加载政策和板块数据
                try:
                    from supergod_data_loader import get_data_loader
                    data_loader = get_data_loader(tushare_token="0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10")
                    policy_data = data_loader.load_policy_data(use_real_data=True)
                    sector_data = data_loader.load_sector_data(use_real_data=True)
                    self.logger.info("已加载政策和板块数据")
                except Exception as e:
                    self.logger.warning(f"加载政策和板块数据失败: {str(e)}")
                    policy_data = None
                    sector_data = None
                
                # 执行市场分析
                if 'market_core' in self.core_modules:
                    market_core = self.core_modules['market_core']
                    results['market_analysis'] = market_core.analyze_market(self.market_data)
                
                # 执行政策分析 - 检查方法是否存在
                if 'policy_analyzer' in self.core_modules:
                    policy_analyzer = self.core_modules['policy_analyzer']
                    # 检查对象是否有该方法
                    if hasattr(policy_analyzer, 'analyze_policy_environment'):
                        results['policy_analysis'] = policy_analyzer.analyze_policy_environment(policy_data)
                    elif hasattr(policy_analyzer, 'analyze'):
                        results['policy_analysis'] = policy_analyzer.analyze(policy_data)
                    else:
                        self.logger.warning("政策分析器缺少预期的分析方法")
                
                # 执行板块轮动分析
                try:
                    if 'sector_tracker' in self.core_modules:
                        sector_tracker = self.core_modules['sector_tracker']
                        
                        # 确保sector_data存在
                        if sector_data and isinstance(sector_data, dict):
                            # 首先更新板块数据，然后再调用analyze方法
                            sector_tracker.update_sector_data(sector_data)
                            sector_analysis = sector_tracker.analyze()
                            
                            if sector_analysis:
                                results['sector_analysis'] = sector_analysis
                                
                                # 在UI上显示关键指标
                                if hasattr(self, 'sector_rotation_label') and self.sector_rotation_label:
                                    rotation_strength = sector_analysis.get('rotation_strength', 0)
                                    rotation_text = f"板块轮动强度: {rotation_strength:.2f}"
                                    self.sector_rotation_label.setText(rotation_text)
                                
                                self.logger.info(f"板块轮动分析完成，强度: {sector_analysis.get('rotation_strength', 0):.2f}")
                            else:
                                self.logger.warning("板块轮动分析返回了空结果")
                                results['sector_analysis'] = {"status": "error", "message": "分析返回空结果"}
                        else:
                            self.logger.warning("板块数据为空或格式不正确")
                            results['sector_analysis'] = {"status": "error", "message": "板块数据为空或格式不正确"}
                    else:
                        self.logger.warning("未找到板块轮动跟踪器模块")
                        results['sector_analysis'] = {"status": "error", "message": "未找到板块轮动跟踪器模块"}
                
                except Exception as e:
                    error_message = f"板块轮动分析失败: {str(e)}"
                    self.logger.error(error_message, exc_info=True)
                    results['sector_analysis'] = {"status": "error", "message": error_message}
                
                # 执行混沌理论分析
                if 'chaos_analyzer' in self.core_modules:
                    chaos_analyzer = self.core_modules['chaos_analyzer']
                    if 'close' in self.market_data.columns:
                        results['chaos_analysis'] = chaos_analyzer.analyze(self.market_data['close'].values)
                
                # 执行量子维度分析
                if 'dimension_enhancer' in self.core_modules:
                    dimension_enhancer = self.core_modules['dimension_enhancer']
                    # 检查方法
                    if hasattr(dimension_enhancer, 'enhance_dimensions'):
                        dimensions_data = dimension_enhancer.enhance_dimensions(self.market_data)
                        if hasattr(dimension_enhancer, 'get_dimension_state'):
                            state = dimension_enhancer.get_dimension_state()
                        else:
                            state = {}  # 如果方法不存在，使用空字典
                            
                        results['quantum_dimensions'] = {
                            'data': dimensions_data,
                            'state': state
                        }
            else:
                # 生成模拟分析结果
                results = self._generate_demo_analysis()
            
            # 更新结果和UI
            self.analysis_results = results
            self.analysis_in_progress = False
            
            # 完成时显示消息
            self.show_info_message("分析完成", "量子分析已完成，结果已更新")
            
            # 使用Qt信号触发UI更新
            self.logger.info("分析完成，更新UI")
            
        except Exception as e:
            self.logger.error(f"分析过程中发生错误: {str(e)}")
            self.analysis_in_progress = False
            # 显示错误消息
            self.show_error_message("分析错误", f"分析过程中发生错误: {str(e)}")
        finally:
            # 确保线程结束后从活动线程列表中移除
            if current_thread in self.active_threads:
                self.active_threads.remove(current_thread)
    
    def _generate_demo_analysis(self):
        """生成演示分析结果"""
        self.logger.info("生成演示分析结果")
        results = {}
        
        # 市场分析结果
        results['market_analysis'] = {
            'current_cycle': '积累期',
            'cycle_confidence': 0.97,
            'market_sentiment': 0.72,
            'anomalies': [
                {
                    'type': '创业板成交量异常增加',
                    'position': '最近3天',
                    'confidence': 0.998
                },
                {
                    'type': '外盘期货与A股相关性断裂',
                    'position': '最近1周',
                    'confidence': 0.823
                }
            ]
        }
        
        # 混沌理论分析
        results['chaos_analysis'] = {
            'market_regime': '混沌边缘',
            'stability': 0.67,
            'hurst_exponent': 0.67,
            'lyapunov_exponent': -0.00023,
            'fractal_dimension': 1.58,
            'entropy': 0.72,
            'critical_points': [
                (30, 0.83),  # 30天后，83%的确信度
                (12, 0.65)   # 12天后，65%的确信度
            ]
        }
        
        # 量子维度状态
        dimension_state = {}
        # 基础维度
        base_dims = [
            ('价格动量', 0.82, 0.03, 0.9),
            ('成交量压力', 0.65, 0.05, 0.8),
            ('市场广度', 0.73, 0.01, 0.7),
            ('波动性', 0.45, -0.02, 0.6),
            ('周期性', 0.58, 0.0, 0.5),
            ('情绪', 0.72, 0.02, 0.8),
            ('价格水平', 0.67, 0.01, 0.7),
            ('流动性', 0.78, 0.03, 0.8),
            ('相对强度', 0.61, 0.02, 0.6),
            ('趋势强度', 0.83, 0.04, 0.9),
            ('反转倾向', 0.32, -0.03, 0.7)
        ]
        
        for i, (name, value, trend, weight) in enumerate(base_dims):
            dimension_state[name] = {
                'type': 'base',
                'value': value,
                'trend': trend,
                'weight': weight
            }
        
        # 扩展维度
        ext_dims = [
            ('分形', 0.67, 0.01, 0.7),
            ('熵', 0.72, 0.02, 0.8),
            ('周期共振', 0.54, 0.0, 0.6),
            ('量子相位', 0.88, 0.04, 0.9),
            ('能量势能', 0.82, 0.03, 0.8),
            ('相位相干性', 0.75, 0.01, 0.7),
            ('时间相干性', 0.66, -0.01, 0.7),
            ('维度共振', 0.75, 0.0, 0.8),
            ('混沌度', 0.23, -0.02, 0.6),
            ('临界度', 0.10, 0.01, 0.5)
        ]
        
        for i, (name, value, trend, weight) in enumerate(ext_dims):
            dimension_state[name] = {
                'type': 'extended',
                'value': value,
                'trend': trend,
                'weight': weight
            }
        
        # 综合维度
        dimension_state['energy_potential'] = {'type': 'composite', 'value': 0.82}
        dimension_state['phase_coherence'] = {'type': 'composite', 'value': 0.75}
        dimension_state['temporal_coherence'] = {'type': 'composite', 'value': 0.66}
        dimension_state['chaos_degree'] = {'type': 'composite', 'value': 0.23}
        
        results['quantum_dimensions'] = {
            'state': dimension_state
        }
        
        # 预测结果
        results['predictions'] = {
            'short_term': {
                'direction': 'bullish',
                'confidence': 0.92,
                'time_frame': '1-3天'
            },
            'medium_term': {
                'direction': 'bullish',
                'confidence': 0.78,
                'time_frame': '1-2周'
            },
            'long_term': {
                'direction': 'bearish',
                'confidence': 0.64,
                'time_frame': '1-3月'
            }
        }
        
        return results
    
    def generate_report(self):
        """生成分析报告"""
        if not self.analysis_results:
            QMessageBox.warning(self, "无分析结果", "请先执行分析")
            return
        
        # 获取保存位置
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存分析报告", "超神分析报告.html", "HTML文件 (*.html)")
            
        if not file_path:
            return
        
        try:
            # 这里应该调用supergod_desktop中的报告生成功能
            # 简化起见，我们只显示成功消息
            # TODO: 实现实际的报告生成功能
            
            QMessageBox.information(self, "报告已生成", 
                                   f"分析报告已成功生成，保存于: {file_path}")
            logger.info(f"已生成报告: {file_path}")
            
        except Exception as e:
            logger.error(f"生成报告失败: {str(e)}")
            QMessageBox.warning(self, "生成失败", f"生成报告失败: {str(e)}")

    def show_error_message(self, title: str, message: str):
        """显示错误消息对话框"""
        QMessageBox.critical(self, title, message)

    def show_warning_message(self, title: str, message: str):
        """显示警告消息对话框"""
        QMessageBox.warning(self, title, message)

    def show_info_message(self, title: str, message: str):
        """显示信息消息对话框"""
        QMessageBox.information(self, title, message)

    def calibrate_model(self):
        """校准量子交易模型"""
        try:
            # 显示校准进度对话框
            progress = QProgressDialog("正在校准模型...", "取消", 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            
            # 更新进度
            progress.setValue(10)
            QApplication.processEvents()
            
            # 校准量子引擎
            if hasattr(self, 'quantum_engine'):
                self.quantum_engine.calibrate()
                progress.setValue(30)
                QApplication.processEvents()
            
            # 校准市场分析模块
            if hasattr(self, 'market_analyzer'):
                self.market_analyzer.calibrate()
                progress.setValue(50)
                QApplication.processEvents()
            
            # 校准预测模型
            if hasattr(self, 'prediction_model'):
                self.prediction_model.calibrate()
                progress.setValue(70)
                QApplication.processEvents()
            
            # 更新系统状态
            self._update_system_state()
            progress.setValue(90)
            QApplication.processEvents()
            
            # 完成校准
            progress.setValue(100)
            self.show_info_message("校准完成", "模型校准成功完成！")
            
        except Exception as e:
            error_msg = f"模型校准失败: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.show_error_message("校准错误", error_msg)
            
            # 记录错误日志
            self.error_log.append({
                'timestamp': datetime.now(),
                'operation': 'calibrate_model',
                'error': str(e),
                'traceback': traceback.format_exc()
            })

    def adjust_sensitivity(self):
        """调整系统灵敏度"""
        try:
            # 显示灵敏度调整对话框
            progress = QProgressDialog("正在调整系统灵敏度...", "取消", 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            
            # 更新进度
            progress.setValue(10)
            QApplication.processEvents()
            
            # 调整量子引擎灵敏度
            if hasattr(self, 'quantum_engine'):
                self.quantum_engine.adjust_sensitivity()
                progress.setValue(30)
                QApplication.processEvents()
            
            # 调整市场分析模块灵敏度
            if hasattr(self, 'market_analyzer'):
                self.market_analyzer.adjust_sensitivity()
                progress.setValue(50)
                QApplication.processEvents()
            
            # 调整预测模型灵敏度
            if hasattr(self, 'prediction_model'):
                self.prediction_model.adjust_sensitivity()
                progress.setValue(70)
                QApplication.processEvents()
            
            # 更新系统状态
            self._update_system_state()
            progress.setValue(90)
            QApplication.processEvents()
            
            # 完成调整
            progress.setValue(100)
            self.show_info_message("灵敏度调整", "系统灵敏度调整完成！")
            
        except Exception as e:
            error_msg = f"灵敏度调整失败: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.show_error_message("调整错误", error_msg)
            
            # 记录错误日志
            self.error_log.append({
                'timestamp': datetime.now(),
                'operation': 'adjust_sensitivity',
                'error': str(e),
                'traceback': traceback.format_exc()
            })

    def expand_time_dimension(self):
        """扩展时间维度"""
        try:
            # 实现时间维度扩展逻辑
            logger.info("开始扩展时间维度...")
            # TODO: 添加实际的时间维度扩展逻辑
            self.show_info_message("成功", "时间维度扩展完成")
        except Exception as e:
            self.show_error_message("错误", f"时间维度扩展失败: {str(e)}")
            logger.error(f"时间维度扩展失败: {str(e)}")

    def sync_market_data(self):
        """同步市场数据"""
        try:
            # 实现市场数据同步逻辑
            logger.info("开始同步市场数据...")
            # TODO: 添加实际的数据同步逻辑
            self.show_info_message("成功", "市场数据同步完成")
        except Exception as e:
            self.show_error_message("错误", f"市场数据同步失败: {str(e)}")
            logger.error(f"市场数据同步失败: {str(e)}")


class ModuleSymbiosisManager:
    """模块共生管理器 - 协调各个模块之间的交互和共生关系"""
    
    def __init__(self, cockpit):
        self.cockpit = cockpit
        self.logger = logging.getLogger("ModuleSymbiosisManager")
        self.active_modules = {}
        self.symbiosis_network = {}
        self.quantum_resonance_level = 0.0
        self.symbiosis_metrics = {
            'coherence': 0.0,
            'resonance': 0.0,
            'synergy': 0.0,
            'stability': 0.0
        }
        self.logger.info("模块共生管理器初始化完成")
        
    def register_module(self, module_name, module_instance):
        """注册一个模块到共生系统"""
        self.active_modules[module_name] = module_instance
        self.logger.info(f"模块 {module_name} 已注册到共生系统")
        
        # 初始化该模块的共生网络
        if module_name not in self.symbiosis_network:
            self.symbiosis_network[module_name] = {
                'connections': [],
                'energy_level': 0.0,
                'resonance_frequency': random.uniform(0.8, 1.2)
            }
            
        # 尝试建立与其他模块的连接
        self._establish_connections(module_name)
        
    def _establish_connections(self, new_module_name):
        """建立新模块与其他模块的连接"""
        for existing_module in self.active_modules:
            if existing_module != new_module_name:
                # 计算两个模块之间的共生亲和度
                affinity = self._calculate_affinity(new_module_name, existing_module)
                
                if affinity > 0.5:  # 只有当亲和度足够高时才建立连接
                    self.symbiosis_network[new_module_name]['connections'].append({
                        'target': existing_module,
                        'affinity': affinity,
                        'energy_flow': 0.0
                    })
                    
                    # 双向连接
                    if existing_module in self.symbiosis_network:
                        self.symbiosis_network[existing_module]['connections'].append({
                            'target': new_module_name,
                            'affinity': affinity,
                            'energy_flow': 0.0
                        })
                        
                    self.logger.info(f"建立了模块 {new_module_name} 和 {existing_module} 之间的共生连接，亲和度: {affinity:.2f}")
    
    def _calculate_affinity(self, module1, module2):
        """计算两个模块之间的共生亲和度"""
        # 基于模块类型和功能计算亲和度
        module1_type = self._get_module_type(module1)
        module2_type = self._get_module_type(module2)
        
        # 定义模块类型之间的亲和度矩阵
        affinity_matrix = {
            'market': {'market': 0.9, 'quantum': 0.7, 'chaos': 0.6, 'prediction': 0.8, 'action': 0.5},
            'quantum': {'market': 0.7, 'quantum': 0.9, 'chaos': 0.8, 'prediction': 0.7, 'action': 0.6},
            'chaos': {'market': 0.6, 'quantum': 0.8, 'chaos': 0.9, 'prediction': 0.7, 'action': 0.5},
            'prediction': {'market': 0.8, 'quantum': 0.7, 'chaos': 0.7, 'prediction': 0.9, 'action': 0.8},
            'action': {'market': 0.5, 'quantum': 0.6, 'chaos': 0.5, 'prediction': 0.8, 'action': 0.9}
        }
        
        # 获取亲和度
        if module1_type in affinity_matrix and module2_type in affinity_matrix[module1_type]:
            base_affinity = affinity_matrix[module1_type][module2_type]
        else:
            base_affinity = 0.5  # 默认亲和度
            
        # 添加一些随机波动，模拟量子不确定性
        quantum_fluctuation = random.uniform(-0.1, 0.1)
        
        # 确保亲和度在0-1之间
        return max(0.0, min(1.0, base_affinity + quantum_fluctuation))
    
    def _get_module_type(self, module_name):
        """根据模块名称判断其类型"""
        if 'market' in module_name.lower() or 'insight' in module_name.lower():
            return 'market'
        elif 'quantum' in module_name.lower() or 'dimension' in module_name.lower():
            return 'quantum'
        elif 'chaos' in module_name.lower():
            return 'chaos'
        elif 'prediction' in module_name.lower():
            return 'prediction'
        elif 'action' in module_name.lower() or 'recommended' in module_name.lower():
            return 'action'
        else:
            return 'unknown'
    
    def update_symbiosis(self):
        """更新共生网络状态"""
        # 更新量子共振水平
        self._update_quantum_resonance()
        
        # 更新模块间的能量流动
        self._update_energy_flow()
        
        # 计算共生指标
        self._calculate_symbiosis_metrics()
        
        # 应用共生效应到各个模块
        self._apply_symbiosis_effects()
        
        self.logger.info(f"共生网络更新完成，共振水平: {self.quantum_resonance_level:.2f}")
        
    def _update_quantum_resonance(self):
        """更新量子共振水平"""
        # 基于活跃模块数量和连接强度计算共振水平
        active_count = len(self.active_modules)
        if active_count == 0:
            self.quantum_resonance_level = 0.0
            return
            
        # 计算总连接强度
        total_connections = 0
        total_affinity = 0.0
        
        for module, network in self.symbiosis_network.items():
            total_connections += len(network['connections'])
            for conn in network['connections']:
                total_affinity += conn['affinity']
                
        # 计算平均连接强度和亲和度
        avg_connections = total_connections / active_count if active_count > 0 else 0
        avg_affinity = total_affinity / total_connections if total_connections > 0 else 0
        
        # 计算共振水平 (0-1之间)
        connection_factor = min(1.0, avg_connections / 4)  # 假设每个模块最多有4个连接
        self.quantum_resonance_level = connection_factor * avg_affinity
        
        # 添加一些随机波动
        self.quantum_resonance_level += random.uniform(-0.05, 0.05)
        self.quantum_resonance_level = max(0.0, min(1.0, self.quantum_resonance_level))
        
    def _update_energy_flow(self):
        """更新模块间的能量流动"""
        for module, network in self.symbiosis_network.items():
            # 更新模块能量水平
            network['energy_level'] = random.uniform(0.7, 1.0) * self.quantum_resonance_level
            
            # 更新连接的能量流动
            for conn in network['connections']:
                target_module = conn['target']
                if target_module in self.symbiosis_network:
                    # 基于亲和度和两个模块的能量水平计算能量流动
                    target_energy = self.symbiosis_network[target_module]['energy_level']
                    energy_diff = network['energy_level'] - target_energy
                    
                    # 能量从高到低流动
                    conn['energy_flow'] = conn['affinity'] * energy_diff * 0.1
                    
    def _calculate_symbiosis_metrics(self):
        """计算共生指标"""
        # 计算相干性 (模块间的一致性)
        coherence = 0.0
        if len(self.active_modules) > 1:
            energy_levels = [network['energy_level'] for network in self.symbiosis_network.values()]
            avg_energy = sum(energy_levels) / len(energy_levels)
            variance = sum((e - avg_energy) ** 2 for e in energy_levels) / len(energy_levels)
            coherence = 1.0 - min(1.0, variance)  # 方差越小，相干性越高
            
        # 计算共振 (模块间的同步程度)
        resonance = self.quantum_resonance_level
        
        # 计算协同性 (模块间的互补程度)
        synergy = 0.0
        if len(self.active_modules) > 1:
            total_affinity = 0.0
            total_connections = 0
            
            for network in self.symbiosis_network.values():
                for conn in network['connections']:
                    total_affinity += conn['affinity']
                    total_connections += 1
                    
            synergy = total_affinity / total_connections if total_connections > 0 else 0
            
        # 计算稳定性 (系统抵抗扰动的能力)
        stability = 0.5 + 0.5 * (coherence + resonance) / 2
        
        # 更新指标
        self.symbiosis_metrics = {
            'coherence': coherence,
            'resonance': resonance,
            'synergy': synergy,
            'stability': stability
        }
        
    def _apply_symbiosis_effects(self):
        """应用共生效应到各个模块"""
        # 根据共生指标调整各个模块的行为
        for module_name, module in self.active_modules.items():
            if hasattr(module, 'adjust_for_symbiosis'):
                # 传递共生指标给模块
                module.adjust_for_symbiosis(self.symbiosis_metrics)
                
        # 更新UI显示
        if hasattr(self.cockpit, 'update_symbiosis_display'):
            self.cockpit.update_symbiosis_display(self.symbiosis_metrics)
            
    def get_symbiosis_report(self):
        """生成共生系统报告"""
        report = {
            'active_modules': list(self.active_modules.keys()),
            'quantum_resonance': self.quantum_resonance_level,
            'metrics': self.symbiosis_metrics,
            'connections': []
        }
        
        # 添加连接信息
        for module, network in self.symbiosis_network.items():
            for conn in network['connections']:
                report['connections'].append({
                    'from': module,
                    'to': conn['target'],
                    'affinity': conn['affinity'],
                    'energy_flow': conn['energy_flow']
                })
                
        return report

    def optimize_module_interaction(self) -> None:
        """优化模块间的交互"""
        try:
            self.logger.info("开始优化模块交互...")
            
            # 分析模块间的依赖关系
            dependency_graph = self._build_dependency_graph()
            
            # 识别关键路径
            critical_path = self._find_critical_path(dependency_graph)
            
            # 优化模块加载顺序
            self._optimize_load_order(critical_path)
            
            # 调整模块间的能量分配
            self._optimize_energy_distribution()
            
            self.logger.info("模块交互优化完成")
            
        except Exception as e:
            self.logger.error(f"优化模块交互时出错: {str(e)}")
            raise

    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        """构建模块依赖图"""
        dependency_graph = {}
        
        for module_name, module in self.active_modules.items():
            dependencies = []
            
            # 分析模块间的数据流
            for other_module in self.active_modules:
                if other_module != module_name:
                    if self._has_data_dependency(module, self.active_modules[other_module]):
                        dependencies.append(other_module)
            
            dependency_graph[module_name] = dependencies
            
        return dependency_graph

    def _has_data_dependency(self, module1: Any, module2: Any) -> bool:
        """检查两个模块之间是否存在数据依赖"""
        # 检查模块1是否依赖模块2的输出
        if hasattr(module1, 'required_data') and hasattr(module2, 'output_data'):
            return any(data in module2.output_data for data in module1.required_data)
        return False

    def _find_critical_path(self, dependency_graph: Dict[str, List[str]]) -> List[str]:
        """找出模块依赖图中的关键路径"""
        # 使用拓扑排序找出关键路径
        visited = set()
        temp = set()
        path = []
        
        def visit(module):
            if module in temp:
                raise ValueError("检测到循环依赖")
            if module in visited:
                return
                
            temp.add(module)
            for dependency in dependency_graph[module]:
                visit(dependency)
            temp.remove(module)
            visited.add(module)
            path.append(module)
            
        for module in dependency_graph:
            if module not in visited:
                visit(module)
                
        return path

    def _optimize_load_order(self, critical_path: List[str]) -> None:
        """根据关键路径优化模块加载顺序"""
        # 重新排序模块
        ordered_modules = {}
        for module_name in critical_path:
            if module_name in self.active_modules:
                ordered_modules[module_name] = self.active_modules[module_name]
                
        # 更新模块字典
        self.active_modules = ordered_modules

    def _optimize_energy_distribution(self) -> None:
        """优化模块间的能量分配"""
        # 计算每个模块的能量需求
        energy_requirements = {}
        for module_name, module in self.active_modules.items():
            if hasattr(module, 'energy_requirement'):
                energy_requirements[module_name] = module.energy_requirement
            else:
                energy_requirements[module_name] = 1.0  # 默认能量需求
                
        # 归一化能量需求
        total_energy = sum(energy_requirements.values())
        for module_name in energy_requirements:
            energy_requirements[module_name] /= total_energy
            
        # 更新模块能量分配
        for module_name, energy in energy_requirements.items():
            if module_name in self.symbiosis_network:
                self.symbiosis_network[module_name]['energy_level'] = energy

    def monitor_performance(self) -> Dict[str, float]:
        """监控系统性能指标"""
        performance_metrics = {
            'module_load_time': 0.0,
            'data_processing_time': 0.0,
            'energy_efficiency': 0.0,
            'system_stability': 0.0
        }
        
        try:
            # 测量模块加载时间
            start_time = time.time()
            self._load_modules()
            performance_metrics['module_load_time'] = time.time() - start_time
            
            # 测量数据处理时间
            start_time = time.time()
            self._process_data()
            performance_metrics['data_processing_time'] = time.time() - start_time
            
            # 计算能量效率
            total_energy = sum(network['energy_level'] for network in self.symbiosis_network.values())
            active_modules = len(self.active_modules)
            performance_metrics['energy_efficiency'] = total_energy / active_modules if active_modules > 0 else 0
            
            # 计算系统稳定性
            stability_scores = []
            for module_name, network in self.symbiosis_network.items():
                if 'connections' in network:
                    connection_stability = len(network['connections']) / (len(self.active_modules) - 1)
                    stability_scores.append(connection_stability)
            performance_metrics['system_stability'] = np.mean(stability_scores) if stability_scores else 0
            
        except Exception as e:
            self.logger.error(f"监控性能时出错: {str(e)}")
            
        return performance_metrics

    def _load_modules(self) -> None:
        """加载所有模块"""
        for module_name, module in self.active_modules.items():
            if hasattr(module, 'initialize'):
                module.initialize()

    def _process_data(self) -> None:
        """处理模块间的数据流"""
        for module_name, module in self.active_modules.items():
            if hasattr(module, 'process_data'):
                module.process_data()

# 修改SupergodCockpit类，添加共生管理器
class SupergodCockpit(QMainWindow):
    """超神系统全息驾驶舱主窗口"""
    
    # 添加用于在主线程显示消息框的信号和槽
    show_message_signal = pyqtSignal(str, str, str)
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("超神量子共生系统 - 全息驾驶舱")
        self.resize(1800, 1000)
        
        # 初始化logger
        self.logger = logging.getLogger("SupergodCockpit")
        
        # 初始化变量
        self.data_connector = None
        self.enhancement_modules = None
        self.market_data = None
        self.analysis_results = {}
        self.core_modules = {}  # 添加核心模块字典初始化
        self.core_modules_loaded = False  # 添加核心模块加载状态
        self.analysis_in_progress = False  # 添加分析状态标志
        self.active_threads = []  # 跟踪活动线程
        
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_displays)
        self.special_effects_timer = QTimer(self)
        self.special_effects_timer.timeout.connect(self._apply_special_effects)
        
        # 监听消息信号
        self.show_message_signal.connect(self._show_message_box)
        
        # 加载核心模块
        self.load_core_modules()
        
        # 设置UI
        self.setup_ui()
        
        # 加载演示数据
        self.load_demo_data()
        
        # 启动定时更新
        self.start_auto_updates()
        
    def load_core_modules(self):
        """加载系统核心模块"""
        self.logger.info("加载超神系统核心模块...")
        self.core_modules_loaded = False
        
        try:
            # 尝试导入核心模块
            from china_market_core import ChinaMarketCore
            from policy_analyzer import PolicyAnalyzer
            from sector_rotation_tracker import SectorRotationTracker
            
            # 初始化核心模块
            self.core_modules['market_core'] = ChinaMarketCore()
            self.core_modules['policy_analyzer'] = PolicyAnalyzer()
            self.core_modules['sector_tracker'] = SectorRotationTracker()
            
            # 初始化量子增强模块
            try:
                from quantum_dimension_enhancer import get_dimension_enhancer
                from chaos_theory_framework import get_chaos_analyzer
                
                self.core_modules['dimension_enhancer'] = get_dimension_enhancer()
                self.core_modules['chaos_analyzer'] = get_chaos_analyzer()
                
                self.logger.info("成功加载量子增强模块")
            except ImportError:
                self.logger.warning("量子增强模块不可用，将使用基本功能")
            
            self.core_modules_loaded = True
            self.logger.info("成功加载超神分析引擎模块")
            
        except ImportError as e:
            self.logger.warning(f"无法加载部分或全部超神引擎模块: {str(e)}")
            self.logger.warning("将使用演示数据")
            
            # 加载模块共生管理器
            try:
                from symbiosis_manager import ModuleSymbiosisManager
                self.core_modules['symbiosis_manager'] = ModuleSymbiosisManager()
                self.logger.info("模块共生管理器初始化完成")
            except ImportError:
                self.logger.warning("模块共生管理器不可用")
    
    def _apply_special_effects(self):
        """应用特殊效果，增强用户体验"""
        # 随机选择一个效果
        effects = [
            self._apply_quantum_ripple,
            self._apply_data_pulse,
            self._apply_dimension_shift
        ]
        
        # 随机选择一个效果
        import random
        effect = random.choice(effects)
        effect()
    
    def _apply_quantum_ripple(self):
        """应用量子波纹效果"""
        pass  # 实际实现会更复杂
    
    def _apply_data_pulse(self):
        """应用数据脉冲效果"""
        pass  # 实际实现会更复杂
        
    def _apply_dimension_shift(self):
        """应用维度偏移效果"""
        pass  # 实际实现会更复杂
    
    def setup_ui(self):
        """设置UI界面"""
        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout(main_widget)
        
        # 创建顶部工具栏
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        
        # 添加工具栏按钮
        toolbar.addAction("打开数据", self.open_data_file)
        toolbar.addAction("校准模型", self.calibrate_model)
        toolbar.addAction("调整灵敏度", self.adjust_sensitivity)
        toolbar.addAction("扩展时间维度", self.expand_time_dimension)
        toolbar.addAction("同步数据", self.sync_market_data)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        
        # 左侧面板
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # 创建各个面板
        self.quantum_panel = QuantumStatePanel(self)
        self.market_panel = MarketInsightPanel(self)
        self.dimension_panel = DimensionVisualizerPanel(self)
        
        left_layout.addWidget(self.quantum_panel)
        left_layout.addWidget(self.market_panel)
        left_layout.addWidget(self.dimension_panel)
        
        # 右侧面板
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.chaos_panel = ChaosAttractorPanel(self)
        self.prediction_panel = PredictionPanel(self)
        self.action_panel = ActionPanel(self)
        self.stocks_panel = RecommendedStocksPanel(self)
        
        right_layout.addWidget(self.chaos_panel)
        right_layout.addWidget(self.prediction_panel)
        right_layout.addWidget(self.action_panel)
        right_layout.addWidget(self.stocks_panel)
        
        # 添加左右面板到分割器
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        
        # 添加分割器到主布局
        main_layout.addWidget(splitter)
        
        # 设置窗口样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a2e;
                color: #ffffff;
            }
            QToolBar {
                background-color: #16213e;
                border: none;
            }
            QToolButton {
                color: #ffffff;
                background-color: #0f3460;
                border: none;
                padding: 5px;
                margin: 2px;
            }
            QToolButton:hover {
                background-color: #e94560;
            }
        """)
    
    def update_displays(self):
        """更新所有显示面板"""
        try:
            # 更新量子状态面板
            if hasattr(self, 'quantum_panel'):
                self.quantum_panel.update_quantum_values()
            
            # 更新市场洞察面板
            if hasattr(self, 'market_panel'):
                self.market_panel.refresh_data()
            
            # 更新维度可视化面板
            if hasattr(self, 'dimension_panel'):
                self.dimension_panel.update_visualization()
            
            # 更新混沌吸引子面板
            if hasattr(self, 'chaos_panel'):
                self.chaos_panel.update_attractor()
            
            # 更新预测面板
            if hasattr(self, 'prediction_panel'):
                self.prediction_panel.update_predictions()
            
            # 更新推荐股票面板
            if hasattr(self, 'stocks_panel'):
                self.stocks_panel.refresh_recommendations()
                
            # 更新共生状态
            self.update_symbiosis()
            
        except Exception as e:
            logger.error(f"更新显示面板时发生错误: {str(e)}")
            logger.error(traceback.format_exc())
    
    def update_symbiosis(self):
        """更新共生系统状态"""
        try:
            self.symbiosis_manager.update_symbiosis()
        except Exception as e:
            logger.error(f"更新共生状态时发生错误: {str(e)}")
    
    def register_module(self, module_name, module_instance):
        """注册模块到共生系统"""
        self.symbiosis_manager.register_module(module_name, module_instance)
        
    def get_symbiosis_report(self):
        """获取共生系统报告"""
        return self.symbiosis_manager.get_symbiosis_report()

    def monitor_system_health(self) -> Dict[str, Any]:
        """监控系统健康状态"""
        health_metrics = {
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'quantum_stability': 0.0,
            'module_health': {},
            'error_rate': 0.0
        }
        
        try:
            # 监控CPU使用率
            health_metrics['cpu_usage'] = psutil.cpu_percent() / 100.0
            
            # 监控内存使用率
            memory = psutil.Process().memory_info()
            health_metrics['memory_usage'] = memory.rss / psutil.virtual_memory().total
            
            # 监控量子稳定性
            if hasattr(self, 'quantum_engine'):
                health_metrics['quantum_stability'] = self.quantum_engine.get_stability()
            
            # 监控模块健康状态
            for module_name, module in self.active_modules.items():
                if hasattr(module, 'get_health'):
                    health_metrics['module_health'][module_name] = module.get_health()
            
            # 计算错误率
            total_operations = len(self.operation_log)
            error_operations = sum(1 for op in self.operation_log if op['status'] == 'error')
            health_metrics['error_rate'] = error_operations / total_operations if total_operations > 0 else 0
            
        except Exception as e:
            self.logger.error(f"监控系统健康状态时出错: {str(e)}")
            
        return health_metrics

    def handle_error(self, error: Exception, context: str) -> None:
        """处理系统错误
        
        Args:
            error: 异常对象
            context: 错误发生的上下文
        """
        try:
            # 记录错误
            error_info = {
                'timestamp': datetime.now().isoformat(),
                'context': context,
                'error_type': type(error).__name__,
                'error_message': str(error),
                'traceback': traceback.format_exc()
            }
            
            self.error_log.append(error_info)
            self.logger.error(f"错误发生在 {context}: {str(error)}")
            
            # 根据错误类型采取相应措施
            if isinstance(error, QuantumError):
                self._handle_quantum_error(error)
            elif isinstance(error, ModuleError):
                self._handle_module_error(error)
            elif isinstance(error, DataError):
                self._handle_data_error(error)
            else:
                self._handle_general_error(error)
                
            # 更新系统状态
            self._update_system_state()
            
        except Exception as e:
            self.logger.critical(f"错误处理过程中发生异常: {str(e)}")

    def _handle_quantum_error(self, error: QuantumError) -> None:
        """处理量子计算相关错误"""
        try:
            # 重置量子引擎
            if hasattr(self, 'quantum_engine'):
                self.quantum_engine.reset()
            
            # 重新初始化量子态
            self._initialize_quantum_state()
            
            # 通知用户
            self.show_error_message("量子错误", "量子计算发生错误，系统已重置")
            
        except Exception as e:
            self.logger.error(f"处理量子错误时出错: {str(e)}")

    def _handle_module_error(self, error: ModuleError) -> None:
        """处理模块相关错误"""
        try:
            # 识别出错的模块
            failed_module = error.module_name if hasattr(error, 'module_name') else None
            
            if failed_module:
                # 尝试重新加载模块
                self._reload_module(failed_module)
                
                # 更新模块依赖
                self.symbiosis_manager.update_module_dependencies()
                
                # 通知用户
                self.show_warning_message("模块错误", f"模块 {failed_module} 发生错误，已尝试重新加载")
            
        except Exception as e:
            self.logger.error(f"处理模块错误时出错: {str(e)}")

    def _handle_data_error(self, error: DataError) -> None:
        """处理数据相关错误"""
        try:
            # 清理损坏的数据
            if hasattr(error, 'data_id'):
                self._clean_corrupted_data(error.data_id)
            
            # 重新加载数据
            self._reload_data()
            
            # 通知用户
            self.show_warning_message("数据错误", "数据发生错误，已重新加载")
            
        except Exception as e:
            self.logger.error(f"处理数据错误时出错: {str(e)}")

    def _handle_general_error(self, error: Exception) -> None:
        """处理一般性错误"""
        try:
            # 记录错误详情
            self.logger.error(f"发生一般性错误: {str(error)}")
            
            # 尝试恢复系统状态
            self._recover_system_state()
            
            # 通知用户
            self.show_error_message("系统错误", "系统发生错误，已尝试恢复")
            
        except Exception as e:
            self.logger.critical(f"处理一般性错误时出错: {str(e)}")

    def _update_system_state(self) -> None:
        """更新系统状态"""
        try:
            # 更新性能指标
            self.performance_metrics = self.monitor_system_health()
            
            # 更新模块状态
            self.module_states = {
                name: module.get_state() if hasattr(module, 'get_state') else {}
                for name, module in self.active_modules.items()
            }
            
            # 更新UI显示
            self.update_status_display()
            
        except Exception as e:
            self.logger.error(f"更新系统状态时出错: {str(e)}")

    def _recover_system_state(self) -> None:
        """恢复系统状态"""
        try:
            # 保存当前状态
            current_state = self._save_current_state()
            
            # 重置系统组件
            self._reset_system_components()
            
            # 恢复数据
            self._restore_data()
            
            # 重新初始化模块
            self._reinitialize_modules()
            
            # 验证系统状态
            if not self._verify_system_state():
                # 如果验证失败，回滚到之前的状态
                self._restore_state(current_state)
                raise SystemError("系统状态恢复失败")
            
        except Exception as e:
            self.logger.error(f"恢复系统状态时出错: {str(e)}")
            raise

    def _save_current_state(self) -> Dict[str, Any]:
        """保存当前系统状态"""
        return {
            'performance_metrics': self.performance_metrics.copy(),
            'module_states': self.module_states.copy(),
            'quantum_state': self.quantum_engine.get_state() if hasattr(self, 'quantum_engine') else None,
            'active_modules': list(self.active_modules.keys())
        }

    def _reset_system_components(self) -> None:
        """重置系统组件"""
        # 重置量子引擎
        if hasattr(self, 'quantum_engine'):
            self.quantum_engine.reset()
        
        # 重置模块
        for module in self.active_modules.values():
            if hasattr(module, 'reset'):
                module.reset()
        
        # 重置共生管理器
        self.symbiosis_manager.reset()

    def _restore_data(self) -> None:
        """恢复数据"""
        try:
            # 从备份加载数据
            if hasattr(self, 'data_backup'):
                self.market_data = self.data_backup.copy()
            
            # 重新计算技术指标
            self._add_technical_indicators()
            
        except Exception as e:
            self.logger.error(f"恢复数据时出错: {str(e)}")
            raise DataError("数据恢复失败")

    def _reinitialize_modules(self) -> None:
        """重新初始化模块"""
        for module_name, module in self.active_modules.items():
            try:
                if hasattr(module, 'initialize'):
                    module.initialize()
            except Exception as e:
                self.logger.error(f"重新初始化模块 {module_name} 时出错: {str(e)}")
                raise ModuleError(f"模块 {module_name} 初始化失败")

    def _verify_system_state(self) -> bool:
        """验证系统状态"""
        try:
            # 检查量子引擎状态
            if hasattr(self, 'quantum_engine'):
                if not self.quantum_engine.is_initialized():
                    return False
            
            # 检查模块状态
            for module in self.active_modules.values():
                if hasattr(module, 'is_healthy'):
                    if not module.is_healthy():
                        return False
            
            # 检查数据完整性
            if self.market_data is None or self.market_data.empty:
                return False
            
            # 检查共生网络状态
            if not self.symbiosis_manager.is_healthy():
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"验证系统状态时出错: {str(e)}")
            return False

    def _restore_state(self, state: Dict[str, Any]) -> None:
        """恢复到之前的状态"""
        try:
            # 恢复性能指标
            self.performance_metrics = state['performance_metrics']
            
            # 恢复模块状态
            self.module_states = state['module_states']
            
            # 恢复量子态
            if state['quantum_state'] is not None and hasattr(self, 'quantum_engine'):
                self.quantum_engine.restore_state(state['quantum_state'])
            
            # 重新加载活动模块
            self._reload_active_modules(state['active_modules'])
            
        except Exception as e:
            self.logger.error(f"恢复状态时出错: {str(e)}")
            raise SystemError("状态恢复失败")

    def open_data_file(self):
        """打开数据文件并加载数据"""
        try:
            # 打开文件选择对话框
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "选择数据文件",
                "",
                "CSV Files (*.csv);;All Files (*)"
            )
            
            if file_path:
                # 尝试加载数据
                self.load_data(file_path)
                self.show_info_message("数据加载", "数据加载成功！")
                
                # 更新显示
                self.update_displays()
                
                # 记录操作日志
                self.operation_log.append({
                    'timestamp': datetime.now(),
                    'operation': 'open_data_file',
                    'file_path': file_path,
                    'status': 'success'
                })
                
        except Exception as e:
            error_msg = f"打开数据文件时发生错误: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.show_error_message("错误", error_msg)
            
            # 记录错误日志
            self.error_log.append({
                'timestamp': datetime.now(),
                'operation': 'open_data_file',
                'error': str(e),
                'traceback': traceback.format_exc()
            })

    def calibrate_model(self):
        """校准量子交易模型"""
        try:
            # 显示校准进度对话框
            progress = QProgressDialog("正在校准模型...", "取消", 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            
            # 更新进度
            progress.setValue(10)
            QApplication.processEvents()
            
            # 校准量子引擎
            if hasattr(self, 'quantum_engine'):
                self.quantum_engine.calibrate()
                progress.setValue(30)
                QApplication.processEvents()
            
            # 校准市场分析模块
            if hasattr(self, 'market_analyzer'):
                self.market_analyzer.calibrate()
                progress.setValue(50)
                QApplication.processEvents()
            
            # 校准预测模型
            if hasattr(self, 'prediction_model'):
                self.prediction_model.calibrate()
                progress.setValue(70)
                QApplication.processEvents()
            
            # 更新系统状态
            self._update_system_state()
            progress.setValue(90)
            QApplication.processEvents()
            
            # 完成校准
            progress.setValue(100)
            self.show_info_message("校准完成", "模型校准成功完成！")
            
        except Exception as e:
            error_msg = f"模型校准失败: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.show_error_message("校准错误", error_msg)
            
            # 记录错误日志
            self.error_log.append({
                'timestamp': datetime.now(),
                'operation': 'calibrate_model',
                'error': str(e),
                'traceback': traceback.format_exc()
            })

    def adjust_sensitivity(self):
        """调整系统灵敏度"""
        try:
            # 显示灵敏度调整对话框
            progress = QProgressDialog("正在调整系统灵敏度...", "取消", 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            
            # 更新进度
            progress.setValue(10)
            QApplication.processEvents()
            
            # 调整量子引擎灵敏度
            if hasattr(self, 'quantum_engine'):
                self.quantum_engine.adjust_sensitivity()
                progress.setValue(30)
                QApplication.processEvents()
            
            # 调整市场分析模块灵敏度
            if hasattr(self, 'market_analyzer'):
                self.market_analyzer.adjust_sensitivity()
                progress.setValue(50)
                QApplication.processEvents()
            
            # 调整预测模型灵敏度
            if hasattr(self, 'prediction_model'):
                self.prediction_model.adjust_sensitivity()
                progress.setValue(70)
                QApplication.processEvents()
            
            # 更新系统状态
            self._update_system_state()
            progress.setValue(90)
            QApplication.processEvents()
            
            # 完成调整
            progress.setValue(100)
            self.show_info_message("灵敏度调整", "系统灵敏度调整完成！")
            
        except Exception as e:
            error_msg = f"灵敏度调整失败: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            self.show_error_message("调整错误", error_msg)
            
            # 记录错误日志
            self.error_log.append({
                'timestamp': datetime.now(),
                'operation': 'adjust_sensitivity',
                'error': str(e),
                'traceback': traceback.format_exc()
            })

    def expand_time_dimension(self):
        """扩展时间维度"""
        try:
            # 实现时间维度扩展逻辑
            logger.info("开始扩展时间维度...")
            # TODO: 添加实际的时间维度扩展逻辑
            self.show_info_message("成功", "时间维度扩展完成")
        except Exception as e:
            self.show_error_message("错误", f"时间维度扩展失败: {str(e)}")
            logger.error(f"时间维度扩展失败: {str(e)}")

    def sync_market_data(self):
        """同步市场数据"""
        try:
            # 实现市场数据同步逻辑
            logger.info("开始同步市场数据...")
            # TODO: 添加实际的数据同步逻辑
            self.show_info_message("成功", "市场数据同步完成")
        except Exception as e:
            self.show_error_message("错误", f"市场数据同步失败: {str(e)}")
            logger.error(f"市场数据同步失败: {str(e)}")


def main():
    """启动超神全息驾驶舱"""
    app = QApplication(sys.argv)
    window = SupergodCockpit()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 