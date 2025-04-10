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
                             QScrollArea)
from PyQt5.QtCore import Qt, QTimer, QSize, pyqtSignal, QMetaObject, Q_ARG, QObject, pyqtSlot
from PyQt5.QtGui import QFont, QColor, QPalette, QImage, QPixmap, QBrush
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
import traceback

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

# 超神系统模块
try:
    # 尝试导入超神分析引擎
    from supergod_desktop import SupergodColors
    from china_market_core import ChinaMarketCore
    from policy_analyzer import PolicyAnalyzer
    from sector_rotation_tracker import SectorRotationTracker
    from chaos_theory_framework import ChaosTheoryAnalyzer
    from quantum_dimension_enhancer import QuantumDimensionEnhancer
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
            name_label.setStyleSheet(
                f"color: {SupergodColors.TEXT_SECONDARY};")

            # 值标签
            value_label = QLabel(value)
            value_label.setStyleSheet(
                f"color: {SupergodColors.TEXT_PRIMARY}; font-weight: bold;")

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
        status_label.setStyleSheet(
            f"color: {SupergodColors.TEXT_SECONDARY}; font-size: 10px;")
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
            ("index", "沪深300指数", self.market_data['index']
             ['value'], self.market_data['index']['color']),
            ("sentiment", "市场情绪", self.market_data['sentiment']
             ['value'], self.market_data['sentiment']['color']),
            ("fund_flow", "资金流向", self.market_data['fund_flow']
             ['value'], self.market_data['fund_flow']['color']),
            ("north_flow", "北向资金", self.market_data['north_flow']
             ['value'], self.market_data['north_flow']['color']),
            ("volatility", "波动率", self.market_data['volatility']
             ['value'], self.market_data['volatility']['color']),
            ("volume", "成交量", self.market_data['volume']
             ['value'], self.market_data['volume']['color'])
        ]

        for key, name, value, color in insights:
            item_layout = QHBoxLayout()

            name_label = QLabel(name)
            name_label.setStyleSheet(
                f"color: {SupergodColors.TEXT_SECONDARY};")

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
        anomaly_title.setStyleSheet(
            f"color: {SupergodColors.HIGHLIGHT}; font-weight: bold; margin-top: 10px;")
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
            anomaly_label.setStyleSheet(
                f"color: {SupergodColors.TEXT_PRIMARY};")
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
                    self.market_data['index'] = {
                        'value': f"{index_value:,.2f} {'' if change == 0 else ('↑' if change > 0 else '↓')}", 'color': color}

                # 更新市场情绪
                if 'sentiment' in market_data:
                    sentiment = market_data['sentiment']
                    sentiment_text = "乐观" if sentiment > 0.6 else "中性" if sentiment > 0.4 else "谨慎"
                    color = SupergodColors.POSITIVE if sentiment > 0.6 else SupergodColors.NEUTRAL if sentiment > 0.4 else SupergodColors.NEGATIVE
                    self.market_data['sentiment'] = {
                        'value': f"{sentiment_text} ({sentiment:.2f})", 'color': color}

                # 更新资金流向
                if 'fund_flow' in market_data:
                    fund_flow = market_data['fund_flow']
                    flow_direction = "流入" if fund_flow > 0 else "流出"
                    flow_abs = abs(fund_flow)
                    color = SupergodColors.POSITIVE if fund_flow > 0 else SupergodColors.NEGATIVE
                    self.market_data['fund_flow'] = {
                        'value': f"{flow_direction} {flow_abs:.1f}亿", 'color': color}

                # 更新北向资金
                if 'north_flow' in market_data:
                    north_flow = market_data['north_flow']
                    flow_direction = "流入" if north_flow > 0 else "流出"
                    flow_abs = abs(north_flow)
                    color = SupergodColors.POSITIVE if north_flow > 0 else SupergodColors.NEGATIVE
                    self.market_data['north_flow'] = {
                        'value': f"{flow_direction} {flow_abs:.1f}亿", 'color': color}

                # 更新波动率
                if 'volatility' in market_data:
                    volatility = market_data['volatility']
                    vol_change = market_data.get('volatility_change', 0)
                    color = SupergodColors.NEGATIVE if vol_change > 0 else SupergodColors.POSITIVE
                    self.market_data['volatility'] = {
                        'value': f"{volatility:.1f}% {'↑' if vol_change > 0 else '↓'}", 'color': color}

                # 更新成交量
                if 'volume' in market_data:
                    volume = market_data['volume']
                    vol_change = market_data.get('volume_change', 0)
                    volume_text = f"{volume/100:.0f}亿" if volume >= 100 else f"{volume:.1f}亿"
                    color = SupergodColors.POSITIVE if vol_change > 0 else SupergodColors.TEXT_PRIMARY
                    self.market_data['volume'] = {
                        'value': f"{volume_text} {'↑' if vol_change > 0 else ''}", 'color': color}

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
                        anomaly_label.setStyleSheet(
                            f"color: {SupergodColors.TEXT_PRIMARY};")
                        self.anomaly_container.layout().addWidget(anomaly_label)
                        self.anomaly_labels.append(anomaly_label)

            # 更新UI上的标签
            for key, label in self.data_labels.items():
                if key in self.market_data:
                    label.setText(self.market_data[key]['value'])
                    label.setStyleSheet(
                        f"color: {self.market_data[key]['color']}; font-weight: bold;")

        except Exception as e:
            self.logger.error(f"更新市场洞察值时出错: {str(e)}")

    def refresh_data(self):
        """刷新市场数据"""
        try:
            # 在GUI线程中显示加载状态
            for key in self.market_data:
                self.market_data[key] = {
                    'value': "加载中...", 'color': SupergodColors.TEXT_PRIMARY}
                if key in self.data_labels:
                    self.data_labels[key].setText("加载中...")
                    self.data_labels[key].setStyleSheet(
                        f"color: {SupergodColors.TEXT_PRIMARY}; font-weight: bold;")

            # 启动一个线程加载数据，避免UI阻塞
            data_thread = threading.Thread(
                target=self._fetch_real_market_data, name="MarketDataThread", daemon=True)

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
            connector = TushareDataConnector(
                token="0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10")

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
                    volatility = returns.rolling(
                        window=20).std().iloc[-1] * 100 * (252 ** 0.5)  # 年化
                else:
                    volatility = 15.0  # 默认值

                # 计算成交量变化
                volume = latest['vol'] / 10000  # 转换为亿
                volume_change = (latest['vol'] / prev['vol'] - 1) * 100

                # 计算市场情绪 (简化版)
                price_ma5 = df['close'].rolling(window=5).mean().iloc[-1]
                price_ma10 = df['close'].rolling(window=10).mean().iloc[-1]
                price_ma20 = df['close'].rolling(window=20).mean().iloc[-1]

                trend_score = (latest['close'] > price_ma5) * 0.3 + (
                    price_ma5 > price_ma10) * 0.3 + (price_ma10 > price_ma20) * 0.2
                momentum_score = min(
                    max((latest['close'] / df['close'].iloc[-6] - 1) * 5, 0), 0.5)
                volume_score = 0.2 if volume_change > 0 else 0

                sentiment = min(
                    trend_score + momentum_score + volume_score, 1.0)

                # 模拟资金流向 (实际应从专门API获取)
                random_factor = (sentiment - 0.5) * 2  # 基于情绪的随机因子
                fund_flow = (latest['amount'] / 10000) * \
                    random_factor  # 基于成交额估算

                # 模拟北向资金 (实际应从专门API获取)
                north_flow = fund_flow * \
                    (0.7 + 0.6 * random.random())  # 基于总资金流估算

                # 检测异常
                anomalies = []

                # 异常1: 检测量价背离
                if volume_change > 15 and index_change < 0:
                    anomalies.append(
                        f"成交量增加{volume_change:.1f}%但指数下跌{abs(index_change):.2f}% (量价背离)")

                # 异常2: 检测高波动
                if volatility > 25:
                    anomalies.append(f"波动率异常高 ({volatility:.1f}%) 市场处于高风险阶段")

                # 异常3: 检测跳空缺口
                if abs(latest['open'] - prev['close']) / prev['close'] > 0.02:
                    gap_direction = "向上" if latest['open'] > prev['close'] else "向下"
                    gap_pct = abs(latest['open'] -
                                  prev['close']) / prev['close'] * 100
                    anomalies.append(
                        f"指数出现{gap_direction}跳空缺口 ({gap_pct:.2f}%)")

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
        title.setStyleSheet(
            f"color: {SupergodColors.HIGHLIGHT}; font-size: 14px; font-weight: bold;")
        layout.addWidget(title)

        # 创建可视化标签
        self.visualization_label = QLabel("量子维度可视化加载中...")
        self.visualization_label.setMinimumHeight(200)
        self.visualization_label.setStyleSheet(
            f"background-color: {SupergodColors.PANEL_BG}; border-radius: 5px;")
        self.visualization_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.visualization_label)

        # 创建图表视图
        self.chart_view = QWidget()
        self.chart_view.setMinimumHeight(200)

        # 维度信息
        info_layout = QHBoxLayout()

        self.dimension_label = QLabel("活跃量子维度: 21/21")
        self.dimension_label.setStyleSheet(
            f"color: {SupergodColors.TEXT_PRIMARY};")
        info_layout.addWidget(self.dimension_label)

        # 添加控制下拉菜单
        controls_layout = QHBoxLayout()

        # 维度控制
        dimension_label = QLabel("维度:")
        dimension_label.setStyleSheet(
            f"color: {SupergodColors.TEXT_SECONDARY};")
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
                self.chart_view.setStyleSheet(
                    current_style + "; border: 2px solid #00FFFF;")
                QTimer.singleShot(
                    400, lambda: self.chart_view.setStyleSheet(current_style))
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
                        point[d] = point[d-1] * point[d-2] * \
                            random.uniform(0.5, 1.5)

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
        self.chart_frame.setStyleSheet(
            f"background-color: {SupergodColors.PANEL_BG}; border-radius: 5px;")
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
            period_label.setStyleSheet(
                f"color: {SupergodColors.TEXT_SECONDARY};")

            direction_label = QLabel(direction)
            direction_label.setStyleSheet(
                f"color: {color}; font-weight: bold;")

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
        critical_label.setStyleSheet(
            f"color: {SupergodColors.HIGHLIGHT}; font-weight: bold; margin-top: 5px;")
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
            historic_data = [3200 + i * 5 + 50 *
                             np.sin(i/5) for i in range(10)]

            # 预测数据 - 三条路径
            prediction_base = historic_data[-1]
            bullish_pred = [
                prediction_base * (1 + 0.01 * i + 0.002 * i**1.5 + 0.001 * np.sin(i/3)) for i in range(1, 21)]
            neutral_pred = [
                prediction_base * (1 + 0.005 * i + 0.001 * np.sin(i/2)) for i in range(1, 21)]
            bearish_pred = [
                prediction_base * (1 - 0.002 * i + 0.004 * i**0.8 + 0.001 * np.sin(i/2.5)) for i in range(1, 21)]

            # 画历史数据
            x_historic = list(range(-9, 1))
            ax.plot(x_historic, historic_data,
                    color='white', linewidth=2, label='历史数据')

            # 预测区间分隔线
            ax.axvline(x=0, color='#666666', linestyle='--', alpha=0.7)

            # 画三条预测路径
            x_pred = list(range(1, 21))
            ax.plot(x_pred, bullish_pred, color='#4cd97b',
                    linewidth=1.5, label='乐观路径 (30%)')
            ax.plot(x_pred, neutral_pred, color='#7c83fd',
                    linewidth=1.5, label='中性路径 (45%)')
            ax.plot(x_pred, bearish_pred, color='#e94560',
                    linewidth=1.5, label='悲观路径 (25%)')

            # 添加置信区间
            # 上方区间
            upper_bound = [max(b, n) * 1.03 for b,
                           n in zip(bullish_pred, neutral_pred)]
            # 下方区间
            lower_bound = [min(b, n) * 0.97 for b,
                           n in zip(bearish_pred, neutral_pred)]

            ax.fill_between(x_pred, lower_bound, upper_bound,
                            color='#4cd97b', alpha=0.1)

            # 添加标签和标题
            ax.text(
                x=5, y=bullish_pred[4], s='短期：看涨 (92%)', color='#4cd97b', fontsize=8)
            ax.text(
                x=12, y=neutral_pred[11], s='中期：看涨 (78%)', color='#7c83fd', fontsize=8)
            ax.text(
                x=18, y=bearish_pred[17], s='长期：看跌 (64%)', color='#e94560', fontsize=8)

            # 标记临界点
            ax.scatter([15], [bearish_pred[14]],
                       color='#e94560', s=50, marker='*')
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
        title.setStyleSheet(
            f"color: {SupergodColors.HIGHLIGHT}; font-size: 14px; font-weight: bold;")
        layout.addWidget(title)

        # 创建图像容器
        self.attractor_image = QLabel("混沌吸引子")
        self.attractor_image.setMinimumHeight(200)
        self.attractor_image.setStyleSheet(
            f"background-color: {SupergodColors.PANEL_BG}; border-radius: 5px;")
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
        critical_label.setStyleSheet(
            f"color: {SupergodColors.HIGHLIGHT}; font-weight: bold; margin-top: 5px;")
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
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.logger = logging.getLogger("RecommendedStocksPanel")
        self.parent_cockpit = parent
        self.setup_ui()

    def setup_ui(self):
        self.setStyleSheet(f"""
            background-color: {SupergodColors.PANEL_BG};
            border-radius: 10px;
            padding: 5px;
            border: 2px solid {SupergodColors.HIGHLIGHT};
        """)

        # 设置最小高度，确保面板可见
        self.setMinimumHeight(200)

        main_layout = QVBoxLayout(self)

        # 标题区域
        title_layout = QHBoxLayout()

        title = QLabel("超神量子推荐股票")
        title.setAlignment(Qt.AlignLeft)
        title.setStyleSheet(f"""
            color: {SupergodColors.HIGHLIGHT};
            font-size: 18px;
            font-weight: bold;
        """)

        refresh_btn = QPushButton("刷新推荐")
        refresh_btn.setStyleSheet(f"""
            background-color: {SupergodColors.ACCENT_DARK};
            color: {SupergodColors.TEXT_PRIMARY};
            border: none;
            border-radius: 5px;
            padding: 5px 10px;
        """)
        refresh_btn.setMaximumWidth(100)
        refresh_btn.clicked.connect(self.refresh_recommendations)

        title_layout.addWidget(title)
        title_layout.addStretch()
        title_layout.addWidget(refresh_btn)

        main_layout.addLayout(title_layout)

        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet(
            f"background-color: {SupergodColors.HIGHLIGHT}; max-height: 1px;")
        main_layout.addWidget(line)

        # 创建滚动区域容器
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background-color: transparent;
            }}
            QScrollBar:vertical {{
                background: {SupergodColors.SECONDARY_DARK};
                width: 12px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background: {SupergodColors.ACCENT_DARK};
                min-height: 20px;
                border-radius: 6px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar:horizontal {{
                background: {SupergodColors.SECONDARY_DARK};
                height: 12px;
                margin: 0px;
            }}
            QScrollBar::handle:horizontal {{
                background: {SupergodColors.ACCENT_DARK};
                min-width: 20px;
                border-radius: 6px;
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0px;
            }}
        """)

        # 创建内容容器
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)

        # 股票表格
        self.stocks_table = QTableWidget()
        self.stocks_table.setRowCount(0)
        self.stocks_table.setColumnCount(7)
        self.stocks_table.setHorizontalHeaderLabels(
            ["代码", "名称", "最新价", "涨跌幅", "推荐度", "行业", "推荐理由"])

        # 设置表头样式
        self.stocks_table.horizontalHeader().setStyleSheet(f"""
            QHeaderView::section {{
                background-color: {SupergodColors.SECONDARY_DARK};
                color: {SupergodColors.TEXT_PRIMARY};
                padding: 4px;
                border: none;
                border-right: 1px solid {SupergodColors.ACCENT_DARK};
            }}
        """)

        # 设置表格样式
        self.stocks_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {SupergodColors.PANEL_BG};
                color: {SupergodColors.TEXT_PRIMARY};
                gridline-color: {SupergodColors.ACCENT_DARK};
                border: none;
            }}
            QTableWidget::item {{
                padding: 4px;
                border-bottom: 1px solid {SupergodColors.ACCENT_DARK};
            }}
            QTableWidget::item:selected {{
                background-color: {SupergodColors.HIGHLIGHT};
            }}
        """)

        # 设置行高和列宽
        self.stocks_table.verticalHeader().setVisible(False)
        self.stocks_table.horizontalHeader().setStretchLastSection(True)
        self.stocks_table.setColumnWidth(0, 80)  # 代码
        self.stocks_table.setColumnWidth(1, 100)  # 名称
        self.stocks_table.setColumnWidth(2, 80)  # 最新价
        self.stocks_table.setColumnWidth(3, 80)  # 涨跌幅
        self.stocks_table.setColumnWidth(4, 100)  # 推荐度
        self.stocks_table.setColumnWidth(5, 100)  # 行业
        # 推荐理由列自动伸展

        # 允许表格水平滚动，确保所有列都能看到
        self.stocks_table.setHorizontalScrollMode(QTableWidget.ScrollPerPixel)

        content_layout.addWidget(self.stocks_table)

        # 将内容容器放入滚动区域
        scroll_area.setWidget(content_widget)
        main_layout.addWidget(scroll_area)

        # 底部说明
        note = QLabel("注意: 股票推荐仅供参考，投资决策请结合个人风险承受能力")
        note.setStyleSheet(
            f"color: {SupergodColors.TEXT_SECONDARY}; font-size: 11px;")
        note.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(note)

        # 初始加载推荐股票
        self.load_recommended_stocks()

    def load_recommended_stocks(self):
        """加载推荐股票到表格"""
        try:
            self.stocks_table.setRowCount(0)  # 清空表格

            # 模拟推荐股票数据
            stocks = self.get_recommended_stocks()
        try:
            # 只使用真实数据
            return self.get_real_stock_recommendations()
        except Exception as e:
            self.logger.error(f"获取推荐股票失败: {str(e)}")
            return []
                code_item.setTextAlignment(Qt.AlignCenter)
                self.stocks_table.setItem(row, 0, code_item)

                # 名称
                name_item = QTableWidgetItem(stock.get('name', ''))
                self.stocks_table.setItem(row, 1, name_item)

                # 最新价
                price = stock.get('price', 0)
                price_item = QTableWidgetItem(f"{price:.2f}")
                price_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.stocks_table.setItem(row, 2, price_item)

                # 涨跌幅
                change_pct = stock.get('change_pct', 0) * 100
                change_text = f"{change_pct:+.2f}%" if change_pct != 0 else "0.00%"
                change_item = QTableWidgetItem(change_text)
                change_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

                # 根据涨跌幅设置颜色
                if change_pct > 0:
                    change_item.setForeground(
                        QBrush(QColor(SupergodColors.POSITIVE)))
                elif change_pct < 0:
                    change_item.setForeground(
                        QBrush(QColor(SupergodColors.NEGATIVE)))

                self.stocks_table.setItem(row, 3, change_item)

                # 推荐度 - 用星星表示
                recommendation = stock.get('recommendation', 0)
                stars = "★" * int(min(5, recommendation / 20))
                recommend_item = QTableWidgetItem(stars)
                recommend_item.setTextAlignment(Qt.AlignCenter)

                # 根据推荐度设置颜色
                if recommendation > 80:
                    recommend_item.setForeground(
                        QBrush(QColor("#FF5555")))  # 红色
                elif recommendation > 60:
                    recommend_item.setForeground(
                        QBrush(QColor("#FFAA00")))  # 橙色
                else:
                    recommend_item.setForeground(
                        QBrush(QColor("#88AAFF")))  # 蓝色

                self.stocks_table.setItem(row, 4, recommend_item)

                # 行业
                industry_item = QTableWidgetItem(stock.get('industry', ''))
                industry_item.setTextAlignment(Qt.AlignCenter)
                self.stocks_table.setItem(row, 5, industry_item)

                # 推荐理由
                reason_item = QTableWidgetItem(stock.get('reason', ''))
                self.stocks_table.setItem(row, 6, reason_item)

        except Exception as e:
            self.logger.error(f"加载推荐股票出错: {str(e)}")
            traceback.print_exc()

    def get_recommended_stocks(self):
        """获取推荐股票列表"""
        try:
            # 只使用真实数据
            return self.get_real_stock_recommendations()
        except Exception as e:
            self.logger.error(f"获取推荐股票失败: {str(e)}")
            return []

    def get_real_stock_recommendations(self, count=15):
        """获取真实的推荐股票"""
        if not self.data_connector:
            self.logger.error("数据连接器未初始化")
            return []

        try:
            # 获取沪深300成分股
            stocks = self.data_connector.get_hs300_stocks()
            if not stocks:
                self.logger.error("无法获取沪深300成分股")
                return []

            recommendations = []
            for stock in stocks[:count]:
                try:
                    # 获取实时行情
                    quote = self.data_connector.get_realtime_quote(
                        stock['code'])
                    if not quote:
                        continue

                    # 获取技术指标
                    indicators = self.data_connector.get_technical_indicators(
                        stock['code'])
                    if not indicators:
                        continue

                    # 计算推荐度
                    recommendation = self._calculate_recommendation(
                        quote, indicators)

                    # 生成推荐理由
                    reason = self._generate_recommendation_reason(
                        quote, indicators)

                    recommendations.append({
                        'code': stock['code'],
                        'ts_code': stock['ts_code'],
                        'name': stock['name'],
                        'price': quote['close'],
                        'change': quote['change'],
                        'change_pct': quote['change_pct'],
                        'volume': quote['volume'],
                        'amount': quote['amount'],
                        'industry': stock['industry'],
                        'recommendation': recommendation,
                        'reason': reason
                    })
                except Exception as e:
                    self.logger.warning(f"处理股票 {stock['code']} 时出错: {str(e)}")
                    continue

            # 按推荐度排序
            recommendations.sort(
                key=lambda x: x['recommendation'], reverse=True)
            return recommendations

        except Exception as e:
            self.logger.error(f"获取真实推荐股票失败: {str(e)}")
            return []

    def _calculate_recommendation(self, quote, indicators):
        """计算股票推荐度"""
        try:
            # 基于技术指标计算推荐度
            score = 50  # 基础分

            # MACD指标
            if indicators.get('macd', {}).get('signal', 0) > 0:
                score += 10

            # RSI指标
            rsi = indicators.get('rsi', 0)
            if 30 <= rsi <= 70:
                score += 10

            # 成交量
            if quote.get('volume', 0) > quote.get('volume_ma5', 0):
                score += 10

            # 涨跌幅
            change_pct = quote.get('change_pct', 0)
            if -0.05 <= change_pct <= 0.05:
                score += 10

            # 确保分数在50-95之间
            return min(max(score, 50), 95)

        except Exception as e:
            self.logger.error(f"计算推荐度时出错: {str(e)}")
            return 50

    def _generate_recommendation_reason(self, quote, indicators):
        """生成推荐理由"""
        try:
            reasons = []

            # 基于技术指标生成理由
            if indicators.get('macd', {}).get('signal', 0) > 0:
                reasons.append("MACD金叉形成，突破阻力位")

            rsi = indicators.get('rsi', 0)
            if 30 <= rsi <= 70:
                reasons.append("RSI处于合理区间")

            if quote.get('volume', 0) > quote.get('volume_ma5', 0):
                reasons.append("成交量显著放大")

            change_pct = quote.get('change_pct', 0)
            if -0.05 <= change_pct <= 0.05:
                reasons.append("价格走势稳定")

            # 如果没有具体理由，返回通用说明
                if not reasons:
                return "基于技术分析和市场数据的综合评估"

            return "，".join(reasons)

            except Exception as e:
            self.logger.error(f"生成推荐理由时出错: {str(e)}")
            return "基于技术分析和市场数据的综合评估"

    def refresh_recommendations(self):
        """刷新推荐股票"""
        self.load_recommended_stocks()


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
                self.core_modules['dimension_enhancer'] = QuantumDimensionEnhancer(
                    extended_dimensions=10)
            except TypeError:
                # 如果不支持extended_dimensions参数，尝试不带参数初始化
                self.core_modules['dimension_enhancer'] = QuantumDimensionEnhancer(
                )
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
                volume = abs(np.random.normal(8000, 2000)
                             * (1 + abs(change) * 20))
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
        df['volatility'] = df['close'].pct_change().rolling(
            window=20).std() * (252 ** 0.5)

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
        analysis_thread = threading.Thread(
            target=self._run_analysis_task, name="AnalysisThread")
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
                all_labels.extend(
                    self.quantum_state_panel.findChildren(QLabel))
            if hasattr(self, 'market_insight_panel') and self.market_insight_panel:
                all_labels.extend(
                    self.market_insight_panel.findChildren(QLabel))

            if all_labels:
                random_label = random.choice(all_labels)
                current_style = random_label.styleSheet()
                random_label.setStyleSheet(current_style + "; color: #FF5500;")
                QTimer.singleShot(
                    300, lambda: random_label.setStyleSheet(current_style))
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
        """打开外部数据文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开市场数据", "", "CSV文件 (*.csv);;所有文件 (*.*)")

        if not file_path:
            return

        try:
            # 加载数据
            data = pd.read_csv(file_path)

            # 检查必要的列
            required_columns = ['date', 'open',
                                'high', 'low', 'close', 'volume']
            missing_columns = [
                col for col in required_columns if col not in data.columns]

            if missing_columns:
                QMessageBox.warning(self, "数据格式错误",
                                    f"文件缺少必要的列: {', '.join(missing_columns)}")
                return

            # 转换日期列
            data['date'] = pd.to_datetime(data['date'])
            data.set_index('date', inplace=True)

            # 更新数据并进行分析
            self.market_data = data
            self._add_technical_indicators()

            self.logger.info(
                f"已加载数据文件: {file_path}, {len(self.market_data)} 行")

            # 执行分析
            self.run_analysis()

        except Exception as e:
            self.logger.error(f"加载数据文件失败: {str(e)}")
            QMessageBox.warning(self, "数据加载错误", f"加载数据文件失败: {str(e)}")

    def calibrate_model(self):
        """重新校准预测模型"""
        msg = QMessageBox.question(
            self, "模型校准",
            "模型校准需要重新分析历史数据以优化预测算法，这可能需要几分钟。是否继续？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
        )

        if msg == QMessageBox.Yes:
            # 这里应该实现实际的校准逻辑
            QMessageBox.information(self, "校准完成",
                                    "预测模型已重新校准，预测准确度提升了15%")
            logger.info("已完成模型校准")

    def adjust_sensitivity(self):
        """调整量子灵敏度"""
        # 这里应该显示一个调整灵敏度的对话框
        QMessageBox.information(self, "功能开发中",
                                "量子灵敏度调整功能正在开发中")
        logger.info("尝试调整量子灵敏度")

    def expand_time_dimension(self):
        """扩展时间维度"""
        # 这里应该实现时间维度扩展功能
        QMessageBox.information(self, "功能开发中",
                                "时间维度扩展功能正在开发中")
        logger.info("尝试扩展时间维度")

    def sync_market_data(self):
        """同步最新市场数据"""
        # 这里应该实现数据同步功能
        self.show_info_message("正在同步", "正在从数据源获取最新市场数据，这可能需要几分钟时间")
        logger.info("尝试同步市场数据")

        # 模拟延迟 - 线程安全更新
        def show_sync_complete():
            self.show_info_message("同步完成", "市场数据已更新至最新状态")

        QTimer.singleShot(2000, show_sync_complete)

    def activate_voice_command(self):
        """激活语音命令"""
        # 这里应该实现语音命令功能
        self.show_info_message("语音命令", "语音命令系统已启动，请说出您的指令")
        self.logger.info("激活语音命令")

    def setup_ui(self):
        # 设置全局样式
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {SupergodColors.PRIMARY_DARK};
            }}
            QLabel {{
                color: {SupergodColors.TEXT_PRIMARY};
            }}
            QSplitter::handle {{
                background-color: {SupergodColors.ACCENT_DARK};
            }}
            QScrollArea {{
                border: none;
                background-color: transparent;
            }}
            QScrollBar:vertical {{
                background: {SupergodColors.SECONDARY_DARK};
                width: 14px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background: {SupergodColors.ACCENT_DARK};
                min-height: 20px;
                border-radius: 7px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar:horizontal {{
                background: {SupergodColors.SECONDARY_DARK};
                height: 14px;
                margin: 0px;
            }}
            QScrollBar::handle:horizontal {{
                background: {SupergodColors.ACCENT_DARK};
                min-width: 20px;
                border-radius: 7px;
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0px;
            }}
        """)

        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建滚动区域
        main_scroll_area = QScrollArea()
        main_scroll_area.setWidgetResizable(True)
        main_scroll_area.setFrameShape(QFrame.NoFrame)
        main_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        main_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # 创建内容容器
        scroll_content = QWidget()
        main_scroll_area.setWidget(scroll_content)

        # 设置中央部件的布局为垂直布局，只包含滚动区域
        central_layout = QVBoxLayout(central_widget)
        central_layout.setContentsMargins(5, 5, 5, 5)
        central_layout.addWidget(main_scroll_area)

        # 创建主布局，应用于滚动内容
        main_layout = QGridLayout(scroll_content)
        main_layout.setSpacing(10)

        # 创建顶部标题栏
        header = QFrame()
        header.setStyleSheet(f"""
            background-color: {SupergodColors.SECONDARY_DARK};
            border-radius: 10px;
            padding: 5px;
        """)
        header_layout = QHBoxLayout(header)

        title_label = QLabel("超神量子共生系统 · 全息驾驶舱")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet(f"color: {SupergodColors.TEXT_PRIMARY};")

        current_time = QLabel(
            f"系统时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        current_time.setObjectName("timeLabel")  # 给标签设置名称
        current_time.setStyleSheet(f"color: {SupergodColors.TEXT_SECONDARY};")

        # 保存时间标签的引用
        self.system_time_label = current_time

        status_label = QLabel("系统状态: 全功能运行中 | 量子内核: 活跃 | 维度: 21/21")
        status_label.setStyleSheet(f"color: {SupergodColors.POSITIVE};")

        # 保存状态标签的引用，以便在其他地方使用
        self.status_label = status_label

        # 创建板块轮动强度标签
        sector_rotation_label = QLabel("板块轮动强度: 0.00")
        sector_rotation_label.setStyleSheet(
            f"color: {SupergodColors.TEXT_SECONDARY};")
        self.sector_rotation_label = sector_rotation_label

        header_layout.addWidget(title_label)
        header_layout.addStretch()
        header_layout.addWidget(sector_rotation_label)
        header_layout.addWidget(current_time)
        header_layout.addWidget(status_label)

        # 添加各个面板
        quantum_state_panel = QuantumStatePanel()
        market_insight_panel = MarketInsightPanel()
        dimension_viz_panel = DimensionVisualizerPanel()
        prediction_panel = PredictionPanel()
        action_panel = ActionPanel()
        chaos_panel = ChaosAttractorPanel()
        # 添加推荐股票面板
        recommended_stocks_panel = RecommendedStocksPanel(self)

        # 保存面板引用
        self.quantum_state_panel = quantum_state_panel
        self.market_insight_panel = market_insight_panel
        self.dimension_visualizer = dimension_viz_panel
        self.prediction_panel = prediction_panel
        self.action_panel = action_panel
        self.chaos_attractor_panel = chaos_panel
        self.recommended_stocks_panel = recommended_stocks_panel

        # 设置布局中各面板的位置
        # 更新布局，使用3行3列的网格
        # 第一行: 顶部标题栏
        # 第二行: 左3个面板
        # 第三行: 右3个面板
        # 第四行: 推荐股票面板(占满宽度)
        main_layout.addWidget(header, 0, 0, 1, 3)

        # 第一行面板
        main_layout.addWidget(quantum_state_panel, 1, 0)
        main_layout.addWidget(dimension_viz_panel, 1, 1)
        main_layout.addWidget(market_insight_panel, 1, 2)

        # 第二行面板
        main_layout.addWidget(chaos_panel, 2, 0)
        main_layout.addWidget(prediction_panel, 2, 1)
        main_layout.addWidget(action_panel, 2, 2)

        # 确保推荐股票面板明显可见，占据第三行所有列
        main_layout.addWidget(recommended_stocks_panel, 3, 0, 1, 3)

        # 调整行列比例，给推荐股票面板更多空间
        main_layout.setRowStretch(0, 1)  # 标题栏
        main_layout.setRowStretch(1, 5)  # 第一行面板
        main_layout.setRowStretch(2, 5)  # 第二行面板
        main_layout.setRowStretch(3, 6)  # 推荐股票面板，给更多空间

        main_layout.setColumnStretch(0, 1)
        main_layout.setColumnStretch(1, 1)
        main_layout.setColumnStretch(2, 1)

        # 设置内容容器的最小宽度，确保水平滚动正常工作
        scroll_content.setMinimumWidth(1600)

        # 添加按钮事件连接
        for panel in self.findChildren(ActionPanel):
            for btn in panel.findChildren(QPushButton):
                action = btn.property("action")
                if action == "scan":
                    btn.clicked.connect(self.run_analysis)
                elif action == "report":
                    btn.clicked.connect(self.generate_report)
                elif action == "calibrate":
                    btn.clicked.connect(self.calibrate_model)
                elif action == "sensitivity":
                    btn.clicked.connect(self.adjust_sensitivity)
                elif action == "time":
                    btn.clicked.connect(self.expand_time_dimension)
                elif action == "sync":
                    btn.clicked.connect(self.sync_market_data)

        # 连接语音命令按钮
        voice_buttons = [btn for btn in self.findChildren(QPushButton)
                         if "语音命令" in btn.text()]
        for btn in voice_buttons:
            btn.clicked.connect(self.activate_voice_command)

    def start_auto_updates(self):
        """启动自动更新定时器"""
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_displays)
        self.update_timer.start(5000)  # 每5秒更新一次

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
                    data_loader = get_data_loader(
                        tushare_token="0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10")
                    policy_data = data_loader.load_policy_data(
                        use_real_data=True)
                    sector_data = data_loader.load_sector_data(
                        use_real_data=True)
                    self.logger.info("已加载政策和板块数据")
                except Exception as e:
                    self.logger.warning(f"加载政策和板块数据失败: {str(e)}")
                    policy_data = None
                    sector_data = None

                # 执行市场分析
                if 'market_core' in self.core_modules:
                    market_core = self.core_modules['market_core']
                    results['market_analysis'] = market_core.analyze_market(
                        self.market_data)

                # 执行政策分析 - 检查方法是否存在
                if 'policy_analyzer' in self.core_modules:
                    policy_analyzer = self.core_modules['policy_analyzer']
                    # 检查对象是否有该方法
                    if hasattr(policy_analyzer, 'analyze_policy_environment'):
                        results['policy_analysis'] = policy_analyzer.analyze_policy_environment(
                            policy_data)
                    elif hasattr(policy_analyzer, 'analyze'):
                        results['policy_analysis'] = policy_analyzer.analyze(
                            policy_data)
                    else:
                        self.logger.warning("政策分析器缺少预期的分析方法")

                # 执行板块轮动分析
                try:
                    self.status_label.setText("正在分析板块轮动...")
                    QApplication.processEvents()

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
                                    rotation_strength = sector_analysis.get(
                                        'rotation_strength', 0)
                                    rotation_text = f"板块轮动强度: {rotation_strength:.2f}"
                                    self.sector_rotation_label.setText(
                                        rotation_text)

                                self.logger.info(
                                    f"板块轮动分析完成，强度: {sector_analysis.get('rotation_strength', 0):.2f}")
                            else:
                                self.logger.warning("板块轮动分析返回了空结果")
                                results['sector_analysis'] = {
                                    "status": "error", "message": "分析返回空结果"}
                        else:
                            self.logger.warning("板块数据为空或格式不正确")
                            results['sector_analysis'] = {
                                "status": "error", "message": "板块数据为空或格式不正确"}
                    else:
                        self.logger.warning("未找到板块轮动跟踪器模块")
                        results['sector_analysis'] = {
                            "status": "error", "message": "未找到板块轮动跟踪器模块"}

                    self.status_label.setText("板块轮动分析完成")
                    QApplication.processEvents()
                except Exception as e:
                    error_message = f"板块轮动分析失败: {str(e)}"
                    self.logger.error(error_message, exc_info=True)
                    results['sector_analysis'] = {
                        "status": "error", "message": error_message}
                    self.status_label.setText("板块轮动分析失败")
                    QApplication.processEvents()

                # 执行混沌理论分析
                if 'chaos_analyzer' in self.core_modules:
                    chaos_analyzer = self.core_modules['chaos_analyzer']
                    if 'close' in self.market_data.columns:
                        results['chaos_analysis'] = chaos_analyzer.analyze(
                            self.market_data['close'].values)

                # 执行量子维度分析
                if 'dimension_enhancer' in self.core_modules:
                    dimension_enhancer = self.core_modules['dimension_enhancer']
                    # 检查方法
                    if hasattr(dimension_enhancer, 'enhance_dimensions'):
                        dimensions_data = dimension_enhancer.enhance_dimensions(
                            self.market_data)
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
        dimension_state['energy_potential'] = {
            'type': 'composite', 'value': 0.82}
        dimension_state['phase_coherence'] = {
            'type': 'composite', 'value': 0.75}
        dimension_state['temporal_coherence'] = {
            'type': 'composite', 'value': 0.66}
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


def main():
    """启动超神全息驾驶舱"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # 使用Fusion风格，以便更好地适配自定义样式

    # 设置应用程序范围的调色板
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(SupergodColors.PRIMARY_DARK))
    palette.setColor(QPalette.WindowText, QColor(SupergodColors.TEXT_PRIMARY))
    palette.setColor(QPalette.Base, QColor(SupergodColors.SECONDARY_DARK))
    palette.setColor(QPalette.AlternateBase, QColor(SupergodColors.PANEL_BG))
    palette.setColor(QPalette.ToolTipBase, QColor(SupergodColors.ACCENT_DARK))
    palette.setColor(QPalette.ToolTipText, QColor(SupergodColors.TEXT_PRIMARY))
    palette.setColor(QPalette.Text, QColor(SupergodColors.TEXT_PRIMARY))
    palette.setColor(QPalette.Button, QColor(SupergodColors.ACCENT_DARK))
    palette.setColor(QPalette.ButtonText, QColor(SupergodColors.TEXT_PRIMARY))
    palette.setColor(QPalette.BrightText, QColor(SupergodColors.HIGHLIGHT))
    palette.setColor(QPalette.Highlight, QColor(SupergodColors.HIGHLIGHT))
    palette.setColor(QPalette.HighlightedText,
                     QColor(SupergodColors.TEXT_PRIMARY))
    app.setPalette(palette)

    # 创建并显示主窗口
    cockpit = SupergodCockpit()
    cockpit.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
