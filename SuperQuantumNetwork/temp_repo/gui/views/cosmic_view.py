#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - 宇宙视图
展示宇宙共振和能量可视化
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                           QProgressBar, QFrame, QScrollArea, QSizePolicy,
                           QTabWidget, QGridLayout, QSpacerItem, QTextBrowser,
                           QListWidget, QListWidgetItem)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPalette, QGradient
from datetime import datetime

# 导入宇宙能量可视化组件
from gui.views.cosmic_energy_visualization import CosmicEnergyVisualization

class CosmicView(QWidget):
    """宇宙视图"""
    
    update_requested = pyqtSignal()
    analyze_market_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("cosmicView")
        
        # 设置样式
        self.setStyleSheet("""
            QWidget#cosmicView {
                background-color: #050510;
            }
            QLabel#titleLabel {
                color: #c0c0ff;
                font-size: 18px;
                font-weight: bold;
            }
            QLabel#sectionLabel {
                color: #a0a0ff;
                font-size: 16px;
                font-weight: bold;
            }
            QLabel#valueLabel {
                color: #80d0ff;
                font-size: 14px;
            }
            QLabel#eventLabel {
                color: #70b0ff;
                font-size: 14px;
                padding: 10px;
                background-color: rgba(10, 20, 40, 0.6);
                border-radius: 5px;
            }
            QFrame#cosmicFrame {
                border: 1px solid #336;
                border-radius: 10px;
                background-color: rgba(10, 15, 35, 0.7);
                padding: 5px;
            }
            QProgressBar {
                border: 1px solid #446;
                border-radius: 5px;
                background-color: #111144;
                text-align: center;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2277ff, stop:1 #00ddff);
                border-radius: 5px;
            }
            QTabWidget::pane {
                border: 1px solid #446;
                border-radius: 5px;
                background-color: rgba(10, 15, 35, 0.7);
            }
            QTabBar::tab {
                background-color: #223;
                color: #aac;
                border: 1px solid #446;
                padding: 5px 10px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {
                background-color: #335;
                color: #ccf;
            }
        """)
        
        # 初始化UI
        self.init_ui()
        
        # 设置更新计时器
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.request_update)
        self.update_timer.start(3000)  # 每3秒更新一次
        
        # 市场分析计时器
        self.market_timer = QTimer(self)
        self.market_timer.timeout.connect(self.request_market_analysis)
        self.market_timer.start(15000)  # 每15秒分析一次市场
    
    def init_ui(self):
        """初始化UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # 标题
        title_label = QLabel("宇宙共振终极引擎")
        title_label.setObjectName("titleLabel")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 创建标签页
        tab_widget = QTabWidget()
        
        # 共振状态标签页
        resonance_tab = QWidget()
        resonance_layout = QVBoxLayout(resonance_tab)
        self.setup_resonance_tab(resonance_layout)
        tab_widget.addTab(resonance_tab, "共振状态")
        
        # 宇宙事件标签页
        events_tab = QWidget()
        events_layout = QVBoxLayout(events_tab)
        self.setup_events_tab(events_layout)
        tab_widget.addTab(events_tab, "宇宙事件")
        
        # 市场模式标签页
        patterns_tab = QWidget()
        patterns_layout = QVBoxLayout(patterns_tab)
        self.setup_patterns_tab(patterns_layout)
        tab_widget.addTab(patterns_tab, "市场模式")
        
        # 能量可视化标签页
        visualization_tab = QWidget()
        visualization_layout = QVBoxLayout(visualization_tab)
        self.setup_visualization_tab(visualization_layout)
        tab_widget.addTab(visualization_tab, "能量可视化")
        
        main_layout.addWidget(tab_widget)
    
    def setup_resonance_tab(self, layout):
        """设置共振状态标签页"""
        # 共振状态面板
        resonance_frame = QFrame()
        resonance_frame.setObjectName("cosmicFrame")
        resonance_layout = QVBoxLayout(resonance_frame)
        
        # 共振参数
        params_grid = QGridLayout()
        params_grid.setColumnStretch(1, 1)
        params_grid.setColumnStretch(3, 1)
        
        # 创建参数标签和进度条
        param_labels = ["共振强度", "维度同步率", "和谐指数", "宇宙精准度"]
        self.param_bars = {}
        
        for i, label in enumerate(param_labels):
            row, col = i // 2, i % 2 * 2
            
            # 标签
            param_label = QLabel(f"{label}:")
            param_label.setObjectName("valueLabel")
            params_grid.addWidget(param_label, row, col)
            
            # 进度条
            progress_bar = QProgressBar()
            progress_bar.setRange(0, 100)
            progress_bar.setValue(0)
            progress_bar.setFormat("%v%")
            params_grid.addWidget(progress_bar, row, col + 1)
            
            # 保存引用
            self.param_bars[label] = progress_bar
        
        resonance_layout.addLayout(params_grid)
        
        # 场参数
        field_layout = QGridLayout()
        field_layout.setColumnStretch(1, 1)
        field_layout.setColumnStretch(3, 1)
        
        # 场频率
        freq_label = QLabel("场频率 (Hz):")
        freq_label.setObjectName("valueLabel")
        self.freq_value = QLabel("7.83")
        self.freq_value.setObjectName("valueLabel")
        self.freq_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        field_layout.addWidget(freq_label, 0, 0)
        field_layout.addWidget(self.freq_value, 0, 1)
        
        # 场振幅
        amp_label = QLabel("场振幅:")
        amp_label.setObjectName("valueLabel")
        self.amp_value = QLabel("0.50")
        self.amp_value.setObjectName("valueLabel")
        self.amp_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        field_layout.addWidget(amp_label, 0, 2)
        field_layout.addWidget(self.amp_value, 0, 3)
        
        # 矩阵相干性
        coherence_label = QLabel("矩阵相干性:")
        coherence_label.setObjectName("valueLabel")
        self.coherence_value = QLabel("0.30")
        self.coherence_value.setObjectName("valueLabel")
        self.coherence_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        field_layout.addWidget(coherence_label, 1, 0)
        field_layout.addWidget(self.coherence_value, 1, 1)
        
        # 共振状态
        status_label = QLabel("共振状态:")
        status_label.setObjectName("valueLabel")
        self.status_value = QLabel("活跃")
        self.status_value.setObjectName("valueLabel")
        self.status_value.setStyleSheet("color: #00ffcc; font-weight: bold;")
        self.status_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        field_layout.addWidget(status_label, 1, 2)
        field_layout.addWidget(self.status_value, 1, 3)
        
        resonance_layout.addLayout(field_layout)
        
        # 更新时间
        time_layout = QHBoxLayout()
        time_label = QLabel("上次更新:")
        time_label.setObjectName("valueLabel")
        self.time_value = QLabel("--")
        self.time_value.setObjectName("valueLabel")
        self.time_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        time_layout.addWidget(time_label)
        time_layout.addWidget(self.time_value, 1)
        
        resonance_layout.addLayout(time_layout)
        
        layout.addWidget(resonance_frame)
        layout.addStretch()
    
    def setup_events_tab(self, layout):
        """设置宇宙事件标签页"""
        # 事件标签
        events_label = QLabel("宇宙事件")
        events_label.setObjectName("sectionLabel")
        events_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(events_label)
        
        # 事件滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        
        # 事件容器
        self.events_container = QWidget()
        self.events_layout = QVBoxLayout(self.events_container)
        self.events_layout.setContentsMargins(0, 0, 0, 0)
        self.events_layout.setSpacing(10)
        self.events_layout.addStretch()
        
        scroll_area.setWidget(self.events_container)
        layout.addWidget(scroll_area, 1)
    
    def setup_patterns_tab(self, layout):
        """设置市场模式标签页"""
        # 模式标签
        patterns_label = QLabel("宇宙市场模式")
        patterns_label.setObjectName("sectionLabel")
        patterns_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(patterns_label)
        
        # 时间戳
        self.pattern_time = QLabel("上次分析: --")
        self.pattern_time.setObjectName("valueLabel")
        self.pattern_time.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.pattern_time)
        
        # 创建模式面板
        patterns_frame = QFrame()
        patterns_frame.setObjectName("cosmicFrame")
        patterns_layout = QVBoxLayout(patterns_frame)
        
        # 模式标题
        patterns_title = QLabel("检测到的模式")
        patterns_title.setObjectName("valueLabel")
        patterns_title.setStyleSheet("color: #a0d0ff; font-weight: bold;")
        patterns_layout.addWidget(patterns_title)
        
        # 模式列表
        self.patterns_container = QWidget()
        self.patterns_list = QVBoxLayout(self.patterns_container)
        self.patterns_list.setContentsMargins(0, 0, 0, 0)
        self.patterns_list.setSpacing(5)
        
        # 添加默认消息
        default_pattern = QLabel("等待宇宙模式分析...")
        default_pattern.setObjectName("valueLabel")
        default_pattern.setStyleSheet("color: #7799cc;")
        self.patterns_list.addWidget(default_pattern)
        
        patterns_layout.addWidget(self.patterns_container)
        layout.addWidget(patterns_frame)
        
        # 创建预测面板
        predictions_frame = QFrame()
        predictions_frame.setObjectName("cosmicFrame")
        predictions_layout = QVBoxLayout(predictions_frame)
        
        # 预测标题
        predictions_title = QLabel("高维预测")
        predictions_title.setObjectName("valueLabel")
        predictions_title.setStyleSheet("color: #c0d0ff; font-weight: bold;")
        predictions_layout.addWidget(predictions_title)
        
        # 预测列表
        self.predictions_container = QWidget()
        self.predictions_list = QVBoxLayout(self.predictions_container)
        self.predictions_list.setContentsMargins(0, 0, 0, 0)
        self.predictions_list.setSpacing(5)
        
        # 添加默认消息
        default_prediction = QLabel("等待宇宙预测生成...")
        default_prediction.setObjectName("valueLabel")
        default_prediction.setStyleSheet("color: #7799cc;")
        self.predictions_list.addWidget(default_prediction)
        
        predictions_layout.addWidget(self.predictions_container)
        layout.addWidget(predictions_frame)
        
        # 添加弹性空间
        layout.addStretch()
    
    def setup_visualization_tab(self, layout):
        """设置能量可视化标签页"""
        # 能量可视化标签
        viz_label = QLabel("宇宙能量可视化")
        viz_label.setObjectName("sectionLabel")
        viz_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(viz_label)
        
        # 创建能量可视化组件
        self.energy_viz = CosmicEnergyVisualization()
        self.energy_viz.setMinimumHeight(300)
        layout.addWidget(self.energy_viz, 1)
        
        # 能量说明
        energy_desc = QLabel("实时显示宇宙能量流动和系统节点之间的共振状态")
        energy_desc.setObjectName("valueLabel")
        energy_desc.setAlignment(Qt.AlignCenter)
        layout.addWidget(energy_desc)
    
    def update_resonance_state(self, state):
        """更新共振状态显示"""
        if not state:
            return
            
        # 更新共振参数进度条
        param_mappings = {
            "共振强度": "resonance_level",
            "维度同步率": "dimensional_sync",
            "和谐指数": "harmony_index",
            "宇宙精准度": "cosmic_accuracy"
        }
        
        for label, key in param_mappings.items():
            if key in state and label in self.param_bars:
                value = state[key]
                self.param_bars[label].setValue(int(value * 100))
        
        # 更新场参数
        if "field_frequency" in state:
            self.freq_value.setText(f"{state['field_frequency']:.2f}")
        
        if "field_amplitude" in state:
            self.amp_value.setText(f"{state['field_amplitude']:.2f}")
        
        if "matrix_coherence" in state:
            self.coherence_value.setText(f"{state['matrix_coherence']:.2f}")
        
        # 更新共振状态
        if "is_resonating" in state:
            status = "活跃" if state["is_resonating"] else "休眠"
            color = "#00ffcc" if state["is_resonating"] else "#ff7777"
            self.status_value.setText(status)
            self.status_value.setStyleSheet(f"color: {color}; font-weight: bold;")
        
        # 更新时间戳
        if "timestamp" in state:
            self.time_value.setText(state["timestamp"])
        
        # 更新能量可视化的能量水平
        if "resonance_level" in state and hasattr(self, 'energy_viz'):
            self.energy_viz.set_energy_level(state["resonance_level"])
    
    def display_cosmic_events(self, events):
        """
        显示宇宙事件
        
        Args:
            events: 宇宙事件列表
        """
        self.clear_cosmic_events()
        
        if not events:
            self.add_cosmic_message("未检测到宇宙事件", "warning")
            return
            
        self.add_cosmic_message(f"检测到 {len(events)} 个宇宙事件", "info")
        
        # 按日期排序事件
        sorted_events = sorted(events, key=lambda x: x.get("date", ""))
        
        for event in sorted_events:
            event_date = event.get("date", "未知日期")
            event_type = event.get("type", "未知类型")
            event_strength = event.get("strength", 0)
            event_content = event.get("content", "无内容")
            event_impact = event.get("impact", "未知")
            
            # 根据强度确定样式类别
            style_class = ""
            if event_strength > 0.8:
                style_class = "critical-event"
            elif event_strength > 0.6:
                style_class = "major-event"
            elif event_strength > 0.4:
                style_class = "moderate-event"
            else:
                style_class = "minor-event"
                
            # 创建事件标题
            title_html = f"""
            <div class='event-header {style_class}'>
                <span class='event-date'>{event_date}</span>
                <span class='event-type'>{event_type}</span>
                <span class='event-impact'>{event_impact}</span>
                <span class='event-strength'>强度: {event_strength:.2f}</span>
            </div>
            """
            
            # 创建事件内容
            content_html = f"""
            <div class='event-content'>
                <p>{event_content}</p>
            """
            
            # 添加宇宙洞察（如果有）
            if "cosmic_insight" in event:
                content_html += f"""
                <div class='cosmic-insight'>
                    <span class='insight-title'>宇宙洞察:</span>
                    <p>{event['cosmic_insight']}</p>
                </div>
                """
                
            # 添加行动建议（如果有）
            if "action_suggestion" in event:
                content_html += f"""
                <div class='action-suggestion'>
                    <span class='suggestion-title'>行动建议:</span>
                    <p>{event['action_suggestion']}</p>
                </div>
                """
                
            # 关闭内容div
            content_html += "</div>"
            
            # 将事件添加到视图
            self.cosmic_events_layout.addWidget(
                self._create_event_widget(title_html + content_html)
            )
            
        # 更新视图
        self.cosmic_events_container.setLayout(self.cosmic_events_layout)
        
    def _create_event_widget(self, html_content):
        """
        创建事件小部件
        
        Args:
            html_content: HTML内容
            
        Returns:
            QWidget: 事件小部件
        """
        widget = QWidget()
        layout = QVBoxLayout()
        
        text_browser = QTextBrowser()
        text_browser.setHtml(html_content)
        text_browser.setOpenExternalLinks(True)
        text_browser.setStyleSheet("""
            QTextBrowser {
                background-color: rgba(20, 20, 40, 0.7);
                border: 1px solid rgba(100, 100, 180, 0.5);
                border-radius: 5px;
                padding: 8px;
                color: #ddd;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
        """)
        
        layout.addWidget(text_browser)
        widget.setLayout(layout)
        return widget
    
    def update_market_patterns(self, analysis):
        """更新市场模式分析"""
        if not analysis:
            return
            
        # 更新时间戳
        timestamp = analysis.get("timestamp", "--")
        self.pattern_time.setText(f"上次分析: {timestamp}")
        
        # 清除现有模式
        while self.patterns_list.count() > 0:
            item = self.patterns_list.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # 添加新模式
        patterns = analysis.get("patterns", [])
        if patterns:
            for pattern in patterns:
                pattern_layout = QHBoxLayout()
                
                # 模式描述
                desc = pattern.get("description", "")
                desc_label = QLabel(desc)
                desc_label.setObjectName("valueLabel")
                desc_label.setWordWrap(True)
                pattern_layout.addWidget(desc_label, 1)
                
                # 置信度
                confidence = pattern.get("confidence", 0)
                conf_label = QLabel(f"{confidence:.2f}")
                conf_label.setStyleSheet(f"color: {'#00ffaa' if confidence > 0.7 else '#aaff00'}; font-weight: bold;")
                pattern_layout.addWidget(conf_label)
                
                # 添加到列表
                self.patterns_list.addLayout(pattern_layout)
        else:
            # 无模式
            no_patterns = QLabel("当前未检测到明显的宇宙模式")
            no_patterns.setObjectName("valueLabel")
            no_patterns.setStyleSheet("color: #7799cc;")
            self.patterns_list.addWidget(no_patterns)
        
        # 清除现有预测
        while self.predictions_list.count() > 0:
            item = self.predictions_list.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # 添加新预测
        predictions = analysis.get("predictions", [])
        if predictions:
            for prediction in predictions:
                pred_frame = QFrame()
                pred_frame.setFrameShape(QFrame.StyledPanel)
                pred_frame.setStyleSheet("background-color: rgba(20, 30, 60, 0.5); border-radius: 5px; padding: 5px;")
                pred_layout = QVBoxLayout(pred_frame)
                pred_layout.setSpacing(3)
                
                # 预测描述
                desc = prediction.get("description", "")
                desc_label = QLabel(desc)
                desc_label.setObjectName("valueLabel")
                desc_label.setWordWrap(True)
                desc_label.setStyleSheet("color: #90d0ff; font-weight: bold;")
                pred_layout.addWidget(desc_label)
                
                # 底部信息
                footer_layout = QHBoxLayout()
                
                # 时间框架
                timeframe = prediction.get("timeframe", "")
                time_label = QLabel(timeframe)
                time_label.setStyleSheet("color: #7799cc; font-size: 12px;")
                footer_layout.addWidget(time_label)
                
                footer_layout.addStretch()
                
                # 概率
                probability = prediction.get("probability", 0)
                prob_label = QLabel(f"概率: {probability:.2f}")
                prob_label.setStyleSheet(f"color: {'#00ffaa' if probability > 0.7 else '#aaff00'}; font-size: 12px; font-weight: bold;")
                footer_layout.addWidget(prob_label)
                
                pred_layout.addLayout(footer_layout)
                
                # 添加到列表
                self.predictions_list.addWidget(pred_frame)
        else:
            # 无预测
            no_predictions = QLabel("当前未生成高维预测")
            no_predictions.setObjectName("valueLabel")
            no_predictions.setStyleSheet("color: #7799cc;")
            self.predictions_list.addWidget(no_predictions)
    
    def update_energy_level(self, level):
        """更新能量水平"""
        if hasattr(self, 'energy_viz'):
            self.energy_viz.set_energy_level(level)
    
    def request_update(self):
        """请求更新数据"""
        self.update_requested.emit()
    
    def request_market_analysis(self):
        """请求市场分析"""
        self.analyze_market_requested.emit()
    
    def clear_cosmic_events(self):
        """清除所有宇宙事件"""
        # 创建新的布局
        if hasattr(self, 'cosmic_events_layout'):
            # 删除旧的布局中的所有小部件
            while self.cosmic_events_layout.count():
                item = self.cosmic_events_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
        else:
            # 创建新的布局
            self.cosmic_events_layout = QVBoxLayout()
            self.cosmic_events_layout.setContentsMargins(5, 5, 5, 5)
            self.cosmic_events_layout.setSpacing(10)
            
        # 创建新的容器(如果不存在)
        if not hasattr(self, 'cosmic_events_container'):
            self.cosmic_events_container = QWidget()
    
    def add_cosmic_message(self, message, message_type="info"):
        """
        添加宇宙消息
        
        Args:
            message: 消息内容
            message_type: 消息类型 (info, warning, error)
        """
        if not hasattr(self, 'cosmic_events_layout'):
            self.clear_cosmic_events()
            
        # 创建消息HTML
        message_html = f"""
        <div class='cosmic-message {message_type}'>
            {message}
        </div>
        """
        
        # 添加消息小部件
        self.cosmic_events_layout.addWidget(
            self._create_event_widget(message_html)
        )
    
    def update_cosmic_events(self, events):
        """更新宇宙事件列表
        
        Args:
            events: 宇宙事件列表
        """
        try:
            # 清空现有事件
            self.cosmic_events_list.clear()
            
            # 添加新事件
            for event in events:
                event_type = event.get("type", "未知")
                event_message = event.get("message", "")
                event_time = event.get("timestamp", datetime.now().strftime("%H:%M:%S"))
                
                # 创建列表项
                item = QListWidgetItem(f"[{event_time}] [{event_type}] {event_message}")
                
                # 根据事件类型设置不同颜色
                if "意识" in event_type:
                    item.setForeground(QColor(153, 51, 255))  # 紫色
                elif "量子" in event_type:
                    item.setForeground(QColor(0, 128, 255))   # 蓝色
                elif "维度" in event_type:
                    item.setForeground(QColor(255, 153, 0))   # 橙色
                elif "预警" in event_type:
                    item.setForeground(QColor(255, 0, 0))     # 红色
                
                # 添加到列表
                self.cosmic_events_list.addItem(item)
            
            # 滚动到最新事件
            self.cosmic_events_list.scrollToBottom()
            
        except Exception as e:
            print(f"更新宇宙事件时出错: {str(e)}")
    
    def add_cosmic_event(self, event_type, event_message):
        """添加单个宇宙事件
        
        Args:
            event_type: 事件类型
            event_message: 事件消息
        """
        try:
            # 获取当前时间
            event_time = datetime.now().strftime("%H:%M:%S")
            
            # 创建列表项
            item = QListWidgetItem(f"[{event_time}] [{event_type}] {event_message}")
            
            # 根据事件类型设置不同颜色
            if "意识" in event_type:
                item.setForeground(QColor(153, 51, 255))  # 紫色
            elif "量子" in event_type:
                item.setForeground(QColor(0, 128, 255))   # 蓝色
            elif "维度" in event_type:
                item.setForeground(QColor(255, 153, 0))   # 橙色
            elif "预警" in event_type:
                item.setForeground(QColor(255, 0, 0))     # 红色
            
            # 添加到列表
            self.cosmic_events_list.addItem(item)
            
            # 滚动到最新事件
            self.cosmic_events_list.scrollToBottom()
            
        except Exception as e:
            print(f"添加宇宙事件时出错: {str(e)}") 