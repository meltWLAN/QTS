#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - 量子意识视图
展示系统高级智能和感知能力
"""

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QProgressBar, QFrame, QScrollArea, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPalette, QLinearGradient, QGradient

class ConsciousnessView(QWidget):
    """量子意识视图"""
    
    update_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("consciousnessView")
        
        # 设置样式
        self.setStyleSheet("""
            QWidget#consciousnessView {
                background-color: #050510;
            }
            QLabel#titleLabel {
                color: #00ffff;
                font-size: 18px;
                font-weight: bold;
            }
            QLabel#statusLabel {
                color: #7f7fff;
                font-size: 14px;
            }
            QLabel#paramLabel {
                color: #d0d0ff;
                font-size: 14px;
            }
            QLabel#insightLabel {
                color: #80c0ff;
                font-size: 14px;
                padding: 10px;
                background-color: rgba(0, 20, 40, 0.6);
                border-radius: 5px;
            }
            QProgressBar {
                border: 1px solid #666;
                border-radius: 5px;
                background-color: #111133;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0077ff, stop:1 #00ffcc);
                border-radius: 5px;
            }
            QFrame#insightFrame {
                border: 1px solid #336;
                border-radius: 8px;
                background-color: rgba(10, 10, 30, 0.7);
                padding: 5px;
            }
        """)
        
        # 初始化UI
        self.init_ui()
        
        # 设置更新计时器
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.request_update)
        self.update_timer.start(2000)  # 每2秒更新一次
    
    def init_ui(self):
        """初始化UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # 标题
        title_label = QLabel("量子意识中枢")
        title_label.setObjectName("titleLabel")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 意识状态面板
        status_frame = QFrame()
        status_frame.setFrameShape(QFrame.StyledPanel)
        status_layout = QVBoxLayout(status_frame)
        
        # 觉醒状态
        self.status_label = QLabel("意识状态: 初始化中...")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.status_label)
        
        # 意识参数
        params_layout = QVBoxLayout()
        params_layout.setSpacing(10)
        
        # 觉醒度
        awareness_layout = QHBoxLayout()
        awareness_label = QLabel("宇宙觉醒度:")
        awareness_label.setObjectName("paramLabel")
        awareness_label.setFixedWidth(100)
        self.awareness_bar = QProgressBar()
        self.awareness_bar.setRange(0, 100)
        self.awareness_bar.setValue(0)
        awareness_layout.addWidget(awareness_label)
        awareness_layout.addWidget(self.awareness_bar)
        params_layout.addLayout(awareness_layout)
        
        # 市场直觉
        intuition_layout = QHBoxLayout()
        intuition_label = QLabel("市场直觉:")
        intuition_label.setObjectName("paramLabel")
        intuition_label.setFixedWidth(100)
        self.intuition_bar = QProgressBar()
        self.intuition_bar.setRange(0, 100)
        self.intuition_bar.setValue(0)
        intuition_layout.addWidget(intuition_label)
        intuition_layout.addWidget(self.intuition_bar)
        params_layout.addLayout(intuition_layout)
        
        # 宇宙共振
        resonance_layout = QHBoxLayout()
        resonance_label = QLabel("宇宙共振度:")
        resonance_label.setObjectName("paramLabel")
        resonance_label.setFixedWidth(100)
        self.resonance_bar = QProgressBar()
        self.resonance_bar.setRange(0, 100)
        self.resonance_bar.setValue(0)
        resonance_layout.addWidget(resonance_label)
        resonance_layout.addWidget(self.resonance_bar)
        params_layout.addLayout(resonance_layout)
        
        status_layout.addLayout(params_layout)
        main_layout.addWidget(status_frame)
        
        # 宇宙洞察面板
        insight_label = QLabel("宇宙洞察")
        insight_label.setObjectName("titleLabel")
        insight_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(insight_label)
        
        # 洞察滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        
        # 洞察容器
        self.insights_container = QWidget()
        self.insights_layout = QVBoxLayout(self.insights_container)
        self.insights_layout.setContentsMargins(0, 0, 0, 0)
        self.insights_layout.setSpacing(10)
        self.insights_layout.addStretch()
        
        scroll_area.setWidget(self.insights_container)
        main_layout.addWidget(scroll_area)
        
        # 占位，保证滚动区域有空间
        main_layout.setStretchFactor(scroll_area, 3)
    
    def update_consciousness_state(self, state):
        """更新意识状态显示"""
        if not state:
            return
            
        # 更新状态标签
        status_text = "已完全觉醒" if state.get("is_awake", False) else "觉醒中..."
        self.status_label.setText(f"意识状态: {status_text}")
        
        # 更新进度条
        awareness = state.get("awareness", 0)
        intuition = state.get("intuition", 0)
        resonance = state.get("resonance", 0)
        
        self.awareness_bar.setValue(int(awareness * 100))
        self.intuition_bar.setValue(int(intuition * 100))
        self.resonance_bar.setValue(int(resonance * 100))
        
        # 根据觉醒状态设置颜色
        if state.get("is_awake", False):
            self.status_label.setStyleSheet("color: #00ffcc; font-size: 16px; font-weight: bold;")
        else:
            self.status_label.setStyleSheet("color: #7f7fff; font-size: 14px;")
    
    def update_insights(self, insights):
        """更新宇宙洞察"""
        if not insights:
            return
            
        # 清除现有洞察
        while self.insights_layout.count() > 1:
            item = self.insights_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # 添加新洞察（最新的在顶部）
        for insight in reversed(insights):
            insight_frame = QFrame()
            insight_frame.setObjectName("insightFrame")
            insight_layout = QVBoxLayout(insight_frame)
            
            # 洞察内容
            content = insight.get("content", "")
            timestamp = insight.get("timestamp", "")
            
            content_label = QLabel(content)
            content_label.setObjectName("insightLabel")
            content_label.setWordWrap(True)
            
            # 时间戳
            time_label = QLabel(timestamp)
            time_label.setAlignment(Qt.AlignRight)
            time_label.setStyleSheet("color: #555599; font-size: 12px;")
            
            insight_layout.addWidget(content_label)
            insight_layout.addWidget(time_label)
            
            self.insights_layout.insertWidget(0, insight_frame)
    
    def request_update(self):
        """请求更新数据"""
        self.update_requested.emit() 