"""
组件状态小部件 - 显示系统组件状态
"""

import logging
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                         QPushButton, QFrame, QGridLayout)
from PyQt5.QtCore import Qt, QSize, pyqtSlot
from PyQt5.QtGui import QColor, QPainter, QBrush, QPen

logger = logging.getLogger("QuantumDesktop.ComponentStatusWidget")

class StatusIndicator(QWidget):
    """状态指示器 - 显示组件状态的指示灯"""
    
    def __init__(self, status='stopped', parent=None):
        super().__init__(parent)
        self.status = status
        self.setMinimumSize(16, 16)
        self.setMaximumSize(16, 16)
        
    def set_status(self, status):
        """设置状态"""
        self.status = status
        self.update()  # 触发重绘
        
    def paintEvent(self, event):
        """绘制事件处理"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 设置颜色
        if self.status == 'running':
            color = QColor(46, 204, 113)  # 绿色
        elif self.status == 'warning':
            color = QColor(241, 196, 15)  # 黄色
        elif self.status == 'error':
            color = QColor(231, 76, 60)   # 红色
        else:  # stopped
            color = QColor(149, 165, 166)  # 灰色
            
        # 绘制圆形
        painter.setPen(QPen(Qt.gray, 1))
        painter.setBrush(QBrush(color))
        painter.drawEllipse(2, 2, 12, 12)
        
        # 确保结束绘图
        painter.end()

class ComponentStatusWidget(QWidget):
    """组件状态小部件 - 显示系统各组件的运行状态"""
    
    def __init__(self, system_manager, parent=None):
        super().__init__(parent)
        self.system_manager = system_manager
        
        # 初始化UI
        self._init_ui()
        
        # 注册事件回调
        self.system_manager.register_event_callback(
            'component_status_changed', self._on_component_status_changed
        )
        
        # 初始状态更新
        self._update_status()
        
        logger.info("组件状态小部件初始化完成")
        
    def _init_ui(self):
        """初始化用户界面"""
        # 主布局
        self.main_layout = QVBoxLayout(self)
        
        # 标题
        title_label = QLabel("系统组件状态")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.main_layout.addWidget(title_label)
        
        # 分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        self.main_layout.addWidget(separator)
        
        # 组件状态网格
        self.status_grid = QGridLayout()
        self.status_grid.setVerticalSpacing(10)
        self.status_grid.setHorizontalSpacing(15)
        self.main_layout.addLayout(self.status_grid)
        
        # 添加组件状态行
        self._create_component_rows()
        
        # 添加底部控制按钮
        self.control_layout = QHBoxLayout()
        self.main_layout.addLayout(self.control_layout)
        
        # 刷新按钮
        self.refresh_button = QPushButton("刷新状态")
        self.refresh_button.clicked.connect(self._update_status)
        self.control_layout.addWidget(self.refresh_button)
        
        # 弹性空间
        self.main_layout.addStretch(1)
        
    def _create_component_rows(self):
        """创建各组件状态行"""
        # 组件定义
        components = [
            {'name': 'backend', 'label': '量子后端'},
            {'name': 'converter', 'label': '市场到量子转换器'},
            {'name': 'interpreter', 'label': '量子解释器'},
            {'name': 'analyzer', 'label': '市场分析器'}
        ]
        
        # 添加标题行
        self.status_grid.addWidget(QLabel("组件"), 0, 0)
        self.status_grid.addWidget(QLabel("状态"), 0, 1)
        
        # 存储组件状态指示器
        self.status_indicators = {}
        self.status_labels = {}
        
        # 添加组件行
        for i, component in enumerate(components):
            # 组件名称
            name_label = QLabel(component['label'])
            self.status_grid.addWidget(name_label, i + 1, 0)
            
            # 状态指示器
            indicator = StatusIndicator()
            self.status_indicators[component['name']] = indicator
            
            # 状态文本
            status_label = QLabel("未启动")
            self.status_labels[component['name']] = status_label
            
            # 添加到布局
            status_layout = QHBoxLayout()
            status_layout.addWidget(indicator)
            status_layout.addWidget(status_label)
            status_layout.addStretch(1)
            
            self.status_grid.addLayout(status_layout, i + 1, 1)
            
    def _update_status(self):
        """更新所有组件状态"""
        component_status = self.system_manager.get_component_status()
        
        for name, status in component_status.items():
            self._update_component_status(name, status)
            
    def _update_component_status(self, name, status):
        """更新单个组件状态"""
        if name not in self.status_indicators:
            return
            
        # 更新状态指示器
        self.status_indicators[name].set_status(status)
        
        # 更新状态文本
        status_text = self._get_status_text(status)
        self.status_labels[name].setText(status_text)
        
    def _get_status_text(self, status):
        """获取状态的文本描述"""
        if status == 'running':
            return "运行中"
        elif status == 'warning':
            return "警告"
        elif status == 'error':
            return "错误"
        else:  # stopped
            return "未启动"
            
    @pyqtSlot(object)
    def _on_component_status_changed(self, data):
        """组件状态变化回调处理"""
        if isinstance(data, dict) and 'component' in data and 'status' in data:
            self._update_component_status(data['component'], data['status']) 