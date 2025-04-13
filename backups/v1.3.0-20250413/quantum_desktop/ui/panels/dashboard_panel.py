"""
仪表盘面板 - 系统概览显示
"""

import logging
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                         QFrame, QGridLayout, QSplitter, QProgressBar)
from PyQt5.QtCore import Qt, QTimer, pyqtSlot, QMetaObject
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QFont, QLinearGradient, QGradient
from PyQt5.QtChart import QChart, QChartView, QLineSeries, QValueAxis, QSplineSeries

logger = logging.getLogger("QuantumDesktop.DashboardPanel")

class ResourceMeter(QWidget):
    """资源仪表 - 显示系统资源使用情况"""
    
    def __init__(self, title, value=0, max_value=100, color=Qt.green, parent=None):
        super().__init__(parent)
        self.title = title
        self.value = value
        self.max_value = max_value
        self.color = color
        
        self.setMinimumHeight(120)
        
    def set_value(self, value):
        """设置当前值"""
        self.value = max(0, min(value, self.max_value))
        self.update()
        
    def paintEvent(self, event):
        """绘制事件处理"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 计算绘制区域
        rect = self.rect()
        width = rect.width()
        height = rect.height()
        
        # 绘制标题
        title_rect = rect.adjusted(0, 0, 0, int(-height * 0.7))
        painter.setPen(Qt.white)
        painter.setFont(QFont("Arial", 10, QFont.Bold))
        painter.drawText(title_rect, Qt.AlignCenter, self.title)
        
        # 绘制仪表背景
        gauge_rect = rect.adjusted(int(width * 0.1), int(height * 0.35), int(-width * 0.1), int(-height * 0.2))
        painter.setPen(QPen(Qt.gray, 1))
        painter.setBrush(QBrush(QColor(50, 50, 50)))
        painter.drawRoundedRect(gauge_rect, 5, 5)
        
        # 计算填充宽度
        fill_width = int(gauge_rect.width() * (self.value / self.max_value))
        
        # 创建渐变填充
        gradient = QLinearGradient(gauge_rect.topLeft(), gauge_rect.topRight())
        gradient.setColorAt(0, self.color)
        gradient.setColorAt(1, self.color.lighter(120))
        gradient.setSpread(QGradient.PadSpread)
        
        # 绘制填充部分
        fill_rect = gauge_rect.adjusted(0, 0, fill_width - gauge_rect.width(), 0)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(gradient))
        painter.drawRoundedRect(fill_rect, 5, 5)
        
        # 绘制数值
        painter.setPen(Qt.white)
        painter.setFont(QFont("Arial", 9))
        value_text = f"{self.value:.1f} / {self.max_value:.1f}"
        painter.drawText(gauge_rect, Qt.AlignCenter, value_text)
        
        # 绘制百分比
        percent_text = f"{(self.value / self.max_value * 100):.1f}%"
        percent_rect = rect.adjusted(0, int(height * 0.8), 0, 0)
        painter.drawText(percent_rect, Qt.AlignCenter, percent_text)
        
        # 确保在退出时结束绘图
        painter.end()

class SystemStatusChart(QChartView):
    """系统状态图表 - 显示系统资源使用随时间变化"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 创建图表
        self.chart = QChart()
        self.chart.setTitle("系统资源监控")
        self.chart.setAnimationOptions(QChart.SeriesAnimations)
        self.chart.legend().setVisible(True)
        self.chart.legend().setAlignment(Qt.AlignBottom)
        
        # 设置抗锯齿
        self.setRenderHint(QPainter.Antialiasing)
        
        # 设置图表
        self.setChart(self.chart)
        
        # 创建时间轴
        self.axis_x = QValueAxis()
        self.axis_x.setRange(0, 60)  # 60秒
        self.axis_x.setLabelFormat("%d秒")
        self.axis_x.setTitleText("时间")
        
        # 创建数值轴
        self.axis_y = QValueAxis()
        self.axis_y.setRange(0, 100)  # 0-100%
        self.axis_y.setLabelFormat("%d%%")
        self.axis_y.setTitleText("使用率")
        
        # 添加轴到图表
        self.chart.addAxis(self.axis_x, Qt.AlignBottom)
        self.chart.addAxis(self.axis_y, Qt.AlignLeft)
        
        # 创建CPU系列
        self.cpu_series = QSplineSeries()
        self.cpu_series.setName("CPU使用率")
        self.cpu_series.setPen(QPen(QColor(46, 204, 113), 2))
        
        # 创建内存系列
        self.memory_series = QSplineSeries()
        self.memory_series.setName("内存使用率")
        self.memory_series.setPen(QPen(QColor(52, 152, 219), 2))
        
        # 添加系列到图表
        self.chart.addSeries(self.cpu_series)
        self.chart.addSeries(self.memory_series)
        
        # 将系列附加到轴
        self.cpu_series.attachAxis(self.axis_x)
        self.cpu_series.attachAxis(self.axis_y)
        self.memory_series.attachAxis(self.axis_x)
        self.memory_series.attachAxis(self.axis_y)
        
        # 初始时间点
        self.time_point = 0
        
        # 初始数据点
        for i in range(60):
            self.cpu_series.append(i, 0)
            self.memory_series.append(i, 0)
            
    def update_data(self, cpu_usage, memory_usage):
        """更新图表数据"""
        # 增加时间点
        self.time_point += 1
        
        # 如果超过60秒，移动窗口
        if self.time_point > 60:
            self.axis_x.setRange(self.time_point - 60, self.time_point)
            
        # 添加数据点
        self.cpu_series.append(self.time_point, cpu_usage)
        self.memory_series.append(self.time_point, memory_usage)
        
        # 如果有超过100个点，删除旧点以优化性能
        if len(self.cpu_series.points()) > 120:
            points = self.cpu_series.pointsVector()
            self.cpu_series.replace(points[60:])
            
            points = self.memory_series.pointsVector()
            self.memory_series.replace(points[60:])

class DashboardPanel(QWidget):
    """仪表盘面板 - 显示系统概览"""
    
    def __init__(self, system_manager, parent=None):
        super().__init__(parent)
        self.system_manager = system_manager
        
        # 初始化更新间隔
        self.update_interval = 1000  # 默认1秒
        
        # 初始化UI
        self._init_ui()
        
        # 注册事件回调
        self.system_manager.register_event_callback(
            'resources_updated', self._on_resources_updated
        )
        
        # 更新定时器
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._update_displays)
        
        logger.info("仪表盘面板初始化完成")
        
    def _init_ui(self):
        """初始化用户界面"""
        # 主布局
        self.main_layout = QVBoxLayout(self)
        
        # 顶部状态区域
        self.status_frame = QFrame()
        self.status_frame.setFrameShape(QFrame.StyledPanel)
        self.status_frame.setFrameShadow(QFrame.Raised)
        self.status_layout = QHBoxLayout(self.status_frame)
        
        # CPU资源仪表
        self.cpu_meter = ResourceMeter("CPU使用率", 0, 100, QColor(46, 204, 113))
        self.status_layout.addWidget(self.cpu_meter)
        
        # 内存资源仪表
        self.memory_meter = ResourceMeter("内存使用", 0, 100, QColor(52, 152, 219))
        self.status_layout.addWidget(self.memory_meter)
        
        # 组件状态仪表
        self.component_meter = ResourceMeter("活动组件", 0, 4, QColor(155, 89, 182))
        self.status_layout.addWidget(self.component_meter)
        
        # 添加状态区域到主布局
        self.main_layout.addWidget(self.status_frame)
        
        # 资源图表
        self.status_chart = SystemStatusChart()
        self.main_layout.addWidget(self.status_chart)
        
        # 底部状态网格
        self.bottom_frame = QFrame()
        self.bottom_frame.setFrameShape(QFrame.StyledPanel)
        self.bottom_frame.setFrameShadow(QFrame.Raised)
        self.bottom_layout = QGridLayout(self.bottom_frame)
        
        # 标签
        self.bottom_layout.addWidget(QLabel("组件"), 0, 0)
        self.bottom_layout.addWidget(QLabel("内存使用"), 0, 1)
        self.bottom_layout.addWidget(QLabel("状态"), 0, 2)
        
        # 组件状态行
        components = [
            {'name': 'backend', 'label': '量子后端'},
            {'name': 'converter', 'label': '市场到量子转换器'},
            {'name': 'interpreter', 'label': '量子解释器'},
            {'name': 'analyzer', 'label': '市场分析器'}
        ]
        
        # 组件状态进度条和标签
        self.component_bars = {}
        self.component_status_labels = {}
        
        for i, component in enumerate(components):
            # 组件名
            name_label = QLabel(component['label'])
            self.bottom_layout.addWidget(name_label, i + 1, 0)
            
            # 内存使用进度条
            progress_bar = QProgressBar()
            progress_bar.setMinimum(0)
            progress_bar.setMaximum(100)
            progress_bar.setValue(0)
            self.component_bars[component['name']] = progress_bar
            self.bottom_layout.addWidget(progress_bar, i + 1, 1)
            
            # 状态
            status_label = QLabel("未启动")
            self.component_status_labels[component['name']] = status_label
            self.bottom_layout.addWidget(status_label, i + 1, 2)
            
        # 添加底部区域到主布局
        self.main_layout.addWidget(self.bottom_frame)
        
    def set_update_interval(self, interval_ms):
        """设置更新间隔
        
        Args:
            interval_ms: 更新间隔，单位为毫秒
        """
        self.update_interval = interval_ms
        logger.info(f"仪表盘更新间隔设置为 {interval_ms}ms")
        
        # 如果定时器已经运行，则重新启动它
        if self.update_timer.isActive():
            self.update_timer.stop()
            self.update_timer.start(self.update_interval)
    
    def apply_settings(self, config):
        """应用配置设置
        
        Args:
            config: 配置字典
        """
        logger.info("应用配置到仪表盘面板")
        
        # 应用UI设置
        if 'ui' in config:
            ui_config = config['ui']
            if 'update_interval' in ui_config:
                self.set_update_interval(ui_config['update_interval'])
                
        # 应用系统设置
        if 'system' in config:
            sys_config = config['system']
            if 'max_threads' in sys_config:
                # 更新CPU仪表的最大值
                max_threads = sys_config['max_threads']
                logger.info(f"设置CPU仪表最大值为 {max_threads}")
                
        # 应用量子设置
        if 'quantum' in config:
            # 可以在这里处理量子相关设置，例如显示量子后端类型
            # 或者更新量子相关组件的状态
            pass
    
    def on_system_started(self):
        """系统启动时调用"""
        # 启动更新定时器
        self.update_timer.start(self.update_interval)
        
    def on_system_stopped(self):
        """系统停止时调用"""
        # 停止更新定时器
        self.update_timer.stop()
        
        # 重置显示
        self.cpu_meter.set_value(0)
        self.memory_meter.set_value(0)
        self.component_meter.set_value(0)
        
        # 重置组件进度条
        for bar in self.component_bars.values():
            bar.setValue(0)
        
    @pyqtSlot(object)
    def _on_resources_updated(self, resources):
        """资源更新回调处理"""
        # 将资源数据存储为成员变量，以便在_update_displays中使用
        self.latest_resources = resources
        # 使用QTimer.singleShot确保在主线程中安全更新UI
        QTimer.singleShot(0, self._update_displays)
        
    def _update_displays(self):
        """更新仪表盘显示（确保在UI线程中调用）"""
        try:
            # 使用回调函数提供的资源数据或从系统管理器获取最新数据
            resources = getattr(self, 'latest_resources', None) or self.system_manager.get_latest_resources()
            if not resources:
                return
                
            # 更新CPU使用率
            cpu_usage = resources.get('cpu_usage', 0)
            self.cpu_meter.set_value(cpu_usage)
            
            # 更新内存使用率
            memory_usage = resources.get('memory_usage', 0)
            self.memory_meter.set_value(memory_usage)
            
            # 更新图表
            self.status_chart.update_data(cpu_usage, memory_usage)
            
            # 获取最新组件状态（确保直接从系统管理器获取最新状态）
            component_statuses = self.system_manager.get_component_status()
            active_components = sum(1 for status in component_statuses.values() if status == 'running')
            self.component_meter.set_value(active_components)
            
            # 更新组件进度条和状态标签
            component_memory = resources.get('component_memory', {})
            for name, bar in self.component_bars.items():
                # 计算组件内存占总内存的百分比
                component_mem = component_memory.get(name, 0)
                if memory_usage > 0:
                    percent = min(component_mem / (memory_usage / 100), 100)
                else:
                    percent = 0
                    
                bar.setValue(int(percent))
                
                # 更新组件状态标签
                status = component_statuses.get(name, 'unknown')
                status_text = "运行中" if status == 'running' else "未启动"
                status_label = self.component_status_labels.get(name)
                if status_label:
                    status_label.setText(status_text)
                    # 根据状态设置样式
                    if status == 'running':
                        status_label.setStyleSheet("color: #2ecc71; font-weight: bold;")  # 绿色
                    else:
                        status_label.setStyleSheet("color: #e74c3c;")  # 红色
                    
        except Exception as e:
            logger.error(f"更新显示错误: {str(e)}") 