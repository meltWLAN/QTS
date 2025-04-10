#!/usr/bin/env python3
"""
超神量子系统 - 可视化组件
专为中国市场分析设计的量子风格UI组件
"""

import numpy as np
import pandas as pd
from datetime import datetime
from PyQt5.QtWidgets import (QLabel, QFrame, QVBoxLayout, QHBoxLayout, 
                             QWidget, QSizePolicy, QGraphicsDropShadowEffect, QMenu)
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QSize, QRectF, QPointF, QMargins
from PyQt5.QtGui import QPixmap, QColor, QPainter, QPen, QPainterPath, QLinearGradient, QRadialGradient, QFont, QFontMetrics, QBrush
from PyQt5.QtChart import QChart, QChartView, QLineSeries, QValueAxis, QSplineSeries, QScatterSeries, QAreaSeries, QDateTimeAxis
import random
import math
import logging

logger = logging.getLogger(__name__)


class QuantumLogo(QLabel):
    """量子特效Logo组件"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(130, 130)
        self.setMaximumSize(130, 130)
        self.angle = 0
        self.pulse_scale = 1.0
        self.pulse_direction = 0.01
        self.energy_level = 0.3
        self.energy_direction = 0.005
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(30)  # 每30ms更新一次，更流畅的动画
        # 增加更多粒子和轨道
        self.particle_positions = np.random.rand(30, 2)  # 30个粒子，每个粒子有x,y坐标
        self.particle_speeds = np.random.rand(30, 2) * 0.02 - 0.01  # 随机速度
        self.particle_sizes = np.random.randint(2, 8, 30)  # 随机大小
        self.particle_energy = np.random.rand(30) * 0.5 + 0.5  # 粒子能量
        self.particle_orbits = np.random.rand(30) * 0.5 + 0.5  # 轨道半径
        # 新增量子特效参数
        self.waveform_phase = 0
        self.resonance_points = np.random.rand(5, 2)  # 量子共振点

    def update_animation(self):
        """更新动画状态"""
        # 更新旋转角度
        self.angle = (self.angle + 2) % 360
        
        # 更新脉冲效果
        self.pulse_scale += self.pulse_direction
        if self.pulse_scale > 1.1 or self.pulse_scale < 0.9:
            self.pulse_direction *= -1
            
        # 更新能量水平
        self.energy_level += self.energy_direction
        if self.energy_level > 0.8 or self.energy_level < 0.3:
            self.energy_direction *= -1
            
        # 更新波形相位
        self.waveform_phase = (self.waveform_phase + 0.1) % (2 * np.pi)
            
        # 更新粒子位置
        self.particle_positions += self.particle_speeds * self.particle_energy.reshape(-1, 1)
        
        # 边界反弹
        bounce_mask = (self.particle_positions < 0) | (self.particle_positions > 1)
        self.particle_speeds[bounce_mask] *= -1
        
        # 随机调整共振点
        if np.random.random() < 0.05:  # 5%的概率
            idx = np.random.randint(0, len(self.resonance_points))
            self.resonance_points[idx] = np.random.rand(2)
            
        self.update()  # 重绘

    def paintEvent(self, event):
        """绘制量子特效Logo"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制背景辐射效果
        center_x, center_y = self.width() / 2, self.height() / 2
        
        # 创建径向渐变背景
        gradient = QRadialGradient(center_x, center_y, 70 * self.pulse_scale)
        gradient.setColorAt(0, QColor(40, 130, 200, 100))
        gradient.setColorAt(0.7, QColor(20, 80, 160, 50))
        gradient.setColorAt(1, QColor(10, 40, 120, 10))
        painter.setBrush(gradient)
        painter.setPen(Qt.NoPen)
        
        # 修复: 使用QRectF代替浮点数参数
        rect = QRectF(
            center_x - 65 * self.pulse_scale, 
            center_y - 65 * self.pulse_scale, 
            130 * self.pulse_scale, 
            130 * self.pulse_scale
        )
        painter.drawEllipse(rect)
        
        # 绘制量子能量波
        painter.setPen(QPen(QColor(80, 180, 255, 40), 1))
        for i in range(3):
            scale = 0.7 + i * 0.15
            size = 100 * scale * self.pulse_scale
            x = center_x - size/2
            y = center_y - size/2
            # 修复: 使用QRectF
            painter.drawEllipse(QRectF(x, y, size, size))
        
        # 绘制量子轨道系统
        painter.translate(center_x, center_y)
        
        # 绘制主量子轨道
        painter.rotate(self.angle)
        for i in range(3):
            opacity = 150 - i * 30
            width = 2 - i * 0.5
            color = QColor(100, 200, 255, opacity)
            painter.setPen(QPen(color, width))
            size = (50 + i * 15) * self.pulse_scale
            # 修复: 使用QRectF
            painter.drawEllipse(QRectF(-size/2, -size/2, size, size))
        
        # 绘制正交轨道
        painter.rotate(60)
        painter.setPen(QPen(QColor(0, 180, 220, 100), 1.5))
        # 修复: 使用QRectF
        painter.drawEllipse(QRectF(-55 * self.pulse_scale/2, -55 * self.pulse_scale/2, 
                          55 * self.pulse_scale, 55 * self.pulse_scale))
        
        painter.rotate(60)
        painter.setPen(QPen(QColor(140, 210, 255, 80), 1))
        # 修复: 使用QRectF
        painter.drawEllipse(QRectF(-45 * self.pulse_scale/2, -45 * self.pulse_scale/2, 
                          45 * self.pulse_scale, 45 * self.pulse_scale))
        
        # 绘制波动效果
        painter.resetTransform()
        painter.translate(center_x, center_y)
        phase = self.waveform_phase
        painter.setPen(QPen(QColor(100, 200, 255, 60), 1, Qt.DashLine))
        points = []
        for i in range(0, 360, 10):
            angle = i * np.pi / 180
            r = 40 * (1 + 0.1 * np.sin(6 * angle + phase)) * self.pulse_scale
            x = float(r * np.cos(angle))
            y = float(r * np.sin(angle))
            points.append(QPointF(x, y))
        
        for i in range(len(points)-1):
            painter.drawLine(points[i], points[i+1])
        painter.drawLine(points[-1], points[0])
        
        # 绘制共振点
        painter.resetTransform()
        for point in self.resonance_points:
            x = float(center_x + (point[0] - 0.5) * 100)
            y = float(center_y + (point[1] - 0.5) * 100)
            # 创建共振点渐变
            res_gradient = QRadialGradient(x, y, 15)
            res_gradient.setColorAt(0, QColor(160, 230, 255, 180))
            res_gradient.setColorAt(0.5, QColor(80, 160, 255, 100))
            res_gradient.setColorAt(1, QColor(40, 100, 220, 0))
            painter.setBrush(res_gradient)
            painter.setPen(Qt.NoPen)
            size = float(12 * (0.8 + 0.4 * np.sin(phase + point[0] * 10)))
            # 修复: 使用QRectF
            painter.drawEllipse(QRectF(x - size/2, y - size/2, size, size))
        
        # 绘制中心核心
        painter.resetTransform()
        painter.translate(center_x, center_y)
        core_size = float(16 * (1 + 0.2 * np.sin(phase * 2)))
        core_gradient = QRadialGradient(0, 0, core_size)
        core_energy = 120 + int(80 * self.energy_level)
        core_gradient.setColorAt(0, QColor(120, core_energy, 255))
        core_gradient.setColorAt(0.7, QColor(60, core_energy//2, 220))
        core_gradient.setColorAt(1, QColor(0, 40, 180, 180))
        painter.setBrush(core_gradient)
        painter.setPen(Qt.NoPen)
        # 修复: 使用QRectF
        painter.drawEllipse(QRectF(-core_size/2, -core_size/2, core_size, core_size))
        
        # 绘制光晕
        glow_gradient = QRadialGradient(0, 0, core_size * 1.5)
        glow_gradient.setColorAt(0, QColor(180, 230, 255, 100))
        glow_gradient.setColorAt(1, QColor(100, 180, 255, 0))
        painter.setBrush(glow_gradient)
        # 修复: 使用QRectF
        painter.drawEllipse(QRectF(-core_size, -core_size, core_size * 2, core_size * 2))
        
        # 绘制量子粒子
        painter.resetTransform()
        painter.setPen(Qt.NoPen)
        for i, (x, y) in enumerate(self.particle_positions):
            # 粒子颜色随位置和能量变化
            hue = (x * 360 + self.angle) % 360
            energy = self.particle_energy[i]
            particle_color = QColor()
            particle_color.setHsv(int(hue), 200, 255, int(150 * energy))
            painter.setBrush(particle_color)
            
            # 粒子大小有微小的波动
            size = float(self.particle_sizes[i] * (0.8 + 0.4 * np.sin(phase + i * 0.2)))
            
            # 转换为实际坐标并绘制
            px = float(center_x + (x - 0.5) * 100)
            py = float(center_y + (y - 0.5) * 100)
            painter.drawEllipse(QRectF(px - size/2, py - size/2, size, size))
            
            # 粒子核心
            painter.setBrush(particle_color)
            # 修复: 使用QRectF
            painter.drawEllipse(QRectF(px - size/2, py - size/2, size, size))


class QuantumLoadingBar(QFrame):
    """量子风格的加载进度条"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(6)
        self.setMaximumHeight(6)
        self._progress = 0
        self._particles = []
        self._wave_offset = 0
        self._pulse_direction = 1
        self._pulse_value = 0
        self._generate_particles()
        
        # 添加动画定时器
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_animation)
        self._timer.start(50)  # 50ms一帧
        
    def _generate_particles(self):
        """生成粒子效果"""
        self._particles = []
        for _ in range(25):  # 增加粒子数量
            pos = np.random.rand()  # 0-1之间的随机位置
            speed = np.random.rand() * 0.03 + 0.005  # 随机速度
            size = np.random.randint(3, 9)  # 随机大小
            hue = np.random.randint(180, 250)  # 随机色调 (蓝色系)
            energy = np.random.rand() * 0.5 + 0.5  # 粒子能量
            self._particles.append({
                "pos": pos, 
                "speed": speed, 
                "size": size, 
                "hue": hue,
                "energy": energy,
                "glow": np.random.randint(0, 2)  # 是否有发光效果
            })
    
    def _update_animation(self):
        """更新动画状态"""
        # 更新波浪偏移
        self._wave_offset = (self._wave_offset + 0.1) % (2 * np.pi)
        
        # 更新脉冲效果
        self._pulse_value += 0.1 * self._pulse_direction
        if self._pulse_value > 1.0:
            self._pulse_direction = -1
        elif self._pulse_value < 0.0:
            self._pulse_direction = 1
            
        # 更新粒子
        for particle in self._particles:
            if particle["pos"] <= self._progress / 100:
                # 更新粒子位置
                particle["pos"] += particle["speed"] * particle["energy"]
                if particle["pos"] > 1:
                    # 重置回起点
                    particle["pos"] = 0
                    particle["size"] = np.random.randint(3, 9)
                    particle["hue"] = np.random.randint(180, 250)
                    particle["energy"] = np.random.rand() * 0.5 + 0.5
                    particle["glow"] = np.random.randint(0, 2)
                    
        self.update()  # 重绘
        
    def set_progress(self, value):
        """设置进度值 (0-100)"""
        self._progress = max(0, min(100, value))
        self.update()
        
    def paintEvent(self, event):
        """绘制进度条"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        width = self.width()
        height = self.height()
        
        # 创建进度条区域
        rect = QRectF(0, 0, width, height)
        
        # 绘制背景
        gradient = QLinearGradient(0, 0, width, 0)
        gradient.setColorAt(0, QColor(10, 20, 30))
        gradient.setColorAt(1, QColor(30, 40, 60))
        painter.setPen(Qt.NoPen)
        painter.setBrush(gradient)
        painter.drawRect(rect)
        
        # 绘制进度
        if self._progress > 0:
            progress_width = width * self._progress / 100
            active_rect = QRectF(0, 0, progress_width, height)
            
            progress_gradient = QLinearGradient(0, 0, progress_width, 0)
            progress_gradient.setColorAt(0, QColor(40, 120, 200))
            progress_gradient.setColorAt(0.5, QColor(60, 160, 220))
            progress_gradient.setColorAt(1, QColor(80, 180, 255))
            painter.setBrush(progress_gradient)
            painter.drawRect(active_rect)
            
            # 绘制波动效果
            wave_pen = QPen(QColor(100, 200, 255, 180), 1)
            painter.setPen(wave_pen)
            wave_offset = self._wave_offset
            wave_path = QPainterPath()
            
            wave_path.moveTo(0, height/2)
            for x in range(0, int(progress_width), 3):
                wave_y = height/2 + np.sin(x/10 + wave_offset) * (height/3)
                wave_path.lineTo(x, wave_y)
                
            painter.drawPath(wave_path)
            
            # 绘制粒子效果
            painter.setPen(Qt.NoPen)
            for particle in self._particles:
                pos = particle["pos"]
                if pos <= self._progress / 100:  # 只显示进度范围内的粒子
                    size = particle["size"]
                    hue = particle["hue"]
                    energy = particle["energy"]
                    px = width * pos
                    py = height/2 + np.sin(px/10 + wave_offset) * (height/3) * particle["energy"]
                    
                    color = QColor()
                    color.setHsv(hue, 200, 255, int(200 * energy))
                    painter.setBrush(color)
                    
                    # 绘制发光效果
                    if particle["glow"] == 1:
                        glow = QRadialGradient(px, py, size*2)
                        glow.setColorAt(0, color)
                        glow_color = QColor(color)
                        glow_color.setAlpha(0)
                        glow.setColorAt(1, glow_color)
                        painter.setBrush(glow)
                        # 修复: 使用QRectF
                        painter.drawEllipse(QRectF(px-size, py-size, size*2, size*2))
                    
                    # 绘制粒子
                    painter.setBrush(color)
                    # 修复: 使用QRectF
                    painter.drawEllipse(QRectF(px-size/2, py-size/2, size, size))


class QuantumCard(QFrame):
    """量子风格的信息卡片"""
    def __init__(self, title, value="", description="", bg_color=(30, 40, 60), parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: rgba({bg_color[0]}, {bg_color[1]}, {bg_color[2]}, 180);
                border-radius: 8px;
                border: 1px solid #3a3a6a;
            }}
        """)
        self.setMinimumHeight(120)
        
        # 添加阴影效果
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(0, 3)
        self.setGraphicsEffect(shadow)
        
        # 创建布局
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(15, 10, 15, 10)
        
        # 标题
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("""
            font-family: '微软雅黑';
            font-size: 14px;
            color: #8090b0;
        """)
        self.layout.addWidget(self.title_label)
        
        # 值
        self.value_label = QLabel(str(value))
        self.value_label.setStyleSheet("""
            font-family: '微软雅黑';
            font-size: 20px;
            font-weight: bold;
            color: #40a0ff;
        """)
        self.layout.addWidget(self.value_label)
        
        # 描述
        self.desc_label = QLabel(description)
        self.desc_label.setStyleSheet("""
            font-family: '微软雅黑';
            font-size: 12px;
            color: #a0b0c0;
        """)
        self.layout.addWidget(self.desc_label)
    
    def update_value(self, value, description=None):
        """更新卡片的值和描述"""
        self.value_label.setText(str(value))
        if description is not None:
            self.desc_label.setText(description)


class QuantumChart(QChartView):
    """量子图表组件"""
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setBackgroundBrush(QColor(20, 25, 35))
        
        # 创建图表
        self.chart = QChart()
        self.chart.setTitle(title)
        self.chart.setTitleFont(QFont("Arial", 12, QFont.Bold))
        self.chart.setTitleBrush(QColor(220, 230, 255))
        self.chart.setBackgroundBrush(QColor(20, 25, 35))
        self.chart.legend().setLabelBrush(QColor(200, 210, 255))
        self.chart.legend().setAlignment(Qt.AlignBottom)
        
        # 设置图表外观
        self.chart.layout().setContentsMargins(0, 0, 0, 0)
        self.chart.setMargins(QMargins(10, 10, 10, 10))
        self.chart.setAnimationOptions(QChart.SeriesAnimations)
        
        # 设置为视图的图表
        self.setChart(self.chart)
        
        # 保存所有系列以便更新
        self.series_dict = {}
        
        # 高维数据支持
        self.dimension_view_mode = "2D"  # 可以是 "2D", "3D", "Quantum"
        self.dimension_reduction_method = "PCA"  # 降维方法
        self.high_dimension_data = {}  # 存储原始高维数据
        
        # 量子渲染支持
        self.quantum_particles = []  # 量子渲染模式下的粒子
        self.quantum_animation_timer = QTimer(self)
        self.quantum_animation_timer.timeout.connect(self.update_quantum_particles)
        
        # 添加右键菜单支持
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        
    def set_dimension_view_mode(self, mode):
        """设置维度视图模式"""
        if mode in ["2D", "3D", "Quantum"]:
            self.dimension_view_mode = mode
            if mode == "Quantum" and not self.quantum_animation_timer.isActive():
                self._generate_quantum_particles()
                self.quantum_animation_timer.start(30)  # 30ms更新一次
            elif mode != "Quantum" and self.quantum_animation_timer.isActive():
                self.quantum_animation_timer.stop()
            self.update()
    
    def show_context_menu(self, pos):
        """显示右键菜单"""
        menu = QMenu(self)
        
        # 添加视图模式切换
        view_menu = menu.addMenu("视图模式")
        modes = ["2D", "3D", "Quantum"]
        for mode in modes:
            action = view_menu.addAction(mode)
            action.setCheckable(True)
            action.setChecked(self.dimension_view_mode == mode)
            action.triggered.connect(lambda checked, m=mode: self.set_dimension_view_mode(m))
        
        # 添加降维方法选择
        if len(self.high_dimension_data) > 0:
            dim_menu = menu.addMenu("降维方法")
            methods = ["PCA", "TSNE", "UMAP"]
            for method in methods:
                action = dim_menu.addAction(method)
                action.setCheckable(True)
                action.setChecked(self.dimension_reduction_method == method)
                action.triggered.connect(lambda checked, m=method: self.set_dimension_reduction_method(m))
        
        menu.exec_(self.mapToGlobal(pos))
    
    def _generate_quantum_particles(self, count=150):
        """生成量子粒子用于量子渲染模式"""
        self.quantum_particles = []
        plot_area = self.chart.plotArea()
        for _ in range(count):
            x = random.uniform(plot_area.left(), plot_area.right())
            y = random.uniform(plot_area.top(), plot_area.bottom())
            size = random.uniform(2, 6)
            speed = random.uniform(0.5, 2.0)
            energy = random.uniform(0.5, 1.0)
            color = QColor(
                int(100 + 155 * random.random()), 
                int(150 + 105 * random.random()), 
                int(200 + 55 * random.random()), 
                int(100 + 100 * energy)
            )
            self.quantum_particles.append({
                'x': x, 'y': y, 'size': size, 'speed': speed, 
                'color': color, 'energy': energy,
                'angle': random.uniform(0, 2 * np.pi),
                'phase': random.uniform(0, 2 * np.pi)
            })
    
    def update_quantum_particles(self):
        """更新量子粒子状态"""
        if not self.quantum_particles or self.dimension_view_mode != "Quantum":
            return
            
        plot_area = self.chart.plotArea()
        width = plot_area.width()
        height = plot_area.height()
        
        # 更新每个粒子
        for particle in self.quantum_particles:
            # 更新位置
            particle['x'] += math.cos(particle['angle']) * particle['speed']
            particle['y'] += math.sin(particle['angle']) * particle['speed']
            
            # 添加量子效果 - 波动性
            particle['phase'] += 0.05
            wave_effect = math.sin(particle['phase']) * 0.5
            particle['size'] = particle['size'] * (1 + wave_effect * 0.2)
            
            # 随机微调角度 - 量子不确定性
            particle['angle'] += random.uniform(-0.1, 0.1)
            
            # 边界检查并反弹
            if particle['x'] < plot_area.left() or particle['x'] > plot_area.right():
                particle['angle'] = math.pi - particle['angle']
            if particle['y'] < plot_area.top() or particle['y'] > plot_area.bottom():
                particle['angle'] = -particle['angle']
        
        self.update()
    
    def update_quantum_view(self):
        """更新量子视图"""
        if self.dimension_view_mode == "Quantum":
            self._generate_quantum_particles()
            
    def add_high_dimension_data(self, data, name, dimensions=None):
        """添加高维数据并进行降维可视化"""
        try:
            from sklearn.decomposition import PCA
            from sklearn.manifold import TSNE
            
            # 存储原始高维数据
            self.high_dimension_data[name] = {
                'data': data,
                'dimensions': dimensions if dimensions else list(range(data.shape[1]))
            }
            
            # 根据当前模式进行降维
            reduced_data = None
            if self.dimension_reduction_method == "PCA":
                pca = PCA(n_components=2)
                reduced_data = pca.fit_transform(data)
            elif self.dimension_reduction_method == "TSNE":
                tsne = TSNE(n_components=2)
                reduced_data = tsne.fit_transform(data)
            else:  # 默认使用PCA
                pca = PCA(n_components=2)
                reduced_data = pca.fit_transform(data)
                
            # 创建散点序列
            scatter = self.add_scatter_series(
                reduced_data[:, 0], reduced_data[:, 1], 
                name=f"{name} ({self.dimension_reduction_method})"
            )
            
            return scatter
        except ImportError:
            # 如果没有sklearn，回退到简单的2D投影
            logger.warning("未安装sklearn，使用简单2D投影")
            if data.shape[1] >= 2:
                return self.add_scatter_series(data[:, 0], data[:, 1], name=name)
            return None
        except Exception as e:
            logger.error(f"高维数据可视化错误: {str(e)}")
            return None
    
    def set_dimension_reduction_method(self, method):
        """设置降维方法"""
        if method in ["PCA", "TSNE", "UMAP"]:
            self.dimension_reduction_method = method
            # 重新处理现有数据
            for name, data_info in self.high_dimension_data.items():
                if name in self.series_dict:
                    self.chart.removeSeries(self.series_dict[name])
                self.add_high_dimension_data(data_info['data'], name, data_info['dimensions'])
    
    def paintEvent(self, event):
        """自定义绘制事件"""
        if self.dimension_view_mode == "Quantum":
            # 在标准绘制之前先绘制量子效果
            painter = QPainter(self.viewport())
            painter.setRenderHint(QPainter.Antialiasing)
            
            # 绘制量子粒子
            for particle in self.quantum_particles:
                # 创建径向渐变
                gradient = QRadialGradient(particle['x'], particle['y'], particle['size'] * 2)
                color = particle['color']
                gradient.setColorAt(0, color)
                gradient.setColorAt(1, QColor(color.red(), color.green(), color.blue(), 0))
                
                painter.setBrush(QBrush(gradient))
                painter.setPen(Qt.NoPen)
                painter.drawEllipse(
                    QRectF(
                        particle['x'] - particle['size'],
                        particle['y'] - particle['size'],
                        particle['size'] * 2,
                        particle['size'] * 2
                    )
                )
                
                # 绘制能量线
                if random.random() < 0.3 * particle['energy']:
                    painter.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 120), 0.5))
                    line_length = particle['size'] * 3
                    end_x = float(particle['x'] + math.cos(particle['angle']) * line_length)
                    end_y = float(particle['y'] + math.sin(particle['angle']) * line_length)
                    painter.drawLine(QPointF(particle['x'], particle['y']), QPointF(end_x, end_y))
            
        # 调用原始绘制
        super().paintEvent(event)


class QuantumHeatmap(QWidget):
    """量子风格热力图"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 200)
        self.data = None
        self.row_labels = []
        self.col_labels = []
        self.title = ""
        self.max_value = 1.0
        self.min_value = -1.0
        
        # 设置字体
        self.title_font = QFont("微软雅黑", 12, QFont.Bold)
        self.label_font = QFont("微软雅黑", 9)
        
        # 添加阴影效果
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 100))
        shadow.setOffset(0, 5)
        self.setGraphicsEffect(shadow)
    
    def set_data(self, data, row_labels=None, col_labels=None, title="相关性矩阵"):
        """设置热力图数据"""
        self.data = data
        self.row_labels = row_labels if row_labels is not None else []
        self.col_labels = col_labels if col_labels is not None else []
        self.title = title
        
        # 计算数值范围
        self.max_value = max(data.max().max(), abs(data.min().min()))
        self.min_value = -self.max_value
        
        self.update()
    
    def get_color(self, value):
        """根据值获取颜色"""
        if value > 0:
            # 正值 - 蓝色系
            intensity = min(1.0, value / self.max_value) if self.max_value > 0 else 0
            r = int(50 + 70 * intensity)
            g = int(100 + 120 * intensity)
            b = int(180 + 75 * intensity)
            return QColor(r, g, b, 200)
        else:
            # 负值 - 红色系
            intensity = min(1.0, abs(value) / abs(self.min_value)) if self.min_value < 0 else 0
            r = int(180 + 75 * intensity)
            g = int(70 + 80 * intensity)
            b = int(80 + 20 * intensity)
            return QColor(r, g, b, 200)
    
    def paintEvent(self, event):
        """绘制热力图"""
        if self.data is None:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制背景
        painter.fillRect(self.rect(), QColor(20, 25, 45, 200))
        
        # 绘制标题
        painter.setFont(self.title_font)
        painter.setPen(QColor(160, 190, 255))
        title_rect = QRectF(0, 10, self.width(), 25)
        painter.drawText(title_rect, Qt.AlignCenter, self.title)
        
        # 计算热力图区域
        margin_top = 45
        margin_left = 80 if self.row_labels else 40
        margin_right = 20
        margin_bottom = 80 if self.col_labels else 40
        
        heatmap_width = self.width() - margin_left - margin_right
        heatmap_height = self.height() - margin_top - margin_bottom
        
        # 确保数据有效
        if self.data.shape[0] <= 0 or self.data.shape[1] <= 0:
            return
        
        # 计算单元格大小
        cell_width = heatmap_width / self.data.shape[1]
        cell_height = heatmap_height / self.data.shape[0]
        
        # 绘制行标签
        if self.row_labels:
            painter.setFont(self.label_font)
            painter.setPen(QColor(180, 190, 210))
            for i, label in enumerate(self.row_labels):
                if i < self.data.shape[0]:
                    text_rect = QRectF(5, margin_top + i * cell_height, margin_left - 10, cell_height)
                    painter.drawText(text_rect, Qt.AlignRight | Qt.AlignVCenter, label)
        
        # 绘制列标签
        if self.col_labels:
            painter.setFont(self.label_font)
            painter.setPen(QColor(180, 190, 210))
            for j, label in enumerate(self.col_labels):
                if j < self.data.shape[1]:
                    text_rect = QRectF(margin_left + j * cell_width, self.height() - margin_bottom + 5, 
                                       cell_width, margin_bottom - 10)
                    painter.save()
                    painter.translate(margin_left + (j + 0.5) * cell_width, self.height() - margin_bottom + 10)
                    painter.rotate(-45)  # 旋转45度
                    painter.drawText(QRectF(-50, 0, 100, 20), Qt.AlignCenter, label)
                    painter.restore()
        
        # 绘制热力图
        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                if i < len(self.data) and j < len(self.data.iloc[i]):
                    value = self.data.iloc[i, j]
                    color = self.get_color(value)
                    
                    x = margin_left + j * cell_width
                    y = margin_top + i * cell_height
                    
                    # 填充单元格
                    painter.fillRect(QRectF(x, y, cell_width, cell_height), color)
                    
                    # 绘制单元格边框
                    painter.setPen(QPen(QColor(30, 35, 55), 1))
                    painter.drawRect(QRectF(x, y, cell_width, cell_height))
                    
                    # 绘制数值
                    painter.setFont(QFont("微软雅黑", 8))
                    painter.setPen(QColor(240, 240, 250))
                    text_rect = QRectF(x, y, cell_width, cell_height)
                    painter.drawText(text_rect, Qt.AlignCenter, f"{value:.2f}")


class QuantumRadarChart(QWidget):
    """量子风格雷达图"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 300)
        self.categories = []
        self.values = []
        self.title = ""
        
        # 设置字体
        self.title_font = QFont("微软雅黑", 12, QFont.Bold)
        self.label_font = QFont("微软雅黑", 9)
        
        # 添加阴影效果
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 100))
        shadow.setOffset(0, 5)
        self.setGraphicsEffect(shadow)
    
    def set_data(self, categories, values, title="特征雷达图"):
        """设置雷达图数据"""
        self.categories = categories
        self.values = values
        self.title = title
        self.update()
    
    def paintEvent(self, event):
        """绘制雷达图"""
        if not self.categories or not self.values:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制背景
        painter.fillRect(self.rect(), QColor(20, 25, 45, 200))
        
        # 绘制标题
        painter.setFont(self.title_font)
        painter.setPen(QColor(160, 190, 255))
        title_rect = QRectF(0, 10, self.width(), 25)
        painter.drawText(title_rect, Qt.AlignCenter, self.title)
        
        # 计算雷达图区域
        margin = 40
        center_x = self.width() / 2
        center_y = self.height() / 2
        radius = min(self.width(), self.height()) / 2 - margin
        
        # 确保数据有效
        if len(self.categories) <= 2 or len(self.values) != len(self.categories):
            return
        
        # 计算每个类别的角度
        angle_step = 2 * np.pi / len(self.categories)
        
        # 绘制背景网格和刻度线
        painter.setPen(QPen(QColor(60, 70, 90), 1, Qt.DashLine))
        for i in range(1, 6):  # 5个环
            r = radius * i / 5
            painter.drawEllipse(QRectF(center_x - r, center_y - r, 2 * r, 2 * r))
        
        # 绘制径向线
        for i in range(len(self.categories)):
            angle = i * angle_step - np.pi / 2  # 从顶部开始
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            painter.drawLine(QPointF(center_x, center_y), QPointF(float(x), float(y)))
            
            # 绘制类别标签
            label_x = center_x + (radius + 15) * np.cos(angle)
            label_y = center_y + (radius + 15) * np.sin(angle)
            painter.setFont(self.label_font)
            painter.setPen(QColor(180, 190, 210))
            
            text_rect = QRectF(label_x - 50, label_y - 10, 100, 20)
            painter.drawText(text_rect, Qt.AlignCenter, self.categories[i])
        
        # 绘制雷达多边形
        path = QPainterPath()
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0, QColor(0, 120, 240, 150))
        gradient.setColorAt(1, QColor(120, 0, 240, 150))
        
        for i in range(len(self.values)):
            angle = i * angle_step - np.pi / 2  # 从顶部开始
            value = max(0, min(1, self.values[i]))  # 确保值在0-1之间
            x = center_x + radius * value * np.cos(angle)
            y = center_y + radius * value * np.sin(angle)
            
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        
        path.closeSubpath()
        painter.setBrush(gradient)
        painter.setPen(QPen(QColor(100, 200, 255, 200), 2))
        painter.drawPath(path)
        
        # 绘制数据点
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 255, 200))
        for i in range(len(self.values)):
            angle = i * angle_step - np.pi / 2  # 从顶部开始
            value = max(0, min(1, self.values[i]))  # 确保值在0-1之间
            x = center_x + radius * value * np.cos(angle)
            y = center_y + radius * value * np.sin(angle)
            painter.drawEllipse(QRectF(x - 4, y - 4, 8, 8))


class QuantumInfoPanel(QFrame):
    """量子风格信息面板"""
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background-color: rgba(25, 30, 50, 180);
                border-radius: 8px;
                border: 1px solid #303060;
            }
        """)
        
        # 添加阴影效果
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(0, 3)
        self.setGraphicsEffect(shadow)
        
        # 创建布局
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(15, 10, 15, 10)
        
        # 标题
        if title:
            self.title_label = QLabel(title)
            self.title_label.setStyleSheet("""
                font-family: '微软雅黑';
                font-size: 16px;
                font-weight: bold;
                color: #60a0e0;
                padding: 5px;
            """)
            self.layout.addWidget(self.title_label)
    
    def add_item(self, text, color="#b0c0e0", indent=0):
        """添加文本项"""
        label = QLabel(text)
        label.setWordWrap(True)
        indent_str = "&nbsp;" * (4 * indent)
        label.setStyleSheet(f"""
            font-family: '微软雅黑';
            font-size: 14px;
            color: {color};
            padding: 3px 10px;
        """)
        if indent > 0:
            label.setText(f"{indent_str}{text}")
        self.layout.addWidget(label)
        return label
    
    def add_separator(self):
        """添加分隔线"""
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("background-color: #303050;")
        separator.setFixedHeight(1)
        self.layout.addWidget(separator)
    
    def add_spacer(self):
        """添加弹性空间"""
        self.layout.addStretch() 