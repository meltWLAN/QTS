#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
维度可视化面板 - 量子维度的3D图形表示
"""

import logging
import random
import io
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QFrame, QSlider, QCheckBox,
                             QSplitter, QComboBox)
from PyQt5.QtCore import Qt, pyqtSlot, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用Agg后端，避免需要图形界面
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D

logger = logging.getLogger('quantum_desktop.ui.panels.dimension_visualizer_panel')

class DimensionVisualizerPanel(QFrame):
    """维度可视化面板 - 量子维度的图形化表示"""
    
    def __init__(self, system_manager, parent=None):
        super().__init__(parent)
        self.system_manager = system_manager
        
        # 可视化属性
        self.dimensions = 21
        self.data_points = 200
        self.rotation_speed = 0.5
        self.auto_rotate = True
        self.current_angle = 0
        self.visualization_type = "散点图"  # 可选：散点图、线图、表面图
        
        # 定时器，用于自动旋转和更新可视化
        self.rotation_timer = QTimer(self)
        self.rotation_timer.timeout.connect(self.rotate_visualization)
        self.rotation_timer.start(50)  # 每50毫秒更新一次
        
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_visualization)
        self.update_timer.start(5000)  # 每5秒更新一次数据
        
        # 初始化UI
        self._init_ui()
        
        logger.info("维度可视化面板初始化完成")
        
    def _init_ui(self):
        """初始化UI"""
        # 设置面板样式
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setStyleSheet("""
            QFrame {
                background-color: #1A1E2E;
                border-radius: 10px;
                border: 1px solid #2A3142;
            }
            QLabel {
                color: #E0E0E0;
                font-size: 14px;
            }
            QPushButton {
                background-color: #3A4254;
                color: #E0E0E0;
                border-radius: 5px;
                padding: 5px 10px;
                border: none;
            }
            QPushButton:hover {
                background-color: #4A5264;
            }
            QPushButton:pressed {
                background-color: #2A3244;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #2A3142;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #6C87EF;
                border: 1px solid #5C77DF;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QComboBox {
                background-color: #3A4254;
                color: #E0E0E0;
                border-radius: 5px;
                padding: 5px 10px;
                border: 1px solid #4A5264;
            }
            QCheckBox {
                color: #E0E0E0;
            }
        """)
        
        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)
        
        # 标题
        title_label = QLabel("量子维度可视化")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #6C87EF;")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 信息行
        info_layout = QHBoxLayout()
        self.dimension_label = QLabel(f"活跃量子维度: {self.dimensions}/21")
        info_layout.addWidget(self.dimension_label)
        
        info_layout.addStretch()
        
        point_label = QLabel(f"量子点数量: {self.data_points}")
        info_layout.addWidget(point_label)
        
        main_layout.addLayout(info_layout)
        
        # 可视化区域
        self.visualization_label = QLabel()
        self.visualization_label.setMinimumSize(400, 300)
        self.visualization_label.setAlignment(Qt.AlignCenter)
        self.visualization_label.setStyleSheet("background-color: #1A1E2E; border: 1px solid #2A3142;")
        main_layout.addWidget(self.visualization_label, 1)
        
        # 控制面板
        controls_layout = QHBoxLayout()
        
        # 维度滑块
        dim_layout = QVBoxLayout()
        dim_label = QLabel("维度")
        dim_layout.addWidget(dim_label)
        
        self.dim_slider = QSlider(Qt.Horizontal)
        self.dim_slider.setMinimum(3)
        self.dim_slider.setMaximum(21)
        self.dim_slider.setValue(self.dimensions)
        self.dim_slider.setTickPosition(QSlider.TicksBelow)
        self.dim_slider.setTickInterval(2)
        self.dim_slider.valueChanged.connect(self.on_dim_changed)
        dim_layout.addWidget(self.dim_slider)
        
        controls_layout.addLayout(dim_layout)
        
        # 点数滑块
        points_layout = QVBoxLayout()
        points_label = QLabel("数据点")
        points_layout.addWidget(points_label)
        
        self.points_slider = QSlider(Qt.Horizontal)
        self.points_slider.setMinimum(50)
        self.points_slider.setMaximum(500)
        self.points_slider.setValue(self.data_points)
        self.points_slider.setTickPosition(QSlider.TicksBelow)
        self.points_slider.setTickInterval(50)
        self.points_slider.valueChanged.connect(self.on_points_changed)
        points_layout.addWidget(self.points_slider)
        
        controls_layout.addLayout(points_layout)
        
        # 可视化类型
        type_layout = QVBoxLayout()
        type_label = QLabel("可视化类型")
        type_layout.addWidget(type_label)
        
        self.type_combo = QComboBox()
        self.type_combo.addItems(["散点图", "线图", "表面图", "热图"])
        self.type_combo.currentTextChanged.connect(self.on_type_changed)
        type_layout.addWidget(self.type_combo)
        
        controls_layout.addLayout(type_layout)
        
        # 自动旋转
        rotate_layout = QVBoxLayout()
        self.rotate_check = QCheckBox("自动旋转")
        self.rotate_check.setChecked(self.auto_rotate)
        self.rotate_check.stateChanged.connect(self.on_auto_rotate_changed)
        rotate_layout.addWidget(self.rotate_check)
        
        # 旋转速度滑块
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(10)
        self.speed_slider.setValue(int(self.rotation_speed * 10))
        self.speed_slider.valueChanged.connect(self.on_speed_changed)
        rotate_layout.addWidget(self.speed_slider)
        
        controls_layout.addLayout(rotate_layout)
        
        # 更新按钮
        self.update_btn = QPushButton("刷新可视化")
        self.update_btn.clicked.connect(self.update_visualization)
        controls_layout.addWidget(self.update_btn)
        
        main_layout.addLayout(controls_layout)
        
        # 初始可视化
        self.update_visualization()
    
    def on_dim_changed(self, value):
        """维度滑块值改变"""
        self.dimensions = value
        self.dimension_label.setText(f"活跃量子维度: {self.dimensions}/21")
        self.update_visualization()
    
    def on_points_changed(self, value):
        """数据点滑块值改变"""
        self.data_points = value
        self.update_visualization()
    
    def on_type_changed(self, value):
        """可视化类型改变"""
        self.visualization_type = value
        self.update_visualization()
    
    def on_auto_rotate_changed(self, state):
        """自动旋转选项改变"""
        self.auto_rotate = (state == Qt.Checked)
        if self.auto_rotate:
            self.rotation_timer.start()
        else:
            self.rotation_timer.stop()
    
    def on_speed_changed(self, value):
        """旋转速度改变"""
        self.rotation_speed = value / 10.0
    
    def rotate_visualization(self):
        """旋转3D可视化"""
        if hasattr(self, 'ax') and self.ax:
            try:
                self.current_angle += self.rotation_speed
                if self.current_angle >= 360:
                    self.current_angle = 0
                    
                # 设置新视角
                self.ax.view_init(30, self.current_angle)
                
                # 重新绘制
                self.canvas.draw()
                
                # 转换为QImage并显示
                buf = io.BytesIO()
                self.canvas.print_png(buf)
                buf.seek(0)
                
                image = QImage.fromData(buf.getvalue())
                pixmap = QPixmap.fromImage(image)
                
                # 更新UI
                if hasattr(self, 'visualization_label') and self.visualization_label:
                    self.visualization_label.setPixmap(pixmap)
                    self.visualization_label.setScaledContents(True)
            except Exception as e:
                logger.error(f"旋转可视化时出错: {str(e)}")
    
    def update_visualization(self):
        """更新维度可视化"""
        try:
            # 生成量子维度数据
            dimensions = self.dimensions
            data_points = self.data_points
            
            # 生成量子点数据
            quantum_points = []
            for i in range(data_points):
                # 创建一个N维的量子点，每个维度的值在[-1,1]之间
                point = [random.uniform(-1, 1) for _ in range(dimensions)]
                
                # 添加一些类似于超空间的结构
                for d in range(2, dimensions):
                    # 随机添加维度间的关联，创造量子纠缠效果
                    if random.random() < 0.3:
                        # 维度之间的非线性关系
                        point[d] = point[d-1] * point[d-2] * random.uniform(0.5, 1.5)
                
                quantum_points.append(point)
            
            # t-SNE降维 (简化为基本的PCA)
            viz_points = []
            if len(quantum_points) > 0:
                # 简化的降维实现
                xs = []
                ys = []
                zs = []
                
                for point in quantum_points:
                    # 使用简单的投影映射到3D空间
                    x = sum(point[0:dimensions:3]) / (dimensions/3 + 1)
                    y = sum(point[1:dimensions:3]) / (dimensions/3 + 1) 
                    z = sum(point[2:dimensions:3]) / (dimensions/3 + 1)
                    
                    # 添加一些非线性关系，使可视化更有趣
                    x = x * 10 
                    y = y * 10
                    z = z * 10 + x*y/20
                    
                    xs.append(x)
                    ys.append(y)
                    zs.append(z)
                    
                    viz_points.append((x, y, z))
            else:
                # 如果没有点，创建一些随机点
                for _ in range(data_points):
                    x = random.uniform(-10, 10)
                    y = random.uniform(-10, 10)
                    z = random.uniform(-10, 10)
                    # 为了形成更有趣的结构，添加一些非线性关系
                    if random.random() < 0.7:
                        z = x*y/10 + z*0.2
                    viz_points.append((x, y, z))
                
                xs = [p[0] for p in viz_points]
                ys = [p[1] for p in viz_points]
                zs = [p[2] for p in viz_points]
            
            # 更新3D图表
            figure = Figure(figsize=(8, 6), dpi=100)
            self.canvas = FigureCanvas(figure)
            self.ax = figure.add_subplot(111, projection='3d')
            
            # 根据可视化类型创建不同的图
            if self.visualization_type == "散点图":
                scatter = self.ax.scatter(xs, ys, zs, c=range(len(xs)), cmap='plasma', 
                                     marker='o', s=20, alpha=0.6)
                                     
                # 添加一些连线，表示维度间的关联
                for i in range(0, len(xs)-1, 10):
                    if random.random() < 0.3:  # 只连接30%的点，避免过度拥挤
                        self.ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]], [zs[i], zs[i+1]], 
                                color='cyan', alpha=0.3, linewidth=0.5)
            
            elif self.visualization_type == "线图":
                # 创建线图，连接所有点
                self.ax.plot(xs, ys, zs, color='cyan', alpha=0.6, linewidth=1)
                self.ax.scatter(xs, ys, zs, c=range(len(xs)), cmap='plasma', 
                           marker='o', s=10, alpha=0.4)
            
            elif self.visualization_type == "表面图":
                # 创建规则网格
                grid_size = int(np.sqrt(min(data_points, 400)))  # 限制网格大小以避免性能问题
                x = np.linspace(-10, 10, grid_size)
                y = np.linspace(-10, 10, grid_size)
                X, Y = np.meshgrid(x, y)
                Z = np.sin(np.sqrt(X**2 + Y**2)) + X*Y/10
                
                # 创建表面图
                surf = self.ax.plot_surface(X, Y, Z, cmap='plasma', alpha=0.8,
                                      linewidth=0, antialiased=True)
            
            elif self.visualization_type == "热图":
                # 创建连续的3D散点热图
                cm = self.ax.scatter(xs, ys, zs, c=zs, cmap='inferno', 
                              marker='o', s=30, alpha=0.7)
                figure.colorbar(cm, ax=self.ax, shrink=0.5, aspect=5)
            
            # 设置图表样式
            self.ax.set_facecolor('#1A1E2E')  # 深色背景
            figure.patch.set_facecolor('#1A1E2E')
            self.ax.set_title(f"{dimensions}维量子空间投影", color='white')
            
            # 是否显示坐标轴
            if self.visualization_type == "热图":
                self.ax.set_xlabel('X', color='white')
                self.ax.set_ylabel('Y', color='white')
                self.ax.set_zlabel('Z', color='white')
                # 设置刻度颜色
                self.ax.tick_params(axis='x', colors='white')
                self.ax.tick_params(axis='y', colors='white')
                self.ax.tick_params(axis='z', colors='white')
            else:
                self.ax.set_axis_off()  # 隐藏坐标轴
            
            # 添加维度标签
            self.ax.text2D(0.02, 0.98, f"维度: {dimensions}/21", transform=self.ax.transAxes, 
                      color='white', fontsize=8)
            self.ax.text2D(0.02, 0.93, f"活跃量子点: {data_points}", transform=self.ax.transAxes, 
                      color='white', fontsize=8)
            
            # 设置初始视角
            self.ax.view_init(30, self.current_angle)
            
            # 设置轴范围 - 确保所有点都可见
            x_range = max(abs(min(xs)), abs(max(xs))) * 1.1
            y_range = max(abs(min(ys)), abs(max(ys))) * 1.1
            z_range = max(abs(min(zs)), abs(max(zs))) * 1.1
            
            max_range = max(x_range, y_range, z_range)
            self.ax.set_xlim(-max_range, max_range)
            self.ax.set_ylim(-max_range, max_range)
            self.ax.set_zlim(-max_range, max_range)
            
            # 渲染图表
            self.canvas.draw()
            
            # 转换为QImage并显示
            buf = io.BytesIO()
            self.canvas.print_png(buf)
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
            logger.error(f"更新维度可视化时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            self.visualization_label.setText(f"可视化错误: {str(e)}") 