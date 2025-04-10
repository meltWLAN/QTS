#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - 宇宙能量可视化模块
展示超级能量流动和市场维度共振
"""

import numpy as np
import random
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QRectF, QPointF
from PyQt5.QtGui import QPainter, QColor, QRadialGradient, QPen, QBrush, QPainterPath
import math

class CosmicEnergyParticle:
    """宇宙能量粒子"""
    
    def __init__(self, x, y, size, speed, color, lifespan=100):
        self.x = x
        self.y = y
        self.size = size
        self.original_size = size
        self.speed = speed
        self.color = color
        self.angle = random.uniform(0, 2 * math.pi)
        self.lifespan = lifespan
        self.age = 0
        self.alive = True
    
    def update(self):
        """更新粒子状态"""
        self.age += 1
        
        # 移动粒子
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed
        
        # 随机微调角度，形成流动感
        self.angle += random.uniform(-0.1, 0.1)
        
        # 随着年龄增长，粒子变小
        life_ratio = 1 - (self.age / self.lifespan)
        self.size = self.original_size * life_ratio
        
        # 检查生命周期
        if self.age >= self.lifespan:
            self.alive = False

class EnergyNode:
    """能量节点"""
    
    def __init__(self, x, y, size, energy, name=""):
        self.x = x
        self.y = y
        self.size = size
        self.energy = energy  # 0-1
        self.name = name
        self.pulse_phase = random.uniform(0, 2 * math.pi)
        self.connections = []  # 连接到其他节点
    
    def update(self):
        """更新节点状态"""
        self.pulse_phase += 0.05
        if self.pulse_phase > 2 * math.pi:
            self.pulse_phase -= 2 * math.pi
    
    def get_pulse_size(self):
        """获取脉冲大小"""
        pulse = math.sin(self.pulse_phase) * 0.2 + 0.8
        return self.size * pulse * (0.8 + self.energy * 0.4)
    
    def connect_to(self, other_node, strength=0.5):
        """连接到另一个节点"""
        if other_node not in self.connections:
            self.connections.append((other_node, strength))

class CosmicEnergyVisualization(QWidget):
    """宇宙能量可视化组件"""
    
    energy_updated = pyqtSignal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 200)
        
        # 初始化能量粒子
        self.particles = []
        self.max_particles = 150
        
        # 初始化能量节点
        self.nodes = []
        self.setup_nodes()
        
        # 能量水平
        self.energy_level = 0.5
        self.target_energy = 0.5
        
        # 设置定时器
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_visualization)
        self.update_timer.start(30)  # 30ms更新一次
        
        # 能量浮动定时器
        self.energy_timer = QTimer(self)
        self.energy_timer.timeout.connect(self.fluctuate_energy)
        self.energy_timer.start(3000)  # 3秒更新一次能量目标
    
    def setup_nodes(self):
        """设置能量节点"""
        # 创建主要节点
        self.nodes = [
            EnergyNode(0.5, 0.5, 40, 0.9, "中央能量"),
            EnergyNode(0.3, 0.3, 25, 0.7, "量子意识"),
            EnergyNode(0.7, 0.3, 25, 0.8, "市场感知"),
            EnergyNode(0.3, 0.7, 25, 0.6, "交易引擎"),
            EnergyNode(0.7, 0.7, 25, 0.75, "预测单元")
        ]
        
        # 连接节点
        self.nodes[0].connect_to(self.nodes[1], 0.9)
        self.nodes[0].connect_to(self.nodes[2], 0.8)
        self.nodes[0].connect_to(self.nodes[3], 0.7)
        self.nodes[0].connect_to(self.nodes[4], 0.85)
        self.nodes[1].connect_to(self.nodes[2], 0.6)
        self.nodes[2].connect_to(self.nodes[4], 0.7)
        self.nodes[3].connect_to(self.nodes[4], 0.5)
        self.nodes[1].connect_to(self.nodes[3], 0.4)
    
    def update_visualization(self):
        """更新可视化状态"""
        # 更新节点
        for node in self.nodes:
            node.update()
        
        # 平滑过渡到目标能量
        self.energy_level += (self.target_energy - self.energy_level) * 0.05
        
        # 根据能量水平添加新粒子
        spawn_chance = self.energy_level * 0.5  # 0-0.5的概率
        if random.random() < spawn_chance and len(self.particles) < self.max_particles:
            self.add_particle()
        
        # 更新粒子
        for particle in self.particles[:]:
            particle.update()
            if not particle.alive:
                self.particles.remove(particle)
        
        # 触发重绘
        self.update()
    
    def add_particle(self):
        """添加能量粒子"""
        # 随机选择一个节点作为起点
        node = random.choice(self.nodes)
        
        # 获取相对于widget的位置
        x = node.x * self.width()
        y = node.y * self.height()
        
        # 随机颜色
        colors = [
            QColor(30, 160, 255, 180),  # 蓝色
            QColor(0, 200, 200, 180),   # 青色
            QColor(140, 80, 255, 180),  # 紫色
            QColor(0, 220, 180, 180),   # 绿松石色
            QColor(100, 160, 255, 180)  # 浅蓝色
        ]
        
        # 创建粒子
        size = random.uniform(3, 6) * (0.5 + self.energy_level)
        speed = random.uniform(1, 3) * (0.5 + self.energy_level)
        color = random.choice(colors)
        lifespan = random.randint(50, 100)
        
        self.particles.append(CosmicEnergyParticle(x, y, size, speed, color, lifespan))
    
    def fluctuate_energy(self):
        """能量水平浮动"""
        # 随机调整目标能量
        self.target_energy = min(1.0, max(0.3, self.target_energy + random.uniform(-0.2, 0.2)))
        self.energy_updated.emit(self.energy_level)
    
    def set_energy_level(self, level):
        """设置能量水平"""
        self.target_energy = max(0.0, min(1.0, level))
    
    def paintEvent(self, event):
        """绘制事件"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制背景
        painter.fillRect(self.rect(), QColor(5, 5, 16))
        
        # 绘制连接
        self.draw_connections(painter)
        
        # 绘制能量粒子
        self.draw_particles(painter)
        
        # 绘制节点
        self.draw_nodes(painter)
    
    def draw_connections(self, painter):
        """绘制节点之间的连接"""
        for node in self.nodes:
            x1 = node.x * self.width()
            y1 = node.y * self.height()
            
            for connection in node.connections:
                other_node, strength = connection
                x2 = other_node.x * self.width()
                y2 = other_node.y * self.height()
                
                # 设置透明度基于强度和能量
                alpha = int(100 * strength * (0.3 + self.energy_level * 0.7))
                color = QColor(100, 180, 255, alpha)
                
                # 设置线宽基于强度
                width = 1 + strength * 2
                
                # 绘制连接线
                pen = QPen(color, width)
                painter.setPen(pen)
                
                # 创建路径
                path = QPainterPath()
                path.moveTo(x1, y1)
                
                # 控制点，使连接呈弧形
                cx = (x1 + x2) / 2 - (y2 - y1) * 0.2
                cy = (y1 + y2) / 2 + (x2 - x1) * 0.2
                
                path.quadTo(cx, cy, x2, y2)
                painter.drawPath(path)
                
                # 在连接线上绘制能量流动点
                if random.random() < 0.3 * self.energy_level:
                    # 随机位置
                    t = random.uniform(0.2, 0.8)
                    px = x1 + t * (x2 - x1) + (y1 - y2) * 0.1 * math.sin(t * math.pi)
                    py = y1 + t * (y2 - y1) + (x2 - x1) * 0.1 * math.sin(t * math.pi)
                    
                    # 绘制能量点
                    pulse_size = 3 + 2 * math.sin(self.nodes[0].pulse_phase * 2 + t * 5)
                    radial = QRadialGradient(px, py, pulse_size * 2)
                    radial.setColorAt(0, QColor(200, 230, 255, 180))
                    radial.setColorAt(1, QColor(100, 180, 255, 0))
                    
                    painter.setBrush(QBrush(radial))
                    painter.setPen(Qt.NoPen)
                    painter.drawEllipse(QPointF(px, py), pulse_size, pulse_size)
    
    def draw_nodes(self, painter):
        """绘制能量节点"""
        for node in self.nodes:
            x = node.x * self.width()
            y = node.y * self.height()
            
            # 脉冲大小
            pulse_size = node.get_pulse_size()
            
            # 绘制光晕
            radial = QRadialGradient(x, y, pulse_size * 1.5)
            glow_color = QColor(100, 180, 255, 50)
            radial.setColorAt(0, glow_color)
            radial.setColorAt(1, QColor(100, 180, 255, 0))
            
            painter.setBrush(QBrush(radial))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(x, y), pulse_size * 1.5, pulse_size * 1.5)
            
            # 绘制核心
            core_radial = QRadialGradient(x, y, pulse_size)
            energy_factor = 0.5 + node.energy * 0.5
            core_color = QColor(
                int(100 * energy_factor), 
                int(180 * energy_factor), 
                int(255 * energy_factor), 
                200
            )
            
            core_radial.setColorAt(0, core_color)
            core_radial.setColorAt(0.7, QColor(
                int(60 * energy_factor), 
                int(120 * energy_factor), 
                int(200 * energy_factor), 
                150
            ))
            core_radial.setColorAt(1, QColor(
                int(30 * energy_factor), 
                int(90 * energy_factor), 
                int(150 * energy_factor), 
                100
            ))
            
            painter.setBrush(QBrush(core_radial))
            painter.drawEllipse(QPointF(x, y), pulse_size, pulse_size)
            
            # 绘制节点名称
            if node.name:
                painter.setPen(QColor(200, 230, 255))
                painter.drawText(
                    QRectF(x - 50, y + pulse_size + 5, 100, 20), 
                    Qt.AlignHCenter, 
                    node.name
                )
    
    def draw_particles(self, painter):
        """绘制能量粒子"""
        for particle in self.particles:
            # 设置画刷
            alpha = int(150 * (1 - particle.age / particle.lifespan))
            color = QColor(
                particle.color.red(),
                particle.color.green(),
                particle.color.blue(),
                alpha
            )
            
            # 创建径向渐变
            radial = QRadialGradient(particle.x, particle.y, particle.size)
            radial.setColorAt(0, color)
            radial.setColorAt(1, QColor(color.red(), color.green(), color.blue(), 0))
            
            painter.setBrush(QBrush(radial))
            painter.setPen(Qt.NoPen)
            
            # 绘制粒子
            painter.drawEllipse(
                QPointF(particle.x, particle.y), 
                particle.size, 
                particle.size
            ) 