#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - 高级启动画面
提供更现代化的启动体验
"""

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QProgressBar, 
    QGraphicsDropShadowEffect, QHBoxLayout
)
from PyQt5.QtCore import Qt, QTimer, QSize, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont, QColor, QIcon, QPixmap, QPainter, QLinearGradient, QPen
import sys
import random

class QuantumParticle:
    """量子粒子动画元素"""
    def __init__(self, x, y, speed=0.5, size=4):
        self.x = x
        self.y = y
        self.size = size
        self.speed = speed
        self.angle = random.uniform(0, 360)
        self.color = QColor(
            random.randint(30, 255), 
            random.randint(30, 255), 
            random.randint(90, 255),
            150
        )
        self.connections = []  # 与其他粒子的连接
        
    def update(self, width, height):
        """更新粒子位置"""
        dx = self.speed * 2 * (0.5 - random.random())
        dy = self.speed * 2 * (0.5 - random.random())
        
        self.x += dx
        self.y += dy
        
        # 确保粒子不会离开边界
        border = 20
        self.x = max(border, min(self.x, width - border))
        self.y = max(border, min(self.y, height - border))
        
        # 随机改变一点颜色
        r, g, b, a = self.color.getRgb()
        r = max(30, min(255, r + random.randint(-5, 5)))
        g = max(30, min(255, g + random.randint(-5, 5)))
        b = max(90, min(255, b + random.randint(-5, 5)))
        self.color = QColor(r, g, b, a)

class SuperGodSplashScreen(QWidget):
    """超神启动画面"""
    
    # 自定义信号
    progressChanged = pyqtSignal(int, str)
    finished = pyqtSignal()
    
    def __init__(self):
        super().__init__(None)
        
        # 窗口设置
        self.setWindowTitle("启动中")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(600, 400)
        
        # 初始化粒子
        self.particles = []
        for _ in range(30):
            x = random.uniform(50, 550)
            y = random.uniform(50, 350)
            speed = random.uniform(0.2, 0.8)
            size = random.uniform(2, 6)
            self.particles.append(QuantumParticle(x, y, speed, size))
        
        # 设置UI
        self._setup_ui()
        
        # 动画定时器
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self._update_animation)
        self.animation_timer.start(30)  # 30ms刷新一次，约33fps
        
        # 连接信号
        self.progressChanged.connect(self._update_progress)
        
        # 居中显示
        self._center_on_screen()
    
    def _setup_ui(self):
        """设置UI"""
        # 主布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 顶部空间
        layout.addSpacing(20)
        
        # 标题
        title_label = QLabel("超神量子共生网络交易系统")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 20, QFont.Bold))
        title_label.setStyleSheet("color: white;")
        
        # 添加阴影效果
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setColor(QColor(0, 0, 0, 180))
        shadow.setOffset(0, 0)
        title_label.setGraphicsEffect(shadow)
        
        layout.addWidget(title_label)
        
        # 副标题
        subtitle_label = QLabel("SUPER GOD-LEVEL QUANTUM SYMBIOTIC SYSTEM")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setFont(QFont("Arial", 10))
        subtitle_label.setStyleSheet("color: rgba(255, 255, 255, 180);")
        layout.addWidget(subtitle_label)
        
        layout.addSpacing(40)
        
        # 版本信息
        version_label = QLabel("v0.2.0")
        version_label.setAlignment(Qt.AlignCenter)
        version_label.setStyleSheet("color: rgba(255, 255, 255, 150);")
        layout.addWidget(version_label)
        
        layout.addStretch()
        
        # 进度文本
        self.status_label = QLabel("正在初始化量子网络...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: white;")
        layout.addWidget(self.status_label)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: rgba(255, 255, 255, 50);
                border-radius: 2px;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3D88F4, stop:1 #5D4CEC);
                border-radius: 2px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # 底部版权
        copyright_label = QLabel("© 2025 量子共生网络研发团队")
        copyright_label.setAlignment(Qt.AlignCenter)
        copyright_label.setStyleSheet("color: rgba(255, 255, 255, 120);")
        layout.addWidget(copyright_label)
    
    def paintEvent(self, event):
        """绘制事件"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 绘制背景
        rect = self.rect()
        
        # 创建渐变背景
        gradient = QLinearGradient(0, 0, rect.width(), rect.height())
        gradient.setColorAt(0, QColor(28, 35, 65))
        gradient.setColorAt(1, QColor(12, 15, 35))
        painter.fillRect(rect, gradient)
        
        # 绘制边框
        pen = QPen(QColor(80, 120, 220, 60))
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 10, 10)
        
        # 绘制粒子
        for particle in self.particles:
            # 绘制连接线
            for other in particle.connections:
                # 计算距离，距离越远线越淡
                dx = particle.x - other.x
                dy = particle.y - other.y
                distance = (dx**2 + dy**2)**0.5
                max_distance = 100  # 超过此距离不绘制连接线
                
                if distance < max_distance:
                    alpha = int(150 * (1 - distance / max_distance))
                    color = QColor(particle.color.red(), particle.color.green(), 
                                particle.color.blue(), alpha)
                    pen = QPen(color)
                    pen.setWidth(1)
                    painter.setPen(pen)
                    painter.drawLine(int(particle.x), int(particle.y), 
                                    int(other.x), int(other.y))
            
            # 绘制粒子
            painter.setPen(Qt.NoPen)
            painter.setBrush(particle.color)
            painter.drawEllipse(
                int(particle.x - particle.size/2), 
                int(particle.y - particle.size/2), 
                int(particle.size), 
                int(particle.size)
            )
    
    def _update_animation(self):
        """更新粒子动画"""
        width = self.width()
        height = self.height()
        
        # 更新所有粒子位置
        for particle in self.particles:
            particle.update(width, height)
            # 清除之前的连接
            particle.connections = []
        
        # 创建粒子间的连接
        for i, particle in enumerate(self.particles):
            for j in range(i+1, len(self.particles)):
                other = self.particles[j]
                dx = particle.x - other.x
                dy = particle.y - other.y
                distance = (dx**2 + dy**2)**0.5
                
                if distance < 100:  # 距离小于100时创建连接
                    particle.connections.append(other)
                    other.connections.append(particle)
        
        # 更新UI
        self.update()
    
    def _center_on_screen(self):
        """将窗口居中显示"""
        screen_geometry = QApplication.desktop().screenGeometry()
        x = (screen_geometry.width() - self.width()) // 2
        y = (screen_geometry.height() - self.height()) // 2
        self.move(x, y)
    
    def _update_progress(self, value, message):
        """更新进度和状态消息"""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
        
        if value >= 100:
            # 短暂延迟后关闭
            QTimer.singleShot(500, self._finish)
    
    def _finish(self):
        """完成动画，发出完成信号"""
        # 停止动画
        self.animation_timer.stop()
        
        # 发出完成信号
        self.finished.emit()
        
        # 关闭窗口
        self.close()

def show_splash_screen(app):
    """显示启动画面并返回实例"""
    splash = SuperGodSplashScreen()
    splash.show()
    app.processEvents()
    return splash

if __name__ == "__main__":
    # 测试代码
    app = QApplication(sys.argv)
    splash = show_splash_screen(app)
    
    # 模拟加载过程
    def simulate_loading():
        current = 0
        stages = [
            "正在初始化量子网络...",
            "加载市场数据...",
            "校准量子共振频率...",
            "同步交易引擎...",
            "激活AI预测模块..."
        ]
        
        for i, stage in enumerate(stages):
            current = (i+1) * 100 // len(stages)
            splash.progressChanged.emit(current, stage)
            QTimer.singleShot((i+1)*1000, lambda: None)  # 延迟
    
    QTimer.singleShot(100, simulate_loading)
    
    # 主窗口
    main_window = QWidget()
    main_window.setWindowTitle("主窗口")
    main_window.resize(800, 600)
    
    def show_main():
        main_window.show()
    
    splash.finished.connect(show_main)
    
    sys.exit(app.exec_()) 