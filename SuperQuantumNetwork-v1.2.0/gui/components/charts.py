#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - 图表组件
提供高级图表绘制功能
"""

import logging
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

class PieChart(QWidget):
    """饼图组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger("PieChart")
        
        # 初始化数据
        self.values = []
        self.labels = []
        self.colors = []
        
        # 创建matplotlib图形
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        
        # 创建布局
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        # 设置尺寸策略
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()
        
        # 绘制空饼图
        self._draw_pie_chart()
    
    def update_data(self, values, labels, colors=None):
        """更新数据
        
        Args:
            values: 数值列表
            labels: 标签列表
            colors: 颜色列表
        """
        self.values = values
        self.labels = labels
        
        if colors:
            self.colors = colors
        else:
            # 生成随机颜色
            self.colors = [QColor(
                np.random.randint(0, 256), 
                np.random.randint(0, 256), 
                np.random.randint(0, 256)
            ) for _ in range(len(values))]
        
        # 重新绘制饼图
        self._draw_pie_chart()
    
    def _draw_pie_chart(self):
        """绘制饼图"""
        try:
            # 清空图形
            self.figure.clear()
            
            # 检查数据是否有效
            if not self.values or sum(self.values) == 0:
                self.canvas.draw()
                return
                
            # 创建子图
            ax = self.figure.add_subplot(111)
            
            # 准备颜色
            if not self.colors:
                # 使用默认颜色
                colors = None
            else:
                # 转换QColor为matplotlib颜色格式
                colors = [(c.red()/255, c.green()/255, c.blue()/255) for c in self.colors]
            
            # 绘制饼图
            wedges, texts, autotexts = ax.pie(
                self.values, 
                labels=self.labels, 
                autopct='%1.1f%%',
                startangle=90,
                shadow=False,
                colors=colors,
                wedgeprops={'edgecolor': 'w', 'linewidth': 1},
                textprops={'fontsize': 9}
            )
            
            # 设置标签字体
            for text in texts:
                text.set_fontsize(9)
            
            # 设置百分比字体
            for autotext in autotexts:
                autotext.set_fontsize(8)
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            # 设置标题
            ax.set_title("资产配置比例", fontsize=12)
            
            # 设置等比例
            ax.axis('equal')
            
            # 设置布局
            self.figure.tight_layout()
            
            # 绘制
            self.canvas.draw()
            
        except Exception as e:
            self.logger.error(f"绘制饼图失败: {str(e)}")

class LineChart(QWidget):
    """折线图组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger("LineChart")
        
        # 创建matplotlib图形
        self.figure = Figure(figsize=(8, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        
        # 创建布局
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        # 设置尺寸策略
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()
        
        # 绘制空折线图
        self._draw_empty_chart()
    
    def update_line_chart(self, x_data, y_data_list, labels=None, colors=None):
        """更新折线图数据
        
        Args:
            x_data: x轴数据
            y_data_list: y轴数据列表
            labels: 数据标签列表
            colors: 颜色列表
        """
        try:
            # 清空图形
            self.figure.clear()
            
            # 创建子图
            ax = self.figure.add_subplot(111)
            
            # 绘制每条折线
            for i, y_data in enumerate(y_data_list):
                # 获取标签
                label = labels[i] if labels and i < len(labels) else f"数据{i+1}"
                
                # 获取颜色
                color = colors[i] if colors and i < len(colors) else None
                
                # 绘制折线
                ax.plot(x_data, y_data, label=label, color=color)
            
            # 添加图例
            ax.legend(loc='upper left')
            
            # 设置网格
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # 设置标题
            ax.set_title("投资组合绩效", fontsize=12)
            
            # 格式化x轴
            if len(x_data) > 10:
                # 显示部分刻度
                step = max(1, len(x_data) // 10)
                ax.set_xticks(x_data[::step])
                ax.set_xticklabels([x.strftime('%m-%d') if hasattr(x, 'strftime') else str(x) for x in x_data[::step]], rotation=45)
            
            # 设置布局
            self.figure.tight_layout()
            
            # 绘制
            self.canvas.draw()
            
        except Exception as e:
            self.logger.error(f"更新折线图失败: {str(e)}")
            # 绘制空图表
            self._draw_empty_chart()
    
    def _draw_empty_chart(self):
        """绘制空折线图"""
        try:
            # 清空图形
            self.figure.clear()
            
            # 创建子图
            ax = self.figure.add_subplot(111)
            
            # 设置网格
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # 设置标题
            ax.set_title("投资组合绩效", fontsize=12)
            
            # 设置文本
            ax.text(0.5, 0.5, "无数据", ha='center', va='center', transform=ax.transAxes)
            
            # 设置布局
            self.figure.tight_layout()
            
            # 绘制
            self.canvas.draw()
            
        except Exception as e:
            self.logger.error(f"绘制空折线图失败: {str(e)}")

class BarChart(QWidget):
    """柱状图组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger("BarChart")
        
        # 创建matplotlib图形
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        
        # 创建布局
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        # 设置尺寸策略
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()
        
        # 绘制空柱状图
        self._draw_empty_chart()
    
    def update_bar_chart(self, categories, values, title="", color=None):
        """更新柱状图数据
        
        Args:
            categories: 类别列表
            values: 值列表
            title: 图表标题
            color: 柱状图颜色
        """
        try:
            # 清空图形
            self.figure.clear()
            
            # 创建子图
            ax = self.figure.add_subplot(111)
            
            # 绘制柱状图
            bars = ax.bar(categories, values, color=color)
            
            # 在柱状图上添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f"{height:.2f}", ha='center', va='bottom')
            
            # 设置网格
            ax.grid(True, linestyle='--', alpha=0.7, axis='y')
            
            # 设置标题
            ax.set_title(title, fontsize=12)
            
            # 旋转x轴标签
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # 设置布局
            self.figure.tight_layout()
            
            # 绘制
            self.canvas.draw()
            
        except Exception as e:
            self.logger.error(f"更新柱状图失败: {str(e)}")
            # 绘制空图表
            self._draw_empty_chart()
    
    def _draw_empty_chart(self):
        """绘制空柱状图"""
        try:
            # 清空图形
            self.figure.clear()
            
            # 创建子图
            ax = self.figure.add_subplot(111)
            
            # 设置网格
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # 设置标题
            ax.set_title("柱状图", fontsize=12)
            
            # 设置文本
            ax.text(0.5, 0.5, "无数据", ha='center', va='center', transform=ax.transAxes)
            
            # 设置布局
            self.figure.tight_layout()
            
            # 绘制
            self.canvas.draw()
            
        except Exception as e:
            self.logger.error(f"绘制空柱状图失败: {str(e)}")

class HeatmapChart(QWidget):
    """热力图组件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger("HeatmapChart")
        
        # 创建matplotlib图形
        self.figure = Figure(figsize=(6, 5), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        
        # 创建布局
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        # 设置尺寸策略
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()
        
        # 绘制空热力图
        self._draw_empty_chart()
    
    def update_heatmap(self, data, row_labels, col_labels, title="", cmap="coolwarm"):
        """更新热力图数据
        
        Args:
            data: 2D数据矩阵
            row_labels: 行标签
            col_labels: 列标签
            title: 图表标题
            cmap: 颜色映射
        """
        try:
            # 清空图形
            self.figure.clear()
            
            # 创建子图
            ax = self.figure.add_subplot(111)
            
            # 绘制热力图
            im = ax.imshow(data, cmap=cmap)
            
            # 添加颜色条
            cbar = self.figure.colorbar(im, ax=ax)
            cbar.set_label('值')
            
            # 设置刻度和标签
            ax.set_xticks(np.arange(len(col_labels)))
            ax.set_yticks(np.arange(len(row_labels)))
            ax.set_xticklabels(col_labels)
            ax.set_yticklabels(row_labels)
            
            # 旋转x轴标签
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # 添加文本标注
            for i in range(len(row_labels)):
                for j in range(len(col_labels)):
                    text = ax.text(j, i, f"{data[i, j]:.2f}",
                                  ha="center", va="center", color="white" if data[i, j] > 0.5 else "black")
            
            # 设置标题
            ax.set_title(title, fontsize=12)
            
            # 设置布局
            self.figure.tight_layout()
            
            # 绘制
            self.canvas.draw()
            
        except Exception as e:
            self.logger.error(f"更新热力图失败: {str(e)}")
            # 绘制空图表
            self._draw_empty_chart()
    
    def _draw_empty_chart(self):
        """绘制空热力图"""
        try:
            # 清空图形
            self.figure.clear()
            
            # 创建子图
            ax = self.figure.add_subplot(111)
            
            # 设置标题
            ax.set_title("热力图", fontsize=12)
            
            # 设置文本
            ax.text(0.5, 0.5, "无数据", ha='center', va='center', transform=ax.transAxes)
            
            # 设置布局
            self.figure.tight_layout()
            
            # 绘制
            self.canvas.draw()
            
        except Exception as e:
            self.logger.error(f"绘制空热力图失败: {str(e)}") 