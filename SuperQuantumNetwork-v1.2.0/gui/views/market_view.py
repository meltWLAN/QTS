#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - 市场视图
显示市场概览、指数行情、热门股票和推荐股票
"""

import os
import sys
import logging
from datetime import datetime
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, 
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter,
    QPushButton, QComboBox, QGroupBox, QGridLayout, QTabWidget,
    QProgressBar, QLineEdit, QToolButton, QCompleter, QAbstractItemView,
    QDialog, QListWidget, QMessageBox, QApplication
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QSize, pyqtSlot
from PyQt5.QtGui import QColor, QIcon, QPixmap, QFont, QPalette, QBrush
import numpy as np

try:
    import pyqtgraph as pg
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False
    logging.warning("PyQtGraph未安装，图表功能受限")

try:
    import qtawesome as qta
    QTA_AVAILABLE = True
except ImportError:
    QTA_AVAILABLE = False
    logging.warning("QtAwesome未安装，图标功能受限")


class MarketStatusWidget(QFrame):
    """市场状态小部件"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            MarketStatusWidget {
                background-color: rgba(20, 20, 40, 180);
                border-radius: 5px;
                padding: 5px;
            }
            QLabel {
                color: white;
            }
        """)
        
        # 创建布局
        layout = QGridLayout(self)
        
        # 市场状态标题
        self.title_label = QLabel("市场状态")
        self.title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #88aaff;")
        layout.addWidget(self.title_label, 0, 0, 1, 2)
        
        # 状态指示器
        self.status_label = QLabel("未知")
        self.status_label.setStyleSheet("font-weight: bold; color: yellow; font-size: 14px;")
        layout.addWidget(QLabel("当前状态:"), 1, 0)
        layout.addWidget(self.status_label, 1, 1)
        
        # 更新时间
        self.time_label = QLabel("-")
        layout.addWidget(QLabel("更新时间:"), 2, 0)
        layout.addWidget(self.time_label, 2, 1)
        
        # 连接状态
        self.connection_label = QLabel("连接中...")
        self.connection_label.setStyleSheet("color: yellow;")
        layout.addWidget(QLabel("数据源:"), 3, 0)
        layout.addWidget(self.connection_label, 3, 1)
        
        # 自动刷新
        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(60 * 1000)  # 60秒自动刷新
    
    def update_status(self, status_data):
        """更新市场状态"""
        # 更新状态标签
        status = status_data.get("status", "未知")
        self.status_label.setText(status)
        
        # 根据状态设置颜色
        if status == "交易中":
            self.status_label.setStyleSheet("font-weight: bold; color: #00ff00; font-size: 14px;")
        elif status == "已收盘":
            self.status_label.setStyleSheet("font-weight: bold; color: #ff9900; font-size: 14px;")
        elif status == "休市":
            self.status_label.setStyleSheet("font-weight: bold; color: #ff3333; font-size: 14px;")
        elif status == "未开盘":
            self.status_label.setStyleSheet("font-weight: bold; color: #ffcc00; font-size: 14px;")
        else:
            self.status_label.setStyleSheet("font-weight: bold; color: #cccccc; font-size: 14px;")
        
        # 更新时间
        self.time_label.setText(status_data.get("time", "-"))
        
        # 更新连接状态
        if "source" in status_data and status_data["source"] == "tushare":
            self.connection_label.setText("TuShare实时数据")
            self.connection_label.setStyleSheet("color: #00ff00;")
        else:
            self.connection_label.setText("模拟数据")
            self.connection_label.setStyleSheet("color: #ff9900;")


class IndexCardWidget(QFrame):
    """指数卡片小部件"""
    
    def __init__(self, code="000001.SH", name="上证指数", parent=None):
        super().__init__(parent)
        self.code = code
        self.name = name
        
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            IndexCardWidget {
                background-color: rgba(20, 20, 40, 180);
                border-radius: 5px;
                padding: 5px;
            }
            QLabel {
                color: white;
            }
        """)
        
        # 创建布局
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        
        # 名称标签
        self.name_label = QLabel(name)
        self.name_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #88aaff;")
        self.name_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.name_label)
        
        # 价格标签
        self.price_label = QLabel("---.--")
        self.price_label.setStyleSheet("font-size: 18px; font-weight: bold; color: white;")
        self.price_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.price_label)
        
        # 变化标签
        self.change_label = QLabel("-.--% (-.--)")
        self.change_label.setStyleSheet("font-size: 14px; color: gray;")
        self.change_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.change_label)
        
        # 图表
        if PYQTGRAPH_AVAILABLE:
            self.chart_widget = pg.PlotWidget()
            self.chart_widget.setBackground(None)
            self.chart_widget.setMinimumHeight(80)
            self.chart_widget.setMaximumHeight(120)
            self.chart_widget.setMenuEnabled(False)
            self.chart_widget.hideAxis('left')
            self.chart_widget.hideAxis('bottom')
            self.layout.addWidget(self.chart_widget)
            self.chart_curve = self.chart_widget.plot([], [], pen=pg.mkPen(color='c', width=2))
        else:
            self.chart_widget = QLabel("图表功能需要pyqtgraph库")
            self.chart_widget.setStyleSheet("color: gray; font-size: 10px;")
            self.chart_widget.setAlignment(Qt.AlignCenter)
            self.layout.addWidget(self.chart_widget)
    
    def update_data(self, index_data):
        """更新指数数据"""
        if not index_data:
            return
            
        # 更新价格
        price = index_data.get("price", 0)
        self.price_label.setText(f"{price:.2f}")
        
        # 更新变化
        change = index_data.get("change", 0)
        prev_close = index_data.get("prev_close", 0)
        change_value = price - prev_close if prev_close else 0
        
        # 根据变化设置颜色
        if change > 0:
            color = "#ff5555"  # 上涨红色
            change_text = f"+{change*100:.2f}% (+{change_value:.2f})"
        elif change < 0:
            color = "#00ff00"  # 下跌绿色
            change_text = f"{change*100:.2f}% ({change_value:.2f})"
        else:
            color = "gray"
            change_text = f"{change*100:.2f}% ({change_value:.2f})"
        
        self.change_label.setText(change_text)
        self.change_label.setStyleSheet(f"font-size: 14px; color: {color};")
        
        # 更新图表
        if PYQTGRAPH_AVAILABLE and "history" in index_data:
            prices = index_data["history"].get("prices", [])
            if prices:
                # 标准化价格用于绘图
                x = list(range(len(prices)))
                self.chart_curve.setData(x, prices)
                
                # 设置图表颜色
                pen_color = 'r' if change >= 0 else 'g'
                self.chart_curve.setPen(pg.mkPen(color=pen_color, width=2))


class StockTableWidget(QTableWidget):
    """股票列表表格小部件"""
    
    stock_selected = pyqtSignal(str)  # 发出信号：选择的股票代码
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 设置表格属性
        self.setColumnCount(6)
        self.setHorizontalHeaderLabels(["名称", "代码", "最新价", "涨跌幅", "推荐度", "原因"])
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setAlternatingRowColors(True)
        self.setStyleSheet("""
            QTableWidget {
                background-color: rgba(30, 30, 50, 150);
                alternate-background-color: rgba(35, 35, 55, 150);
                border: none;
                color: white;
            }
            QHeaderView::section {
                background-color: rgba(40, 40, 60, 200);
                color: #88aaff;
                border: none;
                padding: 4px;
            }
            QTableWidget::item:selected {
                background-color: rgba(100, 100, 150, 150);
            }
        """)
        
        # 设置列宽
        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)  # 名称列自适应
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)  # 代码列自适应内容
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)  # 价格列自适应内容
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)  # 涨跌幅列自适应内容
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)  # 推荐度列自适应内容
        header.setSectionResizeMode(5, QHeaderView.Stretch)  # 原因列自适应
        
        # 连接信号
        self.cellClicked.connect(self.on_cell_clicked)
    
    def on_cell_clicked(self, row, col):
        """处理单元格点击事件"""
        # 获取点击行的股票代码
        code_item = self.item(row, 1)
        if code_item:
            code = code_item.text()
            self.stock_selected.emit(code)
    
    def update_stocks(self, stocks_data):
        """更新股票列表"""
        # 清空表格
        self.setRowCount(0)
        
        # 填充数据
        for row, stock in enumerate(stocks_data):
            self.insertRow(row)
            
            # 名称
            name_item = QTableWidgetItem(stock.get("name", ""))
            self.setItem(row, 0, name_item)
            
            # 代码
            code = stock.get("code", "") or stock.get("ts_code", "").split(".")[0]
            code_item = QTableWidgetItem(code)
            self.setItem(row, 1, code_item)
            
            # 价格
            price = stock.get("price", 0)
            price_item = QTableWidgetItem(f"{price:.2f}")
            self.setItem(row, 2, price_item)
            
            # 涨跌幅
            change = stock.get("change", 0)
            change_text = f"{change*100:+.2f}%" if change != 0 else "0.00%"
            change_item = QTableWidgetItem(change_text)
            # 设置颜色
            if change > 0:
                change_item.setForeground(QBrush(QColor("#ff5555")))
            elif change < 0:
                change_item.setForeground(QBrush(QColor("#00ff00")))
            self.setItem(row, 3, change_item)
            
            # 推荐度
            recommendation = stock.get("recommendation", 0)
            if recommendation > 0:
                stars = "★" * int(recommendation * 5)
                recommendation_item = QTableWidgetItem(stars)
                # 根据推荐度设置颜色
                if recommendation > 0.8:
                    recommendation_item.setForeground(QBrush(QColor("#ff5555")))  # 高推荐为红色
                elif recommendation > 0.6:
                    recommendation_item.setForeground(QBrush(QColor("#ffcc00")))  # 中推荐为黄色
                else:
                    recommendation_item.setForeground(QBrush(QColor("#88aaff")))  # 低推荐为蓝色
            else:
                recommendation_item = QTableWidgetItem("-")
            self.setItem(row, 4, recommendation_item)
            
            # 原因
            reason = stock.get("reason", "")
            reason_item = QTableWidgetItem(reason)
            self.setItem(row, 5, reason_item)


class MarketOverviewWidget(QWidget):
    """市场概览部件"""
    
    def __init__(self, data_controller, parent=None):
        super().__init__(parent)
        self.data_controller = data_controller
        
        # 创建布局
        main_layout = QVBoxLayout(self)
        
        # 顶部布局（市场状态和搜索）
        top_layout = QHBoxLayout()
        
        # 市场状态
        self.market_status = MarketStatusWidget()
        top_layout.addWidget(self.market_status, 2)
        
        # 搜索框
        search_group = QGroupBox("股票搜索")
        search_group.setStyleSheet("""
            QGroupBox {
                color: #88aaff;
                font-weight: bold;
                border: 1px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                background-color: rgba(20, 20, 40, 180);
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        search_layout = QVBoxLayout(search_group)
        
        search_input_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("输入股票代码或名称")
        search_input_layout.addWidget(self.search_input)
        
        if QTA_AVAILABLE:
            search_button = QToolButton()
            search_button.setIcon(qta.icon('fa5s.search'))
        else:
            search_button = QPushButton("搜索")
        search_button.clicked.connect(self.on_search)
        search_input_layout.addWidget(search_button)
        
        search_layout.addLayout(search_input_layout)
        top_layout.addWidget(search_group, 3)
        
        main_layout.addLayout(top_layout)
        
        # 指数卡片布局
        indices_layout = QHBoxLayout()
        
        # 创建四个指数卡片
        self.index_cards = {}
        indices_info = [
            ("000001.SH", "上证指数"),
            ("399001.SZ", "深证成指"),
            ("399006.SZ", "创业板指"),
            ("000688.SH", "科创50")
        ]
        
        for code, name in indices_info:
            card = IndexCardWidget(code, name)
            self.index_cards[code] = card
            indices_layout.addWidget(card)
        
        main_layout.addLayout(indices_layout)
        
        # 热门股和推荐股切换
        tabs_layout = QHBoxLayout()
        
        # 热门和推荐股Tab
        self.stocks_tabs = QTabWidget()
        self.stocks_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: none;
                background: transparent;
            }
            QTabBar::tab {
                background: rgba(40, 40, 60, 150);
                color: white;
                border: none;
                padding: 5px 10px;
                margin: 1px;
            }
            QTabBar::tab:selected {
                background: rgba(60, 60, 100, 200);
                color: #88aaff;
                font-weight: bold;
            }
        """)
        
        # 热门股票表格
        self.hot_stocks_table = StockTableWidget()
        self.stocks_tabs.addTab(self.hot_stocks_table, "热门股票")
        
        # 推荐股票表格
        self.recommended_stocks_table = StockTableWidget()
        self.stocks_tabs.addTab(self.recommended_stocks_table, "推荐股票")
        
        tabs_layout.addWidget(self.stocks_tabs)
        main_layout.addLayout(tabs_layout)
        
        # 连接信号
        if hasattr(self.data_controller, 'market_status_updated_signal'):
            self.data_controller.market_status_updated_signal.connect(self.update_market_status)
        
        # 创建刷新定时器
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_data)
        self.refresh_timer.start(60 * 1000)  # 每分钟刷新一次
        
        # 初始加载数据
        self.refresh_data()
    
    def refresh_data(self):
        """刷新数据"""
        try:
            # 获取市场状态
            market_status = self.data_controller.get_market_status()
            self.update_market_status(market_status)
            
            # 获取指数数据
            indices = self.data_controller.get_indices_data()
            self.update_indices(indices)
            
            # 获取热门股票
            hot_stocks = self.data_controller.get_hot_stocks()
            self.hot_stocks_table.update_stocks(hot_stocks)
            
            # 获取推荐股票
            recommended_stocks = self.data_controller.get_recommended_stocks()
            self.recommended_stocks_table.update_stocks(recommended_stocks)
        except Exception as e:
            logging.error(f"刷新市场数据失败: {str(e)}")
    
    def update_market_status(self, status_data):
        """更新市场状态"""
        self.market_status.update_status(status_data)
    
    def update_indices(self, indices_data):
        """更新指数数据"""
        for code, card in self.index_cards.items():
            if code in indices_data:
                card.update_data(indices_data[code])
    
    def on_search(self):
        """处理搜索操作"""
        search_text = self.search_input.text().strip()
        if not search_text:
            return
        
        # TODO: 实现股票搜索功能
        logging.info(f"搜索股票: {search_text}")
        
        # 可以在这里实现股票搜索功能，在表格中高亮匹配的股票或显示搜索结果


class RealTimeMarketView(QWidget):
    """实时市场视图"""
    
    def __init__(self, data_controller, parent=None):
        """初始化视图"""
        super().__init__(parent)
        
        # 保存数据控制器引用
        self.data_controller = data_controller
        
        # 初始化UI
        self._setup_ui()
        
        # 连接信号
        self._connect_signals()
        
        # 加载初始数据
        self.refresh_data()
        
    def _setup_ui(self):
        """设置UI"""
        # 主布局
        main_layout = QVBoxLayout(self)
        
        # 创建市场状态区域
        status_group = QGroupBox("市场状态")
        status_layout = QGridLayout(status_group)
        
        # 市场状态
        self.market_status_label = QLabel("当前状态:")
        self.market_status_value = QLabel("未知")
        self.market_status_value.setStyleSheet("color: yellow; font-weight: bold;")
        status_layout.addWidget(self.market_status_label, 0, 0)
        status_layout.addWidget(self.market_status_value, 0, 1)
        
        # 更新时间
        self.update_time_label = QLabel("更新时间:")
        self.update_time_value = QLabel("-")
        status_layout.addWidget(self.update_time_label, 1, 0)
        status_layout.addWidget(self.update_time_value, 1, 1)
        
        # 数据源
        self.data_source_label = QLabel("数据源:")
        self.data_source_value = QLabel("连接中...")
        self.data_source_value.setStyleSheet("color: yellow;")
        status_layout.addWidget(self.data_source_label, 2, 0)
        status_layout.addWidget(self.data_source_value, 2, 1)
        
        # 添加进度条
        self.loading_progress = QProgressBar()
        self.loading_progress.setRange(0, 100)
        self.loading_progress.setValue(0)
        self.loading_progress.setVisible(False)  # 默认隐藏
        status_layout.addWidget(self.loading_progress, 3, 0, 1, 2)
        
        # 添加刷新按钮
        self.refresh_button = QPushButton("刷新数据")
        self.refresh_button.clicked.connect(self.refresh_data)
        status_layout.addWidget(self.refresh_button, 4, 0, 1, 2)
        
        main_layout.addWidget(status_group)
        
        # 创建指数区域
        indices_layout = QHBoxLayout()
        
        # 创建上证指数卡片
        self.sh_index_card = self._create_index_card("上证指数")
        indices_layout.addWidget(self.sh_index_card)
        
        # 创建深证成指卡片
        self.sz_index_card = self._create_index_card("深证成指")
        indices_layout.addWidget(self.sz_index_card)
        
        # 创建创业板指卡片
        self.cyb_index_card = self._create_index_card("创业板指")
        indices_layout.addWidget(self.cyb_index_card)
        
        # 创建科创板指卡片
        self.kcb_index_card = self._create_index_card("科创50")
        indices_layout.addWidget(self.kcb_index_card)
        
        main_layout.addLayout(indices_layout)
        
        # 创建股票列表区域
        stocks_group = QGroupBox("热门股票")
        stocks_layout = QVBoxLayout(stocks_group)
        
        # 创建股票表格
        self.stocks_table = QTableWidget()
        self.stocks_table.setColumnCount(6)
        self.stocks_table.setHorizontalHeaderLabels(["名称", "代码", "最新价", "涨跌幅", "推荐度", "原因"])
        self.stocks_table.horizontalHeader().setStretchLastSection(True)
        self.stocks_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.stocks_table.verticalHeader().setVisible(False)
        self.stocks_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.stocks_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.stocks_table.setAlternatingRowColors(True)
        stocks_layout.addWidget(self.stocks_table)
        
        main_layout.addWidget(stocks_group)
        
        # 搜索区域
        search_group = QGroupBox("股票搜索")
        search_layout = QHBoxLayout(search_group)
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("输入股票代码或名称")
        search_layout.addWidget(self.search_input)
        
        self.search_button = QPushButton("搜索")
        self.search_button.clicked.connect(self._on_search)
        search_layout.addWidget(self.search_button)
        
        main_layout.addWidget(search_group)
        
    def _connect_signals(self):
        """连接信号"""
        # 连接数据控制器的信号
        self.data_controller.market_status_updated_signal.connect(self.update_market_status)
        self.data_controller.market_indices_updated_signal.connect(self.update_indices)
        self.data_controller.recommended_stocks_updated_signal.connect(self.update_recommended_stocks)
        self.data_controller.loading_progress_signal.connect(self.update_loading_progress)
        self.data_controller.error_signal.connect(self.show_error)
        
    def _create_index_card(self, title):
        """创建指数卡片"""
        card = QGroupBox(title)
        card.setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #555; border-radius: 5px; padding: 10px; }")
        
        layout = QVBoxLayout(card)
        
        # 指数值
        value_label = QLabel("----.--")
        value_label.setStyleSheet("font-size: 18px; font-weight: bold; color: white;")
        value_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(value_label)
        
        # 涨跌幅
        change_label = QLabel("-.-% (-.--)")
        change_label.setStyleSheet("font-size: 14px; color: gray;")
        change_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(change_label)
        
        # 添加K线图占位区域
        chart_placeholder = QFrame()
        chart_placeholder.setMinimumHeight(100)
        chart_placeholder.setFrameShape(QFrame.StyledPanel)
        chart_placeholder.setStyleSheet("background-color: #1a1a1a;")
        layout.addWidget(chart_placeholder)
        
        # 保存引用
        card.value_label = value_label
        card.change_label = change_label
        card.chart_placeholder = chart_placeholder
        
        return card
        
    def refresh_data(self):
        """刷新数据"""
        try:
            # 显示加载状态
            self.data_source_value.setText("正在刷新...")
            self.data_source_value.setStyleSheet("color: yellow;")
            self.loading_progress.setVisible(True)
            self.loading_progress.setValue(10)
            
            # 更新时间
            self.update_time_value.setText(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            
            # 刷新市场数据
            try:
                QApplication.processEvents()  # 确保UI更新
            except Exception as e:
                logging.warning(f"更新UI显示时出错: {str(e)}")
            
            if hasattr(self.data_controller, 'refresh_market_data'):
                success = self.data_controller.refresh_market_data()
                
                # 更新数据源状态
                if success:
                    # 检查是使用TuShare还是备用数据
                    if self.data_controller.is_tushare_ready:
                        self.data_source_value.setText("TuShare Pro")
                        self.data_source_value.setStyleSheet("color: green;")
                    else:
                        self.data_source_value.setText("备用数据源")
                        self.data_source_value.setStyleSheet("color: orange;")
                else:
                    self.data_source_value.setText("刷新失败")
                    self.data_source_value.setStyleSheet("color: red;")
                
                self.loading_progress.setValue(100)
                # 0.5秒后隐藏进度条
                QTimer.singleShot(500, lambda: self.loading_progress.setVisible(False))
            else:
                self.data_source_value.setText("接口不可用")
                self.data_source_value.setStyleSheet("color: red;")
                self.loading_progress.setVisible(False)
                
        except Exception as e:
            self.data_source_value.setText("刷新出错")
            self.data_source_value.setStyleSheet("color: red;")
            self.loading_progress.setVisible(False)
            logging.error(f"刷新数据失败: {str(e)}")
            self.show_error(f"刷新数据失败: {str(e)}")
    
    def update_market_status(self, status_data):
        """更新市场状态"""
        try:
            status = status_data.get("status", "未知")
            time_str = status_data.get("time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            
            # 更新状态显示
            self.market_status_value.setText(status)
            self.update_time_value.setText(time_str)
            
            # 根据状态设置颜色
            if status == "交易中":
                self.market_status_value.setStyleSheet("color: #00ff00; font-weight: bold;")
            elif status == "已收盘":
                self.market_status_value.setStyleSheet("color: #ff9900; font-weight: bold;")
            elif status == "未开盘":
                self.market_status_value.setStyleSheet("color: #00ccff; font-weight: bold;")
            elif status == "休市":
                self.market_status_value.setStyleSheet("color: #ff3333; font-weight: bold;")
            elif status == "午休":
                self.market_status_value.setStyleSheet("color: #ffcc00; font-weight: bold;")
            else:
                self.market_status_value.setStyleSheet("color: #cccccc; font-weight: bold;")
                
        except Exception as e:
            logging.error(f"更新市场状态失败: {str(e)}")
    
    def update_indices(self, indices):
        """更新指数数据"""
        try:
            # 指数代码映射到卡片
            index_cards = {
                "000001.SH": self.sh_index_card,  # 上证指数
                "000001": self.sh_index_card,     # 兼容简写
                "sh000001": self.sh_index_card,   # 兼容行情简写
                
                "399001.SZ": self.sz_index_card,  # 深证成指
                "399001": self.sz_index_card,     # 兼容简写
                "sz399001": self.sz_index_card,   # 兼容行情简写
                
                "399006.SZ": self.cyb_index_card, # 创业板指
                "399006": self.cyb_index_card,    # 兼容简写
                "sz399006": self.cyb_index_card,  # 兼容行情简写
                
                "000688.SH": self.kcb_index_card, # 科创50
                "000688": self.kcb_index_card,    # 兼容简写
                "sh000688": self.kcb_index_card   # 兼容行情简写
            }
            
            name_to_card = {
                "上证指数": self.sh_index_card,
                "深证成指": self.sz_index_card,
                "创业板指": self.cyb_index_card,
                "科创50": self.kcb_index_card
            }
            
            # 更新指数卡片
            for index in indices:
                code = index.get("code", "")
                name = index.get("name", "")
                price = index.get("price", 0)
                change = index.get("change", 0)
                change_pct = index.get("change_pct", 0)
                
                # 找到对应的卡片
                card = index_cards.get(code) or name_to_card.get(name)
                
                if card:
                    # 更新卡片数据
                    card.value_label.setText(f"{price:.2f}")
                    card.change_label.setText(f"{change_pct:+.2f}% ({change:+.2f})")
                    
                    # 设置颜色
                    if change_pct > 0:
                        card.value_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #ff3333;")
                        card.change_label.setStyleSheet("font-size: 14px; color: #ff3333;")
                    elif change_pct < 0:
                        card.value_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #00aa00;")
                        card.change_label.setStyleSheet("font-size: 14px; color: #00aa00;")
                    else:
                        card.value_label.setStyleSheet("font-size: 18px; font-weight: bold; color: white;")
                        card.change_label.setStyleSheet("font-size: 14px; color: white;")
            
        except Exception as e:
            logging.error(f"更新指数数据失败: {str(e)}")
    
    def update_recommended_stocks(self, stocks):
        """更新推荐股票"""
        try:
            # 清空表格
            self.stocks_table.setRowCount(0)
            
            # 添加股票数据
            for i, stock in enumerate(stocks):
                self.stocks_table.insertRow(i)
                
                # 获取数据
                name = stock.get("name", "")
                code = stock.get("code", "")
                price = stock.get("price", 0)
                change_pct = stock.get("change_pct", 0)
                recommendation = stock.get("recommendation", 0)
                reason = stock.get("reason", "")
                
                # 创建单元格
                name_item = QTableWidgetItem(name)
                code_item = QTableWidgetItem(code)
                price_item = QTableWidgetItem(f"{price:.2f}")
                change_item = QTableWidgetItem(f"{change_pct:+.2f}%")
                recommendation_item = QTableWidgetItem(f"{recommendation:.2f}")
                reason_item = QTableWidgetItem(reason)
                
                # 设置颜色
                if change_pct > 0:
                    change_item.setForeground(QColor("#ff3333"))
                elif change_pct < 0:
                    change_item.setForeground(QColor("#00aa00"))
                
                # 根据推荐度设置颜色
                if recommendation > 0.8:
                    recommendation_item.setForeground(QColor("#ff3333"))
                elif recommendation > 0.6:
                    recommendation_item.setForeground(QColor("#ff9900"))
                else:
                    recommendation_item.setForeground(QColor("#ffcc00"))
                
                # 添加单元格
                self.stocks_table.setItem(i, 0, name_item)
                self.stocks_table.setItem(i, 1, code_item)
                self.stocks_table.setItem(i, 2, price_item)
                self.stocks_table.setItem(i, 3, change_item)
                self.stocks_table.setItem(i, 4, recommendation_item)
                self.stocks_table.setItem(i, 5, reason_item)
            
        except Exception as e:
            logging.error(f"更新推荐股票失败: {str(e)}")
    
    def update_loading_progress(self, progress, message):
        """更新加载进度"""
        try:
            self.loading_progress.setValue(progress)
            self.loading_progress.setVisible(True)
            
            if message:
                self.data_source_value.setText(message)
                
            if progress >= 100:
                # 0.5秒后隐藏进度条
                QTimer.singleShot(500, lambda: self.loading_progress.setVisible(False))
                
        except Exception as e:
            logging.error(f"更新加载进度失败: {str(e)}")
    
    def show_error(self, message):
        """显示错误信息"""
        try:
            logging.error(f"错误: {message}")
            
            # 仅在真正严重的错误时显示弹窗
            if "初始化失败" in message or "无法连接" in message:
                QMessageBox.critical(self, "错误", message)
        except Exception as e:
            logging.error(f"显示错误信息失败: {str(e)}")
    
    def _on_search(self):
        """处理搜索请求"""
        try:
            query = self.search_input.text().strip()
            if not query:
                return
                
            # 调用数据控制器搜索
            self.data_controller.search_stocks(query, self._on_search_result)
            
            # 显示加载状态
            self.search_button.setEnabled(False)
            self.search_button.setText("搜索中...")
            
        except Exception as e:
            logging.error(f"搜索失败: {str(e)}")
            self.show_error(f"搜索失败: {str(e)}")
            self.search_button.setEnabled(True)
            self.search_button.setText("搜索")
    
    def _on_search_result(self, results):
        """处理搜索结果"""
        try:
            # 恢复按钮状态
            self.search_button.setEnabled(True)
            self.search_button.setText("搜索")
            
            if not results:
                QMessageBox.information(self, "搜索结果", "没有找到匹配的股票")
                return
                
            # 创建结果对话框
            dialog = QDialog(self)
            dialog.setWindowTitle("搜索结果")
            dialog.setMinimumWidth(400)
            
            # 创建布局
            layout = QVBoxLayout(dialog)
            
            # 创建列表
            result_list = QListWidget()
            for stock in results:
                name = stock.get("name", "")
                code = stock.get("code", "")
                item = QListWidgetItem(f"{name} ({code})")
                item.setData(Qt.UserRole, stock)
                result_list.addItem(item)
                
            layout.addWidget(result_list)
            
            # 创建按钮
            button_layout = QHBoxLayout()
            select_button = QPushButton("查看")
            close_button = QPushButton("关闭")
            
            button_layout.addWidget(select_button)
            button_layout.addWidget(close_button)
            layout.addLayout(button_layout)
            
            # 连接信号
            select_button.clicked.connect(lambda: self._select_stock(result_list.currentItem()))
            close_button.clicked.connect(dialog.accept)
            result_list.itemDoubleClicked.connect(lambda item: self._select_stock(item))
            
            # 显示对话框
            dialog.exec_()
            
        except Exception as e:
            logging.error(f"处理搜索结果失败: {str(e)}")
            self.show_error(f"处理搜索结果失败: {str(e)}")
    
    def _select_stock(self, item):
        """选择股票"""
        try:
            if not item:
                return
                
            # 获取股票数据
            stock = item.data(Qt.UserRole)
            if not stock:
                return
                
            # 弹出消息框
            code = stock.get("code", "")
            name = stock.get("name", "")
            
            # 关闭父对话框
            parent_dialog = item.listWidget().parent()
            if parent_dialog:
                parent_dialog.accept()
                
            # 显示股票详情
            QMessageBox.information(self, "股票信息", f"您选择了: {name} ({code})\n\n该功能正在开发中，敬请期待！")
            
        except Exception as e:
            logging.error(f"选择股票失败: {str(e)}")
            self.show_error(f"选择股票失败: {str(e)}") 