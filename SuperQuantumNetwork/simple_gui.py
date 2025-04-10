#!/usr/bin/env python
"""
简化版图形界面模块 - 在完整GUI不可用时作为备用
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QComboBox, QTableWidget, QTableWidgetItem, QTabWidget,
    QLineEdit, QFormLayout, QGroupBox, QSplitter, QMessageBox, QFrame,
    QHeaderView, QDateEdit
)
from PyQt5.QtCore import Qt, QTimer, QDate
from PyQt5.QtGui import QColor, QPalette, QFont

logger = logging.getLogger("SimpleGUI")

class SimpleMainWindow(QMainWindow):
    """简化版主窗口"""
    
    def __init__(self, data_connector):
        super().__init__()
        
        self.data_connector = data_connector
        self.watched_stocks = ["000001.SZ", "600519.SH", "000300.SH", "399006.SZ"]
        self.stock_data = {}
        self.init_ui()
        
        # 启动定时器更新数据
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_data)
        self.timer.start(60000)  # 每分钟更新一次
        
        # 立即加载数据
        self.update_data()
        
    def init_ui(self):
        """初始化UI"""
        # 设置窗口属性
        self.setWindowTitle("量子交易系统 - 简化版")
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建中央控件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 创建标签
        header_label = QLabel("超神量子交易系统 - 简化版")
        header_label.setStyleSheet("font-size: 20px; font-weight: bold; margin: 10px;")
        header_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header_label)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # 左侧控制面板
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # 创建市场监控组
        market_group = QGroupBox("市场监控")
        market_layout = QVBoxLayout(market_group)
        
        # 添加操作控件
        control_layout = QHBoxLayout()
        
        # 股票代码输入
        self.stock_input = QLineEdit()
        self.stock_input.setPlaceholderText("输入股票代码，例如: 000001.SZ")
        control_layout.addWidget(self.stock_input)
        
        # 添加按钮
        add_button = QPushButton("添加监控")
        add_button.clicked.connect(self.add_stock)
        control_layout.addWidget(add_button)
        
        market_layout.addLayout(control_layout)
        
        # 创建股票表格
        self.stock_table = QTableWidget()
        self.stock_table.setColumnCount(6)
        self.stock_table.setHorizontalHeaderLabels(["代码", "名称", "最新价", "涨跌幅", "量子评分", "操作"])
        self.stock_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        market_layout.addWidget(self.stock_table)
        
        left_layout.addWidget(market_group)
        
        # 创建量子分析组
        quantum_group = QGroupBox("量子分析")
        quantum_layout = QVBoxLayout(quantum_group)
        
        # 添加日期选择
        date_layout = QHBoxLayout()
        date_layout.addWidget(QLabel("分析日期:"))
        
        self.date_edit = QDateEdit()
        self.date_edit.setDate(QDate.currentDate())
        self.date_edit.setCalendarPopup(True)
        date_layout.addWidget(self.date_edit)
        
        analyze_button = QPushButton("分析")
        analyze_button.clicked.connect(self.run_analysis)
        date_layout.addWidget(analyze_button)
        
        quantum_layout.addLayout(date_layout)
        
        # 添加分析结果标签
        self.analysis_label = QLabel("尚未进行分析")
        self.analysis_label.setStyleSheet("font-size: 14px; margin: 10px;")
        self.analysis_label.setAlignment(Qt.AlignCenter)
        quantum_layout.addWidget(self.analysis_label)
        
        # 添加分析结果文本
        self.analysis_text = QLabel()
        self.analysis_text.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px;")
        self.analysis_text.setWordWrap(True)
        self.analysis_text.setMinimumHeight(200)
        quantum_layout.addWidget(self.analysis_text)
        
        left_layout.addWidget(quantum_group)
        
        # 右侧图表面板
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # 创建标签
        chart_label = QLabel("股票图表")
        chart_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        right_layout.addWidget(chart_label)
        
        # 创建选择器
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("选择股票:"))
        
        self.chart_combo = QComboBox()
        for stock in self.watched_stocks:
            self.chart_combo.addItem(stock)
        self.chart_combo.currentTextChanged.connect(self.update_chart)
        selector_layout.addWidget(self.chart_combo)
        
        selector_layout.addWidget(QLabel("周期:"))
        
        self.period_combo = QComboBox()
        self.period_combo.addItems(["日K", "周K", "月K"])
        self.period_combo.currentTextChanged.connect(self.update_chart)
        selector_layout.addWidget(self.period_combo)
        
        right_layout.addLayout(selector_layout)
        
        # 图表占位符
        self.chart_placeholder = QLabel("图表加载中...")
        self.chart_placeholder.setStyleSheet("font-size: 18px; color: gray;")
        self.chart_placeholder.setAlignment(Qt.AlignCenter)
        self.chart_placeholder.setMinimumHeight(400)
        right_layout.addWidget(self.chart_placeholder)
        
        # 添加面板到分割器
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 800])
        
        # 创建状态栏
        self.statusBar().showMessage("系统就绪")
        
    def add_stock(self):
        """添加股票到监控列表"""
        stock_code = self.stock_input.text().strip()
        if not stock_code:
            QMessageBox.warning(self, "输入错误", "请输入有效的股票代码")
            return
            
        if stock_code in self.watched_stocks:
            QMessageBox.information(self, "提示", "该股票已在监控列表中")
            return
            
        # 添加到列表
        self.watched_stocks.append(stock_code)
        self.chart_combo.addItem(stock_code)
        
        # 更新数据
        self.update_data()
        
        # 清空输入框
        self.stock_input.clear()
        
    def update_data(self):
        """更新股票数据"""
        self.statusBar().showMessage("正在更新数据...")
        
        # 更新表格大小
        self.stock_table.setRowCount(len(self.watched_stocks))
        
        for i, stock_code in enumerate(self.watched_stocks):
            try:
                # 获取股票数据
                end_date = datetime.now().strftime("%Y%m%d")
                start_date = (datetime.now().replace(year=datetime.now().year - 1)).strftime("%Y%m%d")
                
                df = self.data_connector.get_daily_data(stock_code, start_date, end_date)
                if df is None or df.empty:
                    logger.warning(f"无法获取股票数据: {stock_code}")
                    continue
                    
                # 保存数据
                self.stock_data[stock_code] = df
                
                # 计算涨跌幅
                latest_price = float(df.iloc[-1]['close'])
                prev_price = float(df.iloc[-2]['close']) if len(df) > 1 else latest_price
                change_pct = (latest_price - prev_price) / prev_price * 100 if prev_price > 0 else 0
                
                # 生成随机量子评分（模拟）
                quantum_score = np.random.uniform(0, 100)
                
                # 更新表格
                self.stock_table.setItem(i, 0, QTableWidgetItem(stock_code))
                self.stock_table.setItem(i, 1, QTableWidgetItem(self._get_stock_name(stock_code)))
                self.stock_table.setItem(i, 2, QTableWidgetItem(f"{latest_price:.2f}"))
                
                change_item = QTableWidgetItem(f"{change_pct:.2f}%")
                change_item.setForeground(QColor("red" if change_pct >= 0 else "green"))
                self.stock_table.setItem(i, 3, change_item)
                
                score_item = QTableWidgetItem(f"{quantum_score:.1f}")
                score_color = "red" if quantum_score > 70 else "green" if quantum_score < 30 else "black"
                score_item.setForeground(QColor(score_color))
                self.stock_table.setItem(i, 4, score_item)
                
                # 添加删除按钮
                remove_button = QPushButton("删除")
                remove_button.clicked.connect(lambda checked, code=stock_code: self.remove_stock(code))
                self.stock_table.setCellWidget(i, 5, remove_button)
                
            except Exception as e:
                logger.error(f"更新股票数据出错: {str(e)}")
        
        # 更新图表
        self.update_chart()
        
        self.statusBar().showMessage(f"数据更新完成，共 {len(self.watched_stocks)} 只股票")
    
    def _get_stock_name(self, code):
        """获取股票名称（模拟）"""
        stock_names = {
            "000001.SZ": "平安银行",
            "600519.SH": "贵州茅台",
            "000300.SH": "沪深300",
            "399006.SZ": "创业板指"
        }
        return stock_names.get(code, "未知")
    
    def remove_stock(self, stock_code):
        """从监控列表中移除股票"""
        if stock_code in self.watched_stocks:
            self.watched_stocks.remove(stock_code)
            
            # 从下拉框中移除
            index = self.chart_combo.findText(stock_code)
            if index >= 0:
                self.chart_combo.removeItem(index)
            
            # 更新数据
            self.update_data()
    
    def update_chart(self):
        """更新图表"""
        stock_code = self.chart_combo.currentText()
        period = self.period_combo.currentText()
        
        if not stock_code or stock_code not in self.stock_data:
            self.chart_placeholder.setText("无可用数据")
            return
            
        # 更新图表占位符（在实际应用中，这里应该绘制真实图表）
        df = self.stock_data[stock_code]
        latest_price = float(df.iloc[-1]['close'])
        high_price = float(df['high'].max())
        low_price = float(df['low'].min())
        
        chart_text = f"""
        <h3>{self._get_stock_name(stock_code)} ({stock_code})</h3>
        <p>周期: {period}</p>
        <p>最新价: {latest_price:.2f}</p>
        <p>最高价: {high_price:.2f}</p>
        <p>最低价: {low_price:.2f}</p>
        <p>成交量: {int(df.iloc[-1]['volume'])}</p>
        <p>【此处在实际应用中应显示K线图】</p>
        """
        
        self.chart_placeholder.setText(chart_text)
    
    def run_analysis(self):
        """运行量子分析"""
        selected_date = self.date_edit.date().toString("yyyyMMdd")
        
        # 模拟分析运行
        self.analysis_label.setText(f"量子分析结果 ({selected_date})")
        
        # 生成模拟分析结果
        analysis_result = """
        <h3>市场分析</h3>
        <p>市场趋势: <span style='color:red'>上涨概率 58%</span></p>
        <p>量子纠缠强度: 0.72 (较强)</p>
        <p>市场波动率预测: 中等</p>
        
        <h3>个股量子评分</h3>
        <p>平安银行(000001.SZ): <span style='color:green'>67.5分</span> - 建议持有</p>
        <p>贵州茅台(600519.SH): <span style='color:red'>82.3分</span> - 建议买入</p>
        
        <h3>量子策略建议</h3>
        <p>根据量子场分析，市场处于震荡上行阶段，建议适度配置大盘蓝筹股，增加科技成长股比例。</p>
        """
        
        self.analysis_text.setText(analysis_result)
        
        # 更新状态栏
        self.statusBar().showMessage("量子分析完成")


def run_simple_gui(data_connector):
    """运行简化版图形界面"""
    app = QApplication(sys.argv)
    
    # 设置全局样式
    app.setStyle("Fusion")
    
    # 创建主窗口
    main_window = SimpleMainWindow(data_connector)
    main_window.show()
    
    # 运行应用
    sys.exit(app.exec_())


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建模拟数据连接器
    class MockDataConnector:
        def __init__(self):
            self.name = "MockConnector"
            
        def get_daily_data(self, symbol, start_date, end_date):
            # 生成模拟数据
            dates = pd.date_range(start=start_date, end=end_date)
            prices = np.random.normal(100, 5, size=len(dates))
            volumes = np.random.normal(1000000, 500000, size=len(dates))
            
            df = pd.DataFrame({
                'date': dates,
                'open': prices,
                'high': prices * 1.02,
                'low': prices * 0.98,
                'close': prices * (1 + np.random.normal(0, 0.01, size=len(dates))),
                'volume': volumes
            })
            
            return df
    
    # 运行简化版GUI
    run_simple_gui(MockDataConnector()) 