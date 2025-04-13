#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
市场分析面板 - 提供高级的市场分析功能
"""

import sys
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QComboBox, QPushButton, QTableWidget, QTableWidgetItem,
    QTabWidget, QSplitter, QFrame, QGridLayout, QSpinBox,
    QDoubleSpinBox, QCheckBox, QGroupBox, QFormLayout,
    QLineEdit, QTextEdit, QProgressBar, QMessageBox, QFileDialog
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPalette
import pyqtgraph as pg
from pyqtgraph import PlotWidget, PlotItem

from quantum_core.market_to_quantum import MarketToQuantumConverter
from quantum_core.multidimensional_analyzer import MultidimensionalAnalyzer

class MarketAnalysisPanel(QWidget):
    """市场分析面板 - 提供高级的市场分析功能"""
    
    # 信号定义
    analysis_started = pyqtSignal(str)  # 分析开始信号
    analysis_finished = pyqtSignal(str, dict)  # 分析完成信号
    analysis_error = pyqtSignal(str, str)  # 分析错误信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger('quantum_desktop.ui.market_analysis_panel')
        self.init_ui()
        self.logger.info("市场分析面板初始化完成")
        
    def init_ui(self):
        """初始化UI"""
        # 创建主布局
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        
        # 创建控制面板
        control_panel = self._create_control_panel()
        main_layout.addWidget(control_panel)
        
        # 创建分析面板
        analysis_panel = self._create_analysis_panel()
        main_layout.addWidget(analysis_panel)
        
        # 创建结果面板
        result_panel = self._create_result_panel()
        main_layout.addWidget(result_panel)
        
    def _create_control_panel(self) -> QWidget:
        """创建控制面板"""
        panel = QGroupBox("分析控制")
        layout = QGridLayout()
        
        # 股票选择
        stock_label = QLabel("股票代码:")
        self.stock_input = QLineEdit()
        self.stock_input.setPlaceholderText("输入股票代码")
        layout.addWidget(stock_label, 0, 0)
        layout.addWidget(self.stock_input, 0, 1)
        
        # 时间范围选择
        time_label = QLabel("时间范围:")
        self.time_range = QComboBox()
        self.time_range.addItems(["1天", "1周", "1月", "3月", "6月", "1年"])
        layout.addWidget(time_label, 0, 2)
        layout.addWidget(self.time_range, 0, 3)
        
        # 分析类型选择
        analysis_label = QLabel("分析类型:")
        self.analysis_type = QComboBox()
        self.analysis_type.addItems([
            "技术分析",
            "基本面分析",
            "量子分析",
            "情绪分析",
            "综合分析"
        ])
        layout.addWidget(analysis_label, 1, 0)
        layout.addWidget(self.analysis_type, 1, 1)
        
        # 分析参数设置
        params_label = QLabel("分析参数:")
        self.params_widget = QWidget()
        params_layout = QHBoxLayout()
        self.params_widget.setLayout(params_layout)
        layout.addWidget(params_label, 1, 2)
        layout.addWidget(self.params_widget, 1, 3)
        
        # 开始分析按钮
        self.start_button = QPushButton("开始分析")
        self.start_button.clicked.connect(self._start_analysis)
        layout.addWidget(self.start_button, 2, 0, 1, 4)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar, 3, 0, 1, 4)
        
        panel.setLayout(layout)
        return panel
        
    def _create_analysis_panel(self) -> QWidget:
        """创建分析面板"""
        panel = QTabWidget()
        
        # 技术分析标签页
        technical_tab = QWidget()
        technical_layout = QVBoxLayout()
        
        # 技术指标选择
        indicators_group = QGroupBox("技术指标")
        indicators_layout = QGridLayout()
        
        self.ma_check = QCheckBox("移动平均线")
        self.macd_check = QCheckBox("MACD")
        self.rsi_check = QCheckBox("RSI")
        self.bollinger_check = QCheckBox("布林带")
        self.volume_check = QCheckBox("成交量")
        
        indicators_layout.addWidget(self.ma_check, 0, 0)
        indicators_layout.addWidget(self.macd_check, 0, 1)
        indicators_layout.addWidget(self.rsi_check, 1, 0)
        indicators_layout.addWidget(self.bollinger_check, 1, 1)
        indicators_layout.addWidget(self.volume_check, 2, 0)
        
        indicators_group.setLayout(indicators_layout)
        technical_layout.addWidget(indicators_group)
        
        # 图表显示区域
        self.technical_plot = PlotWidget()
        technical_layout.addWidget(self.technical_plot)
        
        technical_tab.setLayout(technical_layout)
        panel.addTab(technical_tab, "技术分析")
        
        # 基本面分析标签页
        fundamental_tab = QWidget()
        fundamental_layout = QVBoxLayout()
        
        # 基本面指标选择
        fundamental_group = QGroupBox("基本面指标")
        fundamental_layout_grid = QGridLayout()
        
        self.pe_check = QCheckBox("市盈率")
        self.pb_check = QCheckBox("市净率")
        self.roe_check = QCheckBox("净资产收益率")
        self.revenue_check = QCheckBox("营收增长")
        self.profit_check = QCheckBox("利润增长")
        
        fundamental_layout_grid.addWidget(self.pe_check, 0, 0)
        fundamental_layout_grid.addWidget(self.pb_check, 0, 1)
        fundamental_layout_grid.addWidget(self.roe_check, 1, 0)
        fundamental_layout_grid.addWidget(self.revenue_check, 1, 1)
        fundamental_layout_grid.addWidget(self.profit_check, 2, 0)
        
        fundamental_group.setLayout(fundamental_layout_grid)
        fundamental_layout.addWidget(fundamental_group)
        
        # 基本面数据表格
        self.fundamental_table = QTableWidget()
        fundamental_layout.addWidget(self.fundamental_table)
        
        fundamental_tab.setLayout(fundamental_layout)
        panel.addTab(fundamental_tab, "基本面分析")
        
        # 量子分析标签页
        quantum_tab = QWidget()
        quantum_layout = QVBoxLayout()
        
        # 量子分析参数
        quantum_group = QGroupBox("量子分析参数")
        quantum_layout_grid = QGridLayout()
        
        self.qubits_label = QLabel("量子比特数:")
        self.qubits_spin = QSpinBox()
        self.qubits_spin.setRange(1, 100)
        self.qubits_spin.setValue(24)
        
        self.depth_label = QLabel("电路深度:")
        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(1, 1000)
        self.depth_spin.setValue(100)
        
        self.shots_label = QLabel("测量次数:")
        self.shots_spin = QSpinBox()
        self.shots_spin.setRange(100, 10000)
        self.shots_spin.setValue(1000)
        
        quantum_layout_grid.addWidget(self.qubits_label, 0, 0)
        quantum_layout_grid.addWidget(self.qubits_spin, 0, 1)
        quantum_layout_grid.addWidget(self.depth_label, 1, 0)
        quantum_layout_grid.addWidget(self.depth_spin, 1, 1)
        quantum_layout_grid.addWidget(self.shots_label, 2, 0)
        quantum_layout_grid.addWidget(self.shots_spin, 2, 1)
        
        quantum_group.setLayout(quantum_layout_grid)
        quantum_layout.addWidget(quantum_group)
        
        # 量子分析算法选择
        self.quantum_pca_check = QCheckBox("量子PCA分析")
        self.quantum_cluster_check = QCheckBox("量子聚类分析")
        self.quantum_anomaly_check = QCheckBox("量子异常检测")
        self.quantum_trend_check = QCheckBox("量子趋势分析")
        self.quantum_entanglement_check = QCheckBox("量子纠缠分析")
        self.quantum_phase_check = QCheckBox("量子相位分析")
        self.quantum_amplitude_check = QCheckBox("量子振幅分析")
        
        # 添加量子机器学习模型选项
        self.quantum_ml_check = QCheckBox("量子机器学习模型")
        self.quantum_nn_check = QCheckBox("量子神经网络")
        self.quantum_ga_check = QCheckBox("量子遗传算法")
        
        quantum_layout_grid.addWidget(self.quantum_pca_check, 3, 0)
        quantum_layout_grid.addWidget(self.quantum_cluster_check, 3, 1)
        quantum_layout_grid.addWidget(self.quantum_anomaly_check, 4, 0)
        quantum_layout_grid.addWidget(self.quantum_trend_check, 4, 1)
        quantum_layout_grid.addWidget(self.quantum_entanglement_check, 5, 0)
        quantum_layout_grid.addWidget(self.quantum_phase_check, 5, 1)
        quantum_layout_grid.addWidget(self.quantum_amplitude_check, 6, 0)
        quantum_layout_grid.addWidget(self.quantum_ml_check, 6, 1)
        quantum_layout_grid.addWidget(self.quantum_nn_check, 7, 0)
        quantum_layout_grid.addWidget(self.quantum_ga_check, 7, 1)
        
        # 量子机器学习参数
        ml_params_group = QGroupBox("量子机器学习参数")
        ml_params_layout = QGridLayout()
        
        # 模型类型选择
        self.ml_model_type = QComboBox()
        self.ml_model_type.addItems(["分类模型", "回归模型", "聚类模型", "异常检测模型"])
        ml_params_layout.addWidget(QLabel("模型类型:"), 0, 0)
        ml_params_layout.addWidget(self.ml_model_type, 0, 1)
        
        # 训练轮数
        self.ml_epochs_spin = QSpinBox()
        self.ml_epochs_spin.setRange(10, 1000)
        self.ml_epochs_spin.setValue(100)
        ml_params_layout.addWidget(QLabel("训练轮数:"), 1, 0)
        ml_params_layout.addWidget(self.ml_epochs_spin, 1, 1)
        
        # 学习率
        self.ml_lr_spin = QDoubleSpinBox()
        self.ml_lr_spin.setRange(0.0001, 0.1)
        self.ml_lr_spin.setValue(0.01)
        self.ml_lr_spin.setSingleStep(0.001)
        self.ml_lr_spin.setDecimals(4)
        ml_params_layout.addWidget(QLabel("学习率:"), 2, 0)
        ml_params_layout.addWidget(self.ml_lr_spin, 2, 1)
        
        ml_params_group.setLayout(ml_params_layout)
        ml_params_group.setVisible(False)
        
        # 量子神经网络参数
        nn_params_group = QGroupBox("量子神经网络参数")
        nn_params_layout = QGridLayout()
        
        # 网络层数
        self.nn_layers_spin = QSpinBox()
        self.nn_layers_spin.setRange(1, 10)
        self.nn_layers_spin.setValue(3)
        nn_params_layout.addWidget(QLabel("网络层数:"), 0, 0)
        nn_params_layout.addWidget(self.nn_layers_spin, 0, 1)
        
        # 每层神经元数
        self.nn_neurons_spin = QSpinBox()
        self.nn_neurons_spin.setRange(1, 100)
        self.nn_neurons_spin.setValue(10)
        nn_params_layout.addWidget(QLabel("每层神经元数:"), 1, 0)
        nn_params_layout.addWidget(self.nn_neurons_spin, 1, 1)
        
        # 激活函数
        self.nn_activation = QComboBox()
        self.nn_activation.addItems(["ReLU", "Sigmoid", "Tanh", "Softmax"])
        nn_params_layout.addWidget(QLabel("激活函数:"), 2, 0)
        nn_params_layout.addWidget(self.nn_activation, 2, 1)
        
        nn_params_group.setLayout(nn_params_layout)
        nn_params_group.setVisible(False)
        
        # 量子遗传算法参数
        ga_params_group = QGroupBox("量子遗传算法参数")
        ga_params_layout = QGridLayout()
        
        # 种群大小
        self.ga_population_spin = QSpinBox()
        self.ga_population_spin.setRange(10, 1000)
        self.ga_population_spin.setValue(100)
        ga_params_layout.addWidget(QLabel("种群大小:"), 0, 0)
        ga_params_layout.addWidget(self.ga_population_spin, 0, 1)
        
        # 迭代次数
        self.ga_iterations_spin = QSpinBox()
        self.ga_iterations_spin.setRange(10, 1000)
        self.ga_iterations_spin.setValue(100)
        ga_params_layout.addWidget(QLabel("迭代次数:"), 1, 0)
        ga_params_layout.addWidget(self.ga_iterations_spin, 1, 1)
        
        # 变异率
        self.ga_mutation_spin = QDoubleSpinBox()
        self.ga_mutation_spin.setRange(0.01, 0.5)
        self.ga_mutation_spin.setValue(0.1)
        self.ga_mutation_spin.setSingleStep(0.01)
        self.ga_mutation_spin.setDecimals(2)
        ga_params_layout.addWidget(QLabel("变异率:"), 2, 0)
        ga_params_layout.addWidget(self.ga_mutation_spin, 2, 1)
        
        ga_params_group.setLayout(ga_params_layout)
        ga_params_group.setVisible(False)
        
        # 连接信号
        self.quantum_ml_check.stateChanged.connect(lambda state: ml_params_group.setVisible(state == Qt.Checked))
        self.quantum_nn_check.stateChanged.connect(lambda state: nn_params_group.setVisible(state == Qt.Checked))
        self.quantum_ga_check.stateChanged.connect(lambda state: ga_params_group.setVisible(state == Qt.Checked))
        
        quantum_group.setLayout(quantum_layout_grid)
        quantum_layout.addWidget(quantum_group)
        layout.addWidget(quantum_layout)
        
        # 量子分析图表
        self.quantum_plot = pg.GraphicsLayoutWidget()
        self.quantum_plot.setBackground('w')
        layout.addWidget(self.quantum_plot)
        
        # 分析结果报告
        self.quantum_report = QTextEdit()
        self.quantum_report.setReadOnly(True)
        layout.addWidget(self.quantum_report)
        
        # 报告操作按钮
        report_buttons_layout = QHBoxLayout()
        
        self.export_chart_button = QPushButton("导出图表")
        self.export_chart_button.clicked.connect(self._export_chart)
        report_buttons_layout.addWidget(self.export_chart_button)
        
        self.export_report_button = QPushButton("导出报告")
        self.export_report_button.clicked.connect(self._export_report)
        report_buttons_layout.addWidget(self.export_report_button)
        
        self.apply_template_button = QPushButton("应用模板")
        self.apply_template_button.clicked.connect(self._apply_report_template)
        report_buttons_layout.addWidget(self.apply_template_button)
        
        layout.addLayout(report_buttons_layout)
        
        panel.setLayout(layout)
        return panel
        
    def _create_result_panel(self) -> QWidget:
        """创建结果面板"""
        panel = QGroupBox("分析结果")
        layout = QVBoxLayout()
        
        # 结果摘要
        summary_group = QGroupBox("结果摘要")
        summary_layout = QFormLayout()
        
        self.trend_label = QLabel()
        self.score_label = QLabel()
        self.risk_label = QLabel()
        self.recommendation_label = QLabel()
        
        summary_layout.addRow("趋势:", self.trend_label)
        summary_layout.addRow("得分:", self.score_label)
        summary_layout.addRow("风险:", self.risk_label)
        summary_layout.addRow("建议:", self.recommendation_label)
        
        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)
        
        # 详细结果
        details_group = QGroupBox("详细结果")
        details_layout = QVBoxLayout()
        
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        details_layout.addWidget(self.details_text)
        
        details_group.setLayout(details_layout)
        layout.addWidget(details_group)
        
        panel.setLayout(layout)
        return panel
        
    def _start_analysis(self):
        """开始分析"""
        try:
            # 获取输入参数
            stock_code = self.stock_input.text().strip()
            if not stock_code:
                QMessageBox.warning(self, "警告", "请输入股票代码")
                return
                
            time_range = self.time_range.currentText()
            analysis_type = self.analysis_type.currentText()
            
            # 显示进度条
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # 发送分析开始信号
            self.analysis_started.emit(stock_code)
            
            # 根据分析类型执行不同的分析
            if analysis_type == "技术分析":
                self._perform_technical_analysis(stock_code, time_range)
            elif analysis_type == "基本面分析":
                self._perform_fundamental_analysis(stock_code, time_range)
            elif analysis_type == "量子分析":
                self._perform_quantum_analysis(stock_code, time_range)
            elif analysis_type == "情绪分析":
                self._perform_sentiment_analysis(stock_code, time_range)
            else:  # 综合分析
                self._perform_comprehensive_analysis(stock_code, time_range)
                
        except Exception as e:
            self.logger.error(f"分析过程出错: {str(e)}")
            self.analysis_error.emit(stock_code, str(e))
            QMessageBox.critical(self, "错误", f"分析过程出错: {str(e)}")
        finally:
            self.progress_bar.setVisible(False)
            
    def _perform_technical_analysis(self, stock_code: str, time_range: str):
        """执行技术分析"""
        # 更新进度
        self.progress_bar.setValue(20)
        
        # 获取选中的技术指标
        indicators = []
        if self.ma_check.isChecked():
            indicators.append("MA")
        if self.macd_check.isChecked():
            indicators.append("MACD")
        if self.rsi_check.isChecked():
            indicators.append("RSI")
        if self.bollinger_check.isChecked():
            indicators.append("BOLL")
        if self.volume_check.isChecked():
            indicators.append("VOLUME")
            
        # 执行技术分析
        try:
            # 获取市场数据
            self.logger.info(f"获取股票 {stock_code} 的市场数据...")
            market_data = self._get_market_data(stock_code, time_range)
            
            # 更新进度
            self.progress_bar.setValue(40)
            
            # 更新图表
            self._update_technical_plot(market_data, indicators)
            
            # 生成分析结果
            analysis_result = {
                'type': 'technical',
                'indicators': indicators,
                'time_range': time_range,
                'data': market_data.to_dict() if not market_data.empty else {}
            }
            
            # 更新进度
            self.progress_bar.setValue(100)
            
            # 发送分析完成信号
            self.analysis_finished.emit(stock_code, analysis_result)
            
        except Exception as e:
            self.logger.error(f"技术分析出错: {str(e)}")
            self.analysis_error.emit(stock_code, str(e))
            raise
        
    def _perform_fundamental_analysis(self, stock_code: str, time_range: str):
        """执行基本面分析"""
        # 更新进度
        self.progress_bar.setValue(20)
        
        # 获取选中的基本面指标
        indicators = []
        if self.pe_check.isChecked():
            indicators.append("PE")
        if self.pb_check.isChecked():
            indicators.append("PB")
        if self.roe_check.isChecked():
            indicators.append("ROE")
        if self.revenue_check.isChecked():
            indicators.append("REVENUE")
        if self.profit_check.isChecked():
            indicators.append("PROFIT")
            
        # 执行基本面分析
        # TODO: 实现实际的基本面分析逻辑
        
        # 更新进度
        self.progress_bar.setValue(60)
        
        # 更新表格
        this._update_fundamental_table()
        
        # 更新进度
        this.progress_bar.setValue(100)
        
        # 发送分析完成信号
        this.analysis_finished.emit(stock_code, {
            'type': 'fundamental',
            'indicators': indicators,
            'time_range': time_range
        })
        
    def _perform_quantum_analysis(self, stock_code: str, time_range: str):
        """执行量子分析"""
        try:
            # 更新进度条
            self.progress_bar.setValue(30)
            
            # 获取市场数据
            market_data = self._get_market_data(stock_code, time_range)
            
            # 更新进度条
            self.progress_bar.setValue(50)
            
            # 初始化量子转换器和分析器
            converter = MarketToQuantumConverter()
            analyzer = MultidimensionalAnalyzer()
            
            # 将市场数据转换为量子态
            quantum_states = converter.encode_market_data(market_data)
            
            # 更新进度条
            self.progress_bar.setValue(70)
            
            # 执行量子分析
            analysis_results = {}
            
            if self.quantum_pca_check.isChecked():
                # 量子PCA分析
                pca_result = analyzer.quantum_pca(quantum_states)
                analysis_results['pca'] = pca_result
                
            if self.quantum_cluster_check.isChecked():
                # 量子聚类分析
                cluster_result = analyzer.quantum_clustering(quantum_states)
                analysis_results['cluster'] = cluster_result
                
            if self.quantum_anomaly_check.isChecked():
                # 量子异常检测
                anomaly_result = analyzer.quantum_anomaly_detection(quantum_states)
                analysis_results['anomaly'] = anomaly_result
                
            if self.quantum_trend_check.isChecked():
                # 量子趋势分析
                trend_result = analyzer.quantum_trend_analysis(quantum_states)
                analysis_results['trend'] = trend_result
                
            if self.quantum_entanglement_check.isChecked():
                # 量子纠缠分析
                entanglement_result = analyzer.quantum_entanglement_analysis(quantum_states)
                analysis_results['entanglement'] = entanglement_result
                
            if self.quantum_phase_check.isChecked():
                # 量子相位分析
                phase_result = analyzer.quantum_phase_analysis(quantum_states)
                analysis_results['phase'] = phase_result
                
            if self.quantum_amplitude_check.isChecked():
                # 量子振幅分析
                amplitude_result = analyzer.quantum_amplitude_analysis(quantum_states)
                analysis_results['amplitude'] = amplitude_result
                
            if self.quantum_ml_check.isChecked():
                # 量子机器学习模型
                ml_params = {
                    'model_type': self.ml_model_type.currentText(),
                    'epochs': self.ml_epochs_spin.value(),
                    'learning_rate': self.ml_lr_spin.value()
                }
                ml_result = analyzer.quantum_machine_learning(quantum_states, ml_params)
                analysis_results['ml'] = ml_result
                
            if self.quantum_nn_check.isChecked():
                # 量子神经网络
                nn_params = {
                    'layers': self.nn_layers_spin.value(),
                    'neurons_per_layer': self.nn_neurons_spin.value(),
                    'activation': self.nn_activation.currentText()
                }
                nn_result = analyzer.quantum_neural_network(quantum_states, nn_params)
                analysis_results['nn'] = nn_result
                
            if self.quantum_ga_check.isChecked():
                # 量子遗传算法
                ga_params = {
                    'population_size': self.ga_population_spin.value(),
                    'iterations': self.ga_iterations_spin.value(),
                    'mutation_rate': self.ga_mutation_spin.value()
                }
                ga_result = analyzer.quantum_genetic_algorithm(quantum_states, ga_params)
                analysis_results['ga'] = ga_result
                
            # 更新量子分析图表
            self._update_quantum_plot(market_data, analysis_results)
            
            # 生成分析报告
            self._generate_quantum_report(analysis_results)
            
            # 生成分析建议
            self._generate_analysis_recommendations(analysis_results)
            
            # 更新进度条
            self.progress_bar.setValue(100)
            
            # 发送分析完成信号
            this.analysis_finished.emit(stock_code, analysis_results)
            
        except Exception as e:
            this.logger.error(f"量子分析出错: {str(e)}")
            raise
            
    def _update_quantum_plot(self, market_data: pd.DataFrame, analysis_results: Dict):
        """更新量子分析图表"""
        # 清除现有图表
        this.quantum_plot.clear()
        
        # 创建子图布局
        layout = this.quantum_plot.ci.layout
        layout.setRowStretchFactor(0, 3)  # 价格图占3份
        layout.setRowStretchFactor(1, 2)  # 分析图占2份
        
        # 创建价格走势图
        price_plot = this.quantum_plot.addPlot(row=0, col=0)
        price_plot.setBackground('w')
        price_plot.showGrid(x=True, y=True, alpha=0.3)
        
        # 绘制价格走势
        price_plot.plot(market_data.index, market_data['close'], 
                       pen=pg.mkPen('b', width=2), name='价格')
        
        # 创建分析结果图
        analysis_plot = this.quantum_plot.addPlot(row=1, col=0)
        analysis_plot.setBackground('w')
        analysis_plot.showGrid(x=True, y=True, alpha=0.3)
        
        # 根据分析结果绘制不同的图表
        if 'pca' in analysis_results:
            # 绘制PCA结果
            pca_result = analysis_results['pca']
            analysis_plot.plot(pca_result['components'][:, 0], 
                             pen=pg.mkPen('r', width=2), name='主成分1')
            analysis_plot.plot(pca_result['components'][:, 1], 
                             pen=pg.mkPen('g', width=2), name='主成分2')
            
        if 'cluster' in analysis_results:
            # 绘制聚类结果
            cluster_result = analysis_results['cluster']
            for i, cluster in enumerate(cluster_result['clusters']):
                scatter = pg.ScatterPlotItem(
                    x=cluster['x'], y=cluster['y'],
                    pen=None, brush=pg.mkBrush(f'C{i}'),
                    size=10, name=f'聚类{i+1}'
                )
                price_plot.addItem(scatter)
                
        if 'anomaly' in analysis_results:
            # 绘制异常点
            anomaly_result = analysis_results['anomaly']
            anomalies = anomaly_result['anomalies']
            scatter = pg.ScatterPlotItem(
                x=anomalies['x'], y=anomalies['y'],
                pen=None, brush=pg.mkBrush('r'),
                size=15, symbol='x', name='异常点'
            )
            price_plot.addItem(scatter)
            
        if 'trend' in analysis_results:
            # 绘制趋势线
            trend_result = analysis_results['trend']
            price_plot.plot(trend_result['trend_line']['x'],
                          trend_result['trend_line']['y'],
                          pen=pg.mkPen('y', width=2, style=Qt.DashLine),
                          name='趋势线')
            
        if 'entanglement' in analysis_results:
            # 绘制纠缠分析结果
            entanglement_result = analysis_results['entanglement']
            analysis_plot.plot(entanglement_result['correlation'],
                             pen=pg.mkPen('m', width=2),
                             name='纠缠度')
            
        if 'phase' in analysis_results:
            # 绘制相位分析结果
            phase_result = analysis_results['phase']
            analysis_plot.plot(phase_result['phase'],
                             pen=pg.mkPen('c', width=2),
                             name='相位')
            
        if 'amplitude' in analysis_results:
            # 绘制振幅分析结果
            amplitude_result = analysis_results['amplitude']
            analysis_plot.plot(amplitude_result['amplitude'],
                             pen=pg.mkPen('k', width=2),
                             name='振幅')
            
        if 'ml' in analysis_results:
            # 绘制机器学习模型结果
            ml_result = analysis_results['ml']
            if 'predictions' in ml_result:
                analysis_plot.plot(ml_result['predictions'],
                                 pen=pg.mkPen('b', width=2, style=Qt.DashLine),
                                 name='预测值')
                
        if 'nn' in analysis_results:
            # 绘制神经网络结果
            nn_result = analysis_results['nn']
            if 'predictions' in nn_result:
                analysis_plot.plot(nn_result['predictions'],
                                 pen=pg.mkPen('g', width=2, style=Qt.DashLine),
                                 name='神经网络预测')
                
        if 'ga' in analysis_results:
            # 绘制遗传算法结果
            ga_result = analysis_results['ga']
            if 'best_solutions' in ga_result:
                analysis_plot.plot(ga_result['best_solutions'],
                                 pen=pg.mkPen('r', width=2, style=Qt.DashLine),
                                 name='最优解')
            
        # 设置图表标题和标签
        price_plot.setTitle('价格走势与量子分析', size='12pt')
        price_plot.setLabel('left', '价格', size='10pt')
        price_plot.setLabel('bottom', '时间', size='10pt')
        
        analysis_plot.setLabel('left', '量子特征', size='10pt')
        analysis_plot.setLabel('bottom', '时间', size='10pt')
        
        # 添加图例
        price_plot.addLegend(offset=(-10, 10))
        analysis_plot.addLegend(offset=(-10, 10))
        
        # 同步X轴范围
        price_plot.setXLink(analysis_plot)
        
    def _generate_quantum_report(self, analysis_results: Dict):
        """生成量子分析报告"""
        report = []
        report.append("量子分析报告")
        report.append("=" * 50)
        report.append("")
        
        # PCA分析结果
        if 'pca' in analysis_results:
            pca_result = analysis_results['pca']
            report.append("量子PCA分析")
            report.append("-" * 30)
            report.append(f"主成分数量: {pca_result['n_components']}")
            report.append(f"解释方差比: {pca_result['explained_variance_ratio']}")
            report.append(f"累计方差比: {pca_result['cumulative_variance_ratio']}")
            report.append("")
            
        # 聚类分析结果
        if 'cluster' in analysis_results:
            cluster_result = analysis_results['cluster']
            report.append("量子聚类分析")
            report.append("-" * 30)
            report.append(f"聚类数量: {cluster_result['n_clusters']}")
            report.append(f"聚类中心: {cluster_result['centroids']}")
            report.append(f"聚类标签: {cluster_result['labels']}")
            report.append("")
            
        # 异常检测结果
        if 'anomaly' in analysis_results:
            anomaly_result = analysis_results['anomaly']
            report.append("量子异常检测")
            report.append("-" * 30)
            report.append(f"异常点数量: {len(anomaly_result['anomalies'])}")
            report.append(f"异常分数: {anomaly_result['anomaly_scores']}")
            report.append("")
            
        # 趋势分析结果
        if 'trend' in analysis_results:
            trend_result = analysis_results['trend']
            report.append("量子趋势分析")
            report.append("-" * 30)
            report.append(f"趋势方向: {trend_result['direction']}")
            report.append(f"趋势强度: {trend_result['strength']}")
            report.append(f"趋势持续时间: {trend_result['duration']}")
            report.append("")
            
        # 纠缠分析结果
        if 'entanglement' in analysis_results:
            entanglement_result = analysis_results['entanglement']
            report.append("量子纠缠分析")
            report.append("-" * 30)
            report.append(f"平均纠缠度: {entanglement_result['mean_entanglement']}")
            report.append(f"最大纠缠度: {entanglement_result['max_entanglement']}")
            report.append(f"纠缠模式: {entanglement_result['entanglement_pattern']}")
            report.append("")
            
        # 相位分析结果
        if 'phase' in analysis_results:
            phase_result = analysis_results['phase']
            report.append("量子相位分析")
            report.append("-" * 30)
            report.append(f"相位变化: {phase_result['phase_changes']}")
            report.append(f"相位稳定性: {phase_result['phase_stability']}")
            report.append("")
            
        # 振幅分析结果
        if 'amplitude' in analysis_results:
            amplitude_result = analysis_results['amplitude']
            report.append("量子振幅分析")
            report.append("-" * 30)
            report.append(f"振幅范围: {amplitude_result['amplitude_range']}")
            report.append(f"振幅稳定性: {amplitude_result['amplitude_stability']}")
            report.append("")
            
        # 量子机器学习结果
        if 'ml' in analysis_results:
            ml_result = analysis_results['ml']
            report.append("量子机器学习模型")
            report.append("-" * 30)
            report.append(f"模型类型: {ml_result['model_type']}")
            report.append(f"训练轮数: {ml_result['epochs']}")
            report.append(f"学习率: {ml_result['learning_rate']}")
            report.append(f"准确率: {ml_result['accuracy']}")
            report.append(f"损失值: {ml_result['loss']}")
            report.append("")
            
        # 量子神经网络结果
        if 'nn' in analysis_results:
            nn_result = analysis_results['nn']
            report.append("量子神经网络")
            report.append("-" * 30)
            report.append(f"网络层数: {nn_result['layers']}")
            report.append(f"每层神经元数: {nn_result['neurons_per_layer']}")
            report.append(f"激活函数: {nn_result['activation']}")
            report.append(f"准确率: {nn_result['accuracy']}")
            report.append(f"损失值: {nn_result['loss']}")
            report.append("")
            
        # 量子遗传算法结果
        if 'ga' in analysis_results:
            ga_result = analysis_results['ga']
            report.append("量子遗传算法")
            report.append("-" * 30)
            report.append(f"种群大小: {ga_result['population_size']}")
            report.append(f"迭代次数: {ga_result['iterations']}")
            report.append(f"变异率: {ga_result['mutation_rate']}")
            report.append(f"适应度: {ga_result['fitness']}")
            report.append(f"最优解: {ga_result['best_solution']}")
            report.append("")
            
        # 更新报告文本
        this.quantum_report.setText("\n".join(report))
        
    def _generate_analysis_recommendations(self, analysis_results: Dict):
        """生成分析建议"""
        recommendations = []
        recommendations.append("分析建议")
        recommendations.append("=" * 50)
        recommendations.append("")
        
        # 基于趋势分析的建议
        if 'trend' in analysis_results:
            trend_result = analysis_results['trend']
            direction = trend_result['direction']
            strength = trend_result['strength']
            
            if direction == "上升" and strength > 0.7:
                recommendations.append("趋势分析建议:")
                recommendations.append("- 当前处于强势上升趋势，可考虑持有或加仓")
                recommendations.append("- 关注成交量配合情况，确认趋势有效性")
                recommendations.append("")
            elif direction == "下降" and strength > 0.7:
                recommendations.append("趋势分析建议:")
                recommendations.append("- 当前处于强势下降趋势，建议谨慎操作")
                recommendations.append("- 可考虑设置止损，控制风险")
                recommendations.append("")
            elif direction == "震荡":
                recommendations.append("趋势分析建议:")
                recommendations.append("- 当前处于震荡整理阶段，建议观望")
                recommendations.append("- 可等待明确方向后再做决策")
                recommendations.append("")
                
        # 基于异常检测的建议
        if 'anomaly' in analysis_results:
            anomaly_result = analysis_results['anomaly']
            anomaly_count = len(anomaly_result['anomalies'])
            
            if anomaly_count > 5:
                recommendations.append("异常检测建议:")
                recommendations.append("- 检测到多个异常点，市场可能存在不稳定因素")
                recommendations.append("- 建议关注这些异常点前后的市场变化")
                recommendations.append("")
                
        # 基于量子机器学习的建议
        if 'ml' in analysis_results:
            ml_result = analysis_results['ml']
            accuracy = ml_result['accuracy']
            
            if accuracy > 0.8:
                recommendations.append("机器学习模型建议:")
                recommendations.append("- 模型预测准确率较高，可参考其预测结果")
                recommendations.append("- 建议结合其他分析方法，综合判断")
                recommendations.append("")
            elif accuracy < 0.6:
                recommendations.append("机器学习模型建议:")
                recommendations.append("- 模型预测准确率较低，建议谨慎参考")
                recommendations.append("- 可尝试调整模型参数或使用其他分析方法")
                recommendations.append("")
                
        # 基于量子神经网络的建议
        if 'nn' in analysis_results:
            nn_result = analysis_results['nn']
            accuracy = nn_result['accuracy']
            
            if accuracy > 0.8:
                recommendations.append("神经网络模型建议:")
                recommendations.append("- 神经网络预测准确率较高，可参考其预测结果")
                recommendations.append("- 建议关注模型识别的关键特征")
                recommendations.append("")
                
        # 基于量子遗传算法的建议
        if 'ga' in analysis_results:
            ga_result = analysis_results['ga']
            fitness = ga_result['fitness']
            
            if fitness > 0.8:
                recommendations.append("遗传算法建议:")
                recommendations.append("- 算法找到的解决方案质量较高")
                recommendations.append("- 建议参考算法优化的参数设置")
                recommendations.append("")
                
        # 综合建议
        recommendations.append("综合建议:")
        recommendations.append("- 建议结合多种分析方法，不要依赖单一指标")
        recommendations.append("- 关注市场整体环境和行业趋势")
        recommendations.append("- 设置合理的止损和止盈点，控制风险")
        recommendations.append("- 定期回顾和调整分析策略")
        recommendations.append("")
        
        # 将建议添加到报告中
        current_report = this.quantum_report.toPlainText()
        this.quantum_report.setText(current_report + "\n" + "\n".join(recommendations))
        
    def _export_chart(self):
        """导出图表"""
        try:
            # 获取保存路径
            file_path, _ = QFileDialog.getSaveFileName(
                this, "导出图表", "", "PNG图片 (*.png);;JPEG图片 (*.jpg);;所有文件 (*.*)"
            )
            
            if not file_path:
                return
                
            # 导出图表
            exporter = pg.exporters.ImageExporter(this.quantum_plot.scene())
            exporter.export(file_path)
            
            this.logger.info(f"图表已导出到: {file_path}")
            QMessageBox.information(this, "成功", f"图表已成功导出到: {file_path}")
            
        except Exception as e:
            this.logger.error(f"导出图表出错: {str(e)}")
            QMessageBox.critical(this, "错误", f"导出图表出错: {str(e)}")
            
    def _export_report(self):
        """导出报告"""
        try:
            # 获取保存路径
            file_path, _ = QFileDialog.getSaveFileName(
                this, "导出报告", "", "文本文件 (*.txt);;HTML文件 (*.html);;所有文件 (*.*)"
            )
            
            if not file_path:
                return
                
            # 获取报告内容
            report_content = this.quantum_report.toPlainText()
            
            # 根据文件类型导出
            if file_path.endswith('.html'):
                # 导出为HTML
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="UTF-8">
                    <title>量子分析报告</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1 {{ color: #333; }}
                        h2 {{ color: #666; border-bottom: 1px solid #ccc; }}
                        pre {{ white-space: pre-wrap; }}
                    </style>
                </head>
                <body>
                    <h1>量子分析报告</h1>
                    <pre>{report_content}</pre>
                </body>
                </html>
                """
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
            else:
                # 导出为文本
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                    
            this.logger.info(f"报告已导出到: {file_path}")
            QMessageBox.information(this, "成功", f"报告已成功导出到: {file_path}")
            
        except Exception as e:
            this.logger.error(f"导出报告出错: {str(e)}")
            QMessageBox.critical(this, "错误", f"导出报告出错: {str(e)}")
            
    def _apply_report_template(self):
        """应用报告模板"""
        try:
            # 获取模板文件路径
            file_path, _ = QFileDialog.getOpenFileName(
                this, "选择报告模板", "", "文本文件 (*.txt);;HTML文件 (*.html);;所有文件 (*.*)"
            )
            
            if not file_path:
                return
                
            # 读取模板内容
            with open(file_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
                
            # 获取当前报告内容
            current_report = this.quantum_report.toPlainText()
            
            # 替换模板中的占位符
            template_content = template_content.replace("{{REPORT_CONTENT}}", current_report)
            
            # 更新报告
            this.quantum_report.setText(template_content)
            
            this.logger.info(f"已应用报告模板: {file_path}")
            QMessageBox.information(this, "成功", f"已成功应用报告模板: {file_path}")
            
        except Exception as e:
            this.logger.error(f"应用报告模板出错: {str(e)}")
            QMessageBox.critical(this, "错误", f"应用报告模板出错: {str(e)}")
            
    def _setup_real_time_update(self):
        """设置实时数据更新"""
        # 创建定时器
        this.update_timer = QTimer()
        this.update_timer.timeout.connect(this._update_real_time_data)
        
        # 设置更新间隔（毫秒）
        this.update_interval = 5000  # 5秒
        
        # 创建数据缓存
        this.data_cache = {}
        this.cache_size = 1000  # 缓存大小
        
        # 创建增量更新标志
        this.incremental_update = True
        
    def _start_real_time_update(self):
        """开始实时数据更新"""
        if not hasattr(this, 'update_timer'):
            this._setup_real_time_update()
        this.update_timer.start(this.update_interval)
        
    def _stop_real_time_update(self):
        """停止实时数据更新"""
        if hasattr(this, 'update_timer'):
            this.update_timer.stop()
            
    def _update_real_time_data(self):
        """更新实时数据"""
        try:
            # 获取当前选中的股票代码
            stock_code = this.stock_input.text().strip()
            if not stock_code:
                return
                
            # 获取最新的市场数据
            new_data = this._get_market_data(stock_code, "1天")
            
            # 检查是否需要增量更新
            if this.incremental_update and stock_code in this.data_cache:
                # 获取缓存数据
                cached_data = this.data_cache[stock_code]
                
                # 找出新增的数据点
                new_points = new_data[~new_data.index.isin(cached_data.index)]
                
                if len(new_points) > 0:
                    # 合并新旧数据
                    updated_data = pd.concat([cached_data, new_points])
                    
                    # 更新缓存
                    this.data_cache[stock_code] = updated_data
                    
                    # 更新图表和分析结果
                    this._update_quantum_plot(updated_data, this.analysis_result)
                    this._generate_quantum_report(this.analysis_result)
                    this._generate_analysis_recommendations(this.analysis_result)
                else:
                    # 没有新数据，不需要更新
                    pass
            else:
                # 全量更新
                this.data_cache[stock_code] = new_data
                
                # 更新图表和分析结果
                this._update_quantum_plot(new_data, this.analysis_result)
                this._generate_quantum_report(this.analysis_result)
                this._generate_analysis_recommendations(this.analysis_result)
                
            # 控制缓存大小
            if len(this.data_cache) > this.cache_size:
                # 删除最早的缓存
                oldest_key = next(iter(this.data_cache))
                del this.data_cache[oldest_key]
                
        except Exception as e:
            this.logger.error(f"实时数据更新出错: {str(e)}")
            
    def set_update_interval(self, interval_ms: int):
        """设置更新间隔
        
        Args:
            interval_ms: 更新间隔（毫秒）
        """
        this.update_interval = interval_ms
        if hasattr(this, 'update_timer') and this.update_timer.isActive():
            this.update_timer.setInterval(interval_ms)
            
    def set_incremental_update(self, enabled: bool):
        """设置是否启用增量更新
        
        Args:
            enabled: 是否启用增量更新
        """
        this.incremental_update = enabled
        
    def set_cache_size(self, size: int):
        """设置缓存大小
        
        Args:
            size: 缓存大小
        """
        this.cache_size = size
        
        # 如果当前缓存超过新大小，删除多余的缓存
        if len(this.data_cache) > size:
            # 删除最早的缓存
            keys_to_remove = list(this.data_cache.keys())[:len(this.data_cache) - size]
            for key in keys_to_remove:
                del this.data_cache[key]
                
    def closeEvent(self, event):
        """关闭事件处理"""
        # 停止实时更新
        this._stop_real_time_update()
        super().closeEvent(event)

    def _update_fundamental_table(self):
        """更新基本面分析表格"""
        # 清除现有表格
        this.fundamental_table.clear()
        
        # 创建新的表格
        # TODO: 实现实际的表格更新逻辑
        
    def _update_sentiment_plot(self):
        """更新情绪分析图表"""
        # 清除现有图表
        this.sentiment_plot.clear()
        
        # 创建新的图表
        # TODO: 实现实际的图表更新逻辑
        
    def update_analysis_results(self, results: Dict[str, Any]):
        """更新分析结果
        
        Args:
            results: 分析结果字典
        """
        # 更新趋势标签
        trend = results.get('trend', '未知')
        self.trend_label.setText(trend)
        
        # 更新得分标签
        score = results.get('score', 0)
        self.score_label.setText(f"{score:.2f}")
        
        # 更新风险标签
        risk = results.get('risk', '未知')
        self.risk_label.setText(risk)
        
        # 更新建议标签
        recommendation = results.get('recommendation', '无')
        self.recommendation_label.setText(recommendation)
        
        # 更新详细结果
        details = results.get('details', '')
        self.details_text.setText(details)
        
    def clear_results(self):
        """清除分析结果"""
        # 清除标签
        self.trend_label.setText("")
        self.score_label.setText("")
        self.risk_label.setText("")
        self.recommendation_label.setText("")
        
        # 清除详细结果
        self.details_text.clear()
        
        # 清除图表
        self.technical_plot.clear()
        self.quantum_plot.clear()
        this.sentiment_plot.clear()
        
        # 清除表格
        this.fundamental_table.clear()
        
    def set_dark_theme(self):
        """设置暗色主题"""
        # 设置背景色
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        
        self.setPalette(palette)
        
        # 设置图表样式
        this.technical_plot.setBackground('k')
        this.quantum_plot.setBackground('k')
        this.sentiment_plot.setBackground('k')
        
        # 设置文本颜色
        for widget in this.findChildren(QWidget):
            if isinstance(widget, (QLabel, QTextEdit)):
                widget.setStyleSheet("color: white;")
                
    def set_light_theme(self):
        """设置亮色主题"""
        # 设置背景色
        palette = QPalette()
        palette.setColor(QPalette.Window, Qt.white)
        palette.setColor(QPalette.WindowText, Qt.black)
        palette.setColor(QPalette.Base, Qt.white)
        palette.setColor(QPalette.AlternateBase, QColor(240, 240, 240))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.black)
        palette.setColor(QPalette.Text, Qt.black)
        palette.setColor(QPalette.Button, QColor(240, 240, 240))
        palette.setColor(QPalette.ButtonText, Qt.black)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(0, 0, 255))
        palette.setColor(QPalette.Highlight, QColor(0, 120, 215))
        palette.setColor(QPalette.HighlightedText, Qt.white)
        
        self.setPalette(palette)
        
        # 设置图表样式
        this.technical_plot.setBackground('w')
        this.quantum_plot.setBackground('w')
        this.sentiment_plot.setBackground('w')
        
        # 设置文本颜色
        for widget in this.findChildren(QWidget):
            if isinstance(widget, (QLabel, QTextEdit)):
                widget.setStyleSheet("color: black;")
                
    def _get_market_data(self, stock_code: str, time_range: str) -> pd.DataFrame:
        """获取市场数据
        
        Args:
            stock_code: 股票代码
            time_range: 时间范围
            
        Returns:
            市场数据DataFrame
        """
        # 根据时间范围确定开始和结束日期
        end_date = datetime.now()
        if time_range == "1天":
            start_date = end_date - timedelta(days=1)
        elif time_range == "1周":
            start_date = end_date - timedelta(weeks=1)
        elif time_range == "1月":
            start_date = end_date - timedelta(days=30)
        elif time_range == "3月":
            start_date = end_date - timedelta(days=90)
        elif time_range == "6月":
            start_date = end_date - timedelta(days=180)
        elif time_range == "1年":
            start_date = end_date - timedelta(days=365)
        else:
            start_date = end_date - timedelta(days=30)  # 默认1个月
            
        # 格式化日期
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # 获取市场数据
        try:
            # 导入tushare
            import tushare as ts
            
            # 设置tushare token
            ts.set_token('your_tushare_token')  # 请替换为实际的token
            pro = ts.pro_api()
            
            # 获取日线数据
            df = pro.daily(ts_code=stock_code, start_date=start_date_str, end_date=end_date_str)
            
            # 按日期排序
            df = df.sort_values('trade_date')
            
            # 提取需要的列
            market_data = df[['trade_date', 'open', 'high', 'low', 'close', 'vol', 'amount']]
            
            # 计算技术指标
            market_data['ma5'] = market_data['close'].rolling(window=5).mean()
            market_data['ma10'] = market_data['close'].rolling(window=10).mean()
            market_data['ma20'] = market_data['close'].rolling(window=20).mean()
            
            # 计算MACD
            exp1 = market_data['close'].ewm(span=12, adjust=False).mean()
            exp2 = market_data['close'].ewm(span=26, adjust=False).mean()
            market_data['macd'] = exp1 - exp2
            market_data['signal'] = market_data['macd'].ewm(span=9, adjust=False).mean()
            market_data['hist'] = market_data['macd'] - market_data['signal']
            
            # 计算RSI
            delta = market_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            market_data['rsi'] = 100 - (100 / (1 + rs))
            
            # 计算布林带
            market_data['boll_mid'] = market_data['close'].rolling(window=20).mean()
            market_data['boll_std'] = market_data['close'].rolling(window=20).std()
            market_data['boll_upper'] = market_data['boll_mid'] + 2 * market_data['boll_std']
            market_data['boll_lower'] = market_data['boll_mid'] - 2 * market_data['boll_std']
            
            # 删除NaN值
            market_data = market_data.dropna()
            
            return market_data
            
        except Exception as e:
            this.logger.error(f"获取市场数据出错: {str(e)}")
            # 返回模拟数据
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            market_data = pd.DataFrame({
                'trade_date': [d.strftime('%Y%m%d') for d in dates],
                'open': np.random.normal(100, 5, len(dates)),
                'high': np.random.normal(105, 5, len(dates)),
                'low': np.random.normal(95, 5, len(dates)),
                'close': np.random.normal(100, 5, len(dates)),
                'vol': np.random.normal(1000000, 200000, len(dates)),
                'amount': np.random.normal(100000000, 20000000, len(dates))
            })
            
            # 确保high >= open, close, low
            for i in range(len(market_data)):
                market_data.loc[i, 'high'] = max(market_data.loc[i, 'open'], 
                                               market_data.loc[i, 'close'], 
                                               market_data.loc[i, 'high'])
                market_data.loc[i, 'low'] = min(market_data.loc[i, 'open'], 
                                              market_data.loc[i, 'close'], 
                                              market_data.loc[i, 'low'])
                
            # 计算技术指标
            market_data['ma5'] = market_data['close'].rolling(window=5).mean()
            market_data['ma10'] = market_data['close'].rolling(window=10).mean()
            market_data['ma20'] = market_data['close'].rolling(window=20).mean()
            
            # 计算MACD
            exp1 = market_data['close'].ewm(span=12, adjust=False).mean()
            exp2 = market_data['close'].ewm(span=26, adjust=False).mean()
            market_data['macd'] = exp1 - exp2
            market_data['signal'] = market_data['macd'].ewm(span=9, adjust=False).mean()
            market_data['hist'] = market_data['macd'] - market_data['signal']
            
            # 计算RSI
            delta = market_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            market_data['rsi'] = 100 - (100 / (1 + rs))
            
            # 计算布林带
            market_data['boll_mid'] = market_data['close'].rolling(window=20).mean()
            market_data['boll_std'] = market_data['close'].rolling(window=20).std()
            market_data['boll_upper'] = market_data['boll_mid'] + 2 * market_data['boll_std']
            market_data['boll_lower'] = market_data['boll_mid'] - 2 * market_data['boll_std']
            
            # 删除NaN值
            market_data = market_data.dropna()
            
            return market_data

class CandlestickItem(pg.GraphicsObject):
    """K线图项"""
    
    def __init__(self, data):
        pg.GraphicsObject.__init__(self)
        self.data = data
        self.generatePicture()
        
    def generatePicture(self):
        self.picture = QPicture()
        p = QPainter(self.picture)
        
        w = 0.4
        for i in range(len(self.data)):
            t = i
            open = self.data['open'].iloc[i]
            high = self.data['high'].iloc[i]
            low = self.data['low'].iloc[i]
            close = self.data['close'].iloc[i]
            
            if open > close:
                p.setPen(pg.mkPen('r'))
                p.setBrush(pg.mkBrush('r'))
            else:
                p.setPen(pg.mkPen('g'))
                p.setBrush(pg.mkBrush('g'))
                
            p.drawLine(QtCore.QPointF(t, low), QtCore.QPointF(t, high))
            p.drawRect(QtCore.QRectF(t-w, open, w*2, close-open))
            
        p.end()
        
    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)
        
    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect()) 