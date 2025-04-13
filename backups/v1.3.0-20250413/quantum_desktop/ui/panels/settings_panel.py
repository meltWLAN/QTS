"""
设置面板 - 配置系统参数和首选项
"""

import logging
import os
import json
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                         QPushButton, QLineEdit, QFrame, QGridLayout, 
                         QTabWidget, QCheckBox, QSpinBox, QDoubleSpinBox,
                         QComboBox, QFileDialog, QScrollArea, QGroupBox)
from PyQt5.QtCore import Qt, pyqtSlot, QSettings

logger = logging.getLogger("QuantumDesktop.SettingsPanel")

class SettingItem(QFrame):
    """表示单个设置项的组件"""
    
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        
        # 主布局
        self.layout = QHBoxLayout(self)
        
        # 标题标签
        self.title_label = QLabel(title)
        self.title_label.setMinimumWidth(150)
        self.layout.addWidget(self.title_label)
        
        # 控件容器
        self.widget_container = QHBoxLayout()
        self.layout.addLayout(self.widget_container)
        
        # 添加伸缩空间
        self.layout.addStretch(1)
        
    def add_widget(self, widget):
        """添加控件到设置项"""
        self.widget_container.addWidget(widget)
        return widget
        
class SettingsPanel(QWidget):
    """设置面板 - 配置系统参数和首选项"""
    
    def __init__(self, system_manager, parent=None):
        super().__init__(parent)
        self.system_manager = system_manager
        
        # 设置数据
        self.settings = QSettings("QuantumTradingSystem", "QuantumDesktop")
        
        # 初始化UI
        self._init_ui()
        
        # 加载设置
        self._load_settings()
        
        logger.info("设置面板初始化完成")
        
    def _init_ui(self):
        """初始化用户界面"""
        # 主布局
        self.main_layout = QVBoxLayout(self)
        
        # 标签页组件
        self.tab_widget = QTabWidget()
        self.main_layout.addWidget(self.tab_widget)
        
        # 创建设置页
        self._create_general_tab()
        self._create_quantum_tab()
        self._create_data_tab()
        self._create_advanced_tab()
        
        # 底部按钮布局
        self.button_layout = QHBoxLayout()
        self.main_layout.addLayout(self.button_layout)
        
        # 添加弹性空间
        self.button_layout.addStretch(1)
        
        # 添加按钮
        self.reset_button = QPushButton("重置为默认")
        self.reset_button.clicked.connect(self._reset_settings)
        self.button_layout.addWidget(self.reset_button)
        
        self.save_button = QPushButton("保存设置")
        self.save_button.clicked.connect(self._save_settings)
        self.button_layout.addWidget(self.save_button)
        
    def _create_general_tab(self):
        """创建通用设置标签页"""
        self.general_tab = QWidget()
        self.general_layout = QVBoxLayout(self.general_tab)
        
        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.general_layout.addWidget(scroll_area)
        
        # 创建内容容器
        content_widget = QWidget()
        scroll_area.setWidget(content_widget)
        content_layout = QVBoxLayout(content_widget)
        
        # 界面设置组
        ui_group = QGroupBox("界面设置")
        ui_group_layout = QVBoxLayout(ui_group)
        content_layout.addWidget(ui_group)
        
        # 主题选择
        theme_item = SettingItem("主题")
        self.theme_combo = theme_item.add_widget(QComboBox())
        self.theme_combo.addItems(["浅色", "深色", "系统默认"])
        ui_group_layout.addWidget(theme_item)
        
        # 字体大小
        font_size_item = SettingItem("字体大小")
        self.font_size_spin = font_size_item.add_widget(QSpinBox())
        self.font_size_spin.setRange(8, 20)
        self.font_size_spin.setValue(10)
        ui_group_layout.addWidget(font_size_item)
        
        # 启动设置组
        startup_group = QGroupBox("启动设置")
        startup_group_layout = QVBoxLayout(startup_group)
        content_layout.addWidget(startup_group)
        
        # 自动启动系统
        auto_start_item = SettingItem("启动时自动启动系统")
        self.auto_start_check = auto_start_item.add_widget(QCheckBox())
        startup_group_layout.addWidget(auto_start_item)
        
        # 记住上次布局
        remember_layout_item = SettingItem("记住上次布局")
        self.remember_layout_check = remember_layout_item.add_widget(QCheckBox())
        startup_group_layout.addWidget(remember_layout_item)
        
        # 添加标签页
        self.tab_widget.addTab(self.general_tab, "通用")
        
    def _create_quantum_tab(self):
        """创建量子设置标签页"""
        self.quantum_tab = QWidget()
        self.quantum_layout = QVBoxLayout(self.quantum_tab)
        
        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.quantum_layout.addWidget(scroll_area)
        
        # 创建内容容器
        content_widget = QWidget()
        scroll_area.setWidget(content_widget)
        content_layout = QVBoxLayout(content_widget)
        
        # 模拟器设置组
        simulator_group = QGroupBox("量子模拟器设置")
        simulator_group_layout = QVBoxLayout(simulator_group)
        content_layout.addWidget(simulator_group)
        
        # 量子比特数
        qubits_item = SettingItem("量子比特数")
        self.qubits_spin = qubits_item.add_widget(QSpinBox())
        self.qubits_spin.setRange(1, 30)
        self.qubits_spin.setValue(10)
        simulator_group_layout.addWidget(qubits_item)
        
        # 噪声模型
        noise_model_item = SettingItem("噪声模型")
        self.noise_model_combo = noise_model_item.add_widget(QComboBox())
        self.noise_model_combo.addItems(["无噪声", "去极化噪声", "振幅阻尼", "相位阻尼"])
        simulator_group_layout.addWidget(noise_model_item)
        
        # 噪声参数
        noise_param_item = SettingItem("噪声参数")
        self.noise_param_spin = noise_param_item.add_widget(QDoubleSpinBox())
        self.noise_param_spin.setRange(0.0, 1.0)
        self.noise_param_spin.setSingleStep(0.01)
        self.noise_param_spin.setValue(0.01)
        simulator_group_layout.addWidget(noise_param_item)
        
        # 优化设置组
        optimizer_group = QGroupBox("电路优化设置")
        optimizer_group_layout = QVBoxLayout(optimizer_group)
        content_layout.addWidget(optimizer_group)
        
        # 优化级别
        opt_level_item = SettingItem("优化级别")
        self.opt_level_combo = opt_level_item.add_widget(QComboBox())
        self.opt_level_combo.addItems(["0 (无优化)", "1 (轻度优化)", "2 (中度优化)", "3 (高度优化)"])
        optimizer_group_layout.addWidget(opt_level_item)
        
        # 启用反比特优化
        qubit_opt_item = SettingItem("启用反比特优化")
        self.qubit_opt_check = qubit_opt_item.add_widget(QCheckBox())
        optimizer_group_layout.addWidget(qubit_opt_item)
        
        # 添加标签页
        self.tab_widget.addTab(self.quantum_tab, "量子设置")
        
    def _create_data_tab(self):
        """创建数据设置标签页"""
        self.data_tab = QWidget()
        self.data_layout = QVBoxLayout(self.data_tab)
        
        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.data_layout.addWidget(scroll_area)
        
        # 创建内容容器
        content_widget = QWidget()
        scroll_area.setWidget(content_widget)
        content_layout = QVBoxLayout(content_widget)
        
        # 数据源设置组
        data_source_group = QGroupBox("数据源设置")
        data_source_layout = QVBoxLayout(data_source_group)
        content_layout.addWidget(data_source_group)
        
        # API密钥
        api_key_item = SettingItem("API密钥")
        self.api_key_edit = api_key_item.add_widget(QLineEdit())
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        data_source_layout.addWidget(api_key_item)
        
        # 数据源选择
        data_source_item = SettingItem("数据源")
        self.data_source_combo = data_source_item.add_widget(QComboBox())
        self.data_source_combo.addItems(["Alpha Vantage", "Yahoo Finance", "本地数据"])
        data_source_layout.addWidget(data_source_item)
        
        # 本地数据目录
        local_data_item = SettingItem("本地数据目录")
        self.local_data_edit = local_data_item.add_widget(QLineEdit())
        self.browse_button = QPushButton("浏览...")
        self.browse_button.clicked.connect(self._browse_data_dir)
        local_data_item.add_widget(self.browse_button)
        data_source_layout.addWidget(local_data_item)
        
        # 数据缓存设置组
        cache_group = QGroupBox("数据缓存设置")
        cache_layout = QVBoxLayout(cache_group)
        content_layout.addWidget(cache_group)
        
        # 启用数据缓存
        enable_cache_item = SettingItem("启用数据缓存")
        self.enable_cache_check = enable_cache_item.add_widget(QCheckBox())
        self.enable_cache_check.setChecked(True)
        cache_layout.addWidget(enable_cache_item)
        
        # 缓存过期时间
        cache_expire_item = SettingItem("缓存过期时间(小时)")
        self.cache_expire_spin = cache_expire_item.add_widget(QSpinBox())
        self.cache_expire_spin.setRange(1, 168)  # 1小时到7天
        self.cache_expire_spin.setValue(24)
        cache_layout.addWidget(cache_expire_item)
        
        # 添加标签页
        self.tab_widget.addTab(self.data_tab, "数据设置")
        
    def _create_advanced_tab(self):
        """创建高级设置标签页"""
        self.advanced_tab = QWidget()
        self.advanced_layout = QVBoxLayout(self.advanced_tab)
        
        # 创建滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.advanced_layout.addWidget(scroll_area)
        
        # 创建内容容器
        content_widget = QWidget()
        scroll_area.setWidget(content_widget)
        content_layout = QVBoxLayout(content_widget)
        
        # 系统设置组
        system_group = QGroupBox("系统设置")
        system_layout = QVBoxLayout(system_group)
        content_layout.addWidget(system_group)
        
        # 日志级别
        log_level_item = SettingItem("日志级别")
        self.log_level_combo = log_level_item.add_widget(QComboBox())
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        self.log_level_combo.setCurrentText("INFO")
        system_layout.addWidget(log_level_item)
        
        # 日志文件路径
        log_path_item = SettingItem("日志文件路径")
        self.log_path_edit = log_path_item.add_widget(QLineEdit())
        self.log_path_edit.setText("./logs")
        self.log_browse_button = QPushButton("浏览...")
        self.log_browse_button.clicked.connect(self._browse_log_dir)
        log_path_item.add_widget(self.log_browse_button)
        system_layout.addWidget(log_path_item)
        
        # 性能设置组
        performance_group = QGroupBox("性能设置")
        performance_layout = QVBoxLayout(performance_group)
        content_layout.addWidget(performance_group)
        
        # 线程数
        threads_item = SettingItem("最大线程数(0=自动)")
        self.threads_spin = threads_item.add_widget(QSpinBox())
        self.threads_spin.setRange(0, 32)
        self.threads_spin.setValue(0)
        performance_layout.addWidget(threads_item)
        
        # 内存限制
        memory_item = SettingItem("内存限制(MB)")
        self.memory_spin = memory_item.add_widget(QSpinBox())
        self.memory_spin.setRange(512, 16384)
        self.memory_spin.setValue(2048)
        performance_layout.addWidget(memory_item)
        
        # 添加标签页
        self.tab_widget.addTab(self.advanced_tab, "高级")
        
    def _browse_data_dir(self):
        """浏览数据目录"""
        directory = QFileDialog.getExistingDirectory(self, "选择数据目录", self.local_data_edit.text())
        if directory:
            self.local_data_edit.setText(directory)
            
    def _browse_log_dir(self):
        """浏览日志目录"""
        directory = QFileDialog.getExistingDirectory(self, "选择日志目录", self.log_path_edit.text())
        if directory:
            self.log_path_edit.setText(directory)
            
    def _save_settings(self):
        """保存设置"""
        try:
            # 保存通用设置
            self.settings.setValue("theme", self.theme_combo.currentText())
            self.settings.setValue("fontSize", self.font_size_spin.value())
            self.settings.setValue("autoStart", self.auto_start_check.isChecked())
            self.settings.setValue("rememberLayout", self.remember_layout_check.isChecked())
            
            # 保存量子设置
            self.settings.setValue("qubits", self.qubits_spin.value())
            self.settings.setValue("noiseModel", self.noise_model_combo.currentText())
            self.settings.setValue("noiseParam", self.noise_param_spin.value())
            self.settings.setValue("optimizationLevel", self.opt_level_combo.currentIndex())
            self.settings.setValue("qubitOptimization", self.qubit_opt_check.isChecked())
            
            # 保存数据设置
            self.settings.setValue("apiKey", self.api_key_edit.text())
            self.settings.setValue("dataSource", self.data_source_combo.currentText())
            self.settings.setValue("localDataDir", self.local_data_edit.text())
            self.settings.setValue("enableCache", self.enable_cache_check.isChecked())
            self.settings.setValue("cacheExpireHours", self.cache_expire_spin.value())
            
            # 保存高级设置
            self.settings.setValue("logLevel", self.log_level_combo.currentText())
            self.settings.setValue("logPath", self.log_path_edit.text())
            self.settings.setValue("maxThreads", self.threads_spin.value())
            self.settings.setValue("memoryLimit", self.memory_spin.value())
            
            # 同步设置到系统
            self._apply_settings_to_system()
            
            logger.info("设置已保存")
            
        except Exception as e:
            logger.error(f"保存设置时出错: {str(e)}")
            
    def _reset_settings(self):
        """重置设置为默认值"""
        # 通用设置
        self.theme_combo.setCurrentText("系统默认")
        self.font_size_spin.setValue(10)
        self.auto_start_check.setChecked(False)
        self.remember_layout_check.setChecked(True)
        
        # 量子设置
        self.qubits_spin.setValue(10)
        self.noise_model_combo.setCurrentText("无噪声")
        self.noise_param_spin.setValue(0.01)
        self.opt_level_combo.setCurrentIndex(1)
        self.qubit_opt_check.setChecked(True)
        
        # 数据设置
        self.api_key_edit.clear()
        self.data_source_combo.setCurrentText("Yahoo Finance")
        self.local_data_edit.setText("./data")
        self.enable_cache_check.setChecked(True)
        self.cache_expire_spin.setValue(24)
        
        # 高级设置
        self.log_level_combo.setCurrentText("INFO")
        self.log_path_edit.setText("./logs")
        self.threads_spin.setValue(0)
        self.memory_spin.setValue(2048)
        
    def _load_settings(self):
        """加载设置"""
        try:
            # 加载通用设置
            self.theme_combo.setCurrentText(self.settings.value("theme", "系统默认"))
            self.font_size_spin.setValue(int(self.settings.value("fontSize", 10)))
            self.auto_start_check.setChecked(self.settings.value("autoStart", False, type=bool))
            self.remember_layout_check.setChecked(self.settings.value("rememberLayout", True, type=bool))
            
            # 加载量子设置
            self.qubits_spin.setValue(int(self.settings.value("qubits", 10)))
            self.noise_model_combo.setCurrentText(self.settings.value("noiseModel", "无噪声"))
            self.noise_param_spin.setValue(float(self.settings.value("noiseParam", 0.01)))
            self.opt_level_combo.setCurrentIndex(int(self.settings.value("optimizationLevel", 1)))
            self.qubit_opt_check.setChecked(self.settings.value("qubitOptimization", True, type=bool))
            
            # 加载数据设置
            self.api_key_edit.setText(self.settings.value("apiKey", ""))
            self.data_source_combo.setCurrentText(self.settings.value("dataSource", "Yahoo Finance"))
            self.local_data_edit.setText(self.settings.value("localDataDir", "./data"))
            self.enable_cache_check.setChecked(self.settings.value("enableCache", True, type=bool))
            self.cache_expire_spin.setValue(int(self.settings.value("cacheExpireHours", 24)))
            
            # 加载高级设置
            self.log_level_combo.setCurrentText(self.settings.value("logLevel", "INFO"))
            self.log_path_edit.setText(self.settings.value("logPath", "./logs"))
            self.threads_spin.setValue(int(self.settings.value("maxThreads", 0)))
            self.memory_spin.setValue(int(self.settings.value("memoryLimit", 2048)))
            
            logger.info("设置已加载")
            
        except Exception as e:
            logger.error(f"加载设置时出错: {str(e)}")
            self._reset_settings()
            
    def _apply_settings_to_system(self):
        """应用设置到系统"""
        # 检查系统管理器是否可用
        if not self.system_manager:
            logger.warning("系统管理器不可用，无法应用设置")
            return
            
        try:
            # 创建设置字典
            config = {
                "general": {
                    "theme": self.theme_combo.currentText(),
                    "fontSize": self.font_size_spin.value(),
                    "autoStart": self.auto_start_check.isChecked(),
                    "rememberLayout": self.remember_layout_check.isChecked()
                },
                "quantum": {
                    "qubits": self.qubits_spin.value(),
                    "noiseModel": self.noise_model_combo.currentText(),
                    "noiseParam": self.noise_param_spin.value(),
                    "optimizationLevel": self.opt_level_combo.currentIndex(),
                    "qubitOptimization": self.qubit_opt_check.isChecked()
                },
                "data": {
                    "apiKey": self.api_key_edit.text(),
                    "dataSource": self.data_source_combo.currentText(),
                    "localDataDir": self.local_data_edit.text(),
                    "enableCache": self.enable_cache_check.isChecked(),
                    "cacheExpireHours": self.cache_expire_spin.value()
                },
                "advanced": {
                    "logLevel": self.log_level_combo.currentText(),
                    "logPath": self.log_path_edit.text(),
                    "maxThreads": self.threads_spin.value(),
                    "memoryLimit": self.memory_spin.value()
                }
            }
            
            # 更新系统配置
            self.system_manager.update_config(config)
            
            # 应用样式设置
            self._apply_ui_settings()
            
            logger.info("设置已应用到系统")
            
        except Exception as e:
            logger.error(f"应用设置到系统时出错: {str(e)}")
            
    def _apply_ui_settings(self):
        """应用UI设置"""
        # 应用主题
        theme = self.theme_combo.currentText()
        # 主题应用逻辑...
        
        # 应用字体大小
        font_size = self.font_size_spin.value()
        # 字体大小应用逻辑...
        
    def on_system_started(self):
        """系统启动时调用"""
        pass
        
    def on_system_stopped(self):
        """系统停止时调用"""
        pass 