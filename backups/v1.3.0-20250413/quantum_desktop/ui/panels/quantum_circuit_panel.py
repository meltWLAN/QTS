"""
量子电路面板 - 用于设计和执行量子电路
"""

import logging
import uuid
import time
import numpy as np
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                         QPushButton, QComboBox, QFrame, QGridLayout, 
                         QSpinBox, QTextEdit, QSplitter, QToolButton, 
                         QGraphicsView, QGraphicsScene, QGraphicsItem,
                         QMenu, QAction, QMessageBox, QLineEdit, QProgressDialog,
                         QFileDialog)
from PyQt5.QtCore import Qt, QRectF, QPointF, pyqtSlot, QSizeF
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPainterPath
from PyQt5.QtWidgets import QApplication
from datetime import datetime
from PyQt5.QtCore import QTimer

logger = logging.getLogger("QuantumDesktop.QuantumCircuitPanel")

class QuantumGateItem:
    """量子门项 - 表示电路中的一个量子门"""
    
    def __init__(self, gate_type, targets, params=None):
        self.gate_type = gate_type
        self.targets = targets if isinstance(targets, list) else [targets]
        self.params = params or {}
        self.id = str(uuid.uuid4())
        
    def to_dict(self):
        """转换为字典表示"""
        return {
            'type': self.gate_type,
            'targets': self.targets,
            'params': self.params,
            'id': self.id
        }
        
    @classmethod
    def from_dict(cls, data):
        """从字典创建实例"""
        gate = cls(
            data['type'],
            data['targets'],
            data.get('params', {})
        )
        gate.id = data.get('id', str(uuid.uuid4()))
        return gate
        
    def __str__(self):
        params_str = ""
        if self.params:
            params_str = f", params={self.params}"
            
        return f"{self.gate_type}(targets={self.targets}{params_str})"

class QuantumCircuitCanvas(QGraphicsView):
    """量子电路画布 - 可视化和编辑量子电路"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 设置场景
        self.scene = QGraphicsScene(self)
        self.scene.setSceneRect(0, 0, 800, 400)
        self.setScene(self.scene)
        
        # 设置视图属性
        self.setRenderHint(QPainter.Antialiasing)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        
        # 电路属性
        self.num_qubits = 3
        self.grid_size = 60
        self.gates = []
        
        # 绘制电路
        self._draw_circuit()
        
    def _draw_circuit(self):
        """绘制电路"""
        # 清除场景
        self.scene.clear()
        
        # 计算场景尺寸
        width = max(800, (len(self.gates) + 2) * self.grid_size)
        height = (self.num_qubits + 1) * self.grid_size
        
        # 设置场景矩形
        self.scene.setSceneRect(0, 0, width, height)
        
        # 绘制量子比特线
        for i in range(self.num_qubits):
            y = (i + 1) * self.grid_size
            
            # 量子比特标签
            text = self.scene.addText(f"|q{i}⟩")
            text.setPos(10, y - 10)
            
            # 量子比特线
            self.scene.addLine(
                self.grid_size, y, 
                width - self.grid_size / 2, y,
                QPen(Qt.gray, 1, Qt.SolidLine)
            )
            
        # 绘制量子门
        for i, gate in enumerate(self.gates):
            x = (i + 1) * self.grid_size
            self._draw_gate(gate, x)
            
    def _draw_gate(self, gate, x):
        """绘制量子门"""
        gate_colors = {
            'H': QColor(52, 152, 219),     # 蓝色
            'X': QColor(231, 76, 60),      # 红色
            'Z': QColor(46, 204, 113),     # 绿色
            'RZ': QColor(155, 89, 182),    # 紫色
            'CX': QColor(241, 196, 15),    # 黄色
            'M': QColor(149, 165, 166)     # 灰色
        }
        
        # 默认颜色
        color = gate_colors.get(gate.gate_type, QColor(52, 73, 94))
        
        if gate.gate_type == 'CX':
            # 绘制CNOT门
            control = gate.targets[0]
            target = gate.targets[1]
            
            # 控制点
            control_y = (control + 1) * self.grid_size
            self.scene.addEllipse(
                x - 10, control_y - 10, 
                20, 20, 
                QPen(color, 2),
                QBrush(Qt.white)
            )
            
            # 目标点
            target_y = (target + 1) * self.grid_size
            self.scene.addEllipse(
                x - 15, target_y - 15, 
                30, 30, 
                QPen(color, 2),
                QBrush(Qt.white)
            )
            
            # 添加X
            text = self.scene.addText("X")
            text.setPos(x - 5, target_y - 12)
            
            # 连接线
            self.scene.addLine(
                x, control_y, 
                x, target_y,
                QPen(color, 2, Qt.SolidLine)
            )
        else:
            # 绘制单量子位门
            for target in gate.targets:
                y = (target + 1) * self.grid_size
                
                # 门矩形
                rect = self.scene.addRect(
                    x - 20, y - 20, 
                    40, 40, 
                    QPen(color.darker(), 2),
                    QBrush(color.lighter(120))
                )
                
                # 门标签
                text = gate.gate_type
                if gate.gate_type == 'RZ' and 'theta' in gate.params:
                    theta = gate.params['theta']
                    text = f"RZ({theta:.2f})"
                    
                gate_label = self.scene.addText(text)
                gate_label.setPos(
                    x - gate_label.boundingRect().width() / 2,
                    y - 10
                )
                
    def set_num_qubits(self, num_qubits):
        """设置量子比特数量"""
        self.num_qubits = num_qubits
        self._draw_circuit()
        
    def add_gate(self, gate):
        """添加量子门"""
        self.gates.append(gate)
        self._draw_circuit()
        
    def clear_gates(self):
        """清除所有量子门"""
        self.gates = []
        self._draw_circuit()
        
    def get_circuit(self):
        """获取电路描述"""
        return {
            'num_qubits': self.num_qubits,
            'gates': [gate.to_dict() for gate in self.gates]
        }

class QuantumCircuitPanel(QWidget):
    """量子电路面板 - 用于设计和执行量子电路"""
    
    def __init__(self, system_manager, parent=None):
        super().__init__(parent)
        self.system_manager = system_manager
        
        # 电路属性
        self.circuit = None
        self.circuit_id = None
        self.job_id = None
        self.results = None
        
        # AI进化相关属性
        self.evolution_level = 1  # AI演化级别
        self.learning_history = []  # 学习历史
        self.optimization_metrics = {'execution_time': 0}  # 优化指标
        self.optimization_history = []  # 优化历史
        self.circuit_stats = {}  # 电路统计
        self.execution_count = []  # 执行次数记录
        
        # AI进化相关属性
        self.circuit_performance = {}
        self.auto_optimization_enabled = True
        self.template_database = {
            'basic': ['bell_state', 'ghz_state', 'qft'],
            'intermediate': ['grover', 'phase_estimation', 'shor'],
            'advanced': ['qnn', 'vqe', 'qaoa', 'quantum_walk']
        }
        self.user_preferences = {
            'complexity_level': 'intermediate',
            'preferred_gates': ['h', 'cx', 'rz', 'rx'],
            'optimization_priority': 'depth'  # 可选：depth, fidelity, speed
        }
        self.learning_rate = 0.05  # AI学习率
        self.adaptation_threshold = 0.7  # 适应阈值
        
        # 初始化UI
        self._init_ui()
        
        logger.info("量子电路面板初始化完成")
        
    def _init_ui(self):
        """初始化UI组件 - 简化版"""
        # 创建布局
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # 简化顶部区域 - 只保留必要的控件
        top_layout = QHBoxLayout()
        
        # 电路名称输入框
        self.circuit_name_input = QLineEdit()
        self.circuit_name_input.setPlaceholderText("电路名称")
        self.circuit_name_input.setMinimumWidth(200)
        self.circuit_name_input.setText("量子电路")
        top_layout.addWidget(self.circuit_name_input)
        
        # 量子比特设置
        qubits_layout = QHBoxLayout()
        qubits_layout.addWidget(QLabel("量子比特:"))
        self.qubits_spinbox = QSpinBox()
        self.qubits_spinbox.setRange(1, 20)
        self.qubits_spinbox.setValue(5)
        self.qubits_spinbox.valueChanged.connect(self._on_qubits_changed)
        qubits_layout.addWidget(self.qubits_spinbox)
        top_layout.addLayout(qubits_layout)
        
        # 添加间隔
        top_layout.addSpacing(20)
        
        # 简化为一个AI智能设计按钮
        self.auto_design_button = QPushButton("一键AI设计与执行")
        self.auto_design_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; font-size: 14px; padding: 8px 16px;")
        self.auto_design_button.clicked.connect(self._one_click_design)
        self.auto_design_button.setMinimumWidth(200)
        top_layout.addWidget(self.auto_design_button)
        
        # 添加弹性空间
        top_layout.addStretch(1)
        
        main_layout.addLayout(top_layout)
        
        # 状态和信息栏
        status_bar = QWidget()
        status_layout = QHBoxLayout(status_bar)
        status_layout.setContentsMargins(5, 0, 5, 0)
        
        # 状态标签
        self.status_label = QLabel("就绪 - 点击一键AI设计与执行")
        self.status_label.setStyleSheet("color: #4CAF50;")
        status_layout.addWidget(self.status_label)
        
        # 电路信息标签
        self.circuit_info_label = QLabel("电路统计: 0个门 | 深度: 0")
        status_layout.addWidget(self.circuit_info_label)
        
        # AI助手状态
        self.ai_status_label = QLabel(f"AI助手: 已激活 (级别 {self.evolution_level}) {'★' * min(self.evolution_level, 3)}")
        self.ai_status_label.setStyleSheet("color: #2196F3; font-weight: bold;")
        status_layout.addWidget(self.ai_status_label)
        
        # 添加弹性空间
        status_layout.addStretch(1)
        
        main_layout.addWidget(status_bar)
        
        # 分割器
        self.splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(self.splitter, 1)  # 添加拉伸因子，使分割器占用更多空间
        
        # 电路画布
        self.circuit_canvas = QuantumCircuitCanvas()
        self.splitter.addWidget(self.circuit_canvas)
        
        # 底部区域
        self.bottom_widget = QWidget()
        self.bottom_layout = QVBoxLayout(self.bottom_widget)
        self.splitter.addWidget(self.bottom_widget)
        
        # 结果标签
        self.result_label = QLabel("执行结果")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.bottom_layout.addWidget(self.result_label)
        
        # 结果文本框
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.bottom_layout.addWidget(self.result_text)
        
        # 添加电路可视化视图
        self.viz_view = QLabel("电路可视化区域")
        self.viz_view.setAlignment(Qt.AlignCenter)
        self.viz_view.setMinimumHeight(150)
        self.viz_view.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ddd;")
        self.bottom_layout.addWidget(self.viz_view)
        
        # 设置分割比例
        self.splitter.setSizes([400, 300])
        
        # 初始化变量
        self.current_circuit = None
        
        # 设置主布局到widget
        self.setLayout(main_layout)
        
        # 显示欢迎消息
        self.result_text.append("✨ 量子电路AI助手已激活")
        self.result_text.append(f"当前AI进化级别: {self.evolution_level} {'★' * min(self.evolution_level, 3)}")
        self.result_text.append("\n👉 点击「一键AI设计与执行」按钮开始")
        self.result_text.append("AI将根据量子比特数量自动设计最优电路并执行")
        self.result_text.append("随着使用次数增加，AI会自动进化，提供更高级的设计")
        
    def _on_qubits_changed(self, value):
        """量子比特数量变化处理"""
        self.circuit_canvas.set_num_qubits(value)
        
    def _on_template_selected(self, index):
        """模板选择处理"""
        if index == 0:
            # "选择模板..."选项 - 不做任何操作
            return
            
        # 获取当前量子比特数
        num_qubits = self.qubits_spinbox.value()
        
        # 确保有足够的量子比特
        required_qubits = max(2, index)
        if num_qubits < required_qubits:
            self.qubits_spinbox.setValue(required_qubits)
            num_qubits = required_qubits
            
        # 清除当前电路
        self.circuit_canvas.clear_gates()
        
        # 根据选择的模板创建电路
        if index == 1:
            # 贝尔态(纠缠态)
            self._create_bell_state()
        elif index == 2:
            # GHZ态(多体纠缠)
            self._create_ghz_state()
        elif index == 3:
            # 量子傅里叶变换
            self._create_qft_circuit()
        elif index == 4:
            # 量子随机数生成器
            self._create_quantum_random()
        elif index == 5:
            # 超级量子搜索
            self._create_quantum_search()
        elif index == 6:
            # 量子相位估计
            self._create_phase_estimation()
        elif index == 7:
            # 量子神经网络
            self._create_quantum_neural_network()
        
        # 重置下拉框
        self.template_combo.setCurrentIndex(0)
        
    def _create_bell_state(self):
        """创建贝尔态电路"""
        # 贝尔态需要2个量子比特
        self.circuit_canvas.add_gate(QuantumGateItem('H', 0))
        self.circuit_canvas.add_gate(QuantumGateItem('CX', [0, 1]))
        self.circuit_canvas.add_gate(QuantumGateItem('M', 0))
        self.circuit_canvas.add_gate(QuantumGateItem('M', 1))
        
        # 显示电路描述
        self.result_text.clear()
        self.result_text.append("✨ 贝尔态量子电路")
        self.result_text.append("这是量子力学中最基础的纠缠态之一")
        self.result_text.append("两个量子比特将处于完全纠缠状态")
        self.result_text.append("测量结果：要么都是0，要么都是1")
        self.result_text.append("\n🔄 请点击「创建电路」按钮将设计转换为量子电路")
        
    def _create_ghz_state(self):
        """创建GHZ态电路"""
        num_qubits = self.qubits_spinbox.value()
        
        # GHZ态 - 多体纠缠态
        self.circuit_canvas.add_gate(QuantumGateItem('H', 0))
        
        # 添加CNOT门连接所有量子比特
        for i in range(1, num_qubits):
            self.circuit_canvas.add_gate(QuantumGateItem('CX', [0, i]))
            
        # 添加测量
        for i in range(num_qubits):
            self.circuit_canvas.add_gate(QuantumGateItem('M', i))
        
        # 显示电路描述
        self.result_text.clear()
        self.result_text.append(f"✨ GHZ态量子电路 ({num_qubits}量子比特)")
        self.result_text.append("这是一种多体量子纠缠态")
        self.result_text.append("所有量子比特将处于完全纠缠状态")
        self.result_text.append("测量结果：要么全0，要么全1")
        self.result_text.append("\n🔄 请点击「创建电路」按钮将设计转换为量子电路")
        
    def _create_qft_circuit(self):
        """创建量子傅里叶变换电路"""
        num_qubits = self.qubits_spinbox.value()
        
        # 为每个量子比特创建叠加态
        for i in range(num_qubits):
            self.circuit_canvas.add_gate(QuantumGateItem('H', i))
            
        # 添加旋转门和CNOT门，模拟QFT
        for i in range(num_qubits):
            for j in range(i+1, num_qubits):
                self.circuit_canvas.add_gate(QuantumGateItem('CX', [i, j]))
                self.circuit_canvas.add_gate(QuantumGateItem('RZ', j, {'theta': 3.14159 / 2}))
                self.circuit_canvas.add_gate(QuantumGateItem('CX', [i, j]))
        
        # 添加测量
        for i in range(num_qubits):
            self.circuit_canvas.add_gate(QuantumGateItem('M', i))
            
        # 显示电路描述
        self.result_text.clear()
        self.result_text.append(f"✨ 量子傅里叶变换电路 ({num_qubits}量子比特)")
        self.result_text.append("量子版本的傅里叶变换，用于分析量子态的频谱")
        self.result_text.append("应用：量子相位估计、Shor算法等")
        self.result_text.append("\n🔄 请点击「创建电路」按钮将设计转换为量子电路")
    
    def _create_quantum_random(self):
        """创建量子随机数生成器电路"""
        num_qubits = self.qubits_spinbox.value()
        
        # 为每个量子比特创建叠加态
        for i in range(num_qubits):
            self.circuit_canvas.add_gate(QuantumGateItem('H', i))
            
        # 添加测量
        for i in range(num_qubits):
            self.circuit_canvas.add_gate(QuantumGateItem('M', i))
            
        # 显示电路描述
        self.result_text.clear()
        self.result_text.append(f"✨ 量子随机数生成器 ({num_qubits}位随机数)")
        self.result_text.append("利用量子测量的随机性生成真随机数")
        self.result_text.append(f"可生成2^{num_qubits}个不同的随机数")
        self.result_text.append("\n🔄 请点击「创建电路」按钮将设计转换为量子电路")
    
    def _create_quantum_search(self):
        """创建超级量子搜索电路"""
        num_qubits = self.qubits_spinbox.value()
        
        # 至少需要3个量子比特
        if num_qubits < 3:
            num_qubits = 3
            self.qubits_spinbox.setValue(num_qubits)
            
        # 为所有量子比特创建叠加态
        for i in range(num_qubits):
            self.circuit_canvas.add_gate(QuantumGateItem('H', i))
            
        # 创建搜索算法主体（模拟Grover迭代）
        # 相位翻转
        for i in range(num_qubits):
            if i % 2 == 0:
                self.circuit_canvas.add_gate(QuantumGateItem('Z', i))
                
        # 扩散算子
        for i in range(num_qubits):
            self.circuit_canvas.add_gate(QuantumGateItem('H', i))
            
        for i in range(num_qubits):
            self.circuit_canvas.add_gate(QuantumGateItem('X', i))
            
        # 多控制相位门（用CNOT门和Z门模拟）
        control = 0
        for i in range(1, num_qubits-1):
            self.circuit_canvas.add_gate(QuantumGateItem('CX', [control, i]))
            
        self.circuit_canvas.add_gate(QuantumGateItem('Z', num_qubits-1))
        
        for i in range(num_qubits-2, 0, -1):
            self.circuit_canvas.add_gate(QuantumGateItem('CX', [control, i]))
            
        # 还原
        for i in range(num_qubits):
            self.circuit_canvas.add_gate(QuantumGateItem('X', i))
            
        for i in range(num_qubits):
            self.circuit_canvas.add_gate(QuantumGateItem('H', i))
            
        # 添加测量
        for i in range(num_qubits):
            self.circuit_canvas.add_gate(QuantumGateItem('M', i))
            
        # 显示电路描述
        self.result_text.clear()
        self.result_text.append(f"✨ 超级量子搜索电路 ({num_qubits}量子比特)")
        self.result_text.append("灵感来自Grover搜索算法")
        self.result_text.append(f"搜索空间：2^{num_qubits}，搜索步骤：√(2^{num_qubits})")
        self.result_text.append("比任何经典算法都要快")
        self.result_text.append("\n🔄 请点击「创建电路」按钮将设计转换为量子电路")
    
    def _create_phase_estimation(self):
        """创建量子相位估计电路"""
        num_qubits = self.qubits_spinbox.value()
        
        # 至少需要4个量子比特
        if num_qubits < 4:
            num_qubits = 4
            self.qubits_spinbox.setValue(num_qubits)
            
        precision_qubits = num_qubits - 1
        target_qubit = num_qubits - 1
        
        # 步骤1: 对所有精度量子比特应用H门
        for i in range(precision_qubits):
            self.circuit_canvas.add_gate(QuantumGateItem('H', i))
            
        # 步骤2: 应用受控U操作
        # 这里用Z旋转门模拟相位门
        for i in range(precision_qubits):
            self.circuit_canvas.add_gate(QuantumGateItem('CX', [i, target_qubit]))
            power = 2 ** i
            theta = 3.14159 / power
            self.circuit_canvas.add_gate(QuantumGateItem('RZ', target_qubit, {'theta': theta}))
            self.circuit_canvas.add_gate(QuantumGateItem('CX', [i, target_qubit]))
            
        # 步骤3: 逆量子傅里叶变换
        # 这里简化为H门和CNOT门的组合
        for i in range(precision_qubits//2):
            self.circuit_canvas.add_gate(QuantumGateItem('CX', [i, precision_qubits-i-1]))
            self.circuit_canvas.add_gate(QuantumGateItem('H', i))
            self.circuit_canvas.add_gate(QuantumGateItem('H', precision_qubits-i-1))
            
        # 步骤4: 测量所有精度量子比特
        for i in range(precision_qubits):
            self.circuit_canvas.add_gate(QuantumGateItem('M', i))
            
        # 显示电路描述
        self.result_text.clear()
        self.result_text.append(f"✨ 量子相位估计电路 ({precision_qubits}位精度)")
        self.result_text.append("用于估计量子门的特征值的相位")
        self.result_text.append("应用：量子算法中的核心子程序")
        self.result_text.append(f"精度：2^-{precision_qubits}")
        self.result_text.append("\n🔄 请点击「创建电路」按钮将设计转换为量子电路")
    
    def _create_quantum_neural_network(self):
        """创建量子神经网络电路"""
        num_qubits = self.qubits_spinbox.value()
        
        # 至少需要4个量子比特
        if num_qubits < 4:
            num_qubits = 4
            self.qubits_spinbox.setValue(num_qubits)
            
        # 步骤1: 输入编码层 - 应用H门创建叠加态
        for i in range(num_qubits):
            self.circuit_canvas.add_gate(QuantumGateItem('H', i))
            
        # 步骤2: 隐藏层1 - 纠缠层
        for i in range(num_qubits-1):
            self.circuit_canvas.add_gate(QuantumGateItem('CX', [i, i+1]))
        
        # 步骤3: 参数化旋转层 - 使用RZ门
        for i in range(num_qubits):
            theta = 3.14159 / (i+2)  # 不同的角度
            self.circuit_canvas.add_gate(QuantumGateItem('RZ', i, {'theta': theta}))
            
        # 步骤4: 隐藏层2 - 第二次纠缠
        for i in range(num_qubits-1, 0, -1):
            self.circuit_canvas.add_gate(QuantumGateItem('CX', [i, i-1]))
            
        # 步骤5: 输出层 - 最终旋转和测量
        for i in range(num_qubits):
            if i % 2 == 0:
                self.circuit_canvas.add_gate(QuantumGateItem('H', i))
            else:
                self.circuit_canvas.add_gate(QuantumGateItem('X', i))
                
        # 步骤6: 测量
        for i in range(num_qubits):
            self.circuit_canvas.add_gate(QuantumGateItem('M', i))
            
        # 显示电路描述
        self.result_text.clear()
        self.result_text.append(f"✨ 量子神经网络电路 ({num_qubits}量子比特)")
        self.result_text.append("变分量子电路模拟神经网络")
        self.result_text.append("结构: 输入层→隐藏层1→参数层→隐藏层2→输出层")
        self.result_text.append("应用: 量子机器学习、量子分类、量子特征提取")
        self.result_text.append("\n🔄 请点击「创建电路」按钮将设计转换为量子电路")
        
    def _add_gate(self, gate_type):
        """添加量子门"""
        if gate_type == 'CX':
            # CNOT门需要两个目标比特
            if self.circuit_canvas.num_qubits < 2:
                QMessageBox.warning(self, "错误", "CNOT门需要至少2个量子比特")
                return
                
            # 默认使用0和1
            control = 0
            target = 1
            gate = QuantumGateItem(gate_type, [control, target])
        elif gate_type == 'RZ':
            # RZ门需要角度参数
            target = 0
            theta = 3.14159 / 4  # 默认π/4
            gate = QuantumGateItem(gate_type, target, {'theta': theta})
        else:
            # 单量子位门
            target = 0
            gate = QuantumGateItem(gate_type, target)
            
        self.circuit_canvas.add_gate(gate)
        
    def _clear_circuit(self):
        """清除电路"""
        self.circuit_canvas.clear_gates()
        self.result_text.clear()
        self.circuit_id = None
        self.job_id = None
        self.results = None
        
    def _create_circuit(self):
        """创建量子电路对象"""
        try:
            from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
            
            # 获取量子比特和经典比特数量
            num_qubits = self.qubits_spinbox.value()
            # 经典比特数量与量子比特相同，因为我们移除了clbits_spinbox
            num_clbits = num_qubits  # 使用与量子比特相同的数量
            
            # 创建寄存器
            qreg = QuantumRegister(num_qubits, 'q')
            creg = ClassicalRegister(num_clbits, 'c')
            
            # 创建电路
            circuit = QuantumCircuit(qreg, creg)
            self.circuit = circuit
            
            # 设置电路ID
            self.circuit_id = str(uuid.uuid4())
            
            # 获取电路设计
            circuit_design = self.circuit_canvas.get_circuit()
            
            # 添加门
            for gate in circuit_design['gates']:
                # 解析门信息
                gate_type = gate['type']
                qubits = gate['targets']
                params = gate.get('params', {})
                
                # 添加门到电路
                if gate_type == 'H':
                    circuit.h(qubits)
                elif gate_type == 'X':
                    circuit.x(qubits)
                elif gate_type == 'Y':
                    circuit.y(qubits)
                elif gate_type == 'Z':
                    circuit.z(qubits)
                elif gate_type == 'CX' or gate_type == 'CNOT':
                    circuit.cx(qubits[0], qubits[1])
                elif gate_type == 'CZ':
                    circuit.cz(qubits[0], qubits[1])
                elif gate_type == 'CCX' or gate_type == 'CCNOT' or gate_type == 'Toffoli':
                    circuit.ccx(qubits[0], qubits[1], qubits[2])
                elif gate_type == 'RX':
                    theta = params.get('theta', 0.0)
                    circuit.rx(theta, qubits)
                elif gate_type == 'RY':
                    theta = params.get('theta', 0.0)
                    circuit.ry(theta, qubits)
                elif gate_type == 'RZ':
                    theta = params.get('theta', 0.0)
                    circuit.rz(theta, qubits)
                elif gate_type == 'S':
                    circuit.s(qubits)
                elif gate_type == 'SDG':
                    circuit.sdg(qubits)
                elif gate_type == 'T':
                    circuit.t(qubits)
                elif gate_type == 'TDG':
                    circuit.tdg(qubits)
                elif gate_type == 'M' or gate_type == 'Measure':
                    circuit.measure(qubits, qubits)  # 量子比特到对应的经典比特
                elif gate_type == 'SWAP':
                    circuit.swap(qubits[0], qubits[1])
                elif gate_type == 'CP':
                    theta = params.get('theta', 0.0)
                    circuit.cp(theta, qubits[0], qubits[1])
                elif gate_type == 'CSWAP' or gate_type == 'Fredkin':
                    circuit.cswap(qubits[0], qubits[1], qubits[2])
                    
            # 更新电路信息标签
            self._update_circuit_info()
            
            # 如果有编辑器，更新编辑器
            if hasattr(self, 'circuit_editor') and self.circuit_editor:
                self.circuit_editor.set_circuit(self.circuit)
                
            # 更新电路可视化
            self._update_circuit_visualization()
            
            logger.info(f"量子电路创建成功，ID: {self.circuit_id}")
            
            # 清除结果显示
            if hasattr(self, 'results_view') and self.results_view:
                self.results_view.clear()
                
            # 更新状态
            self.status_label.setText("状态: 电路已创建")
            
            return self.circuit
        except Exception as e:
            logger.error(f"创建量子电路时出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"创建量子电路失败: {str(e)}")
            return None
        
    def _run_circuit(self):
        """运行量子电路"""
        try:
            # 获取当前电路
            circuit = self._get_current_circuit()
            if circuit is None:
                QMessageBox.warning(self, "警告", "没有可运行的量子电路")
                return
            
            # 检查电路是否有内容
            if circuit.size() == 0:
                QMessageBox.warning(self, "警告", "电路为空，请添加门操作后再运行")
                return
            
            # 更新状态
            self.status_label.setText("状态: 正在运行电路...")
            QApplication.processEvents()
            
            # 获取量子后端
            backend = None
            try:
                if hasattr(self.system_manager, 'get_component'):
                    backend = self.system_manager.get_component('backend')
                elif hasattr(self.system_manager, 'get_quantum_backend'):
                    backend = self.system_manager.get_quantum_backend()
                elif hasattr(self.system_manager, 'components') and 'backend' in self.system_manager.components:
                    backend = self.system_manager.components['backend']
            except Exception as e:
                logger.warning(f"获取系统量子后端时出错: {str(e)}")
                
            if backend is None:
                # 使用Qiskit的模拟器
                from qiskit.providers.aer import AerSimulator
                backend = AerSimulator()
                logger.warning("使用默认的Qiskit模拟器，因为未找到系统量子后端")
            
            # 记录开始时间
            start_time = time.time()
            
            # 运行电路
            try:
                # 使用不依赖后端属性的转译方式
                from qiskit import transpile
                
                # 直接指定基础门和优化级别，不使用后端信息
                transpiled_circuit = transpile(
                    circuit, 
                    basis_gates=['u1', 'u2', 'u3', 'cx'],
                    optimization_level=1,
                    seed_transpiler=42
                )
                
                # 使用转译后的电路运行
                job = backend.run(transpiled_circuit, shots=1024)
                self.job_id = job.job_id() if hasattr(job, 'job_id') else str(uuid.uuid4())
                self.results = job.result()
            except Exception as e:
                logger.error(f"使用基础门转译失败: {str(e)}, 尝试直接运行...")
                job = backend.run(circuit, shots=1024)
                self.job_id = job.job_id() if hasattr(job, 'job_id') else str(uuid.uuid4())
                self.results = job.result()
            
            # 记录执行时间
            execution_time = time.time() - start_time
            self.optimization_metrics['execution_time'] = execution_time
            
            # 记录电路性能
            self._record_circuit_performance(circuit)
            
            # 更新结果显示
            self._display_results(self.results)
            
            # 更新状态
            self.status_label.setText(f"状态: 电路执行完成，用时 {execution_time:.4f} 秒")
            
        except Exception as e:
            logger.error(f"运行量子电路时出错: {str(e)}")
            self.status_label.setText(f"状态: 运行失败 - {str(e)}")
            QMessageBox.critical(self, "错误", f"运行量子电路失败: {str(e)}")

    def _display_results(self, results):
        """显示运行结果"""
        try:
            if not results:
                return
                
            # 获取计数结果
            counts = results.get_counts() if hasattr(results, 'get_counts') else {}
            
            # 更新结果文本区域
            results_text = "运行结果:\n\n"
            
            if counts:
                # 按照计数排序
                sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
                
                for state, count in sorted_counts:
                    probability = count / sum(counts.values())
                    results_text += f"|{state}⟩: {count} ({probability:.2%})\n"
                    
                # 添加一些统计信息
                results_text += f"\n总样本数: {sum(counts.values())}\n"
                results_text += f"不同状态数: {len(counts)}\n"
                
                # 计算最可能的状态
                most_probable = sorted_counts[0][0]
                results_text += f"最可能的状态: |{most_probable}⟩\n"
            else:
                results_text += "无计数结果\n"
                
            # 添加电路信息
            if self.circuit:
                results_text += f"\n电路深度: {self.circuit.depth()}\n"
                results_text += f"量子门数量: {sum(self.circuit.count_ops().values())}\n"
                results_text += f"量子比特数: {len(self.circuit.qubits)}\n"
                
            # 更新结果文本
            if hasattr(self, 'results_view') and self.results_view:
                self.results_view.setText(results_text)
            elif hasattr(self, 'result_text') and self.result_text:
                self.result_text.setPlainText(results_text)
            
        except Exception as e:
            logger.error(f"显示结果时出错: {str(e)}")
            if hasattr(self, 'results_view') and self.results_view:
                self.results_view.setText(f"显示结果时出错: {str(e)}")
            elif hasattr(self, 'result_text') and self.result_text:
                self.result_text.setPlainText(f"显示结果时出错: {str(e)}")

    def _record_circuit_performance(self, circuit):
        """记录电路执行性能和结果"""
        if not circuit or not self.results:
            return
            
        try:
            # 计算电路的哈希值作为标识
            circuit_id = hash(str(circuit))
            
            # 记录性能指标
            performance = {
                'depth': getattr(circuit, 'depth', lambda: 0)(),
                'gate_count': sum(getattr(circuit, 'count_ops', lambda: {})().values()),
                'execution_time': self.optimization_metrics.get('execution_time', 0),
                'success_rate': 1.0,  # 假设成功率为100%
                'fidelity': 0.9,  # 假设保真度为90%
            }
            
            # 记录到优化历史
            self.optimization_history.append({
                'circuit_id': circuit_id,
                'timestamp': time.time(),
                'performance': performance
            })
            
            # 记录到电路统计
            self.circuit_stats[circuit_id] = performance
            
            # 检查是否满足进化条件
            self._check_evolution_conditions()
        except Exception as e:
            logger.error(f"记录电路性能时出错: {str(e)}")

    def _estimate_fidelity(self, results):
        """估计电路执行保真度"""
        try:
            # 简单实现：基于结果分布评估保真度
            # 实际应用中应该使用更复杂的方法
            counts = results.get_counts() if hasattr(results, 'get_counts') else {}
            if not counts:
                return 0.0
                
            # 计算哪个状态有最高概率
            total = sum(counts.values())
            max_count = max(counts.values())
            
            # 保真度可以简单估计为主要状态的概率
            return max_count / total if total > 0 else 0.0
        except Exception as e:
            logger.error(f"估计保真度时出错: {str(e)}")
            return 0.0

    def _optimize_circuit(self):
        """优化量子电路"""
        try:
            # 获取当前电路
            circuit = self._get_current_circuit()
            if circuit is None:
                QMessageBox.warning(self, "警告", "没有可优化的量子电路")
                return
                
            # 检查电路是否有内容
            if circuit.size() == 0:
                QMessageBox.warning(self, "警告", "电路为空，无法优化")
                return
                
            # 更新状态
            original_depth = circuit.depth()
            original_gates = sum(circuit.count_ops().values())
            
            self.status_label.setText("状态: 正在优化电路...")
            
            # 使用transpile进行优化
            from qiskit import transpile
            optimized_circuit = transpile(circuit, basis_gates=['u1', 'u2', 'u3', 'cx'], 
                                optimization_level=3)
            
            # 获取优化后信息
            optimized_depth = optimized_circuit.depth()
            optimized_gates = sum(optimized_circuit.count_ops().values())
            
            # 计算优化效果
            depth_reduction = original_depth - optimized_depth
            gate_reduction = original_gates - optimized_gates
            
            depth_percent = 100 * depth_reduction / original_depth if original_depth > 0 else 0
            gate_percent = 100 * gate_reduction / original_gates if original_gates > 0 else 0
            
            # 更新电路
            self.circuit = optimized_circuit
            
            # 如果有编辑器，更新编辑器
            if hasattr(self, 'circuit_editor') and self.circuit_editor:
                self.circuit_editor.set_circuit(self.circuit)
                
            # 更新电路可视化
            self._update_circuit_visualization()
            
            # 更新电路信息
            self._update_circuit_info()
            
            # 更新状态
            self.status_label.setText(f"状态: 电路优化完成，深度减少: {depth_percent:.1f}%，门数减少: {gate_percent:.1f}%")
            
            # 添加学习历史
            self.learning_history.append({
                'original_depth': original_depth,
                'optimized_depth': optimized_depth,
                'original_gates': original_gates,
                'optimized_gates': optimized_gates,
                'depth_reduction': depth_reduction,
                'gate_reduction': gate_reduction,
                'timestamp': time.time()
            })
            
            # 检查是否可以进化
            self._check_evolution_conditions()
            
            return optimized_circuit
            
        except Exception as e:
            logger.error(f"优化电路时出错: {str(e)}")
            self.status_label.setText(f"状态: 优化失败 - {str(e)}")
            QMessageBox.critical(self, "错误", f"优化量子电路失败: {str(e)}")
            return None
            
    def _get_current_circuit(self):
        """获取当前量子电路"""
        try:
            # 首先尝试从编辑器获取电路
            if hasattr(self, 'circuit_editor') and self.circuit_editor:
                return self.circuit_editor.get_circuit()
            
            # 如果没有从编辑器获取到，则获取当前缓存的电路
            if hasattr(self, 'circuit') and self.circuit:
                return self.circuit
                
            # 如果还没有电路，则创建一个新电路
            if not hasattr(self, 'circuit') or self.circuit is None:
                self._create_circuit()
                return self.circuit
                
            logger.error("无法获取当前量子电路")
            return None
        except Exception as e:
            logger.error(f"获取当前量子电路时出错: {str(e)}")
            return None

    def _check_evolution_conditions(self):
        """检查是否满足进化条件"""
        # 至少需要5次成功的电路执行才能考虑进化
        if len(self.learning_history) < 5:
            return
            
        # 获取最近的5个执行记录
        recent_history = self.learning_history[-5:]
        
        # 计算平均成功率和保真度
        avg_success_rate = sum(h['performance']['success_rate'] for h in recent_history) / 5
        avg_fidelity = sum(h['performance']['fidelity'] for h in recent_history) / 5
        
        # 如果平均成功率高且保真度好，考虑进化
        if avg_success_rate > 0.9 and avg_fidelity > 0.8:
            # 计算平均深度和门数
            avg_depth = sum(h['performance']['depth'] for h in recent_history) / 5
            avg_gate_count = sum(h['performance']['gate_count'] for h in recent_history) / 5
            
            # 计算执行时间的改善情况
            first_time = recent_history[0]['performance']['execution_time']
            last_time = recent_history[-1]['performance']['execution_time']
            time_improvement = (first_time - last_time) / first_time if first_time > 0 else 0
            
            # 如果执行时间有所改善或门数减少，则进化
            if time_improvement > 0.1 or avg_gate_count < 0.9 * recent_history[0]['performance']['gate_count']:
                self._evolve()
                
    def _evolve(self):
        """电路面板进化，提升自我优化能力"""
        # 提高进化级别
        self.evolution_level += 1
        
        # 记录进化信息
        logger.info(f"量子电路面板进化到级别 {self.evolution_level}")
        
        # 显示进化消息
        if hasattr(self, 'status_label'):
            self.status_label.setText(f"状态: 进化到级别 {self.evolution_level}")
            self.status_label.setStyleSheet("color: blue; font-weight: bold;")
            
        # 根据进化级别调整优化策略
        if self.evolution_level >= 2:
            # 提高转译器优化级别
            if hasattr(self.system_manager, 'get_quantum_backend'):
                backend = self.system_manager.get_quantum_backend()
                if backend:
                    backend.transpilation_level = min(3, self.evolution_level)
                    
        # 执行能力扩展
        if self.evolution_level >= 3:
            # 启用错误缓解
            if hasattr(self.system_manager, 'get_quantum_backend'):
                backend = self.system_manager.get_quantum_backend()
                if backend:
                    backend.error_mitigation = True
                    
        # 更新UI以反映新的进化能力
        self._update_evolution_ui()
                    
        # 发送进化信号
        self.system_manager.signal_manager.emit_signal('quantum_circuit_panel_evolved', self.evolution_level)
        
    def _update_evolution_ui(self):
        """更新UI以反映进化状态"""
        # 更新AI助手状态
        if hasattr(self, 'ai_status_label'):
            evolution_text = f"AI助手: 已激活 (级别 {self.evolution_level})"
            
            if self.evolution_level >= 3:
                self.ai_status_label.setStyleSheet("color: #9C27B0; font-weight: bold;")
                evolution_text += " ★★★"
            elif self.evolution_level >= 2:
                self.ai_status_label.setStyleSheet("color: #2196F3; font-weight: bold;")
                evolution_text += " ★★"
            else:
                self.ai_status_label.setStyleSheet("color: #2196F3;")
                evolution_text += " ★"
                
            self.ai_status_label.setText(evolution_text)
        
    def _update_circuit_info(self):
        """更新电路信息"""
        if self.circuit is None:
            self.circuit_info_label.setText("电路: 未创建")
            return
            
        try:
            # 获取门计数
            gate_counts = self.circuit.count_ops()
            gates_str = ", ".join([f"{gate}: {count}" for gate, count in gate_counts.items()])
            if not gates_str:
                gates_str = "无门操作"
                
            # 更新电路信息标签
            self.circuit_info_label.setText(f"深度: {self.circuit.depth()}, 门: {gates_str}")
        except Exception as e:
            logger.error(f"更新电路信息时出错: {str(e)}")
            self.circuit_info_label.setText("电路信息更新失败")
    
    def _update_circuit_visualization(self):
        """更新电路可视化"""
        try:
            if self.circuit is None:
                self.viz_view.clear()
                return
                
            # 使用Qiskit绘制电路图
            circuit_drawer = self.circuit.draw(output='mpl', style={'name': 'iqx-dark'})
            
            # 转换为PyQt图像
            from io import BytesIO
            import matplotlib.pyplot as plt
            
            # 保存为PNG图像
            buf = BytesIO()
            circuit_drawer.savefig(buf, format='png', dpi=150)
            plt.close(circuit_drawer)
            
            # 显示图像
            from PyQt5.QtGui import QPixmap
            from PyQt5.QtCore import QByteArray
            
            buf.seek(0)
            data = buf.read()
            pixmap = QPixmap()
            pixmap.loadFromData(QByteArray(data))
            
            # 更新图像视图
            self.viz_view.setPixmap(pixmap)
            self.viz_view.setScaledContents(True)
            
        except Exception as e:
            logger.error(f"更新电路可视化时出错: {str(e)}")
            self.viz_view.setText(f"可视化错误: {str(e)}")
        
    def on_system_started(self):
        """系统启动时调用"""
        pass
        
    def on_system_stopped(self):
        """系统停止时调用"""
        self.circuit_id = None
        self.job_id = None

    def _one_click_design(self):
        """一键设计量子电路 - 自适应智能版，自动执行"""
        try:
            # 获取当前量子比特数
            num_qubits = self.qubits_spinbox.value()
            
            # 显示进度对话框
            progress = QProgressDialog("AI正在智能设计并执行量子电路...", "取消", 0, 100, self)
            progress.setWindowTitle("AI设计中")
            progress.setWindowModality(Qt.WindowModal)
            progress.setValue(10)
            
            # 清除当前电路
            self.circuit_canvas.clear_gates()
            
            # 进度推进
            progress.setValue(20)
            
            # 选择适合的电路类型
            circuit_type = self._choose_optimal_circuit_type(num_qubits)
            
            # 根据量子比特数量智能设计电路
            self._design_circuit_by_type(circuit_type, num_qubits)
            
            # 进度推进
            progress.setValue(80)
            
            # 创建并运行电路
            try:
                # 创建电路
                circuit = self._create_circuit()
                if circuit is None:
                    QMessageBox.warning(self, "警告", "电路创建失败")
                    progress.setValue(100)
                    return
                
                # 进度推进
                progress.setValue(90)
                
                # 运行电路
                self._run_circuit()
                
                # 完成进度
                progress.setValue(100)
                
                # 添加到学习历史
                self._add_to_learning_history(circuit_type, num_qubits)
                
                # 检查是否应该进化
                if len(self.execution_count) % 3 == 0 and self.evolution_level < 3:
                    self._evolve()
                
                # 更新状态
                self.status_label.setText("状态: 电路设计与执行完成")
                self.result_text.append("\n✨ AI智能设计与执行已完成")
                self.result_text.append(f"量子电路类型: {circuit_type}")
                self.result_text.append(f"量子比特: {num_qubits}")
                self.result_text.append(f"当前AI级别: {self.evolution_level} {'★' * min(self.evolution_level, 3)}")
                
            except Exception as e:
                logger.error(f"一键设计执行时出错: {str(e)}")
                QMessageBox.critical(self, "错误", f"执行失败: {str(e)}")
                progress.setValue(100)
            
        except Exception as e:
            logger.error(f"AI设计电路时出错: {str(e)}")
            self.result_text.append(f"❌ 设计过程中出错: {str(e)}")
            self.status_label.setText("AI设计失败")
            self.status_label.setStyleSheet("color: #F44336;")

    def _auto_evolve(self):
        """自动进化检查"""
        # 只有在运行过至少3次电路后才考虑进化
        if len(self.learning_history) < 3:
            return
            
        # 如果已经达到最高级别，不再进化
        if self.evolution_level >= 3:
            self.ai_status_label.setText(f"AI助手: 已达到最高级别 (级别 {self.evolution_level}) ★★★")
            self.ai_status_label.setStyleSheet("color: #9C27B0; font-weight: bold;")
            return
            
        # 简化的进化逻辑，每运行3次电路自动进化一级
        self.evolution_level += 1
        
        # 记录进化信息
        logger.info(f"量子电路AI助手自动进化到级别 {self.evolution_level}")
        
        # 更新AI助手状态
        evolution_text = f"AI助手: 已进化 (级别 {self.evolution_level})"
        
        if self.evolution_level >= 3:
            self.ai_status_label.setStyleSheet("color: #9C27B0; font-weight: bold;")
            evolution_text += " ★★★"
        elif self.evolution_level >= 2:
            self.ai_status_label.setStyleSheet("color: #2196F3; font-weight: bold;")
            evolution_text += " ★★"
        else:
            self.ai_status_label.setStyleSheet("color: #2196F3;")
            evolution_text += " ★"
            
        self.ai_status_label.setText(evolution_text)
        
        # 显示进化消息
        self.result_text.append("\n🚀 AI助手已自动进化！")
        self.result_text.append(f"新等级: {self.evolution_level} {' ★' * self.evolution_level}")
        self.result_text.append("AI助手现在能够设计更高级的量子电路并提供更好的优化")
        
        # 根据进化级别调整优化策略
        if self.evolution_level >= 2:
            # 提高转译器优化级别
            if hasattr(self.system_manager, 'get_quantum_backend'):
                backend = self.system_manager.get_quantum_backend()
                if backend:
                    backend.transpilation_level = min(3, self.evolution_level)
                    
        # 如果进化到级别3，启用错误缓解
        if self.evolution_level >= 3:
            # 启用错误缓解
            if hasattr(self.system_manager, 'get_quantum_backend'):
                backend = self.system_manager.get_quantum_backend()
                if backend:
                    backend.error_mitigation = True
                    
        # 发送进化信号
        if hasattr(self.system_manager, 'signal_manager'):
            self.system_manager.signal_manager.emit_signal('quantum_circuit_panel_evolved', self.evolution_level)

    def _choose_optimal_circuit_type(self, num_qubits):
        """基于量子比特数量和系统状态智能选择电路类型"""
        if num_qubits == 1:
            return "single_qubit"
        elif num_qubits == 2:
            return "bell_state"
        elif num_qubits >= 3 and num_qubits <= 5:
            # 基于进化程度选择
            evolution_level = getattr(self, 'evolution_level', 1)
            if evolution_level >= 2 and num_qubits >= 4:
                return "qft"
            else:
                return "ghz_state"
        elif num_qubits >= 6 and num_qubits <= 8:
            # 适合机器学习的规模
            evolution_level = getattr(self, 'evolution_level', 1)
            if evolution_level >= 2:
                return "quantum_ml"
            else:
                return "advanced_circuit"
        else:
            return "advanced_circuit"
            
    def _add_to_learning_history(self, circuit_type, num_qubits):
        """添加到学习历史"""
        history_entry = {
            'timestamp': time.time(),
            'circuit_type': circuit_type,
            'num_qubits': num_qubits,
            'evolution_level': self.evolution_level,
            'performance': {
                'success_rate': 1.0,
                'fidelity': 0.9,
                'depth': getattr(self.circuit, 'depth', lambda: 0)(),
                'gate_count': sum(getattr(self.circuit, 'count_ops', lambda: {})().values()),
                'execution_time': self.optimization_metrics.get('execution_time', 0)
            }
        }
        
        # 添加到学习历史
        self.learning_history.append(history_entry)
        
        # 添加到执行次数记录
        self.execution_count.append({
            'timestamp': time.time(),
            'circuit_type': circuit_type
        })
        
        # 限制历史记录大小
        if len(self.learning_history) > 100:
            self.learning_history = self.learning_history[-100:]
        
        if len(self.execution_count) > 100:
            self.execution_count = self.execution_count[-100:]

    def _save_circuit(self):
        """保存量子电路"""
        try:
            # 获取当前电路
            circuit = self._get_current_circuit()
            if not circuit:
                QMessageBox.warning(self, "警告", "没有可保存的量子电路")
                return
                
            # 获取文件名
            file_name, _ = QFileDialog.getSaveFileName(self, "保存量子电路", "", "Qiskit Quantum Circuit Files (*.qasm);;All Files (*)")
            if not file_name:
                return
                
            # 保存电路
            circuit.save(file_name)
            
            QMessageBox.information(self, "保存成功", f"量子电路已成功保存到 {file_name}")
            
        except Exception as e:
            logger.error(f"保存量子电路时出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"保存量子电路失败: {str(e)}")

    def _load_circuit(self):
        """加载量子电路"""
        try:
            # 获取文件名
            file_name, _ = QFileDialog.getOpenFileName(self, "加载量子电路", "", "Qiskit Quantum Circuit Files (*.qasm);;All Files (*)")
            if not file_name:
                return
                
            # 加载电路
            from qiskit import QuantumCircuit
            self.circuit = QuantumCircuit.from_qasm_file(file_name)
            
            # 更新电路可视化
            self.circuit_canvas.gates = self.circuit.gates()
            self.circuit_canvas._draw_circuit()
            
            # 更新状态
            self.status_label.setText("状态: 电路已加载")
            
        except Exception as e:
            logger.error(f"加载量子电路时出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"加载量子电路失败: {str(e)}")

    def _design_circuit_by_type(self, circuit_type, num_qubits):
        """根据电路类型设计电路"""
        if circuit_type == "single_qubit":
            # 单量子比特电路 - 简单的叠加和旋转
            self.circuit_canvas.add_gate(QuantumGateItem('H', 0))
            self.circuit_canvas.add_gate(QuantumGateItem('RX', 0, {'theta': 0.5}))
            self.circuit_canvas.add_gate(QuantumGateItem('RZ', 0, {'theta': 0.3}))
            self.circuit_canvas.add_gate(QuantumGateItem('M', 0))
        
        elif circuit_type == "bell_state":
            # 两量子比特电路 - 制备Bell态
            self.circuit_canvas.add_gate(QuantumGateItem('H', 0))
            self.circuit_canvas.add_gate(QuantumGateItem('CX', [0, 1]))
            self.circuit_canvas.add_gate(QuantumGateItem('M', 0))
            self.circuit_canvas.add_gate(QuantumGateItem('M', 1))
            
        elif circuit_type == "ghz_state":
            # 3-5量子比特 - GHZ态 (更加高效的构建)
            self.circuit_canvas.add_gate(QuantumGateItem('H', 0))
            
            # 使用级联方式连接量子比特，相比原来的方式更高效
            for i in range(1, num_qubits):
                self.circuit_canvas.add_gate(QuantumGateItem('CX', [i-1, i]))
                
            # 添加测量
            for i in range(num_qubits):
                self.circuit_canvas.add_gate(QuantumGateItem('M', i))
                
        elif circuit_type == "qft":
            # 量子傅里叶变换 (优化版)
            # 第一层Hadamard门
            for i in range(num_qubits):
                self.circuit_canvas.add_gate(QuantumGateItem('H', i))
            
            # 添加受控旋转门
            for i in range(num_qubits):
                for j in range(i+1, num_qubits):
                    # 添加控制旋转门，角度随距离增加而减小
                    phase = 2.0 * 3.14159 / (2**(j-i))
                    self.circuit_canvas.add_gate(QuantumGateItem('CP', [i, j], {'theta': phase}))
            
            # 最终的Hadamard层
            for i in range(num_qubits):
                self.circuit_canvas.add_gate(QuantumGateItem('H', i))
                
            # 测量
            for i in range(num_qubits):
                self.circuit_canvas.add_gate(QuantumGateItem('M', i))
                
        elif circuit_type == "quantum_ml":
            # 量子机器学习电路原型
            # 输入编码层
            for i in range(num_qubits):
                self.circuit_canvas.add_gate(QuantumGateItem('RY', i, {'theta': 0.1 * (i+1)}))
            
            # 纠缠层
            for i in range(num_qubits-1):
                self.circuit_canvas.add_gate(QuantumGateItem('CX', [i, i+1]))
            
            # 参数化旋转层
            for i in range(num_qubits):
                self.circuit_canvas.add_gate(QuantumGateItem('RZ', i, {'theta': 0.2 * (i+1)}))
                self.circuit_canvas.add_gate(QuantumGateItem('RX', i, {'theta': 0.15 * (i+1)}))
            
            # 第二纠缠层
            for i in range(num_qubits-1):
                self.circuit_canvas.add_gate(QuantumGateItem('CX', [i, i+1]))
            
            # 测量
            for i in range(num_qubits):
                self.circuit_canvas.add_gate(QuantumGateItem('M', i))
                
        else:  # advanced_circuit
            # 复杂的高级电路，自动适应量子比特数量
            layers = min(5, num_qubits // 2 + 1)  # 确定电路的层数
            
            # 使用多种量子门类型
            gate_types = ['H', 'X', 'Y', 'Z', 'RX', 'RY', 'RZ']
            
            # 创建多层结构
            for layer in range(layers):
                # 单比特门层
                for i in range(num_qubits):
                    if (i + layer) % len(gate_types) < 3:  # 使用部分门减少复杂度
                        gate = gate_types[(i + layer) % len(gate_types)]
                        if gate in ['RX', 'RY', 'RZ']:
                            self.circuit_canvas.add_gate(QuantumGateItem(gate, i, {'theta': 0.1 * (i+layer)}))
                        else:
                            self.circuit_canvas.add_gate(QuantumGateItem(gate, i))
                
                # 纠缠层 - 实现不同的连接模式
                if layer % 3 == 0:  # 线性连接
                    for i in range(0, num_qubits-1, 2):
                        self.circuit_canvas.add_gate(QuantumGateItem('CX', [i, i+1]))
                elif layer % 3 == 1:  # 反向连接
                    for i in range(num_qubits-1, 0, -2):
                        self.circuit_canvas.add_gate(QuantumGateItem('CX', [i, i-1]))
                else:  # 长距离连接
                    for i in range(num_qubits//2):
                        if i + num_qubits//2 < num_qubits:
                            self.circuit_canvas.add_gate(QuantumGateItem('CX', [i, i + num_qubits//2]))
            
            # 最后加测量
            for i in range(num_qubits):
                self.circuit_canvas.add_gate(QuantumGateItem('M', i))