"""
量子电路面板 - 用于设计和执行量子电路
"""

import logging
import uuid
import time
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                         QPushButton, QComboBox, QFrame, QGridLayout, 
                         QSpinBox, QTextEdit, QSplitter, QToolButton, 
                         QGraphicsView, QGraphicsScene, QGraphicsItem,
                         QMenu, QAction, QMessageBox)
from PyQt5.QtCore import Qt, QRectF, QPointF, pyqtSlot, QSizeF
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QFont, QPainterPath

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
        self.circuit_id = None
        self.job_id = None
        self.results = None
        
        # 初始化UI
        self._init_ui()
        
        logger.info("量子电路面板初始化完成")
        
    def _init_ui(self):
        """初始化用户界面"""
        # 主布局
        self.main_layout = QVBoxLayout(self)
        
        # 顶部工具栏
        self._create_toolbar()
        
        # 分割器
        self.splitter = QSplitter(Qt.Vertical)
        self.main_layout.addWidget(self.splitter)
        
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
        self.bottom_layout.addWidget(self.result_label)
        
        # 结果文本框
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.bottom_layout.addWidget(self.result_text)
        
        # 设置分割比例
        self.splitter.setSizes([400, 200])
        
    def _create_toolbar(self):
        """创建工具栏"""
        # 工具栏容器
        self.toolbar_frame = QFrame()
        self.toolbar_frame.setFrameShape(QFrame.StyledPanel)
        self.toolbar_frame.setFrameShadow(QFrame.Raised)
        self.toolbar_layout = QHBoxLayout(self.toolbar_frame)
        self.main_layout.addWidget(self.toolbar_frame)
        
        # 量子比特数量
        self.qubit_label = QLabel("量子比特:")
        self.toolbar_layout.addWidget(self.qubit_label)
        
        self.qubit_spin = QSpinBox()
        self.qubit_spin.setMinimum(1)
        self.qubit_spin.setMaximum(10)
        self.qubit_spin.setValue(3)
        self.qubit_spin.valueChanged.connect(self._on_qubit_change)
        self.toolbar_layout.addWidget(self.qubit_spin)
        
        # 添加间隔
        self.toolbar_layout.addSpacing(20)
        
        # 添加门按钮
        gates = [
            {'label': 'H门', 'gate': 'H', 'tooltip': 'Hadamard门 - 创建叠加态'},
            {'label': 'X门', 'gate': 'X', 'tooltip': 'Pauli-X门 - 比特翻转（NOT）'},
            {'label': 'Z门', 'gate': 'Z', 'tooltip': 'Pauli-Z门 - 相位翻转'},
            {'label': 'RZ门', 'gate': 'RZ', 'tooltip': 'RZ门 - Z轴旋转'},
            {'label': 'CNOT门', 'gate': 'CX', 'tooltip': '受控非门 - 条件比特翻转'},
            {'label': '测量', 'gate': 'M', 'tooltip': '测量量子比特'}
        ]
        
        for gate_info in gates:
            button = QPushButton(gate_info['label'])
            button.setToolTip(gate_info['tooltip'])
            button.clicked.connect(lambda checked, g=gate_info['gate']: self._add_gate(g))
            self.toolbar_layout.addWidget(button)
            
        # 添加弹性空间
        self.toolbar_layout.addStretch(1)
        
        # 创建电路按钮
        self.create_button = QPushButton("创建电路")
        self.create_button.clicked.connect(self._create_circuit)
        self.toolbar_layout.addWidget(self.create_button)
        
        # 执行电路按钮
        self.execute_button = QPushButton("执行电路")
        self.execute_button.clicked.connect(self._execute_circuit)
        self.execute_button.setEnabled(False)
        self.toolbar_layout.addWidget(self.execute_button)
        
        # 清除按钮
        self.clear_button = QPushButton("清除")
        self.clear_button.clicked.connect(self._clear_circuit)
        self.toolbar_layout.addWidget(self.clear_button)
        
    def _on_qubit_change(self, value):
        """量子比特数量变化处理"""
        self.circuit_canvas.set_num_qubits(value)
        
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
        self.execute_button.setEnabled(False)
        
    def _create_circuit(self):
        """创建量子电路"""
        # 检查量子后端是否已经启动
        if not self._check_backend():
            return
            
        circuit_data = self.circuit_canvas.get_circuit()
        
        try:
            # 获取量子后端
            backend = self.system_manager.get_component('backend')
            
            # 创建电路
            num_qubits = circuit_data['num_qubits']
            self.circuit_id = backend.create_circuit(num_qubits)
            
            # 添加门
            for gate_dict in circuit_data['gates']:
                gate = QuantumGateItem.from_dict(gate_dict)
                
                if gate.gate_type == 'M':
                    # 测量操作
                    for target in gate.targets:
                        backend.add_measurement(self.circuit_id, target)
                else:
                    # 量子门
                    backend.add_gate(
                        self.circuit_id,
                        gate.gate_type,
                        gate.targets,
                        gate.params
                    )
                    
            # 更新UI
            self.execute_button.setEnabled(True)
            self.result_text.setPlainText(f"电路已创建，ID: {self.circuit_id}\n")
            self.result_text.append(f"量子比特数: {num_qubits}")
            self.result_text.append(f"量子门数: {len(circuit_data['gates'])}")
            
            logger.info(f"量子电路创建成功，ID: {self.circuit_id}")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"创建电路时出错: {str(e)}")
            logger.error(f"创建量子电路时出错: {str(e)}")
            
    def _execute_circuit(self):
        """执行量子电路"""
        if not self.circuit_id:
            QMessageBox.warning(self, "错误", "请先创建电路")
            return
            
        # 检查量子后端是否已经启动
        if not self._check_backend():
            return
            
        try:
            # 获取量子后端
            backend = self.system_manager.get_component('backend')
            
            # 更新UI
            self.result_text.append("\n正在执行电路...")
            
            # 执行电路
            shots = 1024
            self.job_id = backend.execute_circuit(self.circuit_id, shots)
            
            # 显示作业ID
            self.result_text.append(f"作业ID: {self.job_id}")
            
            # 等待执行完成
            self._wait_for_result()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"执行电路时出错: {str(e)}")
            logger.error(f"执行量子电路时出错: {str(e)}")
            
    def _wait_for_result(self):
        """等待执行结果"""
        backend = self.system_manager.get_component('backend')
        
        # 检查作业状态
        status = backend.get_job_status(self.job_id)
        
        while status['status'] not in ['completed', 'failed', 'error']:
            # 更新UI
            self.result_text.append(f"状态: {status['status']}")
            
            # 等待一段时间
            time.sleep(0.1)
            
            # 更新状态
            status = backend.get_job_status(self.job_id)
            
        # 获取结果
        if status['status'] == 'completed':
            self.results = backend.get_result(self.job_id)
            self._display_results()
        else:
            self.result_text.append(f"执行失败: {status['status']}")
            
    def _display_results(self):
        """显示执行结果"""
        if not self.results:
            return
            
        # 清除旧结果
        self.result_text.clear()
        
        # 显示基本信息
        self.result_text.append(f"电路ID: {self.circuit_id}")
        self.result_text.append(f"作业ID: {self.job_id}")
        self.result_text.append(f"状态: {self.results.get('status', '未知')}")
        self.result_text.append(f"执行时间: {self.results.get('execution_time', 0):.3f}秒")
        self.result_text.append(f"测量次数: {self.results.get('shots', 0)}")
        
        # 显示计数结果
        self.result_text.append("\n测量结果:")
        counts = self.results.get('counts', {})
        
        for state, count in counts.items():
            percent = count / self.results.get('shots', 1) * 100
            self.result_text.append(f"|{state}⟩: {count} ({percent:.1f}%)")
            
    def _check_backend(self):
        """检查量子后端是否已启动"""
        backend = self.system_manager.get_component('backend')
        
        if not backend:
            QMessageBox.warning(self, "错误", "量子后端未初始化")
            return False
            
        if not self.system_manager.get_component_status('backend').get('backend') == 'running':
            QMessageBox.warning(self, "错误", "量子后端未启动")
            return False
            
        return True
        
    def on_system_started(self):
        """系统启动时调用"""
        pass
        
    def on_system_stopped(self):
        """系统停止时调用"""
        self.execute_button.setEnabled(False)
        self.circuit_id = None
        self.job_id = None 