"""
系统管理器模块 - 管理量子核心系统组件
"""

import logging
import threading
import time
import sys
import os
from typing import Dict, List, Any, Optional

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
sys.path.insert(0, root_dir)  # 使用insert而不是append，确保优先级

# 直接导入特定目录下的模块
quantum_backend_path = os.path.join(root_dir, 'quantum_core')
if quantum_backend_path not in sys.path:
    sys.path.insert(0, quantum_backend_path)

# 导入量子核心组件
try:
    from quantum_core.quantum_backend import QuantumBackend
    from quantum_core.market_to_quantum import MarketToQuantumConverter
    from quantum_core.quantum_interpreter import QuantumInterpreter
    from quantum_core.multidimensional_analysis import MultidimensionalAnalyzer
except ImportError as e:
    print(f"无法导入量子核心组件: {e}")
    print(f"当前Python路径: {sys.path}")
    raise

logger = logging.getLogger("QuantumDesktop.SystemManager")

class SystemManager:
    """系统管理器 - 管理量子核心系统的各个组件"""
    
    def __init__(self):
        # 系统状态
        self.running = False
        
        # 组件实例
        self.components = {
            'backend': None,
            'converter': None,
            'interpreter': None,
            'analyzer': None
        }
        
        # 组件状态
        self.component_status = {
            'backend': 'stopped',
            'converter': 'stopped',
            'interpreter': 'stopped',
            'analyzer': 'stopped'
        }
        
        # 系统监控线程
        self.monitor_thread = None
        self.monitor_active = False
        
        # 系统资源使用
        self.system_resources = {
            'memory_usage': 0,
            'cpu_usage': 0,
            'component_memory': {}
        }
        
        # 系统事件回调
        self.event_callbacks = {
            'system_started': [],
            'system_stopped': [],
            'component_status_changed': [],
            'resources_updated': []
        }
        
        logger.info("系统管理器初始化完成")
        
    def start_system(self) -> bool:
        """启动量子核心系统"""
        if self.running:
            logger.warning("系统已在运行")
            return False
            
        logger.info("启动量子核心系统...")
        
        try:
            # 初始化并启动量子后端
            self.components['backend'] = QuantumBackend(backend_type='simulator')
            self.components['backend'].start()
            self.component_status['backend'] = 'running'
            logger.info("量子后端启动成功")
            
            # 初始化并启动市场到量子转换器
            self.components['converter'] = MarketToQuantumConverter()
            self.components['converter'].start()
            self.component_status['converter'] = 'running'
            logger.info("市场到量子转换器启动成功")
            
            # 初始化并启动量子解释器
            self.components['interpreter'] = QuantumInterpreter()
            self.components['interpreter'].start()
            self.component_status['interpreter'] = 'running'
            logger.info("量子解释器启动成功")
            
            # 初始化并启动市场分析器
            self.components['analyzer'] = MultidimensionalAnalyzer()
            self.components['analyzer'].start()
            self.component_status['analyzer'] = 'running'
            logger.info("市场分析器启动成功")
            
            # 更新系统状态
            self.running = True
            
            # 启动监控线程
            self._start_monitoring()
            
            # 触发系统启动事件
            self._trigger_event('system_started')
            
            logger.info("量子核心系统启动完成")
            return True
            
        except Exception as e:
            logger.error(f"启动系统时出错: {str(e)}")
            self.stop_system()  # 清理已启动的组件
            return False
            
    def stop_system(self) -> bool:
        """停止量子核心系统"""
        if not self.running:
            logger.warning("系统已停止")
            return False
            
        logger.info("停止量子核心系统...")
        
        # 停止监控线程
        self._stop_monitoring()
        
        # 停止各组件
        component_names = ['analyzer', 'interpreter', 'converter', 'backend']
        
        for name in component_names:
            try:
                if self.components[name]:
                    self.components[name].stop()
                    self.component_status[name] = 'stopped'
                    logger.info(f"{name} 停止成功")
            except Exception as e:
                logger.error(f"停止 {name} 时出错: {str(e)}")
                
        # 更新系统状态
        self.running = False
        
        # 清空组件实例
        for name in component_names:
            self.components[name] = None
            
        # 触发系统停止事件
        self._trigger_event('system_stopped')
        
        logger.info("量子核心系统已停止")
        return True
        
    def get_component(self, name: str) -> Any:
        """获取指定组件实例"""
        return self.components.get(name)
        
    def get_component_status(self, name: str = None) -> Dict[str, str]:
        """获取组件状态"""
        if name:
            return {name: self.component_status.get(name, 'unknown')}
        return self.component_status
        
    def is_system_running(self) -> bool:
        """检查系统是否在运行"""
        return self.running
        
    def get_system_resources(self) -> Dict:
        """获取系统资源信息"""
        return self.system_resources.copy()
        
    def get_latest_resources(self) -> Dict:
        """获取最新系统资源信息（线程安全方式）"""
        with threading.Lock():
            return self.system_resources.copy()
        
    def register_event_callback(self, event_type: str, callback) -> bool:
        """注册事件回调"""
        if event_type not in self.event_callbacks:
            logger.warning(f"未知事件类型: {event_type}")
            return False
            
        self.event_callbacks[event_type].append(callback)
        return True
        
    def unregister_event_callback(self, event_type: str, callback) -> bool:
        """注销事件回调"""
        if event_type not in self.event_callbacks:
            return False
            
        if callback in self.event_callbacks[event_type]:
            self.event_callbacks[event_type].remove(callback)
            return True
            
        return False
        
    def _trigger_event(self, event_type: str, data: Any = None) -> None:
        """触发事件"""
        if event_type not in self.event_callbacks:
            return
            
        for callback in self.event_callbacks[event_type]:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"执行事件回调时出错: {str(e)}")
                
    def _start_monitoring(self) -> None:
        """启动系统监控"""
        if self.monitor_thread and self.monitor_thread.is_alive():
            return
            
        self.monitor_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def _stop_monitoring(self) -> None:
        """停止系统监控"""
        self.monitor_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2)
            
    def _monitoring_loop(self) -> None:
        """监控循环"""
        logger.info("开始系统监控")
        
        try:
            import psutil
            has_psutil = True
        except ImportError:
            logger.warning("未安装psutil，将使用模拟数据")
            has_psutil = False
            
        process = psutil.Process() if has_psutil else None
        
        while self.monitor_active:
            try:
                # 更新系统资源使用
                if has_psutil:
                    # 获取内存使用
                    memory_info = process.memory_info()
                    self.system_resources['memory_usage'] = memory_info.rss / (1024 * 1024)  # MB
                    
                    # 获取CPU使用
                    self.system_resources['cpu_usage'] = process.cpu_percent(interval=0.1)
                    
                    # 更新组件资源使用（简化模拟）
                    for name, component in self.components.items():
                        if component:
                            self.system_resources['component_memory'][name] = self.system_resources['memory_usage'] / 4
                else:
                    # 模拟数据
                    self.system_resources['memory_usage'] = 100 + (time.time() % 50)
                    self.system_resources['cpu_usage'] = 5 + (time.time() % 10)
                    
                    for name, component in self.components.items():
                        if component:
                            self.system_resources['component_memory'][name] = 25 + (time.time() % 10)
                            
                # 触发资源更新事件
                self._trigger_event('resources_updated', self.system_resources)
                
                # 检查组件状态
                for name, component in self.components.items():
                    old_status = self.component_status[name]
                    
                    if component:
                        if hasattr(component, 'is_running'):
                            # 组件有is_running属性
                            new_status = 'running' if component.is_running else 'stopped'
                        else:
                            # 假设组件仍在运行
                            new_status = 'running'
                    else:
                        new_status = 'stopped'
                        
                    # 更新状态并触发事件
                    if old_status != new_status:
                        self.component_status[name] = new_status
                        self._trigger_event('component_status_changed', {
                            'component': name,
                            'status': new_status
                        })
                        
            except Exception as e:
                logger.error(f"监控循环中出错: {str(e)}")
                
            # 休眠一段时间
            time.sleep(1)
            
        logger.info("系统监控结束") 