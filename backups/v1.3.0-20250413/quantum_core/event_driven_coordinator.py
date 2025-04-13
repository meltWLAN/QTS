"""
event_driven_coordinator - 量子核心组件
事件驱动协调器 - 管理和协调所有组件的运行时环境
"""

import logging
import threading
import time
from typing import Dict, Any, List, Set

logger = logging.getLogger(__name__)

class RuntimeEnvironment:
    """运行时环境 - 协调系统中所有组件"""
    
    def __init__(self):
        self.components = {}  # 组件名称 -> 组件实例
        self.component_states = {}  # 组件名称 -> 状态
        self.event_manager = None  # 事件管理器引用
        self.is_running = False
        self.startup_lock = threading.RLock()
        self.shutdown_lock = threading.RLock()
        logger.info("运行时环境初始化完成")
        
    def register_component(self, name: str, component: Any):
        """注册组件"""
        if name in self.components:
            logger.warning(f"组件'{name}'已存在，将被替换")
            
        self.components[name] = component
        self.component_states[name] = 'registered'
        logger.info(f"组件'{name}'已注册")
        return True
        
    def unregister_component(self, name: str):
        """注销组件"""
        if name not in self.components:
            logger.warning(f"组件'{name}'不存在，无法注销")
            return False
            
        # 如果组件已启动，先停止它
        if self.component_states.get(name) == 'running':
            self.stop_component(name)
            
        # 移除组件
        del self.components[name]
        del self.component_states[name]
        logger.info(f"组件'{name}'已注销")
        return True
        
    def get_component(self, name: str) -> Any:
        """获取组件"""
        return self.components.get(name)
        
    def get_component_state(self, name: str) -> str:
        """获取组件状态"""
        return self.component_states.get(name, 'unknown')
        
    def start(self):
        """启动运行时环境"""
        with self.startup_lock:
            if self.is_running:
                logger.warning("运行时环境已在运行中")
                return
                
            logger.info("启动运行时环境...")
            self.is_running = True
            
            # 查找事件管理器组件
            if 'event_system' in self.components:
                self.event_manager = self.components['event_system']
                logger.info("找到事件系统组件")
            
            # 按顺序启动组件
            for name, component in self.components.items():
                self._start_component(name, component)
                
            logger.info("运行时环境启动完成")
                
    def stop(self):
        """停止运行时环境"""
        with self.shutdown_lock:
            if not self.is_running:
                logger.warning("运行时环境已经停止")
                return
                
            logger.info("停止运行时环境...")
            self.is_running = False
            
            # 按照相反的顺序停止组件
            for name, component in reversed(list(self.components.items())):
                self._stop_component(name, component)
                
            logger.info("运行时环境停止完成")
            
    def _start_component(self, name, component):
        """启动单个组件"""
        logger.info(f"正在启动组件'{name}'...")
        
        try:
            # 检查组件是否有start方法
            if hasattr(component, 'start') and callable(getattr(component, 'start')):
                component.start()
                
            self.component_states[name] = 'running'
            logger.info(f"组件'{name}'启动成功")
            return True
        except Exception as e:
            self.component_states[name] = 'error'
            logger.error(f"组件'{name}'启动失败: {str(e)}")
            return False
            
    def _stop_component(self, name, component):
        """停止单个组件"""
        logger.info(f"正在停止组件'{name}'...")
        
        try:
            # 检查组件是否有stop方法
            if hasattr(component, 'stop') and callable(getattr(component, 'stop')):
                component.stop()
                
            self.component_states[name] = 'stopped'
            logger.info(f"组件'{name}'停止成功")
            return True
        except Exception as e:
            self.component_states[name] = 'error'
            logger.error(f"组件'{name}'停止失败: {str(e)}")
            return False
            
    def restart_component(self, name):
        """重启单个组件"""
        if name not in self.components:
            logger.warning(f"组件'{name}'不存在，无法重启")
            return False
            
        component = self.components[name]
        
        # 先停止组件
        self._stop_component(name, component)
        
        # 等待一小段时间
        time.sleep(0.5)
        
        # 再启动组件
        return self._start_component(name, component)
        
    def emit_event(self, event_type, event_data):
        """发送事件"""
        if not self.event_manager:
            logger.warning("未找到事件管理器，无法发送事件")
            return None
            
        if hasattr(self.event_manager, 'emit_event') and callable(getattr(self.event_manager, 'emit_event')):
            return self.event_manager.emit_event(event_type, event_data)
        else:
            logger.warning("事件管理器不支持emit_event方法")
            return None
            
    def subscribe(self, event_type, callback):
        """订阅事件"""
        if not self.event_manager:
            logger.warning("未找到事件管理器，无法订阅事件")
            return False
            
        if hasattr(self.event_manager, 'subscribe') and callable(getattr(self.event_manager, 'subscribe')):
            return self.event_manager.subscribe(event_type, callback)
        else:
            logger.warning("事件管理器不支持subscribe方法")
            return False
            
    def get_all_components(self):
        """获取所有组件"""
        return {name: {
            'instance': component,
            'state': self.component_states.get(name, 'unknown')
        } for name, component in self.components.items()}

