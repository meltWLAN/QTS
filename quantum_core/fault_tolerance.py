"""
fault_tolerance - 量子核心组件
容错机制模块 - 提供断路器和优雅降级功能
"""

import logging
import time
import threading
from enum import Enum
from typing import Dict, Callable, Any, List

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """断路器状态"""
    CLOSED = 0      # 正常状态，请求可以通过
    OPEN = 1        # 断开状态，请求被阻止
    HALF_OPEN = 2   # 半开状态，试探性允许请求通过

class CircuitBreaker:
    """断路器模式实现"""
    
    def __init__(self, 
                 name,
                 failure_threshold=5,
                 recovery_timeout=30,
                 failure_counter_window=60,
                 on_state_change=None):
        self.name = name
        self.failure_threshold = failure_threshold  # 触发断路的失败次数
        self.recovery_timeout = recovery_timeout    # 恢复尝试的超时时间（秒）
        self.failure_counter_window = failure_counter_window  # 失败计数窗口（秒）
        
        self.state = CircuitState.CLOSED
        self.failures = []  # 记录失败时间戳
        self.last_failure_time = 0  # 最后一次失败的时间
        self.half_open_success = 0  # 半开状态下成功的请求计数
        
        self.on_state_change = on_state_change  # 状态变化回调
        self.logger = logging.getLogger(f"CircuitBreaker.{name}")
        
        self._lock = threading.RLock()
        
        self.logger.info(f"断路器[{name}]初始化完成")
        
    def execute(self, func, *args, **kwargs):
        """执行被保护的函数"""
        with self._lock:
            # 检查断路器状态
            if self.state == CircuitState.OPEN:
                # 检查是否可以切换到半开状态
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    self._set_state(CircuitState.HALF_OPEN)
                else:
                    self.logger.warning(f"断路器[{self.name}]处于断开状态，拒绝请求")
                    raise CircuitBreakerOpenException(f"断路器[{self.name}]处于断开状态")
                    
        try:
            result = func(*args, **kwargs)
            
            # 如果是半开状态且成功，增加成功计数
            if self.state == CircuitState.HALF_OPEN:
                with self._lock:
                    self.half_open_success += 1
                    if self.half_open_success >= 3:  # 3次成功后恢复
                        self._set_state(CircuitState.CLOSED)
                        
            return result
        except Exception as e:
            with self._lock:
                self._record_failure()
                
                # 根据当前状态决定处理方式
                if self.state == CircuitState.CLOSED and len(self.failures) >= self.failure_threshold:
                    self._set_state(CircuitState.OPEN)
                elif self.state == CircuitState.HALF_OPEN:
                    self._set_state(CircuitState.OPEN)
                    
            # 重新抛出异常
            raise
            
    def _record_failure(self):
        """记录失败"""
        current_time = time.time()
        self.last_failure_time = current_time
        self.failures.append(current_time)
        
        # 清理超出窗口期的失败记录
        cutoff = current_time - self.failure_counter_window
        self.failures = [t for t in self.failures if t >= cutoff]
        
    def _set_state(self, new_state):
        """设置断路器状态"""
        if self.state != new_state:
            old_state = self.state
            self.state = new_state
            self.logger.info(f"断路器[{self.name}]状态从{old_state}变为{new_state}")
            
            # 重置半开状态下的成功计数
            if new_state == CircuitState.HALF_OPEN:
                self.half_open_success = 0
                
            # 触发状态变化回调
            if self.on_state_change:
                try:
                    self.on_state_change(self.name, old_state, new_state)
                except Exception as e:
                    self.logger.error(f"状态变化回调出错: {str(e)}")
                    
    def reset(self):
        """重置断路器状态"""
        with self._lock:
            self.failures = []
            self.last_failure_time = 0
            self.half_open_success = 0
            self._set_state(CircuitState.CLOSED)
            
    def get_state(self):
        """获取当前状态"""
        return self.state
        
    def get_failure_count(self):
        """获取当前失败计数"""
        return len(self.failures)
        
    def get_state_info(self):
        """获取状态信息"""
        with self._lock:
            return {
                'name': self.name,
                'state': self.state.name,
                'failure_count': len(self.failures),
                'last_failure_time': self.last_failure_time,
                'failure_threshold': self.failure_threshold,
                'recovery_timeout': self.recovery_timeout
            }

class CircuitBreakerOpenException(Exception):
    """断路器打开异常"""
    pass
    
class GracefulDegradation:
    """优雅降级实现"""
    
    def __init__(self):
        self.fallbacks = {}
        self.logger = logging.getLogger("GracefulDegradation")
        self.logger.info("优雅降级机制初始化完成")
        
    def register_fallback(self, service_name, primary_func, fallback_func):
        """注册服务及其降级方案"""
        self.fallbacks[service_name] = {
            'primary': primary_func,
            'fallback': fallback_func
        }
        self.logger.debug(f"已为服务[{service_name}]注册降级方案")
        return True
        
    def unregister_fallback(self, service_name):
        """注销降级方案"""
        if service_name in self.fallbacks:
            del self.fallbacks[service_name]
            self.logger.debug(f"已移除服务[{service_name}]的降级方案")
            return True
        return False
        
    def execute(self, service_name, *args, **kwargs):
        """执行服务，失败时降级"""
        if service_name not in self.fallbacks:
            raise ValueError(f"未找到服务[{service_name}]的降级配置")
            
        config = self.fallbacks[service_name]
        try:
            # 尝试执行主要功能
            result = config['primary'](*args, **kwargs)
            return result
        except Exception as e:
            self.logger.warning(f"服务[{service_name}]执行失败，启用降级: {str(e)}")
            # 执行降级方案
            return config['fallback'](*args, **kwargs)
            
    def get_registered_services(self):
        """获取已注册的服务列表"""
        return list(self.fallbacks.keys())

class FaultToleranceManager:
    """容错管理器"""
    
    def __init__(self):
        self.circuit_breakers = {}
        self.degradation = GracefulDegradation()
        self.logger = logging.getLogger("FaultToleranceManager")
        self.logger.info("容错管理器初始化完成")
        
    def create_circuit_breaker(self, name, **config):
        """创建断路器"""
        if name in self.circuit_breakers:
            self.logger.warning(f"断路器[{name}]已存在，将被替换")
            
        circuit_breaker = CircuitBreaker(
            name=name,
            failure_threshold=config.get('failure_threshold', 5),
            recovery_timeout=config.get('recovery_timeout', 30),
            failure_counter_window=config.get('failure_counter_window', 60),
            on_state_change=config.get('on_state_change')
        )
        
        self.circuit_breakers[name] = circuit_breaker
        return circuit_breaker
        
    def get_circuit_breaker(self, name):
        """获取断路器"""
        return self.circuit_breakers.get(name)
        
    def remove_circuit_breaker(self, name):
        """移除断路器"""
        if name in self.circuit_breakers:
            del self.circuit_breakers[name]
            return True
        return False
        
    def register_fallback(self, service_name, primary_func, fallback_func):
        """注册降级方案"""
        return self.degradation.register_fallback(service_name, primary_func, fallback_func)
        
    def execute_with_fallback(self, service_name, *args, **kwargs):
        """执行带降级的服务"""
        return self.degradation.execute(service_name, *args, **kwargs)
        
    def execute_with_circuit_breaker(self, circuit_name, func, *args, **kwargs):
        """执行带断路器保护的函数"""
        circuit_breaker = self.get_circuit_breaker(circuit_name)
        if not circuit_breaker:
            self.logger.warning(f"未找到断路器[{circuit_name}]，创建默认配置")
            circuit_breaker = self.create_circuit_breaker(circuit_name)
            
        return circuit_breaker.execute(func, *args, **kwargs)
        
    def get_all_circuit_breakers(self):
        """获取所有断路器状态"""
        return {name: breaker.get_state_info() for name, breaker in self.circuit_breakers.items()}
        
    def get_all_fallback_services(self):
        """获取所有降级服务"""
        return self.degradation.get_registered_services()

