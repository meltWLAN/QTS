"""
system_monitor - 量子核心组件
系统健康监控 - 监控系统性能和健康状态
"""

import logging
import threading
import time
import psutil
import os
import platform
from typing import Dict, List, Any, Callable

logger = logging.getLogger(__name__)

class SystemHealthMonitor:
    """系统健康监控器 - 监控系统资源使用情况"""
    
    def __init__(self, check_interval: float = 5.0):
        self.is_running = False
        self.check_interval = check_interval  # 检查间隔（秒）
        self.monitor_thread = None
        self.system_stats = {}
        self.resources_history = {
            'cpu': [],
            'memory': [],
            'disk': [],
            'timestamps': []
        }
        self.max_history_size = 100  # 历史记录最大长度
        self.alert_callbacks = []  # 警报回调函数列表
        self.alert_thresholds = {
            'cpu': 80.0,  # CPU使用率警报阈值（百分比）
            'memory': 80.0,  # 内存使用率警报阈值（百分比）
            'disk': 90.0  # 磁盘使用率警报阈值（百分比）
        }
        
        # 初始化系统信息
        self._init_system_info()
        
        logger.info("系统健康监控器初始化完成")
        
    def _init_system_info(self):
        """初始化系统信息"""
        self.system_stats['platform'] = {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor()
        }
        
        self.system_stats['python'] = {
            'version': platform.python_version(),
            'implementation': platform.python_implementation()
        }
        
        # 获取磁盘信息
        try:
            disk_usage = psutil.disk_usage('/')
            self.system_stats['disk'] = {
                'total': disk_usage.total,
                'used': disk_usage.used,
                'free': disk_usage.free,
                'percent': disk_usage.percent
            }
        except Exception as e:
            logger.warning(f"获取磁盘信息失败: {str(e)}")
            self.system_stats['disk'] = {}
            
        # 获取CPU信息
        try:
            self.system_stats['cpu_count'] = psutil.cpu_count(logical=True)
            self.system_stats['cpu_count_physical'] = psutil.cpu_count(logical=False)
        except Exception as e:
            logger.warning(f"获取CPU信息失败: {str(e)}")
            self.system_stats['cpu_count'] = 0
            self.system_stats['cpu_count_physical'] = 0
            
        # 获取内存信息
        try:
            memory = psutil.virtual_memory()
            self.system_stats['memory'] = {
                'total': memory.total,
                'available': memory.available,
                'used': memory.used,
                'percent': memory.percent
            }
        except Exception as e:
            logger.warning(f"获取内存信息失败: {str(e)}")
            self.system_stats['memory'] = {}
            
        # 获取当前进程信息
        try:
            process = psutil.Process(os.getpid())
            self.system_stats['process'] = {
                'pid': process.pid,
                'memory_info': dict(process.memory_info()._asdict()),
                'cpu_percent': process.cpu_percent(),
                'memory_percent': process.memory_percent()
            }
        except Exception as e:
            logger.warning(f"获取进程信息失败: {str(e)}")
            self.system_stats['process'] = {}
            
    def start(self):
        """启动监控"""
        if self.is_running:
            logger.warning("监控器已在运行")
            return
            
        logger.info("启动系统监控...")
        self.is_running = True
        
        # 启动监控线程
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitor_thread.start()
        
        logger.info("系统监控已启动")
        
    def stop(self):
        """停止监控"""
        if not self.is_running:
            logger.warning("监控器已停止")
            return
            
        logger.info("停止系统监控...")
        self.is_running = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2)
            
        logger.info("系统监控已停止")
        
    def _monitoring_loop(self):
        """监控循环"""
        logger.info("监控循环启动")
        
        while self.is_running:
            try:
                # 更新系统状态
                self._update_system_stats()
                
                # 检查是否需要触发警报
                self._check_alerts()
                
                # 等待下一次检查
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"监控过程中出错: {str(e)}")
                time.sleep(1)  # 出错后等待1秒再继续
                
        logger.info("监控循环结束")
        
    def _update_system_stats(self):
        """更新系统状态"""
        # 更新CPU使用率
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.system_stats['cpu_percent'] = cpu_percent
        
        # 更新内存使用率
        memory = psutil.virtual_memory()
        self.system_stats['memory'] = {
            'total': memory.total,
            'available': memory.available,
            'used': memory.used,
            'percent': memory.percent
        }
        
        # 更新磁盘使用率
        disk_usage = psutil.disk_usage('/')
        self.system_stats['disk'] = {
            'total': disk_usage.total,
            'used': disk_usage.used,
            'free': disk_usage.free,
            'percent': disk_usage.percent
        }
        
        # 更新进程信息
        process = psutil.Process(os.getpid())
        self.system_stats['process'] = {
            'pid': process.pid,
            'memory_info': dict(process.memory_info()._asdict()),
            'cpu_percent': process.cpu_percent(),
            'memory_percent': process.memory_percent()
        }
        
        # 更新历史记录
        self._update_history(
            cpu_percent,
            memory.percent,
            disk_usage.percent
        )
        
    def _update_history(self, cpu: float, memory: float, disk: float):
        """更新资源使用历史记录"""
        self.resources_history['cpu'].append(cpu)
        self.resources_history['memory'].append(memory)
        self.resources_history['disk'].append(disk)
        self.resources_history['timestamps'].append(time.time())
        
        # 限制历史记录大小
        if len(self.resources_history['cpu']) > self.max_history_size:
            self.resources_history['cpu'].pop(0)
            self.resources_history['memory'].pop(0)
            self.resources_history['disk'].pop(0)
            self.resources_history['timestamps'].pop(0)
            
    def _check_alerts(self):
        """检查是否需要触发警报"""
        alerts = []
        
        # 检查CPU使用率
        if self.system_stats['cpu_percent'] > self.alert_thresholds['cpu']:
            alerts.append({
                'type': 'cpu',
                'level': 'warning',
                'message': f"CPU使用率过高: {self.system_stats['cpu_percent']}%",
                'value': self.system_stats['cpu_percent'],
                'threshold': self.alert_thresholds['cpu']
            })
            
        # 检查内存使用率
        if self.system_stats['memory']['percent'] > self.alert_thresholds['memory']:
            alerts.append({
                'type': 'memory',
                'level': 'warning',
                'message': f"内存使用率过高: {self.system_stats['memory']['percent']}%",
                'value': self.system_stats['memory']['percent'],
                'threshold': self.alert_thresholds['memory']
            })
            
        # 检查磁盘使用率
        if self.system_stats['disk']['percent'] > self.alert_thresholds['disk']:
            alerts.append({
                'type': 'disk',
                'level': 'warning',
                'message': f"磁盘使用率过高: {self.system_stats['disk']['percent']}%",
                'value': self.system_stats['disk']['percent'],
                'threshold': self.alert_thresholds['disk']
            })
            
        # 触发警报回调
        if alerts and self.alert_callbacks:
            for callback in self.alert_callbacks:
                try:
                    callback(alerts)
                except Exception as e:
                    logger.error(f"执行警报回调时出错: {str(e)}")
                    
    def get_system_stats(self) -> Dict:
        """获取系统状态信息"""
        return self.system_stats
        
    def get_resources_history(self) -> Dict:
        """获取资源使用历史记录"""
        return self.resources_history
        
    def register_alert_callback(self, callback: Callable[[List[Dict]], None]):
        """注册警报回调函数"""
        self.alert_callbacks.append(callback)
        logger.info(f"注册警报回调，当前回调数量: {len(self.alert_callbacks)}")
        return True
        
    def unregister_alert_callback(self, callback: Callable):
        """取消注册警报回调函数"""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
            logger.info(f"取消警报回调，当前回调数量: {len(self.alert_callbacks)}")
            return True
        return False
        
    def set_alert_thresholds(self, thresholds: Dict[str, float]):
        """设置警报阈值"""
        for key, value in thresholds.items():
            if key in self.alert_thresholds:
                self.alert_thresholds[key] = value
                
        logger.info(f"更新警报阈值: {self.alert_thresholds}")
        return True
        
    def get_alert_thresholds(self) -> Dict[str, float]:
        """获取警报阈值"""
        return self.alert_thresholds

