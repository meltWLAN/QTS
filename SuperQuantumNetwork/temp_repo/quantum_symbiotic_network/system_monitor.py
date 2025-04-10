#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - 系统监控模块
监控系统资源使用情况，确保系统稳定运行
"""

import os
import sys
import time
import json
import logging
import threading
import platform
from datetime import datetime
import psutil  # 需要安装: pip install psutil

# 设置日志记录器
logger = logging.getLogger("SystemMonitor")

class SystemMonitor:
    """系统监控类，监控CPU、内存和磁盘使用情况"""
    
    def __init__(self, config=None):
        """初始化系统监控器
        
        Args:
            config: 监控配置字典，包含阈值和监控间隔等
        """
        self.logger = logging.getLogger("SystemMonitor")
        self.logger.info("初始化系统监控模块")
        
        # 默认配置
        self.config = {
            'memory_threshold': 0.85,        # 内存使用阈值 (85%)
            'cpu_threshold': 0.90,           # CPU使用阈值 (90%)
            'disk_threshold': 0.90,          # 磁盘使用阈值 (90%)
            'monitor_interval': 60,          # 监控间隔 (秒)
            'collect_metrics': True,         # 是否收集指标
            'metrics_retention_days': 7,     # 指标保留天数
            'enable_auto_optimization': True, # 是否启用自动优化
            'logging_level': 'INFO'          # 日志级别
        }
        
        # 更新配置
        if config:
            self.config.update(config)
        
        # 初始化状态和指标
        self.running = False
        self.monitor_thread = None
        self.status = {
            'system': platform.system(),
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'initialized',
            'warnings': [],
            'errors': []
        }
        
        # 性能指标历史
        self.metrics_history = []
        
        # 性能优化设置
        self.gc_threshold = 0.75  # 触发垃圾回收的内存阈值
        self.last_gc_time = 0     # 上次垃圾回收时间
        
        # 优化操作回调函数
        self.optimization_callbacks = {
            'memory': [],
            'cpu': [],
            'disk': []
        }
        
        # 指标保存路径
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.metrics_dir = os.path.join(self.base_dir, "logs", "metrics")
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # 设置日志级别
        log_level = getattr(logging, self.config['logging_level'])
        self.logger.setLevel(log_level)
    
    def start(self):
        """启动系统监控"""
        if self.running:
            logger.warning("系统监控已经在运行中")
            return
        
        self.running = True
        self.status['status'] = 'running'
        
        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, name="SystemMonitorThread")
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        logger.info("系统监控已启动")
    
    def stop(self):
        """停止系统监控"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        self.status['status'] = 'stopped'
        logger.info("系统监控已停止")
    
    def _monitoring_loop(self):
        """监控循环，定期检查系统资源"""
        while self.running:
            try:
                # 收集系统指标
                metrics = self._collect_system_metrics()
                
                # 检查是否超过阈值
                warnings = self._check_thresholds(metrics)
                
                # 如果有警告，记录并触发优化
                if warnings:
                    self.status['warnings'] = warnings
                    logger.warning(f"系统资源警告: {warnings}")
                    
                    # 如果启用了自动优化，则执行优化
                    if self.config['enable_auto_optimization']:
                        self._perform_optimization(metrics)
                
                # 保存指标历史
                if self.config['collect_metrics']:
                    self.metrics_history.append(metrics)
                    
                    # 定期保存指标
                    if len(self.metrics_history) >= 60:  # 保存约1小时的数据
                        self._save_metrics()
                        self.metrics_history = []
                
                # 清理过期的指标文件
                if datetime.now().hour == 0 and datetime.now().minute < 5:
                    self._cleanup_old_metrics()
                
            except Exception as e:
                error_msg = f"监控循环发生错误: {str(e)}"
                logger.error(error_msg)
                self.status['errors'].append(error_msg)
            
            # 休眠指定间隔
            time.sleep(self.config['monitor_interval'])
    
    def _collect_system_metrics(self):
        """收集系统指标"""
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 内存使用
        memory = psutil.virtual_memory()
        memory_used_percent = memory.percent / 100.0
        
        # CPU使用
        cpu_percent = psutil.cpu_percent(interval=0.5) / 100.0
        
        # 磁盘使用
        disk = psutil.disk_usage('/')
        disk_used_percent = disk.percent / 100.0
        
        # 进程信息
        process = psutil.Process(os.getpid())
        process_memory = process.memory_info().rss / (1024 * 1024)  # MB
        process_cpu = process.cpu_percent(interval=0.1)
        
        # 收集指标
        metrics = {
            'timestamp': current_time,
            'memory': {
                'total': memory.total / (1024 * 1024 * 1024),  # GB
                'available': memory.available / (1024 * 1024 * 1024),  # GB
                'used_percent': memory_used_percent,
                'swap_used_percent': psutil.swap_memory().percent / 100.0 if hasattr(psutil, 'swap_memory') else 0
            },
            'cpu': {
                'used_percent': cpu_percent,
                'core_count': psutil.cpu_count(logical=True),
                'physical_core_count': psutil.cpu_count(logical=False)
            },
            'disk': {
                'total': disk.total / (1024 * 1024 * 1024),  # GB
                'used': disk.used / (1024 * 1024 * 1024),  # GB
                'used_percent': disk_used_percent
            },
            'process': {
                'memory_mb': process_memory,
                'cpu_percent': process_cpu,
                'threads': process.num_threads() if hasattr(process, 'num_threads') else 0
            },
            'system': {
                'boot_time': datetime.fromtimestamp(psutil.boot_time()).strftime('%Y-%m-%d %H:%M:%S'),
                'uptime_hours': (time.time() - psutil.boot_time()) / 3600
            }
        }
        
        return metrics
    
    def _check_thresholds(self, metrics):
        """检查是否超过阈值"""
        warnings = []
        
        # 检查内存
        if metrics['memory']['used_percent'] > self.config['memory_threshold']:
            warnings.append(f"内存使用率过高: {metrics['memory']['used_percent']*100:.1f}%")
        
        # 检查CPU
        if metrics['cpu']['used_percent'] > self.config['cpu_threshold']:
            warnings.append(f"CPU使用率过高: {metrics['cpu']['used_percent']*100:.1f}%")
        
        # 检查磁盘
        if metrics['disk']['used_percent'] > self.config['disk_threshold']:
            warnings.append(f"磁盘使用率过高: {metrics['disk']['used_percent']*100:.1f}%")
        
        return warnings
    
    def _perform_optimization(self, metrics):
        """执行资源优化"""
        # 内存优化
        if metrics['memory']['used_percent'] > self.config['memory_threshold']:
            self._optimize_memory(metrics)
        
        # CPU优化
        if metrics['cpu']['used_percent'] > self.config['cpu_threshold']:
            self._optimize_cpu(metrics)
        
        # 磁盘优化
        if metrics['disk']['used_percent'] > self.config['disk_threshold']:
            self._optimize_disk(metrics)
    
    def _optimize_memory(self, metrics):
        """内存优化"""
        logger.info("执行内存优化...")
        
        # 强制垃圾回收
        current_time = time.time()
        if current_time - self.last_gc_time > 300:  # 至少间隔5分钟
            try:
                import gc
                collected = gc.collect()
                self.last_gc_time = current_time
                logger.info(f"垃圾回收完成，清理了 {collected} 个对象")
            except Exception as e:
                logger.error(f"垃圾回收失败: {str(e)}")
        
        # 执行回调
        for callback in self.optimization_callbacks['memory']:
            try:
                callback(metrics)
            except Exception as e:
                logger.error(f"内存优化回调失败: {str(e)}")
    
    def _optimize_cpu(self, metrics):
        """CPU优化"""
        logger.info("执行CPU优化...")
        
        # 降低线程优先级
        if platform.system() == "Windows":
            try:
                import win32api
                import win32process
                import win32con
                win32process.SetThreadPriority(win32api.GetCurrentThread(), win32con.THREAD_PRIORITY_BELOW_NORMAL)
                logger.info("已降低CPU优先级")
            except:
                pass
        else:
            try:
                os.nice(10)  # 增加nice值，降低优先级
                logger.info("已降低CPU优先级")
            except:
                pass
        
        # 执行回调
        for callback in self.optimization_callbacks['cpu']:
            try:
                callback(metrics)
            except Exception as e:
                logger.error(f"CPU优化回调失败: {str(e)}")
    
    def _optimize_disk(self, metrics):
        """磁盘优化"""
        logger.info("执行磁盘优化...")
        
        # 清理临时文件
        temp_dirs = []
        if platform.system() == "Windows":
            temp_dirs.append(os.environ.get('TEMP', 'C:\\Windows\\Temp'))
        else:
            temp_dirs.append('/tmp')
        
        # 添加应用临时目录
        app_temp = os.path.join(self.base_dir, "temp")
        if os.path.exists(app_temp):
            temp_dirs.append(app_temp)
        
        # 清理缓存文件
        cache_dir = os.path.join(self.base_dir, "cache")
        if os.path.exists(cache_dir):
            temp_dirs.append(cache_dir)
        
        # 删除临时文件
        files_deleted = 0
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                try:
                    # 只删除3天前的临时文件
                    cutoff = time.time() - (3 * 24 * 3600)
                    for root, dirs, files in os.walk(temp_dir):
                        for f in files:
                            if f.endswith('.tmp') or f.endswith('.temp'):
                                file_path = os.path.join(root, f)
                                try:
                                    if os.path.getmtime(file_path) < cutoff:
                                        os.remove(file_path)
                                        files_deleted += 1
                                except:
                                    pass
                except Exception as e:
                    logger.error(f"清理临时文件失败: {str(e)}")
        
        if files_deleted > 0:
            logger.info(f"已删除 {files_deleted} 个临时文件")
        
        # 执行回调
        for callback in self.optimization_callbacks['disk']:
            try:
                callback(metrics)
            except Exception as e:
                logger.error(f"磁盘优化回调失败: {str(e)}")
    
    def _save_metrics(self):
        """保存监控指标到文件"""
        if not self.metrics_history:
            return
        
        current_date = datetime.now().strftime('%Y%m%d')
        metrics_file = os.path.join(self.metrics_dir, f"metrics_{current_date}.json")
        
        try:
            # 读取已有数据（如果存在）
            existing_metrics = []
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    existing_metrics = json.load(f)
            
            # 合并数据并保存
            all_metrics = existing_metrics + self.metrics_history
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(all_metrics, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"保存了 {len(self.metrics_history)} 条性能指标到 {metrics_file}")
            
        except Exception as e:
            logger.error(f"保存性能指标失败: {str(e)}")
    
    def _cleanup_old_metrics(self):
        """清理过期的指标文件"""
        retention_days = self.config['metrics_retention_days']
        current_time = time.time()
        cutoff_time = current_time - (retention_days * 24 * 3600)
        
        try:
            for filename in os.listdir(self.metrics_dir):
                file_path = os.path.join(self.metrics_dir, filename)
                if os.path.isfile(file_path) and os.path.getmtime(file_path) < cutoff_time:
                    os.remove(file_path)
                    logger.info(f"已删除过期指标文件: {filename}")
        except Exception as e:
            logger.error(f"清理过期指标文件失败: {str(e)}")
    
    def get_system_status(self):
        """获取当前系统状态"""
        # 更新状态信息
        metrics = self._collect_system_metrics()
        
        # 计算健康得分
        health_score = self._calculate_health_score(metrics)
        
        status = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'system_info': self.status,
            'current_metrics': metrics,
            'health_score': health_score,
            'warnings': self.status['warnings'],
            'monitor_status': self.status['status']
        }
        
        return status
    
    def _calculate_health_score(self, metrics):
        """计算系统健康得分 (0-100)"""
        # 内存得分 (0-40)
        memory_score = 40 * (1 - metrics['memory']['used_percent'] / self.config['memory_threshold'])
        memory_score = max(0, min(40, memory_score))
        
        # CPU得分 (0-40)
        cpu_score = 40 * (1 - metrics['cpu']['used_percent'] / self.config['cpu_threshold'])
        cpu_score = max(0, min(40, cpu_score))
        
        # 磁盘得分 (0-20)
        disk_score = 20 * (1 - metrics['disk']['used_percent'] / self.config['disk_threshold'])
        disk_score = max(0, min(20, disk_score))
        
        # 总得分
        total_score = round(memory_score + cpu_score + disk_score)
        return total_score
    
    def register_optimization_callback(self, resource_type, callback):
        """注册资源优化回调函数
        
        Args:
            resource_type: 资源类型 ('memory', 'cpu', 'disk')
            callback: 回调函数，接受metrics参数
        """
        if resource_type in self.optimization_callbacks:
            self.optimization_callbacks[resource_type].append(callback)
            logger.info(f"已注册{resource_type}优化回调函数")
            return True
        else:
            logger.warning(f"未知的资源类型: {resource_type}")
            return False

# 创建系统监控单例
_monitor_instance = None

def get_monitor(config=None):
    """获取系统监控单例实例
    
    Args:
        config: 监控配置字典，包含阈值和监控间隔等
        
    Returns:
        SystemMonitor: 系统监控实例
    """
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = SystemMonitor(config)
    elif config is not None:
        _monitor_instance.config.update(config)
    
    return _monitor_instance

def start_monitoring():
    """启动系统监控"""
    monitor = get_monitor()
    monitor.start()
    return monitor

def get_system_status():
    """获取当前系统状态"""
    monitor = get_monitor()
    return monitor.get_system_status()

def optimize_memory():
    """强制执行内存优化"""
    monitor = get_monitor()
    metrics = monitor._collect_system_metrics()
    monitor._optimize_memory(metrics)

def optimize_system():
    """强制执行全面系统优化"""
    monitor = get_monitor()
    metrics = monitor._collect_system_metrics()
    monitor._optimize_memory(metrics)
    monitor._optimize_cpu(metrics)
    monitor._optimize_disk(metrics)
    return True 