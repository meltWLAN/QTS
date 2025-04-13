#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
系统监控器 - 提供高级的系统监控功能
"""

import logging
import time
import psutil
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
from threading import Thread, Lock
from queue import Queue
import json
import os

class SystemMonitor:
    """系统监控器 - 提供高级的系统监控功能"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger('quantum_core.system_monitor')
        self.is_running = False
        self.config = config or {}
        self.monitoring_thread = None
        self.data_queue = Queue()
        self.monitoring_interval = self.config.get('monitoring_interval', 1.0)  # 秒
        self.history_size = self.config.get('history_size', 3600)  # 1小时的数据点
        self.metrics_history = {}
        self.alerts = []
        self.alert_callbacks = []
        self.lock = Lock()
        
        # 监控指标
        self.metrics = {
            'cpu': {
                'usage': [],
                'temperature': [],
                'frequency': [],
                'power': []
            },
            'memory': {
                'used': [],
                'available': [],
                'percent': []
            },
            'disk': {
                'read_bytes': [],
                'write_bytes': [],
                'read_count': [],
                'write_count': []
            },
            'network': {
                'bytes_sent': [],
                'bytes_recv': [],
                'packets_sent': [],
                'packets_recv': []
            },
            'quantum': {
                'circuit_depth': [],
                'gate_count': [],
                'error_rate': [],
                'coherence_time': []
            }
        }
        
        # 告警阈值
        self.alert_thresholds = {
            'cpu_usage': 90.0,  # CPU使用率超过90%
            'memory_usage': 85.0,  # 内存使用率超过85%
            'disk_usage': 90.0,  # 磁盘使用率超过90%
            'quantum_error_rate': 0.01,  # 量子错误率超过1%
            'quantum_coherence_time': 100.0  # 量子相干时间低于100微秒
        }
        
        self.logger.info("系统监控器初始化完成")
        
    def start(self):
        """启动监控器"""
        try:
            self.logger.info("启动系统监控器...")
            self.is_running = True
            self.monitoring_thread = Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            self.logger.info("系统监控器启动完成")
            return True
        except Exception as e:
            self.logger.error(f"启动失败: {str(e)}")
            return False
            
    def stop(self):
        """停止监控器"""
        try:
            self.logger.info("停止系统监控器...")
            self.is_running = False
            if self.monitoring_thread:
                self.monitoring_thread.join(timeout=5.0)
            self.monitoring_thread = None
            self.logger.info("系统监控器已停止")
            return True
        except Exception as e:
            self.logger.error(f"停止失败: {str(e)}")
            return False
        
    def _monitoring_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                # 收集系统指标
                metrics = self._collect_metrics()
                
                # 更新历史数据
                self._update_history(metrics)
                
                # 检查告警条件
                self._check_alerts(metrics)
                
                # 将数据放入队列
                self.data_queue.put(metrics)
                
                # 等待下一个监控间隔
                time.sleep(self.monitoring_interval)
            except Exception as e:
                self.logger.error(f"监控循环出错: {str(e)}")
                time.sleep(1.0)  # 出错后等待1秒再继续
                
    def _collect_metrics(self) -> Dict[str, Any]:
        """收集系统指标"""
        timestamp = datetime.now().timestamp()
        
        # CPU指标
        cpu_metrics = {
            'usage': psutil.cpu_percent(interval=None),
            'temperature': self._get_cpu_temperature(),
            'frequency': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            'power': self._get_cpu_power()
        }
        
        # 内存指标
        memory = psutil.virtual_memory()
        memory_metrics = {
            'used': memory.used,
            'available': memory.available,
            'percent': memory.percent
        }
        
        # 磁盘指标
        disk_io = psutil.disk_io_counters()
        disk_metrics = {
            'read_bytes': disk_io.read_bytes,
            'write_bytes': disk_io.write_bytes,
            'read_count': disk_io.read_count,
            'write_count': disk_io.write_count
        }
        
        # 网络指标
        net_io = psutil.net_io_counters()
        network_metrics = {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv
        }
        
        # 量子指标（模拟数据）
        quantum_metrics = {
            'circuit_depth': np.random.randint(10, 100),
            'gate_count': np.random.randint(50, 500),
            'error_rate': np.random.random() * 0.02,
            'coherence_time': np.random.uniform(50, 200)
        }
        
        return {
            'timestamp': timestamp,
            'cpu': cpu_metrics,
            'memory': memory_metrics,
            'disk': disk_metrics,
            'network': network_metrics,
            'quantum': quantum_metrics
        }
        
    def _get_cpu_temperature(self) -> float:
        """获取CPU温度"""
        try:
            # 尝试从系统获取CPU温度
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps and 'coretemp' in temps:
                    return sum(temp.current for temp in temps['coretemp']) / len(temps['coretemp'])
            # 如果无法获取，返回模拟数据
            return np.random.uniform(40, 80)
        except:
            return np.random.uniform(40, 80)
            
    def _get_cpu_power(self) -> float:
        """获取CPU功耗"""
        try:
            # 尝试从系统获取CPU功耗
            # 这里需要根据具体系统实现
            # 如果无法获取，返回模拟数据
            return np.random.uniform(10, 50)
        except:
            return np.random.uniform(10, 50)
            
    def _update_history(self, metrics: Dict[str, Any]):
        """更新历史数据"""
        with self.lock:
            timestamp = metrics['timestamp']
            
            # 更新CPU历史
            self.metrics['cpu']['usage'].append((timestamp, metrics['cpu']['usage']))
            self.metrics['cpu']['temperature'].append((timestamp, metrics['cpu']['temperature']))
            self.metrics['cpu']['frequency'].append((timestamp, metrics['cpu']['frequency']))
            self.metrics['cpu']['power'].append((timestamp, metrics['cpu']['power']))
            
            # 更新内存历史
            self.metrics['memory']['used'].append((timestamp, metrics['memory']['used']))
            self.metrics['memory']['available'].append((timestamp, metrics['memory']['available']))
            self.metrics['memory']['percent'].append((timestamp, metrics['memory']['percent']))
            
            # 更新磁盘历史
            self.metrics['disk']['read_bytes'].append((timestamp, metrics['disk']['read_bytes']))
            self.metrics['disk']['write_bytes'].append((timestamp, metrics['disk']['write_bytes']))
            self.metrics['disk']['read_count'].append((timestamp, metrics['disk']['read_count']))
            self.metrics['disk']['write_count'].append((timestamp, metrics['disk']['write_count']))
            
            # 更新网络历史
            self.metrics['network']['bytes_sent'].append((timestamp, metrics['network']['bytes_sent']))
            self.metrics['network']['bytes_recv'].append((timestamp, metrics['network']['bytes_recv']))
            self.metrics['network']['packets_sent'].append((timestamp, metrics['network']['packets_sent']))
            self.metrics['network']['packets_recv'].append((timestamp, metrics['network']['packets_recv']))
            
            # 更新量子历史
            self.metrics['quantum']['circuit_depth'].append((timestamp, metrics['quantum']['circuit_depth']))
            self.metrics['quantum']['gate_count'].append((timestamp, metrics['quantum']['gate_count']))
            self.metrics['quantum']['error_rate'].append((timestamp, metrics['quantum']['error_rate']))
            self.metrics['quantum']['coherence_time'].append((timestamp, metrics['quantum']['coherence_time']))
            
            # 限制历史数据大小
            for category in self.metrics:
                for metric in self.metrics[category]:
                    if len(self.metrics[category][metric]) > self.history_size:
                        self.metrics[category][metric] = self.metrics[category][metric][-self.history_size:]
                        
    def _check_alerts(self, metrics: Dict[str, Any]):
        """检查告警条件"""
        # CPU告警
        if metrics['cpu']['usage'] > self.alert_thresholds['cpu_usage']:
            self._add_alert('cpu_usage', f"CPU使用率过高: {metrics['cpu']['usage']}%")
            
        # 内存告警
        if metrics['memory']['percent'] > self.alert_thresholds['memory_usage']:
            self._add_alert('memory_usage', f"内存使用率过高: {metrics['memory']['percent']}%")
            
        # 量子错误率告警
        if metrics['quantum']['error_rate'] > self.alert_thresholds['quantum_error_rate']:
            self._add_alert('quantum_error_rate', 
                           f"量子错误率过高: {metrics['quantum']['error_rate']*100:.2f}%")
            
        # 量子相干时间告警
        if metrics['quantum']['coherence_time'] < self.alert_thresholds['quantum_coherence_time']:
            self._add_alert('quantum_coherence_time', 
                           f"量子相干时间过低: {metrics['quantum']['coherence_time']:.2f}微秒")
            
    def _add_alert(self, alert_type: str, message: str):
        """添加告警"""
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': datetime.now().timestamp(),
            'severity': 'high' if alert_type in ['cpu_usage', 'memory_usage'] else 'medium'
        }
        
        with self.lock:
            self.alerts.append(alert)
            
            # 限制告警历史大小
            if len(self.alerts) > 100:
                self.alerts = self.alerts[-100:]
                
            # 调用告警回调
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"告警回调出错: {str(e)}")
                    
    def register_alert_callback(self, callback: callable):
        """注册告警回调函数"""
        self.alert_callbacks.append(callback)
        
    def get_current_metrics(self) -> Dict[str, Any]:
        """获取当前指标"""
        if not self.is_running:
            raise RuntimeError("监控器未运行")
            
        try:
            return self.data_queue.get_nowait()
        except:
            return self._collect_metrics()
            
    def get_metrics_history(self, category: str = None, metric: str = None, 
                          start_time: float = None, end_time: float = None) -> Dict[str, Any]:
        """获取指标历史数据
        
        Args:
            category: 指标类别，如'cpu', 'memory'等
            metric: 具体指标名称，如'usage', 'temperature'等
            start_time: 开始时间戳
            end_time: 结束时间戳
            
        Returns:
            历史数据字典
        """
        with self.lock:
            if category is None:
                # 返回所有类别的数据
                result = {}
                for cat, metrics in self.metrics.items():
                    result[cat] = {}
                    for met, data in metrics.items():
                        filtered_data = self._filter_data_by_time(data, start_time, end_time)
                        result[cat][met] = filtered_data
                return result
            elif metric is None:
                # 返回指定类别的所有指标
                if category not in self.metrics:
                    raise ValueError(f"不支持的指标类别: {category}")
                result = {}
                for met, data in self.metrics[category].items():
                    filtered_data = self._filter_data_by_time(data, start_time, end_time)
                    result[met] = filtered_data
                return result
            else:
                # 返回指定类别和指标的数据
                if category not in self.metrics or metric not in self.metrics[category]:
                    raise ValueError(f"不支持的指标: {category}.{metric}")
                data = self.metrics[category][metric]
                return self._filter_data_by_time(data, start_time, end_time)
                
    def _filter_data_by_time(self, data: List[Tuple[float, float]], 
                           start_time: float = None, end_time: float = None) -> List[Tuple[float, float]]:
        """按时间过滤数据"""
        if start_time is None and end_time is None:
            return data
            
        filtered_data = []
        for timestamp, value in data:
            if start_time is not None and timestamp < start_time:
                continue
            if end_time is not None and timestamp > end_time:
                continue
            filtered_data.append((timestamp, value))
            
        return filtered_data
        
    def get_alerts(self, alert_type: str = None, severity: str = None, 
                  start_time: float = None, end_time: float = None) -> List[Dict[str, Any]]:
        """获取告警历史
        
        Args:
            alert_type: 告警类型
            severity: 告警级别
            start_time: 开始时间戳
            end_time: 结束时间戳
            
        Returns:
            告警列表
        """
        with self.lock:
            filtered_alerts = []
            
            for alert in self.alerts:
                # 按类型过滤
                if alert_type is not None and alert['type'] != alert_type:
                    continue
                    
                # 按级别过滤
                if severity is not None and alert['severity'] != severity:
                    continue
                    
                # 按时间过滤
                if start_time is not None and alert['timestamp'] < start_time:
                    continue
                if end_time is not None and alert['timestamp'] > end_time:
                    continue
                    
                filtered_alerts.append(alert)
                
            return filtered_alerts
            
    def clear_alerts(self):
        """清除告警历史"""
        with self.lock:
            self.alerts.clear()
            
    def export_metrics(self, file_path: str, format: str = 'json'):
        """导出指标数据
        
        Args:
            file_path: 导出文件路径
            format: 导出格式，支持'json'和'csv'
        """
        with self.lock:
            if format == 'json':
                # 转换为可序列化的格式
                export_data = {}
                for category, metrics in self.metrics.items():
                    export_data[category] = {}
                    for metric, data in metrics.items():
                        export_data[category][metric] = [
                            {'timestamp': ts, 'value': val} for ts, val in data
                        ]
                        
                # 写入JSON文件
                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
                    
            elif format == 'csv':
                # 创建DataFrame
                data_frames = {}
                for category, metrics in self.metrics.items():
                    for metric, data in metrics.items():
                        df = pd.DataFrame(data, columns=['timestamp', 'value'])
                        df['category'] = category
                        df['metric'] = metric
                        data_frames[f"{category}_{metric}"] = df
                        
                # 合并所有DataFrame
                combined_df = pd.concat(data_frames.values(), ignore_index=True)
                
                # 写入CSV文件
                combined_df.to_csv(file_path, index=False)
                
            else:
                raise ValueError(f"不支持的导出格式: {format}")
                
    def import_metrics(self, file_path: str, format: str = 'json'):
        """导入指标数据
        
        Args:
            file_path: 导入文件路径
            format: 导入格式，支持'json'和'csv'
        """
        with self.lock:
            if format == 'json':
                # 从JSON文件读取
                with open(file_path, 'r') as f:
                    import_data = json.load(f)
                    
                # 转换为内部格式
                for category, metrics in import_data.items():
                    if category not in self.metrics:
                        self.metrics[category] = {}
                    for metric, data in metrics.items():
                        if metric not in self.metrics[category]:
                            self.metrics[category][metric] = []
                        self.metrics[category][metric] = [
                            (item['timestamp'], item['value']) for item in data
                        ]
                        
            elif format == 'csv':
                # 从CSV文件读取
                df = pd.read_csv(file_path)
                
                # 转换为内部格式
                for category in df['category'].unique():
                    if category not in self.metrics:
                        self.metrics[category] = {}
                    category_df = df[df['category'] == category]
                    
                    for metric in category_df['metric'].unique():
                        metric_df = category_df[category_df['metric'] == metric]
                        if metric not in self.metrics[category]:
                            self.metrics[category][metric] = []
                        self.metrics[category][metric] = [
                            (row['timestamp'], row['value']) 
                            for _, row in metric_df.iterrows()
                        ]
                        
            else:
                raise ValueError(f"不支持的导入格式: {format}")
                
    def get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标摘要"""
        with self.lock:
            summary = {}
            
            for category, metrics in self.metrics.items():
                summary[category] = {}
                for metric, data in metrics.items():
                    if not data:
                        continue
                        
                    values = [val for _, val in data]
                    summary[category][metric] = {
                        'current': values[-1],
                        'min': min(values),
                        'max': max(values),
                        'mean': sum(values) / len(values),
                        'std': np.std(values) if len(values) > 1 else 0
                    }
                    
            return summary
            
    def reset_metrics(self):
        """重置所有指标"""
        with self.lock:
            for category in self.metrics:
                for metric in self.metrics[category]:
                    self.metrics[category][metric] = []
                    
    def set_alert_threshold(self, alert_type: str, threshold: float):
        """设置告警阈值
        
        Args:
            alert_type: 告警类型
            threshold: 阈值
        """
        if alert_type not in self.alert_thresholds:
            raise ValueError(f"不支持的告警类型: {alert_type}")
            
        self.alert_thresholds[alert_type] = threshold
        self.logger.info(f"设置告警阈值 {alert_type} = {threshold}")
        
    def get_alert_thresholds(self) -> Dict[str, float]:
        """获取所有告警阈值"""
        return self.alert_thresholds.copy()
        
    def set_monitoring_interval(self, interval: float):
        """设置监控间隔
        
        Args:
            interval: 监控间隔（秒）
        """
        if interval < 0.1:
            raise ValueError("监控间隔不能小于0.1秒")
            
        self.monitoring_interval = interval
        self.logger.info(f"设置监控间隔 = {interval}秒")
        
    def set_history_size(self, size: int):
        """设置历史数据大小
        
        Args:
            size: 历史数据点数量
        """
        if size < 100:
            raise ValueError("历史数据大小不能小于100")
            
        self.history_size = size
        self.logger.info(f"设置历史数据大小 = {size}")
        
        # 裁剪现有历史数据
        with self.lock:
            for category in self.metrics:
                for metric in self.metrics[category]:
                    if len(self.metrics[category][metric]) > size:
                        self.metrics[category][metric] = self.metrics[category][metric][-size:]

