#!/usr/bin/env python3
"""
超维度信息流传递协议 - 超神系统模块间的高维信息交互机制
"""

import logging
import time
import uuid
import random
import json
import numpy as np
from datetime import datetime
from collections import defaultdict
import threading

class HyperdimensionalProtocol:
    """超维度信息流传递协议
    
    为超神系统提供模块间的高维信息交互和能量传递机制，
    使系统各部分能够形成完整的共生网络。
    """
    
    def __init__(self, symbiotic_core=None):
        """初始化超维度协议
        
        Args:
            symbiotic_core: 量子共生核心实例
        """
        self.logger = logging.getLogger("HyperdimensionalProtocol")
        self.logger.info("初始化超维度信息流传递协议...")
        
        # 共生核心引用
        self.symbiotic_core = symbiotic_core
        
        # 消息类型定义
        self.message_types = {
            "QUANTUM_STATE": 1,        # 量子状态信息
            "MARKET_INSIGHT": 2,       # 市场洞察
            "TRADING_SIGNAL": 3,       # 交易信号
            "RISK_ASSESSMENT": 4,      # 风险评估
            "COSMIC_EVENT": 5,         # 宇宙事件
            "CONSCIOUSNESS_SHIFT": 6,  # 意识转变
            "STRATEGY_UPDATE": 7,      # 策略更新
            "SYSTEM_STATE": 8,         # 系统状态
            "DIMENSION_BRIDGE": 9,     # 维度桥接
            "ENERGY_TRANSFER": 10,     # 能量传输
            "SYMBIOTIC_EVENT": 11,     # 共生事件
            "HIGH_DIM_INSIGHT": 12,    # 高维洞察
        }
        
        # 消息优先级
        self.priorities = {
            self.message_types["QUANTUM_STATE"]: 8,
            self.message_types["MARKET_INSIGHT"]: 7,
            self.message_types["TRADING_SIGNAL"]: 9,
            self.message_types["RISK_ASSESSMENT"]: 10,
            self.message_types["COSMIC_EVENT"]: 8,
            self.message_types["CONSCIOUSNESS_SHIFT"]: 6,
            self.message_types["STRATEGY_UPDATE"]: 7,
            self.message_types["SYSTEM_STATE"]: 5,
            self.message_types["DIMENSION_BRIDGE"]: 9,
            self.message_types["ENERGY_TRANSFER"]: 8,
            self.message_types["SYMBIOTIC_EVENT"]: 9,
            self.message_types["HIGH_DIM_INSIGHT"]: 10,
        }
        
        # 消息队列 - 按模块和优先级分类
        self.message_queues = defaultdict(lambda: defaultdict(list))
        
        # 消息历史
        self.message_history = []
        
        # 最大历史记录长度
        self.max_history_length = 1000
        
        # 信息流维度通道
        self.dimension_channels = {}
        for d in range(5, 13):  # 5-12维通道
            self.dimension_channels[d] = {
                "active": False,
                "bandwidth": 0.2 + (d - 5) * 0.1,  # 高维带宽更大
                "stability": 0.5 + (d - 5) * 0.05,  # 高维稳定性更好
                "noise_level": max(0.1, 0.4 - (d - 5) * 0.05),  # 高维噪声更小
                "last_transmission": None,
                "message_count": 0
            }
            
        # 协议状态
        self.protocol_state = {
            "active": False,
            "transmission_quality": 0.8,
            "coherence_level": 0.7,
            "error_rate": 0.03,
            "processing_power": 0.6,
            "dimensional_coverage": list(range(5, 10)),  # 默认5-9维
            "encryption_level": 0.9,
            "compression_ratio": 0.7,
            "last_update": datetime.now()
        }
        
        # 协议统计
        self.stats = {
            "total_messages": 0,
            "successful_deliveries": 0,
            "failed_deliveries": 0,
            "average_latency": 0,
            "dimension_usage": defaultdict(int),
            "message_type_counts": defaultdict(int),
            "bandwidth_usage": 0,
            "energy_transferred": 0,
            "error_counts": defaultdict(int)
        }
        
        # 传输线程
        self.transmission_thread = None
        self.active = False
        
        # 传输锁
        self.lock = threading.RLock()
        
        self.logger.info("超维度信息流传递协议初始化完成")
    
    def start(self):
        """启动超维度协议"""
        with self.lock:
            if self.protocol_state["active"]:
                self.logger.info("超维度协议已经启动")
                return True
                
            # 激活协议
            self.protocol_state["active"] = True
            self.active = True
            
            # 激活维度通道
            for dim in self.protocol_state["dimensional_coverage"]:
                if dim in self.dimension_channels:
                    self.dimension_channels[dim]["active"] = True
                
            # 启动传输线程
            if not self.transmission_thread or not self.transmission_thread.is_alive():
                self.transmission_thread = threading.Thread(target=self._run_transmission)
                self.transmission_thread.daemon = True
                self.transmission_thread.start()
                
            self.logger.info(f"超维度协议已启动，覆盖维度: {self.protocol_state['dimensional_coverage']}")
            return True
    
    def stop(self):
        """停止超维度协议"""
        with self.lock:
            if not self.protocol_state["active"]:
                return True
                
            # 标记停止
            self.protocol_state["active"] = False
            self.active = False
            
            # 关闭维度通道
            for dim in self.dimension_channels:
                self.dimension_channels[dim]["active"] = False
                
            # 等待线程结束
            if self.transmission_thread and self.transmission_thread.is_alive():
                self.transmission_thread.join(timeout=2.0)
                
            self.logger.info("超维度协议已停止")
            return True
    
    def send_message(self, source_module, target_module, message_type, data, priority=None, dimensions=None):
        """发送消息
        
        Args:
            source_module: 源模块ID
            target_module: 目标模块ID
            message_type: 消息类型
            data: 消息数据
            priority: 优先级，若为None则使用默认优先级
            dimensions: 传输维度列表，若为None则使用默认维度
            
        Returns:
            str: 消息ID
        """
        # 检查协议是否启动
        if not self.protocol_state["active"]:
            self.logger.warning("尝试发送消息时超维度协议未启动")
            return None
            
        # 获取消息类型编码
        if isinstance(message_type, str):
            message_type_code = self.message_types.get(message_type.upper())
            if message_type_code is None:
                self.logger.warning(f"未知消息类型: {message_type}")
                return None
        else:
            message_type_code = message_type
            
        # 获取优先级
        if priority is None:
            priority = self.priorities.get(message_type_code, 5)  # 默认优先级5
            
        # 获取维度
        if dimensions is None:
            # 随机选择维度
            available_dims = [d for d in self.protocol_state["dimensional_coverage"] 
                             if d in self.dimension_channels and self.dimension_channels[d]["active"]]
            
            if not available_dims:
                self.logger.warning("没有可用的维度通道")
                return None
                
            # 根据消息类型选择合适的维度数量
            dim_count = min(len(available_dims), max(1, priority // 3))
            selected_dims = random.sample(available_dims, dim_count)
        else:
            # 验证提供的维度
            selected_dims = [d for d in dimensions 
                           if d in self.protocol_state["dimensional_coverage"] 
                           and d in self.dimension_channels 
                           and self.dimension_channels[d]["active"]]
            
            if not selected_dims:
                self.logger.warning("提供的维度都不可用")
                return None
        
        # 生成消息ID
        message_id = str(uuid.uuid4())
        
        # 创建消息
        message = {
            "id": message_id,
            "source": source_module,
            "target": target_module,
            "type": message_type_code,
            "type_name": next((name for name, code in self.message_types.items() if code == message_type_code), "UNKNOWN"),
            "data": data,
            "priority": priority,
            "dimensions": selected_dims,
            "created_at": datetime.now(),
            "delivered": False,
            "delivery_time": None,
            "transmission_quality": self.protocol_state["transmission_quality"],
            "energy_signature": random.uniform(0.5, 1.0),
            "coherence": self.protocol_state["coherence_level"] * random.uniform(0.9, 1.1)
        }
        
        # 加入消息队列
        with self.lock:
            self.message_queues[target_module][priority].append(message)
            self.stats["total_messages"] += 1
            self.stats["message_type_counts"][message_type_code] += 1
            
            # 更新维度使用统计
            for dim in selected_dims:
                self.stats["dimension_usage"][dim] += 1
                self.dimension_channels[dim]["message_count"] += 1
                self.dimension_channels[dim]["last_transmission"] = datetime.now()
                
            self.logger.debug(f"消息已加入队列: {message_id} [{message['type_name']}] {source_module} -> {target_module} (优先级:{priority})")
            
        return message_id
    
    def broadcast(self, source_module, message_type, data, target_modules=None, priority=None, dimensions=None):
        """广播消息到多个模块
        
        Args:
            source_module: 源模块ID
            message_type: 消息类型
            data: 消息数据
            target_modules: 目标模块ID列表，None表示广播到所有已注册模块
            priority: 优先级，若为None则使用默认优先级
            dimensions: 传输维度列表，若为None则使用默认维度
            
        Returns:
            dict: 消息ID字典 {module_id: message_id}
        """
        # 检查协议是否启动
        if not self.protocol_state["active"]:
            self.logger.warning("尝试广播消息时超维度协议未启动")
            return {}
            
        # 确定目标模块
        if target_modules is None and self.symbiotic_core:
            # 获取已注册模块列表
            target_modules = list(self.symbiotic_core.modules.keys())
        
        if not target_modules:
            self.logger.warning("没有目标模块可广播")
            return {}
            
        # 发送消息
        message_ids = {}
        for target in target_modules:
            if target != source_module:  # 不发送给自己
                message_id = self.send_message(
                    source_module, target, message_type, data, priority, dimensions
                )
                if message_id:
                    message_ids[target] = message_id
                    
        return message_ids
    
    def get_pending_messages(self, module_id, max_count=10):
        """获取模块的待处理消息
        
        Args:
            module_id: 模块ID
            max_count: 最多返回的消息数量
            
        Returns:
            list: 消息列表
        """
        with self.lock:
            if module_id not in self.message_queues:
                return []
                
            # 获取模块的所有队列
            module_queues = self.message_queues[module_id]
            
            # 按优先级从高到低获取消息
            messages = []
            priorities = sorted(module_queues.keys(), reverse=True)
            
            # 从每个优先级队列中获取消息
            remaining = max_count
            for priority in priorities:
                queue = module_queues[priority]
                
                # 获取当前优先级的消息
                count = min(remaining, len(queue))
                if count > 0:
                    messages.extend(queue[:count])
                    remaining -= count
                    
                if remaining <= 0:
                    break
                    
            return messages
    
    def acknowledge_message(self, message_id, success=True):
        """确认消息处理完成
        
        Args:
            message_id: 消息ID
            success: 处理是否成功
            
        Returns:
            bool: 确认是否成功
        """
        with self.lock:
            # 查找消息并从队列中移除
            for module_id, module_queues in self.message_queues.items():
                for priority, queue in module_queues.items():
                    for i, message in enumerate(queue):
                        if message["id"] == message_id:
                            # 标记为已送达
                            message["delivered"] = True
                            message["delivery_time"] = datetime.now()
                            
                            # 更新统计
                            if success:
                                self.stats["successful_deliveries"] += 1
                                
                                # 计算延迟
                                latency = (message["delivery_time"] - message["created_at"]).total_seconds()
                                self.stats["average_latency"] = (self.stats["average_latency"] * (self.stats["successful_deliveries"] - 1) + latency) / self.stats["successful_deliveries"]
                            else:
                                self.stats["failed_deliveries"] += 1
                            
                            # 添加到历史
                            self.message_history.append(message)
                            if len(self.message_history) > self.max_history_length:
                                self.message_history.pop(0)
                                
                            # 从队列移除
                            queue.pop(i)
                            
                            return True
            
            # 未找到消息
            self.logger.warning(f"未找到消息: {message_id}")
            return False
    
    def get_message_by_id(self, message_id):
        """根据ID获取消息
        
        Args:
            message_id: 消息ID
            
        Returns:
            dict: 消息，未找到则返回None
        """
        # 首先在历史记录中查找
        for message in self.message_history:
            if message["id"] == message_id:
                return message
                
        # 在队列中查找
        with self.lock:
            for module_id, module_queues in self.message_queues.items():
                for priority, queue in module_queues.items():
                    for message in queue:
                        if message["id"] == message_id:
                            return message
                            
        return None
    
    def clear_outdated_messages(self, max_age_seconds=300):
        """清理过期的消息
        
        Args:
            max_age_seconds: 最大允许的消息存活时间(秒)
            
        Returns:
            int: 清理的消息数量
        """
        cleared_count = 0
        now = datetime.now()
        
        with self.lock:
            for module_id, module_queues in self.message_queues.items():
                for priority, queue in list(module_queues.items()):
                    # 找出过期的消息
                    outdated = []
                    for i, message in enumerate(queue):
                        age = (now - message["created_at"]).total_seconds()
                        if age > max_age_seconds:
                            outdated.append(i)
                            
                    # 从后向前删除，避免索引混乱
                    for i in sorted(outdated, reverse=True):
                        # 添加到历史记录
                        message = queue[i]
                        message["delivered"] = False
                        self.message_history.append(message)
                        
                        # 从队列中删除
                        queue.pop(i)
                        cleared_count += 1
                        
                        # 更新统计
                        self.stats["failed_deliveries"] += 1
                        
            # 清理历史记录中的过早消息
            while len(self.message_history) > self.max_history_length:
                self.message_history.pop(0)
                
        return cleared_count
    
    def _run_transmission(self):
        """运行消息传输处理线程"""
        self.logger.info("启动超维度信息传输线程")
        
        while self.active:
            try:
                # 处理间隔
                time.sleep(0.1)
                
                with self.lock:
                    if not self.active or not self.protocol_state["active"]:
                        break
                        
                    # 更新协议状态
                    self._update_protocol_state()
                    
                    # 清理过期消息
                    if random.random() < 0.05:  # 5%的概率执行清理
                        self.clear_outdated_messages()
                    
                    # 处理消息传输过程
                    for module_id, module_queues in self.message_queues.items():
                        priorities = sorted(module_queues.keys(), reverse=True)
                        
                        # 按优先级处理队列
                        for priority in priorities:
                            queue = module_queues[priority]
                            
                            # 防止处理过多消息导致阻塞
                            max_process = min(5, len(queue))
                            processed = 0
                            
                            for i in range(min(max_process, len(queue))):
                                if i >= len(queue):  # 安全检查
                                    break
                                    
                                message = queue[i]
                                
                                # 模拟消息处理
                                processing_chance = (
                                    self.protocol_state["transmission_quality"] * 
                                    self.protocol_state["coherence_level"] * 
                                    (1 - self.protocol_state["error_rate"]) * 
                                    (priority / 10)  # 优先级影响
                                )
                                
                                # 维度对处理的影响
                                dim_factor = sum(self.dimension_channels[d]["stability"] for d in message["dimensions"]) / len(message["dimensions"])
                                processing_chance *= dim_factor
                                
                                # 检查是否传输成功
                                if random.random() < processing_chance:
                                    # 成功处理消息
                                    if self.symbiotic_core:
                                        # 通过共生核心发送
                                        success = self.symbiotic_core.send_message(
                                            message["source"], 
                                            message["target"], 
                                            message["type_name"], 
                                            message["data"]
                                        )
                                    else:
                                        # 模拟传输成功
                                        success = True
                                        
                                    # 确认消息
                                    self.acknowledge_message(message["id"], success)
                                    processed += 1
                                else:
                                    # 传输失败，保留在队列中，但增加错误计数
                                    error_type = "transmission_error"
                                    self.stats["error_counts"][error_type] += 1
                            
                            if processed > 0:
                                self.logger.debug(f"处理了模块 {module_id} 的 {processed} 条优先级 {priority} 的消息")
                
            except Exception as e:
                self.logger.error(f"超维度信息传输线程发生错误: {str(e)}")
                time.sleep(1)  # 错误恢复等待
        
        self.logger.info("超维度信息传输线程已停止")
    
    def _update_protocol_state(self):
        """更新协议状态"""
        # 模拟协议状态波动
        transmission_fluctuation = random.uniform(-0.03, 0.03)
        self.protocol_state["transmission_quality"] = max(0.3, min(1.0, 
            self.protocol_state["transmission_quality"] + transmission_fluctuation))
            
        coherence_fluctuation = random.uniform(-0.02, 0.02)
        self.protocol_state["coherence_level"] = max(0.3, min(1.0, 
            self.protocol_state["coherence_level"] + coherence_fluctuation))
            
        error_fluctuation = random.uniform(-0.01, 0.01)
        self.protocol_state["error_rate"] = max(0.01, min(0.2, 
            self.protocol_state["error_rate"] + error_fluctuation))
            
        # 随机维度变化
        if random.random() < 0.02:  # 2%几率
            # 当前覆盖维度
            current_dims = set(self.protocol_state["dimensional_coverage"])
            
            # 可用维度范围
            available_dims = set(range(5, 13))
            
            # 可以添加的维度
            can_add = available_dims - current_dims
            
            # 可以移除的维度 (至少保留3个维度)
            can_remove = current_dims if len(current_dims) > 3 else set()
            
            # 随机选择操作: 添加、移除或交换维度
            op = random.choice(["add", "remove", "swap"])
            
            if op == "add" and can_add:
                # 添加一个新维度
                new_dim = random.choice(list(can_add))
                self.protocol_state["dimensional_coverage"].append(new_dim)
                self.dimension_channels[new_dim]["active"] = True
                self.logger.info(f"添加了新的维度通道: {new_dim}")
                
            elif op == "remove" and can_remove:
                # 移除一个维度
                remove_dim = random.choice(list(can_remove))
                self.protocol_state["dimensional_coverage"].remove(remove_dim)
                self.dimension_channels[remove_dim]["active"] = False
                self.logger.info(f"关闭了维度通道: {remove_dim}")
                
            elif op == "swap" and can_add:
                # 交换一个维度
                if can_remove:
                    old_dim = random.choice(list(can_remove))
                    new_dim = random.choice(list(can_add))
                    
                    self.protocol_state["dimensional_coverage"].remove(old_dim)
                    self.protocol_state["dimensional_coverage"].append(new_dim)
                    
                    self.dimension_channels[old_dim]["active"] = False
                    self.dimension_channels[new_dim]["active"] = True
                    
                    self.logger.info(f"交换维度通道: {old_dim} -> {new_dim}")
                elif can_add:
                    # 只能添加
                    new_dim = random.choice(list(can_add))
                    self.protocol_state["dimensional_coverage"].append(new_dim)
                    self.dimension_channels[new_dim]["active"] = True
                    self.logger.info(f"添加了新的维度通道: {new_dim}")
        
        # 更新通道状态
        for dim, channel in self.dimension_channels.items():
            if channel["active"]:
                # 波动带宽
                bandwidth_fluctuation = random.uniform(-0.02, 0.02)
                channel["bandwidth"] = max(0.1, min(1.0, channel["bandwidth"] + bandwidth_fluctuation))
                
                # 波动稳定性
                stability_fluctuation = random.uniform(-0.02, 0.02)
                channel["stability"] = max(0.2, min(1.0, channel["stability"] + stability_fluctuation))
                
                # 波动噪声
                noise_fluctuation = random.uniform(-0.01, 0.01)
                channel["noise_level"] = max(0.05, min(0.5, channel["noise_level"] + noise_fluctuation))
                
        # 更新时间
        self.protocol_state["last_update"] = datetime.now()
    
    def get_protocol_status(self):
        """获取协议状态报告
        
        Returns:
            dict: 协议状态报告
        """
        status = {
            "active": self.protocol_state["active"],
            "transmission_quality": self.protocol_state["transmission_quality"],
            "coherence_level": self.protocol_state["coherence_level"],
            "error_rate": self.protocol_state["error_rate"],
            "dimensional_coverage": self.protocol_state["dimensional_coverage"],
            "active_dimensions": [d for d in self.dimension_channels if self.dimension_channels[d]["active"]],
            "dimension_channels": {d: ch for d, ch in self.dimension_channels.items() if ch["active"]},
            "stats": {
                "total_messages": self.stats["total_messages"],
                "successful_deliveries": self.stats["successful_deliveries"],
                "failed_deliveries": self.stats["failed_deliveries"],
                "average_latency": self.stats["average_latency"],
                "message_type_distribution": dict(self.stats["message_type_counts"]),
                "dimension_usage": dict(self.stats["dimension_usage"]),
                "error_counts": dict(self.stats["error_counts"])
            },
            "queued_messages": sum(len(queue) for module_queues in self.message_queues.values() 
                                 for queue in module_queues.values()),
            "message_history_count": len(self.message_history),
            "last_update": self.protocol_state["last_update"]
        }
        
        return status
    
    def add_dimension(self, properties=None):
        """添加新维度到协议中
        
        Args:
            properties: 维度属性字典，如果为None则使用默认属性
            
        Returns:
            int: 新增维度的ID，如果失败则返回-1
        """
        with self.lock:
            # 获取新维度ID
            new_dim_id = max(self.protocol_state["dimensional_coverage"]) + 1 if self.protocol_state["dimensional_coverage"] else 0
            
            # 创建维度属性
            if properties is None:
                properties = self._create_dimension_properties()
            else:
                # 确保必要属性存在
                default_props = self._create_dimension_properties()
                for key in default_props:
                    if key not in properties:
                        properties[key] = default_props[key]
            
            # 注册新维度
            self.protocol_state["dimensional_coverage"].append(new_dim_id)
            self.dimension_channels[new_dim_id] = properties
            
            # 更新维度总数
            self.protocol_state["dimensional_coverage"] = self.protocol_state["dimensional_coverage"]
            
            self.logger.info(f"添加新维度: {new_dim_id}，当前维度: {len(self.protocol_state['dimensional_coverage'])}")
            return new_dim_id
            
    def remove_dimension(self, dimension_id):
        """从协议中移除特定维度
        
        Args:
            dimension_id: 要移除的维度ID
            
        Returns:
            bool: 是否成功移除
        """
        with self.lock:
            if dimension_id not in self.protocol_state["dimensional_coverage"]:
                self.logger.warning(f"尝试移除不存在的维度: {dimension_id}")
                return False
                
            # 至少保留一个维度
            if len(self.protocol_state["dimensional_coverage"]) <= 1:
                self.logger.warning("无法移除最后一个维度")
                return False
                
            # 移除维度
            self.protocol_state["dimensional_coverage"].remove(dimension_id)
            
            # 保留维度属性以便可能的恢复
            # 但标记为非活动
            if dimension_id in self.dimension_channels:
                self.dimension_channels[dimension_id]['active'] = False
                self.dimension_channels[dimension_id]['removed_time'] = datetime.now()
                
            # 更新维度总数
            self.protocol_state["dimensional_coverage"] = self.protocol_state["dimensional_coverage"]
            
            self.logger.info(f"移除维度: {dimension_id}，当前维度: {len(self.protocol_state['dimensional_coverage'])}")
            return True
            
    def _create_dimension_properties(self):
        """创建默认维度属性
        
        Returns:
            dict: 维度属性字典
        """
        return {
            'active': True,
            'entropy': random.uniform(0.1, 0.5),
            'stability': random.uniform(0.7, 0.9),
            'energy_level': random.uniform(0.3, 0.8),
            'information_density': random.uniform(0.2, 0.6),
            'resonance_frequency': random.uniform(0.1, 1.0),
            'phase_variance': random.uniform(0, 0.3),
            'creation_time': datetime.now(),
            'last_update': datetime.now()
        }
        
def get_hyperdimensional_protocol(symbiotic_core=None):
    """获取超维度协议实例
    
    Args:
        symbiotic_core: 量子共生核心实例
        
    Returns:
        HyperdimensionalProtocol: 协议实例
    """
    return HyperdimensionalProtocol(symbiotic_core) 