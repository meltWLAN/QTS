#!/usr/bin/env python3
"""
共振场 - 超神系统宇宙共振模块的场域组件

管理高维共振场，提供市场环境感知和能量交互能力
"""

import logging
import uuid
import time
import threading
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

class ResonanceField:
    """共振场类
    
    创建和管理高维共振场，用于市场能量和信息的收集与分析
    """
    
    def __init__(self, 
                field_id: str = None, 
                creator_id: str = None,
                config: Dict[str, Any] = None):
        """初始化共振场
        
        Args:
            field_id: 共振场ID
            creator_id: 创建者ID
            config: 配置参数
        """
        self.logger = logging.getLogger("ResonanceField")
        
        # 生成共振场ID (如果未提供)
        self.field_id = field_id if field_id else str(uuid.uuid4())
        self.creator_id = creator_id
        
        # 默认配置
        self.default_config = {
            "dimension": 9,               # 场维度
            "radius": 5.0,                # 场半径
            "energy_decay": 0.01,         # 能量衰减率
            "stability_factor": 0.85,     # 稳定性因子
            "resonance_frequency": 7.83,  # 共振频率 (Hz)
            "harmonic_levels": 5,         # 谐波级别
            "field_memory": True,         # 场记忆功能
            "auto_harmonize": True,       # 自动谐波调整
        }
        
        # 合并配置
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
            
        # 初始化场状态
        self.state = {
            "active": True,
            "creation_time": datetime.now(),
            "energy_level": 0.5,          # 初始能量水平
            "harmonization": 0.0,         # 谐波化程度
            "stability": self.config["stability_factor"],
            "frequency": self.config["resonance_frequency"],
            "radius": self.config["radius"],
            "information_density": 0.0,
            "entities_count": 0,
            "last_update": datetime.now()
        }
        
        # 场内实体
        self.entities = []
        
        # 场记忆 (如果启用)
        self.memory = [] if self.config["field_memory"] else None
        
        # 共振谐波
        self.harmonics = []
        self.initialize_harmonics()
        
        # 能量扰动
        self.perturbations = []
        
        # 维护线程
        self.maintenance_thread = None
        self.maintenance_active = False
        
        self.logger.info(f"共振场初始化完成 (ID: {self.field_id})")
        
    def activate(self) -> bool:
        """激活共振场
        
        Returns:
            bool: 是否成功激活
        """
        if self.state["active"]:
            self.logger.warning("共振场已处于活动状态")
            return False
            
        try:
            self.logger.info("激活共振场...")
            
            # 设置状态
            self.state["active"] = True
            
            # 启动维护线程
            self.maintenance_active = True
            self.maintenance_thread = threading.Thread(target=self._maintenance_worker)
            self.maintenance_thread.daemon = True
            self.maintenance_thread.start()
            
            return True
            
        except Exception as e:
            self.logger.error(f"激活共振场失败: {str(e)}")
            self.state["active"] = False
            return False
            
    def deactivate(self) -> bool:
        """停用共振场
        
        Returns:
            bool: 是否成功停用
        """
        if not self.state["active"]:
            self.logger.warning("共振场已处于停用状态")
            return False
            
        try:
            self.logger.info("停用共振场...")
            
            # 设置停用标志
            self.maintenance_active = False
            
            # 等待线程结束
            if self.maintenance_thread and self.maintenance_thread.is_alive():
                self.maintenance_thread.join(timeout=2.0)
                
            # 更新状态
            self.state["active"] = False
            
            return True
            
        except Exception as e:
            self.logger.error(f"停用共振场失败: {str(e)}")
            return False
    
    def initialize_harmonics(self) -> None:
        """初始化共振谐波"""
        self.harmonics = []
        base_frequency = self.config["resonance_frequency"]
        
        # 创建谐波
        for i in range(self.config["harmonic_levels"]):
            harmonic = {
                "level": i + 1,
                "frequency": base_frequency * (i + 1),
                "amplitude": 1.0 / (i + 1),
                "phase": 2 * np.pi * np.random.random(),
                "energy": 0.5 / (i + 1),
                "active": i == 0  # 只有基本谐波一开始是活跃的
            }
            self.harmonics.append(harmonic)
            
    def add_entity(self, entity: Dict[str, Any]) -> str:
        """添加实体到共振场
        
        Args:
            entity: 实体数据
            
        Returns:
            str: 实体ID
        """
        if not self.state["active"]:
            self.logger.warning("共振场未激活，无法添加实体")
            return None
            
        # 确保实体有ID
        if "id" not in entity:
            entity["id"] = str(uuid.uuid4())
            
        # 添加时间戳
        entity["added_at"] = datetime.now().isoformat()
        
        # 添加到实体列表
        self.entities.append(entity)
        
        # 更新计数
        self.state["entities_count"] = len(self.entities)
        
        self.logger.debug(f"实体添加到共振场 (ID: {entity['id']})")
        return entity["id"]
        
    def remove_entity(self, entity_id: str) -> bool:
        """从共振场移除实体
        
        Args:
            entity_id: 实体ID
            
        Returns:
            bool: 是否成功移除
        """
        for i, entity in enumerate(self.entities):
            if entity["id"] == entity_id:
                self.entities.pop(i)
                
                # 更新计数
                self.state["entities_count"] = len(self.entities)
                
                self.logger.debug(f"实体从共振场移除 (ID: {entity_id})")
                return True
                
        self.logger.warning(f"未找到实体 (ID: {entity_id})")
        return False
        
    def add_perturbation(self, perturbation: Dict[str, Any]) -> str:
        """添加能量扰动
        
        Args:
            perturbation: 扰动数据
            
        Returns:
            str: 扰动ID
        """
        # 确保扰动有ID
        if "id" not in perturbation:
            perturbation["id"] = str(uuid.uuid4())
            
        # 添加时间戳
        perturbation["created_at"] = datetime.now().isoformat()
        
        # 默认持续时间 (如果未指定)
        if "duration" not in perturbation:
            perturbation["duration"] = 60.0  # 60秒
            
        # 计算过期时间
        now = datetime.now()
        expiry = now + timedelta(seconds=perturbation["duration"])
        perturbation["expires_at"] = expiry.isoformat()
        
        # 添加到扰动列表
        self.perturbations.append(perturbation)
        
        # 立即应用扰动效果
        self._apply_perturbation(perturbation)
        
        self.logger.debug(f"能量扰动添加到共振场 (ID: {perturbation['id']})")
        return perturbation["id"]
        
    def _apply_perturbation(self, perturbation: Dict[str, Any]) -> None:
        """应用扰动效果
        
        Args:
            perturbation: 扰动数据
        """
        # 根据扰动类型应用不同效果
        p_type = perturbation.get("type", "energy")
        strength = perturbation.get("strength", 0.5)
        
        if p_type == "energy":
            # 影响场能量
            change = strength * 0.2
            self.state["energy_level"] = max(0.0, min(1.0, self.state["energy_level"] + change))
            
        elif p_type == "frequency":
            # 影响场频率
            change = strength * 0.5
            current = self.state["frequency"]
            self.state["frequency"] = current * (1.0 + change)
            
        elif p_type == "stability":
            # 影响场稳定性
            change = -strength * 0.15  # 扰动通常降低稳定性
            self.state["stability"] = max(0.1, min(1.0, self.state["stability"] + change))
            
        elif p_type == "harmonic":
            # 影响场谐波
            target_level = perturbation.get("harmonic_level", 1)
            
            # 找到目标谐波
            for harmonic in self.harmonics:
                if harmonic["level"] == target_level:
                    # 增加谐波能量
                    harmonic["energy"] = min(1.0, harmonic["energy"] + strength * 0.3)
                    
                    # 激活谐波
                    harmonic["active"] = True
                    break
        
    def get_state(self) -> Dict[str, Any]:
        """获取共振场状态
        
        Returns:
            Dict[str, Any]: 状态数据
        """
        # 更新最后访问时间
        self.state["last_update"] = datetime.now()
        
        # 复制当前状态
        state = self.state.copy()
        
        # 添加场年龄
        age = (datetime.now() - self.state["creation_time"]).total_seconds()
        state["age_seconds"] = age
        
        # 添加活跃谐波数
        active_harmonics = sum(1 for h in self.harmonics if h["active"])
        state["active_harmonics"] = active_harmonics
        
        # 添加场标识信息
        state["field_id"] = self.field_id
        state["creator_id"] = self.creator_id
        
        return state
        
    def scan(self, depth: float = 0.5) -> Dict[str, Any]:
        """扫描共振场
        
        Args:
            depth: 扫描深度 (0-1)
            
        Returns:
            Dict[str, Any]: 扫描结果
        """
        if not self.state["active"]:
            self.logger.warning("共振场未激活，无法扫描")
            return {"success": False, "error": "field_inactive"}
            
        try:
            self.logger.info(f"扫描共振场 (深度: {depth:.2f})...")
            
            # 扫描结果
            scan_result = {
                "timestamp": datetime.now().isoformat(),
                "field_id": self.field_id,
                "scan_depth": depth,
                "energy_map": [],
                "harmonic_analysis": [],
                "entity_count": len(self.entities),
                "perturbation_count": len(self.perturbations),
                "field_state": self.get_state()
            }
            
            # 生成能量地图
            scan_points = int(10 + 30 * depth)
            energy_map = []
            
            for _ in range(scan_points):
                # 随机坐标
                x = (np.random.random() * 2 - 1) * self.state["radius"]
                y = (np.random.random() * 2 - 1) * self.state["radius"]
                z = (np.random.random() * 2 - 1) * self.state["radius"]
                
                # 计算到中心的距离
                distance = np.sqrt(x**2 + y**2 + z**2)
                
                # 计算点能量 (随距离衰减)
                energy = self.state["energy_level"] * np.exp(-distance / self.state["radius"])
                
                # 添加随机变化
                energy *= 0.8 + 0.4 * np.random.random()
                
                # 添加到地图
                energy_map.append({
                    "x": x,
                    "y": y,
                    "z": z,
                    "energy": energy,
                    "distance": distance
                })
                
            scan_result["energy_map"] = energy_map
            
            # 谐波分析
            for harmonic in self.harmonics:
                if harmonic["active"]:
                    analysis = {
                        "level": harmonic["level"],
                        "frequency": harmonic["frequency"],
                        "amplitude": harmonic["amplitude"],
                        "energy": harmonic["energy"],
                        "phase": harmonic["phase"],
                        "purity": 0.5 + 0.5 * np.random.random(),
                        "stability": 0.4 + 0.6 * self.state["stability"] * np.random.random()
                    }
                    scan_result["harmonic_analysis"].append(analysis)
                    
            # 计算平均能量
            if energy_map:
                avg_energy = sum(p["energy"] for p in energy_map) / len(energy_map)
                scan_result["average_energy"] = avg_energy
                
            # 如果深度足够大，添加实体抽样
            if depth > 0.7 and self.entities:
                # 抽取样本实体
                sample_size = min(5, len(self.entities))
                entity_sample = np.random.choice(self.entities, size=sample_size, replace=False)
                scan_result["entity_sample"] = entity_sample.tolist()
                
            return scan_result
            
        except Exception as e:
            self.logger.error(f"扫描共振场失败: {str(e)}")
            return {"success": False, "error": str(e)}
            
    def harmonize(self, target_frequency: float = None) -> Dict[str, Any]:
        """谐波化共振场频率
        
        Args:
            target_frequency: 目标频率
            
        Returns:
            Dict[str, Any]: 谐波化结果
        """
        if not self.state["active"]:
            self.logger.warning("共振场未激活，无法谐波化")
            return {"success": False, "error": "field_inactive"}
            
        self.logger.info("谐波化共振场...")
        
        # 如果未指定目标频率，使用当前频率
        if target_frequency is None:
            target_frequency = self.state["frequency"]
            
        try:
            # 记录开始状态
            initial_harmonization = self.state["harmonization"]
            initial_frequency = self.state["frequency"]
            
            # 计算频率差异
            freq_diff = abs(target_frequency - initial_frequency) / initial_frequency
            
            # 根据频率差异和稳定性计算谐波化难度
            difficulty = freq_diff / self.state["stability"]
            
            # 计算谐波化结果
            success_probability = np.exp(-difficulty * 3)
            success_roll = np.random.random()
            
            # 谐波化结果
            result = {
                "timestamp": datetime.now().isoformat(),
                "initial_frequency": initial_frequency,
                "target_frequency": target_frequency,
                "initial_harmonization": initial_harmonization,
                "difficulty": difficulty,
                "success_probability": success_probability
            }
            
            if success_roll < success_probability:
                # 谐波化成功
                # 计算达成度
                achievement = 0.5 + 0.5 * (1 - difficulty) * np.random.random()
                
                # 应用部分频率变化
                new_frequency = initial_frequency + achievement * (target_frequency - initial_frequency)
                self.state["frequency"] = new_frequency
                
                # 增加谐波化程度
                harmonization_gain = 0.1 + 0.2 * achievement
                new_harmonization = min(1.0, initial_harmonization + harmonization_gain)
                self.state["harmonization"] = new_harmonization
                
                # 更新谐波
                self._update_harmonics()
                
                # 完成结果
                result.update({
                    "success": True,
                    "achievement": achievement,
                    "new_frequency": new_frequency,
                    "new_harmonization": new_harmonization,
                    "message": "谐波化成功"
                })
                
            else:
                # 谐波化失败
                # 小幅频率变化
                random_shift = 0.02 * (2 * np.random.random() - 1)
                new_frequency = initial_frequency * (1 + random_shift)
                self.state["frequency"] = new_frequency
                
                # 降低稳定性
                stability_penalty = 0.05 + 0.1 * np.random.random()
                self.state["stability"] = max(0.1, self.state["stability"] - stability_penalty)
                
                # 完成结果
                result.update({
                    "success": False,
                    "new_frequency": new_frequency,
                    "stability_penalty": stability_penalty,
                    "message": "谐波化失败"
                })
                
            return result
            
        except Exception as e:
            self.logger.error(f"谐波化共振场失败: {str(e)}")
            return {"success": False, "error": str(e)}
            
    def _update_harmonics(self) -> None:
        """更新谐波状态"""
        base_frequency = self.state["frequency"]
        harmonization = self.state["harmonization"]
        
        # 更新每个谐波
        for harmonic in self.harmonics:
            level = harmonic["level"]
            
            # 更新频率
            harmonic["frequency"] = base_frequency * level
            
            # 根据谐波化程度决定谐波是否活跃
            activation_threshold = 0.2 * level
            harmonic["active"] = harmonization > activation_threshold
            
            if harmonic["active"]:
                # 更新振幅
                harmonic["amplitude"] = (1.0 / level) * (0.5 + 0.5 * harmonization)
                
                # 更新能量 (随谐波化程度增加)
                energy_base = 1.0 / level
                energy_boost = harmonization * (1.0 - 1.0 / level)
                harmonic["energy"] = energy_base + energy_boost
            else:
                # 不活跃谐波能量衰减
                harmonic["energy"] *= 0.9
                
    def record_memory(self, data: Dict[str, Any]) -> bool:
        """记录场记忆
        
        Args:
            data: 记忆数据
            
        Returns:
            bool: 是否成功记录
        """
        if not self.config["field_memory"]:
            self.logger.warning("场记忆功能未启用")
            return False
            
        if not self.memory:
            self.memory = []
            
        # 添加时间戳
        if "timestamp" not in data:
            data["timestamp"] = datetime.now().isoformat()
            
        # 添加记忆ID
        if "id" not in data:
            data["id"] = str(uuid.uuid4())
            
        # 添加到记忆
        self.memory.append(data)
        
        # 限制记忆大小
        max_memories = 1000
        if len(self.memory) > max_memories:
            self.memory = self.memory[-max_memories:]
            
        # 增加信息密度
        self.state["information_density"] = min(1.0, self.state["information_density"] + 0.01)
        
        return True
        
    def retrieve_memories(self, filter_criteria: Dict[str, Any] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """检索场记忆
        
        Args:
            filter_criteria: 过滤条件
            limit: 最大记录数
            
        Returns:
            List[Dict[str, Any]]: 记忆列表
        """
        if not self.config["field_memory"] or not self.memory:
            return []
            
        # 如果没有过滤条件，返回最近的记忆
        if not filter_criteria:
            return self.memory[-limit:]
            
        # 过滤记忆
        filtered_memories = []
        
        for memory in self.memory:
            matches = True
            
            # 检查每个过滤条件
            for key, value in filter_criteria.items():
                if key not in memory or memory[key] != value:
                    matches = False
                    break
                    
            if matches:
                filtered_memories.append(memory)
                
                # 达到限制时停止
                if len(filtered_memories) >= limit:
                    break
                    
        return filtered_memories
        
    def _maintenance_worker(self) -> None:
        """场维护工作线程"""
        self.logger.debug("共振场维护线程启动")
        
        cycle_count = 0
        last_harmonize_time = time.time()
        
        while self.maintenance_active:
            try:
                # 能量自然衰减
                energy_decay = self.config["energy_decay"] * (0.8 + 0.4 * np.random.random())
                self.state["energy_level"] = max(0.1, self.state["energy_level"] * (1 - energy_decay))
                
                # 处理过期扰动
                now = datetime.now()
                expired = []
                
                for i, p in enumerate(self.perturbations):
                    expires_at = datetime.fromisoformat(p["expires_at"])
                    if now > expires_at:
                        expired.append(i)
                        
                # 从后向前移除
                for i in sorted(expired, reverse=True):
                    self.perturbations.pop(i)
                    
                # 随机稳定性变化
                stability_change = 0.02 * (np.random.random() - 0.5)
                self.state["stability"] = max(0.1, min(1.0, self.state["stability"] + stability_change))
                
                # 自动谐波化
                if self.config["auto_harmonize"] and time.time() - last_harmonize_time > 30:
                    # 随机微调频率
                    target = self.state["frequency"] * (0.98 + 0.04 * np.random.random())
                    self.harmonize(target)
                    last_harmonize_time = time.time()
                    
                # 信息密度自然衰减
                if self.state["information_density"] > 0:
                    info_decay = 0.005 * (0.8 + 0.4 * np.random.random())
                    self.state["information_density"] = max(0.0, self.state["information_density"] - info_decay)
                    
                # 睡眠一段时间
                time.sleep(1.0)
                
                # 循环计数
                cycle_count += 1
                
            except Exception as e:
                self.logger.error(f"共振场维护线程错误: {str(e)}")
                time.sleep(5.0)  # 出错后等待较长时间
                
        self.logger.debug("共振场维护线程结束")


def create_resonance_field(field_dimensions=7, field_strength=0.6):
    """创建共振场实例
    
    Args:
        field_dimensions: 场维度数 (建议3-12)
        field_strength: 初始场强度 (0-1)
        
    Returns:
        ResonanceField: 共振场实例
    """
    field = ResonanceField(field_dimensions, field_strength)
    return field 