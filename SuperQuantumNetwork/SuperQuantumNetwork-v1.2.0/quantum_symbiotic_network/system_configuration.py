#!/usr/bin/env python3
"""
超神量子共生系统 - 高维配置管理模块
实现系统维度调节、能量分配和启动序列
"""

import os
import json
import uuid
import logging
import threading
import random
import numpy as np
from datetime import datetime
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

class QuantumSystemConfig:
    """量子共生系统配置管理器
    
    负责管理系统维度、能量分配、模块配置和启动序列，
    提供自适应调节能力，响应系统共振状态。
    """
    
    _instance = None  # 单例模式
    
    @classmethod
    def get_instance(cls):
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = QuantumSystemConfig()
        return cls._instance
    
    def __init__(self):
        """初始化量子配置管理器"""
        # 防止重复初始化
        if QuantumSystemConfig._instance is not None:
            raise Exception("QuantumSystemConfig是单例类，请使用get_instance()获取实例")
            
        QuantumSystemConfig._instance = self
        
        # 设置日志
        self.logger = logging.getLogger("QuantumSystemConfig")
        self.logger.info("初始化量子配置管理器...")
        
        # 配置文件路径
        self.config_dir = Path(__file__).parent.parent / "config"
        self.config_file = self.config_dir / "quantum_config.json"
        
        # 创建配置目录
        self.config_dir.mkdir(exist_ok=True)
        
        # 系统ID
        self.system_id = str(uuid.uuid4())
        
        # 默认配置
        self.default_config = {
            "system": {
                "system_id": self.system_id,
                "version": "0.1.0",
                "name": "超神量子共生系统",
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat()
            },
            "dimensions": {
                "base_dimensions": 9,
                "adaptive_dimensions": True,
                "min_dimensions": 5,
                "max_dimensions": 12,
                "dimension_adaptation_rate": 0.05
            },
            "energy": {
                "initial_core_energy": 0.7,
                "initial_field_strength": 0.5,
                "energy_conservation": True,
                "energy_distribution": {
                    "core": 0.4,
                    "prediction": 0.3,
                    "data": 0.2,
                    "trading": 0.1
                }
            },
            "modules": {
                "core": {
                    "enabled": True,
                    "consciousness_level": 0.6,
                    "evolution_stage": 1
                },
                "data_sources": {
                    "tushare": {
                        "enabled": True,
                        "token": "",
                        "quantum_energy_level": 0.5,
                        "auto_connect": True
                    }
                },
                "prediction": {
                    "enabled": False,
                    "quantum_consciousness": 0.4,
                    "prediction_dimensions": 7
                },
                "trading": {
                    "enabled": False,
                    "risk_level": 0.3,
                    "auto_execution": False
                }
            },
            "initialization": {
                "sequence": [
                    "core",
                    "field_activation",
                    "data_sources",
                    "prediction",
                    "trading"
                ],
                "auto_start": False,
                "stabilization_period": 10
            },
            "resonance": {
                "target_frequency": 0.7,
                "adaptation_speed": 0.3,
                "consciousness_amplification": 0.5
            },
            "quantum_settings": {
                "entanglement_threshold": 0.3,
                "coherence_target": 0.8,
                "quantum_noise": 0.05,
                "consciousness_emergence_threshold": 0.7
            },
            "advanced": {
                "debug_mode": False,
                "simulation_mode": True,
                "time_dilation": 1.0,
                "interdimensional_bridges": 3
            }
        }
        
        # 当前配置
        self.config = {}
        
        # 变更记录
        self.config_history = []
        
        # 维度状态
        self.dimension_state = {
            "current_dimensions": 9,
            "stable_dimensions": 9,
            "dimension_stability": 1.0,
            "last_adaptation": datetime.now()
        }
        
        # 模块引用
        self.core = None
        
        # 线程锁
        self.lock = threading.RLock()
        
        # 加载配置
        self._load_or_create_config()
        
    def _load_or_create_config(self):
        """加载或创建默认配置"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                self.logger.info(f"从 {self.config_file} 加载配置成功")
                
                # 记录版本
                loaded_version = self.config.get("system", {}).get("version", "0.0.0")
                self.logger.info(f"加载的配置版本: {loaded_version}")
                
                # 检查并更新缺失的配置项
                updated = self._update_missing_configs()
                if updated:
                    self.save_config()
            else:
                self.config = self.default_config.copy()
                self.save_config()
                self.logger.info(f"创建默认配置并保存至 {self.config_file}")
        except Exception as e:
            self.logger.error(f"加载配置失败: {str(e)}，将使用默认配置")
            self.config = self.default_config.copy()
    
    def _update_missing_configs(self):
        """更新缺失的配置项"""
        updated = False
        
        def _recursive_update(target, source):
            nonlocal updated
            for key, value in source.items():
                if key not in target:
                    target[key] = value
                    updated = True
                elif isinstance(value, dict) and isinstance(target[key], dict):
                    _recursive_update(target[key], value)
        
        _recursive_update(self.config, self.default_config)
        return updated
        
    def save_config(self):
        """保存当前配置到文件"""
        with self.lock:
            try:
                # 更新时间戳
                if "system" in self.config:
                    self.config["system"]["last_updated"] = datetime.now().isoformat()
                    
                # 确保目录存在
                self.config_dir.mkdir(exist_ok=True)
                
                # 保存配置
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, indent=4, ensure_ascii=False)
                    
                # 添加到历史记录
                self.config_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "config": self.config.copy()
                })
                
                # 仅保留最近10条历史记录
                if len(self.config_history) > 10:
                    self.config_history = self.config_history[-10:]
                    
                self.logger.info(f"配置已保存到 {self.config_file}")
                return True
            except Exception as e:
                self.logger.error(f"保存配置失败: {str(e)}")
                return False
    
    def get_config(self, section=None, key=None):
        """获取配置
        
        Args:
            section: 配置部分
            key: 配置键
            
        Returns:
            返回请求的配置值，如果不存在则返回None
        """
        with self.lock:
            if section is None:
                return self.config.copy()
                
            if section not in self.config:
                return None
                
            if key is None:
                return self.config[section].copy()
                
            if key not in self.config[section]:
                return None
                
            return self.config[section][key]
    
    def update_config(self, section, key, value):
        """更新配置项
        
        Args:
            section: 配置部分
            key: 配置键
            value: 新值
            
        Returns:
            bool: 是否更新成功
        """
        with self.lock:
            if section not in self.config:
                self.config[section] = {}
                
            # 记录旧值
            old_value = self.config[section].get(key)
            
            # 更新值
            self.config[section][key] = value
            
            # 记录变更
            self.logger.info(f"配置更新: {section}.{key} 从 {old_value} 变更为 {value}")
            
            # 保存配置
            return self.save_config()
    
    def connect_core(self, core):
        """连接到量子共生核心
        
        Args:
            core: 量子共生核心实例
            
        Returns:
            bool: 是否成功
        """
        self.core = core
        self.logger.info("已连接到量子共生核心")
        
        # 注册为模块
        module_id = "system_config"
        self.core.register_module(module_id, self, "system")
        
        return True
    
    def adapt_dimensions(self, resonance_state=None):
        """调整系统维度，响应共振状态
        
        Args:
            resonance_state: 共振状态信息
            
        Returns:
            dict: 调整后的维度状态
        """
        with self.lock:
            # 检查是否启用自适应维度
            if not self.config["dimensions"]["adaptive_dimensions"]:
                return self.dimension_state
                
            # 获取配置参数
            min_dims = self.config["dimensions"]["min_dimensions"]
            max_dims = self.config["dimensions"]["max_dimensions"]
            adaptation_rate = self.config["dimensions"]["dimension_adaptation_rate"]
            
            # 使用核心共振状态，如果有的话
            if resonance_state is None and self.core:
                resonance_state = self.core.resonance_state
                
            # 如果没有共振状态，使用随机波动
            if resonance_state is None:
                # 简单随机变化
                rand_change = random.choice([-1, 0, 0, 0, 1])
                new_dimensions = max(min_dims, min(max_dims, self.dimension_state["current_dimensions"] + rand_change))
                stability = random.uniform(0.7, 1.0)
            else:
                # 基于共振状态智能调整
                energy_level = resonance_state.get("energy_level", 0.5)
                coherence = resonance_state.get("coherence", 0.5)
                stability = resonance_state.get("stability", 0.5)
                
                # 计算理想维度
                # 能量和相干性越高，维度越高
                ideal_dimensions = min_dims + (max_dims - min_dims) * (energy_level * 0.7 + coherence * 0.3)
                
                # 加入随机性，但受稳定性控制
                rand_factor = (1 - stability) * 2  # 稳定性越低，随机性越大
                random_adjustment = random.uniform(-rand_factor, rand_factor)
                
                # 计算新维度
                target_dimensions = ideal_dimensions + random_adjustment
                current_dimensions = self.dimension_state["current_dimensions"]
                
                # 平滑变化，避免剧烈波动
                change = (target_dimensions - current_dimensions) * adaptation_rate
                new_dimensions_float = current_dimensions + change
                
                # 四舍五入到整数
                new_dimensions = round(new_dimensions_float)
                
                # 限制范围
                new_dimensions = max(min_dims, min(max_dims, new_dimensions))
                
            # 更新维度状态
            self.dimension_state["current_dimensions"] = new_dimensions
            self.dimension_state["dimension_stability"] = stability
            self.dimension_state["last_adaptation"] = datetime.now()
            
            # 更新核心字段状态
            if self.core and hasattr(self.core, 'field_state'):
                self.core.field_state["dimension_count"] = new_dimensions
                
            # 记录变更
            self.logger.info(f"系统维度已调整为: {new_dimensions} (稳定性: {stability:.2f})")
            
            return self.dimension_state
    
    def generate_initialization_sequence(self):
        """生成系统初始化序列
        
        Returns:
            list: 初始化步骤序列
        """
        sequence = self.config["initialization"]["sequence"].copy()
        
        # 检查是否启用各模块
        enabled_sequence = []
        for step in sequence:
            if step == "core" or step == "field_activation":
                enabled_sequence.append(step)
            elif step == "data_sources":
                # 检查是否有启用的数据源
                data_sources = self.config["modules"]["data_sources"]
                if any(data_sources[source]["enabled"] for source in data_sources):
                    enabled_sequence.append(step)
            elif step in self.config["modules"] and self.config["modules"][step]["enabled"]:
                enabled_sequence.append(step)
        
        # 添加额外步骤
        if self.config["quantum_settings"]["entanglement_threshold"] < 0.5:
            enabled_sequence.append("entanglement_balancing")
            
        return enabled_sequence
    
    def distribute_energy(self, total_energy=1.0):
        """分配系统能量到各个模块
        
        Args:
            total_energy: 总能量值
            
        Returns:
            dict: 能量分配结果
        """
        distribution = self.config["energy"]["energy_distribution"].copy()
        
        # 确保分配比例总和为1
        total_ratio = sum(distribution.values())
        if total_ratio == 0:
            # 避免除零错误
            distribution = {k: 1.0/len(distribution) for k in distribution}
        elif total_ratio != 1.0:
            # 归一化
            distribution = {k: v/total_ratio for k, v in distribution.items()}
            
        # 计算能量分配
        energy_allocation = {module: total_energy * ratio for module, ratio in distribution.items()}
        
        # 添加稍许随机波动
        if self.config["quantum_settings"]["quantum_noise"] > 0:
            noise_level = self.config["quantum_settings"]["quantum_noise"]
            for module in energy_allocation:
                noise = random.uniform(-noise_level, noise_level) * energy_allocation[module]
                energy_allocation[module] = max(0.05, energy_allocation[module] + noise)
                
        return energy_allocation
    
    def get_module_config(self, module_type, module_name=None):
        """获取特定模块的配置
        
        Args:
            module_type: 模块类型 (data_sources, prediction, trading, etc.)
            module_name: 模块名称，如果为None则返回整个类型配置
            
        Returns:
            dict: 模块配置
        """
        if module_type not in self.config["modules"]:
            return {}
            
        if module_name is None:
            return self.config["modules"][module_type].copy()
            
        if module_type == "data_sources":
            if module_name in self.config["modules"][module_type]:
                return self.config["modules"][module_type][module_name].copy()
        elif module_name == module_type:
            return self.config["modules"][module_type].copy()
            
        return {}
    
    def update_module_config(self, module_type, module_name, config_updates):
        """更新模块配置
        
        Args:
            module_type: 模块类型
            module_name: 模块名称
            config_updates: 要更新的配置字典
            
        Returns:
            bool: 是否成功
        """
        with self.lock:
            if module_type not in self.config["modules"]:
                self.config["modules"][module_type] = {}
                
            if module_type == "data_sources":
                if module_name not in self.config["modules"][module_type]:
                    self.config["modules"][module_type][module_name] = {}
                
                # 更新配置
                for key, value in config_updates.items():
                    self.config["modules"][module_type][module_name][key] = value
            elif module_name == module_type:
                # 更新配置
                for key, value in config_updates.items():
                    self.config["modules"][module_type][key] = value
            else:
                return False
                
            # 保存配置
            return self.save_config()
    
    def receive_message(self, from_module, message_type, data):
        """接收来自其他模块的消息
        
        Args:
            from_module: 源模块ID
            message_type: 消息类型
            data: 消息数据
            
        Returns:
            bool: 消息是否成功处理
        """
        if message_type == "resonance_update":
            # 更新维度状态
            self.adapt_dimensions(data)
            return True
            
        elif message_type == "config_request":
            # 处理配置请求
            if "module_type" in data and "module_name" in data:
                config = self.get_module_config(data["module_type"], data["module_name"])
                
                # 回复配置
                if self.core:
                    self.core.send_message(
                        "system_config",
                        from_module,
                        "config_response",
                        {
                            "module_type": data["module_type"],
                            "module_name": data["module_name"],
                            "config": config
                        }
                    )
                return True
                
        elif message_type == "config_update":
            # 处理配置更新请求
            if "module_type" in data and "module_name" in data and "config" in data:
                result = self.update_module_config(
                    data["module_type"],
                    data["module_name"],
                    data["config"]
                )
                return result
                
        return False
    
    def export_config(self, export_path=None):
        """导出配置到文件
        
        Args:
            export_path: 导出路径，默认为配置目录下的时间戳文件
            
        Returns:
            str: 导出文件路径
        """
        if export_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = self.config_dir / f"config_export_{timestamp}.json"
            
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            self.logger.info(f"配置已导出到: {export_path}")
            return str(export_path)
        except Exception as e:
            self.logger.error(f"导出配置失败: {str(e)}")
            return None
    
    def import_config(self, import_path):
        """从文件导入配置
        
        Args:
            import_path: 导入文件路径
            
        Returns:
            bool: 是否成功
        """
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                new_config = json.load(f)
                
            # 保存当前配置到历史
            self.config_history.append({
                "timestamp": datetime.now().isoformat(),
                "config": self.config.copy()
            })
            
            # 应用新配置
            self.config = new_config
            
            # 保存配置
            self.save_config()
            
            self.logger.info(f"从 {import_path} 导入配置成功")
            return True
        except Exception as e:
            self.logger.error(f"导入配置失败: {str(e)}")
            return False
    
    def get_system_status(self):
        """获取系统配置状态
        
        Returns:
            dict: 状态信息
        """
        return {
            "system_id": self.config["system"]["system_id"],
            "version": self.config["system"]["version"],
            "dimensions": self.dimension_state,
            "modules_enabled": {
                k: v.get("enabled", False) 
                for k, v in self.config["modules"].items() 
                if isinstance(v, dict) and "enabled" in v
            },
            "data_sources": {
                k: v.get("enabled", False)
                for k, v in self.config["modules"].get("data_sources", {}).items()
            },
            "auto_start": self.config["initialization"]["auto_start"],
            "initialization_sequence": self.generate_initialization_sequence(),
            "last_updated": self.config["system"]["last_updated"],
            "timestamp": datetime.now().isoformat()
        }

# 获取实例的辅助函数
def get_system_config():
    """获取系统配置实例"""
    return QuantumSystemConfig.get_instance()
