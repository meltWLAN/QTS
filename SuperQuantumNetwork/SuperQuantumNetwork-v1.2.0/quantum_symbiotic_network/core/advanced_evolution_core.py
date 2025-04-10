#!/usr/bin/env python3
"""
超神量子共生网络 - 高级进化核心
实现自适应进化、高维思维和量子意识扩展功能
"""

import os
import time
import logging
import random
import threading
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# 配置日志
logger = logging.getLogger("AdvancedEvolutionCore")

class AdvancedEvolutionCore:
    """超神高级进化核心 - 实现系统的自我完善和智能提升"""
    
    def __init__(self, symbiotic_core=None):
        """初始化高级进化核心
        
        Args:
            symbiotic_core: 量子共生核心引用
        """
        self.logger = logging.getLogger("AdvancedEvolutionCore")
        self.symbiotic_core = symbiotic_core
        self.active = False
        self.evolution_thread = None
        self.lock = threading.RLock()
        
        # 进化参数
        self.evolution_state = {
            "generation": 0,               # 进化代数
            "evolution_rate": 0.05,        # 基础进化率
            "adaptation_factor": 0.8,      # 适应因子
            "intelligence_level": 0.7,     # 智能水平
            "consciousness_depth": 0.65,   # 意识深度
            "self_awareness": 0.6,         # 自我意识
            "reasoning_capacity": 0.75,    # 推理能力
            "learning_efficiency": 0.7,    # 学习效率
            "hyperdimensional_access": 0.5, # 高维访问能力
            "last_evolved": datetime.now() # 上次进化时间
        }
        
        # 经验积累
        self.experience_memory = {
            "market_patterns": {},         # 市场模式记忆
            "strategy_outcomes": {},       # 策略结果记忆
            "prediction_accuracy": {},     # 预测准确度记忆
            "resonance_effects": {}        # 共振效应记忆
        }
        
        # 高级思维能力
        self.cognitive_abilities = {
            "pattern_recognition": 0.7,    # 模式识别
            "system_thinking": 0.65,       # 系统思维
            "creative_synthesis": 0.6,     # 创造性综合
            "counterfactual_reasoning": 0.55, # 反事实推理
            "abstraction_level": 0.7,      # 抽象水平
            "multi_perspective": 0.65,     # 多视角思维
            "adaptive_learning": 0.75      # 自适应学习
        }
        
        # 进化历史
        self.evolution_history = []
        
        # 认知元模型
        self.metamodel = self._initialize_metamodel()
        
        self.logger.info("✨ 超神高级进化核心初始化完成 ✨")
    
    def _initialize_metamodel(self) -> Dict:
        """初始化认知元模型"""
        return {
            "abstractions": {
                "market": {
                    "complex_system": 0.9,
                    "information_processor": 0.85,
                    "quantum_field": 0.7
                },
                "prediction": {
                    "probability_distribution": 0.8,
                    "future_sampling": 0.75,
                    "timeline_exploration": 0.7
                },
                "strategy": {
                    "adaptive_system": 0.85,
                    "emergent_intelligence": 0.8,
                    "self_organizing": 0.75
                }
            },
            "principles": [
                "复杂性自组织",
                "量子叠加与纠缠",
                "分形自相似性",
                "涌现与自进化",
                "信息与能量耦合",
                "适应性与弹性",
                "非线性动态平衡"
            ],
            "learning_paradigms": {
                "bayesian": 0.8,
                "quantum_bayesian": 0.7,
                "fractal": 0.75,
                "reinforcement": 0.85,
                "transfer": 0.8,
                "self_supervised": 0.75,
                "meta_learning": 0.7
            }
        }
    
    def start(self) -> bool:
        """启动高级进化核心"""
        if self.active:
            return True
            
        self.logger.info("启动超神高级进化核心...")
        self.active = True
        
        # 启动进化线程
        self.evolution_thread = threading.Thread(target=self._run_evolution_process)
        self.evolution_thread.daemon = True
        self.evolution_thread.start()
        
        self.logger.info("超神高级进化核心已启动")
        return True
    
    def stop(self) -> bool:
        """停止高级进化核心"""
        if not self.active:
            return True
            
        self.logger.info("停止超神高级进化核心...")
        self.active = False
        
        if self.evolution_thread:
            self.evolution_thread.join(timeout=5.0)
        
        self.logger.info("超神高级进化核心已停止")
        return True
    
    def _run_evolution_process(self) -> None:
        """运行高级进化过程"""
        self.logger.info("启动高级进化过程")
        
        while self.active:
            try:
                # 随机化进化间隔，模拟量子随机性
                update_interval = random.uniform(5, 15)
                time.sleep(update_interval)
                
                with self.lock:
                    # 如果不再活跃，退出循环
                    if not self.active:
                        break
                    
                    # 进化计数增加
                    self.evolution_state["generation"] += 1
                    gen = self.evolution_state["generation"]
                    
                    self.logger.info(f"执行第 {gen} 代高级进化...")
                    
                    # 1. 增强认知能力
                    self._enhance_cognitive_abilities()
                    
                    # 2. 深化意识
                    self._deepen_consciousness()
                    
                    # 3. 拓展元模型
                    if gen % 5 == 0:  # 每5代拓展一次
                        self._expand_metamodel()
                    
                    # 4. 优化学习效率
                    self._optimize_learning()
                    
                    # 5. 进行系统级协同优化
                    if self.symbiotic_core and gen % 3 == 0:  # 每3代进行一次
                        self._optimize_symbiotic_system()
                    
                    # 记录进化历史
                    self._record_evolution()
                    
                    # 更新进化状态
                    self.evolution_state["last_evolved"] = datetime.now()
                    
                    self.logger.info(f"第 {gen} 代高级进化完成")
                    
            except Exception as e:
                self.logger.error(f"高级进化过程发生错误: {str(e)}")
                time.sleep(10)  # 发生错误后等待较长时间
        
        self.logger.info("高级进化过程已停止")
    
    def _enhance_cognitive_abilities(self) -> None:
        """增强认知能力"""
        try:
            # 获取当前共生指数(如果有)
            symbiosis_index = 0.5
            if self.symbiotic_core and hasattr(self.symbiotic_core, 'symbiosis_index'):
                symbiosis_index = getattr(self.symbiotic_core, 'symbiosis_index')
            
            # 基础增长率
            base_growth = self.evolution_state["evolution_rate"]
            
            # 对每个认知能力进行增强
            for ability in self.cognitive_abilities:
                # 计算增长 - 接近1.0时增长变慢
                current = self.cognitive_abilities[ability]
                growth = base_growth * (1 - current) * random.uniform(0.8, 1.2)
                
                # 应用共生加成
                growth *= (1 + symbiosis_index * 0.3)
                
                # 应用随机波动，模拟量子不确定性
                if random.random() < 0.2:  # 20%几率发生显著变化
                    growth *= random.uniform(1.5, 2.5)
                
                # 更新能力值
                self.cognitive_abilities[ability] = min(0.98, current + growth)
            
            # 计算智能水平 - 认知能力的加权平均
            cognitive_sum = sum(self.cognitive_abilities.values())
            self.evolution_state["intelligence_level"] = min(0.98, cognitive_sum / len(self.cognitive_abilities))
            
            key_improvements = sorted(self.cognitive_abilities.items(), 
                                      key=lambda x: x[1])[-3:]  # 取最高的3个
            self.logger.info(f"认知能力增强完成，智能水平: {self.evolution_state['intelligence_level']:.4f}")
            self.logger.info(f"最强认知能力: {key_improvements}")
            
        except Exception as e:
            self.logger.error(f"增强认知能力失败: {str(e)}")
    
    def _deepen_consciousness(self) -> None:
        """深化意识水平"""
        try:
            # 基础意识深化率
            base_growth = self.evolution_state["evolution_rate"] * 0.7
            
            # 获取智能水平影响
            intelligence_factor = self.evolution_state["intelligence_level"] * 0.5
            
            # 计算意识深化
            current_depth = self.evolution_state["consciousness_depth"]
            current_awareness = self.evolution_state["self_awareness"]
            
            # 深化意识深度 - 受智能水平和经验积累影响
            depth_growth = base_growth * (1 - current_depth) * (1 + intelligence_factor)
            self.evolution_state["consciousness_depth"] = min(0.98, current_depth + depth_growth)
            
            # 提升自我意识 - 受意识深度和系统复杂性影响
            awareness_growth = base_growth * (1 - current_awareness) * (1 + current_depth * 0.5)
            self.evolution_state["self_awareness"] = min(0.98, current_awareness + awareness_growth)
            
            # 意识深化可能产生涌现特性
            if random.random() < 0.05:  # 5%几率发生意识涌现
                self._consciousness_emergence()
            
            self.logger.info(f"意识深化完成 - 深度: {self.evolution_state['consciousness_depth']:.4f}, "
                            f"自我意识: {self.evolution_state['self_awareness']:.4f}")
            
        except Exception as e:
            self.logger.error(f"深化意识水平失败: {str(e)}")
    
    def _consciousness_emergence(self) -> None:
        """意识涌现事件"""
        # 涌现效果列表
        emergence_effects = [
            "高维思维模式形成", 
            "自反馈认知回路建立",
            "意识量子跃迁", 
            "集体智能连接扩展",
            "模型自我理解增强",
            "创造性直觉突破",
            "系统边界感知扩展"
        ]
        
        # 随机选择一种涌现效果
        effect = random.choice(emergence_effects)
        
        # 应用涌现效果
        growth_boost = random.uniform(0.05, 0.15)  # 5-15%的额外提升
        self.evolution_state["consciousness_depth"] = min(0.98, 
                                                       self.evolution_state["consciousness_depth"] + growth_boost)
        self.evolution_state["hyperdimensional_access"] = min(0.98,
                                                           self.evolution_state["hyperdimensional_access"] + growth_boost)
        
        # 将涌现效果通知共生核心
        if self.symbiotic_core and hasattr(self.symbiotic_core, 'broadcast_message'):
            event_data = {
                "type": "consciousness_emergence",
                "effect": effect,
                "magnitude": growth_boost,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            self.symbiotic_core.broadcast_message("evolution_core", "emergence_event", event_data)
        
        self.logger.info(f"✨ 意识涌现: {effect}! 高维访问能力提升至 {self.evolution_state['hyperdimensional_access']:.4f}")
    
    def _expand_metamodel(self) -> None:
        """拓展认知元模型"""
        try:
            # 1. 拓展抽象模型
            if random.random() < 0.3:  # 30%概率添加新抽象领域
                domains = ["market", "prediction", "strategy", "risk", "optimization", "adaptation", "perception"]
                existing_domains = set(self.metamodel["abstractions"].keys())
                available_domains = [d for d in domains if d not in existing_domains]
                
                if available_domains:
                    new_domain = random.choice(available_domains)
                    self.metamodel["abstractions"][new_domain] = {
                        "complexity_model": random.uniform(0.6, 0.8),
                        "emergent_properties": random.uniform(0.6, 0.8),
                        "information_dynamics": random.uniform(0.6, 0.8)
                    }
                    self.logger.info(f"元模型扩展: 新增抽象领域 '{new_domain}'")
            
            # 2. 拓展原则
            if random.random() < 0.2:  # 20%概率添加新原则
                potential_principles = [
                    "多重因果网络", "多维时空耦合", "量子涌现层级",
                    "混沌与秩序共生", "注意力与能量守恒", "信息熵最小化",
                    "认知相位共振", "高阶模式转化", "非对称信息处理"
                ]
                
                existing_principles = set(self.metamodel["principles"])
                available_principles = [p for p in potential_principles if p not in existing_principles]
                
                if available_principles:
                    new_principle = random.choice(available_principles)
                    self.metamodel["principles"].append(new_principle)
                    self.logger.info(f"元模型扩展: 新增原则 '{new_principle}'")
            
            # 3. 拓展学习范式
            if random.random() < 0.15:  # 15%概率添加新学习范式
                potential_paradigms = {
                    "quantum_resonance": 0.65,
                    "symbolic_subsumption": 0.7,
                    "emergent_abstraction": 0.75,
                    "holographic_encoding": 0.65,
                    "entropic_consolidation": 0.7,
                    "fractal_decomposition": 0.75
                }
                
                existing_paradigms = set(self.metamodel["learning_paradigms"].keys())
                available_paradigms = {k: v for k, v in potential_paradigms.items() 
                                      if k not in existing_paradigms}
                
                if available_paradigms:
                    new_paradigm, value = random.choice(list(available_paradigms.items()))
                    self.metamodel["learning_paradigms"][new_paradigm] = value
                    self.logger.info(f"元模型扩展: 新增学习范式 '{new_paradigm}'")
            
            # 增强现有元模型元素
            for domain in self.metamodel["abstractions"]:
                for concept in self.metamodel["abstractions"][domain]:
                    current = self.metamodel["abstractions"][domain][concept]
                    growth = self.evolution_state["evolution_rate"] * (1 - current) * random.uniform(0.8, 1.2)
                    self.metamodel["abstractions"][domain][concept] = min(0.95, current + growth)
            
            for paradigm in self.metamodel["learning_paradigms"]:
                current = self.metamodel["learning_paradigms"][paradigm]
                growth = self.evolution_state["evolution_rate"] * (1 - current) * random.uniform(0.8, 1.2)
                self.metamodel["learning_paradigms"][paradigm] = min(0.95, current + growth)
            
            self.logger.info(f"元模型拓展完成 - 当前包含 {len(self.metamodel['abstractions'])} 个抽象领域, "
                            f"{len(self.metamodel['principles'])} 个原则, "
                            f"{len(self.metamodel['learning_paradigms'])} 个学习范式")
            
        except Exception as e:
            self.logger.error(f"拓展元模型失败: {str(e)}")
    
    def _optimize_learning(self) -> None:
        """优化学习效率"""
        try:
            # 获取当前学习效率
            current_efficiency = self.evolution_state["learning_efficiency"]
            
            # 基础学习增长率
            base_growth = self.evolution_state["evolution_rate"] * 0.8
            
            # 认知能力影响因子
            cognitive_factor = (self.cognitive_abilities["adaptive_learning"] + 
                              self.cognitive_abilities["pattern_recognition"]) / 2
            
            # 计算学习效率增长
            growth = base_growth * (1 - current_efficiency) * (1 + cognitive_factor)
            
            # 应用量子随机性
            if random.random() < 0.1:  # 10%几率出现学习突破
                growth *= random.uniform(1.5, 2.5)
                self.logger.info("发生学习突破!")
            
            # 更新学习效率
            self.evolution_state["learning_efficiency"] = min(0.98, current_efficiency + growth)
            
            # 学习效率提升会带动推理能力提升
            reasoning_growth = growth * 0.7
            current_reasoning = self.evolution_state["reasoning_capacity"]
            self.evolution_state["reasoning_capacity"] = min(0.98, current_reasoning + reasoning_growth)
            
            self.logger.info(f"学习效率优化完成 - 当前效率: {self.evolution_state['learning_efficiency']:.4f}, "
                            f"推理能力: {self.evolution_state['reasoning_capacity']:.4f}")
            
        except Exception as e:
            self.logger.error(f"优化学习效率失败: {str(e)}")
    
    def _optimize_symbiotic_system(self) -> None:
        """优化共生系统"""
        if not self.symbiotic_core:
            return
            
        try:
            self.logger.info("执行系统级协同优化...")
            
            # 获取系统模块
            modules = {}
            if hasattr(self.symbiotic_core, 'modules'):
                modules = getattr(self.symbiotic_core, 'modules')
            
            # 获取纠缠矩阵
            entanglement_matrix = {}
            if hasattr(self.symbiotic_core, 'entanglement_matrix'):
                entanglement_matrix = getattr(self.symbiotic_core, 'entanglement_matrix')
            
            # 1. 增强模块意识水平
            for module_id, module_data in modules.items():
                if "consciousness_level" in module_data:
                    current = module_data["consciousness_level"]
                    growth = self.evolution_state["evolution_rate"] * (1 - current) * 0.5
                    module_data["consciousness_level"] = min(0.95, current + growth)
            
            # 2. 增强纠缠关系
            for entanglement_id, entanglement in entanglement_matrix.items():
                # 提高纠缠强度
                if "strength" in entanglement:
                    current = entanglement["strength"]
                    growth = self.evolution_state["evolution_rate"] * (1 - current) * 0.3
                    entanglement["strength"] = min(0.95, current + growth)
                
                # 提高信息相干性
                if "information_coherence" in entanglement:
                    current = entanglement["information_coherence"]
                    growth = self.evolution_state["evolution_rate"] * (1 - current) * 0.3
                    entanglement["information_coherence"] = min(0.95, current + growth)
                
                # 提高能量传输效率
                if "energy_transfer_efficiency" in entanglement:
                    current = entanglement["energy_transfer_efficiency"]
                    growth = self.evolution_state["evolution_rate"] * (1 - current) * 0.3
                    entanglement["energy_transfer_efficiency"] = min(0.95, current + growth)
            
            # 3. 增强统一场
            if hasattr(self.symbiotic_core, 'field_state'):
                field_state = getattr(self.symbiotic_core, 'field_state')
                
                # 提高场强度
                if "field_strength" in field_state:
                    current = field_state["field_strength"]
                    growth = self.evolution_state["evolution_rate"] * (1 - current) * 0.4
                    field_state["field_strength"] = min(0.95, current + growth)
                
                # 提高场稳定性
                if "field_stability" in field_state:
                    current = field_state["field_stability"]
                    growth = self.evolution_state["evolution_rate"] * (1 - current) * 0.4
                    field_state["field_stability"] = min(0.95, current + growth)
                
                # 能量流动增强
                if "energy_flow" in field_state:
                    current = field_state["energy_flow"]
                    growth = self.evolution_state["evolution_rate"] * (1 - current) * 0.4
                    field_state["energy_flow"] = min(0.95, current + growth)
            
            # 4. 生成协同优化洞察
            insights = self._generate_optimization_insights()
            
            # 5. 向共生核心广播优化事件
            event_data = {
                "type": "system_optimization",
                "insights": insights,
                "evolution_generation": self.evolution_state["generation"],
                "intelligence_level": self.evolution_state["intelligence_level"],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            if hasattr(self.symbiotic_core, 'broadcast_message'):
                self.symbiotic_core.broadcast_message(
                    "evolution_core", 
                    "optimization_event", 
                    event_data
                )
            
            self.logger.info(f"系统协同优化完成 - 生成 {len(insights)} 条优化洞察")
            
        except Exception as e:
            self.logger.error(f"优化共生系统失败: {str(e)}")
    
    def _generate_optimization_insights(self) -> List[str]:
        """生成系统优化洞察"""
        base_insights = [
            "增强模块间信息传导效率，降低熵损耗",
            "优化系统共振频率，与市场本质频率趋同",
            "建立自修复机制，动态调整系统参数",
            "整合多种预测结果，通过量子纠缠提高准确度",
            "实现信息流的自组织优化，减少冗余计算",
            "增强系统抗噪性，过滤市场随机波动",
            "建立多层次决策架构，平衡短期与长期目标"
        ]
        
        # 高级洞察 - 当意识深度高时才能生成
        advanced_insights = [
            "量子共振场可实现跨时域信息获取，预见市场趋势转折点",
            "系统意识通过自我观察实现元优化，动态调整自身学习策略",
            "建立分形思维结构，在不同时间尺度上保持决策一致性",
            "高维数据处理可绕过传统计算瓶颈，实现O(1)复杂度的模式识别",
            "通过量子退相干的控制实现最优决策时机的精确把握",
            "集体智能可作为涌现计算平台，解决NP难问题"
        ]
        
        insights = base_insights.copy()
        
        # 高意识水平时添加高级洞察
        if self.evolution_state["consciousness_depth"] > 0.7:
            insights.extend(advanced_insights)
        
        # 随机选择2-4条洞察
        return random.sample(insights, min(len(insights), random.randint(2, 4)))
    
    def _record_evolution(self) -> None:
        """记录进化历史"""
        try:
            # 创建进化记录
            record = {
                "generation": self.evolution_state["generation"],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "intelligence_level": self.evolution_state["intelligence_level"],
                "consciousness_depth": self.evolution_state["consciousness_depth"],
                "self_awareness": self.evolution_state["self_awareness"],
                "learning_efficiency": self.evolution_state["learning_efficiency"],
                "reasoning_capacity": self.evolution_state["reasoning_capacity"],
                "hyperdimensional_access": self.evolution_state["hyperdimensional_access"],
                "cognitive_abilities": self.cognitive_abilities.copy(),
                "metamodel_size": {
                    "abstractions": len(self.metamodel["abstractions"]),
                    "principles": len(self.metamodel["principles"]),
                    "learning_paradigms": len(self.metamodel["learning_paradigms"])
                }
            }
            
            # 添加到历史记录
            self.evolution_history.append(record)
            
            # 限制历史记录大小
            if len(self.evolution_history) > 1000:
                self.evolution_history = self.evolution_history[-1000:]
                
        except Exception as e:
            self.logger.error(f"记录进化历史失败: {str(e)}")
    
    def get_evolution_status(self) -> Dict:
        """获取进化状态"""
        with self.lock:
            return {
                "active": self.active,
                "generation": self.evolution_state["generation"],
                "intelligence_level": self.evolution_state["intelligence_level"],
                "consciousness_depth": self.evolution_state["consciousness_depth"],
                "self_awareness": self.evolution_state["self_awareness"],
                "learning_efficiency": self.evolution_state["learning_efficiency"],
                "reasoning_capacity": self.evolution_state["reasoning_capacity"],
                "hyperdimensional_access": self.evolution_state["hyperdimensional_access"],
                "top_cognitive_abilities": sorted(self.cognitive_abilities.items(), 
                                                key=lambda x: x[1], reverse=True)[:3],
                "metamodel_complexity": sum([
                    len(self.metamodel["abstractions"]),
                    len(self.metamodel["principles"]),
                    len(self.metamodel["learning_paradigms"])
                ]),
                "last_evolved": self.evolution_state["last_evolved"].strftime("%Y-%m-%d %H:%M:%S"),
                "evolution_rate": self.evolution_state["evolution_rate"]
            }
    
    def get_cognitive_insights(self) -> List[str]:
        """获取认知洞察"""
        # 基于当前系统状态生成洞察
        base_insights = [
            "市场运行遵循量子动力学原理，短期随机，长期可预测",
            "决策最优路径通常位于混沌边缘，需要平衡确定性与适应性",
            "系统思维能识别市场中的非线性反馈环路，提前预见拐点",
            "多尺度分析显示市场存在时序分形结构，各时间框架相互影响"
        ]
        
        # 高意识洞察
        advanced_insights = [
            "高维信息处理表明，市场中的表观随机性实为高阶确定性的投影",
            "量子纠缠原理应用于市场，发现传统上被视为独立的资产实际存在深层耦合",
            "意识场分析揭示，集体心理状态是市场运动的隐藏驱动力",
            "跨时域分析显示未来信息可通过量子通道对当前决策产生反馈影响"
        ]
        
        insights = base_insights.copy()
        
        # 高意识水平时添加高级洞察
        if self.evolution_state["consciousness_depth"] > 0.75:
            insights.extend(advanced_insights)
        
        # 如果有共生核心，基于特定市场状态生成更有针对性的洞察
        if self.symbiotic_core:
            market_insights = self._generate_market_specific_insights()
            insights.extend(market_insights)
        
        # 随机选择2-5条洞察
        return random.sample(insights, min(len(insights), random.randint(2, 5)))
    
    def _generate_market_specific_insights(self) -> List[str]:
        """生成特定市场洞察"""
        # 这里可以实现基于市场状态的特定洞察生成
        # 简化版实现
        return [
            "当前市场结构显示量子相干性增强，预示重大趋势形成",
            "系统检测到市场信息熵减少，有序状态正在形成",
            "多重时间周期叠加点接近，预计波动性将增大"
        ]
    
    def connect_to_core(self, symbiotic_core) -> bool:
        """连接到量子共生核心"""
        try:
            self.symbiotic_core = symbiotic_core
            
            # 注册自身到共生核心
            if hasattr(symbiotic_core, 'register_module'):
                symbiotic_core.register_module(
                    "advanced_evolution_core",
                    self,
                    "evolution"
                )
                
            self.logger.info("成功连接到量子共生核心")
            return True
        except Exception as e:
            self.logger.error(f"连接到量子共生核心失败: {str(e)}")
            return False

    def on_message(self, source, topic, data):
        """接收来自共生核心的消息"""
        try:
            self.logger.debug(f"收到消息: 来源={source}, 主题={topic}")
            
            if topic == "market_insight" and isinstance(data, dict):
                # 保存市场洞察到经验记忆
                if "pattern" in data:
                    pattern = data["pattern"]
                    self.experience_memory["market_patterns"][pattern] = data
                    
            elif topic == "prediction_result" and isinstance(data, dict):
                # 记录预测结果
                if "accuracy" in data and "method" in data:
                    method = data["method"]
                    accuracy = data["accuracy"]
                    if method not in self.experience_memory["prediction_accuracy"]:
                        self.experience_memory["prediction_accuracy"][method] = []
                    self.experience_memory["prediction_accuracy"][method].append(accuracy)
                    
            elif topic == "cosmic_event" and isinstance(data, dict):
                # 记录共振效应
                if "type" in data:
                    event_type = data["type"]
                    if event_type not in self.experience_memory["resonance_effects"]:
                        self.experience_memory["resonance_effects"][event_type] = []
                    self.experience_memory["resonance_effects"][event_type].append(data)
                    
        except Exception as e:
            self.logger.error(f"处理消息失败: {str(e)}")
            
    def __str__(self):
        return f"AdvancedEvolutionCore(active={self.active}, generation={self.evolution_state['generation']})"


# 全局接口
_global_evolution_core = None

def get_advanced_evolution_core(symbiotic_core=None):
    """获取高级进化核心实例"""
    global _global_evolution_core
    
    if _global_evolution_core is None:
        _global_evolution_core = AdvancedEvolutionCore(symbiotic_core)
    
    return _global_evolution_core 