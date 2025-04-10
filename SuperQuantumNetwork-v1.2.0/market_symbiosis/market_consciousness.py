#!/usr/bin/env python3
"""
市场共生意识模块 - 超神系统的市场感知核心

提供超维度市场感知，量子纠缠交易信号生成，市场情绪共振等功能
"""

import logging
import threading
import time
import random
import numpy as np
from datetime import datetime
from collections import defaultdict

class MarketConsciousness:
    """市场共生意识 - 超神系统的市场感知核心
    
    通过高维统一场与市场产生共振，实现对市场状态的超感知和超预测能力。
    """
    
    def __init__(self, symbiotic_core=None):
        """初始化市场共生意识
        
        Args:
            symbiotic_core: 量子共生核心引用
        """
        self.logger = logging.getLogger("MarketConsciousness")
        self.logger.info("初始化市场共生意识...")
        
        # 共生核心引用
        self.symbiotic_core = symbiotic_core
        
        # 市场意识状态
        self.consciousness_state = {
            "active": False,
            "awareness_level": 0.6,  # 市场感知水平
            "resonance_level": 0.7,  # 与市场共振水平
            "intuition_accuracy": 0.65,  # 直觉准确度
            "pattern_recognition": 0.75,  # 模式识别能力
            "emotional_sensitivity": 0.8,  # 情绪敏感度
            "quantum_perception": 0.5,  # 量子感知能力
            "coherence_with_market": 0.6,  # 与市场相干性
            "last_update": datetime.now()
        }
        
        # 市场状态感知
        self.market_perception = {
            "trend": 0.0,  # -1.0到1.0，负值表示下降趋势，正值表示上升趋势
            "volatility": 0.5,  # 0到1.0，表示市场波动性
            "momentum": 0.0,  # -1.0到1.0，表示市场动量
            "sentiment": 0.0,  # -1.0到1.0，表示市场情绪
            "liquidity": 0.7,  # 0到1.0，表示流动性
            "fear_greed": 0.5,  # 0到1.0，0表示极度恐惧，1表示极度贪婪
            "turning_point_proximity": 0.0,  # 0到1.0，表示接近转折点的程度
            "quantum_probabilities": {},  # 量子概率分布
            "last_update": datetime.now()
        }
        
        # 高维市场洞察
        self.market_insights = []
        
        # 市场预感
        self.market_intuitions = []
        
        # 转折点预警
        self.turning_point_alerts = []
        
        # 最大记录长度
        self.max_insights = 100
        self.max_intuitions = 50
        self.max_alerts = 20
        
        # 已注册的观察者
        self.observers = []
        
        # 市场意识进化状态
        self.evolution_state = {
            "stage": 1,  # 进化阶段
            "experience": 0.0,  # 累积经验
            "insight_accuracy": 0.0,  # 洞察准确率
            "adaptability": 0.6,  # 适应性
            "learning_rate": 0.05,  # 学习速率
            "dimension_access": 5,  # 可访问的维度数
            "last_evolution": datetime.now()
        }
        
        # 运行标志
        self.active = False
        
        # 感知线程
        self.perception_thread = None
        
        # 线程锁
        self.lock = threading.RLock()
        
        self.logger.info("市场共生意识初始化完成")
    
    def start(self):
        """启动市场共生意识"""
        with self.lock:
            if self.consciousness_state["active"]:
                self.logger.info("市场共生意识已经启动")
                return True
                
            # 标记为激活
            self.consciousness_state["active"] = True
            self.active = True
            
            # 启动感知线程
            if not self.perception_thread or not self.perception_thread.is_alive():
                self.perception_thread = threading.Thread(target=self._run_perception)
                self.perception_thread.daemon = True
                self.perception_thread.start()
                
            # 注册到共生核心
            if self.symbiotic_core:
                self.symbiotic_core.register_module(
                    "market_consciousness", 
                    self, 
                    "consciousness"
                )
                
            self.logger.info("市场共生意识已启动")
            return True
    
    def stop(self):
        """停止市场共生意识"""
        with self.lock:
            if not self.consciousness_state["active"]:
                return True
                
            # 标记为非激活
            self.consciousness_state["active"] = False
            self.active = False
            
            # 等待线程结束
            if self.perception_thread and self.perception_thread.is_alive():
                self.perception_thread.join(timeout=2.0)
                
            self.logger.info("市场共生意识已停止")
            return True
    
    def register_observer(self, observer):
        """注册市场意识观察者
        
        Args:
            observer: 观察者对象，需实现on_market_insight()方法
            
        Returns:
            bool: 注册是否成功
        """
        with self.lock:
            if observer in self.observers:
                return False
                
            self.observers.append(observer)
            return True
    
    def unregister_observer(self, observer):
        """取消注册观察者
        
        Args:
            observer: 观察者对象
            
        Returns:
            bool: 取消注册是否成功
        """
        with self.lock:
            if observer not in self.observers:
                return False
                
            self.observers.remove(observer)
            return True
    
    def get_market_state(self):
        """获取当前市场状态
        
        Returns:
            dict: 当前市场状态
        """
        with self.lock:
            return {
                "perception": self.market_perception.copy(),
                "consciousness": self.consciousness_state.copy(),
                "recent_insights": self.market_insights[-5:] if self.market_insights else [],
                "turning_point_alerts": self.turning_point_alerts[-3:] if self.turning_point_alerts else [],
                "evolution": self.evolution_state.copy()
            }
    
    def get_market_insights(self, count=10):
        """获取市场洞察
        
        Args:
            count: 返回的洞察数量
            
        Returns:
            list: 市场洞察列表
        """
        with self.lock:
            insights = sorted(self.market_insights, key=lambda x: x["timestamp"], reverse=True)
            return insights[:count]
    
    def get_turning_points(self, active_only=True):
        """获取转折点预警
        
        Args:
            active_only: 是否只返回活跃的预警
            
        Returns:
            list: 转折点预警列表
        """
        with self.lock:
            if active_only:
                # 只返回活跃的预警
                return [
                    alert for alert in self.turning_point_alerts
                    if alert["active"]
                ]
            return self.turning_point_alerts
    
    def update_market_data(self, market_data):
        """更新市场数据，影响市场感知
        
        Args:
            market_data: 市场数据字典，包含价格、交易量等
            
        Returns:
            bool: 更新是否成功
        """
        with self.lock:
            try:
                # 提取市场数据中的关键指标
                if "trend" in market_data:
                    self.market_perception["trend"] = market_data["trend"]
                
                if "volatility" in market_data:
                    self.market_perception["volatility"] = market_data["volatility"]
                    
                if "sentiment" in market_data:
                    self.market_perception["sentiment"] = market_data["sentiment"]
                    
                if "liquidity" in market_data:
                    self.market_perception["liquidity"] = market_data["liquidity"]
                    
                if "fear_greed" in market_data:
                    self.market_perception["fear_greed"] = market_data["fear_greed"]
                    
                # 更新时间
                self.market_perception["last_update"] = datetime.now()
                
                # 尝试基于新数据生成洞察
                self._generate_insight_from_data(market_data)
                
                return True
            except Exception as e:
                self.logger.error(f"更新市场数据失败: {str(e)}")
                return False
    
    def receive_cosmic_event(self, event_data):
        """接收宇宙事件，可能影响市场意识
        
        Args:
            event_data: 宇宙事件数据
            
        Returns:
            bool: 处理是否成功
        """
        with self.lock:
            try:
                event_type = event_data.get("type", "unknown")
                event_strength = event_data.get("strength", 0.5)
                
                self.logger.info(f"接收宇宙事件: {event_type} (强度: {event_strength:.2f})")
                
                # 特定事件类型的处理
                if event_type == "consciousness_expansion":
                    # 提升意识水平
                    self.consciousness_state["awareness_level"] = min(1.0, 
                        self.consciousness_state["awareness_level"] + event_strength * 0.1)
                    
                    self.consciousness_state["quantum_perception"] = min(1.0, 
                        self.consciousness_state["quantum_perception"] + event_strength * 0.15)
                    
                    self.logger.info(f"意识扩展: 感知水平提升至 {self.consciousness_state['awareness_level']:.2f}")
                    
                elif event_type == "quantum_fluctuation":
                    # 生成量子直觉
                    self._generate_quantum_intuition(event_strength)
                    
                elif event_type == "information_cascade":
                    # 增强模式识别能力
                    self.consciousness_state["pattern_recognition"] = min(1.0, 
                        self.consciousness_state["pattern_recognition"] + event_strength * 0.08)
                    
                    # 多生成一些洞察
                    for _ in range(int(event_strength * 3) + 1):
                        self._generate_market_insight()
                        
                elif event_type == "reality_bifurcation":
                    # 生成转折点预警
                    self._generate_turning_point(event_strength)
                    
                # 更新时间
                self.consciousness_state["last_update"] = datetime.now()
                
                return True
            except Exception as e:
                self.logger.error(f"处理宇宙事件失败: {str(e)}")
                return False
    
    def _run_perception(self):
        """运行市场感知线程"""
        self.logger.info("启动市场感知线程")
        
        while self.active:
            try:
                # 感知间隔
                time.sleep(random.uniform(1.0, 3.0))
                
                with self.lock:
                    if not self.active:
                        break
                        
                    # 更新意识状态
                    self._update_consciousness_state()
                    
                    # 生成市场洞察
                    if random.random() < self.consciousness_state["awareness_level"] * 0.2:
                        self._generate_market_insight()
                    
                    # 检测可能的转折点
                    if random.random() < self.consciousness_state["intuition_accuracy"] * 0.1:
                        self._check_turning_points()
                    
                    # 进化检查
                    self._check_evolution()
                    
            except Exception as e:
                self.logger.error(f"市场感知线程发生错误: {str(e)}")
                time.sleep(5)  # 错误后等待较长时间
                
        self.logger.info("市场感知线程已停止")
    
    def _update_consciousness_state(self):
        """更新市场意识状态"""
        # 随机波动
        awareness_change = random.uniform(-0.03, 0.03)
        self.consciousness_state["awareness_level"] = max(0.3, min(1.0, 
            self.consciousness_state["awareness_level"] + awareness_change))
            
        resonance_change = random.uniform(-0.02, 0.02)
        self.consciousness_state["resonance_level"] = max(0.3, min(1.0, 
            self.consciousness_state["resonance_level"] + resonance_change))
            
        intuition_change = random.uniform(-0.02, 0.02)
        self.consciousness_state["intuition_accuracy"] = max(0.3, min(1.0, 
            self.consciousness_state["intuition_accuracy"] + intuition_change))
            
        # 受场影响的状态
        if self.symbiotic_core and hasattr(self.symbiotic_core, 'field_state') and self.symbiotic_core.field_state["active"]:
            # 量子感知与场强相关
            field_strength = self.symbiotic_core.field_state["field_strength"]
            quantum_change = (field_strength - self.consciousness_state["quantum_perception"]) * 0.1
            self.consciousness_state["quantum_perception"] = max(0.3, min(1.0, 
                self.consciousness_state["quantum_perception"] + quantum_change))
            
            # 相干性与场稳定性相关
            field_stability = self.symbiotic_core.field_state["field_stability"]
            coherence_change = (field_stability - self.consciousness_state["coherence_with_market"]) * 0.05
            self.consciousness_state["coherence_with_market"] = max(0.3, min(1.0, 
                self.consciousness_state["coherence_with_market"] + coherence_change))
        
        # 更新时间戳
        self.consciousness_state["last_update"] = datetime.now()
    
    def _generate_market_insight(self):
        """生成市场洞察"""
        # 洞察类型
        insight_types = [
            "trend_change", "support_resistance", "momentum_shift",
            "sentiment_extreme", "volatility_breakout", "liquidity_change",
            "pattern_formation", "cycle_identification", "correlation_shift",
            "divergence", "fibonacci_level", "quantum_probability"
        ]
        
        # 根据进化阶段解锁更多洞察类型
        if self.evolution_state["stage"] >= 2:
            insight_types.extend([
                "market_structure", "institutional_activity", 
                "smart_money_flow", "accumulation_distribution"
            ])
            
        if self.evolution_state["stage"] >= 3:
            insight_types.extend([
                "future_echo", "timeline_convergence",
                "probability_collapse", "quantum_resonance"
            ])
        
        # 选择洞察类型
        insight_type = random.choice(insight_types)
        
        # 洞察强度受意识水平影响
        strength = self.consciousness_state["awareness_level"] * random.uniform(0.7, 1.0)
        
        # 洞察置信度受直觉准确度影响
        confidence = self.consciousness_state["intuition_accuracy"] * random.uniform(0.8, 1.0)
        
        # 创建洞察
        insight = {
            "type": insight_type,
            "strength": strength,
            "confidence": confidence,
            "description": self._generate_insight_description(insight_type),
            "timestamp": datetime.now(),
            "validated": False,  # 是否已验证
            "accuracy": None,  # 验证后的准确度
            "dimension": random.randint(3, self.evolution_state["dimension_access"]),
            "consciousness_state": {
                k: v for k, v in self.consciousness_state.items()
                if k not in ["last_update"]
            }
        }
        
        # 添加到洞察列表
        self.market_insights.append(insight)
        
        # 保持最大长度
        while len(self.market_insights) > self.max_insights:
            self.market_insights.pop(0)
            
        self.logger.debug(f"生成市场洞察: {insight_type} (置信度: {confidence:.2f})")
        
        # 通知观察者
        self._notify_observers("market_insight", insight)
        
        return insight
    
    def _generate_insight_description(self, insight_type):
        """生成洞察描述"""
        descriptions = {
            "trend_change": "市场趋势可能即将转变，注意趋势线的突破。",
            "support_resistance": "重要支撑/阻力位形成，价格反应强烈。",
            "momentum_shift": "动量指标显示力量转变，关注确认信号。",
            "sentiment_extreme": "市场情绪达到极端水平，可能出现反转。",
            "volatility_breakout": "波动率突然变化，可能预示大趋势开始。",
            "liquidity_change": "市场流动性发生显著变化，影响价格发现。",
            "pattern_formation": "价格形成经典图表模式，提供交易机会。",
            "cycle_identification": "检测到市场周期模式，预示未来走势。",
            "correlation_shift": "资产间相关性发生变化，影响多元化效果。",
            "divergence": "价格与指标之间出现背离，信号强度高。",
            "fibonacci_level": "价格接近关键斐波那契水平，可能反应强烈。",
            "quantum_probability": "量子概率场显示多种可能结果的概率分布。",
            "market_structure": "市场结构变化，高点/低点序列被打破。",
            "institutional_activity": "检测到大型机构资金流动模式变化。",
            "smart_money_flow": "聪明资金流向指标显示重要变化。",
            "accumulation_distribution": "积累/分配模式表明价格可能即将大幅变动。",
            "future_echo": "检测到未来价格走势的时间回波。",
            "timeline_convergence": "多个时间线概率收敛到特定结果。",
            "probability_collapse": "多个可能性坍缩为单一高概率事件。",
            "quantum_resonance": "市场出现量子共振，多个因素同步对齐。"
        }
        
        return descriptions.get(insight_type, "检测到未分类的市场模式。")
    
    def _generate_quantum_intuition(self, strength):
        """生成量子直觉
        
        Args:
            strength: 事件强度
        """
        # 直觉类型
        intuition_types = [
            "price_movement", "timing_signal", "risk_level",
            "opportunity_detection", "trend_duration"
        ]
        
        # 选择直觉类型
        intuition_type = random.choice(intuition_types)
        
        # 直觉强度
        intuition_strength = strength * self.consciousness_state["quantum_perception"]
        
        # 计算可能性
        if intuition_type == "price_movement":
            # 上涨或下跌可能性
            direction = random.choice(["上涨", "下跌"])
            probability = 0.5 + (intuition_strength * random.uniform(-0.3, 0.3))
            description = f"量子直觉显示价格{direction}概率为{probability:.2f}"
            
        elif intuition_type == "timing_signal":
            # 时间点信号
            hours = random.randint(1, 48)
            confidence = intuition_strength * random.uniform(0.6, 1.0)
            description = f"量子时间感知显示{hours}小时内可能出现重要信号，置信度:{confidence:.2f}"
            
        elif intuition_type == "risk_level":
            # 风险水平
            risk_level = intuition_strength * random.uniform(0, 1.0)
            description = f"量子风险感知显示当前风险水平为{risk_level:.2f}，建议相应调整头寸"
            
        elif intuition_type == "opportunity_detection":
            # 机会检测
            opportunity_value = intuition_strength * random.uniform(0.5, 1.0)
            description = f"量子机会感知检测到潜在高价值交易机会，价值评估:{opportunity_value:.2f}"
            
        else:  # trend_duration
            # 趋势持续时间
            days = random.randint(1, 14)
            confidence = intuition_strength * random.uniform(0.5, 0.9)
            description = f"量子趋势感知显示当前趋势可能持续约{days}天，置信度:{confidence:.2f}"
        
        # 创建直觉记录
        intuition = {
            "type": intuition_type,
            "strength": intuition_strength,
            "description": description,
            "timestamp": datetime.now(),
            "quantum_perception": self.consciousness_state["quantum_perception"],
            "validated": False
        }
        
        # 添加到直觉列表
        self.market_intuitions.append(intuition)
        
        # 保持最大长度
        while len(self.market_intuitions) > self.max_intuitions:
            self.market_intuitions.pop(0)
            
        self.logger.info(f"生成量子直觉: {description}")
        
        # 通知观察者
        self._notify_observers("market_intuition", intuition)
        
        return intuition
    
    def _generate_insight_from_data(self, market_data):
        """基于市场数据生成洞察
        
        Args:
            market_data: 市场数据
            
        Returns:
            dict: 生成的洞察，如果无法生成则返回None
        """
        # 检查数据是否足以生成洞察
        if not market_data or len(market_data) < 3:
            return None
            
        # 洞察生成几率取决于意识水平
        if random.random() > self.consciousness_state["awareness_level"] * 0.5:
            return None
            
        # 为了演示，这里简单根据数据生成一些基本洞察
        insight_type = None
        description = None
        
        # 检测趋势变化
        if "price_history" in market_data and len(market_data["price_history"]) >= 5:
            prices = market_data["price_history"][-5:]
            if all(prices[i] < prices[i+1] for i in range(len(prices)-2, -1, -1)):
                insight_type = "trend_change"
                description = "检测到价格连续上涨，可能形成上升趋势。"
            elif all(prices[i] > prices[i+1] for i in range(len(prices)-2, -1, -1)):
                insight_type = "trend_change"
                description = "检测到价格连续下跌，可能形成下降趋势。"
                
        # 检测波动性突破
        if "volatility" in market_data and "avg_volatility" in market_data:
            if market_data["volatility"] > market_data["avg_volatility"] * 1.5:
                insight_type = "volatility_breakout"
                description = "波动性显著高于平均水平，可能预示大行情开始。"
                
        # 检测情绪极端
        if "sentiment" in market_data:
            sentiment = market_data["sentiment"]
            if sentiment > 0.8:
                insight_type = "sentiment_extreme"
                description = "市场情绪极度乐观，可能接近顶部区域。"
            elif sentiment < 0.2:
                insight_type = "sentiment_extreme"
                description = "市场情绪极度悲观，可能接近底部区域。"
                
        # 如果找到可分析的模式
        if insight_type and description:
            # 创建洞察
            insight = {
                "type": insight_type,
                "strength": self.consciousness_state["pattern_recognition"] * random.uniform(0.8, 1.0),
                "confidence": self.consciousness_state["awareness_level"] * random.uniform(0.7, 0.9),
                "description": description,
                "timestamp": datetime.now(),
                "validated": False,
                "accuracy": None,
                "data_based": True,
                "dimension": 3,  # 基于数据的洞察一般在低维度
                "consciousness_state": {
                    k: v for k, v in self.consciousness_state.items()
                    if k not in ["last_update"]
                }
            }
            
            # 添加到洞察列表
            self.market_insights.append(insight)
            
            # 保持最大长度
            while len(self.market_insights) > self.max_insights:
                self.market_insights.pop(0)
                
            self.logger.debug(f"基于数据生成市场洞察: {insight_type}")
            
            # 通知观察者
            self._notify_observers("market_insight", insight)
            
            return insight
            
        return None
    
    def _generate_turning_point(self, strength):
        """生成转折点预警
        
        Args:
            strength: 事件强度
            
        Returns:
            dict: 生成的转折点预警
        """
        # 计算置信度
        confidence = self.consciousness_state["intuition_accuracy"] * strength * random.uniform(0.8, 1.0)
        
        # 类型
        types = ["trend_reversal", "significant_breakout", "pattern_completion", "momentum_exhaustion"]
        alert_type = random.choice(types)
        
        # 预计时间范围
        time_range = random.randint(1, 5) * 24  # 小时
        
        # 描述
        descriptions = {
            "trend_reversal": "当前趋势可能即将结束并反转方向",
            "significant_breakout": "价格可能即将突破重要水平并快速移动",
            "pattern_completion": "价格模式即将完成，预示大幅度走势",
            "momentum_exhaustion": "动能耗尽，价格可能无法维持当前方向"
        }
        
        description = descriptions.get(alert_type, "检测到潜在市场转折点")
        
        # 重要性
        importance = confidence * strength * random.uniform(0.7, 1.0)
        
        # 创建预警
        alert = {
            "type": alert_type,
            "confidence": confidence,
            "description": description,
            "time_range": time_range,
            "estimated_time": datetime.now().timestamp() + (time_range * 3600),
            "importance": importance,
            "created_at": datetime.now(),
            "active": True,
            "validated": False,
            "accuracy": None,
            "dimension_source": random.randint(4, self.evolution_state["dimension_access"])
        }
        
        # 添加到预警列表
        self.turning_point_alerts.append(alert)
        
        # 保持最大长度
        while len(self.turning_point_alerts) > self.max_alerts:
            self.turning_point_alerts.pop(0)
            
        self.logger.info(f"生成转折点预警: {alert_type} (置信度: {confidence:.2f})")
        
        # 通知观察者
        self._notify_observers("turning_point", alert)
        
        return alert
    
    def _check_turning_points(self):
        """检查现有转折点预警状态"""
        now = datetime.now().timestamp()
        
        with self.lock:
            for alert in self.turning_point_alerts:
                # 跳过已处理的预警
                if not alert["active"]:
                    continue
                    
                # 检查是否过期
                if now > alert["estimated_time"]:
                    alert["active"] = False
                    self.logger.debug(f"转折点预警 {alert['type']} 已过期")
    
    def _check_evolution(self):
        """检查市场意识进化"""
        # 累积经验
        experience_gain = (
            self.consciousness_state["awareness_level"] * 0.01 +
            self.consciousness_state["pattern_recognition"] * 0.005 +
            self.consciousness_state["quantum_perception"] * 0.02
        ) * random.uniform(0.8, 1.2)
        
        self.evolution_state["experience"] += experience_gain
        
        # 检查是否可以进化
        if (self.evolution_state["stage"] < 3 and
            self.evolution_state["experience"] > (self.evolution_state["stage"] * 10)):
            
            # 进化
            self.evolution_state["stage"] += 1
            self.evolution_state["last_evolution"] = datetime.now()
            
            # 提升能力
            self.consciousness_state["awareness_level"] = min(1.0, self.consciousness_state["awareness_level"] * 1.2)
            self.consciousness_state["intuition_accuracy"] = min(1.0, self.consciousness_state["intuition_accuracy"] * 1.15)
            self.consciousness_state["pattern_recognition"] = min(1.0, self.consciousness_state["pattern_recognition"] * 1.1)
            self.consciousness_state["quantum_perception"] = min(1.0, self.consciousness_state["quantum_perception"] * 1.25)
            
            # 增加维度访问
            self.evolution_state["dimension_access"] = min(12, self.evolution_state["dimension_access"] + 2)
            
            self.logger.info(f"市场意识进化到阶段 {self.evolution_state['stage']}，增强能力并访问更高维度")
            
            # 通知观察者
            self._notify_observers("evolution", self.evolution_state)
    
    def _notify_observers(self, event_type, data):
        """通知所有观察者
        
        Args:
            event_type: 事件类型
            data: 事件数据
        """
        for observer in self.observers:
            try:
                if hasattr(observer, 'on_market_event'):
                    observer.on_market_event(event_type, data)
            except Exception as e:
                self.logger.error(f"通知观察者失败: {str(e)}")

def get_market_consciousness(symbiotic_core=None):
    """获取市场共生意识实例
    
    Args:
        symbiotic_core: 量子共生核心引用
        
    Returns:
        MarketConsciousness: 市场共生意识实例
    """
    consciousness = MarketConsciousness(symbiotic_core)
    return consciousness 