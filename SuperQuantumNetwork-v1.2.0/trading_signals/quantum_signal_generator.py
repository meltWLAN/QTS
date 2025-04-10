#!/usr/bin/env python3
"""
量子交易信号生成器 - 超神系统的交易信号核心

通过高维统一场和量子共生网络，生成超预测交易信号
"""

import logging
import threading
import time
import random
import numpy as np
from datetime import datetime
from collections import defaultdict

class QuantumSignalGenerator:
    """量子交易信号生成器
    
    利用量子共生网络和高维意识生成交易信号
    """
    
    def __init__(self, symbiotic_core=None, market_consciousness=None):
        """初始化量子交易信号生成器
        
        Args:
            symbiotic_core: 量子共生核心引用
            market_consciousness: 市场共生意识引用
        """
        self.logger = logging.getLogger("QuantumSignalGenerator")
        self.logger.info("初始化量子交易信号生成器...")
        
        # 核心组件引用
        self.symbiotic_core = symbiotic_core
        self.market_consciousness = market_consciousness
        
        # 信号生成器状态
        self.generator_state = {
            "active": False,
            "quantum_coherence": 0.7,  # 量子相干性
            "signal_quality": 0.8,  # 信号质量
            "accuracy": 0.75,  # 准确度
            "processing_power": 0.6,  # 处理能力
            "dimensional_access": list(range(3, 9)),  # 可访问维度
            "last_update": datetime.now()
        }
        
        # 活跃信号列表
        self.active_signals = []
        
        # 历史信号列表
        self.signal_history = []
        
        # 最大历史长度
        self.max_history_length = 200
        
        # 信号统计
        self.stats = {
            "total_signals": 0,
            "successful_signals": 0,
            "failed_signals": 0,
            "signal_types": defaultdict(int),
            "dimension_usage": defaultdict(int),
            "average_quality": 0.0,
            "last_signal_time": None
        }
        
        # 观察者列表
        self.observers = []
        
        # 信号生成配置
        self.config = {
            "min_signal_interval": 60,  # 最小信号间隔(秒)
            "base_generation_chance": 0.1,  # 基础生成几率
            "max_active_signals": 10,  # 最大活跃信号数量
            "quality_threshold": 0.6,  # 信号质量阈值
            "confidence_threshold": 0.65  # 信号置信度阈值
        }
        
        # 已接收的市场洞察
        self.market_insights = []
        
        # 线程和标志
        self.active = False
        self.generator_thread = None
        self.lock = threading.RLock()
        
        self.logger.info("量子交易信号生成器初始化完成")
    
    def start(self):
        """启动信号生成器"""
        with self.lock:
            if self.generator_state["active"]:
                self.logger.info("信号生成器已在运行")
                return True
            
            # 设置状态
            self.generator_state["active"] = True
            self.active = True
            
            # 启动生成线程
            if not self.generator_thread or not self.generator_thread.is_alive():
                self.generator_thread = threading.Thread(target=self._run_generator)
                self.generator_thread.daemon = True
                self.generator_thread.start()
            
            # 连接市场意识
            if self.market_consciousness:
                self.market_consciousness.register_observer(self)
            
            self.logger.info("量子交易信号生成器已启动")
            return True
    
    def stop(self):
        """停止信号生成器"""
        with self.lock:
            if not self.generator_state["active"]:
                return True
            
            # 设置状态
            self.generator_state["active"] = False
            self.active = False
            
            # 等待线程结束
            if self.generator_thread and self.generator_thread.is_alive():
                self.generator_thread.join(timeout=2.0)
            
            # 断开与市场意识的连接
            if self.market_consciousness:
                self.market_consciousness.unregister_observer(self)
            
            self.logger.info("量子交易信号生成器已停止")
            return True
    
    def register_observer(self, observer):
        """注册信号观察者
        
        Args:
            observer: 观察者对象，需实现on_trading_signal()方法
            
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
    
    def get_active_signals(self):
        """获取当前活跃的交易信号
        
        Returns:
            list: 活跃信号列表
        """
        with self.lock:
            return self.active_signals.copy()
    
    def get_signal_history(self, count=10):
        """获取历史信号
        
        Args:
            count: 返回的信号数量
            
        Returns:
            list: 历史信号列表
        """
        with self.lock:
            history = sorted(self.signal_history, key=lambda x: x["timestamp"], reverse=True)
            return history[:count]
    
    def on_market_event(self, event_type, data):
        """处理市场事件
        
        实现观察者接口，接收市场共生意识的事件
        
        Args:
            event_type: 事件类型
            data: 事件数据
            
        Returns:
            bool: 处理是否成功
        """
        if not self.active:
            return False
        
        try:
            if event_type == "market_insight":
                # 存储市场洞察
                self.market_insights.append(data)
                
                # 保持合理大小
                while len(self.market_insights) > 20:
                    self.market_insights.pop(0)
                
                # 特定类型的洞察可能触发信号生成
                if data["type"] in ["trend_change", "momentum_shift", "turning_point", "quantum_probability"]:
                    if data["confidence"] > self.config["confidence_threshold"]:
                        # 基于高置信度洞察生成信号
                        self._generate_signal_from_insight(data)
                
            elif event_type == "turning_point":
                # 转折点预警可能触发信号
                if data["confidence"] > self.config["confidence_threshold"] * 1.1:
                    # 基于转折点预警生成信号
                    self._generate_signal_from_turning_point(data)
            
            return True
            
        except Exception as e:
            self.logger.error(f"处理市场事件失败: {str(e)}")
            return False
    
    def _run_generator(self):
        """运行信号生成线程"""
        self.logger.info("启动量子交易信号生成线程")
        
        while self.active:
            try:
                # 信号生成间隔
                time.sleep(random.uniform(1.0, 3.0))
                
                with self.lock:
                    if not self.active:
                        break
                    
                    # 更新生成器状态
                    self._update_generator_state()
                    
                    # 定期尝试生成信号
                    self._try_generate_signal()
                    
                    # 更新活跃信号状态
                    self._update_active_signals()
                    
                    # 检查过期信号
                    self._clean_expired_signals()
                    
            except Exception as e:
                self.logger.error(f"信号生成线程发生错误: {str(e)}")
                time.sleep(5)  # 错误后等待较长时间
        
        self.logger.info("量子交易信号生成线程已停止")
    
    def _update_generator_state(self):
        """更新信号生成器状态"""
        # 随机波动
        coherence_change = random.uniform(-0.03, 0.03)
        self.generator_state["quantum_coherence"] = max(0.3, min(1.0, 
            self.generator_state["quantum_coherence"] + coherence_change))
        
        quality_change = random.uniform(-0.02, 0.02)
        self.generator_state["signal_quality"] = max(0.3, min(1.0, 
            self.generator_state["signal_quality"] + quality_change))
        
        accuracy_change = random.uniform(-0.02, 0.02)
        self.generator_state["accuracy"] = max(0.3, min(1.0, 
            self.generator_state["accuracy"] + accuracy_change))
        
        # 受场影响的状态
        if self.symbiotic_core and hasattr(self.symbiotic_core, 'field_state') and self.symbiotic_core.field_state["active"]:
            # 量子相干性与场稳定性相关
            field_stability = self.symbiotic_core.field_state["field_stability"]
            coherence_change = (field_stability - self.generator_state["quantum_coherence"]) * 0.1
            self.generator_state["quantum_coherence"] = max(0.3, min(1.0, 
                self.generator_state["quantum_coherence"] + coherence_change))
            
            # 处理能力与场强相关
            field_strength = self.symbiotic_core.field_state["field_strength"]
            processing_change = (field_strength - self.generator_state["processing_power"]) * 0.05
            self.generator_state["processing_power"] = max(0.3, min(1.0, 
                self.generator_state["processing_power"] + processing_change))
            
            # 更新可访问维度
            field_dimensions = self.symbiotic_core.field_state["dimension_count"]
            self.generator_state["dimensional_access"] = list(range(3, min(field_dimensions + 1, 12)))
        
        # 更新时间
        self.generator_state["last_update"] = datetime.now()
    
    def _try_generate_signal(self):
        """尝试生成交易信号"""
        # 检查是否超过最大活跃信号数量
        if len(self.active_signals) >= self.config["max_active_signals"]:
            return False
        
        # 检查最小信号间隔
        if (self.stats["last_signal_time"] and 
            (datetime.now() - self.stats["last_signal_time"]).total_seconds() < self.config["min_signal_interval"]):
            return False
        
        # 计算生成几率
        generation_chance = (
            self.config["base_generation_chance"] * 
            self.generator_state["processing_power"] * 
            random.uniform(0.8, 1.2)
        )
        
        # 市场意识可以增加生成几率
        if self.market_consciousness and hasattr(self.market_consciousness, 'consciousness_state'):
            awareness = self.market_consciousness.consciousness_state["awareness_level"]
            generation_chance *= (1 + awareness * 0.5)
        
        # 随机决定是否生成
        if random.random() < generation_chance:
            return self._generate_quantum_signal()
        
        return False
    
    def _generate_quantum_signal(self):
        """生成量子交易信号
        
        Returns:
            bool: 生成是否成功
        """
        # 信号类型
        signal_types = [
            "entry_long", "entry_short", "exit_long", "exit_short",
            "stop_adjustment", "target_adjustment", "position_sizing",
            "market_regime_change", "volatility_adjustment"
        ]
        
        # 根据维度访问解锁更高级的信号类型
        max_dimension = max(self.generator_state["dimensional_access"]) if self.generator_state["dimensional_access"] else 3
        
        if max_dimension >= 7:
            signal_types.extend([
                "multi_timeframe_alignment", "intermarket_correlation", 
                "probability_distribution", "quantum_hedge"
            ])
            
        if max_dimension >= 9:
            signal_types.extend([
                "timeline_optimization", "dimensional_arbitrage",
                "quantum_probability_collapse"
            ])
        
        # 选择信号类型
        signal_type = random.choice(signal_types)
        
        # 信号维度
        signal_dimension = random.choice(self.generator_state["dimensional_access"])
        
        # 计算信号质量
        quality_factors = [
            self.generator_state["quantum_coherence"] * 0.3,
            self.generator_state["signal_quality"] * 0.4,
            self.generator_state["accuracy"] * 0.3,
            random.uniform(0.7, 1.0) * 0.2  # 随机因素
        ]
        signal_quality = sum(quality_factors) / (1 + 0.2)  # 总权重1.2
        
        # 检查信号质量是否达到阈值
        if signal_quality < self.config["quality_threshold"]:
            self.logger.debug(f"信号质量{signal_quality:.2f}未达到阈值{self.config['quality_threshold']:.2f}，放弃生成")
            return False
        
        # 信号置信度
        confidence = signal_quality * random.uniform(0.8, 1.0)
        
        # 计算信号持续时间
        duration = random.randint(15, 240) * 60  # 15分钟到4小时，转换为秒
        
        # 资产选择 (示例资产列表)
        assets = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", 
                  "ADA/USDT", "DOGE/USDT", "MATIC/USDT", "DOT/USDT", "AVAX/USDT"]
        selected_asset = random.choice(assets)
        
        # 创建信号
        signal = {
            "id": f"sig_{int(time.time())}_{random.randint(1000, 9999)}",
            "type": signal_type,
            "asset": selected_asset,
            "quality": signal_quality,
            "confidence": confidence,
            "dimension": signal_dimension,
            "params": self._generate_signal_params(signal_type, selected_asset),
            "timestamp": datetime.now(),
            "expires_at": datetime.now().timestamp() + duration,
            "active": True,
            "executed": False,
            "success": None,
            "description": self._generate_signal_description(signal_type, selected_asset),
            "source": "quantum_generator",
            "generator_state": {
                k: v for k, v in self.generator_state.items()
                if k not in ["last_update", "dimensional_access"]
            }
        }
        
        # 添加到活跃信号
        self.active_signals.append(signal)
        
        # 更新统计信息
        self.stats["total_signals"] += 1
        self.stats["signal_types"][signal_type] += 1
        self.stats["dimension_usage"][signal_dimension] += 1
        self.stats["last_signal_time"] = datetime.now()
        
        # 更新平均质量
        total_quality = self.stats["average_quality"] * (self.stats["total_signals"] - 1) + signal_quality
        self.stats["average_quality"] = total_quality / self.stats["total_signals"]
        
        self.logger.info(f"生成量子交易信号: {signal_type} 资产:{selected_asset} 质量:{signal_quality:.2f}")
        
        # 通知观察者
        self._notify_observers(signal)
        
        return True
    
    def _generate_signal_from_insight(self, insight):
        """从市场洞察生成信号
        
        Args:
            insight: 市场洞察数据
            
        Returns:
            bool: 生成是否成功
        """
        # 检查是否超过最大活跃信号数量
        if len(self.active_signals) >= self.config["max_active_signals"]:
            return False
        
        # 转换洞察类型到信号类型
        insight_to_signal = {
            "trend_change": ["entry_long", "entry_short", "exit_long", "exit_short"],
            "momentum_shift": ["entry_long", "entry_short", "position_sizing"],
            "volatility_breakout": ["entry_long", "entry_short", "volatility_adjustment"],
            "pattern_formation": ["entry_long", "entry_short"],
            "turning_point": ["exit_long", "exit_short", "stop_adjustment"],
            "quantum_probability": ["quantum_probability_collapse", "dimensional_arbitrage"],
            "market_structure": ["market_regime_change"],
            "future_echo": ["timeline_optimization"]
        }
        
        # 获取可能的信号类型
        possible_signals = insight_to_signal.get(insight["type"], ["entry_long", "entry_short"])
        
        # 选择一个信号类型
        signal_type = random.choice(possible_signals)
        
        # 资产选择 (示例资产列表)
        assets = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", 
                  "ADA/USDT", "DOGE/USDT", "MATIC/USDT", "DOT/USDT", "AVAX/USDT"]
        selected_asset = random.choice(assets)
        
        # 计算信号质量，受洞察置信度影响
        quality = insight["confidence"] * self.generator_state["signal_quality"] * random.uniform(0.9, 1.1)
        
        # 计算信号持续时间
        duration = random.randint(30, 360) * 60  # 30分钟到6小时，转换为秒
        
        # 创建信号
        signal = {
            "id": f"isi_{int(time.time())}_{random.randint(1000, 9999)}",
            "type": signal_type,
            "asset": selected_asset,
            "quality": quality,
            "confidence": insight["confidence"],
            "dimension": insight["dimension"],
            "params": self._generate_signal_params(signal_type, selected_asset),
            "timestamp": datetime.now(),
            "expires_at": datetime.now().timestamp() + duration,
            "active": True,
            "executed": False,
            "success": None,
            "description": self._generate_signal_description(signal_type, selected_asset),
            "source": "market_insight",
            "insight_type": insight["type"],
            "generator_state": {
                k: v for k, v in self.generator_state.items()
                if k not in ["last_update", "dimensional_access"]
            }
        }
        
        # 添加到活跃信号
        self.active_signals.append(signal)
        
        # 更新统计信息
        self.stats["total_signals"] += 1
        self.stats["signal_types"][signal_type] += 1
        self.stats["dimension_usage"][insight["dimension"]] += 1
        self.stats["last_signal_time"] = datetime.now()
        
        self.logger.info(f"基于市场洞察生成交易信号: {signal_type} 资产:{selected_asset} 来源:{insight['type']}")
        
        # 通知观察者
        self._notify_observers(signal)
        
        return True
    
    def _generate_signal_from_turning_point(self, alert):
        """从转折点预警生成信号
        
        Args:
            alert: 转折点预警数据
            
        Returns:
            bool: 生成是否成功
        """
        # 转换预警类型到信号类型
        alert_to_signal = {
            "trend_reversal": ["exit_long", "exit_short", "entry_long", "entry_short"],
            "significant_breakout": ["entry_long", "entry_short", "position_sizing"],
            "pattern_completion": ["entry_long", "entry_short"],
            "momentum_exhaustion": ["exit_long", "exit_short", "stop_adjustment"]
        }
        
        # 获取可能的信号类型
        possible_signals = alert_to_signal.get(alert["type"], ["exit_long", "exit_short"])
        
        # 选择一个信号类型
        signal_type = random.choice(possible_signals)
        
        # 资产选择 (示例资产列表)
        assets = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", 
                  "ADA/USDT", "DOGE/USDT", "MATIC/USDT", "DOT/USDT", "AVAX/USDT"]
        selected_asset = random.choice(assets)
        
        # 计算信号质量，受预警重要性影响
        quality = alert["importance"] * self.generator_state["signal_quality"] * random.uniform(0.9, 1.1)
        
        # 计算信号持续时间
        # 使用预警的时间范围
        time_range_hours = alert.get("time_range", 24)
        duration = time_range_hours * 3600  # 转换为秒
        
        # 创建信号
        signal = {
            "id": f"tps_{int(time.time())}_{random.randint(1000, 9999)}",
            "type": signal_type,
            "asset": selected_asset,
            "quality": quality,
            "confidence": alert["confidence"],
            "dimension": alert.get("dimension_source", 5),
            "params": self._generate_signal_params(signal_type, selected_asset),
            "timestamp": datetime.now(),
            "expires_at": datetime.now().timestamp() + duration,
            "active": True,
            "executed": False,
            "success": None,
            "description": f"转折点预警触发: {alert['description']} - {self._generate_signal_description(signal_type, selected_asset)}",
            "source": "turning_point",
            "alert_type": alert["type"],
            "importance": alert["importance"],
            "generator_state": {
                k: v for k, v in self.generator_state.items()
                if k not in ["last_update", "dimensional_access"]
            }
        }
        
        # 添加到活跃信号
        self.active_signals.append(signal)
        
        # 更新统计信息
        self.stats["total_signals"] += 1
        self.stats["signal_types"][signal_type] += 1
        self.stats["dimension_usage"][alert.get("dimension_source", 5)] += 1
        self.stats["last_signal_time"] = datetime.now()
        
        self.logger.info(f"基于转折点预警生成交易信号: {signal_type} 资产:{selected_asset} 预警类型:{alert['type']}")
        
        # 通知观察者
        self._notify_observers(signal)
        
        return True
    
    def _generate_signal_params(self, signal_type, asset):
        """生成信号参数
        
        Args:
            signal_type: 信号类型
            asset: 资产
            
        Returns:
            dict: 信号参数
        """
        # 为不同类型的信号生成不同的参数
        if signal_type in ["entry_long", "entry_short"]:
            return {
                "price": round(random.uniform(10000, 40000), 2) if "BTC" in asset else round(random.uniform(100, 3000), 2),
                "size": round(random.uniform(0.1, 1.0), 3),
                "stop_loss": round(random.uniform(5, 15), 2),  # 百分比
                "take_profit": round(random.uniform(10, 30), 2),  # 百分比
                "timeframe": random.choice(["5m", "15m", "1h", "4h", "1d"]),
                "risk_reward": round(random.uniform(1.5, 3.0), 1)
            }
            
        elif signal_type in ["exit_long", "exit_short"]:
            return {
                "price": round(random.uniform(10000, 40000), 2) if "BTC" in asset else round(random.uniform(100, 3000), 2),
                "percentage": round(random.uniform(50, 100), 0),  # 退出百分比
                "reason": random.choice(["target_reached", "stop_hit", "signal_reversal", "time_exit", "volatility_exit"])
            }
            
        elif signal_type == "stop_adjustment":
            return {
                "old_stop": round(random.uniform(9000, 35000), 2) if "BTC" in asset else round(random.uniform(90, 2500), 2),
                "new_stop": round(random.uniform(9500, 36000), 2) if "BTC" in asset else round(random.uniform(95, 2600), 2),
                "reason": random.choice(["price_action", "volatility_change", "time_based", "support_resistance"])
            }
            
        elif signal_type == "position_sizing":
            return {
                "risk_percentage": round(random.uniform(0.5, 3.0), 2),  # 账户风险百分比
                "position_size": round(random.uniform(0.1, 2.0), 3),
                "optimal_leverage": round(random.uniform(1, 5), 1),
                "risk_adjustment_factor": round(random.uniform(0.8, 1.2), 2)
            }
            
        elif signal_type == "market_regime_change":
            return {
                "old_regime": random.choice(["bullish", "bearish", "ranging", "high_volatility", "low_volatility"]),
                "new_regime": random.choice(["bullish", "bearish", "ranging", "high_volatility", "low_volatility"]),
                "confidence": round(random.uniform(0.6, 0.95), 2),
                "adaptation_required": random.choice([True, False])
            }
            
        elif signal_type == "quantum_probability_collapse":
            return {
                "probability_distribution": {
                    "bullish": round(random.uniform(0, 1), 2),
                    "bearish": round(random.uniform(0, 1), 2),
                    "ranging": round(random.uniform(0, 1), 2)
                },
                "collapsed_state": random.choice(["bullish", "bearish", "ranging"]),
                "confidence": round(random.uniform(0.7, 0.95), 2),
                "quantum_entropy": round(random.uniform(0.1, 0.9), 2)
            }
            
        # 默认参数
        return {
            "timeframe": random.choice(["5m", "15m", "1h", "4h", "1d"]),
            "strength": round(random.uniform(0.5, 1.0), 2),
            "note": "量子生成的交易信号参数"
        }
    
    def _generate_signal_description(self, signal_type, asset):
        """生成信号描述
        
        Args:
            signal_type: 信号类型
            asset: 资产
            
        Returns:
            str: 信号描述
        """
        descriptions = {
            "entry_long": f"{asset} 出现做多信号，价格可能上涨。建议按照风险管理设置止损。",
            "entry_short": f"{asset} 出现做空信号，价格可能下跌。谨慎操作，设置适当止损。",
            "exit_long": f"{asset} 多头信号消失，建议退出多头仓位保护利润。",
            "exit_short": f"{asset} 空头信号消失，建议退出空头仓位规避风险。",
            "stop_adjustment": f"{asset} 止损位应调整以适应新的市场结构。",
            "target_adjustment": f"{asset} 获利目标应调整以优化风险回报。",
            "position_sizing": f"当前{asset}的最佳仓位大小已重新计算，建议调整。",
            "market_regime_change": f"{asset}市场状态发生变化，交易策略应相应调整。",
            "volatility_adjustment": f"{asset}波动率发生显著变化，风险参数需要调整。",
            "multi_timeframe_alignment": f"{asset}多时间周期分析显示趋势一致性。",
            "intermarket_correlation": f"{asset}与其他市场相关性信号，提供交易机会。",
            "probability_distribution": f"{asset}未来价格概率分布已更新，提供决策参考。",
            "quantum_hedge": f"{asset}量子对冲信号，可用于风险管理。",
            "timeline_optimization": f"{asset}最优交易时间窗口信号。",
            "dimensional_arbitrage": f"{asset}跨维度套利机会出现。",
            "quantum_probability_collapse": f"{asset}量子概率坍缩，确定性显著提高。"
        }
        
        return descriptions.get(signal_type, f"{asset} 交易信号触发，建议根据自身风险偏好决策。")
    
    def _update_active_signals(self):
        """更新活跃信号状态"""
        now = time.time()
        
        for signal in self.active_signals:
            # 检查信号是否过期
            if now > signal["expires_at"]:
                signal["active"] = False
                
                # 如果信号已执行但未设置成功状态，假设为失败
                if signal["executed"] and signal["success"] is None:
                    signal["success"] = False
                    self.stats["failed_signals"] += 1
    
    def _clean_expired_signals(self):
        """清理过期信号"""
        # 将不再活跃的信号移至历史记录
        expired_signals = [s for s in self.active_signals if not s["active"]]
        for signal in expired_signals:
            # 添加到历史
            self.signal_history.append(signal)
            # 从活跃列表移除
            self.active_signals.remove(signal)
        
        # 限制历史记录大小
        while len(self.signal_history) > self.max_history_length:
            self.signal_history.pop(0)
    
    def report_signal_execution(self, signal_id, success=True):
        """报告信号执行结果
        
        Args:
            signal_id: 信号ID
            success: 执行是否成功
            
        Returns:
            bool: 更新是否成功
        """
        with self.lock:
            # 在活跃信号中查找
            for signal in self.active_signals:
                if signal["id"] == signal_id:
                    signal["executed"] = True
                    signal["success"] = success
                    
                    # 更新统计
                    if success:
                        self.stats["successful_signals"] += 1
                    else:
                        self.stats["failed_signals"] += 1
                    
                    return True
            
            # 在历史信号中查找
            for signal in self.signal_history:
                if signal["id"] == signal_id:
                    signal["executed"] = True
                    signal["success"] = success
                    
                    # 更新统计
                    if success:
                        self.stats["successful_signals"] += 1
                    else:
                        self.stats["failed_signals"] += 1
                    
                    return True
            
            return False
    
    def _notify_observers(self, signal):
        """通知所有观察者
        
        Args:
            signal: 生成的信号
        """
        for observer in self.observers:
            try:
                if hasattr(observer, 'on_trading_signal'):
                    observer.on_trading_signal(signal)
            except Exception as e:
                self.logger.error(f"通知观察者失败: {str(e)}")

def get_quantum_signal_generator(symbiotic_core=None, market_consciousness=None):
    """获取量子交易信号生成器实例
    
    Args:
        symbiotic_core: 量子共生核心引用
        market_consciousness: 市场共生意识引用
        
    Returns:
        QuantumSignalGenerator: 信号生成器实例
    """
    generator = QuantumSignalGenerator(symbiotic_core, market_consciousness)
    return generator 