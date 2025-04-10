#!/usr/bin/env python3
"""
超神量子共生系统 - 混沌场景推演引擎
模拟多个平行市场路径，识别蝴蝶效应，提供量子决策树
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union
import os
import json
import random

# 配置日志
logger = logging.getLogger("ChaosScenarioEngine")

class ChaosScenarioEngine:
    """混沌场景推演引擎 - 提供市场多路径模拟和决策支持"""
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化混沌场景推演引擎
        
        Args:
            config: 配置参数
        """
        logger.info("初始化混沌场景推演引擎...")
        
        # 设置默认配置
        self.config = {
            'scenario_count': 100,       # 模拟场景数量
            'time_horizon': 30,          # 时间跨度(天)
            'butterfly_sensitivity': 0.6, # 蝴蝶效应敏感度
            'decision_tree_depth': 5,    # 决策树深度
            'probability_precision': 0.95, # 概率精度
            'chaos_degree': 0.7,         # 混沌程度
            'random_seed': 42            # 随机数种子
        }
        
        # 更新自定义配置
        if config:
            self.config.update(config)
        
        # 设置随机数种子
        np.random.seed(self.config['random_seed'])
        random.seed(self.config['random_seed'])
        
        # 初始化场景存储
        self.scenarios = []
        self.butterfly_signals = []
        self.decision_tree = {}
        self.probability_cloud = {}
        
        logger.info(f"混沌场景推演引擎初始化完成，可模拟 {self.config['scenario_count']} 条平行市场路径")
    
    def generate_scenarios(self, market_data: pd.DataFrame, quantum_state: Optional[Dict] = None) -> List[Dict]:
        """生成多条市场路径场景
        
        Args:
            market_data: 市场历史数据
            quantum_state: 量子状态(可选)
            
        Returns:
            List[Dict]: 生成的场景列表
        """
        logger.info(f"开始生成 {self.config['scenario_count']} 条混沌市场场景...")
        
        try:
            if market_data is None or len(market_data) < 10:
                logger.error("市场数据不足，无法生成场景")
                return []
            
            # 清空之前的场景
            self.scenarios = []
            
            # 提取市场特征
            market_features = self._extract_market_features(market_data)
            
            # 计算基准趋势和波动
            trend, volatility = self._calculate_base_parameters(market_data)
            
            # 生成多条路径
            for i in range(self.config['scenario_count']):
                # 生成单条路径
                scenario = self._generate_single_scenario(market_data, trend, volatility, 
                                                          market_features, quantum_state, i)
                self.scenarios.append(scenario)
            
            # 识别蝴蝶效应信号
            self._identify_butterfly_effects()
            
            # 构建决策树
            self._build_decision_tree()
            
            # 生成概率云
            self._generate_probability_cloud()
            
            logger.info(f"成功生成 {len(self.scenarios)} 条混沌市场场景")
            return self.scenarios
            
        except Exception as e:
            logger.error(f"生成混沌场景失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []
    
    def _extract_market_features(self, market_data: pd.DataFrame) -> Dict:
        """提取市场特征
        
        Args:
            market_data: 市场历史数据
            
        Returns:
            Dict: 市场特征
        """
        features = {}
        
        if market_data is None or len(market_data) < 5:
            return features
            
        # 使用最近的数据
        recent_data = market_data.iloc[-20:] if len(market_data) >= 20 else market_data
        
        # 提取价格特征
        if 'close' in recent_data.columns:
            features['last_price'] = float(recent_data['close'].iloc[-1])
            features['price_mean'] = float(recent_data['close'].mean())
            features['price_std'] = float(recent_data['close'].std())
        
        # 提取交易量特征
        if 'volume' in recent_data.columns:
            features['volume_mean'] = float(recent_data['volume'].mean())
            features['volume_std'] = float(recent_data['volume'].std())
        
        # 计算趋势特征
        if 'close' in recent_data.columns and len(recent_data) >= 2:
            closes = recent_data['close'].values
            returns = np.diff(closes) / closes[:-1]
            features['mean_return'] = float(np.mean(returns))
            features['return_std'] = float(np.std(returns))
            
            # 计算动量
            if len(returns) >= 5:
                features['momentum_5d'] = float(np.sum(returns[-5:]))
            
            # 线性趋势强度
            if len(closes) >= 5:
                x = np.arange(len(closes))
                try:
                    slope, _, r_value, _, _ = np.polyfit(x, closes, 1, full=True)
                    features['trend_strength'] = float(r_value[0])
                    features['trend_slope'] = float(slope)
                except:
                    features['trend_strength'] = 0.0
                    features['trend_slope'] = 0.0
        
        return features
    
    def _calculate_base_parameters(self, market_data: pd.DataFrame) -> Tuple[float, float]:
        """计算基准趋势和波动率
        
        Args:
            market_data: 市场历史数据
            
        Returns:
            Tuple[float, float]: (趋势, 波动率)
        """
        trend, volatility = 0.0, 0.02  # 默认值
        
        if market_data is None or len(market_data) < 5:
            return trend, volatility
            
        if 'close' in market_data.columns:
            closes = market_data['close'].values
            if len(closes) >= 2:
                # 计算对数收益率
                log_returns = np.diff(np.log(closes))
                
                # 波动率 - 年化
                volatility = np.std(log_returns) * np.sqrt(252)
                
                # 趋势 - 基于近期平均收益率
                n = min(20, len(log_returns))
                trend = np.mean(log_returns[-n:]) * 252
                
        return trend, volatility
    
    def _generate_single_scenario(self, market_data: pd.DataFrame, base_trend: float, 
                                  base_volatility: float, market_features: Dict, 
                                  quantum_state: Optional[Dict], scenario_id: int) -> Dict:
        """生成单条市场路径场景
        
        Args:
            market_data: 市场历史数据
            base_trend: 基础趋势
            base_volatility: 基础波动率
            market_features: 市场特征
            quantum_state: 量子状态
            scenario_id: 场景ID
            
        Returns:
            Dict: 生成的场景
        """
        # 获取最后价格作为起点
        last_price = market_features.get('last_price', 3000.0)
        
        # 为每个场景生成随机变异的趋势和波动率
        scenario_volatility = base_volatility * (0.5 + random.random())
        
        # 根据混沌程度调整趋势变异
        chaos_factor = self.config['chaos_degree'] * (0.5 + random.random())
        trend_variation = random.normalvariate(0, 0.001) * chaos_factor
        scenario_trend = base_trend + trend_variation
        
        # 生成未来价格序列
        dates = [datetime.now() + timedelta(days=i) for i in range(self.config['time_horizon'])]
        prices = []
        volumes = []
        events = []
        
        current_price = last_price
        
        # 生成一些随机事件
        event_count = random.randint(0, 3)
        event_days = random.sample(range(1, self.config['time_horizon']), event_count)
        
        for i in range(self.config['time_horizon']):
            # 检查是否有事件发生
            if i in event_days:
                event_impact = random.normalvariate(0, 0.03)  # 事件冲击
                event_type = "positive" if event_impact > 0 else "negative"
                events.append({
                    "day": i,
                    "type": event_type,
                    "impact": abs(event_impact),
                    "description": f"{'利好' if event_type == 'positive' else '利空'}事件影响"
                })
                # 应用事件冲击
                current_price *= (1 + event_impact)
            
            # 正常的随机价格变动
            daily_return = random.normalvariate(scenario_trend / 252, scenario_volatility / np.sqrt(252))
            current_price *= (1 + daily_return)
            
            # 确保价格为正
            current_price = max(current_price, 0.01)
            
            prices.append(current_price)
            
            # 生成相应的成交量
            base_volume = market_features.get('volume_mean', 10000)
            volume_std = market_features.get('volume_std', 1000)
            daily_volume = random.normalvariate(base_volume, volume_std)
            daily_volume = max(daily_volume, 0)
            volumes.append(daily_volume)
        
        # 创建场景
        scenario = {
            "id": scenario_id,
            "dates": [d.strftime("%Y-%m-%d") for d in dates],
            "prices": prices,
            "volumes": volumes,
            "events": events,
            "parameters": {
                "trend": scenario_trend,
                "volatility": scenario_volatility,
                "final_price": prices[-1],
                "max_price": max(prices),
                "min_price": min(prices),
                "price_change": (prices[-1] / last_price - 1) * 100,
                "success_probability": random.random()  # 模拟成功概率
            }
        }
        
        return scenario
    
    def _identify_butterfly_effects(self):
        """识别蝴蝶效应信号"""
        if not self.scenarios or len(self.scenarios) < 2:
            return
        
        logger.info("识别蝴蝶效应信号...")
        
        # 清空之前的信号
        self.butterfly_signals = []
        
        # 找出发散路径
        final_prices = [s["parameters"]["final_price"] for s in self.scenarios]
        mean_price = np.mean(final_prices)
        std_price = np.std(final_prices)
        
        # 寻找超过2个标准差的场景
        divergent_scenarios = []
        for scenario in self.scenarios:
            final_price = scenario["parameters"]["final_price"]
            if abs(final_price - mean_price) > 2 * std_price:
                divergent_scenarios.append(scenario)
        
        # 分析每个发散场景的早期特征
        for scenario in divergent_scenarios:
            # 取前5天的价格变动
            if len(scenario["prices"]) > 5:
                early_changes = []
                for i in range(1, 5):
                    daily_change = (scenario["prices"][i] / scenario["prices"][i-1] - 1) * 100
                    early_changes.append(daily_change)
                
                # 找出最显著的早期变动
                max_early_change = max(early_changes, key=abs)
                day_of_max_change = early_changes.index(max_early_change) + 1
                
                # 创建蝴蝶效应信号
                butterfly_signal = {
                    "scenario_id": scenario["id"],
                    "signal_day": day_of_max_change,
                    "signal_change": max_early_change,
                    "final_divergence": (scenario["parameters"]["final_price"] - mean_price) / mean_price * 100,
                    "events": scenario["events"],
                    "sensitivity": abs(max_early_change / (scenario["parameters"]["final_price"] - mean_price) * 100) if mean_price != scenario["parameters"]["final_price"] else 0
                }
                
                self.butterfly_signals.append(butterfly_signal)
        
        logger.info(f"识别到 {len(self.butterfly_signals)} 个蝴蝶效应信号")
    
    def _build_decision_tree(self):
        """构建量子决策树"""
        if not self.scenarios:
            return
            
        logger.info("构建量子决策树...")
        
        # 初始化决策树
        self.decision_tree = {
            "root": {
                "type": "decision",
                "name": "市场策略选择",
                "children": []
            }
        }
        
        # 定义几个基本策略
        strategies = ["积极策略", "保守策略", "观望策略", "对冲策略"]
        
        # 为每个策略创建分支
        for strategy in strategies:
            strategy_node = {
                "type": "strategy",
                "name": strategy,
                "success_probability": 0,
                "expected_return": 0,
                "risk_level": 0,
                "children": []
            }
            
            # 评估策略在不同场景下的表现
            success_count = 0
            total_return = 0
            returns = []
            
            for scenario in self.scenarios:
                # 基于不同策略评估场景
                if strategy == "积极策略":
                    # 积极策略在上涨场景中表现更好
                    is_success = scenario["parameters"]["price_change"] > 0
                    expected_return = scenario["parameters"]["price_change"]
                elif strategy == "保守策略":
                    # 保守策略在震荡市场中表现更好
                    is_success = abs(scenario["parameters"]["price_change"]) < 10
                    expected_return = min(scenario["parameters"]["price_change"], 5)
                elif strategy == "观望策略":
                    # 观望策略避免大幅下跌
                    is_success = scenario["parameters"]["price_change"] > -5
                    expected_return = max(scenario["parameters"]["price_change"], 0) * 0.5
                else:  # 对冲策略
                    # 对冲策略在波动市场中表现更好
                    is_success = True  # 对冲假设总是成功
                    expected_return = abs(scenario["parameters"]["price_change"]) * 0.3
                
                if is_success:
                    success_count += 1
                
                total_return += expected_return
                returns.append(expected_return)
            
            # 计算策略统计数据
            success_probability = success_count / len(self.scenarios)
            avg_return = total_return / len(self.scenarios)
            risk_level = np.std(returns) if returns else 0
            
            # 更新策略节点
            strategy_node["success_probability"] = success_probability
            strategy_node["expected_return"] = avg_return
            strategy_node["risk_level"] = risk_level
            
            # 添加子场景节点
            for scenario_type in ["乐观", "中性", "悲观"]:
                scenario_node = {
                    "type": "scenario",
                    "name": f"{scenario_type}场景",
                    "probability": 0.34 if scenario_type == "中性" else 0.33,
                    "expected_return": avg_return * (1.5 if scenario_type == "乐观" else (1.0 if scenario_type == "中性" else 0.5)),
                    "description": f"在{scenario_type}市场条件下执行{strategy}"
                }
                strategy_node["children"].append(scenario_node)
            
            # 添加策略到决策树
            self.decision_tree["root"]["children"].append(strategy_node)
        
        logger.info("量子决策树构建完成")
    
    def _generate_probability_cloud(self):
        """生成概率云"""
        if not self.scenarios:
            return
            
        logger.info("生成概率云...")
        
        # 提取所有最终价格
        final_prices = [s["parameters"]["final_price"] for s in self.scenarios]
        
        # 计算价格范围
        min_price = min(final_prices)
        max_price = max(final_prices)
        price_range = max_price - min_price
        
        # 创建价格区间
        bins = 10
        bin_size = price_range / bins if price_range > 0 else 1
        price_bins = [min_price + i * bin_size for i in range(bins + 1)]
        
        # 计算每个区间的概率
        probabilities = []
        for i in range(bins):
            lower = price_bins[i]
            upper = price_bins[i + 1]
            count = sum(1 for p in final_prices if lower <= p < upper)
            probability = count / len(final_prices)
            
            probabilities.append({
                "price_range": f"{lower:.2f}-{upper:.2f}",
                "probability": probability,
                "scenario_count": count
            })
        
        # 添加最后一个区间（包含上限）
        if final_prices:
            count = sum(1 for p in final_prices if p == max_price)
            if count > 0:
                probabilities.append({
                    "price_range": f"{max_price:.2f}",
                    "probability": count / len(final_prices),
                    "scenario_count": count
                })
        
        # 设置概率云
        self.probability_cloud = {
            "price_bins": price_bins,
            "probabilities": probabilities,
            "mean_price": np.mean(final_prices),
            "median_price": np.median(final_prices),
            "std_dev": np.std(final_prices),
            "most_likely_range": max(probabilities, key=lambda x: x["probability"])["price_range"]
        }
        
        logger.info("概率云生成完成")
    
    def get_butterfly_signals(self) -> List[Dict]:
        """获取蝴蝶效应信号
        
        Returns:
            List[Dict]: 蝴蝶效应信号列表
        """
        return self.butterfly_signals
    
    def get_decision_tree(self) -> Dict:
        """获取决策树
        
        Returns:
            Dict: 决策树
        """
        return self.decision_tree
    
    def get_probability_cloud(self) -> Dict:
        """获取概率云
        
        Returns:
            Dict: 概率云
        """
        return self.probability_cloud
    
    def get_optimal_strategy(self) -> Dict:
        """获取最优策略
        
        Returns:
            Dict: 最优策略信息
        """
        if not self.decision_tree or "root" not in self.decision_tree:
            return {}
            
        strategies = self.decision_tree["root"]["children"]
        if not strategies:
            return {}
            
        # 基于预期收益和成功概率的综合得分
        for strategy in strategies:
            strategy["score"] = strategy["expected_return"] * strategy["success_probability"] - strategy["risk_level"] * 0.5
            
        # 选择得分最高的策略
        optimal_strategy = max(strategies, key=lambda s: s["score"])
        
        return {
            "name": optimal_strategy["name"],
            "success_probability": optimal_strategy["success_probability"],
            "expected_return": optimal_strategy["expected_return"],
            "risk_level": optimal_strategy["risk_level"],
            "score": optimal_strategy["score"]
        }
    
    def visualize_scenarios(self, output_file: str = "scenario_paths.png"):
        """可视化场景路径
        
        Args:
            output_file: 输出文件路径
            
        Returns:
            bool: 是否成功生成可视化
        """
        try:
            import matplotlib.pyplot as plt
            
            if not self.scenarios:
                logger.warning("没有可视化的场景")
                return False
                
            plt.figure(figsize=(12, 7))
            
            # 随机选择一部分场景以避免图表过于拥挤
            sample_size = min(20, len(self.scenarios))
            sampled_scenarios = random.sample(self.scenarios, sample_size)
            
            # 绘制每个场景
            for scenario in sampled_scenarios:
                prices = scenario["prices"]
                plt.plot(range(len(prices)), prices, alpha=0.3)
                
            # 绘制平均路径
            avg_prices = []
            for i in range(self.config['time_horizon']):
                avg_price = np.mean([s["prices"][i] for s in self.scenarios])
                avg_prices.append(avg_price)
            
            plt.plot(range(len(avg_prices)), avg_prices, 'r-', linewidth=2, label='平均路径')
            
            # 添加蝴蝶效应信号
            for signal in self.butterfly_signals[:3]:  # 只显示前3个信号
                scenario = next((s for s in self.scenarios if s["id"] == signal["scenario_id"]), None)
                if scenario:
                    day = signal["signal_day"]
                    plt.scatter([day], [scenario["prices"][day]], color='yellow', s=100, zorder=5, 
                                label=f"蝴蝶效应信号 (第{day}天)" if signal == self.butterfly_signals[0] else "")
                    plt.plot(range(len(scenario["prices"])), scenario["prices"], 'g-', linewidth=1.5, 
                             label=f"发散路径 {signal['scenario_id']}" if signal == self.butterfly_signals[0] else "")
            
            plt.title("混沌场景推演 - 多条平行市场路径")
            plt.xlabel("天数")
            plt.ylabel("价格")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 保存图表
            plt.savefig(output_file)
            logger.info(f"场景路径可视化已保存到: {output_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"可视化场景路径失败: {str(e)}")
            return False

# 当作为独立模块运行时，执行简单的测试
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 创建混沌场景引擎
    engine = ChaosScenarioEngine({'scenario_count': 30})
    
    # 创建模拟市场数据
    dates = pd.date_range(start='2025-01-01', periods=60)
    market_data = pd.DataFrame({
        'close': np.random.normal(3000, 30, 60) + np.linspace(0, 60, 60) * 2,  # 添加上升趋势
        'volume': np.random.normal(10000, 1000, 60),
        'open': np.random.normal(3000, 30, 60) + np.linspace(0, 60, 60) * 2,
        'high': np.random.normal(3050, 30, 60) + np.linspace(0, 60, 60) * 2,
        'low': np.random.normal(2950, 30, 60) + np.linspace(0, 60, 60) * 2
    }, index=dates)
    
    # 生成场景
    scenarios = engine.generate_scenarios(market_data)
    
    # 获取蝴蝶效应信号
    butterfly_signals = engine.get_butterfly_signals()
    print(f"发现 {len(butterfly_signals)} 个蝴蝶效应信号")
    
    # 获取最优策略
    optimal_strategy = engine.get_optimal_strategy()
    print(f"最优策略: {optimal_strategy.get('name')}")
    print(f"预期收益: {optimal_strategy.get('expected_return'):.2f}%")
    print(f"成功概率: {optimal_strategy.get('success_probability'):.2f}")
    
    # 可视化场景
    engine.visualize_scenarios() 