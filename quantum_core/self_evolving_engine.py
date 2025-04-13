"""
self_evolving_engine - 量子自进化引擎
提供系统自动学习和进化能力，优化选股和交易策略
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
import os
from datetime import datetime, timedelta
import random
import time

logger = logging.getLogger(__name__)

class SelfEvolvingEngine:
    """量子自进化引擎 - 使系统能够从历史数据中学习并自我优化"""
    
    def __init__(self, data_path=None, evolution_interval=24):
        """初始化自进化引擎
        
        Args:
            data_path: 数据存储路径
            evolution_interval: 进化间隔（小时）
        """
        self.data_path = data_path or os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        self.evolution_data = self._load_evolution_data()
        self.evolution_interval = evolution_interval
        self.last_evolution_time = self.evolution_data.get("last_evolution_time", None)
        self.evolution_generation = self.evolution_data.get("generation", 0)
        logger.info(f"量子自进化引擎初始化完成，当前进化代数: {self.evolution_generation}")
        
    def _load_evolution_data(self) -> Dict[str, Any]:
        """加载进化数据"""
        try:
            evolution_file = os.path.join(self.data_path, "evolution_data.json")
            if os.path.exists(evolution_file):
                with open(evolution_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"已加载进化数据，代数: {data.get('generation', 0)}")
                return data
            else:
                logger.info("未找到进化数据，创建初始进化数据")
                return self._create_initial_evolution_data()
        except Exception as e:
            logger.error(f"加载进化数据时出错: {str(e)}")
            return self._create_initial_evolution_data()
            
    def _create_initial_evolution_data(self) -> Dict[str, Any]:
        """创建初始进化数据"""
        return {
            "version": "1.0.0",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "last_evolution_time": None,
            "generation": 0,
            "fitness_history": [],
            "strategy_params": {
                "technical_weights": {
                    "macd": 0.7,
                    "rsi": 0.6,
                    "bollinger": 0.5,
                    "volume": 0.8,
                    "momentum": 0.7
                },
                "fundamental_weights": {
                    "pe_ratio": 0.6,
                    "pb_ratio": 0.5,
                    "roe": 0.8,
                    "profit_growth": 0.9,
                    "revenue_growth": 0.7
                },
                "timeframe_weights": {
                    "short_term": 0.6,
                    "medium_term": 0.8,
                    "long_term": 0.7
                },
                "sector_preferences": {},
                "risk_threshold": 0.5,
                "position_sizing": "equal"
            },
            "mutation_rate": 0.1,
            "crossover_rate": 0.7,
            "selection_pressure": 0.8
        }
        
    def save_evolution_data(self):
        """保存进化数据"""
        try:
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)
                
            evolution_file = os.path.join(self.data_path, "evolution_data.json")
            
            # 更新时间和版本
            self.evolution_data["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open(evolution_file, 'w', encoding='utf-8') as f:
                json.dump(self.evolution_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"已保存进化数据，当前代数: {self.evolution_data['generation']}")
            return True
        except Exception as e:
            logger.error(f"保存进化数据时出错: {str(e)}")
            return False
            
    def check_evolution_needed(self) -> bool:
        """检查是否需要进化
        
        Returns:
            是否需要进化
        """
        # 如果没有上次进化时间，需要进化
        if not self.last_evolution_time:
            return True
            
        # 解析上次进化时间
        try:
            last_time = datetime.strptime(self.last_evolution_time, "%Y-%m-%d %H:%M:%S")
            current_time = datetime.now()
            
            # 计算时间差
            time_diff = (current_time - last_time).total_seconds() / 3600  # 小时
            
            # 如果超过进化间隔，需要进化
            return time_diff >= self.evolution_interval
            
        except Exception as e:
            logger.error(f"检查进化时间时出错: {str(e)}")
            return True  # 出错时默认需要进化
            
    def evolve(self, performance_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """执行进化
        
        Args:
            performance_data: 性能数据，用于评估进化方向
            
        Returns:
            更新后的策略参数
        """
        logger.info(f"开始第 {self.evolution_generation + 1} 代进化")
        
        # 更新上次进化时间
        self.last_evolution_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.evolution_data["last_evolution_time"] = self.last_evolution_time
        
        # 获取当前策略参数
        current_params = self.evolution_data["strategy_params"]
        
        # 创建多个候选策略
        candidates = self._generate_candidates(current_params)
        
        # 如果有性能数据，使用它进行适应度评估
        # 否则使用模拟评估
        if performance_data:
            best_candidate = self._evaluate_with_real_data(candidates, performance_data)
        else:
            best_candidate = self._evaluate_with_simulation(candidates)
            
        # 更新最佳策略
        self.evolution_data["strategy_params"] = best_candidate
        
        # 增加代数
        self.evolution_generation += 1
        self.evolution_data["generation"] = self.evolution_generation
        
        # 保存进化数据
        self.save_evolution_data()
        
        logger.info(f"完成第 {self.evolution_generation} 代进化")
        return best_candidate
        
    def _generate_candidates(self, base_params: Dict[str, Any], 
                           num_candidates: int = 10) -> List[Dict[str, Any]]:
        """生成候选策略
        
        Args:
            base_params: 基础参数
            num_candidates: 候选数量
            
        Returns:
            候选策略列表
        """
        candidates = []
        
        # 添加基础策略（当前策略）
        candidates.append(base_params.copy())
        
        # 突变率
        mutation_rate = self.evolution_data.get("mutation_rate", 0.1)
        
        # 生成突变候选
        for _ in range(num_candidates - 1):
            candidate = self._mutate_params(base_params.copy(), mutation_rate)
            candidates.append(candidate)
            
        return candidates
        
    def _mutate_params(self, params: Dict[str, Any], 
                      mutation_rate: float) -> Dict[str, Any]:
        """对参数进行突变
        
        Args:
            params: 原始参数
            mutation_rate: 突变率
            
        Returns:
            突变后的参数
        """
        # 深拷贝参数
        mutated = params.copy()
        
        # 突变技术因子权重
        if "technical_weights" in mutated:
            for key in mutated["technical_weights"]:
                if random.random() < mutation_rate:
                    # 随机调整权重
                    current = mutated["technical_weights"][key]
                    adjustment = random.uniform(-0.2, 0.2)
                    mutated["technical_weights"][key] = max(0.1, min(1.0, current + adjustment))
                    
        # 突变基本面因子权重
        if "fundamental_weights" in mutated:
            for key in mutated["fundamental_weights"]:
                if random.random() < mutation_rate:
                    # 随机调整权重
                    current = mutated["fundamental_weights"][key]
                    adjustment = random.uniform(-0.2, 0.2)
                    mutated["fundamental_weights"][key] = max(0.1, min(1.0, current + adjustment))
                    
        # 突变时间框架权重
        if "timeframe_weights" in mutated:
            for key in mutated["timeframe_weights"]:
                if random.random() < mutation_rate:
                    # 随机调整权重
                    current = mutated["timeframe_weights"][key]
                    adjustment = random.uniform(-0.2, 0.2)
                    mutated["timeframe_weights"][key] = max(0.1, min(1.0, current + adjustment))
                    
        # 突变风险阈值
        if "risk_threshold" in mutated and random.random() < mutation_rate:
            current = mutated["risk_threshold"]
            adjustment = random.uniform(-0.15, 0.15)
            mutated["risk_threshold"] = max(0.1, min(0.9, current + adjustment))
            
        # 突变仓位策略
        if "position_sizing" in mutated and random.random() < mutation_rate * 0.5:
            strategies = ["equal", "weighted", "kelly"]
            current = mutated["position_sizing"]
            # 排除当前策略
            other_strategies = [s for s in strategies if s != current]
            if other_strategies:
                mutated["position_sizing"] = random.choice(other_strategies)
                
        return mutated
        
    def _evaluate_with_real_data(self, candidates: List[Dict[str, Any]],
                               performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """使用真实性能数据评估候选策略
        
        Args:
            candidates: 候选策略列表
            performance_data: 真实性能数据
            
        Returns:
            最佳候选策略
        """
        logger.info("使用真实性能数据评估候选策略")
        
        # 这里应该使用真实的性能数据评估每个候选策略
        # 对于这个简化实现，我们只是进行简单的随机评分
        
        # 从性能数据中提取关键指标
        success_sectors = performance_data.get("success_sectors", [])
        failed_sectors = performance_data.get("failed_sectors", [])
        best_timeframe = performance_data.get("best_timeframe", "medium_term")
        
        # 对每个候选策略打分
        scores = []
        
        for candidate in candidates:
            score = 0
            
            # 评估行业偏好
            sector_preferences = candidate.get("sector_preferences", {})
            for sector in success_sectors:
                if sector in sector_preferences and sector_preferences[sector] > 0.6:
                    score += 2
                    
            for sector in failed_sectors:
                if sector in sector_preferences and sector_preferences[sector] < 0.4:
                    score += 2
                    
            # 评估时间框架
            timeframe_weights = candidate.get("timeframe_weights", {})
            if best_timeframe in timeframe_weights and timeframe_weights[best_timeframe] > 0.7:
                score += 3
                
            # 添加一些随机性，避免过度拟合
            score += random.uniform(-1, 1)
            
            scores.append(score)
            
        # 选择最高分的候选策略
        best_index = scores.index(max(scores))
        best_candidate = candidates[best_index]
        
        # 记录适应度历史
        self._record_fitness(max(scores))
        
        return best_candidate
        
    def _evaluate_with_simulation(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """使用模拟数据评估候选策略
        
        Args:
            candidates: 候选策略列表
            
        Returns:
            最佳候选策略
        """
        logger.info("使用模拟数据评估候选策略")
        
        # 模拟评估
        scores = []
        
        # 市场行情偏好 (随机生成)
        market_trend = random.choice(["bull", "bear", "sideways"])
        
        # 特定行业表现 (随机生成)
        performing_sectors = random.sample(
            ["科技", "金融", "医药", "消费", "能源", "材料"], 
            k=random.randint(1, 3)
        )
        
        for candidate in candidates:
            # 基础分数
            score = random.uniform(5, 10)
            
            # 根据市场行情评分
            if market_trend == "bull":
                # 牛市中，进取策略更好
                risk_score = candidate.get("risk_threshold", 0.5) * 5
                score += risk_score
            elif market_trend == "bear":
                # 熊市中，保守策略更好
                risk_score = (1 - candidate.get("risk_threshold", 0.5)) * 5
                score += risk_score
                
            # 根据行业表现评分
            sector_prefs = candidate.get("sector_preferences", {})
            for sector in performing_sectors:
                if sector in sector_prefs:
                    sector_score = sector_prefs[sector] * 3
                    score += sector_score
                    
            # 技术和基本面权重平衡评分
            tech_weights = candidate.get("technical_weights", {})
            fund_weights = candidate.get("fundamental_weights", {})
            
            if tech_weights and fund_weights:
                avg_tech = sum(tech_weights.values()) / len(tech_weights)
                avg_fund = sum(fund_weights.values()) / len(fund_weights)
                
                # 不同市场中技术与基本面的平衡
                if market_trend == "bull" and avg_tech > avg_fund:
                    score += 2
                elif market_trend == "bear" and avg_fund > avg_tech:
                    score += 2
                elif market_trend == "sideways" and abs(avg_tech - avg_fund) < 0.2:
                    score += 2
                    
            scores.append(score)
            
        # 选择最高分的候选策略
        best_index = scores.index(max(scores))
        best_candidate = candidates[best_index]
        
        # 记录适应度历史
        self._record_fitness(max(scores))
        
        return best_candidate
        
    def _record_fitness(self, fitness_score: float):
        """记录适应度分数
        
        Args:
            fitness_score: 适应度分数
        """
        if "fitness_history" not in self.evolution_data:
            self.evolution_data["fitness_history"] = []
            
        fitness_entry = {
            "generation": self.evolution_generation,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "fitness": fitness_score
        }
        
        self.evolution_data["fitness_history"].append(fitness_entry)
        
        # 限制历史记录大小
        if len(self.evolution_data["fitness_history"]) > 30:
            self.evolution_data["fitness_history"] = self.evolution_data["fitness_history"][-30:]
            
    def get_current_strategy(self) -> Dict[str, Any]:
        """获取当前策略参数
        
        Returns:
            当前策略参数
        """
        return self.evolution_data["strategy_params"]
        
    def get_evolution_stats(self) -> Dict[str, Any]:
        """获取进化统计信息
        
        Returns:
            进化统计信息
        """
        stats = {
            "generation": self.evolution_generation,
            "last_evolution_time": self.last_evolution_time,
            "fitness_trend": [],
            "strategy_history": []
        }
        
        # 提取适应度趋势
        for entry in self.evolution_data.get("fitness_history", []):
            stats["fitness_trend"].append({
                "generation": entry.get("generation", 0),
                "fitness": entry.get("fitness", 0)
            })
            
        return stats
        
    def manual_adjust(self, param_path: List[str], value: Any) -> bool:
        """手动调整特定参数
        
        Args:
            param_path: 参数路径（例如["technical_weights", "macd"]）
            value: 新值
            
        Returns:
            是否成功调整
        """
        try:
            # 定位到要修改的参数
            target = self.evolution_data["strategy_params"]
            
            for i, key in enumerate(param_path):
                if i == len(param_path) - 1:
                    # 最后一个键，设置值
                    target[key] = value
                else:
                    # 中间键，继续导航
                    if key not in target:
                        target[key] = {}
                    target = target[key]
                    
            # 保存更改
            self.save_evolution_data()
            logger.info(f"手动调整参数: {'.'.join(param_path)} = {value}")
            return True
            
        except Exception as e:
            logger.error(f"手动调整参数时出错: {str(e)}")
            return False 