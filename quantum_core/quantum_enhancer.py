"""
quantum_enhancer - 高级量子增强模块
提供先进的量子算法增强选股能力，实现智能自适应学习
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class QuantumEnhancer:
    """量子增强器 - 利用高级量子算法和自适应学习提升选股能力"""
    
    def __init__(self, data_path=None):
        """初始化量子增强器
        
        Args:
            data_path: 数据存储路径，用于保存学习结果
        """
        self.data_path = data_path or os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        self.learning_data = self._load_learning_data()
        self.market_patterns = {}
        self.sector_weights = {}
        self.evolution_stage = 0
        logger.info("量子增强器初始化完成")
        
    def _load_learning_data(self) -> Dict[str, Any]:
        """加载学习数据"""
        try:
            learning_file = os.path.join(self.data_path, "quantum_learning.json")
            if os.path.exists(learning_file):
                with open(learning_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"已加载量子学习数据，版本: {data.get('version', '未知')}")
                return data
            else:
                logger.info("未找到学习数据，创建新的学习模型")
                return self._create_initial_learning_model()
        except Exception as e:
            logger.error(f"加载学习数据时出错: {str(e)}")
            return self._create_initial_learning_model()
            
    def _create_initial_learning_model(self) -> Dict[str, Any]:
        """创建初始学习模型"""
        return {
            "version": "1.0.0",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "evolution_count": 0,
            "market_patterns": {},
            "sector_weights": {},
            "technical_factor_weights": {
                "macd": 0.8,
                "rsi": 0.7,
                "bollinger": 0.6,
                "volume_trend": 0.9,
                "price_momentum": 0.85
            },
            "fundamental_factor_weights": {
                "pe_ratio": 0.7,
                "pb_ratio": 0.65,
                "net_profit_growth": 0.9,
                "roe": 0.85,
                "debt_ratio": 0.6
            },
            "correlation_matrix": {},
            "performance_history": []
        }
        
    def save_learning_data(self):
        """保存学习数据"""
        try:
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)
                
            learning_file = os.path.join(self.data_path, "quantum_learning.json")
            
            # 更新版本和时间
            self.learning_data["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.learning_data["evolution_count"] += 1
            
            with open(learning_file, 'w', encoding='utf-8') as f:
                json.dump(self.learning_data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"已保存量子学习数据，进化次数: {self.learning_data['evolution_count']}")
            return True
        except Exception as e:
            logger.error(f"保存学习数据时出错: {str(e)}")
            return False
            
    def enhance_stock_selection(self, stocks_data: List[Dict], quantum_power: int) -> List[Dict]:
        """增强股票选择结果
        
        Args:
            stocks_data: 原始股票数据列表
            quantum_power: 量子能力等级
            
        Returns:
            增强后的股票数据
        """
        logger.info(f"使用量子增强器处理 {len(stocks_data)} 只股票，量子能力: {quantum_power}")
        
        if not stocks_data:
            return stocks_data
            
        # 应用增强因子
        enhanced_stocks = []
        for stock in stocks_data:
            enhanced_stock = self._apply_quantum_enhancement(stock, quantum_power)
            enhanced_stocks.append(enhanced_stock)
            
        # 重新排序
        enhanced_stocks = sorted(enhanced_stocks, key=lambda x: x['quantum_score'], reverse=True)
        
        logger.info(f"量子增强完成，平均分数提升: {self._calculate_enhancement_rate(stocks_data, enhanced_stocks):.2f}%")
        return enhanced_stocks
        
    def _apply_quantum_enhancement(self, stock: Dict, quantum_power: int) -> Dict:
        """应用量子增强到单个股票
        
        Args:
            stock: 原始股票数据
            quantum_power: 量子能力等级
            
        Returns:
            增强后的股票数据
        """
        enhanced_stock = stock.copy()
        
        # 获取股票的行业
        sector = stock.get("sector", "未知")
        
        # 应用行业权重
        sector_weight = self.learning_data.get("sector_weights", {}).get(sector, 1.0)
        
        # 应用技术因子权重
        technical_weight = self._calculate_technical_weight(stock)
        
        # 应用基本面因子权重
        fundamental_weight = self._calculate_fundamental_weight(stock)
        
        # 计算量子增强系数
        quantum_factor = 1.0 + (quantum_power / 100) * 0.3
        
        # 应用自适应学习加成
        evolution_bonus = min(0.2, self.learning_data.get("evolution_count", 0) * 0.005)
        
        # 计算最终增强分数
        original_score = stock.get("quantum_score", 0)
        enhanced_score = original_score * sector_weight * technical_weight * fundamental_weight * quantum_factor * (1 + evolution_bonus)
        
        # 限制在0-100范围内
        enhanced_score = max(0, min(100, enhanced_score))
        
        # 更新股票数据
        enhanced_stock["quantum_score"] = enhanced_score
        
        # 调整预期收益
        if "expected_gain" in stock:
            enhanced_stock["expected_gain"] = stock["expected_gain"] * (1 + (enhanced_score - original_score) / 100)
            
        # 增加增强因子说明
        enhancement_factors = [
            f"行业权重: {sector_weight:.2f}",
            f"技术分析: {technical_weight:.2f}",
            f"基本面分析: {fundamental_weight:.2f}",
            f"量子增强: {quantum_factor:.2f}",
            f"自适应学习: {1 + evolution_bonus:.2f}"
        ]
        
        enhanced_stock["enhancement_factors"] = enhancement_factors
        
        return enhanced_stock
        
    def _calculate_technical_weight(self, stock: Dict) -> float:
        """计算技术因子权重
        
        Args:
            stock: 股票数据
            
        Returns:
            技术因子权重
        """
        # 获取技术因子权重
        weights = self.learning_data.get("technical_factor_weights", {})
        
        # 默认权重为1.0
        if not weights:
            return 1.0
            
        # 计算加权平均
        total_weight = 0.0
        total_score = 0.0
        
        for factor, weight in weights.items():
            factor_value = self._get_stock_factor(stock, factor, default=0.5)
            total_weight += weight
            total_score += factor_value * weight
            
        if total_weight == 0:
            return 1.0
            
        avg_score = total_score / total_weight
        return 0.8 + avg_score * 0.4  # 范围在0.8-1.2之间
        
    def _calculate_fundamental_weight(self, stock: Dict) -> float:
        """计算基本面因子权重
        
        Args:
            stock: 股票数据
            
        Returns:
            基本面因子权重
        """
        # 获取基本面因子权重
        weights = self.learning_data.get("fundamental_factor_weights", {})
        
        # 默认权重为1.0
        if not weights:
            return 1.0
            
        # 计算加权平均
        total_weight = 0.0
        total_score = 0.0
        
        for factor, weight in weights.items():
            factor_value = self._get_stock_factor(stock, factor, default=0.5)
            total_weight += weight
            total_score += factor_value * weight
            
        if total_weight == 0:
            return 1.0
            
        avg_score = total_score / total_weight
        return 0.8 + avg_score * 0.4  # 范围在0.8-1.2之间
        
    def _get_stock_factor(self, stock: Dict, factor_name: str, default: float = 0.5) -> float:
        """获取股票因子值
        
        Args:
            stock: 股票数据
            factor_name: 因子名称
            default: 默认值
            
        Returns:
            因子值 (0-1范围)
        """
        # 从股票数据中获取因子值
        if "factors" in stock and factor_name in stock["factors"]:
            return stock["factors"][factor_name]
            
        # 使用默认值
        return default
        
    def _calculate_enhancement_rate(self, original_stocks: List[Dict], 
                                  enhanced_stocks: List[Dict]) -> float:
        """计算增强率
        
        Args:
            original_stocks: 原始股票列表
            enhanced_stocks: 增强后的股票列表
            
        Returns:
            平均增强率（百分比）
        """
        if not original_stocks or not enhanced_stocks:
            return 0.0
            
        total_improvement = 0.0
        count = 0
        
        # 创建代码到增强股票的映射
        enhanced_map = {s["code"]: s for s in enhanced_stocks}
        
        for original in original_stocks:
            code = original.get("code")
            if code in enhanced_map:
                original_score = original.get("quantum_score", 0)
                enhanced_score = enhanced_map[code].get("quantum_score", 0)
                
                if original_score > 0:
                    improvement = (enhanced_score - original_score) / original_score * 100
                    total_improvement += improvement
                    count += 1
                    
        if count == 0:
            return 0.0
            
        return total_improvement / count
        
    def learn_from_backtest(self, backtest_results: Dict[str, Any]) -> bool:
        """从回测结果学习并优化模型
        
        Args:
            backtest_results: 回测结果数据
            
        Returns:
            是否成功学习
        """
        try:
            logger.info("从回测结果学习以优化量子模型")
            
            # 验证输入
            if not backtest_results or "tests" not in backtest_results:
                logger.warning("回测结果数据无效，无法学习")
                return False
                
            # 保存回测性能历史
            self._add_performance_history(backtest_results)
            
            # 更新行业权重
            self._update_sector_weights(backtest_results)
            
            # 更新技术和基本面因子权重
            self._update_factor_weights(backtest_results)
            
            # 更新相关性矩阵
            self._update_correlation_matrix(backtest_results)
            
            # 增加进化阶段
            self.evolution_stage += 1
            self.learning_data["evolution_stage"] = self.evolution_stage
            
            # 保存学习数据
            return self.save_learning_data()
            
        except Exception as e:
            logger.error(f"从回测结果学习时出错: {str(e)}")
            return False
            
    def _add_performance_history(self, backtest_results: Dict[str, Any]):
        """添加性能历史记录
        
        Args:
            backtest_results: 回测结果数据
        """
        # 确保性能历史存在
        if "performance_history" not in self.learning_data:
            self.learning_data["performance_history"] = []
            
        # 创建性能摘要
        summary = backtest_results.get("summary", {})
        
        performance_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "evolution_stage": self.evolution_stage,
            "average_return": summary.get("average_return", 0),
            "win_rate": summary.get("average_win_rate", 0),
            "excess_return": summary.get("average_excess_return", 0),
            "sharpe_ratio": summary.get("average_sharpe", 0),
            "score_correlation": summary.get("score_correlation", 0)
        }
        
        # 添加到历史记录
        self.learning_data["performance_history"].append(performance_entry)
        
        # 限制历史记录大小
        if len(self.learning_data["performance_history"]) > 20:
            self.learning_data["performance_history"] = self.learning_data["performance_history"][-20:]
            
    def _update_sector_weights(self, backtest_results: Dict[str, Any]):
        """更新行业权重
        
        Args:
            backtest_results: 回测结果数据
        """
        # 初始化行业权重字典
        sector_weights = self.learning_data.get("sector_weights", {})
        
        # 分析每个测试的行业表现
        for test in backtest_results.get("tests", []):
            stock_performances = test.get("backtest_result", {}).get("stock_performances", {})
            
            # 收集每个行业的表现
            sector_performance = {}
            
            for stock_code, performance in stock_performances.items():
                # 尝试从选中的股票中找到行业信息
                sector = None
                for stock in test.get("selected_stocks_details", []):
                    if stock.get("code") == stock_code:
                        sector = stock.get("sector")
                        break
                        
                if not sector:
                    continue
                    
                if sector not in sector_performance:
                    sector_performance[sector] = []
                    
                sector_performance[sector].append(performance.get("actual_return", 0))
                
            # 计算每个行业的平均表现
            for sector, returns in sector_performance.items():
                if not returns:
                    continue
                    
                avg_return = sum(returns) / len(returns)
                
                # 调整行业权重
                if sector not in sector_weights:
                    sector_weights[sector] = 1.0
                    
                # 正向回报增加权重，负向回报减少权重
                adjustment = avg_return / 100  # 将百分比转换为小数
                new_weight = sector_weights[sector] * (1 + adjustment)
                
                # 限制权重范围
                sector_weights[sector] = max(0.5, min(1.5, new_weight))
                
        # 更新学习数据
        self.learning_data["sector_weights"] = sector_weights
        
    def _update_factor_weights(self, backtest_results: Dict[str, Any]):
        """更新因子权重
        
        Args:
            backtest_results: 回测结果数据
        """
        # 目前实现简化版，未来可以扩展
        # 这里我们根据整体表现稍微调整权重
        
        # 获取平均超额收益
        avg_excess_return = backtest_results.get("summary", {}).get("average_excess_return", 0)
        
        # 如果超额收益为正，增强当前模型
        if avg_excess_return > 0:
            self._strengthen_factor_weights()
        else:
            self._diversify_factor_weights()
            
    def _strengthen_factor_weights(self):
        """强化当前因子权重"""
        # 技术因子
        tech_weights = self.learning_data.get("technical_factor_weights", {})
        for factor in tech_weights:
            # 增加权重，但有上限
            tech_weights[factor] = min(1.0, tech_weights[factor] * 1.05)
            
        # 基本面因子
        fund_weights = self.learning_data.get("fundamental_factor_weights", {})
        for factor in fund_weights:
            # 增加权重，但有上限
            fund_weights[factor] = min(1.0, fund_weights[factor] * 1.05)
            
    def _diversify_factor_weights(self):
        """多样化因子权重，避免过度拟合"""
        # 技术因子
        tech_weights = self.learning_data.get("technical_factor_weights", {})
        for factor in tech_weights:
            # 轻微降低权重，增加随机性
            tech_weights[factor] = max(0.4, tech_weights[factor] * 0.98)
            
        # 基本面因子
        fund_weights = self.learning_data.get("fundamental_factor_weights", {})
        for factor in fund_weights:
            # 轻微降低权重，增加随机性
            fund_weights[factor] = max(0.4, fund_weights[factor] * 0.98)
            
    def _update_correlation_matrix(self, backtest_results: Dict[str, Any]):
        """更新相关性矩阵
        
        Args:
            backtest_results: 回测结果数据
        """
        # 简化实现，仅记录量子分数与实际回报的相关性
        correlation = backtest_results.get("summary", {}).get("score_correlation", 0)
        
        if "correlation_history" not in self.learning_data:
            self.learning_data["correlation_history"] = []
            
        self.learning_data["correlation_history"].append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "score_correlation": correlation
        })
        
        # 限制历史大小
        if len(self.learning_data["correlation_history"]) > 20:
            self.learning_data["correlation_history"] = self.learning_data["correlation_history"][-20:] 