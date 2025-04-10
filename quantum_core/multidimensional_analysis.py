"""
multidimensional_analysis - 量子核心组件
多维分析器 - 处理市场数据的多维分析
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class MultidimensionalAnalyzer:
    """多维分析器 - 分析市场数据的多维特性"""
    
    def __init__(self):
        self.is_running = False
        self.dimensions = {}
        self.analysis_results = {}
        logger.info("多维分析器初始化完成")
        
    def start(self):
        """启动分析器"""
        if self.is_running:
            logger.warning("多维分析器已在运行")
            return
            
        logger.info("启动多维分析器...")
        self.is_running = True
        logger.info("多维分析器启动完成")
        
    def stop(self):
        """停止分析器"""
        if not self.is_running:
            logger.warning("多维分析器已停止")
            return
            
        logger.info("停止多维分析器...")
        self.is_running = False
        logger.info("多维分析器已停止")
        
    def add_dimension(self, name: str, analyzer_func=None, description: str = ""):
        """添加分析维度"""
        if name in self.dimensions:
            logger.warning(f"维度 '{name}' 已存在，将被覆盖")
            
        self.dimensions[name] = {
            'function': analyzer_func,
            'description': description,
            'active': True
        }
        
        logger.info(f"添加分析维度: {name}")
        return True
        
    def remove_dimension(self, name: str):
        """移除分析维度"""
        if name not in self.dimensions:
            logger.warning(f"维度 '{name}' 不存在")
            return False
            
        del self.dimensions[name]
        logger.info(f"移除分析维度: {name}")
        return True
        
    def analyze(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """执行多维分析"""
        if not self.is_running:
            logger.warning("多维分析器未运行，无法执行分析")
            return {}
            
        if not market_data:
            logger.warning("无市场数据可分析")
            return {}
            
        logger.info("开始多维分析...")
        
        # 清除旧的分析结果
        self.analysis_results = {}
        
        # 执行每个维度的分析
        for name, dimension in self.dimensions.items():
            if not dimension['active']:
                continue
                
            try:
                analyzer_func = dimension['function']
                
                # 如果没有提供自定义分析函数，使用默认分析
                if analyzer_func is None:
                    result = self._default_analysis(name, market_data)
                else:
                    result = analyzer_func(market_data)
                    
                self.analysis_results[name] = result
                logger.debug(f"维度 '{name}' 分析完成")
                
            except Exception as e:
                logger.error(f"维度 '{name}' 分析失败: {str(e)}")
                
        # 综合各个维度的结果
        combined_result = self._combine_results(self.analysis_results)
        self.analysis_results['combined'] = combined_result
        
        logger.info(f"多维分析完成，分析了 {len(self.dimensions)} 个维度")
        
        return self.analysis_results
        
    def _default_analysis(self, dimension_name: str, market_data: Dict[str, pd.DataFrame]) -> Dict:
        """默认分析方法"""
        results = {}
        
        for symbol, df in market_data.items():
            if len(df) == 0:
                continue
                
            # 基本统计分析
            if 'close' in df.columns:
                # 计算收盘价的基本统计数据
                close_prices = df['close'].values
                results[symbol] = {
                    'mean': float(np.mean(close_prices)),
                    'std': float(np.std(close_prices)),
                    'min': float(np.min(close_prices)),
                    'max': float(np.max(close_prices)),
                    'latest': float(close_prices[-1]),
                    'count': len(close_prices)
                }
                
                # 计算简单的趋势指标
                if len(close_prices) > 1:
                    changes = np.diff(close_prices)
                    results[symbol]['trend'] = {
                        'direction': 'up' if changes[-1] > 0 else 'down',
                        'strength': float(abs(changes[-1] / close_prices[-2]) if close_prices[-2] != 0 else 0),
                        'volatility': float(np.std(changes) / np.mean(close_prices) if np.mean(close_prices) != 0 else 0)
                    }
                    
        return {
            'dimension': dimension_name,
            'type': 'default_analysis',
            'data': results
        }
        
    def _combine_results(self, dimension_results: Dict[str, Any]) -> Dict[str, Any]:
        """合并各维度的分析结果"""
        # 提取所有分析的股票代码
        symbols = set()
        for dim_name, result in dimension_results.items():
            if 'data' in result:
                symbols.update(result['data'].keys())
                
        combined = {}
        
        # 对每个股票合并分析结果
        for symbol in symbols:
            symbol_data = {}
            
            for dim_name, result in dimension_results.items():
                if 'data' in result and symbol in result['data']:
                    symbol_data[dim_name] = result['data'][symbol]
                    
            # 计算综合评分
            if symbol_data:
                score = self._calculate_score(symbol_data)
                combined[symbol] = {
                    'dimensions': symbol_data,
                    'score': score
                }
                
        return {
            'type': 'combined_analysis',
            'symbols': {symbol: data['score'] for symbol, data in combined.items()},
            'details': combined
        }
        
    def _calculate_score(self, symbol_data: Dict[str, Any]) -> float:
        """计算股票的综合评分 (0-100)"""
        # 简化的评分计算
        score = 50.0  # 默认中性评分
        
        # 这里可以添加更复杂的评分逻辑
        # 示例：根据趋势方向调整评分
        for dim_name, data in symbol_data.items():
            if isinstance(data, dict) and 'trend' in data:
                trend = data['trend']
                if trend['direction'] == 'up':
                    score += 5.0 * trend['strength'] * 100
                else:
                    score -= 5.0 * trend['strength'] * 100
                    
                # 根据波动性调整
                volatility_penalty = trend.get('volatility', 0) * 20
                score -= volatility_penalty
                
        # 确保评分在0-100范围内
        return max(0, min(100, score))
        
    def get_dimensions(self) -> List[str]:
        """获取所有分析维度"""
        return list(self.dimensions.keys())
        
    def get_latest_results(self) -> Dict[str, Any]:
        """获取最新分析结果"""
        return self.analysis_results

