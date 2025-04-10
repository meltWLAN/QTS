"""
strategy_evaluator - 量子核心组件
策略评估器 - 评估交易策略性能和风险
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Callable, Optional, Union, Tuple

logger = logging.getLogger(__name__)

class StrategyEvaluator:
    """策略评估器 - 评估交易策略的性能和风险指标"""
    
    def __init__(self):
        self.is_running = False
        self.metrics = {}  # 评估指标注册表
        self.results_cache = {}  # 结果缓存
        self._register_default_metrics()
        logger.info("策略评估器初始化完成")
        
    def start(self):
        """启动评估器"""
        if self.is_running:
            logger.warning("策略评估器已在运行")
            return
            
        logger.info("启动策略评估器...")
        self.is_running = True
        logger.info("策略评估器启动完成")
        
    def stop(self):
        """停止评估器"""
        if not self.is_running:
            logger.warning("策略评估器已停止")
            return
            
        logger.info("停止策略评估器...")
        self.is_running = False
        logger.info("策略评估器已停止")
        
    def _register_default_metrics(self):
        """注册默认评估指标"""
        # 收益类指标
        self.register_metric('total_return', self._calculate_total_return, 
                          '计算总收益率', 'return')
        self.register_metric('annualized_return', self._calculate_annualized_return, 
                          '计算年化收益率', 'return')
        self.register_metric('daily_returns', self._calculate_daily_returns, 
                          '计算每日收益率', 'return')
                          
        # 风险类指标
        self.register_metric('volatility', self._calculate_volatility, 
                          '计算波动率', 'risk')
        self.register_metric('sharpe_ratio', self._calculate_sharpe_ratio, 
                          '计算夏普比率', 'risk')
        self.register_metric('max_drawdown', self._calculate_max_drawdown, 
                          '计算最大回撤', 'risk')
        self.register_metric('sortino_ratio', self._calculate_sortino_ratio, 
                          '计算索提诺比率', 'risk')
                          
        # 交易类指标
        self.register_metric('win_rate', self._calculate_win_rate, 
                          '计算胜率', 'trade')
        self.register_metric('profit_factor', self._calculate_profit_factor, 
                          '计算盈亏比', 'trade')
        self.register_metric('avg_trade', self._calculate_avg_trade, 
                          '计算平均每笔交易收益', 'trade')
                          
        logger.info(f"注册了 {len(self.metrics)} 个默认评估指标")
        
    def register_metric(self, name: str, func: Callable, description: str = "", 
                       category: str = "custom"):
        """注册自定义评估指标"""
        if name in self.metrics:
            logger.warning(f"指标 '{name}' 已存在，将被替换")
            
        self.metrics[name] = {
            'function': func,
            'description': description,
            'category': category
        }
        
        logger.info(f"注册评估指标: {name} ({category})")
        return True
        
    def unregister_metric(self, name: str):
        """注销评估指标"""
        if name not in self.metrics:
            logger.warning(f"指标 '{name}' 不存在")
            return False
            
        del self.metrics[name]
        logger.info(f"注销评估指标: {name}")
        return True
        
    def evaluate_strategy(self, strategy_id: str, equity_curve: pd.Series, 
                        trades: Optional[List[Dict]] = None, 
                        benchmark: Optional[pd.Series] = None,
                        metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """评估策略性能"""
        if not self.is_running:
            logger.warning("策略评估器未运行，无法执行评估")
            return {'status': 'error', 'message': '策略评估器未运行'}
            
        logger.info(f"开始评估策略: {strategy_id}")
        
        # 如果未指定要计算的指标，则计算所有指标
        if metrics is None:
            metrics = list(self.metrics.keys())
            
        results = {
            'strategy_id': strategy_id,
            'metrics': {}
        }
        
        # 计算每个指标
        for metric_name in metrics:
            if metric_name not in self.metrics:
                logger.warning(f"指标 '{metric_name}' 不存在，已跳过")
                continue
                
            try:
                metric_func = self.metrics[metric_name]['function']
                metric_value = metric_func(equity_curve, trades, benchmark)
                results['metrics'][metric_name] = metric_value
                
            except Exception as e:
                logger.error(f"计算指标 '{metric_name}' 时出错: {str(e)}")
                results['metrics'][metric_name] = None
                
        # 缓存结果
        self.results_cache[strategy_id] = results
        
        logger.info(f"策略 '{strategy_id}' 评估完成，计算了 {len(results['metrics'])} 个指标")
        
        return results
        
    def get_evaluation_result(self, strategy_id: str) -> Dict[str, Any]:
        """获取缓存的评估结果"""
        if strategy_id not in self.results_cache:
            logger.warning(f"策略 '{strategy_id}' 的评估结果不存在")
            return {
                'status': 'error',
                'message': f"策略 '{strategy_id}' 的评估结果不存在"
            }
            
        return {
            'status': 'success',
            'result': self.results_cache[strategy_id]
        }
        
    def compare_strategies(self, strategy_ids: List[str], 
                         metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """比较多个策略的性能"""
        if not self.is_running:
            logger.warning("策略评估器未运行，无法执行比较")
            return {'status': 'error', 'message': '策略评估器未运行'}
            
        # 检查所有策略的评估结果是否存在
        missing_strategies = []
        for strategy_id in strategy_ids:
            if strategy_id not in self.results_cache:
                missing_strategies.append(strategy_id)
                
        if missing_strategies:
            logger.warning(f"以下策略的评估结果不存在: {', '.join(missing_strategies)}")
            return {
                'status': 'error',
                'message': f"部分策略的评估结果不存在: {', '.join(missing_strategies)}"
            }
            
        # 如果未指定要比较的指标，则使用所有共有的指标
        if metrics is None:
            # 获取所有策略共有的指标
            common_metrics = set(self.results_cache[strategy_ids[0]]['metrics'].keys())
            for strategy_id in strategy_ids[1:]:
                common_metrics &= set(self.results_cache[strategy_id]['metrics'].keys())
            metrics = list(common_metrics)
            
        # 构建比较结果
        comparison = {
            'metrics': {},
            'strategies': strategy_ids
        }
        
        for metric in metrics:
            comparison['metrics'][metric] = {
                strategy_id: self.results_cache[strategy_id]['metrics'].get(metric, None)
                for strategy_id in strategy_ids
            }
            
        logger.info(f"比较了 {len(strategy_ids)} 个策略的 {len(metrics)} 个指标")
        
        return {
            'status': 'success',
            'comparison': comparison
        }
        
    def get_metrics_by_category(self, category: Optional[str] = None) -> List[Dict[str, str]]:
        """获取指定类别的评估指标"""
        result = []
        
        for name, metric in self.metrics.items():
            if category is None or metric['category'] == category:
                result.append({
                    'name': name,
                    'description': metric['description'],
                    'category': metric['category']
                })
                
        return result
        
    def get_all_metrics(self) -> Dict[str, List[Dict[str, str]]]:
        """获取所有评估指标，按类别分组"""
        categories = {}
        
        for name, metric in self.metrics.items():
            category = metric['category']
            if category not in categories:
                categories[category] = []
                
            categories[category].append({
                'name': name,
                'description': metric['description']
            })
            
        return categories
        
    # 指标计算方法
    def _calculate_total_return(self, equity_curve: pd.Series, 
                             trades: Optional[List[Dict]] = None, 
                             benchmark: Optional[pd.Series] = None) -> float:
        """计算总收益率"""
        if len(equity_curve) < 2:
            return 0.0
            
        return float((equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1.0)
        
    def _calculate_annualized_return(self, equity_curve: pd.Series, 
                                  trades: Optional[List[Dict]] = None, 
                                  benchmark: Optional[pd.Series] = None) -> float:
        """计算年化收益率"""
        if len(equity_curve) < 2:
            return 0.0
            
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1.0
        # 假设交易日为252天
        years = len(equity_curve) / 252
        
        if years == 0:
            return 0.0
            
        return float((1 + total_return) ** (1 / years) - 1)
        
    def _calculate_daily_returns(self, equity_curve: pd.Series, 
                              trades: Optional[List[Dict]] = None, 
                              benchmark: Optional[pd.Series] = None) -> List[float]:
        """计算每日收益率"""
        if len(equity_curve) < 2:
            return []
            
        daily_returns = equity_curve.pct_change().dropna()
        return daily_returns.tolist()
        
    def _calculate_volatility(self, equity_curve: pd.Series, 
                           trades: Optional[List[Dict]] = None, 
                           benchmark: Optional[pd.Series] = None) -> float:
        """计算年化波动率"""
        if len(equity_curve) < 2:
            return 0.0
            
        daily_returns = equity_curve.pct_change().dropna()
        annual_vol = daily_returns.std() * np.sqrt(252)
        return float(annual_vol)
        
    def _calculate_sharpe_ratio(self, equity_curve: pd.Series, 
                             trades: Optional[List[Dict]] = None, 
                             benchmark: Optional[pd.Series] = None, 
                             risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        if len(equity_curve) < 2:
            return 0.0
            
        daily_returns = equity_curve.pct_change().dropna()
        
        if daily_returns.empty or daily_returns.std() == 0:
            return 0.0
            
        annual_return = (1 + daily_returns.mean()) ** 252 - 1
        annual_vol = daily_returns.std() * np.sqrt(252)
        
        if annual_vol == 0:
            return 0.0
            
        sharpe = (annual_return - risk_free_rate) / annual_vol
        return float(sharpe)
        
    def _calculate_max_drawdown(self, equity_curve: pd.Series, 
                             trades: Optional[List[Dict]] = None, 
                             benchmark: Optional[pd.Series] = None) -> float:
        """计算最大回撤"""
        if len(equity_curve) < 2:
            return 0.0
            
        # 计算累计最大值
        running_max = equity_curve.cummax()
        # 计算当前回撤
        drawdown = (equity_curve / running_max - 1)
        # 获取最大回撤
        max_drawdown = float(drawdown.min())
        
        return max_drawdown
        
    def _calculate_sortino_ratio(self, equity_curve: pd.Series, 
                              trades: Optional[List[Dict]] = None, 
                              benchmark: Optional[pd.Series] = None, 
                              risk_free_rate: float = 0.02) -> float:
        """计算索提诺比率"""
        if len(equity_curve) < 2:
            return 0.0
            
        daily_returns = equity_curve.pct_change().dropna()
        
        if daily_returns.empty:
            return 0.0
            
        # 只考虑负收益进行计算
        negative_returns = daily_returns[daily_returns < 0]
        
        if negative_returns.empty or negative_returns.std() == 0:
            return 0.0
            
        annual_return = (1 + daily_returns.mean()) ** 252 - 1
        downside_deviation = negative_returns.std() * np.sqrt(252)
        
        sortino = (annual_return - risk_free_rate) / downside_deviation
        return float(sortino)
        
    def _calculate_win_rate(self, equity_curve: pd.Series, 
                         trades: Optional[List[Dict]] = None, 
                         benchmark: Optional[pd.Series] = None) -> float:
        """计算胜率"""
        if trades is None or len(trades) == 0:
            # 如果没有提供交易记录，则使用每日收益率估算
            if len(equity_curve) < 2:
                return 0.0
                
            daily_returns = equity_curve.pct_change().dropna()
            winning_days = (daily_returns > 0).sum()
            total_days = len(daily_returns)
            
            if total_days == 0:
                return 0.0
                
            return float(winning_days / total_days)
        else:
            # 使用交易记录计算
            profitable_trades = sum(1 for trade in trades if trade.get('profit', 0) > 0)
            total_trades = len(trades)
            
            if total_trades == 0:
                return 0.0
                
            return float(profitable_trades / total_trades)
            
    def _calculate_profit_factor(self, equity_curve: pd.Series, 
                              trades: Optional[List[Dict]] = None, 
                              benchmark: Optional[pd.Series] = None) -> float:
        """计算盈亏比"""
        if trades is None or len(trades) == 0:
            # 如果没有提供交易记录，则使用每日收益率估算
            if len(equity_curve) < 2:
                return 0.0
                
            daily_returns = equity_curve.pct_change().dropna()
            gross_profits = daily_returns[daily_returns > 0].sum()
            gross_losses = abs(daily_returns[daily_returns < 0].sum())
            
            if gross_losses == 0:
                return float('inf') if gross_profits > 0 else 0.0
                
            return float(gross_profits / gross_losses)
        else:
            # 使用交易记录计算
            gross_profits = sum(trade.get('profit', 0) for trade in trades if trade.get('profit', 0) > 0)
            gross_losses = abs(sum(trade.get('profit', 0) for trade in trades if trade.get('profit', 0) < 0))
            
            if gross_losses == 0:
                return float('inf') if gross_profits > 0 else 0.0
                
            return float(gross_profits / gross_losses)
            
    def _calculate_avg_trade(self, equity_curve: pd.Series, 
                          trades: Optional[List[Dict]] = None, 
                          benchmark: Optional[pd.Series] = None) -> float:
        """计算平均每笔交易收益"""
        if trades is None or len(trades) == 0:
            # 如果没有提供交易记录，则返回0
            return 0.0
        else:
            # 使用交易记录计算
            total_profit = sum(trade.get('profit', 0) for trade in trades)
            total_trades = len(trades)
            
            if total_trades == 0:
                return 0.0
                
            return float(total_profit / total_trades)

