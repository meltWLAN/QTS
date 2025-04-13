"""
strategy_evaluator - 量子核心组件
策略评估器 - 评估交易策略性能和风险
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Callable, Optional, Union, Tuple
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)

class StrategyEvaluator:
    """策略评估器 - 评估交易策略的性能和风险指标"""
    
    def __init__(self, use_quantum_enhancement=False):
        self.is_running = False
        self.metrics = {}  # 评估指标注册表
        self.results_cache = {}  # 结果缓存
        self.use_quantum_enhancement = use_quantum_enhancement
        self.dimension_analysis_results = {}  # 维度分析结果
        self.strategy_clusters = {}  # 策略聚类结果
        self._register_default_metrics()
        
        # 量子增强配置
        self.quantum_config = {
            'enabled': use_quantum_enhancement,
            'circuit_depth': 3,
            'shots': 1024,
            'noise_model': None,
            'optimization_level': 1
        }
        
        # 多维分析配置
        self.dimension_config = {
            'pca_components': 3,
            'tsne_components': 2,
            'perplexity': 30,
            'kmeans_clusters': 4,
            'random_state': 42
        }
        
        logger.info("策略评估器初始化完成" + 
                  (" (已启用量子增强)" if use_quantum_enhancement else ""))
        
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
                        metrics: Optional[List[str]] = None,
                        use_quantum: Optional[bool] = None) -> Dict[str, Any]:
        """评估策略性能"""
        if not self.is_running:
            logger.warning("策略评估器未运行，无法执行评估")
            return {'status': 'error', 'message': '策略评估器未运行'}
            
        logger.info(f"开始评估策略: {strategy_id}")
        
        # 确定是否使用量子增强
        use_quantum_eval = self.use_quantum_enhancement
        if use_quantum is not None:
            use_quantum_eval = use_quantum
            
        # 如果未指定要计算的指标，则计算所有指标
        if metrics is None:
            metrics = list(self.metrics.keys())
            
        results = {
            'strategy_id': strategy_id,
            'metrics': {},
            'quantum_enhanced': use_quantum_eval
        }
        
        # 计算每个指标
        for metric_name in metrics:
            if metric_name not in self.metrics:
                logger.warning(f"指标 '{metric_name}' 不存在，已跳过")
                continue
                
            try:
                metric_func = self.metrics[metric_name]['function']
                
                # 判断是否使用量子增强版本的指标计算
                if use_quantum_eval and hasattr(self, f"_quantum_{metric_name}"):
                    quantum_func = getattr(self, f"_quantum_{metric_name}")
                    metric_value = quantum_func(equity_curve, trades, benchmark)
                    results['metrics'][metric_name] = {
                        'value': metric_value,
                        'quantum_enhanced': True
                    }
                else:
                    metric_value = metric_func(equity_curve, trades, benchmark)
                    results['metrics'][metric_name] = {
                        'value': metric_value,
                        'quantum_enhanced': False
                    }
                
            except Exception as e:
                logger.error(f"计算指标 '{metric_name}' 时出错: {str(e)}")
                results['metrics'][metric_name] = None
                
        # 添加稳健性评分 (量子噪声抵抗能力)
        if use_quantum_eval:
            try:
                robustness_score = self._calculate_quantum_robustness(equity_curve)
                results['metrics']['quantum_robustness'] = {
                    'value': robustness_score,
                    'quantum_enhanced': True
                }
            except Exception as e:
                logger.error(f"计算量子稳健性评分时出错: {str(e)}")
        
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

    # 新增的多维分析功能
    def perform_dimension_analysis(self, strategy_ids: List[str] = None) -> Dict[str, Any]:
        """执行策略的多维分析
        
        Args:
            strategy_ids: 要分析的策略ID列表，如果为None则分析所有策略
            
        Returns:
            分析结果
        """
        if not self.is_running:
            logger.warning("策略评估器未运行，无法执行多维分析")
            return {'status': 'error', 'message': '策略评估器未运行'}
            
        # 如果未指定策略，则使用所有已评估的策略
        if strategy_ids is None:
            strategy_ids = list(self.results_cache.keys())
            
        if len(strategy_ids) < 2:
            logger.warning("至少需要两个策略才能执行多维分析")
            return {'status': 'error', 'message': '至少需要两个策略才能执行多维分析'}
            
        logger.info(f"开始对 {len(strategy_ids)} 个策略执行多维分析")
        
        # 1. 准备数据矩阵
        metric_names = []
        data_matrix = []
        
        # 找出所有策略共有的指标
        common_metrics = set()
        for strategy_id in strategy_ids:
            if strategy_id in self.results_cache:
                metrics = self.results_cache[strategy_id]['metrics']
                if not common_metrics:
                    common_metrics = set(metrics.keys())
                else:
                    common_metrics &= set(metrics.keys())
        
        common_metrics = list(common_metrics)
        
        # 构建数据矩阵
        for strategy_id in strategy_ids:
            if strategy_id not in self.results_cache:
                logger.warning(f"策略 '{strategy_id}' 的评估结果不存在，已跳过")
                continue
                
            metrics = self.results_cache[strategy_id]['metrics']
            row = []
            
            for metric in common_metrics:
                if metric in metrics and metrics[metric] is not None:
                    if isinstance(metrics[metric], dict) and 'value' in metrics[metric]:
                        value = metrics[metric]['value']
                    else:
                        value = metrics[metric]
                        
                    # 处理列表或数组类型的指标
                    if isinstance(value, (list, np.ndarray)):
                        if len(value) > 0:
                            # 使用均值作为特征
                            row.append(float(np.mean(value)))
                        else:
                            row.append(0.0)
                    else:
                        row.append(float(value))
                else:
                    row.append(0.0)
            
            # 只有当我们有完整的一行数据时才添加
            if len(row) == len(common_metrics):
                data_matrix.append(row)
                metric_names = common_metrics
            
        if len(data_matrix) < 2:
            logger.warning("没有足够的数据执行多维分析")
            return {'status': 'error', 'message': '没有足够的数据执行多维分析'}
            
        # 转换为numpy数组
        data_matrix = np.array(data_matrix)
        
        # 2. 执行PCA分析
        try:
            pca = PCA(n_components=min(self.dimension_config['pca_components'], data_matrix.shape[0], data_matrix.shape[1]),
                    random_state=self.dimension_config['random_state'])
            pca_result = pca.fit_transform(data_matrix)
            
            # 保存PCA结果
            pca_data = {
                'components': pca_result.tolist(),
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'feature_importances': np.abs(pca.components_).mean(axis=0).tolist(),
                'feature_names': metric_names
            }
        except Exception as e:
            logger.error(f"PCA分析失败: {str(e)}")
            pca_data = None
            
        # 3. 执行t-SNE分析
        try:
            tsne = TSNE(n_components=self.dimension_config['tsne_components'],
                       perplexity=min(self.dimension_config['perplexity'], len(data_matrix) - 1),
                       random_state=self.dimension_config['random_state'])
            tsne_result = tsne.fit_transform(data_matrix)
            
            # 保存t-SNE结果
            tsne_data = {
                'components': tsne_result.tolist()
            }
        except Exception as e:
            logger.error(f"t-SNE分析失败: {str(e)}")
            tsne_data = None
            
        # 4. 执行K-means聚类
        try:
            n_clusters = min(self.dimension_config['kmeans_clusters'], len(data_matrix))
            kmeans = KMeans(n_clusters=n_clusters, 
                          random_state=self.dimension_config['random_state'])
            cluster_labels = kmeans.fit_predict(data_matrix)
            
            # 保存聚类结果
            cluster_data = {
                'labels': cluster_labels.tolist(),
                'centroids': kmeans.cluster_centers_.tolist(),
                'inertia': float(kmeans.inertia_)
            }
            
            # 更新策略聚类信息
            for i, strategy_id in enumerate([sid for sid in strategy_ids if sid in self.results_cache]):
                if i < len(cluster_labels):
                    self.strategy_clusters[strategy_id] = int(cluster_labels[i])
        except Exception as e:
            logger.error(f"K-means聚类失败: {str(e)}")
            cluster_data = None
            
        # 5. 计算相关性矩阵
        try:
            corr_matrix = np.corrcoef(data_matrix.T)
            corr_data = {
                'matrix': corr_matrix.tolist(),
                'feature_names': metric_names
            }
        except Exception as e:
            logger.error(f"相关性分析失败: {str(e)}")
            corr_data = None
            
        # 保存分析结果
        analysis_result = {
            'status': 'success',
            'strategy_count': len(data_matrix),
            'metric_count': len(metric_names),
            'metrics': metric_names,
            'strategies': [sid for sid in strategy_ids if sid in self.results_cache],
            'pca': pca_data,
            'tsne': tsne_data,
            'clusters': cluster_data,
            'correlation': corr_data,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        self.dimension_analysis_results = analysis_result
        
        logger.info(f"多维分析完成，分析了 {len(metric_names)} 个指标维度")
        
        return analysis_result
    
    def get_strategy_similarity(self, strategy_id1: str, strategy_id2: str) -> Dict[str, Any]:
        """计算两个策略的相似度
        
        Args:
            strategy_id1: 第一个策略ID
            strategy_id2: 第二个策略ID
            
        Returns:
            相似度分析结果
        """
        if not self.is_running:
            logger.warning("策略评估器未运行，无法计算相似度")
            return {'status': 'error', 'message': '策略评估器未运行'}
            
        if strategy_id1 not in self.results_cache or strategy_id2 not in self.results_cache:
            logger.warning("策略的评估结果不存在")
            return {'status': 'error', 'message': '策略的评估结果不存在'}
            
        # 找出两个策略共有的指标
        metrics1 = self.results_cache[strategy_id1]['metrics']
        metrics2 = self.results_cache[strategy_id2]['metrics']
        
        common_metrics = set(metrics1.keys()) & set(metrics2.keys())
        
        if not common_metrics:
            logger.warning("两个策略没有共同的指标")
            return {'status': 'error', 'message': '两个策略没有共同的指标'}
            
        # 提取指标值
        values1 = []
        values2 = []
        metric_names = []
        
        for metric in common_metrics:
            if metrics1[metric] is not None and metrics2[metric] is not None:
                # 提取值
                value1 = metrics1[metric]
                value2 = metrics2[metric]
                
                # 处理新的字典格式
                if isinstance(value1, dict) and 'value' in value1:
                    value1 = value1['value']
                if isinstance(value2, dict) and 'value' in value2:
                    value2 = value2['value']
                
                # 处理列表或数组类型的指标
                if isinstance(value1, (list, np.ndarray)) and isinstance(value2, (list, np.ndarray)):
                    # 使用均值
                    if len(value1) > 0 and len(value2) > 0:
                        value1 = float(np.mean(value1))
                        value2 = float(np.mean(value2))
                        values1.append(value1)
                        values2.append(value2)
                        metric_names.append(metric)
                elif not isinstance(value1, (list, np.ndarray)) and not isinstance(value2, (list, np.ndarray)):
                    # 标量值
                    values1.append(float(value1))
                    values2.append(float(value2))
                    metric_names.append(metric)
        
        if not values1 or not values2:
            logger.warning("无法比较的指标值")
            return {'status': 'error', 'message': '无法比较的指标值'}
            
        # 计算余弦相似度
        try:
            values1 = np.array(values1)
            values2 = np.array(values2)
            
            dot_product = np.dot(values1, values2)
            norm_product = np.linalg.norm(values1) * np.linalg.norm(values2)
            
            if norm_product == 0:
                cosine_similarity = 0
            else:
                cosine_similarity = dot_product / norm_product
                
            # 计算欧氏距离
            euclidean_distance = np.linalg.norm(values1 - values2)
            
            # 计算相关系数
            correlation = np.corrcoef(values1, values2)[0, 1]
            
            # 分类相似性 (是否在同一聚类中)
            same_cluster = False
            if strategy_id1 in self.strategy_clusters and strategy_id2 in self.strategy_clusters:
                same_cluster = (self.strategy_clusters[strategy_id1] == self.strategy_clusters[strategy_id2])
                
            similarity_result = {
                'status': 'success',
                'cosine_similarity': float(cosine_similarity),
                'euclidean_distance': float(euclidean_distance),
                'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
                'same_cluster': same_cluster,
                'metrics_compared': metric_names,
                'strategy1': strategy_id1,
                'strategy2': strategy_id2
            }
            
            return similarity_result
            
        except Exception as e:
            logger.error(f"计算相似度时出错: {str(e)}")
            return {'status': 'error', 'message': f"计算相似度时出错: {str(e)}"}
            
    def get_dimension_importance(self) -> Dict[str, Any]:
        """获取各维度(指标)的重要性排名"""
        if not self.dimension_analysis_results or 'pca' not in self.dimension_analysis_results:
            logger.warning("没有维度分析结果")
            return {'status': 'error', 'message': '没有维度分析结果'}
            
        pca_data = self.dimension_analysis_results['pca']
        if not pca_data or 'feature_importances' not in pca_data:
            logger.warning("PCA分析结果不完整")
            return {'status': 'error', 'message': 'PCA分析结果不完整'}
            
        # 获取特征重要性
        importances = pca_data['feature_importances']
        feature_names = pca_data['feature_names']
        
        # 按重要性排序
        importance_tuples = sorted(zip(feature_names, importances), 
                                 key=lambda x: x[1], reverse=True)
        
        # 构建结果
        importance_result = {
            'status': 'success',
            'importances': [
                {'metric': name, 'importance': float(importance)}
                for name, importance in importance_tuples
            ]
        }
        
        return importance_result
        
    def reset_quantum_config(self, config: Dict[str, Any]) -> bool:
        """重置量子增强配置
        
        Args:
            config: 量子增强配置
            
        Returns:
            是否成功
        """
        try:
            # 验证配置
            if 'enabled' in config:
                self.quantum_config['enabled'] = bool(config['enabled'])
                self.use_quantum_enhancement = self.quantum_config['enabled']
                
            if 'circuit_depth' in config:
                depth = int(config['circuit_depth'])
                if depth > 0:
                    self.quantum_config['circuit_depth'] = depth
                    
            if 'shots' in config:
                shots = int(config['shots'])
                if shots > 0:
                    self.quantum_config['shots'] = shots
                    
            if 'optimization_level' in config:
                level = int(config['optimization_level'])
                if 0 <= level <= 3:
                    self.quantum_config['optimization_level'] = level
                    
            if 'noise_model' in config:
                self.quantum_config['noise_model'] = config['noise_model']
                
            logger.info(f"量子增强配置已更新: {self.quantum_config}")
            return True
            
        except Exception as e:
            logger.error(f"更新量子增强配置失败: {str(e)}")
            return False
            
    def _calculate_quantum_robustness(self, equity_curve: pd.Series) -> float:
        """计算策略的量子噪声稳健性评分
        
        这个量子增强的指标评估策略在量子噪声下的稳定性
        
        Args:
            equity_curve: 权益曲线
            
        Returns:
            稳健性评分(0-1之间，越高越好)
        """
        try:
            from qiskit.providers.aer.noise import NoiseModel
            from qiskit import QuantumCircuit, Aer, transpile, assemble
            from qiskit.visualization import plot_histogram
            
            # 创建一个简单的噪声模型
            noise_model = NoiseModel()
            
            # 将权益曲线转换为量子态
            n_qubits = min(10, int(np.log2(len(equity_curve))) + 1)
            
            # 创建量子电路
            qc = QuantumCircuit(n_qubits, n_qubits)
            
            # 加载数据到量子态 (简化版)
            for i in range(n_qubits):
                # 基于权益曲线的部分统计特性设置旋转角度
                theta = np.pi * (equity_curve.pct_change().iloc[i*5:i*5+5].mean() + 1) / 2 
                qc.rx(theta, i)
                qc.ry(theta * 0.5, i)
            
            # 添加量子纠缠
            for i in range(n_qubits-1):
                qc.cx(i, i+1)
            
            # 添加QFT
            for i in range(n_qubits // 2):
                qc.swap(i, n_qubits - i - 1)
            for i in range(n_qubits):
                qc.h(i)
                for j in range(i + 1, n_qubits):
                    qc.cp(np.pi / float(2 ** (j - i)), i, j)
            
            # 测量
            qc.measure(range(n_qubits), range(n_qubits))
            
            # 在理想和有噪声的后端上分别执行
            backend = Aer.get_backend('qasm_simulator')
            
            # 无噪声执行
            job_no_noise = backend.run(
                transpile(qc, backend), 
                shots=1024
            )
            result_no_noise = job_no_noise.result()
            counts_no_noise = result_no_noise.get_counts()
            
            # 有噪声执行
            job_noise = backend.run(
                transpile(qc, backend),
                noise_model=noise_model,
                shots=1024
            )
            result_noise = job_noise.result()
            counts_noise = result_noise.get_counts()
            
            # 计算两种结果的相似度
            similarity = 0.0
            total_shots = 1024
            
            # 计算差异
            differences = 0
            for key in set(counts_no_noise.keys()) | set(counts_noise.keys()):
                count1 = counts_no_noise.get(key, 0)
                count2 = counts_noise.get(key, 0)
                differences += abs(count1 - count2)
            
            # 归一化差异
            normalized_diff = differences / (2 * total_shots)
            
            # 计算稳健性评分(1表示完全相同，0表示完全不同)
            robustness_score = 1.0 - normalized_diff
            
            return float(robustness_score)
            
        except Exception as e:
            logger.error(f"计算量子稳健性评分时出错: {str(e)}")
            # 默认返回中等稳健性
            return 0.5

