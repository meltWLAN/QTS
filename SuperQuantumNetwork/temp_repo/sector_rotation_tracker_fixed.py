
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import stats

class SectorRotationTracker:
    """板块轮动跟踪器"""
    
    def __init__(self, config_path=None):
        """
        初始化板块轮动跟踪器
        
        参数:
            config_path: str, 配置文件路径
        """
        # 设置日志
        self.logger = logging.getLogger("SectorRotationTracker")
        
        # 数据存储
        self.sector_data = {}       # 存储板块数据，格式为 {板块名: DataFrame}
        self.sector_performance = {}  # 存储板块表现指标
        
        # 轮动状态
        self.rotation_state = {
            'rotation_detected': False,        # 是否检测到轮动
            'rotation_strength': 0.0,          # 轮动强度 (0-1)
            'rotation_direction': 'none',      # 轮动方向 (value/growth/mixed)
            'rotation_acceleration': 0.0,      # 轮动加速度
            'leading_sectors': [],             # 领先板块
            'lagging_sectors': [],             # 滞后板块
            'avg_sector_correlation': 0.0,     # 平均板块相关性
            'sector_momentum': {},             # 板块动量
            'sector_volatility': {},           # 板块波动性
            'sector_correlation': {},          # 板块相关性矩阵
            'sector_rankings': {}              # 板块排名
        }
        
        # 配置参数
        self.config = {
            'short_term_window': 5,         # 短期窗口(天)
            'medium_term_window': 10,       # 中期窗口(天)
            'long_term_window': 20,         # 长期窗口(天)
            'momentum_window': 10,          # 动量计算窗口
            'volatility_window': 20,        # 波动率计算窗口
            'correlation_window': 30,       # 相关性计算窗口
            'rotation_detection_threshold': 0.2,  # 轮动检测阈值
            'rotation_strength_threshold': 0.3,    # 轮动强度阈值
            'performance_difference_threshold': 0.05,  # 表现差异阈值
            'momentum_threshold': 0.03,     # 动量阈值
            'correlation_threshold': 0.7,   # 相关性阈值
            'prediction_validation_mode': True, # 预测验证模式
            'short_term_prediction_window': 3,   # 短期预测窗口
            'medium_term_prediction_window': 5,  # 中期预测窗口
            'long_term_prediction_window': 10,   # 长期预测窗口
        }
        
        # 如果提供了配置路径，则从文件加载配置
        if config_path:
            self._load_config(config_path)
            
        # 预测存储
        self.rotation_predictions = {
            'short_term': {},    # 短期预测 (3天)
            'medium_term': {},   # 中期预测 (5天)
            'long_term': {},     # 长期预测 (10天)
            'prediction_history': [], # 历史预测记录
            'accuracy_metrics': {     # 准确性指标
                'hit_rate': 0.0,      # 命中率
                'direction_accuracy': 0.0,  # 方向准确性
                'magnitude_error': 0.0      # 幅度误差
            }
        }
        
        # 高级轮动特征
        self.advanced_features = {
            'sector_flow_network': {},  # 板块资金流网络
            'rotation_cycles': [],      # 轮动周期记录
            'typical_rotation_patterns': {},  # 典型轮动模式
            'current_pattern_match': None,    # 当前匹配的模式
            'flow_direction_strength': 0.0,   # 资金流向强度
            'rotation_stability': 0.0,        # 轮动稳定性
            'rotation_sustainability': 0.0     # 轮动可持续性
        }
        
        self.logger.info("板块轮动跟踪器初始化完成")
    
    def update_sector_data(self, sector_data):
        """
        更新板块数据
        
        参数:
            sector_data: dict, 板块数据，格式为 {板块名: DataFrame}
            
        返回:
            bool: 是否成功更新
        """
        if not sector_data:
            self.logger.warning("未提供板块数据")
            return False
            
        # 更新板块数据
        self.sector_data = sector_data
        
        # 计算板块表现
        self._calculate_sector_performance()
        
        # 计算板块排名
        self._calculate_sector_rankings()
        
        # 计算板块相关性
        self._calculate_sector_correlation()
        
        # 计算板块动量
        self._calculate_sector_momentum()
        
        # 计算板块波动性
        self._calculate_sector_volatility()
        
        # 检测板块轮动
        self._detect_rotation()
        
        # 分析资金流向
        self._analyze_fund_flows()
        
        # 识别轮动模式
        self._identify_rotation_patterns()
        
        # 更新轮动加速度
        self._update_rotation_acceleration()
        
        # 更新轮动预测
        self._update_rotation_predictions()
        
        # 验证历史预测
        if self.config['prediction_validation_mode']:
            self._validate_predictions()
        
        self.logger.info(f"更新了 {len(sector_data)} 个板块的数据")
        return True
    
    def analyze(self):
        """
        分析板块轮动状态
        
        返回:
            dict: 轮动分析结果
        """
        self.logger.info("开始分析板块轮动...")
        
        # 检查数据是否足够
        if not self.sector_data or not self.sector_performance:
            self.logger.warning("缺少足够的板块数据进行分析")
            return {
                'rotation_detected': False,
                'rotation_strength': 0.0,
                'rotation_direction': 'none',
                'leading_sectors': [],
                'lagging_sectors': [],
                'sector_divergence': 0.0,
                'recommendations': []
            }
        
        # 构建分析结果
        analysis_result = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'rotation_detected': self.rotation_state['rotation_detected'],
            'rotation_strength': self.rotation_state['rotation_strength'],
            'rotation_direction': self.rotation_state['rotation_direction'],
            'leading_sectors': self._format_sector_list(self.rotation_state['leading_sectors']),
            'lagging_sectors': self._format_sector_list(self.rotation_state['lagging_sectors']),
            'sector_divergence': self.rotation_state['rotation_strength'],
            'avg_sector_correlation': self.rotation_state.get('avg_sector_correlation', 0.0),
            'recommendations': self._generate_recommendations(),
            'rotation_acceleration': self.rotation_state.get('rotation_acceleration', 0.0),
            'rotation_predictions': self._generate_rotation_predictions(),
            'pattern_recognition': self._recognize_rotation_patterns(),
            'flow_analysis': self._analyze_sector_flows()
        }
        
        self.logger.info(f"板块轮动分析完成，轮动强度: {self.rotation_state['rotation_strength']:.2f}")
        return analysis_result
    
    def _format_sector_list(self, sector_list):
        """格式化板块列表，将元组格式转换为字典格式"""
        result = []
        for i, (sector_name, value) in enumerate(sector_list):
            result.append({
                'name': sector_name,
                'rank': i + 1,
                'short_term_return': value
            })
        return result
    
    def _detect_rotation(self):
        """检测板块轮动"""
        self.logger.info("检测板块轮动...")
        
        # 模拟实现，实际应基于性能差异和动量
        # 初始化领先和滞后板块
        leading_sectors = []
        lagging_sectors = []
        
        # 按性能排序 - 注意这里需要使用短期表现进行排序
        if not self.sector_performance:
            self.logger.warning("板块表现数据为空，无法检测轮动")
            self.rotation_state['rotation_detected'] = False
            self.rotation_state['rotation_strength'] = 0.0
            self.rotation_state['rotation_direction'] = 'none'
            return
            
        # 使用短期表现进行排序
        sorted_sectors = []
        for sector, perf in self.sector_performance.items():
            # 确保我们排序的是简单的值，而不是字典
            if isinstance(perf, dict) and 'short_term' in perf:
                sorted_sectors.append((sector, perf['short_term']))
            else:
                # 如果是简单值就直接使用
                sorted_sectors.append((sector, perf))
                
        # 排序
        sorted_sectors.sort(key=lambda x: x[1], reverse=True)
        
        # 区分领先和滞后板块
        if sorted_sectors:
            # 前1/3为领先板块
            leading_count = max(1, len(sorted_sectors) // 3)
            leading_sectors = sorted_sectors[:leading_count]
            
            # 后1/3为滞后板块
            lagging_sectors = sorted_sectors[-leading_count:]
            
        # 设置轮动状态
        self.rotation_state['leading_sectors'] = leading_sectors
        self.rotation_state['lagging_sectors'] = lagging_sectors
        
        # 计算轮动强度
        if leading_sectors and lagging_sectors:
            # 计算领先与滞后板块的性能差异
            leading_perf = sum(perf for _, perf in leading_sectors) / len(leading_sectors)
            lagging_perf = sum(perf for _, perf in lagging_sectors) / len(lagging_sectors)
            perf_diff = leading_perf - lagging_perf
            
            # 简单轮动强度计算
            rotation_strength = min(1.0, max(0.0, perf_diff * 10))  # 标准化到0-1
            self.rotation_state['rotation_strength'] = rotation_strength
            
            # 轮动方向判断
            self.rotation_state['rotation_direction'] = 'mixed'  # 默认为混合
            
            # 简单轮动检测
            if rotation_strength > self.config['rotation_detection_threshold']:
                self.rotation_state['rotation_detected'] = True
            else:
                self.rotation_state['rotation_detected'] = False
        else:
            self.rotation_state['rotation_strength'] = 0.0
            self.rotation_state['rotation_detected'] = False
            self.rotation_state['rotation_direction'] = 'none'
            
        self.logger.info(f"板块轮动检测完成: {self.rotation_state['rotation_detected']}, 强度: {self.rotation_state['rotation_strength']:.2f}")

    def _calculate_sector_momentum(self):
        """计算板块动量"""
        self.logger.info("计算板块动量...")
        
        # 初始化动量字典
        momentum_dict = {}
        
        for sector_name, sector_df in self.sector_data.items():
            try:
                # 确保数据足够
                if len(sector_df) < 20:  # 需要足够的数据点
                    continue
                
                # 获取最新的收盘价
                latest = sector_df['close'].iloc[-1]
                
                # 计算不同时间窗口的动量 (5, 10, 20天)
                momentum_5d = latest / sector_df['close'].iloc[-5] - 1 if len(sector_df) >= 5 else 0
                momentum_10d = latest / sector_df['close'].iloc[-10] - 1 if len(sector_df) >= 10 else 0
                momentum_20d = latest / sector_df['close'].iloc[-20] - 1 if len(sector_df) >= 20 else 0
                
                # 计算加权总动量 (更短期的权重更高)
                total_momentum = momentum_5d * 0.5 + momentum_10d * 0.3 + momentum_20d * 0.2
                
                # 存储动量值
                momentum_dict[sector_name] = total_momentum
                
            except Exception as e:
                self.logger.error(f"计算 {sector_name} 动量时出错: {str(e)}")
        
        # 更新轮动状态
        self.rotation_state['sector_momentum'] = momentum_dict
        
        self.logger.info("板块动量计算完成")

    def _calculate_sector_volatility(self):
        """计算板块波动性"""
        self.logger.info("计算板块波动性...")
        
        # 初始化波动性字典
        volatility_dict = {}
        
        for sector_name, sector_df in self.sector_data.items():
            try:
                # 确保数据足够
                window = min(20, len(sector_df) - 1)
                if window < 5:  # 需要至少5个数据点计算波动性
                    continue
                
                # 计算日收益率
                returns = sector_df['close'].pct_change().iloc[-window:]
                
                # 计算年化波动率 (假设252个交易日)
                annual_vol = returns.std() * np.sqrt(252)
                
                # 存储波动性值
                volatility_dict[sector_name] = annual_vol
                
            except Exception as e:
                self.logger.error(f"计算 {sector_name} 波动性时出错: {str(e)}")
        
        # 更新轮动状态
        self.rotation_state['sector_volatility'] = volatility_dict
        
        self.logger.info("板块波动性计算完成")

    def _calculate_sector_correlation(self):
        """计算板块相关性矩阵"""
        self.logger.info("计算板块相关性...")
        
        # 初始化相关性矩阵和板块收益率数据框
        correlation_matrix = {}
        returns_data = {}
        
        # 计算每个板块的收益率序列
        for sector_name, sector_df in self.sector_data.items():
            try:
                if len(sector_df) > 5:  # 至少需要5个数据点
                    # 计算收益率
                    returns = sector_df['close'].pct_change().iloc[1:]  # 跳过第一个NaN
                    returns_data[sector_name] = returns
            except Exception as e:
                self.logger.error(f"处理 {sector_name} 收益率时出错: {str(e)}")
        
        # 如果板块数量小于2，无法计算相关性
        if len(returns_data) < 2:
            self.logger.warning("板块数量不足，无法计算相关性")
            self.rotation_state['avg_sector_correlation'] = 0.0
            self.rotation_state['sector_correlation'] = {}
            return
        
        # 计算相关性矩阵
        for sector1 in returns_data:
            correlation_matrix[sector1] = {}
            for sector2 in returns_data:
                if sector1 == sector2:
                    correlation_matrix[sector1][sector2] = 1.0
                else:
                    # 计算两个板块间的相关系数
                    try:
                        # 确保两个序列有相同的索引
                        common_idx = returns_data[sector1].index.intersection(returns_data[sector2].index)
                        if len(common_idx) > 5:  # 至少需要5个共同点
                            s1 = returns_data[sector1].loc[common_idx]
                            s2 = returns_data[sector2].loc[common_idx]
                            correlation_matrix[sector1][sector2] = s1.corr(s2)
                        else:
                            correlation_matrix[sector1][sector2] = 0.0
                    except Exception as e:
                        self.logger.error(f"计算 {sector1} 和 {sector2} 相关性时出错: {str(e)}")
                        correlation_matrix[sector1][sector2] = 0.0
        
        # 计算平均相关性
        correlations = []
        for sector1 in correlation_matrix:
            for sector2 in correlation_matrix[sector1]:
                if sector1 != sector2:  # 排除自相关
                    correlations.append(correlation_matrix[sector1][sector2])
        
        avg_correlation = sum(correlations) / len(correlations) if correlations else 0.0
        
        # 更新轮动状态
        self.rotation_state['sector_correlation'] = correlation_matrix
        self.rotation_state['avg_sector_correlation'] = avg_correlation
        
        self.logger.info("板块相关性计算完成")

    def _analyze_fund_flows(self):
        """分析资金流向"""
        # 简单实现，仅做占位
        pass
    
    def _identify_rotation_patterns(self):
        """识别轮动模式"""
        # 简单实现，仅做占位
        pass
    
    def _update_rotation_acceleration(self):
        """更新轮动加速度"""
        # 简单实现，仅做占位
        self.rotation_state['rotation_acceleration'] = 0.0
    
    def _update_rotation_predictions(self):
        """更新轮动预测"""
        # 简单实现，仅做占位
        pass
    
    def _validate_predictions(self):
        """验证历史预测"""
        # 简单实现，仅做占位
        pass
    
    def _generate_rotation_predictions(self):
        """生成轮动预测"""
        # 简单实现，返回基本预测结构
        predictions = {
            'short_term': {
                'rotation_strength': self.rotation_state.get('rotation_strength', 0.0),
                'rotation_direction': self.rotation_state.get('rotation_direction', 'none'),
                'confidence': 0.7
            },
            'medium_term': {
                'rotation_strength': self.rotation_state.get('rotation_strength', 0.0) * 0.8,  # 简单假设中期轮动减弱20%
                'rotation_direction': self.rotation_state.get('rotation_direction', 'none'),
                'confidence': 0.5
            },
            'long_term': {
                'rotation_strength': self.rotation_state.get('rotation_strength', 0.0) * 0.6,  # 简单假设长期轮动减弱40%
                'rotation_direction': self.rotation_state.get('rotation_direction', 'none'),
                'confidence': 0.3
            }
        }
        
        return predictions
        
    def _recognize_rotation_patterns(self):
        """识别轮动模式"""
        # 简单实现，返回模式识别结果
        patterns = {
            'current_pattern': 'mixed',  # 可能的值: 'value_to_growth', 'growth_to_value', 'accelerating', 'decelerating', 'mixed'
            'pattern_confidence': 0.6,
            'pattern_duration': 5,  # 模式持续天数
            'typical_patterns': ['value_to_growth', 'growth_to_value'],  # 历史上识别出的典型模式
            'pattern_completion': 0.7  # 当前模式完成度 (0-1)
        }
        
        return patterns
        
    def _analyze_sector_flows(self):
        """分析板块资金流向"""
        # 简单实现，返回资金流向分析结果
        flows = {
            'net_inflow_sectors': [],
            'net_outflow_sectors': [],
            'flow_intensity': 0.5,  # 资金流动强度 (0-1)
            'flow_concentration': 0.3,  # 资金集中度 (0-1)
            'main_flow_direction': 'mixed'  # 主要资金流向
        }
        
        # 假设前3个领先板块是资金流入板块
        if self.rotation_state['leading_sectors'] and len(self.rotation_state['leading_sectors']) >= 3:
            flows['net_inflow_sectors'] = [sector_name for sector_name, _ in self.rotation_state['leading_sectors'][:3]]
            
        # 假设后3个滞后板块是资金流出板块
        if self.rotation_state['lagging_sectors'] and len(self.rotation_state['lagging_sectors']) >= 3:
            flows['net_outflow_sectors'] = [sector_name for sector_name, _ in self.rotation_state['lagging_sectors'][:3]]
            
        return flows
    
    def _generate_recommendations(self):
        """生成板块轮动建议"""
        recommendations = []
        
        # 如果未检测到明显轮动
        if not self.rotation_state['rotation_detected']:
            recommendations.append({
                'type': 'general',
                'content': "当前未检测到明显板块轮动，市场分化程度较低，可考虑配置指数型产品"
            })
            return recommendations
        
        # 基于轮动方向的建议
        direction = self.rotation_state['rotation_direction']
        if direction == 'growth':
            recommendations.append({
                'type': 'direction',
                'content': "市场风格偏向成长，高景气赛道和高弹性板块受青睐"
            })
        elif direction == 'value':
            recommendations.append({
                'type': 'direction',
                'content': "市场风格偏向价值，低估值、高股息板块表现较好"
            })
        elif direction == 'mixed':
            recommendations.append({
                'type': 'direction',
                'content': "市场风格混合，既有成长性板块也有价值型板块表现较好，建议均衡配置"
            })
        
        # 基于轮动强度的建议
        strength = self.rotation_state['rotation_strength']
        if strength > 0.8:
            recommendations.append({
                'type': 'strength',
                'content': "板块轮动强度极高，建议积极调整持仓，减持滞后板块，增持领先板块"
            })
        elif strength > 0.6:
            recommendations.append({
                'type': 'strength',
                'content': "板块轮动明显，可适当调整持仓结构，关注领先板块机会"
            })
        
        # 基于平均相关性的建议
        avg_correlation = self.rotation_state.get('avg_sector_correlation', 0.0)
        if avg_correlation < 0.3:
            recommendations.append({
                'type': 'correlation',
                'content': "板块间相关性较低，适合进行板块轮动策略，可增加配置领先板块"
            })
        elif avg_correlation > 0.7:
            recommendations.append({
                'type': 'correlation',
                'content': "板块间相关性较高，轮动持续性可能较差，建议保持谨慎"
            })
        
        # 具体板块建议
        if self.rotation_state['leading_sectors']:
            leading_names = [sector_name for sector_name, _ in self.rotation_state['leading_sectors'][:3]]
            recommendations.append({
                'type': 'sector',
                'content': f"关注领先板块: {', '.join(leading_names)}"
            })
        
        if self.rotation_state['lagging_sectors']:
            lagging_names = [sector_name for sector_name, _ in self.rotation_state['lagging_sectors'][:3]]
            recommendations.append({
                'type': 'sector',
                'content': f"减持滞后板块: {', '.join(lagging_names)}"
            })
        
        return recommendations
