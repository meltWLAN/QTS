#!/usr/bin/env python3
"""
超神量子共生系统 - 板块轮动跟踪器
专门分析A股市场板块轮动特性与预测板块轮动趋势
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Any, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

logger = logging.getLogger("SectorRotationTracker")

class SectorRotationTracker:
    """板块轮动跟踪器，分析A股市场的板块轮动特性"""
    
    def __init__(self, config=None):
        """
        初始化板块轮动跟踪器
        
        参数:
            config: 配置信息
        """
        self.logger = logging.getLogger("SectorRotationTracker")
        self.logger.info("初始化板块轮动跟踪器...")
        
        # 默认配置
        self.config = {
            'rotation_detection_threshold': 0.6,  # 轮动检测阈值
            'momentum_window': 20,  # 动量计算窗口
            'correlation_window': 60,  # 相关性计算窗口
            'short_term_window': 5,  # 短期表现窗口
            'medium_term_window': 20,  # 中期表现窗口
            'long_term_window': 60,  # 长期表现窗口
            'sector_weighting': {
                'momentum': 0.4,  # 动量权重
                'volatility': 0.2,  # 波动率权重
                'volume': 0.2,  # 成交量权重
                'breadth': 0.2  # 市场宽度权重
            }
        }
        
        # 更新配置
        if config:
            self.config.update(config)
            
        # 初始化板块数据存储
        self.sector_data = {}
        
        # 板块表现记录
        self.sector_performance = {}
        
        # 初始化轮动状态
        self.rotation_state = {
            'rotation_detected': False,
            'rotation_strength': 0.0,
            'rotation_direction': 'none',
            'leading_sectors': [],
            'lagging_sectors': [],
            'sector_momentum': {},
            'sector_correlation': {},
            'sector_rankings': {},
            'sector_breadth': {},
            'last_update': None
        }
        
        # 板块定义（A股主要板块）
        self.sector_definitions = {
            '银行': {'code': 'bank', 'description': '银行业，包括商业银行、农村信用社等金融机构'},
            '房地产': {'code': 'realestate', 'description': '房地产开发、物业管理等'},
            '医药生物': {'code': 'pharma', 'description': '医药制造、生物技术、医疗服务等'},
            '食品饮料': {'code': 'food', 'description': '食品加工、饮料制造等'},
            '电子': {'code': 'electronics', 'description': '电子元器件、半导体、面板等'},
            '计算机': {'code': 'computer', 'description': '软件开发、IT服务、互联网等'},
            '有色金属': {'code': 'metals', 'description': '有色金属采选、冶炼加工等'},
            '钢铁': {'code': 'steel', 'description': '钢铁冶炼、钢材加工等'},
            '军工': {'code': 'defense', 'description': '航空航天、军工装备等'},
            '汽车': {'code': 'auto', 'description': '汽车整车、零部件等'},
            '家电': {'code': 'appliance', 'description': '家用电器制造等'},
            '电力设备': {'code': 'power', 'description': '电力设备、新能源发电等'},
            '公用事业': {'code': 'utility', 'description': '电力、燃气、水务等公用事业'},
            '传媒': {'code': 'media', 'description': '广告、出版、影视、互联网媒体等'},
            '通信': {'code': 'telecom', 'description': '通信设备、运营服务等'},
            '建筑材料': {'code': 'building', 'description': '水泥、玻璃、装饰材料等'},
            '农林牧渔': {'code': 'agriculture', 'description': '种植业、畜牧业、渔业等'},
            '商贸零售': {'code': 'retail', 'description': '商业贸易、零售等'},
            '交通运输': {'code': 'transport', 'description': '铁路、公路、航空、航运等'},
            '纺织服装': {'code': 'textile', 'description': '纺织制造、服装服饰等'},
            '机械设备': {'code': 'machinery', 'description': '通用设备、专用设备制造等'},
            '化工': {'code': 'chemical', 'description': '基础化工、化学制品等'},
            '煤炭': {'code': 'coal', 'description': '煤炭开采、洗选等'},
            '石油石化': {'code': 'oil', 'description': '石油开采、炼化、石化等'},
            '非银金融': {'code': 'nonbank', 'description': '证券、保险、多元金融等'}
        }
        
        self.logger.info("板块轮动跟踪器初始化完成")
    
    def update_sector_data(self, data):
        """
        更新板块数据
        
        参数:
            data: 字典，键为板块名称，值为包含OHLCV等数据的DataFrame
            
        返回:
            bool: 是否成功更新
        """
        if not data:
            self.logger.warning("未提供板块数据")
            return False
            
        updated_sectors = []
        
        for sector_name, sector_df in data.items():
            # 确保数据格式正确
            if not isinstance(sector_df, pd.DataFrame):
                self.logger.warning(f"板块 {sector_name} 数据格式不正确")
                continue
                
            # 检查是否包含必要的列
            required_columns = ['date', 'close', 'volume']
            if not all(col in sector_df.columns for col in required_columns):
                self.logger.warning(f"板块 {sector_name} 数据缺少必要的列")
                continue
                
            # 确保日期列为索引
            if 'date' in sector_df.columns and not isinstance(sector_df.index, pd.DatetimeIndex):
                sector_df = sector_df.set_index('date')
                
            # 存储数据
            self.sector_data[sector_name] = sector_df
            updated_sectors.append(sector_name)
            
        # 如果有更新，计算板块表现
        if updated_sectors:
            self._calculate_sector_performance()
            self.rotation_state['last_update'] = datetime.now()
            
        self.logger.info(f"更新了 {len(updated_sectors)} 个板块的数据")
        return True
    
    def _calculate_sector_performance(self):
        """计算板块表现"""
        if not self.sector_data:
            return
            
        self.sector_performance = {}
        
        for sector_name, sector_df in self.sector_data.items():
            try:
                # 确保数据足够
                if len(sector_df) < self.config['long_term_window']:
                    self.logger.warning(f"板块 {sector_name} 数据不足，跳过表现计算")
                    continue
                
                # 计算收益率
                sector_df['return'] = sector_df['close'].pct_change()
                
                # 计算不同时间窗口的表现
                short_term = sector_df['close'].iloc[-1] / sector_df['close'].iloc[-self.config['short_term_window']] - 1
                medium_term = sector_df['close'].iloc[-1] / sector_df['close'].iloc[-self.config['medium_term_window']] - 1
                long_term = sector_df['close'].iloc[-1] / sector_df['close'].iloc[-self.config['long_term_window']] - 1
                
                # 计算动量
                momentum = self._calculate_momentum(sector_df)
                
                # 计算波动率
                volatility = sector_df['return'].std() * np.sqrt(252)
                
                # 计算成交量变化
                volume_change = sector_df['volume'].iloc[-5:].mean() / sector_df['volume'].iloc[-20:-5].mean() - 1
                
                # 存储表现指标
                self.sector_performance[sector_name] = {
                    'short_term': short_term,
                    'medium_term': medium_term,
                    'long_term': long_term,
                    'momentum': momentum,
                    'volatility': volatility,
                    'volume_change': volume_change
                }
                
            except Exception as e:
                self.logger.error(f"计算板块 {sector_name} 表现时出错: {str(e)}")
        
        # 计算板块排名
        self._calculate_sector_rankings()
        
        # 计算板块相关性
        self._calculate_sector_correlation()
        
        # 计算板块轮动强度
        self._calculate_rotation_strength()
        
        # 更新轮动状态
        self._update_rotation_state()
    
    def _calculate_momentum(self, sector_df):
        """计算动量"""
        # 使用移动平均比较计算动量
        momentum_window = self.config['momentum_window']
        
        if len(sector_df) < momentum_window * 2:
            return 0.0
            
        short_ma = sector_df['close'].rolling(momentum_window // 2).mean()
        long_ma = sector_df['close'].rolling(momentum_window).mean()
        
        momentum = short_ma.iloc[-1] / long_ma.iloc[-1] - 1
        return momentum
    
    def _calculate_sector_rankings(self):
        """计算板块排名"""
        if not self.sector_performance:
            return
            
        # 基于不同指标的排名
        rankings = {}
        
        # 短期表现排名
        short_term_data = {k: v['short_term'] for k, v in self.sector_performance.items()}
        short_term_ranking = sorted(short_term_data.items(), key=lambda x: x[1], reverse=True)
        rankings['short_term'] = {sector: rank + 1 for rank, (sector, _) in enumerate(short_term_ranking)}
        
        # 中期表现排名
        medium_term_data = {k: v['medium_term'] for k, v in self.sector_performance.items()}
        medium_term_ranking = sorted(medium_term_data.items(), key=lambda x: x[1], reverse=True)
        rankings['medium_term'] = {sector: rank + 1 for rank, (sector, _) in enumerate(medium_term_ranking)}
        
        # 长期表现排名
        long_term_data = {k: v['long_term'] for k, v in self.sector_performance.items()}
        long_term_ranking = sorted(long_term_data.items(), key=lambda x: x[1], reverse=True)
        rankings['long_term'] = {sector: rank + 1 for rank, (sector, _) in enumerate(long_term_ranking)}
        
        # 动量排名
        momentum_data = {k: v['momentum'] for k, v in self.sector_performance.items()}
        momentum_ranking = sorted(momentum_data.items(), key=lambda x: x[1], reverse=True)
        rankings['momentum'] = {sector: rank + 1 for rank, (sector, _) in enumerate(momentum_ranking)}
        
        # 综合排名 (使用加权平均)
        composite_scores = {}
        for sector in self.sector_performance:
            # 计算综合得分 (排名越小越好)
            composite_scores[sector] = (
                rankings['short_term'][sector] * 0.3 +
                rankings['medium_term'][sector] * 0.3 +
                rankings['momentum'][sector] * 0.4
            )
            
        composite_ranking = sorted(composite_scores.items(), key=lambda x: x[1])
        rankings['composite'] = {sector: rank + 1 for rank, (sector, _) in enumerate(composite_ranking)}
        
        self.rotation_state['sector_rankings'] = rankings
    
    def _calculate_sector_correlation(self):
        """计算板块相关性"""
        if len(self.sector_data) < 2:
            return
            
        # 提取所有板块的收益率数据
        returns_data = {}
        for sector, df in self.sector_data.items():
            if 'return' in df.columns and len(df) >= self.config['correlation_window']:
                returns_data[sector] = df['return'].iloc[-self.config['correlation_window']:]
        
        if len(returns_data) < 2:
            return
            
        # 构建收益率DataFrame
        returns_df = pd.DataFrame(returns_data)
        
        # 计算相关系数矩阵
        correlation_matrix = returns_df.corr()
        
        # 存储结果
        self.rotation_state['sector_correlation'] = correlation_matrix.to_dict()
        
        # 计算平均相关性
        correlations = []
        for i, sector1 in enumerate(correlation_matrix.index):
            for j, sector2 in enumerate(correlation_matrix.columns):
                if i < j:  # 避免重复计算
                    correlations.append(correlation_matrix.loc[sector1, sector2])
        
        if correlations:
            avg_correlation = sum(correlations) / len(correlations)
            self.rotation_state['average_correlation'] = avg_correlation
        else:
            self.rotation_state['average_correlation'] = 0.0
    
    def _calculate_rotation_strength(self):
        """计算板块轮动强度"""
        if not self.sector_performance:
            return
            
        # 计算板块表现的离散程度
        short_term_returns = [perf['short_term'] for perf in self.sector_performance.values()]
        medium_term_returns = [perf['medium_term'] for perf in self.sector_performance.values()]
        
        # 使用标准差衡量离散程度
        short_term_dispersion = np.std(short_term_returns) if len(short_term_returns) > 1 else 0
        medium_term_dispersion = np.std(medium_term_returns) if len(medium_term_returns) > 1 else 0
        
        # 计算轮动强度 (离散度越大，轮动越明显)
        rotation_strength = 0.6 * short_term_dispersion + 0.4 * medium_term_dispersion
        
        # 调整为0-1范围
        normalized_strength = min(1.0, rotation_strength * 10)
        
        self.rotation_state['rotation_strength'] = normalized_strength
        
        # 判断是否存在明显轮动
        threshold = self.config['rotation_detection_threshold']
        self.rotation_state['rotation_detected'] = normalized_strength > threshold
    
    def _update_rotation_state(self):
        """更新轮动状态"""
        if not self.rotation_state['rotation_detected'] or not self.sector_performance:
            self.rotation_state['rotation_direction'] = 'none'
            self.rotation_state['leading_sectors'] = []
            self.rotation_state['lagging_sectors'] = []
            return
            
        # 获取排名前5的板块作为领先板块
        composite_ranking = sorted(
            self.rotation_state['sector_rankings']['composite'].items(),
            key=lambda x: x[1]
        )
        
        leading_sectors = [
            {
                'name': sector,
                'rank': rank,
                'momentum': self.sector_performance[sector]['momentum'],
                'short_term_return': self.sector_performance[sector]['short_term'],
                'medium_term_return': self.sector_performance[sector]['medium_term']
            }
            for sector, rank in composite_ranking[:5]
        ]
        
        # 获取排名后5的板块作为滞后板块
        lagging_sectors = [
            {
                'name': sector,
                'rank': rank,
                'momentum': self.sector_performance[sector]['momentum'],
                'short_term_return': self.sector_performance[sector]['short_term'],
                'medium_term_return': self.sector_performance[sector]['medium_term']
            }
            for sector, rank in composite_ranking[-5:]
        ]
        
        self.rotation_state['leading_sectors'] = leading_sectors
        self.rotation_state['lagging_sectors'] = lagging_sectors
        
        # 确定轮动方向
        # 基于领先板块的特性判断轮动方向
        # 计算领先板块的平均特征
        avg_volatility = sum(self.sector_performance[s['name']]['volatility'] for s in leading_sectors) / len(leading_sectors)
        avg_volume_change = sum(self.sector_performance[s['name']]['volume_change'] for s in leading_sectors) / len(leading_sectors)
        
        # 判断方向
        if avg_volatility > 0.3 and avg_volume_change > 0.2:
            direction = 'growth'  # 成长型板块轮动
        elif avg_volatility < 0.2:
            direction = 'value'   # 价值型板块轮动
        else:
            direction = 'mixed'   # 混合型轮动
            
        self.rotation_state['rotation_direction'] = direction
    
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
            'leading_sectors': self.rotation_state['leading_sectors'],
            'lagging_sectors': self.rotation_state['lagging_sectors'],
            'sector_divergence': self.rotation_state['rotation_strength'],
            'average_correlation': self.rotation_state.get('average_correlation', 0.0),
            'recommendations': self._generate_recommendations()
        }
        
        self.logger.info(f"板块轮动分析完成，轮动强度: {self.rotation_state['rotation_strength']:.2f}")
        return analysis_result
    
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
        avg_correlation = self.rotation_state.get('average_correlation', 0.0)
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
            leading_names = [s['name'] for s in self.rotation_state['leading_sectors'][:3]]
            recommendations.append({
                'type': 'sector',
                'content': f"关注领先板块: {', '.join(leading_names)}"
            })
        
        if self.rotation_state['lagging_sectors']:
            lagging_names = [s['name'] for s in self.rotation_state['lagging_sectors'][:3]]
            recommendations.append({
                'type': 'sector',
                'content': f"减持滞后板块: {', '.join(lagging_names)}"
            })
        
        return recommendations
    
    def get_sector_performance(self, sector_name=None):
        """
        获取板块表现数据
        
        参数:
            sector_name: 板块名称，如不提供则返回所有板块
            
        返回:
            dict: 板块表现数据
        """
        if sector_name:
            return self.sector_performance.get(sector_name, {})
        else:
            return self.sector_performance
    
    def get_rotation_state(self):
        """
        获取当前轮动状态
        
        返回:
            dict: 轮动状态
        """
        return self.rotation_state
    
    def get_sector_rankings(self, ranking_type='composite'):
        """
        获取板块排名
        
        参数:
            ranking_type: 排名类型 ('short_term', 'medium_term', 'long_term', 'momentum', 'composite')
            
        返回:
            dict: 排名数据
        """
        if not self.rotation_state['sector_rankings']:
            return {}
            
        if ranking_type in self.rotation_state['sector_rankings']:
            return self.rotation_state['sector_rankings'][ranking_type]
        else:
            return self.rotation_state['sector_rankings'].get('composite', {})
    
    def get_sector_correlation(self, sector1=None, sector2=None):
        """
        获取板块相关性
        
        参数:
            sector1: 板块1名称
            sector2: 板块2名称
            
        返回:
            float/dict: 相关系数或相关系数矩阵
        """
        if not self.rotation_state['sector_correlation']:
            return 0.0 if sector1 and sector2 else {}
            
        if sector1 and sector2:
            # 返回两个板块间的相关系数
            return self.rotation_state['sector_correlation'].get(sector1, {}).get(sector2, 0.0)
        elif sector1:
            # 返回一个板块与所有其他板块的相关系数
            return self.rotation_state['sector_correlation'].get(sector1, {})
        else:
            # 返回完整的相关系数矩阵
            return self.rotation_state['sector_correlation']

    def analyze_rotation(self, sector_data: Optional[Dict] = None) -> Dict[str, Any]:
        """分析板块轮动"""
        try:
            self.sector_data = sector_data or {}
            
            results = {
                'leading_sectors': self._identify_leading_sectors(),
                'lagging_sectors': self._identify_lagging_sectors(),
                'rotation_direction': self._analyze_rotation_direction(),
                'rotation_strength': self._calculate_rotation_strength(),
                'sector_correlations': self._calculate_sector_correlations(),
                'sector_momentum': self._calculate_sector_momentum()
            }
            
            self.rotation_state = results['rotation_state']
            return results
            
        except Exception as e:
            self.logger.error(f"板块轮动分析失败: {str(e)}")
            return {}
            
    def _identify_leading_sectors(self) -> List[Dict[str, Any]]:
        """识别领先板块"""
        try:
            if not self.sector_data or 'sector_performance' not in self.sector_data:
                return []
                
            performances = self.sector_data['sector_performance']
            leading_sectors = []
            
            for sector, data in performances.items():
                if isinstance(data, dict):
                    # 计算综合得分
                    score = self._calculate_sector_score(data)
                    
                    leading_sectors.append({
                        'name': sector,
                        'relative_strength': data.get('relative_strength', 0),
                        'trend': data.get('trend', 0),
                        'heat': data.get('heat', 0),
                        'score': score
                    })
                    
            # 按得分排序，返回前5个
            leading_sectors.sort(key=lambda x: x['score'], reverse=True)
            return leading_sectors[:5]
            
        except Exception as e:
            self.logger.error(f"领先板块识别失败: {str(e)}")
            return []
            
    def _calculate_sector_score(self, data: Dict[str, float]) -> float:
        """计算板块综合得分"""
        try:
            # 权重设置
            weights = {
                'relative_strength': 0.4,  # 相对强度
                'trend': 0.3,              # 趋势
                'heat': 0.2,               # 热度
                'volume': 0.1              # 成交量
            }
            
            score = 0.0
            
            # 相对强度得分
            if 'relative_strength' in data:
                score += data['relative_strength'] * weights['relative_strength']
                
            # 趋势得分
            if 'trend' in data:
                score += data['trend'] * weights['trend']
                
            # 热度得分
            if 'heat' in data:
                score += data['heat'] * weights['heat']
                
            # 成交量得分
            if 'volume' in data:
                score += data['volume'] * weights['volume']
                
            return score
            
        except Exception as e:
            self.logger.error(f"板块得分计算失败: {str(e)}")
            return 0.0
            
    def _identify_lagging_sectors(self) -> List[Dict[str, Any]]:
        """识别落后板块"""
        try:
            if not self.sector_data or 'sector_performance' not in self.sector_data:
                return []
                
            performances = self.sector_data['sector_performance']
            lagging_sectors = []
            
            for sector, data in performances.items():
                if isinstance(data, dict):
                    # 计算综合得分
                    score = self._calculate_sector_score(data)
                    
                    lagging_sectors.append({
                        'name': sector,
                        'relative_strength': data.get('relative_strength', 0),
                        'trend': data.get('trend', 0),
                        'heat': data.get('heat', 0),
                        'score': score
                    })
                    
            # 按得分排序，返回后5个
            lagging_sectors.sort(key=lambda x: x['score'])
            return lagging_sectors[:5]
            
        except Exception as e:
            self.logger.error(f"落后板块识别失败: {str(e)}")
            return []
            
    def _analyze_rotation_direction(self) -> str:
        """分析轮动方向"""
        try:
            if not self.sector_data or 'sector_flows' not in self.sector_data:
                return 'unknown'
                
            flows = self.sector_data['sector_flows']
            
            # 分析资金流向
            defensive_flow = 0
            cyclical_flow = 0
            growth_flow = 0
            
            sector_types = {
                'defensive': ['消费', '医药', '公用事业'],
                'cyclical': ['金融', '地产', '周期'],
                'growth': ['科技', '新能源', '传媒']
            }
            
            for sector, flow in flows.items():
                if any(key in sector for key in sector_types['defensive']):
                    defensive_flow += flow
                elif any(key in sector for key in sector_types['cyclical']):
                    cyclical_flow += flow
                elif any(key in sector for key in sector_types['growth']):
                    growth_flow += flow
                    
            # 判断轮动方向
            flows = [defensive_flow, cyclical_flow, growth_flow]
            max_flow = max(flows)
            min_flow = min(flows)
            
            if max_flow == defensive_flow:
                return 'defensive'  # 防御性
            elif max_flow == cyclical_flow:
                return 'cyclical'   # 周期性
            elif max_flow == growth_flow:
                return 'growth'     # 成长性
            else:
                return 'mixed'      # 混合型
                
        except Exception as e:
            self.logger.error(f"轮动方向分析失败: {str(e)}")
            return 'unknown'
            
    def _calculate_rotation_strength(self) -> float:
        """计算轮动强度"""
        try:
            if not self.sector_data:
                return 0.0
                
            strength_factors = []
            
            # 1. 板块分化度
            if 'sector_performance' in self.sector_data:
                performances = self.sector_data['sector_performance']
                if performances:
                    returns = [data.get('return', 0) for data in performances.values() if isinstance(data, dict)]
                    if returns:
                        # 使用收益率的标准差作为分化度指标
                        divergence = np.std(returns)
                        strength_factors.append(min(1.0, divergence * 5))  # 标准化到[0,1]
                        
            # 2. 资金流动强度
            if 'sector_flows' in self.sector_data:
                flows = self.sector_data['sector_flows']
                if flows:
                    # 使用资金流绝对值的总和
                    total_flow = sum(abs(flow) for flow in flows.values())
                    flow_strength = min(1.0, total_flow / 1000)  # 假设1000亿为基准
                    strength_factors.append(flow_strength)
                    
            # 3. 换手率变化
            if 'turnover_changes' in self.sector_data:
                changes = self.sector_data['turnover_changes']
                if changes:
                    # 使用换手率变化的平均值
                    avg_change = np.mean([abs(change) for change in changes.values()])
                    turnover_strength = min(1.0, avg_change * 2)
                    strength_factors.append(turnover_strength)
                    
            # 计算综合强度
            return np.mean(strength_factors) if strength_factors else 0.0
            
        except Exception as e:
            self.logger.error(f"轮动强度计算失败: {str(e)}")
            return 0.0
            
    def _calculate_sector_correlations(self) -> Dict[str, Dict[str, float]]:
        """计算板块相关性"""
        try:
            if not self.sector_data or 'sector_returns' not in self.sector_data:
                return {}
                
            returns = self.sector_data['sector_returns']
            if not returns or not isinstance(returns, dict):
                return {}
                
            # 创建相关性矩阵
            sectors = list(returns.keys())
            correlations = {}
            
            for sector1 in sectors:
                correlations[sector1] = {}
                for sector2 in sectors:
                    if sector1 == sector2:
                        correlations[sector1][sector2] = 1.0
                    else:
                        # 计算相关系数
                        returns1 = returns[sector1]
                        returns2 = returns[sector2]
                        if len(returns1) == len(returns2):
                            corr = np.corrcoef(returns1, returns2)[0,1]
                            correlations[sector1][sector2] = corr
                        else:
                            correlations[sector1][sector2] = 0.0
                            
            return correlations
            
        except Exception as e:
            self.logger.error(f"板块相关性计算失败: {str(e)}")
            return {}
            
    def _calculate_sector_momentum(self) -> Dict[str, float]:
        """计算板块动量"""
        try:
            if not self.sector_data or 'sector_performance' not in self.sector_data:
                return {}
                
            performances = self.sector_data['sector_performance']
            momentum = {}
            
            for sector, data in performances.items():
                if isinstance(data, dict):
                    # 计算动量得分
                    score = 0.0
                    
                    # 1. 短期动量 (5日)
                    if 'return_5d' in data:
                        score += data['return_5d'] * 0.3
                        
                    # 2. 中期动量 (20日)
                    if 'return_20d' in data:
                        score += data['return_20d'] * 0.5
                        
                    # 3. 长期动量 (60日)
                    if 'return_60d' in data:
                        score += data['return_60d'] * 0.2
                        
                    momentum[sector] = score
                    
            return momentum
            
        except Exception as e:
            self.logger.error(f"板块动量计算失败: {str(e)}")
            return {}
            
    def calibrate(self):
        """校准分析参数"""
        self.logger.info("校准板块轮动跟踪器")
        # 实现校准逻辑
        
    def adjust_sensitivity(self):
        """调整灵敏度"""
        self.logger.info("调整板块轮动跟踪器灵敏度")
        # 实现灵敏度调整逻辑

# 如果直接运行此脚本，则执行示例
if __name__ == "__main__":
    # 创建板块轮动跟踪器
    tracker = SectorRotationTracker()
    
    # 生成一些模拟数据
    sample_data = {}
    
    # 示例日期范围
    dates = pd.date_range(end=datetime.now(), periods=100)
    
    # 为每个板块生成模拟数据
    for sector in ['银行', '房地产', '医药生物', '食品饮料', '电子']:
        # 生成随机价格序列
        np.random.seed(hash(sector) % 100)  # 使不同板块有不同的随机种子
        
        # 基础价格
        base_price = 100
        
        # 生成随机走势
        changes = np.random.normal(0.0005, 0.015, len(dates))
        
        # 添加板块特性
        if sector == '医药生物':
            # 医药板块近期强势
            changes[-20:] += 0.005
        elif sector == '银行':
            # 银行板块近期弱势
            changes[-20:] -= 0.003
        
        # 计算价格序列
        prices = base_price * np.cumprod(1 + changes)
        
        # 生成成交量数据
        volumes = np.random.normal(1000000, 200000, len(dates))
        volumes = np.abs(volumes)  # 确保成交量为正
        
        # 创建DataFrame
        sector_df = pd.DataFrame({
            'date': dates,
            'open': prices * (1 - np.random.uniform(0, 0.01, len(dates))),
            'high': prices * (1 + np.random.uniform(0, 0.02, len(dates))),
            'low': prices * (1 - np.random.uniform(0, 0.02, len(dates))),
            'close': prices,
            'volume': volumes
        })
        
        sample_data[sector] = sector_df
    
    # 更新板块数据
    tracker.update_sector_data(sample_data)
    
    # 执行分析
    result = tracker.analyze()
    
    # 输出分析结果
    print("\n" + "="*60)
    print("超神量子共生系统 - 板块轮动跟踪器")
    print("="*60 + "\n")
    
    print(f"轮动检测: {'是' if result['rotation_detected'] else '否'}")
    print(f"轮动强度: {result['rotation_strength']:.2f}")
    print(f"轮动方向: {result['rotation_direction']}")
    print(f"平均相关性: {result.get('average_correlation', 0.0):.2f}")
    
    print("\n领先板块:")
    for sector in result['leading_sectors']:
        print(f"- {sector['name']}: 排名 {sector['rank']}, 短期收益率 {sector['short_term_return']:.2%}")
    
    print("\n滞后板块:")
    for sector in result['lagging_sectors']:
        print(f"- {sector['name']}: 排名 {sector['rank']}, 短期收益率 {sector['short_term_return']:.2%}")
    
    print("\n建议:")
    for rec in result['recommendations']:
        print(f"- [{rec['type']}] {rec['content']}")
    
    print("\n系统已准备就绪，可以进行中国A股板块轮动分析。")
    print("="*60)