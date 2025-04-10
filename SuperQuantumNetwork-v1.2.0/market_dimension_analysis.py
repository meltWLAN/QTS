#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - 市场维度分析模块
增强功能：板块关联性分析、热点轮动分析、资金流向分析
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
import json
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict, Counter

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MarketDimensionAnalysis")

class MarketDimensionAnalyzer:
    """市场多维度分析器"""
    
    def __init__(self, data_source=None):
        """初始化分析器
        
        Args:
            data_source: 数据源对象，需要具备获取市场数据的能力
        """
        self.logger = logging.getLogger("MarketDimensionAnalyzer")
        self.data_source = data_source
        
        # 创建缓存目录
        self.cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "market_analysis")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 热点板块历史记录
        self.hot_sectors_history = []
        
        # 行业板块关联性矩阵
        self.sector_correlation_matrix = {}
        
        # 资金流向历史
        self.fund_flow_history = {}
        
        # 最新分析结果
        self.latest_analysis = {
            'hot_sectors': [],
            'sector_rotations': [],
            'market_structure': {},
            'capital_flow': {},
            'timestamp': datetime.now().isoformat()
        }
    
    def analyze_sector_correlation(self, period='30d', top_sectors=10, threshold=0.6):
        """分析行业板块关联性
        
        Args:
            period: 分析周期，如'7d', '30d', '90d'
            top_sectors: 分析的主要板块数量
            threshold: 关联性阈值
            
        Returns:
            dict: 板块关联性分析结果
        """
        self.logger.info(f"开始分析行业板块关联性，周期={period}")
        
        if self.data_source is None:
            self.logger.error("未提供数据源，无法进行分析")
            return {}
        
        # 计算天数
        days = int(period[:-1]) if period[:-1].isdigit() else 30
        
        try:
            # 获取主要行业板块数据
            sectors = self._get_major_sectors(top_sectors)
            
            # 获取各行业指数历史数据
            sector_data = {}
            for sector_name, sector_code in sectors.items():
                hist_data = self.data_source.get_index_data(sector_code, 
                                                          start_date=(datetime.now() - timedelta(days=days)).strftime('%Y%m%d'),
                                                          end_date=datetime.now().strftime('%Y%m%d'))
                if hist_data and 'history' in hist_data and 'prices' in hist_data['history']:
                    sector_data[sector_name] = hist_data['history']['prices']
            
            # 计算关联性矩阵
            correlation_matrix = {}
            sectors_list = list(sector_data.keys())
            
            for i, sector1 in enumerate(sectors_list):
                correlation_matrix[sector1] = {}
                prices1 = sector_data[sector1]
                
                for sector2 in sectors_list[i:]:
                    prices2 = sector_data[sector2]
                    
                    # 确保数据长度相等
                    min_len = min(len(prices1), len(prices2))
                    if min_len < 5:  # 至少需要5个数据点
                        correlation = 0
                    else:
                        correlation, _ = stats.pearsonr(prices1[-min_len:], prices2[-min_len:])
                        correlation = round(correlation, 3)
                    
                    correlation_matrix[sector1][sector2] = correlation
                    if sector1 != sector2:
                        if sector2 not in correlation_matrix:
                            correlation_matrix[sector2] = {}
                        correlation_matrix[sector2][sector1] = correlation
            
            # 保存关联性矩阵
            self.sector_correlation_matrix = correlation_matrix
            
            # 提取关键关联性
            key_correlations = []
            for sector1 in sectors_list:
                for sector2 in sectors_list:
                    if sector1 != sector2:
                        corr = correlation_matrix[sector1][sector2]
                        if abs(corr) >= threshold:
                            key_correlations.append({
                                'sector1': sector1,
                                'sector2': sector2,
                                'correlation': corr,
                                'type': '正相关' if corr > 0 else '负相关',
                                'strength': abs(corr)
                            })
            
            # 按相关性强度排序
            key_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            # 构建结果
            result = {
                'period': period,
                'analyzed_sectors': sectors_list,
                'correlation_matrix': correlation_matrix,
                'key_correlations': key_correlations[:20],  # 最多返回20个关键关联
                'timestamp': datetime.now().isoformat()
            }
            
            # 更新最新分析结果
            self.latest_analysis['sector_correlations'] = result
            
            self.logger.info(f"行业板块关联性分析完成，发现 {len(key_correlations)} 个关键关联")
            return result
            
        except Exception as e:
            self.logger.error(f"分析行业板块关联性失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}
    
    def analyze_sector_rotation(self, period='90d', window=5, threshold=0.05):
        """分析板块轮动
        
        Args:
            period: 分析周期，如'30d', '90d', '180d'
            window: 平滑窗口大小
            threshold: 轮动识别阈值
            
        Returns:
            dict: 板块轮动分析结果
        """
        self.logger.info(f"开始分析板块轮动，周期={period}")
        
        if self.data_source is None:
            self.logger.error("未提供数据源，无法进行分析")
            return {}
        
        # 计算天数
        days = int(period[:-1]) if period[:-1].isdigit() else 90
        
        try:
            # 获取主要行业板块数据
            sectors = self._get_major_sectors(20)  # 分析前20个主要板块
            
            # 获取各行业指数历史数据
            sector_data = {}
            for sector_name, sector_code in sectors.items():
                hist_data = self.data_source.get_index_data(sector_code, 
                                                          start_date=(datetime.now() - timedelta(days=days)).strftime('%Y%m%d'),
                                                          end_date=datetime.now().strftime('%Y%m%d'))
                if hist_data and 'history' in hist_data:
                    sector_data[sector_name] = {
                        'prices': hist_data['history'].get('prices', []),
                        'dates': hist_data['history'].get('dates', [])
                    }
            
            # 计算各板块的周收益率变化
            rotation_data = {}
            dates = []
            
            for sector_name, data in sector_data.items():
                prices = data['prices']
                if len(prices) < window:
                    continue
                
                # 计算每日收益率
                daily_returns = [0]
                for i in range(1, len(prices)):
                    daily_return = (prices[i] - prices[i-1]) / prices[i-1]
                    daily_returns.append(daily_return)
                
                # 使用移动窗口计算平滑收益率
                smoothed_returns = []
                for i in range(len(daily_returns) - window + 1):
                    smoothed_return = sum(daily_returns[i:i+window]) / window
                    smoothed_returns.append(smoothed_return)
                
                # 保存数据
                rotation_data[sector_name] = smoothed_returns
                
                # 保存日期（使用第一个有效板块的日期）
                if not dates and 'dates' in data and len(data['dates']) >= window:
                    dates = data['dates'][window-1:]
            
            # 识别每个时间窗口的领先板块
            leading_sectors = []
            for i in range(len(dates)):
                # 计算当前窗口各板块收益率
                period_returns = {}
                for sector_name, returns in rotation_data.items():
                    if i < len(returns):
                        period_returns[sector_name] = returns[i]
                
                # 如果有数据，选出收益率最高的板块
                if period_returns:
                    leading_sector = max(period_returns.items(), key=lambda x: x[1])
                    if leading_sector[1] > threshold:  # 仅保留收益率超过阈值的板块
                        leading_sectors.append({
                            'date': dates[i],
                            'sector': leading_sector[0],
                            'return': leading_sector[1]
                        })
            
            # 分析板块轮动序列
            rotations = []
            current_leader = None
            leader_start_date = None
            
            for entry in leading_sectors:
                if current_leader != entry['sector']:
                    # 保存之前的领先板块
                    if current_leader:
                        rotations.append({
                            'sector': current_leader,
                            'start_date': leader_start_date,
                            'end_date': entry['date'],
                            'duration': _days_between(leader_start_date, entry['date'])
                        })
                    
                    # 更新当前领先板块
                    current_leader = entry['sector']
                    leader_start_date = entry['date']
            
            # 添加最后一个领先板块
            if current_leader and leader_start_date:
                rotations.append({
                    'sector': current_leader,
                    'start_date': leader_start_date,
                    'end_date': dates[-1] if dates else None,
                    'duration': _days_between(leader_start_date, dates[-1]) if dates else 0
                })
            
            # 预测下一个可能轮动的板块
            next_rotation = self._predict_next_rotation(rotations, self.sector_correlation_matrix)
            
            # 构建结果
            result = {
                'period': period,
                'rotations': rotations,
                'current_leading_sector': rotations[-1]['sector'] if rotations else None,
                'next_potential_rotations': next_rotation,
                'timestamp': datetime.now().isoformat()
            }
            
            # 更新最新分析结果
            self.latest_analysis['sector_rotations'] = result
            
            self.logger.info(f"板块轮动分析完成，发现 {len(rotations)} 次轮动")
            return result
            
        except Exception as e:
            self.logger.error(f"分析板块轮动失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}
    
    def analyze_capital_flow(self, period='30d', top_sectors=10):
        """分析资金流向
        
        Args:
            period: 分析周期，如'7d', '30d', '90d'
            top_sectors: 分析的主要板块数量
            
        Returns:
            dict: 资金流向分析结果
        """
        self.logger.info(f"开始分析资金流向，周期={period}")
        
        if self.data_source is None:
            self.logger.error("未提供数据源，无法进行分析")
            return {}
        
        # 计算天数
        days = int(period[:-1]) if period[:-1].isdigit() else 30
        
        try:
            # 获取主要行业板块数据
            sectors = self._get_major_sectors(top_sectors)
            
            # 获取各行业指数历史数据和成交量
            sector_data = {}
            for sector_name, sector_code in sectors.items():
                hist_data = self.data_source.get_index_data(sector_code, 
                                                          start_date=(datetime.now() - timedelta(days=days)).strftime('%Y%m%d'),
                                                          end_date=datetime.now().strftime('%Y%m%d'))
                if hist_data and 'history' in hist_data:
                    sector_data[sector_name] = {
                        'prices': hist_data['history'].get('prices', []),
                        'volumes': hist_data['history'].get('volumes', []),
                        'dates': hist_data['history'].get('dates', [])
                    }
            
            # 计算资金净流入
            capital_flows = {}
            for sector_name, data in sector_data.items():
                prices = data['prices']
                volumes = data['volumes']
                dates = data['dates']
                
                if not prices or not volumes or len(prices) != len(volumes):
                    continue
                
                # 计算每日资金流向
                daily_flows = []
                for i in range(1, len(prices)):
                    price_change = prices[i] - prices[i-1]
                    flow = price_change * volumes[i]  # 简化计算，实际应考虑更多因素
                    daily_flows.append({
                        'date': dates[i] if i < len(dates) else None,
                        'flow': flow
                    })
                
                # 计算周期内总流向
                total_flow = sum(item['flow'] for item in daily_flows)
                
                # 计算近期流向（最后5天）
                recent_flow = sum(item['flow'] for item in daily_flows[-5:]) if len(daily_flows) >= 5 else 0
                
                # 保存资金流向数据
                capital_flows[sector_name] = {
                    'total_flow': total_flow,
                    'recent_flow': recent_flow,
                    'daily_flows': daily_flows
                }
            
            # 按总流向排序
            ranked_flows = sorted(capital_flows.items(), key=lambda x: x[1]['total_flow'], reverse=True)
            
            # 构建结果
            result = {
                'period': period,
                'top_inflow_sectors': [{'sector': sector, 'flow': data['total_flow']} 
                                      for sector, data in ranked_flows[:5]],
                'top_outflow_sectors': [{'sector': sector, 'flow': data['total_flow']} 
                                       for sector, data in ranked_flows[-5:]],
                'recent_momentum_change': [{'sector': sector, 'flow_change': data['recent_flow'] - (data['total_flow']/days)*5} 
                                          for sector, data in capital_flows.items()],
                'timestamp': datetime.now().isoformat()
            }
            
            # 按资金动量变化排序
            result['recent_momentum_change'].sort(key=lambda x: x['flow_change'], reverse=True)
            
            # 更新最新分析结果
            self.latest_analysis['capital_flow'] = result
            
            self.logger.info(f"资金流向分析完成，分析了 {len(capital_flows)} 个板块")
            return result
            
        except Exception as e:
            self.logger.error(f"分析资金流向失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}
    
    def analyze_market_structure(self, top_stocks=100, weight_market_cap=True):
        """分析市场结构
        
        Args:
            top_stocks: 分析的头部股票数量
            weight_market_cap: 是否按市值加权
            
        Returns:
            dict: 市场结构分析结果
        """
        self.logger.info(f"开始分析市场结构，使用头部 {top_stocks} 只股票")
        
        if self.data_source is None:
            self.logger.error("未提供数据源，无法进行分析")
            return {}
        
        try:
            # 获取股票列表并按市值排序
            all_stocks = self.data_source.get_stock_list()
            
            # 获取市值数据并排序
            stocks_with_cap = []
            for stock in all_stocks:
                try:
                    stock_data = self.data_source.get_stock_data(stock['ts_code'])
                    if 'market_cap' in stock_data:
                        stock['market_cap'] = stock_data['market_cap']
                        stocks_with_cap.append(stock)
                except:
                    pass
            
            # 按市值排序
            stocks_with_cap.sort(key=lambda x: x.get('market_cap', 0), reverse=True)
            
            # 截取头部股票
            top_stocks_list = stocks_with_cap[:top_stocks]
            
            # 按行业分组
            industry_groups = defaultdict(list)
            for stock in top_stocks_list:
                industry = stock.get('industry', '其他')
                industry_groups[industry].append(stock)
            
            # 计算行业权重
            industry_weights = {}
            total_cap = sum(stock.get('market_cap', 0) for stock in top_stocks_list)
            
            for industry, stocks in industry_groups.items():
                if weight_market_cap:
                    # 按市值加权
                    industry_cap = sum(stock.get('market_cap', 0) for stock in stocks)
                    weight = industry_cap / total_cap if total_cap > 0 else 0
                else:
                    # 按数量占比
                    weight = len(stocks) / len(top_stocks_list)
                
                industry_weights[industry] = {
                    'weight': weight,
                    'stock_count': len(stocks),
                    'market_cap': sum(stock.get('market_cap', 0) for stock in stocks) if weight_market_cap else 0,
                    'stocks': [{'code': stock['ts_code'], 'name': stock['name']} for stock in stocks[:5]]  # 只取前5个示例
                }
            
            # 按权重排序
            sorted_weights = sorted(industry_weights.items(), key=lambda x: x[1]['weight'], reverse=True)
            
            # 计算集中度
            concentration = {
                'top_3_industries': sum(item[1]['weight'] for item in sorted_weights[:3]),
                'top_5_industries': sum(item[1]['weight'] for item in sorted_weights[:5]),
                'herfindahl_index': sum(weight**2 for industry, data in industry_weights.items() for weight in [data['weight']])
            }
            
            # 构建结果
            result = {
                'top_stocks_count': len(top_stocks_list),
                'industry_weights': dict(sorted_weights),
                'concentration': concentration,
                'weight_method': 'market_cap' if weight_market_cap else 'count',
                'timestamp': datetime.now().isoformat()
            }
            
            # 更新最新分析结果
            self.latest_analysis['market_structure'] = result
            
            self.logger.info(f"市场结构分析完成，分析了 {len(industry_weights)} 个行业")
            return result
            
        except Exception as e:
            self.logger.error(f"分析市场结构失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}
    
    def analyze_hot_sectors(self, period='7d', top_count=10):
        """分析热点板块
        
        Args:
            period: 分析周期，如'1d', '7d', '30d'
            top_count: 返回的热点板块数量
            
        Returns:
            dict: 热点板块分析结果
        """
        self.logger.info(f"开始分析热点板块，周期={period}")
        
        if self.data_source is None:
            self.logger.error("未提供数据源，无法进行分析")
            return {}
        
        # 计算天数
        days = int(period[:-1]) if period[:-1].isdigit() else 7
        
        try:
            # 获取所有行业板块数据
            all_sectors = self._get_all_sectors()
            
            # 获取各板块收益率
            sector_returns = {}
            for sector_name, sector_code in all_sectors.items():
                hist_data = self.data_source.get_index_data(sector_code, 
                                                          start_date=(datetime.now() - timedelta(days=days)).strftime('%Y%m%d'),
                                                          end_date=datetime.now().strftime('%Y%m%d'))
                if hist_data and 'history' in hist_data and 'prices' in hist_data['history']:
                    prices = hist_data['history']['prices']
                    if len(prices) >= 2:
                        # 计算周期收益率
                        period_return = (prices[-1] - prices[0]) / prices[0]
                        sector_returns[sector_name] = {
                            'return': period_return,
                            'code': sector_code,
                            'current_price': prices[-1],
                            'volume': hist_data['history'].get('volumes', [])[-1] if 'volumes' in hist_data['history'] and hist_data['history']['volumes'] else 0
                        }
            
            # 按收益率排序
            ranked_sectors = sorted(sector_returns.items(), key=lambda x: x[1]['return'], reverse=True)
            
            # 提取热点板块
            hot_sectors = [{
                'name': sector_name,
                'code': data['code'],
                'return': data['return'],
                'price': data['current_price'],
                'volume': data['volume']
            } for sector_name, data in ranked_sectors[:top_count]]
            
            # 计算热点变化
            if self.hot_sectors_history:
                previous_hot = {item['name']: idx for idx, item in enumerate(self.hot_sectors_history[-1]['sectors'])}
                
                for sector in hot_sectors:
                    if sector['name'] in previous_hot:
                        prev_rank = previous_hot[sector['name']]
                        sector['rank_change'] = prev_rank - hot_sectors.index(sector)
                    else:
                        sector['rank_change'] = None  # 新进入热点的板块
            
            # 保存热点历史
            hot_record = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'period': period,
                'sectors': hot_sectors
            }
            self.hot_sectors_history.append(hot_record)
            
            # 限制历史记录长度
            if len(self.hot_sectors_history) > 30:
                self.hot_sectors_history = self.hot_sectors_history[-30:]
            
            # 构建结果
            result = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'period': period,
                'hot_sectors': hot_sectors,
                'timestamp': datetime.now().isoformat()
            }
            
            # 更新最新分析结果
            self.latest_analysis['hot_sectors'] = result
            
            self.logger.info(f"热点板块分析完成，找到 {len(hot_sectors)} 个热点板块")
            return result
            
        except Exception as e:
            self.logger.error(f"分析热点板块失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}
    
    def run_full_analysis(self):
        """运行完整的市场多维度分析"""
        self.logger.info("开始运行完整市场多维度分析...")
        
        # 分析热点板块
        self.analyze_hot_sectors(period='7d', top_count=15)
        
        # 分析板块轮动
        self.analyze_sector_rotation(period='90d')
        
        # 分析行业板块关联性
        self.analyze_sector_correlation(period='30d')
        
        # 分析资金流向
        self.analyze_capital_flow(period='30d')
        
        # 分析市场结构
        self.analyze_market_structure(top_stocks=300)
        
        # 更新时间戳
        self.latest_analysis['timestamp'] = datetime.now().isoformat()
        
        # 保存分析结果
        self._save_analysis_results()
        
        self.logger.info("市场多维度分析完成")
        return self.latest_analysis
    
    def _get_major_sectors(self, count=10):
        """获取主要行业板块
        
        Args:
            count: 返回的板块数量
            
        Returns:
            dict: 板块名称到板块代码的映射
        """
        if not hasattr(self, '_major_sectors_cache'):
            # 定义主要行业板块
            self._major_sectors_cache = {
                "银行": "399986.SZ",
                "医药": "399989.SZ",
                "食品饮料": "399997.SZ",
                "家电": "399996.SZ",
                "有色金属": "399993.SZ",
                "钢铁": "399994.SZ",
                "煤炭": "399998.SZ",
                "石油": "399995.SZ",
                "电力": "399992.SZ",
                "房地产": "399999.SZ",
                "汽车": "399957.SZ",
                "电子": "399987.SZ",
                "计算机": "399988.SZ",
                "通信": "399958.SZ",
                "农林牧渔": "399956.SZ",
                "建筑": "399959.SZ",
                "机械": "399908.SZ",
                "纺织服装": "399907.SZ",
                "商业贸易": "399962.SZ",
                "化工": "399909.SZ"
            }
        
        # 返回指定数量的板块
        return dict(list(self._major_sectors_cache.items())[:count])
    
    def _get_all_sectors(self):
        """获取所有行业板块
        
        Returns:
            dict: 板块名称到板块代码的映射
        """
        # 如果已缓存，直接返回
        if hasattr(self, '_all_sectors_cache'):
            return self._all_sectors_cache
        
        # 尝试从数据源获取
        if self.data_source and hasattr(self.data_source, 'get_sectors'):
            sectors = self.data_source.get_sectors()
            if sectors:
                self._all_sectors_cache = sectors
                return sectors
        
        # 否则返回主要板块
        return self._get_major_sectors(20)
    
    def _predict_next_rotation(self, rotation_history, correlation_matrix):
        """预测下一个可能轮动的板块
        
        Args:
            rotation_history: 板块轮动历史
            correlation_matrix: 板块关联性矩阵
            
        Returns:
            list: 可能的下一个轮动板块列表
        """
        if not rotation_history:
            return []
        
        # 获取当前领先板块
        current_sector = rotation_history[-1]['sector']
        
        # 统计历史轮动规律
        rotation_patterns = defaultdict(int)
        for i in range(len(rotation_history) - 1):
            from_sector = rotation_history[i]['sector']
            to_sector = rotation_history[i + 1]['sector']
            rotation_patterns[(from_sector, to_sector)] += 1
        
        # 查找与当前板块相关的过往轮动方向
        potential_next = []
        
        # 方法1: 基于历史轮动规律
        for (from_sector, to_sector), count in rotation_patterns.items():
            if from_sector == current_sector:
                potential_next.append({
                    'sector': to_sector,
                    'confidence': min(0.7, count / len(rotation_history)),
                    'reason': f"历史上在{from_sector}之后出现过{count}次",
                    'source': 'historical_pattern'
                })
        
        # 方法2: 基于相关性矩阵
        if correlation_matrix and current_sector in correlation_matrix:
            correlations = correlation_matrix[current_sector]
            
            # 寻找负相关性高的板块 (轮动通常发生在负相关板块之间)
            for sector, corr in correlations.items():
                if sector != current_sector and corr < -0.3:  # 负相关性阈值
                    potential_next.append({
                        'sector': sector,
                        'confidence': min(0.6, abs(corr)),
                        'reason': f"与当前板块负相关性为{corr:.2f}",
                        'source': 'negative_correlation'
                    })
        
        # 按可信度排序
        potential_next.sort(key=lambda x: x['confidence'], reverse=True)
        
        return potential_next[:5]  # 最多返回5个可能的下一轮动板块
    
    def _save_analysis_results(self):
        """保存分析结果到文件"""
        try:
            filename = os.path.join(self.cache_dir, f"market_analysis_{datetime.now().strftime('%Y%m%d')}.json")
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.latest_analysis, f, ensure_ascii=False, indent=2)
            self.logger.info(f"分析结果已保存到 {filename}")
        except Exception as e:
            self.logger.error(f"保存分析结果失败: {str(e)}")


def _days_between(date1_str, date2_str):
    """计算两个日期字符串之间的天数"""
    try:
        date1 = datetime.strptime(date1_str, '%Y%m%d')
        date2 = datetime.strptime(date2_str, '%Y%m%d')
        return abs((date2 - date1).days)
    except:
        return 0


def example_usage():
    """示例用法"""
    try:
        # 导入数据源
        from quantum_symbiotic_network.data_sources.tushare_data_source import TushareDataSource
        data_source = TushareDataSource()
        
        # 创建分析器
        analyzer = MarketDimensionAnalyzer(data_source)
        
        # 分析热点板块
        hot_sectors = analyzer.analyze_hot_sectors(period='7d')
        print(f"\n热点板块分析:")
        for i, sector in enumerate(hot_sectors.get('hot_sectors', [])):
            print(f"{i+1}. {sector['name']}: {sector['return']*100:.2f}%")
        
        # 分析板块轮动
        rotations = analyzer.analyze_sector_rotation(period='90d')
        print(f"\n板块轮动分析:")
        for rotation in rotations.get('rotations', [])[-3:]:  # 只显示最近3次轮动
            print(f"{rotation['start_date']} 到 {rotation['end_date']}: {rotation['sector']} (持续{rotation['duration']}天)")
        
        # 显示下一个可能的轮动
        print("\n可能的下一轮动板块:")
        for next_sector in rotations.get('next_potential_rotations', []):
            print(f"- {next_sector['sector']}: 置信度{next_sector['confidence']*100:.1f}%, {next_sector['reason']}")
        
        # 运行完整分析并保存结果
        analyzer.run_full_analysis()
        
    except Exception as e:
        print(f"示例运行失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 运行示例
    example_usage() 