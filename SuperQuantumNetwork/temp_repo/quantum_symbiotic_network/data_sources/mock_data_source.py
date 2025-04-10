#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - 数据缓存模块
该模块用于缓存TuShare的真实数据，在TuShare不可用时作为备用数据源
"""

import os
import json
import logging
import traceback
from datetime import datetime, timedelta
import time
import pandas as pd
import numpy as np

logger = logging.getLogger("TushareDataCache")

class TushareDataCache:
    """TuShare数据缓存 - 在TuShare不可用时使用缓存的真实数据

    超神系统确保使用真实市场数据，该类负责缓存TuShare数据并在必要时提供
    """
    
    def __init__(self):
        """初始化数据缓存"""
        self.logger = logging.getLogger("TushareDataCache")
        self.logger.info("初始化TuShare数据缓存模块")
        
        # 初始化数据
        self.stocks = {}
        self.indices = {}
        self.recommended_stocks = []
        self.last_update_time = None
        
        # 缓存目录
        self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 加载缓存的数据
        self._load_cached_data()
    
    def _load_cached_data(self):
        """从缓存文件加载数据"""
        try:
            # 加载股指数据
            indices_cache_file = os.path.join(self.cache_dir, "indices_data.json")
            if os.path.exists(indices_cache_file):
                with open(indices_cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    self.indices = cached_data.get('indices', {})
                    self.last_update_time = cached_data.get('updated_at')
                    self.logger.info(f"从缓存加载了 {len(self.indices)} 个指数的数据，最后更新时间: {self.last_update_time}")
            
            # 加载推荐股票数据
            stocks_cache_file = os.path.join(self.cache_dir, "recommended_stocks.json")
            if os.path.exists(stocks_cache_file):
                with open(stocks_cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    self.recommended_stocks = cached_data.get('recommended_stocks', [])
                    self.logger.info(f"从缓存加载了 {len(self.recommended_stocks)} 只推荐股票的数据")
            
            # 加载股票数据
            stocks_data_file = os.path.join(self.cache_dir, "stocks_data.json")
            if os.path.exists(stocks_data_file):
                with open(stocks_data_file, 'r', encoding='utf-8') as f:
                    self.stocks = json.load(f)
                    self.logger.info(f"从缓存加载了 {len(self.stocks)} 只股票的基本数据")
                    
        except Exception as e:
            self.logger.error(f"加载缓存数据失败: {str(e)}")
            self.logger.error(traceback.format_exc())
    
    def update_cache_from_tushare(self, tushare_source):
        """从TuShare数据源更新缓存
        
        Args:
            tushare_source: TuShare数据源实例
        """
        if not tushare_source or not hasattr(tushare_source, 'api') or not tushare_source.api:
            self.logger.error("TuShare数据源未就绪，无法更新缓存")
            return False
        
        success = True
        try:
            # 更新指数数据
            self.logger.info("开始从TuShare更新指数数据...")
            indices_data = tushare_source.get_index_data()
            if indices_data:
                self.indices = indices_data
                # 保存到缓存
                cache_data = {
                    'indices': indices_data,
                    'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                with open(os.path.join(self.cache_dir, "indices_data.json"), 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, ensure_ascii=False, indent=2)
                self.logger.info(f"已更新 {len(indices_data)} 个指数的缓存数据")
            else:
                self.logger.warning("从TuShare获取指数数据失败")
                success = False
            
            # 更新推荐股票
            self.logger.info("开始从TuShare更新推荐股票数据...")
            recommended_stocks = tushare_source.get_recommended_stocks()
            if recommended_stocks:
                self.recommended_stocks = recommended_stocks
                # 保存到缓存
                cache_data = {
                    'recommended_stocks': recommended_stocks,
                    'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                with open(os.path.join(self.cache_dir, "recommended_stocks.json"), 'w', encoding='utf-8') as f:
                    json.dump(cache_data, f, ensure_ascii=False, indent=2)
                self.logger.info(f"已更新 {len(recommended_stocks)} 只推荐股票的缓存数据")
            else:
                self.logger.warning("从TuShare获取推荐股票数据失败")
                success = False
            
            # 为每只推荐股票更新详细数据
            for stock in self.recommended_stocks:
                code = stock.get('code')
                if code:
                    try:
                        stock_data = tushare_source.get_stock_data(code)
                        if stock_data:
                            self.stocks[code] = stock_data
                    except Exception as e:
                        self.logger.warning(f"更新股票 {code} 详细数据失败: {str(e)}")
            
            # 保存股票数据
            with open(os.path.join(self.cache_dir, "stocks_data.json"), 'w', encoding='utf-8') as f:
                json.dump(self.stocks, f, ensure_ascii=False, indent=2)
            
            self.last_update_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.logger.info(f"数据缓存更新完成，时间: {self.last_update_time}")
            return success
            
        except Exception as e:
            self.logger.error(f"更新缓存数据失败: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
    
    # 公开API方法 - 这些方法将被系统调用
    
    def get_indices(self):
        """获取指数数据"""
        self.logger.warning("使用缓存的真实指数数据 - 请检查TuShare连接")
        return self.indices
    
    def get_recommended_stocks(self):
        """获取推荐股票"""
        self.logger.warning("使用缓存的真实推荐股票数据 - 请检查TuShare连接")
        return self.recommended_stocks
    
    def get_hot_stocks(self, count=5):
        """获取热门股票"""
        self.logger.warning("使用缓存的真实热门股票数据 - 请检查TuShare连接")
        if not self.recommended_stocks:
            self.logger.error("缓存中没有推荐股票数据")
            return []
        
        # 按推荐度排序
        sorted_stocks = sorted(self.recommended_stocks, 
                              key=lambda x: x.get('recommendation', 0) if isinstance(x, dict) else 0, 
                              reverse=True)
        return sorted_stocks[:count]
    
    def get_stock_data(self, code):
        """获取股票数据"""
        self.logger.warning(f"使用缓存的真实股票数据 {code} - 请检查TuShare连接")
        
        # 检查是否已有该股票数据
        if code in self.stocks:
            return self.stocks[code]
        
        # 没有缓存数据
        self.logger.error(f"缓存中没有股票 {code} 的数据")
        return {
            'code': code,
            'name': f"未知股票{code}",
            'price': 0.0,
            'change': 0.0,
            'change_pct': 0.0,
            'volume': 0,
            'turnover': 0,
            'error': '无法获取股票数据'
        }

# 为了保持向后兼容，使用相同的类名
MockDataSource = TushareDataCache 