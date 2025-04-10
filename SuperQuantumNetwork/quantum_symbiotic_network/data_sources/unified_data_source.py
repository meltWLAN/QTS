#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - 统一数据源
整合多个真实数据源，确保系统始终有可用的真实市场数据
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time

# 导入可能的数据源
try:
    from .tushare_data_source import TushareDataSource, TUSHARE_AVAILABLE
except ImportError:
    TUSHARE_AVAILABLE = False
    logging.warning("无法导入TushareDataSource")

try:
    from .akshare_data_source import AKShareDataSource, AKSHARE_AVAILABLE
except ImportError:
    AKSHARE_AVAILABLE = False
    logging.warning("无法导入AKShareDataSource")

# 设置缓存目录
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

class UnifiedDataSource:
    """统一数据源类，整合多个真实数据源，确保始终有可用的真实数据"""
    
    def __init__(self, config=None):
        """初始化统一数据源
        
        Args:
            config: 配置信息，包含各数据源的配置
        """
        self.logger = logging.getLogger("UnifiedDataSource")
        self.config = config or {}
        self.data_sources = []
        self.cache = {}
        self.last_update = {}
        self.is_ready = False
        
        # 创建缓存目录
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        # 初始化所有可用的数据源
        self._initialize_data_sources()
        
        # 记录数据源状态
        self.data_source_status = {}
        
        if not self.data_sources:
            self.logger.critical("没有可用的数据源！系统将无法获取真实市场数据")
        else:
            self.is_ready = True
    
    def _initialize_data_sources(self):
        """初始化所有可用的数据源"""
        # 初始化TuShare数据源
        if TUSHARE_AVAILABLE:
            try:
                tushare_token = self.config.get('tushare_token', '')
                if not tushare_token:
                    tushare_token = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"
                
                self.logger.info(f"初始化TuShare数据源，使用token: {tushare_token[:8]}...")
                tushare_source = TushareDataSource(token=tushare_token)
                
                if tushare_source.is_ready:
                    self.data_sources.append(('tushare', tushare_source))
                    self.data_source_status['tushare'] = True
                    self.logger.info("TuShare数据源初始化成功")
                else:
                    self.data_source_status['tushare'] = False
                    self.logger.warning("TuShare数据源初始化失败")
            except Exception as e:
                self.data_source_status['tushare'] = False
                self.logger.error(f"初始化TuShare数据源出错: {str(e)}")
        else:
            self.data_source_status['tushare'] = False
            self.logger.warning("TuShare库不可用")
        
        # 初始化AKShare数据源
        if AKSHARE_AVAILABLE:
            try:
                self.logger.info("初始化AKShare数据源...")
                akshare_source = AKShareDataSource()
                
                if akshare_source.is_ready:
                    self.data_sources.append(('akshare', akshare_source))
                    self.data_source_status['akshare'] = True
                    self.logger.info("AKShare数据源初始化成功")
                else:
                    self.data_source_status['akshare'] = False
                    self.logger.warning("AKShare数据源初始化失败")
            except Exception as e:
                self.data_source_status['akshare'] = False
                self.logger.error(f"初始化AKShare数据源出错: {str(e)}")
        else:
            self.data_source_status['akshare'] = False
            self.logger.warning("AKShare库不可用")
        
        self.logger.info(f"统一数据源初始化完成，共有 {len(self.data_sources)} 个可用数据源")
    
    def _load_cache(self, cache_name):
        """从缓存加载数据"""
        cache_file = os.path.join(CACHE_DIR, f"unified_{cache_name}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.logger.info(f"从统一缓存加载{cache_name}数据成功")
                return data
            except Exception as e:
                self.logger.error(f"从统一缓存加载{cache_name}数据失败: {str(e)}")
        return None
    
    def _save_cache(self, cache_name, data):
        """保存数据到缓存"""
        cache_file = os.path.join(CACHE_DIR, f"unified_{cache_name}.json")
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"保存{cache_name}数据到统一缓存成功")
            return True
        except Exception as e:
            self.logger.error(f"保存{cache_name}数据到统一缓存失败: {str(e)}")
            return False
    
    def get_market_status(self):
        """获取市场状态
        
        尝试所有可用的数据源，返回第一个成功的结果
        
        Returns:
            dict: 市场状态信息
        """
        # 尝试从缓存加载
        cache_key = "market_status"
        cached_data = self.cache.get(cache_key)
        cache_time = self.last_update.get(cache_key, 0)
        # 缓存有效期为5分钟
        if cached_data and time.time() - cache_time < 300:
            return cached_data
        
        # 尝试所有数据源
        for source_name, source in self.data_sources:
            try:
                market_status = source.get_market_status()
                if market_status:
                    # 添加数据源信息
                    market_status['source'] = source_name
                    
                    # 更新缓存
                    self.cache[cache_key] = market_status
                    self.last_update[cache_key] = time.time()
                    
                    return market_status
            except Exception as e:
                self.logger.warning(f"从 {source_name} 获取市场状态失败: {str(e)}")
        
        # 所有数据源都失败，返回基本状态
        basic_status = {
            "status": "未知",
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "trading_day": self._is_today_likely_trading_day(),
            "next_trading_day": None,
            "last_trading_day": None,
            "source": "fallback"
        }
        return basic_status
    
    def _is_today_likely_trading_day(self):
        """简单判断今天是否为交易日"""
        today = datetime.now()
        # 周末不是交易日
        if today.weekday() >= 5:
            return False
        
        # 判断是否在交易时间内
        now = today.time()
        morning_start = datetime.strptime('09:30', '%H:%M').time()
        morning_end = datetime.strptime('11:30', '%H:%M').time()
        afternoon_start = datetime.strptime('13:00', '%H:%M').time()
        afternoon_end = datetime.strptime('15:00', '%H:%M').time()
        
        if (morning_start <= now <= morning_end) or (afternoon_start <= now <= afternoon_end):
            return True
        
        return False
    
    def get_index_data(self):
        """获取指数数据
        
        尝试所有可用的数据源，返回第一个成功的结果
        
        Returns:
            dict: 指数数据
        """
        # 尝试从缓存加载
        cache_key = "index_data"
        cached_data = self.cache.get(cache_key)
        cache_time = self.last_update.get(cache_key, 0)
        # 缓存有效期为10分钟
        if cached_data and time.time() - cache_time < 600:
            return cached_data
        
        # 尝试所有数据源
        for source_name, source in self.data_sources:
            try:
                if hasattr(source, 'get_index_data'):
                    indices_data = source.get_index_data()
                    if indices_data and len(indices_data) > 0:
                        # 更新缓存
                        self.cache[cache_key] = indices_data
                        self.last_update[cache_key] = time.time()
                        return indices_data
            except Exception as e:
                self.logger.warning(f"从 {source_name} 获取指数数据失败: {str(e)}")
        
        # 尝试加载统一缓存
        cached_indices = self._load_cache("indices_data")
        if cached_indices:
            return cached_indices.get('indices', {})
        
        # 所有数据源都失败
        self.logger.error("所有数据源获取指数数据均失败")
        return {}
    
    def get_recommended_stocks(self, count=10):
        """获取推荐股票
        
        尝试所有可用的数据源，返回第一个成功的结果
        
        Args:
            count: 推荐股票数量
            
        Returns:
            list: 推荐股票列表
        """
        # 尝试从缓存加载
        cache_key = "recommended_stocks"
        cached_data = self.cache.get(cache_key)
        cache_time = self.last_update.get(cache_key, 0)
        # 缓存有效期为30分钟
        if cached_data and time.time() - cache_time < 1800:
            return cached_data[:count] if len(cached_data) > count else cached_data
        
        # 尝试所有数据源
        for source_name, source in self.data_sources:
            try:
                if hasattr(source, 'get_recommended_stocks'):
                    stocks = source.get_recommended_stocks(count)
                    if stocks and len(stocks) > 0:
                        # 更新缓存
                        self.cache[cache_key] = stocks
                        self.last_update[cache_key] = time.time()
                        
                        # 保存到统一缓存
                        cache_data = {
                            'stocks': stocks,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                        self._save_cache("recommended_stocks", cache_data)
                        
                        return stocks
            except Exception as e:
                self.logger.warning(f"从 {source_name} 获取推荐股票失败: {str(e)}")
        
        # 尝试加载统一缓存
        cached_stocks = self._load_cache("recommended_stocks")
        if cached_stocks:
            stocks = cached_stocks.get('stocks', [])
            return stocks[:count] if len(stocks) > count else stocks
        
        # 所有数据源都失败
        self.logger.error("所有数据源获取推荐股票均失败")
        return []
    
    def get_hot_stocks(self, count=5):
        """获取热门股票
        
        尝试所有可用的数据源，返回第一个成功的结果
        
        Args:
            count: 热门股票数量
            
        Returns:
            list: 热门股票列表
        """
        # 尝试所有数据源
        for source_name, source in self.data_sources:
            try:
                if hasattr(source, 'get_hot_stocks'):
                    stocks = source.get_hot_stocks(count)
                    if stocks and len(stocks) > 0:
                        return stocks
            except Exception as e:
                self.logger.warning(f"从 {source_name} 获取热门股票失败: {str(e)}")
        
        # 尝试从推荐股票中获取热门股票
        recommended = self.get_recommended_stocks(count * 2)
        if recommended:
            # 按推荐度排序，取前count个
            sorted_stocks = sorted(recommended, key=lambda x: x.get('recommendation', 0), reverse=True)
            return sorted_stocks[:count]
        
        # 所有数据源都失败
        self.logger.error("所有数据源获取热门股票均失败")
        return []
    
    def get_stock_data(self, code, force_refresh=False):
        """获取单个股票数据
        
        尝试所有可用的数据源，返回第一个成功的结果
        
        Args:
            code: 股票代码
            force_refresh: 是否强制刷新
            
        Returns:
            dict: 股票数据
        """
        # 处理股票代码格式
        if "." in code:
            pure_code = code.split('.')[0]
        else:
            pure_code = code
        
        # 尝试从缓存加载
        cache_key = f"stock_data_{pure_code}"
        if not force_refresh:
            cached_data = self.cache.get(cache_key)
            cache_time = self.last_update.get(cache_key, 0)
            # 缓存有效期为10分钟
            if cached_data and time.time() - cache_time < 600:
                return cached_data
        
        # 尝试所有数据源
        for source_name, source in self.data_sources:
            try:
                if hasattr(source, 'get_stock_data'):
                    stock_data = source.get_stock_data(code, force_refresh)
                    if stock_data and len(stock_data) > 0:
                        # 更新缓存
                        self.cache[cache_key] = stock_data
                        self.last_update[cache_key] = time.time()
                        
                        return stock_data
            except Exception as e:
                self.logger.warning(f"从 {source_name} 获取股票 {code} 数据失败: {str(e)}")
        
        # 所有数据源都失败
        self.logger.error(f"所有数据源获取股票 {code} 数据均失败")
        return {}
    
    def search_stocks(self, keyword, limit=10):
        """搜索股票
        
        尝试所有可用的数据源，返回第一个成功的结果
        
        Args:
            keyword: 搜索关键词
            limit: 最大返回数量
            
        Returns:
            list: 匹配的股票列表
        """
        # 尝试所有数据源
        for source_name, source in self.data_sources:
            try:
                if hasattr(source, 'search_stocks'):
                    results = source.search_stocks(keyword, limit)
                    if results and len(results) > 0:
                        return results
            except Exception as e:
                self.logger.warning(f"从 {source_name} 搜索股票失败: {str(e)}")
        
        # 所有数据源都失败
        self.logger.error("所有数据源搜索股票均失败")
        return []
    
    def get_data_source_status(self):
        """获取数据源状态
        
        Returns:
            dict: 数据源状态信息
        """
        # 更新数据源状态
        for source_name, source in self.data_sources:
            try:
                # 尝试获取市场状态检验数据源是否可用
                if hasattr(source, 'get_market_status'):
                    source.get_market_status()
                    self.data_source_status[source_name] = True
                else:
                    self.data_source_status[source_name] = False
            except Exception:
                self.data_source_status[source_name] = False
        
        return self.data_source_status
    
    def update_all_cache(self):
        """更新所有缓存数据"""
        try:
            self.logger.info("开始更新所有缓存数据...")
            
            # 更新市场状态
            market_status = self.get_market_status()
            
            # 更新指数数据
            indices = self.get_index_data()
            
            # 更新推荐股票
            recommended = self.get_recommended_stocks(20)
            
            self.logger.info("所有缓存数据更新完成")
            
            return {
                "market_status": market_status,
                "indices": indices,
                "recommended": recommended,
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            self.logger.error(f"更新缓存数据失败: {str(e)}")
            return None 