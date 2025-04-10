#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强版数据连接器框架 - 多源数据集成与高频数据支持
"""

import logging
import time
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import threading
import queue
import json
import os

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('DataConnector')

class DataSourceBase(ABC):
    """数据源基类，所有数据源都应继承此类"""
    
    def __init__(self, name, config=None):
        self.name = name
        self.config = config or {}
        self.priority = self.config.get('priority', 5)  # 1-10，优先级越高越先使用
        self.status = "initialized"
        self.logger = logging.getLogger(f'DataSource.{name}')
        self.last_error = None
        self.error_count = 0
        self.max_errors = self.config.get('max_errors', 5)
        self.last_success = None
        
    @abstractmethod
    def initialize(self):
        """初始化数据源连接"""
        pass
    
    @abstractmethod
    def check_connection(self):
        """检查数据源连接状态"""
        pass
    
    @abstractmethod
    def get_market_data(self, code, start_date=None, end_date=None, period='daily'):
        """获取市场数据"""
        pass
    
    @abstractmethod
    def get_sector_data(self, date=None):
        """获取板块数据"""
        pass
    
    @abstractmethod
    def get_stock_list(self, market=None):
        """获取股票列表"""
        pass
        
    def handle_error(self, error, context="operation"):
        """处理错误"""
        self.last_error = error
        self.error_count += 1
        self.logger.error(f"{context} error: {str(error)}")
        
        if self.error_count >= self.max_errors:
            self.status = "failed"
            self.logger.warning(f"Data source {self.name} marked as failed after {self.error_count} errors")
            return False
        return True
    
    def reset_error_count(self):
        """重置错误计数"""
        if self.error_count > 0:
            self.error_count = 0
            self.logger.info(f"Error count reset for {self.name}")


class TushareDataSource(DataSourceBase):
    """Tushare数据源实现"""
    
    def __init__(self, config=None):
        super().__init__("tushare", config)
        self.token = self.config.get('token')
        self.api = None
        
    def initialize(self):
        """初始化Tushare连接"""
        try:
            import tushare as ts
            if not self.token:
                self.logger.error("Tushare token not provided")
                self.status = "failed"
                return False
                
            self.api = ts.pro_api(self.token)
            self.status = "connected"
            self.logger.info("Tushare API initialized successfully")
            return True
        except ImportError:
            self.logger.error("Tushare package not installed")
            self.status = "failed"
            return False
        except Exception as e:
            self.handle_error(e, "initialization")
            return False
    
    def check_connection(self):
        """检查Tushare连接状态"""
        if not self.api:
            return False
            
        try:
            # 简单查询测试连接
            self.api.query('stock_basic', limit=1)
            self.status = "connected"
            self.last_success = datetime.now()
            return True
        except Exception as e:
            self.handle_error(e, "connection check")
            return False
    
    def get_market_data(self, code, start_date=None, end_date=None, period='daily'):
        """获取市场数据"""
        if not self.api or self.status != "connected":
            self.logger.error("Tushare API not connected")
            return None
            
        try:
            # 设置默认日期
            if end_date is None:
                end_date = datetime.now().strftime('%Y%m%d')
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
                
            # 根据周期选择接口
            if period == 'daily':
                df = self.api.daily(ts_code=code, start_date=start_date, end_date=end_date)
            elif period == 'weekly':
                df = self.api.weekly(ts_code=code, start_date=start_date, end_date=end_date)
            elif period == 'monthly':
                df = self.api.monthly(ts_code=code, start_date=start_date, end_date=end_date)
            elif period == 'minute':
                # 分钟数据需要特殊处理，这里简化实现
                df = self.api.mins(ts_code=code, start_date=start_date, end_date=end_date)
            else:
                self.logger.error(f"Unsupported period: {period}")
                return None
                
            if df is not None and not df.empty:
                # 处理日期列
                if 'trade_date' in df.columns:
                    df['date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
                    df.set_index('date', inplace=True)
                
                # 确保列名标准化
                df.rename(columns={
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'vol': 'volume',
                    'amount': 'amount',
                    'pct_chg': 'change_pct'
                }, inplace=True)
                
                self.reset_error_count()
                self.last_success = datetime.now()
                return df
            else:
                self.logger.warning(f"No data returned for {code}")
                return None
        except Exception as e:
            self.handle_error(e, f"market data retrieval for {code}")
            return None
    
    # 其他方法实现...


class SinaDataSource(DataSourceBase):
    """新浪财经数据源实现"""
    
    def __init__(self, config=None):
        super().__init__("sina", config)
        # 实现略...


class DataConnector:
    """
    增强版数据连接器 - 支持多数据源、自动切换和数据缓存
    """
    
    def __init__(self, config_path=None):
        self.logger = logging.getLogger('DataConnector')
        self.config = self._load_config(config_path)
        self.data_sources = {}
        self.active_source = None
        self.initialized = False
        self.data_cache = {}
        self.cache_dir = self.config.get('cache_dir', 'data/cache')
        
        # 创建缓存目录
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 初始化数据源
        self._init_data_sources()
    
    def _load_config(self, config_path):
        """加载配置文件"""
        default_config = {
            'cache_enabled': True,
            'cache_ttl': 24 * 60 * 60,  # 24小时缓存过期
            'data_sources': {
                'tushare': {
                    'enabled': True,
                    'token': '0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10',
                    'priority': 10
                },
                'sina': {
                    'enabled': False,
                    'priority': 5
                },
                'eastmoney': {
                    'enabled': False,
                    'priority': 7
                }
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                # 合并配置
                for key, value in config.items():
                    if key == 'data_sources':
                        for source, source_config in value.items():
                            if source in default_config['data_sources']:
                                default_config['data_sources'][source].update(source_config)
                            else:
                                default_config['data_sources'][source] = source_config
                    else:
                        default_config[key] = value
            except Exception as e:
                self.logger.error(f"Error loading config file: {e}")
        
        return default_config
    
    def _init_data_sources(self):
        """初始化所有数据源"""
        data_sources_config = self.config.get('data_sources', {})
        
        for source_name, source_config in data_sources_config.items():
            if source_config.get('enabled', False):
                self._create_data_source(source_name, source_config)
        
        # 按优先级排序
        self._select_active_source()
        
        if self.active_source:
            self.initialized = True
            self.logger.info(f"Data connector initialized with active source: {self.active_source.name}")
        else:
            self.logger.warning("No active data source available")
    
    def _create_data_source(self, source_name, source_config):
        """创建数据源实例"""
        try:
            if source_name == 'tushare':
                source = TushareDataSource(source_config)
            elif source_name == 'sina':
                source = SinaDataSource(source_config)
            # 添加更多数据源...
            else:
                self.logger.warning(f"Unknown data source: {source_name}")
                return
                
            # 初始化数据源
            if source.initialize():
                self.data_sources[source_name] = source
                self.logger.info(f"Data source {source_name} initialized with priority {source.priority}")
            else:
                self.logger.warning(f"Failed to initialize data source: {source_name}")
        except Exception as e:
            self.logger.error(f"Error creating data source {source_name}: {e}")
    
    def _select_active_source(self):
        """选择活跃的数据源"""
        available_sources = [s for s in self.data_sources.values() 
                            if s.status == "connected" or s.status == "initialized"]
        
        if not available_sources:
            self.active_source = None
            return
            
        # 按优先级排序
        available_sources.sort(key=lambda s: s.priority, reverse=True)
        
        # 选择优先级最高的
        self.active_source = available_sources[0]
        self.logger.info(f"Selected active data source: {self.active_source.name}")
    
    def get_market_data(self, code, start_date=None, end_date=None, period='daily', force_refresh=False):
        """
        获取市场数据，支持缓存和自动切换数据源
        """
        if not self.initialized:
            self.logger.error("Data connector not initialized")
            return None
            
        # 检查缓存
        cache_key = f"{code}_{period}_{start_date}_{end_date}"
        
        if not force_refresh and self.config.get('cache_enabled', True):
            cached_data = self._get_from_cache(cache_key, 'market')
            if cached_data is not None:
                return cached_data
        
        # 尝试从主数据源获取
        if self.active_source:
            data = self.active_source.get_market_data(code, start_date, end_date, period)
            if data is not None and not data.empty:
                # 缓存数据
                if self.config.get('cache_enabled', True):
                    self._save_to_cache(cache_key, data, 'market')
                return data
        
        # 如果主数据源失败，尝试其他数据源
        for source in self.data_sources.values():
            if source != self.active_source and (source.status == "connected" or source.check_connection()):
                self.logger.info(f"Trying alternate data source: {source.name}")
                data = source.get_market_data(code, start_date, end_date, period)
                if data is not None and not data.empty:
                    # 更新活跃数据源
                    self.active_source = source
                    # 缓存数据
                    if self.config.get('cache_enabled', True):
                        self._save_to_cache(cache_key, data, 'market')
                    return data
        
        self.logger.error(f"Failed to get market data for {code} from all sources")
        return None
    
    def _get_from_cache(self, key, data_type):
        """从缓存获取数据"""
        cache_file = os.path.join(self.cache_dir, f"{data_type}_{key}.pkl")
        
        if os.path.exists(cache_file):
            # 检查缓存是否过期
            file_mtime = os.path.getmtime(cache_file)
            if time.time() - file_mtime < self.config.get('cache_ttl', 24 * 60 * 60):
                try:
                    data = pd.read_pickle(cache_file)
                    self.logger.info(f"Loaded {data_type} data from cache: {key}")
                    return data
                except Exception as e:
                    self.logger.warning(f"Error loading cached data: {e}")
        
        return None
    
    def _save_to_cache(self, key, data, data_type):
        """保存数据到缓存"""
        try:
            cache_file = os.path.join(self.cache_dir, f"{data_type}_{key}.pkl")
            data.to_pickle(cache_file)
            self.logger.info(f"Saved {data_type} data to cache: {key}")
            return True
        except Exception as e:
            self.logger.warning(f"Error saving data to cache: {e}")
            return False

# 使用示例
if __name__ == "__main__":
    # 创建数据连接器
    connector = DataConnector()
    
    # 获取沪深300指数数据
    df = connector.get_market_data("000300.SH", period='daily')
    if df is not None:
        print(f"Got data for 000300.SH, shape: {df.shape}")
        print(df.head())
    else:
        print("Failed to get data") 