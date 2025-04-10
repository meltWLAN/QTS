#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
超神量子系统 - 数据集成模块
集成了Tushare及其他数据源，为量子策略提供统一的数据接入层
"""

import logging
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import tushare as ts
import time
import json
from typing import Dict, List, Any, Optional, Union

# 导入TushareIntegrator
from test_tushare_integrator import TushareIntegrator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class DataIntegrationService:
    """超神数据集成服务
    
    为量子策略提供统一的数据访问接口，集成了多种数据源
    """
    
    def __init__(self, tushare_token=None, config_file=None):
        """初始化数据集成服务
        
        Args:
            tushare_token: Tushare API令牌
            config_file: 配置文件路径
        """
        self.config = self._load_config(config_file)
        self.tushare_token = tushare_token or self.config.get('tushare_token', "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10")
        
        # 初始化数据源
        self.tushare = TushareIntegrator(token=self.tushare_token)
        
        # 数据缓存
        self.data_cache = {}
        self.last_update = {}
        
        # 市场数据汇总
        self.market_data = {}
        
        logger.info("超神数据集成服务初始化完成")
    
    def _load_config(self, config_file=None):
        """加载配置文件
        
        Args:
            config_file: 配置文件路径
            
        Returns:
            dict: 配置信息
        """
        config = {
            'tushare_token': "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10",
            'data_cache_dir': 'data',
            'cache_expiry_hours': 24,
            'default_market': 'A股',
            'prefetch_assets': [
                '000001.SH',  # 上证指数
                '399001.SZ',  # 深证成指
                '000300.SH',  # 沪深300
                '000016.SH'   # 上证50
            ]
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                config.update(user_config)
                logger.info(f"已加载配置文件: {config_file}")
            except Exception as e:
                logger.error(f"加载配置文件失败: {str(e)}")
        
        return config
    
    def initialize(self):
        """初始化数据服务并预加载数据"""
        logger.info("开始初始化数据服务...")
        
        # 测试Tushare连接
        if not self.tushare.test_connection():
            logger.error("Tushare连接测试失败，部分功能可能不可用")
        
        # 预加载市场数据
        self._prefetch_market_data()
        
        logger.info("数据服务初始化完成")
        return True
    
    def _prefetch_market_data(self):
        """预加载市场数据"""
        logger.info("开始预加载市场数据...")
        
        # 加载指数数据
        for index_code in self.config.get('prefetch_assets', []):
            logger.info(f"预加载指数数据: {index_code}")
            index_data = self.tushare.get_index_data(
                ts_code=index_code, 
                start_date=(datetime.now() - timedelta(days=30)).strftime('%Y%m%d'),
                end_date=datetime.now().strftime('%Y%m%d')
            )
            if index_data is not None:
                self.data_cache[f"index_{index_code}"] = {
                    'data': index_data,
                    'timestamp': datetime.now(),
                    'type': 'index'
                }
        
        # 加载股票基本信息
        stock_basic = self.tushare.get_stock_basic()
        if stock_basic is not None:
            self.data_cache['stock_basic'] = {
                'data': stock_basic,
                'timestamp': datetime.now(),
                'type': 'basic'
            }
        
        logger.info("预加载市场数据完成")
    
    def get_market_data(self, data_type='stock', date=None, start_date=None, end_date=None):
        """获取市场数据
        
        Args:
            data_type: 数据类型，可选值: 'stock'/'index'/'fund'/'future'/'fx'
            date: 指定日期，格式YYYYMMDD
            start_date: 开始日期，格式YYYYMMDD
            end_date: 结束日期，格式YYYYMMDD
            
        Returns:
            dict: 市场数据
        """
        # 生成缓存键
        cache_key = f"{data_type}_{date or 'latest'}"
        if start_date and end_date:
            cache_key = f"{data_type}_{start_date}_{end_date}"
        
        # 检查缓存
        if cache_key in self.data_cache:
            cache_data = self.data_cache[cache_key]
            cache_age = (datetime.now() - cache_data['timestamp']).total_seconds() / 3600
            
            # 如果缓存未过期，直接返回
            if cache_age < self.config.get('cache_expiry_hours', 24):
                logger.info(f"使用缓存数据: {cache_key}")
                return cache_data['data']
        
        # 获取新数据
        if date is not None:
            logger.info(f"获取 {data_type} 在 {date} 的市场数据")
            market_data = self.tushare.get_market_data(category=data_type, date=date)
        elif start_date is not None and end_date is not None:
            logger.info(f"获取 {data_type} 从 {start_date} 到 {end_date} 的市场数据")
            # 通过特定接口获取数据
            if data_type == 'stock':
                market_data = self.tushare.get_stock_data(start_date=start_date, end_date=end_date)
            elif data_type == 'index':
                market_data = self.tushare.get_index_data(start_date=start_date, end_date=end_date)
            else:
                logger.warning(f"暂不支持获取 {data_type} 类型的日期范围数据")
                market_data = None
        else:
            logger.info(f"获取最新的 {data_type} 市场数据")
            market_data = self.tushare.get_market_data(category=data_type)
        
        # 更新缓存
        if market_data is not None:
            self.data_cache[cache_key] = {
                'data': market_data,
                'timestamp': datetime.now(),
                'type': data_type
            }
            logger.info(f"更新缓存数据: {cache_key}")
        
        return market_data
    
    def get_asset_data(self, asset_code, data_type='daily', start_date=None, end_date=None):
        """获取单个资产的数据
        
        Args:
            asset_code: 资产代码，如'600519.SH'
            data_type: 数据类型，可选值: 'daily'/'weekly'/'monthly'/'basic'/'financial'
            start_date: 开始日期，格式YYYYMMDD
            end_date: 结束日期，格式YYYYMMDD
            
        Returns:
            pandas.DataFrame or dict: 资产数据
        """
        # 生成缓存键
        cache_key = f"{asset_code}_{data_type}"
        if start_date and end_date:
            cache_key = f"{asset_code}_{data_type}_{start_date}_{end_date}"
        
        # 检查缓存
        if cache_key in self.data_cache:
            cache_data = self.data_cache[cache_key]
            cache_age = (datetime.now() - cache_data['timestamp']).total_seconds() / 3600
            
            # 如果缓存未过期，直接返回
            if cache_age < self.config.get('cache_expiry_hours', 24):
                logger.info(f"使用缓存数据: {cache_key}")
                return cache_data['data']
        
        # 判断是指数还是股票
        is_index = False
        if asset_code.endswith('.SH') or asset_code.endswith('.SZ'):
            code = asset_code.split('.')[0]
            if code.startswith('000') or code.startswith('399'):
                is_index = True
        
        # 获取数据
        data = None
        if data_type == 'daily':
            if is_index:
                logger.info(f"获取指数 {asset_code} 日线数据")
                data = self.tushare.get_index_data(
                    ts_code=asset_code, 
                    start_date=start_date, 
                    end_date=end_date
                )
            else:
                logger.info(f"获取股票 {asset_code} 日线数据")
                data = self.tushare.get_stock_data(
                    ts_code=asset_code, 
                    start_date=start_date, 
                    end_date=end_date
                )
        elif data_type == 'financial':
            logger.info(f"获取 {asset_code} 财务数据")
            data = self.tushare.get_financial_data(
                ts_code=asset_code, 
                start_date=start_date, 
                end_date=end_date
            )
        elif data_type == 'basic':
            if is_index:
                logger.info(f"获取指数 {asset_code} 基本信息")
                # TODO: 暂不支持获取指数基本信息
                pass
            else:
                logger.info(f"获取股票基本信息")
                stock_basic = self.tushare.get_stock_basic()
                if stock_basic is not None:
                    data = stock_basic[stock_basic['ts_code'] == asset_code]
        
        # 更新缓存
        if data is not None:
            self.data_cache[cache_key] = {
                'data': data,
                'timestamp': datetime.now(),
                'type': data_type
            }
            logger.info(f"更新缓存数据: {cache_key}")
        
        return data
    
    def get_market_index_data(self, index_code='000001.SH', start_date=None, end_date=None):
        """获取市场指数数据
        
        Args:
            index_code: 指数代码，默认为上证指数'000001.SH'
            start_date: 开始日期，格式YYYYMMDD
            end_date: 结束日期，格式YYYYMMDD
            
        Returns:
            pandas.DataFrame: 指数数据
        """
        logger.info(f"获取市场指数 {index_code} 数据")
        return self.get_asset_data(
            asset_code=index_code, 
            data_type='daily', 
            start_date=start_date, 
            end_date=end_date
        )
    
    def save_data(self, data, filename, directory=None):
        """保存数据到文件
        
        Args:
            data: 数据对象
            filename: 文件名
            directory: 目录，如果为None则使用配置中的data_cache_dir
            
        Returns:
            bool: 是否保存成功
        """
        if directory is None:
            directory = self.config.get('data_cache_dir', 'data')
        
        if isinstance(data, pd.DataFrame) or isinstance(data, dict):
            return self.tushare.save_data_to_csv(data, filename, directory)
        
        return False
    
    def clear_cache(self, older_than_hours=None):
        """清除缓存
        
        Args:
            older_than_hours: 清除超过指定小时数的缓存，如果为None则使用配置中的cache_expiry_hours
            
        Returns:
            int: 清除的缓存数量
        """
        if older_than_hours is None:
            older_than_hours = self.config.get('cache_expiry_hours', 24)
        
        logger.info(f"清除超过 {older_than_hours} 小时的缓存")
        
        count = 0
        keys_to_remove = []
        
        for key, cache_data in self.data_cache.items():
            cache_age = (datetime.now() - cache_data['timestamp']).total_seconds() / 3600
            if cache_age >= older_than_hours:
                keys_to_remove.append(key)
                count += 1
        
        for key in keys_to_remove:
            del self.data_cache[key]
        
        logger.info(f"已清除 {count} 个缓存项")
        return count


def create_integration_service(tushare_token=None, config_file=None):
    """创建数据集成服务
    
    Args:
        tushare_token: Tushare API令牌
        config_file: 配置文件路径
        
    Returns:
        DataIntegrationService: 数据集成服务实例
    """
    service = DataIntegrationService(tushare_token, config_file)
    service.initialize()
    return service


def main():
    """主函数"""
    logger.info("=== 超神量子系统 - 数据集成服务测试 ===")
    
    # 创建数据集成服务
    tushare_token = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"
    service = create_integration_service(tushare_token)
    
    # 测试获取市场数据
    logger.info("测试获取市场指数数据...")
    index_data = service.get_market_index_data(
        index_code='000001.SH',
        start_date='20230101',
        end_date='20230131'
    )
    
    if index_data is not None:
        service.save_data(index_data, "index_test.csv")
    
    # 测试获取股票数据
    logger.info("测试获取股票数据...")
    stock_data = service.get_asset_data(
        asset_code='600519.SH',
        data_type='daily',
        start_date='20230101',
        end_date='20230131'
    )
    
    if stock_data is not None:
        service.save_data(stock_data, "stock_test.csv")
    
    # 测试获取市场数据
    logger.info("测试获取整体市场数据...")
    market_data = service.get_market_data(data_type='stock')
    
    if market_data is not None:
        service.save_data(market_data, "market_test.csv")
    
    logger.info("=== 数据集成服务测试完成 ===")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 