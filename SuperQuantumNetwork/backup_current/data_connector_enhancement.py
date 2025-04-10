#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强版数据连接器框架 - 多源数据集成与高频数据支持

核心功能:
1. 多数据源集成 - 支持Tushare、AKShare等多个数据源，根据优先级自动选择
2. 数据源故障自动切换 - 当主数据源不可用时自动切换到备用数据源
3. 高频数据支持 - 支持分钟级别实时行情数据
4. 数据缓存系统 - 智能本地缓存，避免重复请求，提高响应速度
5. 并发数据获取 - 支持并发请求多个股票数据
6. 数据源健康监控 - 实时监控数据源状态，记录故障信息
7. 统一数据格式 - 统一不同数据源返回的数据格式，方便使用

设计特点:
- 基于标准抽象基类设计，易于扩展新数据源
- 智能缓存管理，自动检测和清理过期缓存
- 高可用性设计，保证数据服务连续性
- 保持数据格式一致性，简化数据处理
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

    def get_sector_data(self, date=None):
        """获取板块数据"""
        if not self.api or self.status != "connected":
            self.logger.error("Tushare API not connected")
            return None
            
        try:
            # 设置默认日期为今天
            if date is None:
                date = datetime.now().strftime('%Y%m%d')
                
            # 获取行业分类
            industry_df = self.api.industry()
            
            # 获取每个行业的涨跌幅等信息
            sectors = []
            
            if industry_df is not None and not industry_df.empty:
                # 获取指数列表
                index_df = self.api.index_basic(market='SW')
                
                if index_df is not None and not index_df.empty:
                    # 获取指数每日指标
                    index_dailies = self.api.index_dailybasic(trade_date=date)
                    
                    if index_dailies is not None and not index_dailies.empty:
                        # 合并数据
                        merged = pd.merge(index_df, index_dailies, on='ts_code', how='inner')
                        
                        # 构建板块数据
                        for _, row in merged.iterrows():
                            sector = {
                                'name': row['name'],
                                'code': row['ts_code'],
                                'change_pct': row.get('pct_change', 0) / 100,  # 转为小数
                                'pe': row.get('pe', 0),
                                'pb': row.get('pb', 0),
                                'turnover_rate': row.get('turnover_rate', 0) / 100,  # 转为小数
                                'volume_ratio': row.get('volume_ratio', 0)
                            }
                            sectors.append(sector)
            
            return {
                'date': date,
                'sectors': sectors
            }
            
        except Exception as e:
            self.handle_error(e, "sector data retrieval")
            return None
            
    def get_stock_list(self, market=None):
        """获取股票列表"""
        if not self.api or self.status != "connected":
            self.logger.error("Tushare API not connected")
            return None
            
        try:
            # 根据市场筛选
            market_param = market if market else ''
            
            # 获取股票基本信息
            df = self.api.stock_basic(exchange=market_param, list_status='L')
            
            stock_list = []
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    stock = {
                        'ts_code': row['ts_code'],
                        'symbol': row['symbol'],
                        'name': row['name'],
                        'industry': row.get('industry', ''),
                        'market': row['ts_code'][-2:],
                        'list_date': row.get('list_date', '')
                    }
                    stock_list.append(stock)
            
            return stock_list
            
        except Exception as e:
            self.handle_error(e, "stock list retrieval")
            return None
            
    def get_high_frequency_data(self, code, interval='1min', count=120):
        """
        获取高频数据
        
        参数:
            code: 股票代码
            interval: 时间间隔 (1min, 5min, 15min, 30min, 60min)
            count: 获取的条数
            
        返回:
            DataFrame: 分钟级数据
        """
        if not self.api or self.status != "connected":
            self.logger.error("Tushare API not connected")
            return None
            
        try:
            import tushare as ts
            
            # 获取今天日期
            today = datetime.now().strftime('%Y%m%d')
            
            # Tushare分钟级数据需要额外处理
            # 由于Tushare Pro分钟级数据接口限制，这里使用普通tushare接口
            
            # 转换周期格式
            freq_map = {
                '1min': '1',
                '5min': '5',
                '15min': '15',
                '30min': '30',
                '60min': '60'
            }
            
            # 检查是否支持该周期
            if interval not in freq_map:
                self.logger.error(f"Unsupported interval: {interval}")
                return None
                
            # 转换代码格式
            if code.endswith('.SH'):
                ts_code = code.replace('.SH', '')
                market = 'sh'
            elif code.endswith('.SZ'):
                ts_code = code.replace('.SZ', '')
                market = 'sz'
            else:
                self.logger.error(f"Invalid code format: {code}")
                return None
                
            # 获取分钟级数据
            df = ts.get_k_data(code=ts_code, ktype=freq_map[interval], 
                             autype='qfq', index=False, start=None, end=None)
            
            if df is not None and not df.empty:
                # 确保列名标准化
                df.rename(columns={
                    'date': 'datetime',
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'volume': 'volume'
                }, inplace=True)
                
                # 转换时间格式
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
                
                # 限制返回的条数
                if len(df) > count:
                    df = df.iloc[-count:]
                
                self.reset_error_count()
                self.last_success = datetime.now()
                return df
            else:
                self.logger.warning(f"No high frequency data returned for {code}")
                return None
        except Exception as e:
            self.handle_error(e, f"high frequency data retrieval for {code}")
            return None
            
    def get_market_status(self):
        """获取市场状态"""
        if not self.api or self.status != "connected":
            self.logger.error("Tushare API not connected")
            return None
            
        try:
            # 获取当前日期
            now = datetime.now()
            today = now.strftime('%Y%m%d')
            
            # 检查是否是交易日
            trade_cal = self.api.trade_cal(exchange='SSE', start_date=today, end_date=today)
            
            is_trading_day = False
            if trade_cal is not None and not trade_cal.empty:
                is_trading_day = trade_cal.iloc[0]['is_open'] == 1
            
            # 检查交易时间
            market_open = False
            if is_trading_day:
                hour = now.hour
                minute = now.minute
                if (hour == 9 and minute >= 30) or (hour == 10) or (hour == 11 and minute <= 30) or \
                   (hour == 13) or (hour == 14):
                    market_open = True
            
            # 获取下一个交易日
            next_day = (now + timedelta(days=1)).strftime('%Y%m%d')
            next_week = (now + timedelta(days=7)).strftime('%Y%m%d')
            next_cal = self.api.trade_cal(exchange='SSE', start_date=next_day, end_date=next_week, is_open=1)
            
            next_trading_day = None
            if next_cal is not None and not next_cal.empty:
                next_trading_day = next_cal.iloc[0]['cal_date']
            
            return {
                "trading_day": is_trading_day,
                "market_open": market_open,
                "next_trading_day": next_trading_day,
                "time": now.strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            self.handle_error(e, "market status retrieval")
            return None


class AKShareDataSource(DataSourceBase):
    """AKShare数据源实现 - 作为Tushare的备用数据源"""
    
    def __init__(self, config=None):
        super().__init__("akshare", config)
        self.cache_dir = self.config.get('cache_dir', 'data/cache/akshare')
        self.cache_ttl = self.config.get('cache_ttl', 3600)  # 默认缓存1小时
        self.api_available = False
        
        # 创建缓存目录
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def initialize(self):
        """初始化AKShare连接"""
        try:
            import akshare as ak
            # 简单测试AKShare是否可用
            test = ak.stock_zh_index_spot()
            if test is not None and not test.empty:
                self.api_available = True
                self.status = "connected"
                self.logger.info("AKShare API initialized successfully")
                return True
            else:
                self.logger.error("AKShare API test failed")
                self.status = "failed"
                return False
        except ImportError:
            self.logger.error("AKShare package not installed")
            self.status = "failed"
            return False
        except Exception as e:
            self.handle_error(e, "initialization")
            return False
    
    def check_connection(self):
        """检查AKShare连接状态"""
        if not self.api_available:
            return False
            
        try:
            import akshare as ak
            # 简单测试
            ak.stock_zh_index_spot()
            self.status = "connected"
            self.last_success = datetime.now()
            return True
        except Exception as e:
            self.handle_error(e, "connection check")
            return False
    
    def get_market_data(self, code, start_date=None, end_date=None, period='daily'):
        """获取市场数据"""
        if not self.api_available or self.status != "connected":
            self.logger.error("AKShare API not connected")
            return None
            
        try:
            import akshare as ak
            
            # 设置默认日期
            if end_date is None:
                end_date = datetime.now().strftime('%Y%m%d')
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
            
            # 转换代码格式
            market = code[-2:].lower()
            symbol = code[:-3]
            
            if market == 'sh':
                full_code = f"sh{symbol}"
            elif market == 'sz':
                full_code = f"sz{symbol}"
            else:
                self.logger.error(f"Unsupported market in code: {code}")
                return None
            
            # 根据周期获取数据
            if period == 'daily':
                if symbol.startswith('0') or symbol.startswith('3'):  # 指数
                    df = ak.stock_zh_index_daily(symbol=full_code)
                else:
                    df = ak.stock_zh_a_hist(symbol=full_code, period="daily", 
                                         start_date=start_date, end_date=end_date)
            elif period == 'weekly':
                df = ak.stock_zh_a_hist(symbol=full_code, period="weekly", 
                                     start_date=start_date, end_date=end_date)
            elif period == 'monthly':
                df = ak.stock_zh_a_hist(symbol=full_code, period="monthly", 
                                      start_date=start_date, end_date=end_date)
            else:
                self.logger.error(f"Unsupported period: {period}")
                return None
                
            if df is not None and not df.empty:
                # 标准化列名
                column_mappings = {
                    '日期': 'date',
                    '开盘': 'open',
                    '收盘': 'close',
                    '最高': 'high',
                    '最低': 'low',
                    '成交量': 'volume',
                    '成交额': 'amount',
                    '涨跌幅': 'change_pct',
                    'open': 'open',
                    'close': 'close',
                    'high': 'high',
                    'low': 'low',
                    'volume': 'volume',
                    'amount': 'amount',
                    'pct_chg': 'change_pct',
                    'date': 'date'
                }
                
                # 重命名存在的列
                rename_dict = {k: v for k, v in column_mappings.items() if k in df.columns}
                if rename_dict:
                    df.rename(columns=rename_dict, inplace=True)
                
                # 处理日期
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                
                # 过滤日期范围
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                df = df[(df.index >= start_dt) & (df.index <= end_dt)]
                
                # 确保返回的数据是按日期排序的
                df = df.sort_index()
                
                self.reset_error_count()
                self.last_success = datetime.now()
                return df
            else:
                self.logger.warning(f"No data returned for {code}")
                return None
        except Exception as e:
            self.handle_error(e, f"market data retrieval for {code}")
            return None
    
    def get_high_frequency_data(self, code, interval='1min', count=120):
        """
        获取高频数据
        
        参数:
            code: 股票代码
            interval: 时间间隔 (1min, 5min, 15min, 30min, 60min)
            count: 获取的条数
            
        返回:
            DataFrame: 分钟级数据
        """
        if not self.api_available or self.status != "connected":
            self.logger.error("AKShare API not connected")
            return None
            
        try:
            import akshare as ak
            
            # 转换代码格式
            market = code[-2:].lower()
            symbol = code[:-3]
            
            if market == 'sh':
                full_code = f"sh{symbol}"
            elif market == 'sz':
                full_code = f"sz{symbol}"
            else:
                self.logger.error(f"Unsupported market in code: {code}")
                return None
            
            # 转换时间间隔
            period_map = {
                '1min': '1',
                '5min': '5',
                '15min': '15',
                '30min': '30',
                '60min': '60'
            }
            
            if interval not in period_map:
                self.logger.error(f"Unsupported interval: {interval}")
                return None
            
            # 获取分钟数据，使用不同函数处理指数和个股
            if symbol.startswith('0') or symbol.startswith('3'):  # 指数
                # AKShare对指数的分钟数据有专门接口
                df = ak.stock_zh_index_min(symbol=full_code, period=period_map[interval])
            else:
                # 个股分钟数据
                df = ak.stock_zh_a_minute(symbol=full_code, period=period_map[interval])
            
            if df is not None and not df.empty:
                # 规范化列名
                column_mappings = {
                    '日期': 'datetime',
                    '时间': 'time',
                    '开盘': 'open',
                    '收盘': 'close',
                    '最高': 'high',
                    '最低': 'low',
                    '成交量': 'volume',
                    '成交额': 'amount',
                    'datetime': 'datetime',
                    'open': 'open',
                    'close': 'close',
                    'high': 'high',
                    'low': 'low',
                    'volume': 'volume',
                    'amount': 'amount'
                }
                
                # 重命名列
                rename_dict = {k: v for k, v in column_mappings.items() if k in df.columns}
                if rename_dict:
                    df.rename(columns=rename_dict, inplace=True)
                
                # 处理日期时间
                if '日期' in df.columns and '时间' in df.columns:
                    df['datetime'] = pd.to_datetime(df['日期'] + ' ' + df['时间'])
                    df.set_index('datetime', inplace=True)
                elif 'datetime' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df.set_index('datetime', inplace=True)
                
                # 限制返回条数
                if len(df) > count:
                    df = df.iloc[-count:]
                
                # 排序
                df = df.sort_index()
                
                self.reset_error_count()
                self.last_success = datetime.now()
                return df
            else:
                self.logger.warning(f"No high frequency data returned for {code}")
                return None
        
        except Exception as e:
            self.handle_error(e, f"high frequency data retrieval for {code}")
            return None
    
    def get_market_status(self):
        """获取市场状态"""
        if not self.api_available:
            self.logger.error("AKShare API not connected")
            return None
            
        try:
            import akshare as ak
            
            # 获取当前日期
            now = datetime.now()
            today = now.strftime('%Y%m%d')
            
            # 尝试获取交易日历
            try:
                calendar_df = ak.tool_trade_date_hist_sina()
                
                if calendar_df is not None and not calendar_df.empty:
                    # 转换日期格式
                    calendar_df['trade_date'] = pd.to_datetime(calendar_df['trade_date']).dt.strftime('%Y%m%d')
                    
                    # 判断今天是否交易日
                    is_trading_day = today in calendar_df['trade_date'].values
                    
                    # 检查交易时间
                    market_open = False
                    if is_trading_day:
                        current_time = now.time()
                        morning_start = datetime.strptime('09:30', '%H:%M').time()
                        morning_end = datetime.strptime('11:30', '%H:%M').time()
                        afternoon_start = datetime.strptime('13:00', '%H:%M').time()
                        afternoon_end = datetime.strptime('15:00', '%H:%M').time()
                        
                        if (morning_start <= current_time <= morning_end) or \
                           (afternoon_start <= current_time <= afternoon_end):
                            market_open = True
                    
                    # 获取下一个交易日
                    next_trading_day = None
                    if is_trading_day:
                        # 如果今天是交易日，找到今天之后的第一个交易日
                        future_days = calendar_df[calendar_df['trade_date'] > today]
                        if not future_days.empty:
                            next_trading_day = future_days.iloc[0]['trade_date']
                    else:
                        # 如果今天不是交易日，找到今天之后的第一个交易日
                        future_days = calendar_df[calendar_df['trade_date'] > today]
                        if not future_days.empty:
                            next_trading_day = future_days.iloc[0]['trade_date']
                    
                    return {
                        "trading_day": is_trading_day,
                        "market_open": market_open,
                        "next_trading_day": next_trading_day,
                        "time": now.strftime("%Y-%m-%d %H:%M:%S")
                    }
            
            except Exception as e:
                self.logger.warning(f"获取交易日历失败: {e}")
            
            # 如果获取日历失败，使用简单判断法
            weekday = now.weekday()
            hour = now.hour
            
            # 周末不是交易日
            is_trading_day = weekday < 5
            
            # 交易时间判断
            market_open = False
            if is_trading_day:
                if (hour == 9 and now.minute >= 30) or (hour == 10) or \
                   (hour == 11 and now.minute <= 30) or (hour == 13) or \
                   (hour == 14):
                    market_open = True
            
            # 估算下一个交易日
            next_trading_day = None
            if weekday == 4:  # 周五
                next_trading_day = (now + timedelta(days=3)).strftime('%Y%m%d')
            elif weekday == 5:  # 周六
                next_trading_day = (now + timedelta(days=2)).strftime('%Y%m%d')
            else:
                next_trading_day = (now + timedelta(days=1)).strftime('%Y%m%d')
            
            return {
                "trading_day": is_trading_day,
                "market_open": market_open,
                "next_trading_day": next_trading_day,
                "time": now.strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            self.handle_error(e, "market status retrieval")
            return None
    
    def get_sector_data(self, date=None):
        """获取板块数据"""
        if not self.api_available:
            self.logger.error("AKShare API not connected")
            return None
            
        try:
            import akshare as ak
            
            # 设置默认日期为今天
            if date is None:
                date = datetime.now().strftime('%Y%m%d')
                
            # 获取行业板块数据
            industry_df = ak.stock_board_industry_name_em()
            
            # 转换为标准格式
            sectors = []
            if industry_df is not None and not industry_df.empty:
                for _, row in industry_df.iterrows():
                    sector = {
                        'name': row.get('板块名称', ''),
                        'change_pct': row.get('涨跌幅', 0) / 100,  # 转为小数
                        'price': row.get('最新价', 0),
                        'volume': row.get('成交量', 0),
                        'amount': row.get('成交额', 0),
                        'leader_stock': row.get('领涨股', ''),
                        'leader_pct': row.get('领涨股涨跌幅', 0) / 100  # 转为小数
                    }
                    sectors.append(sector)
                    
            return {
                'date': date,
                'sectors': sectors
            }
            
        except Exception as e:
            self.handle_error(e, "sector data retrieval")
            return None
    
    def get_stock_list(self, market=None):
        """获取股票列表"""
        if not self.api_available:
            self.logger.error("AKShare API not connected")
            return None
            
        try:
            import akshare as ak
            
            # 从缓存中读取
            cache_file = os.path.join(self.cache_dir, 'stock_list.json')
            if os.path.exists(cache_file):
                cache_time = os.path.getmtime(cache_file)
                if time.time() - cache_time < self.cache_ttl:  # 缓存未过期
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        stock_list = json.load(f)
                    return stock_list
            
            # 获取A股列表
            df = ak.stock_zh_a_spot_em()
            
            stock_list = []
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    code = row.get('代码', '')
                    name = row.get('名称', '')
                    
                    # 确定市场
                    if code.startswith('6'):
                        ts_code = f"{code}.SH"
                    else:
                        ts_code = f"{code}.SZ"
                    
                    stock = {
                        'ts_code': ts_code,
                        'symbol': code,
                        'name': name,
                        'industry': row.get('所处行业', ''),
                        'market': 'SH' if code.startswith('6') else 'SZ'
                    }
                    stock_list.append(stock)
            
            # 保存到缓存
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(stock_list, f, ensure_ascii=False)
            
            return stock_list
            
        except Exception as e:
            self.handle_error(e, "stock list retrieval")
            return None


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
                'akshare': {
                    'enabled': True,
                    'priority': 8
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
            elif source_name == 'akshare':
                source = AKShareDataSource(source_config)
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

    def get_high_frequency_data(self, code, interval='1min', count=120):
        """
        获取高频数据 - 支持分钟级数据
        
        参数:
            code: 股票或指数代码
            interval: 时间间隔 (1min, 5min, 15min, 30min, 60min)
            count: 返回的数据点数量
            
        返回:
            DataFrame: 高频数据
        """
        if not self.initialized:
            self.logger.error("Data connector not initialized")
            return None
            
        # 高频数据缓存时间较短
        cache_key = f"{code}_{interval}_{count}_hf"
        cache_ttl = 300  # 5分钟缓存时间
        
        # 检查缓存
        cache_file = os.path.join(self.cache_dir, f"hf_{cache_key}.pkl")
        if os.path.exists(cache_file):
            file_mtime = os.path.getmtime(cache_file)
            if time.time() - file_mtime < cache_ttl:
                try:
                    data = pd.read_pickle(cache_file)
                    self.logger.info(f"Loaded high frequency data from cache: {cache_key}")
                    return data
                except Exception as e:
                    self.logger.warning(f"Error loading cached high frequency data: {e}")
        
        # 遍历所有数据源尝试获取高频数据
        for source in sorted(self.data_sources.values(), key=lambda s: s.priority, reverse=True):
            if source.status == "connected" or source.check_connection():
                try:
                    # 直接调用数据源的高频数据接口 (如果存在)
                    if hasattr(source, 'get_high_frequency_data'):
                        data = source.get_high_frequency_data(code, interval, count)
                        if data is not None and not data.empty:
                            # 缓存数据
                            try:
                                data.to_pickle(cache_file)
                            except Exception as e:
                                self.logger.warning(f"Error caching high frequency data: {e}")
                            return data
                    else:
                        self.logger.debug(f"Source {source.name} does not support high frequency data")
                except Exception as e:
                    self.logger.warning(f"Error getting high frequency data from {source.name}: {e}")
                    continue
        
        self.logger.error(f"Failed to get high frequency data for {code} from all sources")
        return None
        
    def check_sources_health(self):
        """
        检查所有数据源的健康状态
        
        返回:
            dict: 数据源健康状态
        """
        health_status = {}
        
        for name, source in self.data_sources.items():
            is_connected = source.status == "connected"
            
            # 如果未连接，尝试重新连接
            if not is_connected:
                is_connected = source.check_connection()
                
            # 记录状态
            health_status[name] = {
                "connected": is_connected,
                "status": source.status,
                "last_error": str(source.last_error) if source.last_error else None,
                "error_count": source.error_count,
                "last_success": source.last_success.strftime("%Y-%m-%d %H:%M:%S") if source.last_success else None,
                "priority": source.priority
            }
            
        # 更新活跃数据源
        self._select_active_source()
        
        return health_status
        
    def get_concurrent(self, codes, start_date=None, end_date=None, period='daily'):
        """
        并发获取多个股票的数据
        
        参数:
            codes: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            period: 周期 (daily, weekly, monthly)
            
        返回:
            dict: 以股票代码为键的数据字典
        """
        if not self.initialized:
            self.logger.error("Data connector not initialized")
            return {}
            
        results = {}
        errors = []
        
        # 创建线程池
        from concurrent.futures import ThreadPoolExecutor
        
        def fetch_single(code):
            try:
                data = self.get_market_data(code, start_date, end_date, period)
                return code, data
            except Exception as e:
                self.logger.error(f"Error fetching data for {code}: {e}")
                errors.append((code, str(e)))
                return code, None
        
        # 使用线程池并发请求
        with ThreadPoolExecutor(max_workers=min(10, len(codes))) as executor:
            # 提交所有任务
            futures = [executor.submit(fetch_single, code) for code in codes]
            
            # 收集结果
            for future in futures:
                try:
                    code, data = future.result()
                    if data is not None:
                        results[code] = data
                except Exception as e:
                    self.logger.error(f"Thread error: {e}")
                    
        self.logger.info(f"Concurrent fetch completed: {len(results)} succeeded, {len(errors)} failed")
        
        if errors:
            self.logger.warning(f"Failed to fetch data for: {[e[0] for e in errors]}")
            
        return results
        
    def get_market_status(self):
        """
        获取市场状态
        
        返回:
            dict: 市场状态信息
        """
        # 默认状态
        status = {
            "trading_day": False,
            "market_open": False,
            "next_trading_day": None,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 尝试获取市场状态
        for source in sorted(self.data_sources.values(), key=lambda s: s.priority, reverse=True):
            if (source.status == "connected" or source.check_connection()) and hasattr(source, 'get_market_status'):
                try:
                    source_status = source.get_market_status()
                    if source_status:
                        status.update(source_status)
                        break
                except Exception as e:
                    self.logger.warning(f"Error getting market status from {source.name}: {e}")
        
        return status
        
    def get_data_source_status(self):
        """
        获取数据源状态
        
        返回:
            dict: 数据源状态
        """
        source_status = {}
        
        for name, source in self.data_sources.items():
            source_status[name] = {
                "connected": source.status == "connected",
                "active": source == self.active_source,
                "priority": source.priority
            }
            
        return source_status

# 使用示例
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 创建数据连接器
    connector = DataConnector()
    print("\n=== 数据连接器初始化 ===")
    print(f"初始化状态: {'成功' if connector.initialized else '失败'}")
    print(f"活跃数据源: {connector.active_source.name if connector.active_source else '无'}")
    
    # 数据源状态
    print("\n=== 数据源状态 ===")
    source_status = connector.get_data_source_status()
    for name, status in source_status.items():
        print(f"{name}: {'✓ 连接' if status['connected'] else '✗ 未连接'} " + 
              f"{'[活跃]' if status['active'] else ''} " +
              f"(优先级: {status['priority']})")
    
    # 获取市场状态
    print("\n=== 市场状态 ===")
    market_status = connector.get_market_status()
    print(f"当前时间: {market_status['time']}")
    print(f"交易日: {'是' if market_status['trading_day'] else '否'}")
    print(f"市场开盘: {'是' if market_status['market_open'] else '否'}")
    if market_status['next_trading_day']:
        print(f"下一交易日: {market_status['next_trading_day']}")
    
    # 获取沪深300指数数据
    print("\n=== 沪深300指数日线行情 ===")
    df = connector.get_market_data("000300.SH", period='daily')
    if df is not None:
        print(f"获取数据成功，共{df.shape[0]}条记录")
        print(df.head())
    else:
        print("获取数据失败")
    
    try:
        # 尝试获取高频数据
        print("\n=== 尝试获取上证指数分钟数据 ===")
        hf_df = connector.get_high_frequency_data("000001.SH", interval='5min', count=10)
        if hf_df is not None:
            print(f"获取高频数据成功，共{hf_df.shape[0]}条记录")
            print(hf_df.head())
        else:
            print("获取高频数据失败")
    except Exception as e:
        print(f"获取高频数据出错: {e}")
    
    # 尝试获取多个股票数据
    print("\n=== 尝试并发获取多个股票数据 ===")
    codes = ["000001.SZ", "600000.SH", "000300.SH", "000905.SH"]
    multi_data = connector.get_concurrent(codes)
    print(f"成功获取{len(multi_data)}个股票的数据")
    for code, data in multi_data.items():
        print(f"{code}: {data.shape[0]}条记录")
    
    # 数据源健康检查
    print("\n=== 数据源健康检查 ===")
    health = connector.check_sources_health()
    for name, status in health.items():
        print(f"{name}: 状态={status['status']} " + 
              f"连接={'成功' if status['connected'] else '失败'} " +
              f"错误次数={status['error_count']}")
        if status['last_success']:
            print(f"  最后成功时间: {status['last_success']}")
        if status['last_error']:
            print(f"  最后错误: {status['last_error']}") 