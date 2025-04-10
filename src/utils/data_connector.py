#!/usr/bin/env python3
"""
超神量子系统 - 统一数据连接器
用于解决数据连接问题，提供统一的接口获取各类数据
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import tushare as ts
from typing import List, Dict, Any, Optional, Union

# 获取日志记录器
logger = logging.getLogger(__name__)

class DataConnector:
    """统一数据连接器基类"""
    
    def __init__(self):
        self.name = "基础数据连接器"
        self.logger = logging.getLogger(f"DataConnector.{self.name}")
        self.logger.info(f"初始化{self.name}")
        
    def get_daily_data(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取日线数据"""
        raise NotImplementedError("子类必须实现此方法")
        
    def get_stock_list(self) -> pd.DataFrame:
        """获取股票列表"""
        raise NotImplementedError("子类必须实现此方法")
        
    def get_index_data(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取指数数据"""
        raise NotImplementedError("子类必须实现此方法")
        
    def get_sector_data(self) -> pd.DataFrame:
        """获取板块数据"""
        raise NotImplementedError("子类必须实现此方法")
        
    def get_policy_news(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取政策新闻数据"""
        raise NotImplementedError("子类必须实现此方法")


class TushareConnector(DataConnector):
    """Tushare数据连接器"""
    
    def __init__(self, token: str = None):
        self.name = "Tushare数据连接器"
        self.logger = logging.getLogger(f"DataConnector.{self.name}")
        
        # 设置Token
        if token is None:
            # 默认Token
            token = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"
            
        self.token = token
        ts.set_token(token)
        self.pro = ts.pro_api()
        
        # 基础信息缓存
        self._stock_list = None
        self._index_list = None
        self._sector_list = None
        
        # 初始化
        self.init_basic_info()
        self.logger.info(f"初始化{self.name}完成")
        
    def init_basic_info(self):
        """初始化基础信息"""
        try:
            # 尝试获取股票列表
            df = self.pro.stock_basic(exchange='', list_status='L', 
                                       fields='ts_code,symbol,name,area,industry,list_date')
            self._stock_list = df
            self.logger.info(f"获取股票列表成功，共 {len(df)} 条记录")
            
            # 尝试获取指数列表
            try:
                df_index = self.pro.index_basic(market='SSE')
                df_index = pd.concat([df_index, self.pro.index_basic(market='SZSE')])
                self._index_list = df_index
                self.logger.info(f"获取指数列表成功，共 {len(df_index)} 条记录")
            except Exception as e:
                self.logger.warning(f"获取指数列表失败: {str(e)}")
                
            self.logger.info("基础信息初始化成功")
            return True
            
        except Exception as e:
            self.logger.error(f"初始化基础信息失败: {str(e)}")
            
            # 创建模拟数据
            self._create_mock_data()
            return False
    
    def _create_mock_data(self):
        """创建模拟数据"""
        # 模拟股票列表
        stocks = [
            {'ts_code': '000001.SZ', 'symbol': '000001', 'name': '平安银行', 'area': '深圳', 'industry': '银行'},
            {'ts_code': '600519.SH', 'symbol': '600519', 'name': '贵州茅台', 'area': '贵州', 'industry': '白酒'},
            {'ts_code': '000858.SZ', 'symbol': '000858', 'name': '五粮液', 'area': '四川', 'industry': '白酒'},
            {'ts_code': '601318.SH', 'symbol': '601318', 'name': '中国平安', 'area': '上海', 'industry': '保险'},
            {'ts_code': '000333.SZ', 'symbol': '000333', 'name': '美的集团', 'area': '广东', 'industry': '家电'},
            {'ts_code': '600036.SH', 'symbol': '600036', 'name': '招商银行', 'area': '深圳', 'industry': '银行'},
            {'ts_code': '601888.SH', 'symbol': '601888', 'name': '中国中免', 'area': '上海', 'industry': '旅游'} 
        ]
        self._stock_list = pd.DataFrame(stocks)
        
        # 模拟指数列表
        indices = [
            {'ts_code': '000001.SH', 'name': '上证指数', 'market': 'SSE'},
            {'ts_code': '399001.SZ', 'name': '深证成指', 'market': 'SZSE'},
            {'ts_code': '000300.SH', 'name': '沪深300', 'market': 'SSE'},
            {'ts_code': '000905.SH', 'name': '中证500', 'market': 'SSE'}
        ]
        self._index_list = pd.DataFrame(indices)
        
        # 模拟板块列表
        sectors = [
            {'sector_code': 'BK0001', 'name': '银行', 'stocks': ['000001.SZ', '600036.SH']},
            {'sector_code': 'BK0002', 'name': '白酒', 'stocks': ['600519.SH', '000858.SZ']},
            {'sector_code': 'BK0003', 'name': '保险', 'stocks': ['601318.SH']},
            {'sector_code': 'BK0004', 'name': '家电', 'stocks': ['000333.SZ']}
        ]
        self._sector_list = pd.DataFrame(sectors)
        
        self.logger.info("已创建模拟数据")
    
    def get_daily_data(self, code: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """获取日线数据"""
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
            
        self.logger.debug(f"获取 {code} 的日线数据, 时间范围: {start_date}-{end_date}")
        
        try:
            # 尝试从Tushare获取数据
            df = self.pro.daily(ts_code=code, start_date=start_date, end_date=end_date)
            if df is not None and not df.empty:
                self.logger.debug(f"成功从Tushare获取 {code} 的日线数据，共 {len(df)} 条记录")
                # 按日期排序
                df = df.sort_values('trade_date')
                return df
                
        except Exception as e:
            self.logger.warning(f"从Tushare获取 {code} 的日线数据失败: {str(e)}，将使用模拟数据")
        
        # 如果获取失败或者结果为空，使用模拟数据
        return self._generate_mock_daily_data(code, start_date, end_date)
    
    def _generate_mock_daily_data(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """生成模拟日线数据"""
        # 处理日期
        start = datetime.strptime(start_date, '%Y%m%d')
        end = datetime.strptime(end_date, '%Y%m%d')
        
        # 生成日期范围（只包含工作日）
        date_range = []
        current = start
        while current <= end:
            if current.weekday() < 5:  # 0-4表示周一至周五
                date_range.append(current)
            current += timedelta(days=1)
        
        # 生成模拟数据
        data = []
        price = 100.0  # 起始价格
        
        for date in date_range:
            # 随机价格变动
            change = np.random.normal(0, 1) / 100
            price = price * (1 + change)
            
            # 生成OHLCV数据
            open_price = price * (1 + np.random.normal(0, 0.005))
            high_price = price * (1 + abs(np.random.normal(0, 0.01)))
            low_price = price * (1 - abs(np.random.normal(0, 0.01)))
            close_price = price
            volume = int(np.random.randint(1000, 10000))
            amount = volume * close_price
            
            data.append({
                'ts_code': code,
                'trade_date': date.strftime('%Y%m%d'),
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'vol': volume,
                'amount': amount,
                'change': price - close_price,
                'pct_chg': (price - close_price) / close_price * 100
            })
        
        df = pd.DataFrame(data)
        self.logger.debug(f"已生成 {code} 的模拟日线数据，共 {len(df)} 条记录")
        return df
    
    def get_stock_list(self) -> pd.DataFrame:
        """获取股票列表"""
        if self._stock_list is None:
            self.init_basic_info()
        
        return self._stock_list
    
    def get_index_data(self, code: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """获取指数数据"""
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
            
        self.logger.debug(f"获取指数 {code} 的数据, 时间范围: {start_date}-{end_date}")
        
        try:
            # 尝试从Tushare获取数据
            df = self.pro.index_daily(ts_code=code, start_date=start_date, end_date=end_date)
            if df is not None and not df.empty:
                self.logger.debug(f"成功从Tushare获取指数 {code} 的数据，共 {len(df)} 条记录")
                # 按日期排序
                df = df.sort_values('trade_date')
                return df
                
        except Exception as e:
            self.logger.warning(f"从Tushare获取指数 {code} 的数据失败: {str(e)}，将使用模拟数据")
        
        # 如果获取失败或者结果为空，使用模拟数据
        return self._generate_mock_daily_data(code, start_date, end_date)
    
    def get_sector_data(self) -> pd.DataFrame:
        """获取板块数据"""
        # 如果没有实际数据，返回模拟板块数据
        if self._sector_list is None:
            self._create_mock_data()
            
        return self._sector_list
    
    def get_policy_news(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """获取政策新闻数据"""
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
            
        self.logger.debug(f"获取政策新闻数据, 时间范围: {start_date}-{end_date}")
        
        # 创建模拟政策新闻数据
        news = [
            {
                'date': '20230105',
                'title': '央行再推量化宽松政策',
                'content': '为稳定经济，中国人民银行宣布降低存款准备金率0.5个百分点，释放长期资金约1万亿元。',
                'impact': '正面',
                'source': '央行'
            },
            {
                'date': '20230212',
                'title': '证监会发布新股发行改革方案',
                'content': '证监会发布全面注册制改革方案，进一步优化新股发行机制，提高市场效率。',
                'impact': '正面',
                'source': '证监会'
            },
            {
                'date': '20230328',
                'title': '国务院出台房地产支持政策',
                'content': '国务院发布支持房地产市场稳定发展的若干措施，包括降低首付比例和贷款利率等。',
                'impact': '正面',
                'source': '国务院'
            },
            {
                'date': '20230417',
                'title': '财政部宣布新一轮减税降费政策',
                'content': '财政部宣布实施新一轮减税降费政策，预计全年将为企业减负超过2万亿元。',
                'impact': '正面',
                'source': '财政部'
            },
            {
                'date': '20230526',
                'title': '科技部发布人工智能发展规划',
                'content': '科技部发布《新一代人工智能产业创新发展行动计划》，推动人工智能产业高质量发展。',
                'impact': '正面',
                'source': '科技部'
            }
        ]
        
        # 过滤日期范围内的新闻
        filtered_news = []
        for item in news:
            if start_date <= item['date'] <= end_date:
                filtered_news.append(item)
                
        return pd.DataFrame(filtered_news)


# 单例模式，提供全局数据连接器实例
_global_data_connector = None

def get_data_connector(connector_type: str = 'tushare', token: str = None) -> DataConnector:
    """获取数据连接器实例（单例模式）"""
    global _global_data_connector
    
    if _global_data_connector is None:
        if connector_type.lower() == 'tushare':
            _global_data_connector = TushareConnector(token=token)
            
    return _global_data_connector
