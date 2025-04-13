#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
real_data_connector - 量子核心组件
真实数据连接器 - 从多种来源获取市场真实数据
"""

import logging
import time
import json
import os
import requests
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

# 设置日志
logger = logging.getLogger('quantum_core.real_data_connector')

class RealDataConnector:
    """真实数据连接器 - 从多个数据源获取市场数据"""
    
    def __init__(self):
        """初始化连接器"""
        self.is_running = False
        self.api_keys = {}
        self.data_cache = {}
        self.cache_timestamps = {}
        self.default_cache_age = 3600  # 默认缓存1小时
        self.providers = ['tushare', 'yahoo', 'alphavantage', 'sina', 'eastmoney']
        self.preferred_provider = 'tushare'
        self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "market_cache")
        
        # 确保缓存目录存在
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info("真实数据连接器初始化完成")
        
    def start(self):
        """启动连接器"""
        try:
            logger.info("启动真实数据连接器...")
            self.is_running = True
            # 加载本地缓存
            self._load_cache()
            logger.info("真实数据连接器启动完成")
            return True
        except Exception as e:
            logger.error(f"启动真实数据连接器失败: {str(e)}")
            return False
            
    def stop(self):
        """停止连接器"""
        try:
            logger.info("停止真实数据连接器...")
            # 保存缓存
            self._save_cache()
            self.is_running = False
            logger.info("真实数据连接器已停止")
            return True
        except Exception as e:
            logger.error(f"停止真实数据连接器失败: {str(e)}")
            return False
            
    def set_api_key(self, provider: str, api_key: str):
        """设置数据源API密钥
        
        Args:
            provider: 数据提供者名称
            api_key: API密钥
        """
        self.api_keys[provider] = api_key
        logger.info(f"已设置 {provider} 的API密钥")
        
    def get_market_data(self, symbol: str, provider: Optional[str] = None, 
                      start_date: Optional[str] = None, end_date: Optional[str] = None,
                      use_cache: bool = True) -> Dict[str, Any]:
        """获取市场数据
        
        Args:
            symbol: 股票代码
            provider: 数据提供者，如果为None则使用默认提供者
            start_date: 开始日期，格式: YYYYMMDD
            end_date: 结束日期，格式: YYYYMMDD
            use_cache: 是否使用缓存
            
        Returns:
            市场数据字典
        """
        if not self.is_running:
            logger.warning("数据连接器未运行")
            return {'error': '数据连接器未运行'}
            
        # 确定数据提供者
        provider = provider or self.preferred_provider
        if provider not in self.providers:
            logger.warning(f"不支持的数据提供者: {provider}")
            return {'error': f"不支持的数据提供者: {provider}"}
            
        # 准备日期范围
        if not end_date:
            end_date = datetime.now().strftime('%Y%m%d')
            
        if not start_date:
            # 默认获取一年的数据
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
            
        # 检查缓存
        cache_key = f"{symbol}_{provider}_{start_date}_{end_date}"
        
        if use_cache and cache_key in self.data_cache:
            cache_age = time.time() - self.cache_timestamps.get(cache_key, 0)
            if cache_age < self.default_cache_age:
                logger.info(f"使用缓存数据: {cache_key}")
                return self.data_cache[cache_key]
                
        # 根据不同提供者获取数据
        try:
            if provider == 'tushare':
                data = self._get_from_tushare(symbol, start_date, end_date)
            elif provider == 'yahoo':
                data = self._get_from_yahoo(symbol, start_date, end_date)
            elif provider == 'alphavantage':
                data = self._get_from_alphavantage(symbol, start_date, end_date)
            elif provider == 'sina':
                data = self._get_from_sina(symbol, start_date, end_date)
            elif provider == 'eastmoney':
                data = self._get_from_eastmoney(symbol, start_date, end_date)
            else:
                return {'error': f"未实现的数据提供者: {provider}"}
                
            # 添加元数据
            data.update({
                'symbol': symbol,
                'provider': provider,
                'start_date': start_date,
                'end_date': end_date,
                'fetch_time': time.time()
            })
            
            # 更新缓存
            self.data_cache[cache_key] = data
            self.cache_timestamps[cache_key] = time.time()
            
            # 定期保存缓存
            if len(self.data_cache) % 10 == 0:
                self._save_cache()
                
            return data
            
        except Exception as e:
            logger.error(f"获取市场数据失败 ({provider}): {str(e)}")
            return {'error': f"获取数据失败: {str(e)}"}
            
    def _get_from_tushare(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """从TuShare获取数据"""
        try:
            # 尝试导入tushare
            import tushare as ts
            
            # 设置API密钥
            api_key = self.api_keys.get('tushare')
            if api_key:
                ts.set_token(api_key)
                
            # 获取数据
            pro = ts.pro_api()
            
            # 转换日期格式
            start = start_date
            end = end_date
            
            # 获取日线数据
            df = pro.daily(ts_code=symbol, start_date=start, end_date=end)
            
            # 检查数据
            if df.empty:
                logger.warning(f"TuShare未返回数据: {symbol}")
                return {'error': '未找到数据'}
                
            # 按日期升序排序
            df = df.sort_values('trade_date')
            
            # 转换为标准格式
            dates = df['trade_date'].tolist()
            opens = df['open'].tolist()
            highs = df['high'].tolist()
            lows = df['low'].tolist()
            closes = df['close'].tolist()
            volumes = df['vol'].tolist()
            
            return {
                'dates': dates,
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'volume': volumes
            }
            
        except ImportError:
            logger.error("未找到TuShare库，请安装: pip install tushare")
            return {'error': '未找到TuShare库，请安装: pip install tushare'}
        except Exception as e:
            logger.error(f"TuShare数据获取失败: {str(e)}")
            return {'error': f"TuShare数据获取失败: {str(e)}"}
            
    def _get_from_yahoo(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """从Yahoo Finance获取数据"""
        try:
            # 尝试导入yfinance
            import yfinance as yf
            
            # 调整股票代码格式
            yahoo_symbol = self._convert_to_yahoo_symbol(symbol)
            
            # 转换日期格式
            start = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]}"
            end = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]}"
            
            # 获取数据
            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(start=start, end=end)
            
            # 检查数据
            if df.empty:
                logger.warning(f"Yahoo Finance未返回数据: {yahoo_symbol}")
                return {'error': '未找到数据'}
                
            # 转换为标准格式
            dates = [d.strftime('%Y%m%d') for d in df.index.tolist()]
            opens = df['Open'].tolist()
            highs = df['High'].tolist()
            lows = df['Low'].tolist()
            closes = df['Close'].tolist()
            volumes = df['Volume'].tolist()
            
            return {
                'dates': dates,
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'volume': volumes
            }
            
        except ImportError:
            logger.error("未找到yfinance库，请安装: pip install yfinance")
            return {'error': '未找到yfinance库，请安装: pip install yfinance'}
        except Exception as e:
            logger.error(f"Yahoo Finance数据获取失败: {str(e)}")
            return {'error': f"Yahoo Finance数据获取失败: {str(e)}"}
            
    def _get_from_alphavantage(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """从Alpha Vantage获取数据"""
        try:
            # 获取API密钥
            api_key = self.api_keys.get('alphavantage')
            if not api_key:
                return {'error': 'Alpha Vantage需要API密钥'}
                
            # 构建API URL
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}&outputsize=full"
            
            # 发送请求
            response = requests.get(url)
            data = response.json()
            
            # 检查错误
            if "Error Message" in data:
                return {'error': data["Error Message"]}
                
            # 提取数据
            time_series = data.get("Time Series (Daily)", {})
            
            if not time_series:
                return {'error': '未找到数据'}
                
            # 转换为标准格式
            dates = []
            opens = []
            highs = []
            lows = []
            closes = []
            volumes = []
            
            # 转换日期为datetime对象以方便过滤
            start_dt = datetime.strptime(start_date, '%Y%m%d')
            end_dt = datetime.strptime(end_date, '%Y%m%d')
            
            for date_str, values in time_series.items():
                date_dt = datetime.strptime(date_str, '%Y-%m-%d')
                
                # 检查日期范围
                if start_dt <= date_dt <= end_dt:
                    dates.append(date_dt.strftime('%Y%m%d'))
                    opens.append(float(values['1. open']))
                    highs.append(float(values['2. high']))
                    lows.append(float(values['3. low']))
                    closes.append(float(values['4. close']))
                    volumes.append(int(float(values['5. volume'])))
                    
            # 按日期升序排序
            sorted_data = sorted(zip(dates, opens, highs, lows, closes, volumes), 
                               key=lambda x: x[0])
                               
            # 解压排序后的数据
            dates, opens, highs, lows, closes, volumes = zip(*sorted_data) if sorted_data else ([], [], [], [], [], [])
            
            return {
                'dates': list(dates),
                'open': list(opens),
                'high': list(highs),
                'low': list(lows),
                'close': list(closes),
                'volume': list(volumes)
            }
            
        except Exception as e:
            logger.error(f"Alpha Vantage数据获取失败: {str(e)}")
            return {'error': f"Alpha Vantage数据获取失败: {str(e)}"}
            
    def _get_from_sina(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """从新浪财经API获取数据"""
        try:
            # 转换股票代码为新浪格式
            sina_symbol = self._convert_to_sina_symbol(symbol)
            
            # 构建API URL (新浪历史数据API)
            url = f"https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData?symbol={sina_symbol}&scale=240&ma=no&datalen=100"
            
            # 发送请求
            response = requests.get(url)
            data = response.json()
            
            if not data:
                return {'error': '未找到数据'}
                
            # 转换为标准格式
            dates = []
            opens = []
            highs = []
            lows = []
            closes = []
            volumes = []
            
            # 转换日期为datetime对象以方便过滤
            start_dt = datetime.strptime(start_date, '%Y%m%d')
            end_dt = datetime.strptime(end_date, '%Y%m%d')
            
            for item in data:
                date_str = item['day']
                date_dt = datetime.strptime(date_str, '%Y-%m-%d')
                
                # 检查日期范围
                if start_dt <= date_dt <= end_dt:
                    dates.append(date_dt.strftime('%Y%m%d'))
                    opens.append(float(item['open']))
                    highs.append(float(item['high']))
                    lows.append(float(item['low']))
                    closes.append(float(item['close']))
                    volumes.append(int(float(item['volume'])))
                    
            return {
                'dates': dates,
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'volume': volumes
            }
            
        except Exception as e:
            logger.error(f"新浪财经数据获取失败: {str(e)}")
            return {'error': f"新浪财经数据获取失败: {str(e)}"}
            
    def _get_from_eastmoney(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """从东方财富获取数据"""
        try:
            # 转换股票代码为东方财富格式
            market_id = "0" if symbol.startswith(("0", "3")) else "1"
            secid = f"{market_id}.{symbol}"
            
            # 转换日期格式
            start_timestamp = int(datetime.strptime(start_date, '%Y%m%d').timestamp() * 1000)
            end_timestamp = int(datetime.strptime(end_date, '%Y%m%d').timestamp() * 1000)
            
            # 构建API URL
            url = f"http://push2his.eastmoney.com/api/qt/stock/kline/get?secid={secid}&fields1=f1,f2,f3,f4,f5&fields2=f51,f52,f53,f54,f55,f56,f57,f58&klt=101&fqt=0&beg={start_date}&end={end_date}"
            
            # 发送请求
            response = requests.get(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            data = response.json()
            
            # 检查数据
            if data.get('data') is None or data['data'].get('klines') is None:
                return {'error': '未找到数据'}
                
            klines = data['data']['klines']
            
            # 转换为标准格式
            dates = []
            opens = []
            highs = []
            lows = []
            closes = []
            volumes = []
            
            for kline in klines:
                parts = kline.split(',')
                date_str = parts[0].replace('-', '')
                dates.append(date_str)
                opens.append(float(parts[1]))
                closes.append(float(parts[2]))
                highs.append(float(parts[3]))
                lows.append(float(parts[4]))
                volumes.append(float(parts[5]))
                
            return {
                'dates': dates,
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'volume': volumes
            }
            
        except Exception as e:
            logger.error(f"东方财富数据获取失败: {str(e)}")
            return {'error': f"东方财富数据获取失败: {str(e)}"}
            
    def _convert_to_yahoo_symbol(self, symbol: str) -> str:
        """转换股票代码为Yahoo格式"""
        if symbol.isdigit():
            # 处理中国股票代码
            if symbol.startswith('6'):
                return f"{symbol}.SS"  # 上海
            elif symbol.startswith(('0', '3')):
                return f"{symbol}.SZ"  # 深圳
            else:
                return symbol
        return symbol
        
    def _convert_to_sina_symbol(self, symbol: str) -> str:
        """转换股票代码为新浪格式"""
        if symbol.isdigit():
            # 处理中国股票代码
            if symbol.startswith('6'):
                return f"sh{symbol}"  # 上海
            elif symbol.startswith(('0', '3')):
                return f"sz{symbol}"  # 深圳
            else:
                return symbol
        return symbol
        
    def _load_cache(self):
        """加载缓存数据"""
        try:
            cache_file = os.path.join(self.cache_dir, "market_data_cache.json")
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    self.data_cache = cache_data.get('data', {})
                    self.cache_timestamps = cache_data.get('timestamps', {})
                    logger.info(f"已加载{len(self.data_cache)}个缓存项")
        except Exception as e:
            logger.error(f"加载缓存失败: {str(e)}")
            
    def _save_cache(self):
        """保存缓存数据"""
        try:
            cache_file = os.path.join(self.cache_dir, "market_data_cache.json")
            
            # 仅保留最近的100个缓存项
            if len(self.data_cache) > 100:
                # 按时间戳排序
                sorted_keys = sorted(self.cache_timestamps.items(), 
                                  key=lambda x: x[1], reverse=True)
                # 保留前100个
                keep_keys = [k for k, _ in sorted_keys[:100]]
                # 筛选缓存
                self.data_cache = {k: v for k, v in self.data_cache.items() 
                                if k in keep_keys}
                self.cache_timestamps = {k: v for k, v in self.cache_timestamps.items() 
                                      if k in keep_keys}
                
            # 保存缓存
            cache_data = {
                'data': self.data_cache,
                'timestamps': self.cache_timestamps,
                'saved_at': time.time()
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f)
                
            logger.info(f"已保存{len(self.data_cache)}个缓存项")
            
        except Exception as e:
            logger.error(f"保存缓存失败: {str(e)}")
            
    def clear_cache(self):
        """清除所有缓存"""
        self.data_cache.clear()
        self.cache_timestamps.clear()
        logger.info("已清除所有缓存")
        
    def get_available_providers(self) -> List[str]:
        """获取所有可用的数据提供者"""
        return self.providers
        
    def set_preferred_provider(self, provider: str):
        """设置首选数据提供者"""
        if provider in self.providers:
            self.preferred_provider = provider
            logger.info(f"已设置首选数据提供者: {provider}")
        else:
            logger.warning(f"未知的数据提供者: {provider}")
            
    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """获取股票基本信息
        
        Args:
            symbol: 股票代码
            
        Returns:
            股票信息字典
        """
        try:
            # 尝试从TuShare获取
            if 'tushare' in self.api_keys:
                import tushare as ts
                ts.set_token(self.api_keys['tushare'])
                pro = ts.pro_api()
                
                # 获取股票基本信息
                df = pro.stock_basic(ts_code=symbol, fields='ts_code,name,area,industry,list_date')
                
                if not df.empty:
                    row = df.iloc[0]
                    return {
                        'code': symbol,
                        'name': row['name'],
                        'area': row['area'],
                        'industry': row['industry'],
                        'list_date': row['list_date'],
                        'source': 'tushare'
                    }
            
            # 如果TuShare失败，尝试从东方财富获取
            market_id = "0" if symbol.startswith(("0", "3")) else "1"
            secid = f"{market_id}.{symbol}"
            
            url = f"http://push2.eastmoney.com/api/qt/stock/get?secid={secid}&fields=f57,f58,f84,f85,f86,f1,f2,f3,f4,f5"
            response = requests.get(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            data = response.json()
            
            if data and 'data' in data:
                stock_data = data['data']
                return {
                    'code': symbol,
                    'name': stock_data.get('f58', ''),
                    'industry': stock_data.get('f127', ''),
                    'source': 'eastmoney'
                }
                
            return {'error': '无法获取股票信息'}
            
        except Exception as e:
            logger.error(f"获取股票信息失败: {str(e)}")
            return {'error': f"获取股票信息失败: {str(e)}"} 