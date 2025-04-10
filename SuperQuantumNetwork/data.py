#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据处理器模块
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import tushare as ts
import pandas as pd
import logging
from event import MarketDataEvent, EventType

logger = logging.getLogger(__name__)

class DataHandler(ABC):
    def __init__(self, symbols: List[str], start_date: str, end_date: str):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.data = {}
        self.current_index = 0
        self.dates = []

    @abstractmethod
    def get_latest_data(self, symbol: str) -> Dict[str, Any]:
        """获取最新的市场数据"""
        pass

    @abstractmethod
    def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """获取历史市场数据"""
        pass

    def get_next_data(self) -> Optional[MarketDataEvent]:
        """获取下一个市场数据事件"""
        if self.current_index >= len(self.dates):
            logger.debug("已经处理完所有数据")
            return None

        current_date = self.dates[self.current_index]
        data = {}
        
        for symbol in self.symbols:
            if symbol in self.data and current_date in self.data[symbol]:
                data[symbol] = self.data[symbol][current_date]

        if not data:
            logger.debug(f"在 {current_date} 没有找到任何数据")
            self.current_index += 1
            return self.get_next_data()

        logger.debug(f"生成市场数据事件: {current_date}")
        for symbol in data:
            logger.debug(f"- {symbol}: 开盘={data[symbol]['open']:.2f}, "
                        f"最高={data[symbol]['high']:.2f}, "
                        f"最低={data[symbol]['low']:.2f}, "
                        f"收盘={data[symbol]['close']:.2f}, "
                        f"成交量={data[symbol]['volume']:.0f}")

        event = MarketDataEvent(
            type=EventType.MARKET_DATA,
            timestamp=current_date,
            data=data
        )
        
        self.current_index += 1
        return event

    def reset(self):
        """重置数据处理器状态"""
        self.current_index = 0
        logger.info("数据处理器已重置")

    def get_historical_data(self, symbol: str, days: int = 10) -> List[Dict[str, Any]]:
        """获取指定天数的历史数据"""
        if symbol not in self.data:
            return []
            
        if self.current_index <= 0:
            return []
            
        # 获取当前日期
        end_idx = self.current_index if self.current_index < len(self.dates) else len(self.dates) - 1
        end_date = self.dates[end_idx]
        
        # 确定开始的索引
        start_idx = max(0, end_idx - days)
        
        # 获取数据
        result = []
        for i in range(start_idx, end_idx + 1):
            if i >= len(self.dates):
                break
                
            date = self.dates[i]
            if symbol in self.data and date in self.data[symbol]:
                result.append(self.data[symbol][date])
                
        return result

    def get_data_for_date(self, date: datetime) -> Dict[str, Dict[str, Any]]:
        """获取指定日期的市场数据"""
        data = {}
        for symbol in self.symbols:
            if symbol in self.data and date in self.data[symbol]:
                data[symbol] = self.data[symbol][date]
                
        return data

class TushareDataHandler(DataHandler):
    def __init__(self, symbols: List[str], start_date: str, end_date: str, token: str = None):
        super().__init__(symbols, start_date, end_date)
        self.token = token or "your_tushare_token"  # 替换为你的token
        ts.set_token(self.token)
        self.pro = ts.pro_api()
        # 区分指数和股票
        self.index_symbols = [s for s in symbols if self._is_index_symbol(s)]
        self.stock_symbols = [s for s in symbols if not self._is_index_symbol(s)]
        logger.info(f"加载 {len(self.index_symbols)} 个指数 和 {len(self.stock_symbols)} 只股票")
        self._load_data()

    def _is_index_symbol(self, symbol):
        """判断是否是指数代码"""
        index_patterns = [
            '000001.SH',  # 上证指数
            '399001.SZ',  # 深证成指
            '000300.SH',  # 沪深300
            '000905.SH',  # 中证500
            '000016.SH',  # 上证50
            '399006.SZ',  # 创业板指
            '000688.SH'   # 科创50
        ]
        # 直接匹配特定指数代码
        if symbol in index_patterns:
            return True
        # 通过后缀判断
        if symbol.endswith('.SH') or symbol.endswith('.SZ'):
            symbol_code = symbol.split('.')[0]
            # 上证指数以000开头，深证指数以399开头
            if symbol_code.startswith('000') or symbol_code.startswith('399'):
                return True
        return False

    def _load_data(self):
        """加载历史数据"""
        successful_symbols = []
        
        # 先加载股票数据
        for symbol in self.stock_symbols:
            try:
                # 获取日线数据
                df = self.pro.daily(ts_code=symbol, 
                                  start_date=self.start_date, 
                                  end_date=self.end_date)
                
                if df is None or df.empty:
                    logger.warning(f"无法获取股票 {symbol} 的数据")
                    continue

                # 转换日期格式
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                
                # 按日期排序
                df = df.sort_values('trade_date')
                
                # 存储数据
                self.data[symbol] = {}
                for _, row in df.iterrows():
                    self.data[symbol][row['trade_date']] = {
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': float(row['vol']),
                        'amount': float(row['amount'])
                    }

                # 更新日期列表
                if not self.dates:
                    self.dates = sorted(df['trade_date'].unique())
                else:
                    self.dates = sorted(set(self.dates) | set(df['trade_date'].unique()))

                successful_symbols.append(symbol)
                logger.info(f"成功加载股票 {symbol} 的数据，共 {len(df)} 条记录")

            except Exception as e:
                logger.error(f"加载股票 {symbol} 数据时发生错误: {str(e)}")
        
        # 然后加载指数数据
        for symbol in self.index_symbols:
            try:
                # 使用index_daily接口获取指数数据
                df = self.pro.index_daily(ts_code=symbol, 
                                        start_date=self.start_date, 
                                        end_date=self.end_date)
                
                if df is None or df.empty:
                    logger.warning(f"无法获取指数 {symbol} 的数据，可能是积分不足或API权限限制")
                    continue

                # 转换日期格式
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                
                # 按日期排序
                df = df.sort_values('trade_date')
                
                # 存储数据
                self.data[symbol] = {}
                for _, row in df.iterrows():
                    self.data[symbol][row['trade_date']] = {
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': float(row['vol']),
                        'amount': float(row['amount']) if 'amount' in row else 0.0
                    }

                # 更新日期列表
                if not self.dates:
                    self.dates = sorted(df['trade_date'].unique())
                else:
                    self.dates = sorted(set(self.dates) | set(df['trade_date'].unique()))

                successful_symbols.append(symbol)
                logger.info(f"成功加载指数 {symbol} 的数据，共 {len(df)} 条记录")

            except Exception as e:
                logger.error(f"加载指数 {symbol} 数据时发生错误: {str(e)}")

        # 更新symbols列表，只保留成功加载的股票和指数
        self.symbols = successful_symbols
        
        if not self.symbols:
            raise ValueError("没有成功加载任何股票或指数数据")

        logger.info(f"数据加载完成，共加载 {len(self.symbols)} 只股票/指数的数据")

    def get_latest_data(self, symbol: str) -> Dict[str, Any]:
        """获取最新的市场数据"""
        if symbol not in self.data or not self.data[symbol]:
            return None
        
        latest_date = max(self.data[symbol].keys())
        return self.data[symbol][latest_date]

    def get_historical_data(self, symbol: str, start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]:
        """获取历史市场数据"""
        if isinstance(start_date, int):
            # 如果start_date是整数，表示获取最近N天的数据
            days = start_date
            
            if symbol not in self.data:
                logger.warning(f"找不到 {symbol} 的数据")
                return []
                
            # 直接获取按日期排序的所有数据，然后取最后N天
            all_dates = sorted(self.data[symbol].keys())
            if len(all_dates) == 0:
                return []
                
            # 取最近N天的数据
            target_dates = all_dates[-min(days, len(all_dates)):]
            
            # 组装数据返回
            historical_data = [self.data[symbol][date] for date in target_dates]
            
            logger.debug(f"根据天数请求获取 {symbol} 最近 {days} 天的历史数据，找到 {len(historical_data)} 条记录")
            return historical_data
            
        if symbol not in self.data:
            logger.warning(f"找不到 {symbol} 的数据")
            return []

        if start_date and end_date:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            
            logger.debug(f"获取 {symbol} 从 {start} 到 {end} 的历史数据")
            
            historical_data = []
            for date in sorted(self.data[symbol].keys()):
                if start <= date <= end:
                    historical_data.append(self.data[symbol][date])
        else:
            # 返回所有历史数据
            historical_data = [data for date, data in sorted(self.data[symbol].items())]
        
        logger.debug(f"找到 {len(historical_data)} 条历史数据")
        
        if len(historical_data) > 0:
            logger.debug(f"第一条数据: {historical_data[0]}")
            logger.debug(f"最后一条数据: {historical_data[-1]}")
        
        return historical_data

    def get_data_for_date(self, date: datetime) -> Dict[str, Dict[str, Any]]:
        """获取指定日期的市场数据"""
        return super().get_data_for_date(date)

    def reset(self):
        """重置数据处理器状态"""
        super().reset()
        logger.info("Tushare数据处理器已重置") 