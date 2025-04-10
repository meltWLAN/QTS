#!/usr/bin/env python3
"""
超神量子共生系统 - 数据管理器
负责获取和管理市场数据
"""

import logging
import time
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# 设置日志
logger = logging.getLogger("TradingEngine.DataManager")

class DataManager:
    """数据管理器 - 负责获取和管理市场数据"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化数据管理器
        
        参数:
            config: 配置参数
        """
        self.config = config or {}
        
        # 数据源设置
        self.data_source_type = self.config.get("data_source_type", "tushare")
        self.data_sources = {}  # 数据源实例
        self.data_cache_dir = self.config.get("data_cache_dir", "data/cache")
        self.data_update_interval = self.config.get("data_update_interval", 60)  # 数据更新间隔(秒)
        self.max_cache_age = self.config.get("max_cache_age", 24 * 60 * 60)  # 最大缓存有效期(秒)
        self.use_cache = self.config.get("use_cache", True)
        
        # 市场数据缓存
        self.market_data = {}  # symbol -> DataFrame
        self.last_update_time = {}  # symbol -> timestamp
        self.update_in_progress = False
        
        # 股票池
        self.stock_pool = self.config.get("stock_pool", [])
        self.default_indices = ["000001.SH", "399001.SZ", "399005.SZ", "399006.SZ"]
        
        # 数据字段映射
        self.field_mapping = {
            "tushare": {
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "vol",
                "amount": "amount",
                "change": "pct_chg"
            },
            "eastmoney": {
                "open": "OPEN",
                "high": "HIGH",
                "low": "LOW",
                "close": "CLOSE",
                "volume": "VOLUME",
                "amount": "AMOUNT",
                "change": "CHANGE"
            }
        }
        
        # 创建缓存目录
        os.makedirs(self.data_cache_dir, exist_ok=True)
        
        # 初始化数据源
        self._initialize_data_sources()
        
        logger.info("数据管理器初始化完成")
    
    def _initialize_data_sources(self):
        """初始化数据源"""
        # 尝试初始化 Tushare 数据源
        if self.data_source_type == "tushare" or "tushare" in self.config.get("data_sources", []):
            self._initialize_tushare_data_source()
        
        # 尝试初始化 EastMoney 数据源
        if self.data_source_type == "eastmoney" or "eastmoney" in self.config.get("data_sources", []):
            self._initialize_eastmoney_data_source()
    
    def _initialize_tushare_data_source(self):
        """初始化 Tushare 数据源"""
        try:
            # 尝试导入 Tushare 数据连接器
            from tushare_data_connector import TushareDataConnector
            
            # 获取配置
            token = self.config.get("tushare_token", "")
            
            # 创建数据源实例
            tushare_connector = TushareDataConnector(token=token)
            
            # 检查连接状态
            if tushare_connector.check_connection():
                self.data_sources["tushare"] = tushare_connector
                logger.info("Tushare 数据源初始化成功")
            else:
                logger.warning("Tushare 数据源连接失败")
        except ImportError:
            logger.warning("未找到 Tushare 数据连接器模块")
    
    def _initialize_eastmoney_data_source(self):
        """初始化 EastMoney 数据源"""
        try:
            # 尝试导入 EastMoney 数据连接器
            from eastmoney_data_connector import EastMoneyDataConnector
            
            # 创建数据源实例
            eastmoney_connector = EastMoneyDataConnector()
            
            # 检查连接状态
            if eastmoney_connector.check_connection():
                self.data_sources["eastmoney"] = eastmoney_connector
                logger.info("EastMoney 数据源初始化成功")
            else:
                logger.warning("EastMoney 数据源连接失败")
        except ImportError:
            logger.warning("未找到 EastMoney 数据连接器模块")
    
    def get_market_data(self, symbols: List[str], 
                        start_date: Optional[str] = None, 
                        end_date: Optional[str] = None,
                        freq: str = "D",
                        refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """
        获取市场数据
        
        参数:
            symbols: 股票代码列表
            start_date: 开始日期，格式: 'YYYY-MM-DD'
            end_date: 结束日期，格式: 'YYYY-MM-DD'
            freq: 频率，'D'=日线，'W'=周线，'M'=月线，'60min'=60分钟线，'1min'=1分钟线
            refresh: 是否强制刷新
            
        返回:
            Dict[str, pd.DataFrame]: {股票代码: 数据DataFrame}
        """
        # 默认参数处理
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        if not start_date:
            # 默认获取60个交易日的数据
            start_date = (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d")
        
        # 检查每个股票是否需要更新
        symbols_to_update = []
        result_data = {}
        
        for symbol in symbols:
            # 检查缓存
            if not refresh and symbol in self.market_data:
                last_update = self.last_update_time.get(symbol, datetime(1970, 1, 1))
                if (datetime.now() - last_update).total_seconds() < self.data_update_interval:
                    # 数据足够新，直接使用缓存
                    result_data[symbol] = self.market_data[symbol]
                    continue
            
            # 需要更新的股票
            symbols_to_update.append(symbol)
        
        # 获取需要更新的股票数据
        if symbols_to_update:
            new_data = self._fetch_market_data(symbols_to_update, start_date, end_date, freq)
            
            # 合并结果
            result_data.update(new_data)
        
        return result_data
    
    def _fetch_market_data(self, symbols: List[str], start_date: str, end_date: str, freq: str) -> Dict[str, pd.DataFrame]:
        """
        从数据源获取市场数据
        
        参数:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            freq: 频率
            
        返回:
            Dict[str, pd.DataFrame]: {股票代码: 数据DataFrame}
        """
        results = {}
        
        # 标记更新进行中
        self.update_in_progress = True
        
        try:
            # 检查是否有可用数据源
            if not self.data_sources:
                logger.error("没有可用的数据源")
                return results
            
            # 选择主数据源
            data_source = self.data_sources.get(self.data_source_type)
            if not data_source:
                # 使用第一个可用的数据源
                data_source = list(self.data_sources.values())[0]
            
            # 字段映射
            field_map = self.field_mapping.get(self.data_source_type, {})
            
            # 使用线程池并行获取多个股票数据
            with ThreadPoolExecutor(max_workers=min(10, len(symbols))) as executor:
                future_to_symbol = {}
                
                for symbol in symbols:
                    # 检查缓存
                    cache_file = os.path.join(self.data_cache_dir, f"{symbol}_{freq}_{start_date}_{end_date}.csv")
                    use_cache = self.use_cache and os.path.exists(cache_file)
                    
                    if use_cache:
                        # 检查缓存文件年龄
                        file_age = time.time() - os.path.getmtime(cache_file)
                        if file_age > self.max_cache_age:
                            use_cache = False
                    
                    if use_cache:
                        # 从缓存加载
                        future = executor.submit(self._load_data_from_cache, cache_file)
                    else:
                        # 从数据源获取
                        future = executor.submit(self._fetch_symbol_data, data_source, symbol, start_date, end_date, freq)
                    
                    future_to_symbol[future] = symbol
                
                # 收集结果
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        data = future.result()
                        
                        if data is not None and not data.empty:
                            # 转换列名
                            if field_map and self.data_source_type in ["tushare", "eastmoney"]:
                                data = self._map_column_names(data, field_map)
                            
                            # 添加到结果
                            results[symbol] = data
                            
                            # 更新内部缓存
                            self.market_data[symbol] = data
                            self.last_update_time[symbol] = datetime.now()
                            
                            # 保存到缓存
                            if self.use_cache:
                                cache_file = os.path.join(self.data_cache_dir, f"{symbol}_{freq}_{start_date}_{end_date}.csv")
                                data.to_csv(cache_file)
                        else:
                            logger.warning(f"获取 {symbol} 数据失败或数据为空")
                    except Exception as e:
                        logger.error(f"处理 {symbol} 数据时出错: {str(e)}")
        
        except Exception as e:
            logger.error(f"获取市场数据时出错: {str(e)}")
        
        finally:
            # 更新完成
            self.update_in_progress = False
        
        return results
    
    def _fetch_symbol_data(self, data_source, symbol: str, start_date: str, end_date: str, freq: str) -> Optional[pd.DataFrame]:
        """
        获取单个股票数据
        
        参数:
            data_source: 数据源实例
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            freq: 频率
            
        返回:
            Optional[pd.DataFrame]: 数据DataFrame
        """
        try:
            # 检查是指数还是股票
            if symbol.endswith('.SH') or symbol.endswith('.SZ') and (symbol.startswith('000') or symbol.startswith('399')):
                # 指数
                if hasattr(data_source, 'get_index_data'):
                    data = data_source.get_index_data(symbol, start_date, end_date, freq)
                    return data
            
            # 股票
            if hasattr(data_source, 'get_daily_data'):
                data = data_source.get_daily_data(symbol, start_date, end_date, freq)
                return data
            
            logger.warning(f"数据源不支持获取 {symbol} 的数据")
            return None
        
        except Exception as e:
            logger.error(f"获取 {symbol} 数据时出错: {str(e)}")
            return None
    
    def _load_data_from_cache(self, cache_file: str) -> Optional[pd.DataFrame]:
        """
        从缓存加载数据
        
        参数:
            cache_file: 缓存文件路径
            
        返回:
            Optional[pd.DataFrame]: 数据DataFrame
        """
        try:
            data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            logger.debug(f"从缓存加载数据: {cache_file}")
            return data
        except Exception as e:
            logger.error(f"从缓存加载数据失败: {str(e)}")
            return None
    
    def _map_column_names(self, df: pd.DataFrame, field_map: Dict) -> pd.DataFrame:
        """
        映射列名
        
        参数:
            df: 数据DataFrame
            field_map: 字段映射
            
        返回:
            pd.DataFrame: 映射后的DataFrame
        """
        inv_map = {v: k for k, v in field_map.items()}
        df = df.rename(columns=inv_map)
        return df
    
    def get_real_time_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        获取实时行情
        
        参数:
            symbols: 股票代码列表
            
        返回:
            Dict[str, Dict]: {股票代码: 行情数据}
        """
        results = {}
        
        try:
            # 检查是否有可用数据源
            if not self.data_sources:
                logger.error("没有可用的数据源")
                return results
            
            # 选择主数据源
            data_source = self.data_sources.get(self.data_source_type)
            if not data_source:
                # 使用第一个可用的数据源
                data_source = list(self.data_sources.values())[0]
            
            # 检查是否支持实时行情
            if hasattr(data_source, 'get_realtime_quotes'):
                quotes = data_source.get_realtime_quotes(symbols)
                return quotes
            else:
                logger.warning("数据源不支持获取实时行情")
        
        except Exception as e:
            logger.error(f"获取实时行情时出错: {str(e)}")
        
        return results
    
    def get_stock_list(self, market: str = "A") -> pd.DataFrame:
        """
        获取股票列表
        
        参数:
            market: 市场类型，'A'=A股，'H'=港股，'US'=美股
            
        返回:
            pd.DataFrame: 股票列表
        """
        try:
            # 检查是否有可用数据源
            if not self.data_sources:
                logger.error("没有可用的数据源")
                return pd.DataFrame()
            
            # 选择主数据源
            data_source = self.data_sources.get(self.data_source_type)
            if not data_source:
                # 使用第一个可用的数据源
                data_source = list(self.data_sources.values())[0]
            
            # 检查是否支持获取股票列表
            if hasattr(data_source, 'get_stock_list'):
                stock_list = data_source.get_stock_list(market)
                return stock_list
            else:
                logger.warning("数据源不支持获取股票列表")
        
        except Exception as e:
            logger.error(f"获取股票列表时出错: {str(e)}")
        
        return pd.DataFrame()
    
    def get_index_list(self) -> pd.DataFrame:
        """
        获取指数列表
        
        返回:
            pd.DataFrame: 指数列表
        """
        try:
            # 检查是否有可用数据源
            if not self.data_sources:
                logger.error("没有可用的数据源")
                return pd.DataFrame()
            
            # 选择主数据源
            data_source = self.data_sources.get(self.data_source_type)
            if not data_source:
                # 使用第一个可用的数据源
                data_source = list(self.data_sources.values())[0]
            
            # 检查是否支持获取指数列表
            if hasattr(data_source, 'get_index_list'):
                index_list = data_source.get_index_list()
                return index_list
            else:
                logger.warning("数据源不支持获取指数列表")
        
        except Exception as e:
            logger.error(f"获取指数列表时出错: {str(e)}")
        
        return pd.DataFrame()
    
    def set_stock_pool(self, stock_pool: List[str]):
        """
        设置股票池
        
        参数:
            stock_pool: 股票代码列表
        """
        self.stock_pool = stock_pool
    
    def get_stock_pool(self) -> List[str]:
        """
        获取股票池
        
        返回:
            List[str]: 股票池列表
        """
        return self.stock_pool
    
    def update_stock_pool_data(self, start_date: Optional[str] = None, 
                               end_date: Optional[str] = None,
                               freq: str = "D",
                               include_indices: bool = True) -> Dict[str, pd.DataFrame]:
        """
        更新股票池数据
        
        参数:
            start_date: 开始日期
            end_date: 结束日期
            freq: 频率
            include_indices: 是否包含指数
            
        返回:
            Dict[str, pd.DataFrame]: 更新后的数据
        """
        symbols = self.stock_pool.copy()
        
        # 添加默认指数
        if include_indices:
            for index in self.default_indices:
                if index not in symbols:
                    symbols.append(index)
        
        # 获取数据
        return self.get_market_data(symbols, start_date, end_date, freq, refresh=True)
    
    def mock_market_data(self, symbols: List[str], days: int = 60) -> Dict[str, pd.DataFrame]:
        """
        生成模拟市场数据（用于测试）
        
        参数:
            symbols: 股票代码列表
            days: 天数
            
        返回:
            Dict[str, pd.DataFrame]: 模拟数据
        """
        mock_data = {}
        
        # 生成日期范围
        end_date = datetime.now()
        dates = pd.date_range(end=end_date, periods=days)
        
        # 为每个股票生成模拟数据
        for symbol in symbols:
            # 生成初始价格 (10-1000 之间)
            base_price = np.random.uniform(10, 1000)
            
            # 生成价格序列（添加一些随机波动和趋势）
            prices = np.zeros(days)
            prices[0] = base_price
            
            # 添加一些随机趋势
            trend = np.random.uniform(-0.0003, 0.0003)
            
            for i in range(1, days):
                # 日收益率，均值为trend，标准差为0.015
                daily_return = np.random.normal(trend, 0.015)
                prices[i] = prices[i-1] * (1 + daily_return)
            
            # 生成 OHLC 数据
            high = prices * (1 + np.random.uniform(0.005, 0.02, days))
            low = prices * (1 - np.random.uniform(0.005, 0.02, days))
            open_prices = low + np.random.uniform(0, 1, days) * (high - low)
            
            # 生成成交量
            volume = np.random.normal(1000000, 200000, days)
            volume = np.abs(volume)  # 确保成交量为正
            
            # 创建 DataFrame
            df = pd.DataFrame({
                'open': open_prices,
                'high': high,
                'low': low,
                'close': prices,
                'volume': volume
            }, index=dates)
            
            mock_data[symbol] = df
        
        return mock_data

# 测试函数
def test_data_manager():
    """测试数据管理器"""
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建数据管理器
    manager = DataManager({
        "data_source_type": "tushare",
        "use_cache": True
    })
    
    # 检查数据源
    print(f"可用数据源: {list(manager.data_sources.keys())}")
    
    # 测试获取模拟数据
    symbols = ["000001.SH", "600000.SH", "000001.SZ", "000063.SZ"]
    mock_data = manager.mock_market_data(symbols, days=30)
    
    for symbol, data in mock_data.items():
        print(f"\n{symbol} 模拟数据:")
        print(data.tail())
    
    # 设置股票池
    manager.set_stock_pool(symbols)
    
    # 测试更新股票池数据
    # 注意：实际应用中，这将从真实数据源获取数据
    # 在测试环境中，由于可能没有配置真实数据源，这可能会失败
    try:
        if manager.data_sources:
            print("\n更新股票池数据:")
            pool_data = manager.update_stock_pool_data()
            
            for symbol, data in pool_data.items():
                print(f"{symbol} 数据行数: {len(data)}")
    except Exception as e:
        print(f"更新股票池数据失败: {str(e)}")
        
    return manager, mock_data

if __name__ == "__main__":
    manager, data = test_data_manager() 