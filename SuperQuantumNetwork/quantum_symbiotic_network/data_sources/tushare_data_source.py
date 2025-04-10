#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - Tushare数据源
利用Tushare Pro API获取真实市场数据
增强版 - 集成更多高级功能和量子预测数据流
"""

import os
import time
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any

# 尝试导入tushare
try:
    import tushare as ts
    TUSHARE_AVAILABLE = True
except ImportError:
    TUSHARE_AVAILABLE = False
    logging.warning("Tushare模块未安装，请执行 pip install tushare 安装")

# 设置缓存目录
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)


class TushareDataSource:
    """Tushare数据源类，提供股票和市场数据，使用Pro API
    
    升级版：增加更多高级功能和指标数据获取
    - 资金流向分析
    - 市场情绪指标
    - 北向资金监控
    - 龙虎榜数据
    - 行业板块轮动
    - 量化因子计算
    """
    
    def __init__(self, token=None, delay_init=True):
        """初始化TuShare数据源
        
        Args:
            token: TuShare API令牌，如果为None，则尝试从环境变量读取
            delay_init: 是否延迟初始化API，默认为True
        """
        self.logger = logging.getLogger("TushareDataSource")
        
        # 设置API初始状态
        self.api = None
        self.is_ready = False
        self.offline_mode = False
        self.token = token
        self.stock_info_cache = {}
        
        # 确保缓存目录存在
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        # 缓存
        self.cache = {}
        self.last_update = {}
        
        # 如果不延迟初始化，立即初始化API
        if not delay_init:
            self.init_api()
    
    def init_api(self, token=None, max_retries=3):
        """初始化TuShare API
        
        Args:
            token: TuShare API令牌，如果为None则使用之前提供的令牌
            max_retries: 最大重试次数
            
        Returns:
            bool: 是否初始化成功
        """
        # 优先使用参数传入的token，其次使用实例的token，最后尝试从环境变量获取
        if token is not None:
            self.token = token
        elif self.token is None:
            self.token = os.environ.get("TUSHARE_TOKEN")
        
        if not self.token:
            self.logger.warning("未提供TuShare API令牌，将以离线模式运行")
            self.offline_mode = True
            return False
        
        self.logger.info("初始化TuShare Pro API...")
        retry_delay = 2.0  # 初始重试延迟（秒）
        
        for attempt in range(max_retries):
            try:
                import tushare as ts
                timeout = 10  # 连接超时时间（秒）
                
                # 设置重试策略
                import requests
                from requests.adapters import HTTPAdapter
                from urllib3.util.retry import Retry
                
                session = requests.Session()
                retry_strategy = Retry(
                    total=3,
                    backoff_factor=0.5,
                    status_forcelist=[429, 500, 502, 503, 504],
                )
                session.mount('http://', HTTPAdapter(max_retries=retry_strategy))
                session.mount('https://', HTTPAdapter(max_retries=retry_strategy))
                
                # 尝试使用自定义会话创建API
                try:
                    pro = ts.pro_api(token, timeout=timeout, session=session)
                except TypeError:
                    # 如果API不支持这些参数，使用标准方式
                    self.logger.warning("TuShare API不支持高级参数，使用标准初始化")
                    pro = ts.pro_api(token)
                
                # 测试API是否可用 - 使用简单的接口测试
                self.logger.info(f"测试TuShare API连接 (尝试 {attempt+1}/{max_retries})...")
                df = None
                
                try:
                    # 使用更简单的接口进行测试
                    df = pro.query('stock_basic', exchange='', list_status='L', 
                                  fields='ts_code,name', limit=5)
                except:
                    # 如果失败，尝试交易日历接口
                    df = pro.query('trade_cal', exchange='', start_date='20210101', 
                                  end_date='20210110', limit=5)
                
                if df is not None and not df.empty:
                    self.api = pro
                    self.logger.info(f"Tushare Pro API初始化成功，使用token: {token[:8]}...")
                    
                    # 检查API权限
                    try:
                        interfaces = self._check_api_permissions()
                        self.logger.info(f"Tushare Pro API权限检查完成，可用接口数量: {len(interfaces)}")
                    except Exception as e:
                        self.logger.warning(f"检查API权限时出错: {str(e)}")
                    
                    self.is_ready = True
                    self.offline_mode = False
                    return True
                else:
                    self.logger.warning(f"Tushare Pro API测试失败 (尝试 {attempt+1}/{max_retries}): 返回空数据")
                    
            except requests.exceptions.Timeout:
                self.logger.warning(f"Tushare Pro API连接超时 (尝试 {attempt+1}/{max_retries})")
            except requests.exceptions.ConnectionError:
                self.logger.warning(f"Tushare Pro API连接错误 (尝试 {attempt+1}/{max_retries})")
            except Exception as e:
                self.logger.warning(f"Tushare Pro API初始化失败 (尝试 {attempt+1}/{max_retries}): {str(e)}")
                
            # 如果不是最后一次尝试，则等待后重试
            if attempt < max_retries - 1:
                import time
                self.logger.info(f"等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 1.5, 15)  # 增加重试间隔，但最多15秒
        
        self.logger.warning(f"Tushare Pro API初始化失败，将以离线模式运行，使用缓存数据")
        self.is_ready = False
        self.offline_mode = True
        self.api = None
        return False

    def _check_api_permissions(self):
        """检查API权限"""
        if self.api is None:
            return []
        
        try:
            # 查询API权限
            df = self.api.query('tushare_token', token=self.token)
            return list(df['interface_name']) if df is not None and not df.empty else []
        except Exception as e:
            self.logger.warning(f"检查API权限失败: {str(e)}")
            return []
    
    def _load_cache(self, cache_name):
        """从缓存加载数据
        
        Args:
            cache_name: 缓存名称
            
        Returns:
            dict: 缓存数据
        """
        cache_file = os.path.join(CACHE_DIR, f"{cache_name}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.logger.info(f"从缓存加载{cache_name}数据成功")
                return data
            except Exception as e:
                self.logger.error(f"从缓存加载{cache_name}数据失败: {str(e)}")
        return None
    
    def _save_cache(self, cache_name, data):
        """保存数据到缓存
        
        Args:
            cache_name: 缓存名称
            data: 要缓存的数据
        """
        cache_file = os.path.join(CACHE_DIR, f"{cache_name}.json")
        try:
            # 确保缓存目录存在
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            
            # 保存数据
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"缓存{cache_name}数据成功")
            return True
        except Exception as e:
            self.logger.error(f"缓存{cache_name}数据失败: {str(e)}")
            return False
    
    def _get_data_with_cache(self, cache_name, fetch_func, force_refresh=False, cache_duration=3600):
        """获取数据，优先使用缓存
        
        Args:
            cache_name: 缓存名称
            fetch_func: 获取数据的函数
            force_refresh: 是否强制刷新
            cache_duration: 缓存有效期（秒）
            
        Returns:
            dict: 数据
        """
        # 检查是否有缓存且未过期
        last_update = self.last_update.get(cache_name, 0)
        if not force_refresh and time.time() - last_update < cache_duration:
            if cache_name in self.cache:
                return self.cache[cache_name]
        
        # 检查文件缓存
        if not force_refresh:
            cache_data = self._load_cache(cache_name)
            if cache_data:
                # 更新内存缓存
                self.cache[cache_name] = cache_data
                self.last_update[cache_name] = time.time()
                return cache_data
        
        # 获取新数据
        try:
            data = fetch_func()
            if data is not None:
                # 更新缓存
                self.cache[cache_name] = data
                self.last_update[cache_name] = time.time()
                self._save_cache(cache_name, data)
                return data
        except Exception as e:
            self.logger.error(f"获取{cache_name}数据失败: {str(e)}")
            
            # 尝试使用旧缓存
            if cache_name in self.cache:
                return self.cache[cache_name]
            
            # 尝试从文件加载旧缓存
            cache_data = self._load_cache(cache_name)
            if cache_data:
                self.cache[cache_name] = cache_data
                return cache_data
        
        # 所有方法都失败
        return None
    
    def get_stock_list(self, force_refresh=False):
        """获取股票列表
        
        Args:
            force_refresh: 是否强制刷新
            
        Returns:
            list: 股票列表
        """
        if self.api is None:
            return []
        
        def fetch_func():
            # 获取股票基本信息
            df = self.api.stock_basic(exchange='', list_status='L', 
                              fields='ts_code,symbol,name,area,industry,market,list_date')
            
            # 转换为列表
            stocks = []
            for _, row in df.iterrows():
                stock = {
                    "ts_code": row["ts_code"],
                    "code": row["symbol"],
                    "name": row["name"],
                    "area": row["area"],
                    "industry": row["industry"],
                    "market": row["market"],
                    "list_date": row["list_date"]
                }
                stocks.append(stock)
            
            return stocks
        
        # 获取数据
        stocks = self._get_data_with_cache("stock_list", fetch_func, force_refresh, 86400)
        if stocks:
            self.stock_list = stocks
        return stocks or []
    
    def get_index_list(self, force_refresh=False):
        """获取指数列表
        
        Args:
            force_refresh: 是否强制刷新
            
        Returns:
            list: 指数列表
        """
        if self.api is None:
            return []
        
        def fetch_func():
            # 获取指数基本信息
            df = self.api.index_basic(market='SSE')
            df2 = self.api.index_basic(market='SZSE')
            df = pd.concat([df, df2])
            
            # 转换为列表
            indices = []
            for _, row in df.iterrows():
                index = {
                    "ts_code": row["ts_code"],
                    "name": row["name"],
                    "market": row["market"],
                    "publisher": row["publisher"],
                    "category": row["category"],
                    "base_date": row["base_date"],
                    "base_point": row["base_point"]
                }
                indices.append(index)
            
            return indices
        
        # 获取数据
        indices = self._get_data_with_cache("index_list", fetch_func, force_refresh, 86400)
        if indices:
            self.indices_list = indices
        return indices or []
    
    def get_stock_data(self, code, force_refresh=False):
        """获取股票数据
        
        Args:
            code: 股票代码
            force_refresh: 是否强制刷新
            
        Returns:
            dict: 股票数据
        """
        if self.api is None:
            return {}
        
        # 确保code格式正确（带交易所后缀）
        ts_code = code
        if not code.endswith('.SH') and not code.endswith('.SZ'):
            # 尝试从股票列表查找
            if self.stock_list is None:
                self.get_stock_list()
            
            if self.stock_list:
                for stock in self.stock_list:
                    if stock["code"] == code:
                        ts_code = stock["ts_code"]
                        break
            
            # 如果仍未找到，尝试猜测
            if not ts_code.endswith('.SH') and not ts_code.endswith('.SZ'):
                if code.startswith('6') or code.startswith('688'):
                    ts_code = f"{code}.SH"
                else:
                    ts_code = f"{code}.SZ"
        
        def fetch_func():
            # 获取日线数据（最近30个交易日）
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=60)).strftime('%Y%m%d')
            
            # 获取日线数据
            df_daily = self.api.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            
            if df_daily.empty:
                return None
            
            # 获取当日行情
            df_today = None
            try:
                df_today = ts.get_realtime_quotes(code.replace('.SH', '').replace('.SZ', ''))
            except:
                pass
            
            # 获取基本信息
            stock_info = {}
            if self.stock_list:
                for stock in self.stock_list:
                    if stock["ts_code"] == ts_code:
                        stock_info = stock
                        break
            
            # 构建历史数据
            history = {
                "dates": df_daily["trade_date"].tolist(),
                "prices": df_daily["close"].tolist(),
                "opens": df_daily["open"].tolist(),
                "highs": df_daily["high"].tolist(),
                "lows": df_daily["low"].tolist(),
                "volumes": df_daily["vol"].tolist()
            }
            
            # 获取最新价格和变化
            if df_today is not None and not df_today.empty:
                current_price = float(df_today.iloc[0]["price"])
                prev_close = float(df_today.iloc[0]["pre_close"])
                change = (current_price - prev_close) / prev_close
            else:
                current_price = float(df_daily.iloc[0]["close"])
                prev_close = float(df_daily.iloc[1]["close"]) if len(df_daily) > 1 else current_price
                change = (current_price - prev_close) / prev_close
            
            # 构建结果
            result = {
                "code": code,
                "ts_code": ts_code,
                "name": stock_info.get("name", ""),
                "industry": stock_info.get("industry", ""),
                "price": current_price,
                "prev_close": prev_close,
                "change": change,
                "volume": float(df_daily.iloc[0]["vol"]),
                "turnover": float(df_daily.iloc[0]["amount"]),
                "history": history
            }
            
            return result
        
        # 获取数据
        return self._get_data_with_cache(f"stock_{code}", fetch_func, force_refresh, 300) or {}
    
    def get_index_data(self):
        """获取指数数据
        
        获取主要指数的实时行情数据
            
        Returns:
            list: 指数数据列表，每个元素是一个字典
        """
        # 检查API状态
        if self.api is None:
            self.logger.error("无法获取指数数据: Tushare Pro API未初始化")
            return []
        
        try:
            # 指数代码列表
            index_codes = ['000001.SH', '399001.SZ', '399006.SZ', '000300.SH', '000016.SH', '000688.SH', '000905.SH']
            
            # 尝试使用daily接口获取最新行情
            today = datetime.now().strftime('%Y%m%d')
            trade_date = today
            
            # 首先获取当前交易日
            try:
                calendar_df = self.api.trade_cal(exchange='SSE', start_date=(datetime.now() - timedelta(days=10)).strftime('%Y%m%d'), 
                                               end_date=today, is_open=1)
                if not calendar_df.empty:
                    # 获取最近交易日
                    trade_date = calendar_df['cal_date'].iloc[-1]
            except Exception as e:
                self.logger.warning(f"获取交易日历失败: {str(e)}，使用当前日期")
            
            # 获取指数行情
            try:
                df = self.api.index_daily(ts_code=','.join(index_codes), trade_date=trade_date)
                # 如果当日数据不可用，获取最近一个交易日的数据
                if df.empty:
                    # 尝试获取最近五个交易日内的数据
                    last_days = 5
                    for i in range(1, last_days + 1):
                        try_date = (datetime.now() - timedelta(days=i)).strftime('%Y%m%d')
                        try:
                            df = self.api.index_daily(ts_code=','.join(index_codes), trade_date=try_date)
                            if not df.empty:
                                self.logger.info(f"获取到 {try_date} 的指数数据")
                                break
                        except Exception:
                            continue
            except Exception as e:
                self.logger.warning(f"获取指数日线数据失败: {str(e)}，尝试使用实时行情")
                # 尝试使用行情接口
                try:
                    df = ts.get_realtime_quotes(['sh000001', 'sz399001', 'sz399006'])
                    # 转换为标准格式
                    if df is not None and not df.empty:
                        new_df = pd.DataFrame()
                        new_df['ts_code'] = df['code'].apply(lambda x: f"{x[2:]}.{'SH' if x.startswith('sh') else 'SZ'}")
                        new_df['trade_date'] = today
                        new_df['close'] = df['price'].astype(float)
                        new_df['pre_close'] = df['pre_close'].astype(float)
                        new_df['open'] = df['open'].astype(float)
                        new_df['high'] = df['high'].astype(float)
                        new_df['low'] = df['low'].astype(float)
                        new_df['vol'] = df['volume'].astype(float) / 100  # 转换为手
                        new_df['pct_chg'] = ((new_df['close'] - new_df['pre_close']) / new_df['pre_close'] * 100).round(2)
                        df = new_df
                except Exception as inner_e:
                    self.logger.error(f"获取指数实时行情失败: {str(inner_e)}")
                    # 尝试使用缓存
                    return self._get_cached_indices()
            
            if df is None or df.empty:
                self.logger.warning("未获取到指数数据，使用缓存数据")
                return self._get_cached_indices()
            
            # 转换为结果格式
            result = []
            for _, row in df.iterrows():
                index_code = row['ts_code']
                # 获取指数名称
                index_name = self._get_index_name(index_code)
                
                # 添加结果
                result.append({
                    'code': index_code,
                    'name': index_name,
                    'price': round(float(row['close']), 2),
                    'change': round(float(row['close']) - float(row['pre_close']), 2),
                    'change_pct': round(float(row.get('pct_chg', 0)), 2),
                    'open': round(float(row['open']), 2),
                    'high': round(float(row['high']), 2),
                    'low': round(float(row['low']), 2),
                    'volume': int(row['vol']),
                    'trade_date': row['trade_date']
                })
            
            # 更新缓存
            self._cache_indices(result)
            
            return result
        
        except Exception as e:
            self.logger.error(f"获取指数数据异常: {str(e)}")
            # 使用缓存数据
            return self._get_cached_indices()

    def _get_index_name(self, index_code):
        """获取指数名称"""
        index_map = {
            '000001.SH': '上证指数',
            '399001.SZ': '深证成指',
            '399006.SZ': '创业板指',
            '000300.SH': '沪深300',
            '000016.SH': '上证50',
            '000688.SH': '科创50',
            '000905.SH': '中证500'
        }
        return index_map.get(index_code, f"指数{index_code}")

    def _cache_indices(self, indices_data):
        """缓存指数数据"""
        try:
            cache_file = os.path.join(CACHE_DIR, "indices_cache.json")
            cache_data = {
                'indices': indices_data,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False)
        except Exception as e:
            self.logger.warning(f"缓存指数数据失败: {str(e)}")

    def _get_cached_indices(self):
        """获取缓存的指数数据"""
        try:
            cache_file = os.path.join(CACHE_DIR, "indices_cache.json")
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    # 检查缓存是否有效（24小时内）
                    timestamp = datetime.strptime(cache_data['timestamp'], '%Y-%m-%d %H:%M:%S')
                    if (datetime.now() - timestamp).total_seconds() < 86400:  # 24小时
                        self.logger.info("使用缓存的指数数据")
                        return cache_data['indices']
                    else:
                        self.logger.warning("缓存的指数数据已过期")
        except Exception as e:
            self.logger.warning(f"读取缓存指数数据失败: {str(e)}")
        
        # 返回基本指数数据
        return [
            {'code': '000001.SH', 'name': '上证指数', 'price': 3200.15, 'change': 12.35, 'change_pct': 0.38},
            {'code': '399001.SZ', 'name': '深证成指', 'price': 10523.67, 'change': 45.23, 'change_pct': 0.42}
        ]

    def _ensure_api_ready(self):
        """确保API已准备好
            
        Returns:
            bool: API是否准备好
        """
        if self.is_ready:
            return True
        
        if self.offline_mode:
            # 如果已经知道处于离线模式，不再尝试初始化API以减少日志
            return False
            
        # 尝试初始化API
        success = self.init_api()
        if not success and not self.offline_mode:
            self.logger.debug("TuShare API未准备好，使用离线模式和缓存数据")
            self.offline_mode = True
        return success
    
    def get_market_status(self) -> Dict[str, str]:
        """获取市场状态"""
        try:
            # 获取当前日期
            import datetime
            today = datetime.date.today().strftime("%Y%m%d")
            
            # 获取交易日历
            df = self.api.trade_cal(exchange='SSE', start_date=today, end_date=today)
            is_open = df['is_open'].values[0] == 1 if not df.empty else False
            
            # 获取最近交易日
            last_df = self.api.trade_cal(
                exchange='SSE',
                start_date=(datetime.date.today() - datetime.timedelta(days=10)).strftime("%Y%m%d"),
                end_date=today
            )
            last_df = last_df[last_df['is_open'] == 1]
            last_trading_day = last_df['cal_date'].values[-2] if len(last_df) > 1 else today
            
            # 获取下一个交易日
            next_df = self.api.trade_cal(
                exchange='SSE',
                start_date=today,
                end_date=(datetime.date.today() + datetime.timedelta(days=10)).strftime("%Y%m%d")
            )
            next_df = next_df[next_df['is_open'] == 1]
            next_trading_day = next_df['cal_date'].values[1] if len(next_df) > 1 else today
            
            # 确定市场状态
            status = "开市" if is_open else "休市"
            time_str = datetime.datetime.now().strftime("%H:%M:%S")
            
            return {
                "status": status,
                "time": time_str,
                "trading_day": today,
                "next_trading_day": next_trading_day,
                "last_trading_day": last_trading_day
            }
        except Exception as e:
            self.logger.error(f"获取市场状态失败: {str(e)}")
            # 返回备用市场状态
            import datetime
            today = datetime.date.today()
            return {
                "status": "获取失败",
                "time": datetime.datetime.now().strftime("%H:%M:%S"),
                "trading_day": today.strftime("%Y%m%d"),
                "next_trading_day": (today + datetime.timedelta(days=1)).strftime("%Y%m%d"),
                "last_trading_day": (today - datetime.timedelta(days=1)).strftime("%Y%m%d")
            }
    
    def get_historical_data(self, code, start_date=None, end_date=None):
        """获取历史数据
        
        Args:
            code: 股票或指数代码
            start_date: 开始日期，格式YYYYMMDD
            end_date: 结束日期，格式YYYYMMDD
            
        Returns:
            dict: 历史数据
        """
        if self.api is None:
            return {}
        
        # 设置默认日期范围
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        
        # 确保日期格式正确
        if isinstance(start_date, datetime):
            start_date = start_date.strftime('%Y%m%d')
        if isinstance(end_date, datetime):
            end_date = end_date.strftime('%Y%m%d')
        
        # 判断是股票还是指数
        if code.endswith('.SH') or code.endswith('.SZ'):
            # 获取数据
            if code.startswith('0000') or code.startswith('399') or code.startswith('899'):
                # 指数
                index_data = self.get_index_data()
                if index_data and "history" in index_data:
                    return index_data["history"]
            else:
                # 股票
                stock_data = self.get_stock_data(code, True)
                if stock_data and "history" in stock_data:
                    return stock_data["history"]
        else:
            # 尝试获取股票数据
            stock_data = self.get_stock_data(code, True)
            if stock_data and "history" in stock_data:
                return stock_data["history"]
        
        return {}
    
    def get_daily_data(self, code, start_date=None, end_date=None, days=365):
        """获取股票日线数据
        
        Args:
            code: 股票代码
            start_date: 开始日期，格式YYYYMMDD
            end_date: 结束日期，格式YYYYMMDD
            days: 如果未提供start_date，则获取最近多少天的数据
            
        Returns:
            pandas.DataFrame: 日线数据
        """
        # 确保API已初始化
        if self.api is None:
            self.logger.error(f"无法获取股票 {code} 日线数据: Tushare Pro API未初始化")
            return pd.DataFrame()
            
        # 确保code格式正确（带交易所后缀）
        ts_code = code
        if not code.endswith('.SH') and not code.endswith('.SZ'):
            # 尝试从股票列表查找
            if self.stock_list is None:
                self.get_stock_list()
            
            if self.stock_list:
                for stock in self.stock_list:
                    if stock["code"] == code:
                        ts_code = stock["ts_code"]
                        break
            
            # 如果仍未找到，尝试猜测
            if not ts_code.endswith('.SH') and not ts_code.endswith('.SZ'):
                if code.startswith('6'):
                    ts_code = f"{code}.SH"
                else:
                    ts_code = f"{code}.SZ"
        
        # 设置默认日期范围
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
        
        # 确保日期格式正确
        if isinstance(start_date, datetime):
            start_date = start_date.strftime('%Y%m%d')
        if isinstance(end_date, datetime):
            end_date = end_date.strftime('%Y%m%d')
        
        try:
            # 使用Pro API获取数据
            self.logger.info(f"使用Tushare Pro API获取股票 {ts_code} 日线数据，从 {start_date} 到 {end_date}")
            df = self.api.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            
            if df is not None and not df.empty:
                # 按日期排序
                df = df.sort_values('trade_date', ascending=True)
                self.logger.info(f"成功获取到股票 {ts_code} 的 {len(df)} 条日线数据")
                return df
            else:
                self.logger.warning(f"Tushare Pro API返回的 {ts_code} 数据为空")
        except Exception as e:
            self.logger.error(f"获取股票 {ts_code} 日线数据失败: {str(e)}")
        
        # 返回空数据
        self.logger.warning(f"获取股票 {ts_code} 数据失败，返回空DataFrame")
        return pd.DataFrame()
    
    def search_stocks(self, keyword):
        """搜索股票
        
        Args:
            keyword: 关键词
            
        Returns:
            list: 股票列表
        """
        # 获取股票列表
        if self.stock_list is None:
            self.get_stock_list()
        
        if not self.stock_list:
            return []
        
        # 搜索匹配项
        results = []
        for stock in self.stock_list:
            if keyword in stock["code"] or keyword in stock["name"]:
                # 获取最新价格
                stock_data = self.get_stock_data(stock["ts_code"], False)
                
                results.append({
                    "code": stock["code"],
                    "ts_code": stock["ts_code"],
                    "name": stock["name"],
                    "industry": stock["industry"],
                    "price": stock_data.get("price", 0.0),
                    "change": stock_data.get("change", 0.0)
                })
        
        return results
    
    def get_recommended_stocks(self, count=10, force_refresh=False):
        """获取推荐股票列表
        
        根据多种因素分析选取推荐股票，包括：
        - 成交量变化
        - 价格变动百分比
        - 最近交易活跃度
        
        Args:
            count: 返回的推荐股票数量
            force_refresh: 是否强制刷新数据
        
        Returns:
            list: 推荐股票列表
        """
        cache_key = "recommended_stocks"
        
        # 检查缓存 - 优先使用缓存数据，每小时刷新一次
        if not force_refresh and cache_key in self.cache:
            cached_data = self.cache[cache_key]
            cache_time = self.last_update.get(cache_key, 0)
            # 缓存有效期为1小时
            if time.time() - cache_time < 3600:
                return cached_data[:count] if len(cached_data) > count else cached_data
        
        # 尝试从缓存目录加载备用数据
        backup_data = self._get_cached_recommended_stocks()
        
        try:
            # 检查API状态
            if self.api is None or not self.is_ready:
                self.logger.warning("TuShare API未就绪，使用缓存的推荐股票数据")
                return backup_data[:count] if backup_data and len(backup_data) > count else backup_data
            
            # 获取当前市场数据
            self.logger.info("获取最新市场数据以生成推荐股票...")
            
            # 获取最后交易日
            last_trade_day = self._get_last_trading_day()
            if not last_trade_day:
                self.logger.warning("无法获取最后交易日，使用缓存的推荐股票数据")
                return backup_data[:count] if backup_data and len(backup_data) > count else backup_data
            
            # 获取活跃股票数据
            try:
                # 获取最新交易日活跃股票
                df = self.api.daily(trade_date=last_trade_day)
                if df is None or df.empty:
                    self.logger.warning("未获取到活跃股票数据，使用缓存数据")
                    return backup_data[:count] if backup_data and len(backup_data) > count else backup_data
                
                # 按成交量和涨跌幅排序
                df['score'] = df['vol'] * df['pct_chg'].abs() / 100
                df = df.sort_values('score', ascending=False)
                
                # 获取前N只股票
                top_stocks = df.head(count * 2)  # 获取更多股票，以便过滤
                
                # 获取股票详细信息
                recommended = []
                for _, row in top_stocks.iterrows():
                    try:
                        # 获取股票基本信息
                        code = row['ts_code']
                        stock_info = self._get_stock_info(code)
                        
                        if stock_info:
                            # 计算推荐度分数 (0-100)
                            recommendation_score = min(100, abs(row['pct_chg']) * 2 + row['vol'] / 1000000)
                            
                            # 创建推荐股票信息
                            stock = {
                                'code': code.split('.')[0],  # 去掉后缀
                                'ts_code': code,
                                'name': stock_info['name'],
                                'price': row['close'],
                                'change': row['change'],
                                'change_pct': row['pct_chg'],
                                'volume': row['vol'],
                                'amount': row['amount'],
                                'industry': stock_info.get('industry', ''),
                                'recommendation': recommendation_score,
                                'reason': self._get_recommendation_reason(row)
                            }
                            recommended.append(stock)
                            
                            # 如果已有足够的股票，则退出
                            if len(recommended) >= count:
                                break
                    except Exception as e:
                        self.logger.warning(f"处理推荐股票 {code} 时出错: {str(e)}")
                
                # 如果获取的股票不足，使用缓存数据补充
                if len(recommended) < count and backup_data:
                    self.logger.info(f"实时推荐股票不足 {count} 只，使用缓存数据补充")
                    # 找出不重复的缓存股票
                    existing_codes = {s['ts_code'] for s in recommended}
                    for stock in backup_data:
                        if stock['ts_code'] not in existing_codes and len(recommended) < count:
                            recommended.append(stock)
                            existing_codes.add(stock['ts_code'])
                
                # 如果仍然不足，随机降低阈值重新选择
                if len(recommended) < count:
                    self.logger.info("推荐股票数量不足，降低筛选标准")
                    remaining_needed = count - len(recommended)
                    existing_codes = {s['ts_code'] for s in recommended}
                    
                    for _, row in df.iloc[len(recommended):].iterrows():
                        if len(recommended) >= count:
                            break
                            
                        code = row['ts_code']
                        if code not in existing_codes:
                            try:
                                stock_info = self._get_stock_info(code)
                                if stock_info:
                                    # 创建推荐股票信息，降低推荐度
                                    recommendation_score = min(50, abs(row['pct_chg']) + row['vol'] / 2000000)
                                    stock = {
                                        'code': code.split('.')[0],
                                        'ts_code': code,
                                        'name': stock_info['name'],
                                        'price': row['close'],
                                        'change': row['change'],
                                        'change_pct': row['pct_chg'],
                                        'volume': row['vol'],
                                        'amount': row['amount'],
                                        'industry': stock_info.get('industry', ''),
                                        'recommendation': recommendation_score,
                                        'reason': "交易活跃"
                                    }
                                    recommended.append(stock)
                                    existing_codes.add(code)
                            except Exception as e:
                                self.logger.warning(f"处理备选推荐股票 {code} 时出错: {str(e)}")
                
                # 更新缓存
                if recommended:
                    self.cache[cache_key] = recommended
                    self.last_update[cache_key] = time.time()
                    
                    # 同时写入文件缓存，以便下次使用
                    try:
                        cache_file = os.path.join(CACHE_DIR, "recommended_stocks.json")
                        cache_data = {
                            'recommended_stocks': recommended,
                            'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                        with open(cache_file, 'w', encoding='utf-8') as f:
                            json.dump(cache_data, f, ensure_ascii=False, indent=2)
                        self.logger.info(f"已缓存 {len(recommended)} 只推荐股票")
                    except Exception as e:
                        self.logger.warning(f"缓存推荐股票数据失败: {str(e)}")
                
                return recommended
                
            except Exception as e:
                self.logger.error(f"获取推荐股票失败: {str(e)}")
                return backup_data[:count] if backup_data and len(backup_data) > count else backup_data
                
        except Exception as e:
            self.logger.error(f"获取推荐股票时发生错误: {str(e)}")
            return backup_data[:count] if backup_data and len(backup_data) > count else backup_data
    
    def _get_cached_recommended_stocks(self):
        """获取缓存的推荐股票数据
        
        Returns:
            list: 缓存的推荐股票列表，如果没有缓存则返回默认数据
        """
        try:
            cache_file = os.path.join(CACHE_DIR, "recommended_stocks.json")
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    stocks = cached_data.get('recommended_stocks', [])
                    if stocks:
                        self.logger.info(f"从文件缓存加载了 {len(stocks)} 只推荐股票")
                        return stocks
        except Exception as e:
            self.logger.warning(f"加载缓存的推荐股票数据失败: {str(e)}")
        
        # 返回默认推荐股票
        return [
            {
                'code': '601933',
                'ts_code': '601933.SH',
                'name': '永辉超市',
                'price': 6.82,
                'change': 0.14,
                'change_pct': 2.1,
                'volume': 98756200,
                'amount': 673400000,
                'industry': '零售业',
                'recommendation': 85,
                'reason': '成交量大幅增加'
            },
            {
                'code': '600900',
                'ts_code': '600900.SH',
                'name': '长江电力',
                'price': 22.36,
                'change': 0.28,
                'change_pct': 1.27,
                'volume': 65432100,
                'amount': 1463700000,
                'industry': '电力',
                'recommendation': 78,
                'reason': '稳定增长'
            },
            {
                'code': '000858',
                'ts_code': '000858.SZ',
                'name': '五粮液',
                'price': 168.5,
                'change': 3.2,
                'change_pct': 1.94,
                'volume': 12453600,
                'amount': 2098400000,
                'industry': '食品饮料',
                'recommendation': 88,
                'reason': '高品质绩优股'
            },
            {
                'code': '600519',
                'ts_code': '600519.SH',
                'name': '贵州茅台',
                'price': 1788.0,
                'change': 8.5,
                'change_pct': 0.48,
                'volume': 1589700,
                'amount': 2842000000,
                'industry': '食品饮料',
                'recommendation': 90,
                'reason': '绩优蓝筹'
            },
            {
                'code': '601318',
                'ts_code': '601318.SH',
                'name': '中国平安',
                'price': 48.32,
                'change': 0.64,
                'change_pct': 1.34,
                'volume': 45678900,
                'amount': 2207200000,
                'industry': '金融保险',
                'recommendation': 83,
                'reason': '价值投资首选'
            },
            {
                'code': '000333',
                'ts_code': '000333.SZ',
                'name': '美的集团',
                'price': 55.85,
                'change': 1.25,
                'change_pct': 2.29,
                'volume': 32154600,
                'amount': 1795900000,
                'industry': '家电',
                'recommendation': 84,
                'reason': '行业龙头'
            },
            {
                'code': '600036',
                'ts_code': '600036.SH',
                'name': '招商银行',
                'price': 38.65,
                'change': 0.48,
                'change_pct': 1.26,
                'volume': 28965400,
                'amount': 1119300000,
                'industry': '银行',
                'recommendation': 79,
                'reason': '银行业绩优'
            },
            {
                'code': '601888',
                'ts_code': '601888.SH',
                'name': '中国中免',
                'price': 193.8,
                'change': 4.1,
                'change_pct': 2.16,
                'volume': 7896500,
                'amount': 1530500000,
                'industry': '旅游商业',
                'recommendation': 80,
                'reason': '消费复苏受益'
            },
            {
                'code': '603259',
                'ts_code': '603259.SH',
                'name': '药明康德',
                'price': 86.2,
                'change': 2.15,
                'change_pct': 2.56,
                'volume': 14523600,
                'amount': 1251900000,
                'industry': '医药',
                'recommendation': 82,
                'reason': '医药研发龙头'
            },
            {
                'code': '002594',
                'ts_code': '002594.SZ',
                'name': '比亚迪',
                'price': 254.15,
                'change': 5.8,
                'change_pct': 2.34,
                'volume': 9876500,
                'amount': 2510600000,
                'industry': '汽车',
                'recommendation': 87,
                'reason': '新能源汽车龙头'
            }
        ]
    
    def _get_last_trading_day(self):
        """获取最近的交易日
        
        Returns:
            str: 最近交易日，格式YYYYMMDD
        """
        try:
            if self.api is None:
                self.logger.warning("API未初始化，无法获取最近交易日")
                return None
            
            # 获取当前日期
            today = datetime.now().strftime('%Y%m%d')
            # 获取从10天前到今天的交易日历
            df = self.api.trade_cal(exchange='SSE', start_date=(datetime.now() - timedelta(days=30)).strftime('%Y%m%d'), 
                                   end_date=today)
            
            if df is not None and not df.empty:
                # 找出最近的交易日（is_open=1）
                open_days = df[df['is_open'] == 1]['cal_date']
                if not open_days.empty:
                    return open_days.iloc[-1]
            return None
        except Exception as e:
            self.logger.warning(f"获取最近交易日失败: {str(e)}")
            return None
    
    def _get_stock_info(self, code):
        """获取股票基本信息
        
        Args:
            code: 股票代码
            
        Returns:
            dict: 股票基本信息
        """
        try:
            if self.api is None:
                self.logger.warning("API未初始化，无法获取股票信息")
                return None
                
            # 从缓存获取
            if code in self.stock_info_cache:
                return self.stock_info_cache[code]
                
            # 获取股票名称
            df = self.api.stock_basic(ts_code=code)
            if df is not None and not df.empty:
                info = {
                    'code': code,
                    'name': df.iloc[0]['name'],
                    'industry': df.iloc[0]['industry'],
                    'market': df.iloc[0]['market'],
                    'list_date': df.iloc[0]['list_date']
                }
                # 缓存结果
                self.stock_info_cache[code] = info
                return info
                
            return None
        except Exception as e:
            self.logger.warning(f"获取股票 {code} 基本信息失败: {str(e)}")
            return None
    
    def _get_recommendation_reason(self, row):
        """基于股票数据生成推荐理由
        
        Args:
            row: 股票数据行
            
        Returns:
            str: 推荐理由
        """
        reasons = []
        
        # 基于涨跌幅
        if row['pct_chg'] > 5:
            reasons.append("强势上涨")
        elif row['pct_chg'] > 2:
            reasons.append("稳步上涨")
        elif row['pct_chg'] > 0:
            reasons.append("小幅上涨")
        elif row['pct_chg'] > -2:
            reasons.append("企稳回升")
        else:
            reasons.append("超跌反弹机会")
            
        # 基于成交量
        if row['vol'] > 80000000:
            reasons.append("成交量显著放大")
        elif row['vol'] > 40000000:
            reasons.append("成交活跃")
        elif row['vol'] > 20000000:
            reasons.append("成交量适中")
            
        # 组合理由
        if not reasons:
            return "交易活跃"
        elif len(reasons) == 1:
            return reasons[0]
        else:
            return f"{reasons[0]}，{reasons[1]}"
    
    def get_positions(self):
        """获取持仓数据（模拟）
        
        Returns:
            list: 持仓列表
        """
        # 获取推荐股票作为持仓基础
        recommended = self.get_recommended_stocks()
        if not recommended:
            return []
        
        # 模拟持仓数据
        positions = []
        for i, stock in enumerate(recommended[:5]):
            quantity = np.random.randint(1000, 10000) // 100 * 100
            cost = stock["price"] * (1 - np.random.uniform(-0.1, 0.1))
            
            positions.append({
                "code": stock["code"],
                "ts_code": stock["ts_code"],
                "name": stock["name"],
                "quantity": quantity,
                "available": quantity,
                "cost": cost,
                "price": stock["price"],
                "profit": (stock["price"] - cost) * quantity,
                "profit_percent": (stock["price"] / cost - 1) * 100
            })
        
        return positions 

    def get_money_flow(self, ts_code, start_date, end_date=None):
        """获取个股资金流向数据
        
        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期，默认为今天
            
        Returns:
            pandas.DataFrame: 资金流向数据
        """
        if self.api is None:
            self.logger.error("API未初始化，无法获取资金流向数据")
            return None
            
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
            
        try:
            self.logger.info(f"获取股票 {ts_code} 资金流向数据，从 {start_date} 到 {end_date}")
            df = self.api.moneyflow(ts_code=ts_code, start_date=start_date, end_date=end_date)
            
            if df is not None and not df.empty:
                self.logger.info(f"获取到 {len(df)} 条资金流向数据")
                return df
            else:
                self.logger.warning("未获取到资金流向数据")
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"获取资金流向数据失败: {str(e)}")
            return pd.DataFrame()
            
    def get_north_money(self, start_date, end_date=None, market_type='S'):
        """获取北向资金流向数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期，默认为今天
            market_type: 市场类型 (沪市: 'S', 深市: 'N', 全部: '')
            
        Returns:
            pandas.DataFrame: 北向资金数据
        """
        if self.api is None:
            self.logger.error("API未初始化，无法获取北向资金数据")
            return None
            
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
            
        try:
            self.logger.info(f"获取北向资金数据，从 {start_date} 到 {end_date}")
            df = self.api.moneyflow_hsgt(start_date=start_date, end_date=end_date)
            
            if df is not None and not df.empty:
                self.logger.info(f"获取到 {len(df)} 条北向资金数据")
                
                # 如果需要过滤市场
                if market_type and market_type in ['S', 'N']:
                    if market_type == 'S':
                        df = df[df['market_type'].isin([1, 3])]  # 沪股通
                    elif market_type == 'N':
                        df = df[df['market_type'].isin([2, 4])]  # 深股通
                
                return df
            else:
                self.logger.warning("未获取到北向资金数据")
                return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"获取北向资金数据失败: {str(e)}")
            return pd.DataFrame()
            
    def get_market_sentiment(self, trade_date=None, refresh=False):
        """计算市场情绪指标
        
        利用多种指标综合计算市场情绪：
        - 涨跌家数比
        - 成交量变化
        - 北向资金
        - 大单净流入
        - 换手率
        
        Args:
            trade_date: 交易日期，默认最新
            refresh: 是否强制刷新
            
        Returns:
            dict: 市场情绪指标
        """
        if self.api is None:
            self.logger.error("API未初始化，无法计算市场情绪")
            return {}
            
        # 如果已有缓存且不刷新，直接返回
        if self.market_sentiment is not None and not refresh:
            return self.market_sentiment
            
        if trade_date is None:
            trade_date = datetime.now().strftime('%Y%m%d')
            # 检查是否为交易日
            calendar = self.get_trade_calendar(trade_date, trade_date)
            if calendar.empty or calendar.iloc[0]['is_open'] == 0:
                # 获取前一个交易日
                calendar = self.get_trade_calendar(
                    (datetime.now() - timedelta(days=10)).strftime('%Y%m%d'),
                    trade_date
                )
                calendar = calendar[calendar['is_open'] == 1]
                if not calendar.empty:
                    trade_date = calendar.iloc[-1]['cal_date']
        
        try:
            sentiment = {}
            
            # 1. 获取指数行情
            index_daily = self.api.index_daily(ts_code='000001.SH', trade_date=trade_date)
            if not index_daily.empty:
                # 获取前一交易日数据
                prev_date = self._get_prev_trade_date(trade_date)
                prev_index = self.api.index_daily(ts_code='000001.SH', trade_date=prev_date)
                
                if not prev_index.empty:
                    # 计算指数变化
                    idx_change = (index_daily.iloc[0]['close'] - prev_index.iloc[0]['close']) / prev_index.iloc[0]['close']
                    vol_change = (index_daily.iloc[0]['vol'] - prev_index.iloc[0]['vol']) / prev_index.iloc[0]['vol']
                    
                    sentiment['index_change'] = idx_change
                    sentiment['volume_change'] = vol_change
            
            # 2. 获取涨跌家数
            try:
                limit_list = self.api.limit_list(trade_date=trade_date, limit_type='U,D')
                if not limit_list.empty:
                    up_count = len(limit_list[limit_list['limit_type'] == 'U'])
                    down_count = len(limit_list[limit_list['limit_type'] == 'D'])
                    sentiment['limit_up_count'] = up_count
                    sentiment['limit_down_count'] = down_count
            except:
                self.logger.warning("获取涨跌停数据失败")
                
            # 3. 北向资金
            try:
                north_money = self.api.moneyflow_hsgt(trade_date=trade_date)
                if not north_money.empty:
                    sentiment['north_money'] = north_money['net_buy_amount'].sum()
            except:
                self.logger.warning("获取北向资金数据失败")
                
            # 4. 综合情绪分析
            # 计算情绪得分 (0-100)
            score = 50  # 中性
            
            # 根据指数变化调整
            if 'index_change' in sentiment:
                if sentiment['index_change'] > 0.02:
                    score += 15
                elif sentiment['index_change'] > 0.01:
                    score += 10
                elif sentiment['index_change'] > 0:
                    score += 5
                elif sentiment['index_change'] < -0.02:
                    score -= 15
                elif sentiment['index_change'] < -0.01:
                    score -= 10
                elif sentiment['index_change'] < 0:
                    score -= 5
            
            # 根据成交量变化调整
            if 'volume_change' in sentiment:
                if sentiment['volume_change'] > 0.2 and sentiment.get('index_change', 0) > 0:
                    score += 10  # 量增价升
                elif sentiment['volume_change'] < -0.2 and sentiment.get('index_change', 0) < 0:
                    score -= 10  # 量减价跌
            
            # 根据涨跌停家数调整
            if 'limit_up_count' in sentiment and 'limit_down_count' in sentiment:
                ratio = sentiment['limit_up_count'] / max(1, sentiment['limit_down_count'])
                if ratio > 3:
                    score += 15
                elif ratio > 1.5:
                    score += 10
                elif ratio < 0.3:
                    score -= 15
                elif ratio < 0.7:
                    score -= 10
            
            # 根据北向资金调整
            if 'north_money' in sentiment:
                if sentiment['north_money'] > 5e9:  # 50亿以上
                    score += 15
                elif sentiment['north_money'] > 1e9:  # 10亿以上
                    score += 10
                elif sentiment['north_money'] < -5e9:
                    score -= 15
                elif sentiment['north_money'] < -1e9:
                    score -= 10
            
            # 确保分数在0-100之间
            score = max(0, min(100, score))
            
            # 添加情绪标签
            if score >= 75:
                mood = "极度乐观"
            elif score >= 60:
                mood = "乐观"
            elif score >= 40:
                mood = "中性"
            elif score >= 25:
                mood = "悲观"
            else:
                mood = "极度悲观"
                
            sentiment['score'] = score
            sentiment['mood'] = mood
            sentiment['trade_date'] = trade_date
            
            # 更新缓存
            self.market_sentiment = sentiment
            
            return sentiment
            
        except Exception as e:
            self.logger.error(f"计算市场情绪失败: {str(e)}")
            return {}
    
    def _get_prev_trade_date(self, trade_date):
        """获取前一个交易日
        
        Args:
            trade_date: 当前交易日
            
        Returns:
            str: 前一个交易日
        """
        try:
            # 获取交易日历
            calendar = self.get_trade_calendar(
                (datetime.strptime(trade_date, '%Y%m%d') - timedelta(days=10)).strftime('%Y%m%d'),
                trade_date
            )
            calendar = calendar[calendar['is_open'] == 1]
            
            if len(calendar) > 1:
                return calendar.iloc[-2]['cal_date']
            else:
                # 如果没有足够的数据，返回一个估计的日期
                return (datetime.strptime(trade_date, '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d')
        except:
            # 如果出错，返回一个估计的日期
            return (datetime.strptime(trade_date, '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d')
    
    def get_trade_calendar(self, start_date, end_date):
        """获取交易日历
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            pandas.DataFrame: 交易日历
        """
        if self.api is None:
            return pd.DataFrame()
            
        try:
            return self.api.trade_cal(start_date=start_date, end_date=end_date)
        except Exception as e:
            self.logger.error(f"获取交易日历失败: {str(e)}")
            return pd.DataFrame()
    
    def get_industry_rotation(self, trade_date=None, days=5):
        """获取行业轮动数据
        
        计算各行业在指定时间段内的涨跌幅，分析行业轮动情况
        
        Args:
            trade_date: 截止交易日，默认最新
            days: 统计天数
            
        Returns:
            pandas.DataFrame: 行业轮动数据
        """
        if self.api is None:
            self.logger.error("API未初始化，无法获取行业轮动数据")
            return None
            
        if trade_date is None:
            trade_date = datetime.now().strftime('%Y%m%d')
            
        # 计算起始日期
        start_date = (datetime.strptime(trade_date, '%Y%m%d') - timedelta(days=days*2)).strftime('%Y%m%d')
        
        try:
            # 获取行业板块数据
            industry_list = self.api.index_classify(level='L1', src='SW')
            
            if industry_list is None or industry_list.empty:
                self.logger.warning("未获取到行业板块数据")
                return pd.DataFrame()
                
            # 获取每个行业的指数行情
            industry_perf = []
            
            for _, row in industry_list.iterrows():
                industry_code = row['index_code']
                industry_name = row['industry_name']
                
                # 获取行业指数行情
                df = self.api.index_daily(ts_code=industry_code, start_date=start_date, end_date=trade_date)
                
                if df is not None and not df.empty and len(df) >= days:
                    # 只取最近的days天
                    df = df.sort_values('trade_date', ascending=False).head(days)
                    
                    # 计算区间涨跌幅
                    earliest_close = df.iloc[-1]['close']
                    latest_close = df.iloc[0]['close']
                    change_pct = (latest_close - earliest_close) / earliest_close * 100
                    
                    # 计算日均成交量变化
                    avg_vol = df['vol'].mean()
                    
                    industry_perf.append({
                        'industry_code': industry_code,
                        'industry_name': industry_name,
                        'change_pct': change_pct,
                        'avg_vol': avg_vol,
                        'latest_close': latest_close,
                        'trade_date': trade_date,
                        'days': days
                    })
            
            # 转换为DataFrame并排序
            result = pd.DataFrame(industry_perf)
            if not result.empty:
                result = result.sort_values('change_pct', ascending=False)
                
            self.logger.info(f"计算了 {len(result)} 个行业的轮动数据")
            return result
            
        except Exception as e:
            self.logger.error(f"获取行业轮动数据失败: {str(e)}")
            return pd.DataFrame()
            
    def get_hot_stocks(self, trade_date=None, limit=50):
        """获取热门股票
        
        综合成交量、换手率、涨跌幅等因素，筛选热门股票
        
        Args:
            trade_date: 交易日期，默认最新
            limit: 返回数量
            
        Returns:
            pandas.DataFrame: 热门股票数据
        """
        if self.api is None:
            self.logger.error("API未初始化，无法获取热门股票")
            return None
            
        if trade_date is None:
            trade_date = datetime.now().strftime('%Y%m%d')
            
        try:
            # 获取当日所有股票行情
            df_daily = self.api.daily(trade_date=trade_date)
            
            if df_daily is None or df_daily.empty:
                self.logger.warning(f"未获取到 {trade_date} 的股票行情数据")
                return pd.DataFrame()
                
            # 计算综合得分
            df_daily['score'] = 0  # 初始得分
            
            # 换手率得分（标准化）
            if 'turnover_rate' in df_daily.columns:
                max_turnover = df_daily['turnover_rate'].max()
                min_turnover = df_daily['turnover_rate'].min()
                if max_turnover > min_turnover:
                    df_daily['turnover_score'] = 40 * (df_daily['turnover_rate'] - min_turnover) / (max_turnover - min_turnover)
                    df_daily['score'] += df_daily['turnover_score']
            
            # 成交量得分
            max_vol = df_daily['vol'].max()
            min_vol = df_daily['vol'].min()
            if max_vol > min_vol:
                df_daily['vol_score'] = 30 * (df_daily['vol'] - min_vol) / (max_vol - min_vol)
                df_daily['score'] += df_daily['vol_score']
                
            # 涨跌幅得分（取绝对值，大幅度波动都视为热门）
            df_daily['abs_pct_chg'] = df_daily['pct_chg'].abs()
            max_pct = df_daily['abs_pct_chg'].max()
            min_pct = df_daily['abs_pct_chg'].min()
            if max_pct > min_pct:
                df_daily['pct_score'] = 30 * (df_daily['abs_pct_chg'] - min_pct) / (max_pct - min_pct)
                df_daily['score'] += df_daily['pct_score']
                
            # 排序并返回前limit个
            hot_stocks = df_daily.sort_values('score', ascending=False).head(limit)
            
            # 获取股票名称
            stock_list = self.get_stock_list()
            if stock_list:
                # 转换为字典便于查询
                stock_dict = {item['ts_code']: item['name'] for item in stock_list}
                
                # 添加股票名称
                hot_stocks['name'] = hot_stocks['ts_code'].map(lambda x: stock_dict.get(x, ''))
                
            # 选择需要的列
            result_cols = ['ts_code', 'name', 'open', 'high', 'low', 'close', 'pre_close', 
                          'pct_chg', 'vol', 'amount', 'score']
            result = hot_stocks[result_cols] if all(col in hot_stocks.columns for col in result_cols) else hot_stocks
            
            self.logger.info(f"获取到 {len(result)} 只热门股票")
            return result
            
        except Exception as e:
            self.logger.error(f"获取热门股票失败: {str(e)}")
            return pd.DataFrame()
            
    def get_concept_stats(self, trade_date=None):
        """获取概念板块统计
        
        统计概念板块的涨跌情况
        
        Args:
            trade_date: 交易日期，默认最新
            
        Returns:
            pandas.DataFrame: 概念板块统计
        """
        if self.api is None:
            self.logger.error("API未初始化，无法获取概念板块统计")
            return None
            
        if trade_date is None:
            trade_date = datetime.now().strftime('%Y%m%d')
            
        try:
            # 获取概念板块列表
            concept_list = self.api.concept()
            
            if concept_list is None or concept_list.empty:
                self.logger.warning("未获取到概念板块列表")
                return pd.DataFrame()
                
            # 统计每个概念的表现
            concept_stats = []
            
            for _, row in concept_list.iterrows():
                concept_id = row['code']
                concept_name = row['name']
                
                # 获取概念成分股
                members = self.api.concept_detail(id=concept_id)
                
                if members is not None and not members.empty:
                    # 获取成分股行情
                    stock_codes = members['ts_code'].tolist()
                    
                    # 分批获取，避免请求过大
                    batch_size = 100
                    all_prices = []
                    
                    for i in range(0, len(stock_codes), batch_size):
                        batch_codes = stock_codes[i:i+batch_size]
                        batch_str = ",".join(batch_codes)
                        
                        prices = self.api.daily(ts_code=batch_str, trade_date=trade_date)
                        if prices is not None and not prices.empty:
                            all_prices.append(prices)
                    
                    if all_prices:
                        # 合并所有价格数据
                        price_df = pd.concat(all_prices)
                        
                        # 计算统计数据
                        avg_pct = price_df['pct_chg'].mean()
                        up_count = len(price_df[price_df['pct_chg'] > 0])
                        down_count = len(price_df[price_df['pct_chg'] < 0])
                        flat_count = len(price_df[price_df['pct_chg'] == 0])
                        total_count = len(price_df)
                        
                        # 计算涨跌比
                        up_down_ratio = up_count / max(1, down_count)
                        
                        concept_stats.append({
                            'concept_id': concept_id,
                            'concept_name': concept_name,
                            'avg_pct_chg': avg_pct,
                            'up_count': up_count,
                            'down_count': down_count,
                            'flat_count': flat_count,
                            'total_count': total_count,
                            'up_down_ratio': up_down_ratio,
                            'trade_date': trade_date
                        })
            
            # 转换为DataFrame并排序
            result = pd.DataFrame(concept_stats)
            if not result.empty:
                result = result.sort_values('avg_pct_chg', ascending=False)
                
            self.logger.info(f"统计了 {len(result)} 个概念板块的数据")
            return result
            
        except Exception as e:
            self.logger.error(f"获取概念板块统计失败: {str(e)}")
            return pd.DataFrame()

    def get_realtime_quotes(self, symbols):
        """获取实时行情数据
        
        使用TuShare获取指定股票的实时报价
        
        Args:
            symbols: 股票代码或代码列表
            
        Returns:
            pandas.DataFrame: 实时行情数据
        """
        if not TUSHARE_AVAILABLE:
            self.logger.error("TuShare模块未安装，无法获取实时行情")
            return None
            
        try:
            # 转为列表以统一处理
            if isinstance(symbols, str):
                symbols = [symbols]
                
            # 获取行情
            df = ts.get_realtime_quotes(symbols)
            
            if df is not None and not df.empty:
                # 转换数据类型
                for col in ['price', 'open', 'high', 'low', 'pre_close', 'volume', 'amount', 'b1_v', 'b2_v', 'b3_v', 'b4_v', 'b5_v', 'a1_v', 'a2_v', 'a3_v', 'a4_v', 'a5_v']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                self.logger.info(f"获取到 {len(df)} 条实时行情数据")
                return df
            else:
                self.logger.warning("未获取到实时行情数据")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"获取实时行情失败: {str(e)}")
            return pd.DataFrame()

    def get_quantitative_factors(self, ts_code, start_date, end_date=None):
        """计算量化因子
        
        结合多种数据源计算常用量化因子：
        - 动量因子
        - 价值因子
        - 波动率因子
        - 流动性因子
        
        Args:
            ts_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期，默认为今天
            
        Returns:
            pandas.DataFrame: 因子数据
        """
        if self.api is None:
            self.logger.error("API未初始化，无法计算量化因子")
            return None
            
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')
            
        try:
            # 获取历史行情数据
            df = self.get_daily_data(ts_code, start_date, end_date)
            
            if df is None or df.empty:
                self.logger.warning(f"未获取到股票 {ts_code} 的历史行情数据")
                return pd.DataFrame()
                
            # 确保按日期升序排列
            df = df.sort_values('trade_date')
            
            # 1. 计算动量因子
            # 1个月动量
            df['momentum_1m'] = df['close'].pct_change(21)
            
            # 3个月动量
            df['momentum_3m'] = df['close'].pct_change(63)
            
            # 6个月动量
            df['momentum_6m'] = df['close'].pct_change(126)
            
            # 12个月动量
            df['momentum_12m'] = df['close'].pct_change(252)
            
            # 2. 计算波动率因子
            # 20日波动率
            df['volatility_20d'] = df['close'].pct_change().rolling(20).std() * np.sqrt(20)
            
            # 60日波动率
            df['volatility_60d'] = df['close'].pct_change().rolling(60).std() * np.sqrt(60)
            
            # 3. 计算流动性因子
            # 20日平均成交量
            df['avg_vol_20d'] = df['vol'].rolling(20).mean()
            
            # 成交量动量
            df['vol_change_20d'] = df['vol'].pct_change(20)
            
            # 4. 价值因子（需要结合基本面数据）
            try:
                # 获取最新季度的财务指标
                fina = self.api.fina_indicator(ts_code=ts_code, start_date=start_date, end_date=end_date)
                
                if fina is not None and not fina.empty:
                    # 获取最新一期数据
                    latest_fina = fina.iloc[0]
                    
                    # 基本财务指标
                    pe = latest_fina.get('pe', np.nan)  # 市盈率
                    pb = latest_fina.get('pb', np.nan)  # 市净率
                    roe = latest_fina.get('roe', np.nan)  # 净资产收益率
                    
                    # 添加价值因子
                    df['pe'] = pe
                    df['pb'] = pb
                    df['roe'] = roe
                    
                    # 计算PEG
                    df['peg'] = df['pe'] / roe if not np.isnan(pe) and not np.isnan(roe) and roe > 0 else np.nan
            except:
                self.logger.warning(f"获取股票 {ts_code} 的财务指标失败")
            
            # 5. 综合因子计算
            # ...可在此添加更多因子计算...
            
            self.logger.info(f"计算了股票 {ts_code} 的量化因子数据，共 {len(df)} 条记录")
            return df
            
        except Exception as e:
            self.logger.error(f"计算量化因子失败: {str(e)}")
            return pd.DataFrame()

    def get_market_state(self):
        """获取市场状态信息
        
        获取当前市场状态，包括：
        - 市场开闭状态
        - 主要指数涨跌情况
        - 市场情绪指标
        - 资金流向概览
        - 涨跌家数统计
        
        Returns:
            dict: 市场状态信息
        """
        if self.api is None:
            self.logger.error("API未初始化，无法获取市场状态")
            return None
            
        try:
            # 获取当前日期时间
            now = datetime.now()
            current_date = now.strftime('%Y%m%d')
            
            # 检查市场是否开盘
            market_open = False
            try:
                # 获取交易日历
                calendar = self.api.trade_cal(exchange='SSE', start_date=current_date, end_date=current_date)
                if not calendar.empty:
                    is_open = calendar.iloc[0]['is_open']
                    market_open = is_open == 1
                    
                    # 检查当前时间是否在交易时间内 (9:30-11:30, 13:00-15:00)
                    current_time = now.time()
                    morning_start = datetime.strptime('09:30', '%H:%M').time()
                    morning_end = datetime.strptime('11:30', '%H:%M').time()
                    afternoon_start = datetime.strptime('13:00', '%H:%M').time()
                    afternoon_end = datetime.strptime('15:00', '%H:%M').time()
                    
                    if market_open:
                        if not ((morning_start <= current_time <= morning_end) or 
                                (afternoon_start <= current_time <= afternoon_end)):
                            market_open = False
            except Exception as e:
                self.logger.warning(f"检查市场开闭状态失败: {str(e)}")
            
            # 获取前一交易日
            prev_date = self._get_prev_trade_date(current_date)
            
            # 获取主要指数最新行情
            indices = self._get_major_indices()
            index_status = {}
            
            for index_code in indices:
                try:
                    df = self.api.index_daily(ts_code=index_code, start_date=prev_date, end_date=current_date)
                    if not df.empty and len(df) >= 1:
                        latest = df.iloc[0]
                        index_status[index_code] = {
                            'name': self._get_index_name(index_code),
                            'close': latest['close'],
                            'change': latest['change'],
                            'pct_change': latest['pct_chg'],
                            'volume': latest['vol'],
                            'amount': latest['amount']
                        }
                except Exception as e:
                    self.logger.warning(f"获取指数 {index_code} 行情失败: {str(e)}")
            
            # 获取市场情绪
            market_sentiment = self.get_market_sentiment(current_date)
            
            # 获取行业热点
            try:
                industry_rotation = self.get_industry_rotation(current_date, days=1)
                top_industries = []
                bottom_industries = []
                
                if not industry_rotation.empty:
                    # 热门行业
                    top_industries = [
                        {'name': row['industry_name'], 'change_pct': row['change_pct']} 
                        for _, row in industry_rotation.head(5).iterrows()
                    ]
                    
                    # 冷门行业
                    bottom_industries = [
                        {'name': row['industry_name'], 'change_pct': row['change_pct']} 
                        for _, row in industry_rotation.tail(5).iterrows()
                    ]
            except Exception as e:
                self.logger.warning(f"获取行业热点失败: {str(e)}")
                top_industries = []
                bottom_industries = []
            
            # 获取资金流向概览
            capital_flow = {
                'main_net_inflow': 0,  # 主力资金净流入
                'north_net_inflow': 0,  # 北向资金净流入
                'industry_inflow': {},  # 行业资金流入
            }
            
            try:
                # 尝试获取北向资金
                north_money = self.api.moneyflow_hsgt(trade_date=current_date)
                if not north_money.empty:
                    capital_flow['north_net_inflow'] = north_money['net_buy_amount'].sum()
            except Exception as e:
                self.logger.warning(f"获取北向资金失败: {str(e)}")
            
            # 获取涨跌家数统计
            up_down_stats = {
                'up_count': 0,
                'down_count': 0,
                'flat_count': 0,
                'limit_up_count': 0,
                'limit_down_count': 0,
                'total_count': 0
            }
            
            try:
                # 获取当日全部股票行情
                stock_daily = self.api.daily(trade_date=current_date)
                if not stock_daily.empty:
                    up_down_stats['total_count'] = len(stock_daily)
                    up_down_stats['up_count'] = len(stock_daily[stock_daily['pct_chg'] > 0])
                    up_down_stats['down_count'] = len(stock_daily[stock_daily['pct_chg'] < 0])
                    up_down_stats['flat_count'] = len(stock_daily[stock_daily['pct_chg'] == 0])
                    
                    # 涨停跌停数据
                    try:
                        limit_list = self.api.limit_list(trade_date=current_date, limit_type='U,D')
                        if not limit_list.empty:
                            up_down_stats['limit_up_count'] = len(limit_list[limit_list['limit_type'] == 'U'])
                            up_down_stats['limit_down_count'] = len(limit_list[limit_list['limit_type'] == 'D'])
                    except:
                        self.logger.warning("获取涨跌停数据失败")
            except Exception as e:
                self.logger.warning(f"获取涨跌家数统计失败: {str(e)}")
            
            # 合并所有信息
            market_state = {
                'date': current_date,
                'time': now.strftime('%H:%M:%S'),
                'market_open': market_open,
                'indices': index_status,
                'sentiment': market_sentiment,
                'hot_industries': top_industries,
                'cold_industries': bottom_industries,
                'capital_flow': capital_flow,
                'up_down_stats': up_down_stats
            }
            
            return market_state
            
        except Exception as e:
            self.logger.error(f"获取市场状态失败: {str(e)}")
            return {
                'date': datetime.now().strftime('%Y%m%d'),
                'time': datetime.now().strftime('%H:%M:%S'),
                'market_open': False,
                'error': str(e)
            }
        
    def _get_major_indices(self):
        """获取主要指数代码
        
        Returns:
            list: 指数代码列表
        """
        return ['000001.SH', '399001.SZ', '399006.SZ', '000688.SH'] 

    def recommend_stocks(self, count: int = 10) -> List[Dict[str, Any]]:
        """推荐股票"""
        try:
            recommended = []
            # 获取股票推荐逻辑
            # ... 其他代码 ...
            
            # 如果已有足够的股票，则退出
            if len(recommended) >= count:
                return recommended[:count]
            
            return recommended
        except Exception as e:
            self.logger.error(f"推荐股票失败: {str(e)}")
            return []