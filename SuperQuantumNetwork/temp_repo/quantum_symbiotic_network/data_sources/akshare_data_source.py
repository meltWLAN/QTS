#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - AKShare实时数据源
利用AKShare获取真实市场数据，作为TuShare的备用数据源
"""

import os
import time
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 尝试导入akshare
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    logging.warning("AKShare模块未安装，请执行 pip install akshare 安装")

# 设置缓存目录
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)


class AKShareDataSource:
    """AKShare数据源类，提供股票和市场数据，作为备用真实数据源"""
    
    def __init__(self):
        """初始化AKShare数据源"""
        self.logger = logging.getLogger("AKShareDataSource")
        self.api = None
        self.is_ready = False
        self.cache = {}
        self.stock_info_cache = {}
        
        # 创建缓存目录
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        # 初始化API
        self._init_api()
        
        # 存储股票基本信息
        self.stock_list = None
        self.indices_list = None
        
        # 记录最后更新时间
        self.last_update = {}
    
    def _init_api(self):
        """初始化AKShare API"""
        if not AKSHARE_AVAILABLE:
            self.logger.error("无法初始化AKShare API：模块未安装")
            self.api = None
            return False
        
        # 重试机制
        max_retries = 3
        retry_delay = 2  # 秒
        
        for attempt in range(max_retries):
            try:
                # AKShare不需要token，直接进行接口测试
                self.logger.info(f"测试AKShare API连接 (尝试 {attempt+1}/{max_retries})...")
                
                # 使用一个简单的接口测试
                test_data = ak.stock_zh_index_spot()
                
                if test_data is not None and not test_data.empty:
                    self.api = True  # AKShare没有显式的API对象，我们只需要确认它可用
                    self.logger.info("AKShare API初始化成功")
                    self.is_ready = True
                    return True
                else:
                    self.logger.warning(f"AKShare API测试失败 (尝试 {attempt+1}/{max_retries}): 返回空数据")
                    
            except Exception as e:
                self.logger.warning(f"AKShare API初始化失败 (尝试 {attempt+1}/{max_retries}): {str(e)}")
                
            # 如果不是最后一次尝试，则等待后重试
            if attempt < max_retries - 1:
                import time
                self.logger.info(f"等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 1.5, 10)  # 增加重试间隔，但最多10秒
        
        self.logger.error(f"AKShare API初始化失败，已尝试 {max_retries} 次")
        self.is_ready = False
        self.api = None
        return False

    def _load_cache(self, cache_name):
        """从缓存加载数据"""
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
        """保存数据到缓存"""
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
    
    def get_market_status(self):
        """获取市场状态
        
        返回市场开盘状态、时间等信息
        
        Returns:
            dict: 市场状态信息
        """
        # 基本市场状态
        status = {
            "status": "休市",
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "trading_day": False,
            "next_trading_day": None,
            "last_trading_day": None
        }
        
        try:
            if not AKSHARE_AVAILABLE:
                self.logger.error("无法获取市场状态: AKShare模块未安装")
                return status
            
            # 获取当前日期
            today = datetime.now().strftime('%Y%m%d')
            
            # 获取交易日历
            try:
                # 使用AKShare获取交易日历
                calendar_df = ak.tool_trade_date_hist_sina()
                
                if calendar_df is not None and not calendar_df.empty:
                    # 将日期格式转换为YYYYMMDD
                    calendar_df['trade_date'] = pd.to_datetime(calendar_df['trade_date']).dt.strftime('%Y%m%d')
                    
                    # 检查今天是否交易日
                    if today in calendar_df['trade_date'].values:
                        status["trading_day"] = True
                        
                        # 检查当前时间是否在交易时间内
                        now = datetime.now().time()
                        morning_start = datetime.strptime('09:30', '%H:%M').time()
                        morning_end = datetime.strptime('11:30', '%H:%M').time()
                        afternoon_start = datetime.strptime('13:00', '%H:%M').time()
                        afternoon_end = datetime.strptime('15:00', '%H:%M').time()
                        
                        if (morning_start <= now <= morning_end) or (afternoon_start <= now <= afternoon_end):
                            status["status"] = "交易中"
                        elif now < morning_start:
                            status["status"] = "未开盘"
                        elif morning_end < now < afternoon_start:
                            status["status"] = "午休"
                        else:
                            status["status"] = "已收盘"
                    
                    # 获取下一个交易日和上一个交易日
                    calendar_df = calendar_df.sort_values('trade_date')
                    today_idx = calendar_df[calendar_df['trade_date'] == today].index
                    
                    if len(today_idx) > 0:
                        today_idx = today_idx[0]
                        # 获取下一个交易日
                        if today_idx < len(calendar_df) - 1:
                            status["next_trading_day"] = calendar_df.iloc[today_idx + 1]['trade_date']
                        
                        # 获取上一个交易日
                        if today_idx > 0:
                            status["last_trading_day"] = calendar_df.iloc[today_idx - 1]['trade_date']
                    else:
                        # 如果今天不是交易日，找到最近的一个交易日
                        future_dates = calendar_df[calendar_df['trade_date'] > today]
                        past_dates = calendar_df[calendar_df['trade_date'] < today]
                        
                        if not future_dates.empty:
                            status["next_trading_day"] = future_dates.iloc[0]['trade_date']
                        
                        if not past_dates.empty:
                            status["last_trading_day"] = past_dates.iloc[-1]['trade_date']
            except Exception as e:
                self.logger.warning(f"获取交易日历失败: {str(e)}")
        
        except Exception as e:
            self.logger.error(f"获取市场状态失败: {str(e)}")
        
        return status
    
    def get_index_data(self):
        """获取指数数据
        
        Returns:
            dict: 指数数据字典
        """
        indices = {}
        
        try:
            if not AKSHARE_AVAILABLE:
                self.logger.error("无法获取指数数据: AKShare模块未安装")
                return indices
            
            # 尝试从缓存加载
            cached_indices = self._load_cache("ak_indices_data")
            if cached_indices:
                # 检查缓存是否在今天
                timestamp = cached_indices.get('timestamp', '2000-01-01')
                cache_time = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                if (datetime.now() - cache_time).total_seconds() < 3600:  # 1小时
                    self.logger.info("使用缓存的指数数据")
                    return cached_indices.get('indices', {})
            
            # 获取指数实时行情
            df = ak.stock_zh_index_spot()
            
            if df is not None and not df.empty:
                # 处理指数数据
                index_codes = {
                    "000001": "上证指数",
                    "399001": "深证成指",
                    "399006": "创业板指",
                    "000688": "科创50"
                }
                
                for code, name in index_codes.items():
                    # 找到对应的指数
                    index_row = df[df['代码'] == code]
                    if not index_row.empty:
                        row = index_row.iloc[0]
                        
                        # 创建指数数据
                        ts_code = f"{code}.{'SH' if code.startswith('0') else 'SZ'}"
                        price = row['最新价']
                        change = row['涨跌额']
                        change_pct = row['涨跌幅'] / 100  # 转换为小数
                        
                        # 生成指数对象
                        index_data = {
                            "code": ts_code,
                            "name": name,
                            "price": price,
                            "change": change,
                            "change_pct": change_pct,
                            "prev_close": price - change,
                            "open": row.get('开盘价', price - change),
                            "high": row.get('最高价', price),
                            "low": row.get('最低价', price),
                            "volume": row.get('成交量', 0),
                            "amount": row.get('成交额', 0)
                        }
                        
                        # 获取历史数据
                        try:
                            history_data = self._get_index_history(code)
                            if history_data:
                                index_data["history"] = history_data
                        except Exception as e:
                            self.logger.warning(f"获取指数 {code} 历史数据失败: {str(e)}")
                        
                        # 添加到结果
                        indices[ts_code] = index_data
                
                # 缓存结果
                cache_data = {
                    'indices': indices,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                self._save_cache("ak_indices_data", cache_data)
            
        except Exception as e:
            self.logger.error(f"获取指数数据失败: {str(e)}")
        
        return indices
    
    def _get_index_history(self, index_code):
        """获取指数历史数据
        
        Args:
            index_code: 指数代码
            
        Returns:
            dict: 历史数据
        """
        try:
            # 获取最近30天数据
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
            
            if index_code.startswith('0'):
                # 获取上证指数历史数据
                df = ak.stock_zh_index_daily(symbol=f"sh{index_code}")
            else:
                # 获取深证指数历史数据
                df = ak.stock_zh_index_daily(symbol=f"sz{index_code}")
            
            if df is not None and not df.empty:
                # 将日期列转换为字符串
                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y%m%d')
                
                # 过滤时间范围
                df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                
                # 生成历史数据
                history = {
                    "dates": df['date'].tolist(),
                    "prices": df['close'].tolist(),
                    "opens": df['open'].tolist(),
                    "highs": df['high'].tolist(),
                    "lows": df['low'].tolist(),
                    "volumes": df['volume'].tolist()
                }
                return history
            
            return None
        
        except Exception as e:
            self.logger.warning(f"获取指数 {index_code} 历史数据失败: {str(e)}")
            return None
    
    def get_recommended_stocks(self, count=10):
        """获取推荐股票列表
        
        基于成交量和涨跌幅获取市场热门股票
        
        Args:
            count: 返回的推荐股票数量
            
        Returns:
            list: 推荐股票列表
        """
        try:
            if not AKSHARE_AVAILABLE:
                self.logger.error("无法获取推荐股票: AKShare模块未安装")
                return []
            
            # 尝试从缓存加载
            cached_stocks = self._load_cache("ak_recommended_stocks")
            if cached_stocks:
                # 检查缓存是否在今天
                timestamp = cached_stocks.get('timestamp', '2000-01-01')
                cache_time = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                if (datetime.now() - cache_time).total_seconds() < 3600:  # 1小时
                    self.logger.info("使用缓存的推荐股票数据")
                    return cached_stocks.get('stocks', [])[:count]
            
            # 获取A股实时数据
            df = ak.stock_zh_a_spot_em()
            
            if df is not None and not df.empty:
                # 计算推荐分数 (基于成交量和涨跌幅)
                df['score'] = df['成交量'] * abs(df['涨跌幅']) / 100
                
                # 按分数排序，取前100名
                top_df = df.sort_values('score', ascending=False).head(100)
                
                recommended_stocks = []
                for _, row in top_df.iterrows():
                    code = row['代码']
                    name = row['名称']
                    
                    # 获取股票详细信息
                    ts_code = f"{code}.{'SH' if code.startswith('6') else 'SZ'}"
                    stock_info = self._get_stock_info(code)
                    
                    # 计算推荐度分数 (0-100)
                    recommendation_score = min(100, abs(row['涨跌幅']) * 2 + row['成交量'] / 1000000)
                    
                    # 生成推荐原因
                    if row['涨跌幅'] > 5:
                        reason = "强势上涨"
                    elif row['涨跌幅'] > 2:
                        reason = "稳步上涨"
                    elif row['涨跌幅'] > 0:
                        reason = "小幅上涨"
                    elif row['涨跌幅'] > -2:
                        reason = "企稳回升"
                    else:
                        reason = "超跌反弹机会"
                    
                    # 基于成交量
                    if row['成交量'] > 80000000:
                        reason += "，成交量显著放大"
                    elif row['成交量'] > 40000000:
                        reason += "，成交活跃"
                    
                    # 创建股票对象
                    stock = {
                        'code': code,
                        'ts_code': ts_code,
                        'name': name,
                        'price': row['最新价'],
                        'change': row['涨跌额'],
                        'change_pct': row['涨跌幅'] / 100,  # 转换为小数
                        'volume': row['成交量'],
                        'amount': row['成交额'],
                        'industry': stock_info.get('industry', ''),
                        'recommendation': recommendation_score,
                        'reason': reason
                    }
                    
                    recommended_stocks.append(stock)
                    
                    if len(recommended_stocks) >= count:
                        break
                
                # 缓存推荐结果
                cache_data = {
                    'stocks': recommended_stocks,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                self._save_cache("ak_recommended_stocks", cache_data)
                
                return recommended_stocks
            
            return []
            
        except Exception as e:
            self.logger.error(f"获取推荐股票失败: {str(e)}")
            return []
    
    def _get_stock_info(self, code):
        """获取股票基本信息
        
        Args:
            code: 股票代码
            
        Returns:
            dict: 股票基本信息
        """
        try:
            if not AKSHARE_AVAILABLE:
                return None
            
            # 从缓存获取
            if code in self.stock_info_cache:
                return self.stock_info_cache[code]
            
            # 获取股票基本信息
            df = ak.stock_individual_info_em(symbol=code)
            
            if df is not None and not df.empty:
                # 提取行业信息
                industry_row = df[df['item'] == '所处行业']
                industry = industry_row['value'].values[0] if not industry_row.empty else ""
                
                # 创建股票信息对象
                info = {
                    'code': code,
                    'name': df[df['item'] == '股票简称']['value'].values[0] if not df[df['item'] == '股票简称'].empty else "",
                    'industry': industry,
                    'market': "上海" if code.startswith('6') else "深圳",
                    'list_date': df[df['item'] == '上市时间']['value'].values[0] if not df[df['item'] == '上市时间'].empty else ""
                }
                
                # 缓存结果
                self.stock_info_cache[code] = info
                return info
            
            return None
            
        except Exception as e:
            self.logger.warning(f"获取股票 {code} 基本信息失败: {str(e)}")
            return None
    
    def get_stock_data(self, code, force_refresh=False):
        """获取单个股票数据
        
        Args:
            code: 股票代码
            force_refresh: 是否强制刷新
            
        Returns:
            dict: 股票数据
        """
        try:
            if not AKSHARE_AVAILABLE:
                self.logger.error("无法获取股票数据: AKShare模块未安装")
                return {}
            
            # 处理股票代码格式
            if "." in code:
                # 如果格式是 000001.SZ，提取数字部分
                pure_code = code.split('.')[0]
            else:
                pure_code = code
            
            # 尝试从缓存加载
            cache_key = f"ak_stock_{pure_code}"
            if not force_refresh:
                cached_stock = self._load_cache(cache_key)
                if cached_stock:
                    # 检查缓存是否新鲜
                    timestamp = cached_stock.get('timestamp', '2000-01-01')
                    cache_time = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                    if (datetime.now() - cache_time).total_seconds() < 3600:  # 1小时
                        self.logger.info(f"使用缓存的股票 {pure_code} 数据")
                        return cached_stock.get('data', {})
            
            # 获取股票实时数据
            df = ak.stock_zh_a_spot_em()
            stock_row = df[df['代码'] == pure_code]
            
            if not stock_row.empty:
                row = stock_row.iloc[0]
                
                # 创建股票对象
                ts_code = f"{pure_code}.{'SH' if pure_code.startswith('6') else 'SZ'}"
                stock_info = self._get_stock_info(pure_code)
                
                stock_data = {
                    'code': pure_code,
                    'ts_code': ts_code,
                    'name': row['名称'],
                    'price': row['最新价'],
                    'change': row['涨跌额'],
                    'change_pct': row['涨跌幅'] / 100,  # 转换为小数
                    'open': row['开盘价'],
                    'high': row['最高价'],
                    'low': row['最低价'],
                    'volume': row['成交量'],
                    'amount': row['成交额'],
                    'industry': stock_info.get('industry', '') if stock_info else '',
                    'market_cap': row.get('总市值', 0),
                    'turnover_rate': row.get('换手率', 0)
                }
                
                # 获取历史数据
                try:
                    history_data = self._get_stock_history(pure_code)
                    if history_data:
                        stock_data["history"] = history_data
                except Exception as e:
                    self.logger.warning(f"获取股票 {pure_code} 历史数据失败: {str(e)}")
                
                # 缓存结果
                cache_data = {
                    'data': stock_data,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                self._save_cache(cache_key, cache_data)
                
                return stock_data
            
            return {}
            
        except Exception as e:
            self.logger.error(f"获取股票 {code} 数据失败: {str(e)}")
            return {}
    
    def _get_stock_history(self, code):
        """获取股票历史数据
        
        Args:
            code: 股票代码
            
        Returns:
            dict: 历史数据
        """
        try:
            # 获取最近30天数据
            df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=(datetime.now() - timedelta(days=60)).strftime("%Y%m%d"), end_date=datetime.now().strftime("%Y%m%d"), adjust="qfq")
            
            if df is not None and not df.empty:
                # 转换日期格式
                df['日期'] = pd.to_datetime(df['日期']).dt.strftime('%Y%m%d')
                
                # 生成历史数据
                history = {
                    "dates": df['日期'].tolist(),
                    "prices": df['收盘'].tolist(),
                    "opens": df['开盘'].tolist(),
                    "highs": df['最高'].tolist(),
                    "lows": df['最低'].tolist(),
                    "volumes": df['成交量'].tolist()
                }
                return history
            
            return None
            
        except Exception as e:
            self.logger.warning(f"获取股票 {code} 历史数据失败: {str(e)}")
            return None
    
    def search_stocks(self, keyword, limit=10):
        """搜索股票
        
        Args:
            keyword: 搜索关键词
            limit: 最大返回数量
            
        Returns:
            list: 匹配的股票列表
        """
        try:
            if not AKSHARE_AVAILABLE:
                self.logger.error("无法搜索股票: AKShare模块未安装")
                return []
            
            # 获取A股股票列表
            df = ak.stock_zh_a_spot_em()
            
            if df is not None and not df.empty:
                # 搜索代码或名称中包含关键词的股票
                matches = df[(df['代码'].str.contains(keyword) | df['名称'].str.contains(keyword))]
                
                # 限制返回数量
                matches = matches.head(limit)
                
                # 创建结果列表
                results = []
                for _, row in matches.iterrows():
                    code = row['代码']
                    ts_code = f"{code}.{'SH' if code.startswith('6') else 'SZ'}"
                    
                    # 获取行业信息
                    stock_info = self._get_stock_info(code)
                    
                    # 创建股票对象
                    stock = {
                        'code': code,
                        'ts_code': ts_code,
                        'name': row['名称'],
                        'price': row['最新价'],
                        'change_pct': row['涨跌幅'] / 100,
                        'industry': stock_info.get('industry', '') if stock_info else ''
                    }
                    
                    results.append(stock)
                
                return results
            
            return []
            
        except Exception as e:
            self.logger.error(f"搜索股票失败: {str(e)}")
            return []
    
    def get_hot_stocks(self, count=5):
        """获取热门股票
        
        Args:
            count: 返回的热门股票数量
            
        Returns:
            list: 热门股票列表
        """
        # 获取推荐股票并按推荐度排序
        recommended = self.get_recommended_stocks(count*2)
        if recommended:
            # 按推荐度排序，取前count个
            sorted_stocks = sorted(recommended, key=lambda x: x.get('recommendation', 0), reverse=True)
            return sorted_stocks[:count] 