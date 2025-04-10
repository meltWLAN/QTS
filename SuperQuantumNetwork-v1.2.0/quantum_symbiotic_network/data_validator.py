#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - 数据验证模块
对市场数据和股票数据进行验证，确保数据质量和完整性
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger("DataValidator")

class DataValidator:
    """数据验证器，确保数据的完整性和合理性"""
    
    def __init__(self):
        """初始化数据验证器"""
        self.logger = logging.getLogger("DataValidator")
        self.logger.info("初始化数据验证模块")
        
        # 合理值范围配置
        self.price_range = (0.01, 10000.0)  # 价格合理范围
        self.change_range = (-0.2, 0.2)     # 涨跌幅合理范围（-20%到20%）
        self.volume_min = 0                 # 最小成交量
        
        # 时间相关配置
        self.max_data_age_hours = 24        # 数据最大有效期（小时）
        
        # 加载行业和股票基础信息
        self.industry_list = []
        self.stock_base_info = {}
        self._load_base_info()
    
    def _load_base_info(self):
        """加载基础参考信息"""
        try:
            # 尝试加载行业和股票基础信息（如有）
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            industry_file = os.path.join(base_dir, "cache", "industry_list.json")
            stock_info_file = os.path.join(base_dir, "cache", "stock_list.json")
            
            if os.path.exists(industry_file):
                with open(industry_file, 'r', encoding='utf-8') as f:
                    self.industry_list = json.load(f)
                    self.logger.info(f"加载了 {len(self.industry_list)} 个行业信息")
            
            if os.path.exists(stock_info_file):
                with open(stock_info_file, 'r', encoding='utf-8') as f:
                    self.stock_base_info = json.load(f)
                    self.logger.info(f"加载了 {len(self.stock_base_info)} 支股票的基础信息")
        except Exception as e:
            self.logger.error(f"加载基础信息失败: {str(e)}")
    
    def validate_market_status(self, market_status):
        """验证市场状态数据
        
        Args:
            market_status: 市场状态数据字典
            
        Returns:
            tuple: (是否有效, 问题列表)
        """
        problems = []
        
        # 检查必要字段
        required_fields = ["status", "time"]
        for field in required_fields:
            if field not in market_status:
                problems.append(f"缺少必要字段: {field}")
        
        # 检查状态值是否合法
        valid_statuses = ["交易中", "已收盘", "休市", "未开盘", "未知"]
        if "status" in market_status and market_status["status"] not in valid_statuses:
            problems.append(f"状态值无效: {market_status['status']}")
        
        # 检查时间是否合理
        if "time" in market_status:
            try:
                data_time = datetime.strptime(market_status["time"], "%Y-%m-%d %H:%M:%S")
                time_diff = datetime.now() - data_time
                if time_diff.total_seconds() > self.max_data_age_hours * 3600:
                    problems.append(f"数据过期: {market_status['time']}")
            except Exception as e:
                problems.append(f"时间格式错误: {market_status['time']}")
        
        # 返回验证结果
        is_valid = len(problems) == 0
        return (is_valid, problems)
    
    def validate_indices_data(self, indices_data):
        """验证指数数据
        
        Args:
            indices_data: 指数数据列表
            
        Returns:
            tuple: (是否有效, 问题列表)
        """
        problems = []
        
        # 检查是否为空
        if not indices_data:
            problems.append("指数数据为空")
            return (False, problems)
        
        # 检查每个指数数据
        for index in indices_data:
            # 检查必要字段
            required_fields = ["code", "name", "price"]
            for field in required_fields:
                if field not in index:
                    problems.append(f"指数 {index.get('code', '未知')} 缺少必要字段: {field}")
            
            # 检查价格是否合理
            if "price" in index:
                price = index["price"]
                if not isinstance(price, (int, float)) or price < self.price_range[0] or price > self.price_range[1]:
                    problems.append(f"指数 {index.get('code', '未知')} 价格不合理: {price}")
            
            # 检查涨跌幅是否合理
            if "change" in index:
                change = index["change"]
                if not isinstance(change, (int, float)) or change < self.change_range[0] or change > self.change_range[1]:
                    problems.append(f"指数 {index.get('code', '未知')} 涨跌幅不合理: {change}")
        
        # 检查必要的主要指数是否存在
        major_indices = ["000001.SH", "399001.SZ", "399006.SZ"]
        existing_codes = [index.get("code") for index in indices_data]
        for code in major_indices:
            if code not in existing_codes:
                problems.append(f"缺少主要指数: {code}")
        
        # 返回验证结果
        is_valid = len(problems) == 0
        return (is_valid, problems)
    
    def validate_stock_data(self, stock_data):
        """验证单个股票数据
        
        Args:
            stock_data: 股票数据字典
            
        Returns:
            tuple: (是否有效, 问题列表)
        """
        problems = []
        
        # 检查必要字段
        required_fields = ["code", "name", "price"]
        for field in required_fields:
            if field not in stock_data:
                problems.append(f"缺少必要字段: {field}")
        
        # 检查价格是否合理
        if "price" in stock_data:
            price = stock_data["price"]
            if not isinstance(price, (int, float)) or price < self.price_range[0] or price > self.price_range[1]:
                problems.append(f"价格不合理: {price}")
        
        # 检查涨跌幅是否合理
        if "change" in stock_data:
            change = stock_data["change"]
            if not isinstance(change, (int, float)) or change < self.change_range[0] or change > self.change_range[1]:
                problems.append(f"涨跌幅不合理: {change}")
        
        # 检查成交量是否合理
        if "volume" in stock_data:
            volume = stock_data["volume"]
            if not isinstance(volume, (int, float)) or volume < self.volume_min:
                problems.append(f"成交量不合理: {volume}")
        
        # 检查行业是否有效
        if "industry" in stock_data and stock_data["industry"] and self.industry_list:
            if stock_data["industry"] not in self.industry_list:
                problems.append(f"未知行业: {stock_data['industry']}")
        
        # 返回验证结果
        is_valid = len(problems) == 0
        return (is_valid, problems)
    
    def validate_recommended_stocks(self, stocks, min_count=5):
        """验证推荐股票列表
        
        Args:
            stocks: 推荐股票列表
            min_count: 最小推荐数量
            
        Returns:
            tuple: (是否有效, 问题列表, 有效的股票列表)
        """
        problems = []
        valid_stocks = []
        
        # 检查是否为空或数量不足
        if not stocks:
            problems.append("推荐股票列表为空")
            return (False, problems, valid_stocks)
        
        if len(stocks) < min_count:
            problems.append(f"推荐股票数量不足: {len(stocks)}/{min_count}")
        
        # 检查每只股票
        for stock in stocks:
            stock_valid, stock_problems = self.validate_stock_data(stock)
            if not stock_valid:
                problems.append(f"股票 {stock.get('code', '未知')} 存在问题: {', '.join(stock_problems)}")
            else:
                valid_stocks.append(stock)
        
        # 返回验证结果
        is_valid = len(problems) == 0 and len(valid_stocks) >= min_count
        return (is_valid, problems, valid_stocks)
    
    def validate_historical_data(self, history_data, min_days=30):
        """验证历史数据
        
        Args:
            history_data: 历史数据字典或DataFrame
            min_days: 最小天数
            
        Returns:
            tuple: (是否有效, 问题列表)
        """
        problems = []
        
        # 转换为DataFrame（如果是字典）
        df = None
        if isinstance(history_data, dict):
            try:
                dates = history_data.get("dates", [])
                prices = history_data.get("prices", [])
                df = pd.DataFrame({"date": dates, "price": prices})
            except Exception as e:
                problems.append(f"无法转换历史数据: {str(e)}")
                return (False, problems)
        elif isinstance(history_data, pd.DataFrame):
            df = history_data
        else:
            problems.append(f"不支持的历史数据类型: {type(history_data)}")
            return (False, problems)
        
        # 检查数据长度
        if len(df) < min_days:
            problems.append(f"历史数据天数不足: {len(df)}/{min_days}")
        
        # 检查数据完整性和连续性
        if "date" in df.columns and len(df) > 1:
            try:
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date")
                
                # 检查日期间隔
                date_diff = df["date"].diff().dropna()
                if date_diff.max().total_seconds() > 7 * 24 * 3600:  # 最大间隔超过7天
                    problems.append(f"历史数据存在较大间隔: {date_diff.max().days}天")
            except Exception as e:
                problems.append(f"日期检查失败: {str(e)}")
        
        # 检查价格有效性
        if "price" in df.columns:
            if df["price"].isnull().any():
                problems.append("历史价格数据存在空值")
            
            # 检查异常值
            mean = df["price"].mean()
            std = df["price"].std()
            outliers = df[(df["price"] < mean - 3*std) | (df["price"] > mean + 3*std)]
            if len(outliers) > 0:
                problems.append(f"历史价格存在{len(outliers)}个异常值")
        
        # 返回验证结果
        is_valid = len(problems) == 0
        return (is_valid, problems)
    
    def repair_data(self, data_type, data, repair_mode="conservative"):
        """尝试修复数据问题
        
        Args:
            data_type: 数据类型("market_status", "index", "stock", "recommended_stocks")
            data: 要修复的数据
            repair_mode: 修复模式("conservative"保守, "aggressive"激进)
            
        Returns:
            修复后的数据
        """
        if data_type == "market_status":
            return self._repair_market_status(data, repair_mode)
        elif data_type == "index":
            return self._repair_index_data(data, repair_mode)
        elif data_type == "stock":
            return self._repair_stock_data(data, repair_mode)
        elif data_type == "recommended_stocks":
            return self._repair_recommended_stocks(data, repair_mode)
        else:
            self.logger.warning(f"不支持的数据类型: {data_type}")
            return data
    
    def _repair_market_status(self, market_status, repair_mode):
        """修复市场状态数据"""
        repaired = market_status.copy()
        
        # 填充缺失字段
        if "status" not in repaired or not repaired["status"]:
            repaired["status"] = "未知"
        
        if "time" not in repaired or not repaired["time"]:
            repaired["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if "trading_day" not in repaired:
            # 根据当前时间判断
            now = datetime.now()
            repaired["trading_day"] = (now.weekday() < 5)  # 周一至周五视为交易日
        
        return repaired
    
    def _repair_index_data(self, index_data, repair_mode):
        """修复指数数据"""
        if not index_data:
            return index_data
        
        repaired = index_data.copy()
        
        # 如果是字典类型（单个指数）
        if isinstance(repaired, dict):
            # 填充缺失字段
            if "name" not in repaired and "code" in repaired:
                code_to_name = {
                    "000001.SH": "上证指数",
                    "399001.SZ": "深证成指",
                    "399006.SZ": "创业板指",
                    "000688.SH": "科创50"
                }
                repaired["name"] = code_to_name.get(repaired["code"], f"指数{repaired['code']}")
            
            # 修正价格
            if "price" in repaired:
                price = repaired["price"]
                if not isinstance(price, (int, float)) or price < self.price_range[0]:
                    repaired["price"] = 3000.0  # 默认值
            else:
                repaired["price"] = 3000.0
            
            # 修正涨跌幅
            if "change" in repaired:
                change = repaired["change"]
                if not isinstance(change, (int, float)) or change < self.change_range[0] or change > self.change_range[1]:
                    repaired["change"] = 0.0
            else:
                repaired["change"] = 0.0
        
        # 如果是列表类型（多个指数）
        elif isinstance(repaired, list):
            for i, index in enumerate(repaired):
                repaired[i] = self._repair_index_data(index, repair_mode)
        
        return repaired
    
    def _repair_stock_data(self, stock_data, repair_mode):
        """修复股票数据"""
        if not stock_data:
            return stock_data
        
        repaired = stock_data.copy()
        
        # 填充缺失字段
        if "name" not in repaired and "code" in repaired:
            code = repaired["code"]
            # 从基础信息中查找名称
            if code in self.stock_base_info:
                repaired["name"] = self.stock_base_info[code].get("name", f"股票{code}")
            else:
                repaired["name"] = f"股票{code}"
        
        # 修正价格
        if "price" in repaired:
            price = repaired["price"]
            if not isinstance(price, (int, float)) or price < self.price_range[0]:
                repaired["price"] = 10.0  # 默认值
        else:
            repaired["price"] = 10.0
        
        # 修正涨跌幅
        if "change" in repaired:
            change = repaired["change"]
            if not isinstance(change, (int, float)) or change < self.change_range[0] or change > self.change_range[1]:
                repaired["change"] = 0.0
        else:
            repaired["change"] = 0.0
        
        # 修正成交量
        if "volume" in repaired:
            volume = repaired["volume"]
            if not isinstance(volume, (int, float)) or volume < self.volume_min:
                repaired["volume"] = 10000.0  # 默认值
        
        return repaired
    
    def _repair_recommended_stocks(self, stocks, repair_mode):
        """修复推荐股票列表"""
        if not stocks:
            return stocks
        
        repaired = []
        for stock in stocks:
            fixed_stock = self._repair_stock_data(stock, repair_mode)
            repaired.append(fixed_stock)
        
        return repaired

# 创建单例实例
validator = DataValidator()

# 提供便捷的验证函数
def validate_market_status(data):
    return validator.validate_market_status(data)

def validate_indices_data(data):
    return validator.validate_indices_data(data)

def validate_stock_data(data):
    return validator.validate_stock_data(data)

def validate_recommended_stocks(data, min_count=5):
    return validator.validate_recommended_stocks(data, min_count)

def repair_data(data_type, data, repair_mode="conservative"):
    return validator.repair_data(data_type, data, repair_mode) 