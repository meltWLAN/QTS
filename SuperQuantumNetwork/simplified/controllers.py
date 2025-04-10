#!/usr/bin/env python
"""
简化控制器模块 - 在完整控制器不可用时作为备用
"""

import logging
import pandas as pd
from datetime import datetime

logger = logging.getLogger("SimpleControllers")

class SimpleDataController:
    """简化数据控制器，提供基本的数据获取功能"""
    
    def __init__(self, data_connector):
        """
        初始化数据控制器
        
        参数:
            data_connector: 数据连接器实例
        """
        self.data_connector = data_connector
        self.name = "SimpleDataController"
        self.cache = {}
        logger.info("简化数据控制器已初始化")
    
    def get_stock_data(self, symbol, start_date=None, end_date=None):
        """
        获取股票数据
        
        参数:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        返回:
            DataFrame: 股票数据
        """
        logger.info(f"获取股票 {symbol} 的数据")
        cache_key = f"{symbol}_{start_date}_{end_date}"
        
        # 检查缓存
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # 获取数据
        try:
            data = self.data_connector.get_daily_data(symbol, start_date, end_date)
            self.cache[cache_key] = data
            return data
        except Exception as e:
            logger.error(f"获取股票数据失败: {str(e)}")
            return pd.DataFrame()
    
    def get_index_data(self, index_code, start_date=None, end_date=None):
        """
        获取指数数据
        
        参数:
            index_code: 指数代码
            start_date: 开始日期
            end_date: 结束日期
            
        返回:
            DataFrame: 指数数据
        """
        logger.info(f"获取指数 {index_code} 的数据")
        return self.get_stock_data(index_code, start_date, end_date)
    
    def get_latest_price(self, symbol):
        """
        获取最新价格
        
        参数:
            symbol: 股票代码
            
        返回:
            float: 最新价格
        """
        logger.info(f"获取 {symbol} 的最新价格")
        
        try:
            # 简化实现：获取今天的数据或最近的数据
            today = datetime.now().strftime("%Y%m%d")
            data = self.get_stock_data(symbol, end_date=today)
            
            if data is not None and not data.empty:
                return float(data.iloc[-1]['close'])
            return None
        except Exception as e:
            logger.error(f"获取最新价格失败: {str(e)}")
            return None


class SimpleTradingController:
    """简化交易控制器，提供基本的交易功能"""
    
    def __init__(self):
        """初始化交易控制器"""
        self.name = "SimpleTradingController"
        self.data_controller = None
        self.portfolio_controller = None
        self.signal_generator = None
        self.positions = {}
        self.orders = []
        self.cash = 1000000.0  # 默认初始资金
        logger.info("简化交易控制器已初始化")
    
    def set_data_controller(self, controller):
        """
        设置数据控制器
        
        参数:
            controller: 数据控制器实例
        """
        self.data_controller = controller
        logger.info("已设置数据控制器")
    
    def set_portfolio_controller(self, controller):
        """
        设置投资组合控制器
        
        参数:
            controller: 投资组合控制器实例
        """
        self.portfolio_controller = controller
        logger.info("已设置投资组合控制器")
    
    def set_signal_generator(self, generator):
        """
        设置信号生成器
        
        参数:
            generator: 信号生成器实例
        """
        self.signal_generator = generator
        logger.info("已设置信号生成器")
    
    def place_order(self, symbol, order_type, quantity, price=None):
        """
        下单
        
        参数:
            symbol: 股票代码
            order_type: 订单类型，'BUY'或'SELL'
            quantity: 数量
            price: 价格，如果为None则使用市价
            
        返回:
            bool: 是否成功
        """
        if price is None and self.data_controller:
            price = self.data_controller.get_latest_price(symbol)
            
        if price is None:
            logger.error(f"无法获取 {symbol} 的价格，订单取消")
            return False
            
        order_value = price * quantity
        
        # 检查资金是否足够
        if order_type == 'BUY' and order_value > self.cash:
            logger.warning(f"资金不足，无法购买 {symbol}")
            return False
        
        # 创建订单
        order = {
            'symbol': symbol,
            'type': order_type,
            'quantity': quantity,
            'price': price,
            'value': order_value,
            'timestamp': datetime.now(),
            'status': 'PENDING'
        }
        
        # 保存订单
        self.orders.append(order)
        
        # 模拟执行（简化版）
        if order_type == 'BUY':
            self.cash -= order_value
            self.positions[symbol] = self.positions.get(symbol, 0) + quantity
        else:  # SELL
            self.cash += order_value
            self.positions[symbol] = self.positions.get(symbol, 0) - quantity
            
        order['status'] = 'FILLED'
        
        logger.info(f"订单执行成功: {order_type} {quantity} 股 {symbol} @ {price}")
        return True
    
    def get_positions(self):
        """
        获取当前持仓
        
        返回:
            dict: 当前持仓
        """
        return self.positions
    
    def get_cash(self):
        """
        获取当前现金
        
        返回:
            float: 当前现金
        """
        return self.cash
    
    def get_portfolio_value(self):
        """
        获取投资组合价值
        
        返回:
            float: 投资组合价值
        """
        value = self.cash
        
        # 计算持仓市值
        if self.data_controller:
            for symbol, quantity in self.positions.items():
                price = self.data_controller.get_latest_price(symbol)
                if price:
                    value += price * quantity
        
        return value 