#!/usr/bin/env python3
"""
超神量子共生系统 - 风险管理器
负责交易风险控制和限制管理
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np

# 设置日志
logger = logging.getLogger("TradingEngine.RiskManager")

class RiskManager:
    """风险管理器 - 负责交易风险控制"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化风险管理器
        
        参数:
            config: 配置参数
        """
        self.config = config or {}
        
        # 设置默认风险参数
        self.risk_params = {
            # 单笔交易限制
            "max_single_order_amount": self.config.get("max_single_order_amount", 100000.0),  # 单笔最大交易金额
            "max_single_order_quantity": self.config.get("max_single_order_quantity", 100000),  # 单笔最大交易数量
            
            # 净暴露限制
            "max_net_exposure": self.config.get("max_net_exposure", 0.8),  # 最大净暴露度 (占总资产比例)
            "max_single_symbol_exposure": self.config.get("max_single_symbol_exposure", 0.2),  # 单一标的最大暴露度
            
            # 行业和板块限制
            "max_sector_exposure": self.config.get("max_sector_exposure", 0.3),  # 单一行业最大暴露度
            "max_industry_count": self.config.get("max_industry_count", 5),  # 最大行业数量
            
            # 波动性和风险度量
            "max_drawdown_limit": self.config.get("max_drawdown_limit", 0.1),  # 最大回撤限制
            "volatility_multiplier": self.config.get("volatility_multiplier", 0.5),  # 波动率乘数
            
            # 时间和频率限制
            "min_order_interval": self.config.get("min_order_interval", 5),  # 最小下单间隔(秒)
            "max_daily_trades": self.config.get("max_daily_trades", 50),  # 每日最大交易次数
            "max_daily_turnover": self.config.get("max_daily_turnover", 3.0),  # 每日最大换手率
            
            # 算法交易限制
            "algorithm_trade_limit": self.config.get("algorithm_trade_limit", 0.7),  # 算法交易限制占比
            "quantum_risk_factor": self.config.get("quantum_risk_factor", 0.5),  # 量子风险因子
            
            # 流动性限制
            "min_liquidity_factor": self.config.get("min_liquidity_factor", 0.02),  # 最小流动性因子
            "max_market_impact": self.config.get("max_market_impact", 0.01)  # 最大市场影响
        }
        
        # 风险统计和状态
        self.risk_stats = {
            "last_order_time": None,
            "daily_trade_count": 0,
            "daily_turnover": 0.0,
            "max_drawdown": 0.0,
            "current_exposure": 0.0,
            "sector_exposure": {},
            "symbol_exposure": {},
            "risk_warnings": [],
            "risk_limit_breaches": []
        }
        
        # 账户初始值和峰值
        self.account_peak = 0.0
        self.daily_values = []
        
        # 风险监控标志
        self.monitoring_active = True
        
        logger.info("风险管理器初始化完成")
    
    def check_order_risk(self, account, symbol, direction, price, quantity) -> Tuple[bool, str]:
        """
        检查订单风险
        
        参数:
            account: 账户对象
            symbol: 股票代码
            direction: 交易方向
            price: 价格
            quantity: 数量
            
        返回:
            Tuple[bool, str]: (是否通过风险检查, 消息)
        """
        if not self.monitoring_active:
            return True, "风险监控已停用"
        
        # 计算订单价值
        order_value = price * quantity
        
        # 检查单笔交易限制
        if order_value > self.risk_params["max_single_order_amount"]:
            message = f"单笔交易金额超过限制: {order_value:.2f} > {self.risk_params['max_single_order_amount']:.2f}"
            self._record_risk_breach("max_single_order_amount", message)
            return False, message
        
        if quantity > self.risk_params["max_single_order_quantity"]:
            message = f"单笔交易数量超过限制: {quantity} > {self.risk_params['max_single_order_quantity']}"
            self._record_risk_breach("max_single_order_quantity", message)
            return False, message
        
        # 检查下单频率
        if self.risk_stats["last_order_time"]:
            time_since_last_order = (datetime.now() - self.risk_stats["last_order_time"]).total_seconds()
            if time_since_last_order < self.risk_params["min_order_interval"]:
                message = f"下单频率过高: {time_since_last_order:.2f}秒 < {self.risk_params['min_order_interval']}秒"
                self._record_risk_breach("min_order_interval", message)
                return False, message
        
        # 检查每日交易次数
        if self.risk_stats["daily_trade_count"] >= self.risk_params["max_daily_trades"]:
            message = f"每日交易次数超过限制: {self.risk_stats['daily_trade_count']} >= {self.risk_params['max_daily_trades']}"
            self._record_risk_breach("max_daily_trades", message)
            return False, message
        
        # 检查净暴露度
        total_assets = account.total_value
        current_exposure = account.market_value / total_assets if total_assets > 0 else 0
        
        # 计算新的暴露度
        new_exposure = current_exposure
        if direction == "BUY":
            new_exposure = (account.market_value + order_value) / (total_assets)
        elif direction == "SELL":
            position = account.get_position(symbol)
            if position:
                # 计算卖出后的持仓市值
                remaining_value = position.market_value - min(position.quantity, quantity) * price
                new_exposure = (account.market_value - position.market_value + remaining_value) / total_assets
        
        if new_exposure > self.risk_params["max_net_exposure"]:
            message = f"净暴露度超过限制: {new_exposure:.2f} > {self.risk_params['max_net_exposure']:.2f}"
            self._record_risk_breach("max_net_exposure", message)
            return False, message
        
        # 检查单一标的暴露度
        symbol_exposure = 0.0
        position = account.get_position(symbol)
        if position:
            symbol_exposure = position.market_value / total_assets
        
        if direction == "BUY":
            new_symbol_exposure = (position.market_value if position else 0) + order_value
            new_symbol_exposure /= total_assets
            
            if new_symbol_exposure > self.risk_params["max_single_symbol_exposure"]:
                message = f"{symbol}暴露度超过限制: {new_symbol_exposure:.2f} > {self.risk_params['max_single_symbol_exposure']:.2f}"
                self._record_risk_breach("max_single_symbol_exposure", message)
                return False, message
        
        # 检查特征强度风险 (使用波动率乘数)
        # 实际应用中，这里可以结合具体市场波动情况进行更复杂的计算
        
        # 更新风险统计
        self.risk_stats["last_order_time"] = datetime.now()
        self.risk_stats["daily_trade_count"] += 1
        
        return True, "通过风险检查"
    
    def _record_risk_breach(self, risk_type: str, message: str):
        """记录风险违规"""
        breach = {
            "time": datetime.now(),
            "type": risk_type,
            "message": message
        }
        self.risk_stats["risk_limit_breaches"].append(breach)
        logger.warning(f"风险限制被违反: {message}")
    
    def update_account_stats(self, account):
        """
        更新账户统计信息
        
        参数:
            account: 账户对象
        """
        # 更新账户峰值
        if account.total_value > self.account_peak:
            self.account_peak = account.total_value
        
        # 计算最大回撤
        if self.account_peak > 0:
            current_drawdown = 1 - account.total_value / self.account_peak
            if current_drawdown > self.risk_stats["max_drawdown"]:
                self.risk_stats["max_drawdown"] = current_drawdown
                
                # 检查最大回撤限制
                if current_drawdown > self.risk_params["max_drawdown_limit"]:
                    message = f"最大回撤超过限制: {current_drawdown:.2f} > {self.risk_params['max_drawdown_limit']:.2f}"
                    self._add_risk_warning("max_drawdown_exceeded", message)
        
        # 更新当前暴露度
        self.risk_stats["current_exposure"] = account.market_value / account.total_value if account.total_value > 0 else 0
        
        # 更新标的暴露度
        self.risk_stats["symbol_exposure"] = {}
        for position in account.get_all_positions():
            symbol_exposure = position.market_value / account.total_value
            self.risk_stats["symbol_exposure"][position.symbol] = symbol_exposure
            
            # 检查单一标的暴露度限制
            if symbol_exposure > self.risk_params["max_single_symbol_exposure"]:
                message = f"{position.symbol}暴露度超过限制: {symbol_exposure:.2f} > {self.risk_params['max_single_symbol_exposure']:.2f}"
                self._add_risk_warning("symbol_exposure_exceeded", message)
        
        # 存储每日账户价值用于后续分析
        today = datetime.now().date()
        self.daily_values.append({
            "date": today,
            "total_value": account.total_value,
            "cash": account.cash,
            "market_value": account.market_value
        })
    
    def _add_risk_warning(self, warning_type: str, message: str):
        """添加风险警告"""
        warning = {
            "time": datetime.now(),
            "type": warning_type,
            "message": message
        }
        self.risk_stats["risk_warnings"].append(warning)
        logger.warning(f"风险警告: {message}")
    
    def get_risk_report(self) -> Dict:
        """
        获取风险报告
        
        返回:
            Dict: 风险报告字典
        """
        return {
            "risk_parameters": self.risk_params,
            "risk_statistics": self.risk_stats,
            "account_peak": self.account_peak,
            "monitoring_active": self.monitoring_active
        }
    
    def adjust_risk_parameters(self, risk_params: Dict):
        """
        调整风险参数
        
        参数:
            risk_params: 新的风险参数
        """
        for param, value in risk_params.items():
            if param in self.risk_params:
                self.risk_params[param] = value
                logger.info(f"风险参数已调整: {param} = {value}")
    
    def set_monitoring_status(self, active: bool):
        """
        设置风险监控状态
        
        参数:
            active: 是否激活风险监控
        """
        self.monitoring_active = active
        logger.info(f"风险监控状态: {'激活' if active else '停用'}")
    
    def reset_daily_stats(self):
        """重置每日统计"""
        self.risk_stats["daily_trade_count"] = 0
        self.risk_stats["daily_turnover"] = 0.0
        logger.info("每日风险统计已重置")
    
    def calculate_var(self, positions, confidence_level=0.95, time_horizon=1):
        """
        计算风险价值(Value at Risk)
        
        参数:
            positions: 持仓列表
            confidence_level: 置信水平
            time_horizon: 时间范围(天)
            
        返回:
            float: 风险价值
        """
        # 简化版VaR计算
        # 假设我们有持仓的历史收益率数据
        # 实际应用中，这里应该使用真实的历史数据和更复杂的模型
        
        # 假设一个简单的模拟
        total_value = sum(position.market_value for position in positions)
        
        # 假设波动率
        # 实际应用中应该使用历史数据计算
        volatility = 0.02  # 日波动率
        
        # 计算VaR
        z_score = np.abs(np.percentile(np.random.normal(0, 1, 10000), (1 - confidence_level) * 100))
        var = total_value * volatility * z_score * np.sqrt(time_horizon)
        
        return var

# 测试函数
def test_risk_manager():
    """测试风险管理器"""
    # 创建模拟账户
    class MockAccount:
        def __init__(self):
            self.total_value = 1000000.0
            self.cash = 500000.0
            self.market_value = 500000.0
            self.positions = []
        
        def get_all_positions(self):
            return self.positions
        
        def get_position(self, symbol):
            for pos in self.positions:
                if pos.symbol == symbol:
                    return pos
            return None
    
    class MockPosition:
        def __init__(self, symbol, quantity, market_price):
            self.symbol = symbol
            self.quantity = quantity
            self.market_price = market_price
            self.market_value = quantity * market_price
    
    # 创建模拟账户和持仓
    account = MockAccount()
    account.positions = [
        MockPosition("000001.SZ", 10000, 10.0),
        MockPosition("600000.SH", 5000, 15.0)
    ]
    account.market_value = sum(pos.market_value for pos in account.positions)
    account.total_value = account.cash + account.market_value
    
    # 创建风险管理器
    risk_manager = RiskManager()
    
    # 测试订单风险检查
    print("检查正常订单:")
    result, message = risk_manager.check_order_risk(account, "000001.SZ", "BUY", 10.0, 1000)
    print(f"结果: {result}, 消息: {message}")
    
    print("\n检查超大金额订单:")
    result, message = risk_manager.check_order_risk(account, "000001.SZ", "BUY", 10.0, 100000)
    print(f"结果: {result}, 消息: {message}")
    
    # 更新账户统计
    risk_manager.update_account_stats(account)
    
    # 获取风险报告
    risk_report = risk_manager.get_risk_report()
    print("\n风险报告:")
    for category, items in risk_report.items():
        if isinstance(items, dict):
            print(f"\n{category}:")
            for key, value in items.items():
                if not isinstance(value, list):
                    print(f"  {key}: {value}")
        else:
            print(f"{category}: {items}")
    
    return risk_manager

if __name__ == "__main__":
    # 设置日志输出
    logging.basicConfig(level=logging.INFO)
    
    # 测试风险管理器
    test_risk_manager() 