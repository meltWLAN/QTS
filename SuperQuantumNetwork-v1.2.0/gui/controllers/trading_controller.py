#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - 交易控制器
实现交易功能与交易策略
"""

import logging
import random
import uuid
import threading
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# 自定义账户和订单类
class Account:
    """账户模型"""
    
    def __init__(self):
        """初始化账户"""
        self.id = str(uuid.uuid4())[:8]
        self.name = "超神量子交易账户"
        self.cash = 1000000.0
        self.market_value = 0.0
        self.total_value = self.cash + self.market_value
        self.margin = 0.0
        self.risk_level = "低风险"
        self.created_at = datetime.now()
        self.positions = {}  # 存储持仓
        
    @property
    def profit_ratio(self):
        """获取盈利率"""
        return (self.total_value - 1000000.0) / 1000000.0
        
    def update_market_value(self, market_value):
        """更新市值"""
        self.market_value = market_value
        self.total_value = self.cash + self.market_value
        
    def to_dict(self):
        """转换为字典"""
        return {
            "id": self.id,
            "name": self.name,
            "cash": self.cash,
            "market_value": self.market_value,
            "total_value": self.total_value,
            "margin": self.margin,
            "risk_level": self.risk_level,
            "profit_ratio": self.profit_ratio,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def buy_stock(self, code, name, price, quantity):
        """买入股票
        
        Args:
            code: 股票代码
            name: 股票名称
            price: 价格
            quantity: 数量
            
        Returns:
            tuple: (成功标志, 消息)
        """
        # 检查资金是否足够
        cost = price * quantity
        if cost > self.cash:
            return False, "可用资金不足"
        
        # 更新资金
        self.cash -= cost
        
        # 更新持仓
        if code in self.positions:
            position = self.positions[code]
            # 计算新的成本
            total_cost = position.cost * position.quantity + price * quantity
            position.quantity += quantity
            if position.quantity > 0:
                position.cost = total_cost / position.quantity
            position.update_market_price(price)
        else:
            # 创建新持仓
            position = Position(code, name, quantity, price)
            self.positions[code] = position
        
        # 更新市值
        self._update_market_value()
        return True, "买入成功"
    
    def sell_stock(self, code, price, quantity):
        """卖出股票
        
        Args:
            code: 股票代码
            price: 价格
            quantity: 数量
            
        Returns:
            tuple: (成功标志, 消息)
        """
        # 检查持仓是否足够
        if code not in self.positions or self.positions[code].quantity < quantity:
            return False, "可用持仓不足"
        
        # 获取持仓
        position = self.positions[code]
        
        # 计算收益
        profit = (price - position.cost) * quantity
        
        # 更新持仓
        position.quantity -= quantity
        position.update_market_price(price)
        
        # 如果卖空了，移除持仓
        if position.quantity <= 0:
            del self.positions[code]
        
        # 更新资金
        self.cash += price * quantity
        
        # 更新市值
        self._update_market_value()
        return True, f"卖出成功，收益: {profit:.2f}"
    
    def _update_market_value(self):
        """更新市值"""
        market_value = 0.0
        for position in self.positions.values():
            market_value += position.market_price * position.quantity
        
        self.market_value = market_value
        self.total_value = self.cash + self.market_value

    def get_position_list(self):
        """获取持仓列表
        
        Returns:
            list: 持仓列表
        """
        return [position.to_dict() for position in self.positions.values()]

class Position:
    """持仓模型"""
    
    def __init__(self, code, name, quantity, price):
        """初始化持仓"""
        self.id = str(uuid.uuid4())[:8]
        self.code = code
        self.name = name
        self.quantity = quantity
        self.price = price
        self.cost = price
        self.market_price = price
        self.profit = 0.0
        self.profit_ratio = 0.0
        self.created_at = datetime.now()
        
    def update_market_price(self, price):
        """更新市价"""
        self.market_price = price
        self.profit = (self.market_price - self.cost) * self.quantity
        self.profit_ratio = (self.market_price - self.cost) / self.cost if self.cost != 0 else 0.0
        
    def to_dict(self):
        """转换为字典"""
        return {
            "id": self.id,
            "code": self.code,
            "name": self.name,
            "quantity": self.quantity,
            "price": self.price,
            "cost": self.cost,
            "market_price": self.market_price,
            "profit": self.profit,
            "profit_ratio": self.profit_ratio,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S")
        }

class Order:
    """订单模型"""
    
    def __init__(self, code, name, direction, price, quantity, order_type="限价单"):
        """初始化订单"""
        self.id = str(uuid.uuid4())[:8]
        self.code = code
        self.name = name
        self.direction = direction  # "买入" 或 "卖出"
        self.price = price
        self.quantity = quantity
        self.order_type = order_type
        self.status = "未成交"
        self.created_at = datetime.now()
        self.updated_at = self.created_at
        
    def execute(self):
        """执行订单"""
        self.status = "已成交"
        self.updated_at = datetime.now()
        
    def cancel(self):
        """取消订单"""
        self.status = "已取消"
        self.updated_at = datetime.now()
        
    def to_dict(self):
        """转换为字典"""
        # 处理created_at和updated_at可能是字符串或datetime的情况
        if isinstance(self.created_at, datetime):
            created_at_str = self.created_at.strftime("%Y-%m-%d %H:%M:%S")
        else:
            created_at_str = self.created_at
            
        if isinstance(self.updated_at, datetime):
            updated_at_str = self.updated_at.strftime("%Y-%m-%d %H:%M:%S")
        else:
            updated_at_str = self.updated_at
            
        return {
            "id": self.id,
            "code": self.code,
            "name": self.name,
            "direction": self.direction,
            "price": self.price,
            "quantity": self.quantity,
            "order_type": self.order_type,
            "status": self.status,
            "created_at": created_at_str,
            "updated_at": updated_at_str
        }

class TradingController:
    """交易控制器，负责管理交易功能"""
    
    def __init__(self):
        """初始化交易控制器"""
        self.logger = logging.getLogger("TradingController")
        self.account = Account()
        self.orders = []  # 订单历史
        self.active_orders = []  # 活跃订单
        self.quantum_network = None
        self.symbiosis_core = None
        self._signals = []  # 使用_signals而不是signals
        self.trading_signals = []  # 量子交易信号
        self.signal_threshold = 0.65  # 信号阈值
        
        # 超神级交易功能增强
        self.quantum_trading = {
            "enabled": True,
            "coherence": 0.95,
            "dimension_channels": 9,
            "prediction_confidence": 0.88,
            "execution_precision": 0.92,
            "risk_adaptation": 0.85,
            "learning_rate": 0.75
        }
        
        # 高维市场状态感知
        self.market_state = {
            "volatility": 0.0,
            "trend": 0.0,
            "rhythm": 0.0,
            "energy": 0.0,
            "pattern_stability": 0.0,
            "non_linear_factor": 0.0
        }
        
        # 量子策略引擎
        self.strategy_engine = {
            "active_strategies": [],
            "strategy_weights": {},
            "quantum_signal_strength": 0.0,
            "adaptation_level": 0.0,
            "market_fit_score": 0.0
        }
        
        # 风险管理系统
        self.risk_system = {
            "max_position_size": 0.2,  # 单一仓位上限
            "max_sector_exposure": 0.4,  # 单一行业上限
            "stop_loss_ratio": 0.15,  # 止损比例
            "dynamic_risk_budget": 0.05,  # 动态风险预算
            "correlation_threshold": 0.75,  # 相关性阈值
            "dimension_risk_factor": 0.4  # 维度风险因子
        }
        
        # 尝试初始化量子网络
        try:
            # 动态导入get_quantum_symbiotic_network函数
            from hyperunity_activation import get_quantum_symbiotic_network
            self.quantum_network = get_quantum_symbiotic_network()
            if self.quantum_network:
                self.logger.info("量子网络初始化成功")
                
                # 连接量子预测器
                try:
                    from quantum_symbiotic_network.quantum_prediction import get_predictor
                    self.quantum_predictor = get_predictor()
                    if self.quantum_predictor:
                        self.logger.info("超神量子预测器连接成功")
                except Exception as e:
                    self.logger.error(f"连接量子预测器失败: {str(e)}")
                    self.quantum_predictor = None
        except Exception as e:
            self.logger.error(f"初始化量子网络时出错: {str(e)}")
            self.quantum_predictor = None
        
        # 启动自动交易信号线程
        self.signal_active = True
        self.signal_thread = threading.Thread(target=self._run_signal_generator)
        self.signal_thread.daemon = True
        self.signal_thread.start()
        
        # 启动风险监控线程
        self.risk_active = True
        self.risk_thread = threading.Thread(target=self._run_risk_monitor)
        self.risk_thread.daemon = True
        self.risk_thread.start()
        
        self.logger.info("超神交易控制器初始化完成")
    
    def place_order(self, code, name, direction, price, quantity, order_type="限价单", strategy_id=None):
        """下单 - 超神增强版"""
        try:
            # 创建订单
            order = Order(code, name, direction, price, quantity, order_type)
            
            # 设置策略ID
            if strategy_id:
                order.strategy_id = strategy_id
            
            # 添加量子增强属性
            if self.quantum_trading["enabled"]:
                # 量子信心度
                if self.quantum_predictor:
                    try:
                        # 获取股票预测
                        prediction = self.quantum_predictor.predict_stock(code)
                        if prediction and 'confidence' in prediction:
                            order.quantum_confidence = prediction['confidence']
                        else:
                            order.quantum_confidence = self.quantum_trading["prediction_confidence"]
                    except:
                        order.quantum_confidence = self.quantum_trading["prediction_confidence"]
                else:
                    order.quantum_confidence = self.quantum_trading["prediction_confidence"]
                
                # 时机精度
                order.timing_precision = self.quantum_trading["execution_precision"]
                
                # 维度适应性
                order.dimension_fit = random.uniform(0.7, 0.95)
            
            # 风险检查
            risk_check, risk_message = self._check_order_risk(code, direction, price, quantity)
            if not risk_check:
                order.status = "已拒绝"
                order.message = f"风险检查: {risk_message}"
                order.updated_at = datetime.now()
                self.orders.append(order)
                return False, risk_message, order.to_dict()
                
            # 检查账户状态
            if direction == "买入":
                cost = price * quantity
                if cost > self.account.cash:
                    order.status = "已拒绝"
                    order.updated_at = datetime.now()
                    self.orders.append(order)
                    return False, "可用资金不足", order.to_dict()
            elif direction == "卖出":
                if code not in self.account.positions or self.account.positions[code].quantity < quantity:
                    order.status = "已拒绝"
                    order.updated_at = datetime.now()
                    self.orders.append(order)
                    return False, "可用持仓不足", order.to_dict()
            
            # 添加订单
            self.orders.append(order)
            self.active_orders.append(order)
            
            # 超神级交易增强: 寻找最佳执行时机
            if self.quantum_trading["enabled"] and order.quantum_confidence > 0.7:
                # 异步处理订单，寻找最佳执行时机
                threading.Thread(target=self._find_optimal_execution, args=(order,)).start()
                return True, "下单成功 (超神级执行优化中...)", order.to_dict()
            else:
                # 常规处理 - 模拟成交
                self._process_order(order)
                return True, "下单成功", order.to_dict()
                
        except Exception as e:
            self.logger.error(f"下单失败: {e}")
            return False, f"下单失败: {str(e)}", None
    
    def _process_order(self, order):
        """处理订单（模拟成交）"""
        # 模拟成交率
        fill_prob = 0.8
        
        if np.random.random() < fill_prob:
            # 模拟成交
            if order.direction == "买入":
                success, message = self.account.buy_stock(order.code, order.name, order.price, order.quantity)
                if success:
                    order.execute()
                    if order not in self.active_orders:
                        self.active_orders.remove(order)
            elif order.direction == "卖出":
                success, message = self.account.sell_stock(order.code, order.price, order.quantity)
                if success:
                    order.execute()
                    if order in self.active_orders:
                        self.active_orders.remove(order)
    
    def cancel_order(self, order_id):
        """撤单"""
        try:
            # 查找订单
            order = None
            for o in self.orders:
                if o.id == order_id:
                    order = o
                    break
            
            if not order:
                return False, "订单不存在"
            
            # 如果已经成交，不能撤单
            if order.status == "已成交":
                return False, "订单已成交，无法撤单"
            
            # 撤销订单
            if order.cancel():
                if order in self.active_orders:
                    self.active_orders.remove(order)
                return True, "撤单成功"
            else:
                return False, "撤单失败"
        except Exception as e:
            self.logger.error(f"撤单失败: {e}")
            return False, f"撤单失败: {str(e)}"
    
    def get_order_list(self):
        """获取订单列表"""
        return [order.to_dict() for order in self.orders]
    
    def get_orders(self):
        """获取订单列表（get_order_list的别名）"""
        return self.get_order_list()
    
    def get_active_order_list(self):
        """获取活跃订单列表"""
        return [order.to_dict() for order in self.active_orders]
    
    def get_stock_recommendation(self, code=None):
        """获取股票推荐"""
        if self.quantum_network:
            # 如果有量子网络，使用量子网络做推荐
            try:
                recommendation = self.quantum_network.recommend_stock(code)
                return recommendation
            except Exception as e:
                self.logger.error(f"获取股票推荐失败: {e}")
        
        # 如果没有量子网络或者调用失败，返回模拟推荐
        if code:
            return {
                "buy_probability": np.random.uniform(0.5, 0.9),
                "confidence": np.random.uniform(0.6, 0.95),
                "expected_return": np.random.uniform(0.05, 0.2)
            }
        else:
            # 返回一组推荐股票
            recommendations = []
            stock_codes = ["600000", "600036", "601398", "600519", "600276"]
            stock_names = ["浦发银行", "招商银行", "工商银行", "贵州茅台", "恒瑞医药"]
            
            for code, name in zip(stock_codes, stock_names):
                recommendations.append({
                    "code": code,
                    "name": name,
                    "buy_probability": np.random.uniform(0.5, 0.9),
                    "confidence": np.random.uniform(0.6, 0.95),
                    "expected_return": np.random.uniform(0.05, 0.2)
                })
            
            return recommendations
    
    def get_account_info(self):
        """获取账户信息"""
        return self.account.to_dict()
    
    def get_position_list(self):
        """获取持仓列表"""
        return self.account.get_position_list()
    
    def get_positions(self):
        """获取持仓列表（get_position_list的别名）"""
        return self.get_position_list()
    
    def initialize_mock_positions(self):
        """初始化模拟持仓（用于演示）"""
        # 创建一些模拟持仓
        stocks = [
            {"code": "600000", "name": "浦发银行", "price": 10.5, "quantity": 1000},
            {"code": "600036", "name": "招商银行", "price": 45.8, "quantity": 500},
            {"code": "601398", "name": "工商银行", "price": 5.3, "quantity": 2000},
            {"code": "600519", "name": "贵州茅台", "price": 1800.0, "quantity": 10},
            {"code": "600276", "name": "恒瑞医药", "price": 32.5, "quantity": 800}
        ]
        
        for stock in stocks:
            self.account.buy_stock(
                stock["code"], stock["name"], stock["price"], stock["quantity"]
            )
    
    def initialize_mock_orders(self):
        """初始化模拟订单（用于演示）"""
        # 创建一些模拟订单
        orders = [
            {"code": "600000", "name": "浦发银行", "direction": "买入", "price": 10.2, "quantity": 500},
            {"code": "600036", "name": "招商银行", "direction": "卖出", "price": 45.0, "quantity": 200},
            {"code": "601398", "name": "工商银行", "direction": "买入", "price": 5.1, "quantity": 1000},
            {"code": "600519", "name": "贵州茅台", "direction": "买入", "price": 1795.0, "quantity": 5},
            {"code": "000001", "name": "平安银行", "direction": "买入", "price": 15.8, "quantity": 800}
        ]
        
        # 设置一个较早的时间
        earlier_time = datetime.now() - timedelta(hours=random.randint(1, 48))
        
        for i, order_data in enumerate(orders):
            order = Order(
                order_data["code"],
                order_data["name"],
                order_data["direction"],
                order_data["price"],
                order_data["quantity"]
            )
            
            # 设置一些为已成交状态
            if i % 2 == 0 or np.random.random() < 0.7:
                order.execute()
            
            # 设置创建时间 - 使用datetime对象，不要转为字符串
            order_time = earlier_time - timedelta(minutes=random.randint(5, 60) * i)
            order.created_at = order_time
            order.updated_at = order_time + timedelta(minutes=random.randint(1, 10))
            
            self.orders.append(order)
            if order.status != "已成交":
                self.active_orders.append(order)
    
    def on_connect_symbiosis(self, symbiosis_core):
        """连接到共生核心"""
        try:
            self.symbiosis_core = symbiosis_core
            self.logger.info("交易控制器成功连接到共生核心")
            return True
        except Exception as e:
            self.logger.error(f"连接共生核心失败: {str(e)}")
            return False
    
    def on_disconnect_symbiosis(self):
        """断开与共生核心的连接"""
        self.symbiosis_core = None
        self.logger.info("交易控制器已断开与共生核心的连接")
        return True
    
    def shutdown(self):
        """关闭交易控制器"""
        try:
            self.signal_active = False
            self.risk_active = False
            
            if hasattr(self, "signal_thread") and self.signal_thread and self.signal_thread.is_alive():
                self.signal_thread.join(timeout=2.0)
                
            if hasattr(self, "risk_thread") and self.risk_thread and self.risk_thread.is_alive():
                self.risk_thread.join(timeout=2.0)
            
            self.logger.info("交易控制器已安全关闭")
            return True
        except Exception as e:
            self.logger.error(f"关闭交易控制器时出错: {str(e)}")
            return False
    
    def _find_optimal_execution(self, order):
        """寻找最佳执行时机 - 超神级功能"""
        try:
            # 模拟寻找最佳执行时机的过程
            self.logger.info(f"寻找最佳执行时机: {order.code} ({order.direction})")
            
            # 随机等待1-5秒，模拟寻找最佳时机
            wait_time = random.uniform(1, 5)
            time.sleep(wait_time)
            
            # 计算最优价格调整
            if order.direction == "买入":
                # 买入尝试以更低价格成交
                price_improvement = random.uniform(0, 0.02)  # 最多改善2%
                optimal_price = order.price * (1 - price_improvement)
            else:
                # 卖出尝试以更高价格成交
                price_improvement = random.uniform(0, 0.02)  # 最多改善2%
                optimal_price = order.price * (1 + price_improvement)
            
            # 更新订单价格
            original_price = order.price
            order.price = optimal_price
            
            # 处理订单
            self._process_order(order)
            
            # 记录优化结果
            if order.status == "已成交":
                improvement = abs(optimal_price - original_price) * order.quantity
                self.logger.info(f"超神执行优化: {order.code} 价格改善 {price_improvement*100:.2f}%, 节约/增加 ¥{improvement:.2f}")
            
        except Exception as e:
            self.logger.error(f"最佳执行优化失败: {str(e)}")
            # 如果优化失败，使用原始方式处理订单
            self._process_order(order)
    
    def _check_order_risk(self, code, direction, price, quantity):
        """检查订单风险 - 多维度风险管理"""
        try:
            # 计算订单价值
            order_value = price * quantity
            
            # 检查单笔交易规模
            if order_value > self.account.total_value * self.risk_system["max_position_size"]:
                return False, f"单笔交易规模超过限制 ({self.risk_system['max_position_size']*100}%)"
            
            # 检查行业敞口
            # 此处需要实际实现中添加股票所属行业的判断
            # 简化模拟: 假设每个股票代码的首字符代表行业
            sector = code[0]
            sector_exposure = 0
            
            # 计算当前行业敞口
            for pos_code, position in self.account.positions.items():
                if pos_code[0] == sector:
                    sector_exposure += position.market_price * position.quantity
            
            # 如果是买入，加上新订单价值
            if direction == "买入":
                new_sector_exposure = (sector_exposure + order_value) / self.account.total_value
                if new_sector_exposure > self.risk_system["max_sector_exposure"]:
                    return False, f"行业暴露度超过限制 ({self.risk_system['max_sector_exposure']*100}%)"
            
            # 检查相关性风险 (实际实现中可以使用真实相关性数据)
            # 此处简化模拟
            
            # 检查维度风险 (与量子预测相关)
            if self.quantum_trading["enabled"] and self.quantum_predictor:
                try:
                    dimension_risk = 1.0 - self.quantum_predictor.coherence
                    if dimension_risk > self.risk_system["dimension_risk_factor"]:
                        return False, f"维度风险过高 ({dimension_risk:.2f})"
                except:
                    pass
            
            return True, "通过风险检查"
            
        except Exception as e:
            self.logger.error(f"订单风险检查失败: {str(e)}")
            return True, "风险检查异常，默认通过"
    
    def _run_signal_generator(self):
        """运行交易信号生成器"""
        try:
            # 获取市场洞察数据
            insights = None
            if hasattr(self, 'quantum_predictor') and self.quantum_predictor:
                if hasattr(self.quantum_predictor, 'get_market_insights'):
                    insights = self.quantum_predictor.get_market_insights()
                elif hasattr(self.quantum_predictor, 'generate_market_insights'):
                    insights = self.quantum_predictor.generate_market_insights({})
            
            # 如果无法获取洞察数据，创建默认数据
            if not insights:
                insights = {
                    'market_state': {
                        'volatility': random.uniform(0.2, 0.7),
                        'trend': random.uniform(-0.5, 0.5),
                        'sentiment': random.uniform(-0.3, 0.3)
                    },
                    'predictions': {}
                }
            
            # 确保insights是字典类型
            if not isinstance(insights, dict):
                self.logger.warning(f"获取到的市场洞察不是字典类型: {type(insights)}")
                insights = {'market_state': {}, 'predictions': {}}
                
            # 获取市场状态
            market_state = insights.get('market_state', {})
            volatility = market_state.get('volatility', 0.0)
            trend = market_state.get('trend', 0.0)
            sentiment = market_state.get('sentiment', 0.0)
            
            # 获取股票预测
            predictions = insights.get('predictions', {})
            
            # 如果没有预测，添加一些模拟预测
            if not predictions:
                stocks = [
                    {"code": "600000", "name": "浦发银行"},
                    {"code": "600036", "name": "招商银行"},
                    {"code": "601398", "name": "工商银行"},
                    {"code": "600519", "name": "贵州茅台"},
                    {"code": "600276", "name": "恒瑞医药"}
                ]
                
                for stock in stocks:
                    code = stock["code"]
                    predictions[code] = {
                        'direction': 1 if random.random() > 0.5 else -1,
                        'confidence': random.uniform(0.6, 0.9),
                        'price_target': random.uniform(10, 100)
                    }
            
            # 生成交易信号
            signals = []
            
            for symbol, prediction in predictions.items():
                # 检查prediction是否为字典类型
                if not isinstance(prediction, dict):
                    continue
                    
                direction = prediction.get('direction', 0)
                confidence = prediction.get('confidence', 0)
                price_target = prediction.get('price_target', 0)
                
                # 应用量子增强
                enhanced_confidence = self._apply_quantum_enhancement(confidence)
                
                # 如果增强后的置信度大于阈值，生成信号
                if enhanced_confidence > self.signal_threshold:
                    signal = {
                        'symbol': symbol,
                        'direction': direction,
                        'confidence': enhanced_confidence,
                        'price_target': price_target,
                        'timestamp': datetime.now(),
                        'volatility_factor': volatility,
                        'quantum_adjusted': True
                    }
                    signals.append(signal)
            
            # 更新最新信号
            if signals:
                self.trading_signals = signals
                self.logger.info(f"生成了 {len(signals)} 个量子交易信号")
                
                # 触发信号更新
                if hasattr(self, 'signals_updated_signal'):
                    self.signals_updated_signal.emit(signals)
            
        except Exception as e:
            self.logger.error(f"交易信号生成失败: {str(e)}")
            # 捕获错误但允许继续运行
    
    def _update_market_state(self):
        """更新高维市场状态"""
        try:
            # 更新基本状态
            self.market_state['volatility'] = random.uniform(0.1, 0.9)
            self.market_state['trend'] = random.uniform(-0.8, 0.8)
            self.market_state['rhythm'] = random.uniform(0.2, 0.95)
            self.market_state['energy'] = random.uniform(0.3, 0.9)
            
            # 如果有量子预测器，使用其数据
            if self.quantum_predictor:
                try:
                    insights = self.quantum_predictor.generate_market_insights({})
                    if insights and 'market_sentiment' in insights:
                        sentiment = insights['market_sentiment'].get('score', 0)
                        self.market_state['energy'] = abs(sentiment) * 0.7 + 0.3
                    
                    # 使用预测器的量子参数
                    self.market_state['pattern_stability'] = self.quantum_predictor.coherence
                    
                except Exception as e:
                    self.logger.error(f"获取量子预测数据失败: {str(e)}")
                    self.market_state['pattern_stability'] = random.uniform(0.5, 0.95)
            else:
                self.market_state['pattern_stability'] = random.uniform(0.5, 0.95)
            
            # 非线性因子
            self.market_state['non_linear_factor'] = random.uniform(0.1, 0.7)
                
        except Exception as e:
            self.logger.error(f"更新市场状态失败: {str(e)}")
    
    def _run_risk_monitor(self):
        """运行风险监控"""
        while self.risk_active:
            try:
                # 检查所有持仓的止损条件
                for code, position in list(self.account.positions.items()):
                    # 计算浮动亏损比例
                    loss_ratio = -position.profit_ratio
                    
                    # 如果亏损超过止损线，自动卖出
                    if loss_ratio > self.risk_system["stop_loss_ratio"]:
                        self.logger.warning(f"触发自动止损: {code} {position.name}, 亏损率: {loss_ratio:.2%}")
                        
                        # 自动卖出
                        self.place_order(
                            code=code,
                            name=position.name,
                            direction="卖出",
                            price=position.market_price * 0.97,  # 模拟市价略低
                            quantity=position.quantity,
                            order_type="市价单",
                            strategy_id="risk_management"
                        )
                
                # 动态调整风险预算
                market_volatility = self.market_state.get('volatility', 0.5)
                pattern_stability = self.market_state.get('pattern_stability', 0.7)
                
                # 根据市场状态调整风险参数
                if market_volatility > 0.7 and pattern_stability < 0.5:
                    # 高波动、低稳定性环境 - 降低风险
                    self.risk_system["max_position_size"] = 0.1
                    self.risk_system["max_sector_exposure"] = 0.3
                    self.risk_system["stop_loss_ratio"] = 0.1
                elif market_volatility < 0.3 and pattern_stability > 0.8:
                    # 低波动、高稳定性环境 - 提高风险
                    self.risk_system["max_position_size"] = 0.25
                    self.risk_system["max_sector_exposure"] = 0.5
                    self.risk_system["stop_loss_ratio"] = 0.2
                else:
                    # 正常环境 - 平衡风险
                    self.risk_system["max_position_size"] = 0.2
                    self.risk_system["max_sector_exposure"] = 0.4
                    self.risk_system["stop_loss_ratio"] = 0.15
                
            except Exception as e:
                self.logger.error(f"风险监控运行出错: {str(e)}")
            
            # 每30秒运行一次
            time.sleep(30)
    
    def _record_signal(self, stock_code, action, confidence, source="system"):
        """记录交易信号
        
        Args:
            stock_code: 股票代码
            action: 交易动作 ('buy' 或 'sell')
            confidence: 信心水平 (0.0-1.0)
            source: 信号来源
        """
        signal = {
            "stock_code": stock_code,
            "action": action,
            "confidence": confidence,
            "source": source,
            "timestamp": datetime.now().isoformat()
        }
        
        # 添加到信号列表
        self._signals.append(signal)
        
        # 最多保留100个信号
        if len(self._signals) > 100:
            self._signals = self._signals[-100:]
            
    @property
    def signals(self):
        """获取交易信号历史
        
        Returns:
            list: 交易信号列表
        """
        if not hasattr(self, "_signals"):
            self._signals = []
        return self._signals
    
    def _apply_quantum_enhancement(self, confidence):
        """应用量子增强到置信度
        
        Args:
            confidence: 原始置信度
            
        Returns:
            float: 增强后的置信度
        """
        try:
            # 如果量子交易功能被禁用，直接返回原始置信度
            if not self.quantum_trading["enabled"]:
                return confidence
                
            # 获取量子交易参数
            coherence = self.quantum_trading["coherence"]
            prediction_conf = self.quantum_trading["prediction_confidence"]
            
            # 应用量子增强算法
            # 通过量子相干性和预测置信度增强原始置信度
            enhanced = confidence * (1.0 + coherence * 0.2)
            enhanced = enhanced * prediction_conf
            
            # 确保增强后的置信度在0-1范围内
            enhanced = max(0.0, min(1.0, enhanced))
            
            return enhanced
        except Exception as e:
            self.logger.error(f"应用量子增强失败: {str(e)}")
            return confidence
            
    def get_quantum_signals(self):
        """获取量子交易信号
        
        Returns:
            list: 量子交易信号列表
        """
        try:
            if not hasattr(self, 'trading_signals') or not self.trading_signals:
                # 如果没有信号，生成一些模拟信号
                signals = []
                stocks = [
                    {"code": "600000", "name": "浦发银行"},
                    {"code": "600036", "name": "招商银行"},
                    {"code": "601398", "name": "工商银行"},
                    {"code": "600519", "name": "贵州茅台"},
                    {"code": "600276", "name": "恒瑞医药"}
                ]
                
                for stock in stocks:
                    if random.random() > 0.5:  # 50%概率生成信号
                        direction = 1 if random.random() > 0.4 else -1  # 60%买入，40%卖出
                        signals.append({
                            'symbol': stock["code"],
                            'name': stock["name"],
                            'direction': direction,
                            'confidence': random.uniform(0.7, 0.95),
                            'price_target': random.uniform(10, 100),
                            'timestamp': datetime.now(),
                            'volatility_factor': random.uniform(0.2, 0.6),
                            'quantum_adjusted': True
                        })
                
                return signals
            else:
                return self.trading_signals
        except Exception as e:
            self.logger.error(f"获取量子交易信号失败: {str(e)}")
            return [] 