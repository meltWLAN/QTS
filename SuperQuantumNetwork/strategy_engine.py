import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger("StrategyEngine")

class StrategyEngine:
    """策略引擎"""
    
    def __init__(self):
        """初始化策略引擎"""
        self.strategies = {}
        self.signals = {}
        self.positions = {}
        
    def register_strategy(self, name: str, strategy_func):
        """
        注册策略
        
        参数:
            name: 策略名称
            strategy_func: 策略函数
        """
        self.strategies[name] = strategy_func
        logger.info(f"策略 {name} 注册成功")
        
    def run_strategy(self, name: str, data: pd.DataFrame, params: Dict = None) -> pd.DataFrame:
        """
        运行策略
        
        参数:
            name: 策略名称
            data: 数据
            params: 策略参数
            
        返回:
            DataFrame: 策略结果
        """
        try:
            if name not in self.strategies:
                raise ValueError(f"策略 {name} 不存在")
                
            strategy_func = self.strategies[name]
            result = strategy_func(data, params)
            
            self.signals[name] = result
            return result
            
        except Exception as e:
            logger.error(f"运行策略 {name} 失败: {str(e)}")
            return pd.DataFrame()
    
    def combine_signals(self, weights: Dict[str, float] = None) -> pd.DataFrame:
        """
        组合信号
        
        参数:
            weights: 策略权重
            
        返回:
            DataFrame: 组合信号
        """
        try:
            if not self.signals:
                raise ValueError("没有可用的策略信号")
                
            if weights is None:
                weights = {name: 1.0/len(self.signals) for name in self.signals}
                
            combined = pd.DataFrame()
            for name, signal in self.signals.items():
                if name in weights:
                    combined[name] = signal * weights[name]
                    
            combined['combined_signal'] = combined.sum(axis=1)
            return combined
            
        except Exception as e:
            logger.error(f"组合信号失败: {str(e)}")
            return pd.DataFrame()
    
    def generate_orders(self, signals: pd.DataFrame, capital: float, risk_control: Dict = None) -> pd.DataFrame:
        """
        生成订单
        
        参数:
            signals: 信号数据
            capital: 资金
            risk_control: 风险控制参数
            
        返回:
            DataFrame: 订单数据
        """
        try:
            if risk_control is None:
                risk_control = {
                    'max_position': 0.3,  # 最大仓位
                    'stop_loss': 0.1,     # 止损比例
                    'take_profit': 0.2    # 止盈比例
                }
                
            orders = pd.DataFrame()
            orders['signal'] = signals['combined_signal']
            orders['position'] = orders['signal'].apply(
                lambda x: 1 if x > 0.5 else (-1 if x < -0.5 else 0)
            )
            
            # 计算仓位
            orders['position_value'] = orders['position'] * capital * risk_control['max_position']
            
            # 计算止损止盈价格
            orders['stop_loss_price'] = orders.apply(
                lambda x: x['close'] * (1 - risk_control['stop_loss']) if x['position'] > 0
                else x['close'] * (1 + risk_control['stop_loss']) if x['position'] < 0
                else 0,
                axis=1
            )
            
            orders['take_profit_price'] = orders.apply(
                lambda x: x['close'] * (1 + risk_control['take_profit']) if x['position'] > 0
                else x['close'] * (1 - risk_control['take_profit']) if x['position'] < 0
                else 0,
                axis=1
            )
            
            return orders
            
        except Exception as e:
            logger.error(f"生成订单失败: {str(e)}")
            return pd.DataFrame()
    
    def backtest(self, data: pd.DataFrame, orders: pd.DataFrame, initial_capital: float) -> Dict:
        """
        回测
        
        参数:
            data: 数据
            orders: 订单数据
            initial_capital: 初始资金
            
        返回:
            Dict: 回测结果
        """
        try:
            # 初始化回测结果
            results = {
                'trades': [],
                'positions': [],
                'capital': [initial_capital],
                'returns': [0.0]
            }
            
            current_position = 0
            current_capital = initial_capital
            
            for i in range(len(data)):
                # 更新持仓
                if orders['position'].iloc[i] != current_position:
                    trade = {
                        'date': data.index[i],
                        'type': 'buy' if orders['position'].iloc[i] > current_position else 'sell',
                        'price': data['close'].iloc[i],
                        'volume': abs(orders['position'].iloc[i] - current_position),
                        'value': abs(orders['position_value'].iloc[i] - current_position * data['close'].iloc[i])
                    }
                    results['trades'].append(trade)
                    
                    current_position = orders['position'].iloc[i]
                
                # 检查止损止盈
                if current_position != 0:
                    if (current_position > 0 and data['low'].iloc[i] <= orders['stop_loss_price'].iloc[i]) or \
                       (current_position < 0 and data['high'].iloc[i] >= orders['stop_loss_price'].iloc[i]):
                        # 触发止损
                        trade = {
                            'date': data.index[i],
                            'type': 'stop_loss',
                            'price': orders['stop_loss_price'].iloc[i],
                            'volume': current_position,
                            'value': current_position * orders['stop_loss_price'].iloc[i]
                        }
                        results['trades'].append(trade)
                        current_position = 0
                        
                    elif (current_position > 0 and data['high'].iloc[i] >= orders['take_profit_price'].iloc[i]) or \
                         (current_position < 0 and data['low'].iloc[i] <= orders['take_profit_price'].iloc[i]):
                        # 触发止盈
                        trade = {
                            'date': data.index[i],
                            'type': 'take_profit',
                            'price': orders['take_profit_price'].iloc[i],
                            'volume': current_position,
                            'value': current_position * orders['take_profit_price'].iloc[i]
                        }
                        results['trades'].append(trade)
                        current_position = 0
                
                # 更新资金和收益
                position_value = current_position * data['close'].iloc[i]
                current_capital = initial_capital + position_value
                returns = (current_capital - initial_capital) / initial_capital
                
                results['positions'].append(current_position)
                results['capital'].append(current_capital)
                results['returns'].append(returns)
            
            # 计算回测指标
            results['total_trades'] = len(results['trades'])
            results['win_rate'] = len([t for t in results['trades'] if t['value'] > 0]) / results['total_trades']
            results['max_drawdown'] = min(results['returns'])
            results['final_return'] = results['returns'][-1]
            
            return results
            
        except Exception as e:
            logger.error(f"回测失败: {str(e)}")
            return {
                'trades': [],
                'positions': [],
                'capital': [initial_capital],
                'returns': [0.0],
                'total_trades': 0,
                'win_rate': 0.0,
                'max_drawdown': 0.0,
                'final_return': 0.0
            }
    
    def optimize_parameters(self, data: pd.DataFrame, strategy_name: str, param_grid: Dict) -> Dict:
        """
        优化策略参数
        
        参数:
            data: 数据
            strategy_name: 策略名称
            param_grid: 参数网格
            
        返回:
            Dict: 最优参数
        """
        try:
            best_params = None
            best_return = float('-inf')
            
            # 生成参数组合
            param_combinations = [dict(zip(param_grid.keys(), v)) for v in np.array(np.meshgrid(*param_grid.values())).T.reshape(-1, len(param_grid))]
            
            for params in param_combinations:
                # 运行策略
                signals = self.run_strategy(strategy_name, data, params)
                
                # 生成订单
                orders = self.generate_orders(signals, 1000000)  # 使用100万初始资金进行优化
                
                # 回测
                results = self.backtest(data, orders, 1000000)
                
                # 更新最优参数
                if results['final_return'] > best_return:
                    best_return = results['final_return']
                    best_params = params
            
            return best_params
            
        except Exception as e:
            logger.error(f"优化参数失败: {str(e)}")
            return {} 