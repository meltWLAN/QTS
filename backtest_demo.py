#!/usr/bin/env python3
"""
回测日志模块使用示例
"""

import logging
import pandas as pd
import numpy as np
import time
from src.utils.logger import setup_backtest_logger

class SimpleBacktester:
    """简单的回测演示类"""
    
    def __init__(self, strategy_name="量子策略示例"):
        # 设置回测专用日志记录器
        self.logger = setup_backtest_logger(
            strategy_name=strategy_name,
            log_level=logging.INFO
        )
        self.trades = []
        self.logger.info("=============================================")
        self.logger.info(f"{strategy_name} - 回测启动")
        self.logger.info("=============================================")
    
    def generate_sample_data(self, days=100):
        """生成模拟股票数据用于回测"""
        self.logger.info("生成模拟回测数据")
        dates = pd.date_range(start='2023-01-01', periods=days)
        
        # 创建一个随机波动的价格序列
        price = 100  # 起始价格
        prices = [price]
        for _ in range(days-1):
            # 随机涨跌
            change = np.random.normal(0, 1) / 100  # 每日变化率服从正态分布
            price = price * (1 + change)
            prices.append(price)
        
        # 生成成交量
        volumes = np.random.randint(1000, 10000, size=days)
        
        # 创建DataFrame
        data = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
            'low': [p * (1 - np.random.uniform(0, 0.02)) for p in prices],
            'close': prices,
            'volume': volumes
        })
        
        self.logger.info(f"生成了 {days} 天的模拟数据")
        return data
    
    def run_backtest(self, data, initial_capital=100000.0):
        """运行简单的回测"""
        self.logger.info(f"开始回测，初始资金: {initial_capital:.2f}")
        
        capital = initial_capital
        position = 0
        entry_price = 0
        
        # 遍历数据进行回测
        for i in range(5, len(data)):
            # 简单的移动平均策略
            short_ma = data['close'].iloc[i-5:i].mean()
            long_ma = data['close'].iloc[i-20:i].mean() if i >= 20 else short_ma
            
            current_price = data['close'].iloc[i]
            current_date = data['date'].iloc[i]
            
            # 模拟交易延迟
            time.sleep(0.01)
            
            # 买入信号: 短期均线上穿长期均线
            if short_ma > long_ma and position == 0:
                # 计算可买入数量
                position = int(capital / current_price)
                entry_price = current_price
                cost = position * current_price
                capital -= cost
                
                self.logger.info(f"买入信号! 日期: {current_date.date()}, 价格: {current_price:.2f}, 数量: {position}, 剩余资金: {capital:.2f}")
            
            # 卖出信号: 短期均线下穿长期均线
            elif short_ma < long_ma and position > 0:
                # 卖出所有持仓
                revenue = position * current_price
                profit = revenue - (position * entry_price)
                profit_pct = (profit / (position * entry_price)) * 100
                
                self.trades.append({
                    'entry_date': current_date.date(),
                    'entry_price': entry_price,
                    'exit_date': current_date.date(),
                    'exit_price': current_price,
                    'position': position,
                    'profit': profit,
                    'profit_pct': profit_pct
                })
                
                capital += revenue
                position = 0
                
                self.logger.info(f"卖出信号! 日期: {current_date.date()}, 价格: {current_price:.2f}, 数量: {position}, 当前资金: {capital:.2f}, 收益: {profit:.2f} ({profit_pct:.2f}%)")
        
        # 计算回测结果
        if position > 0:
            # 如果还有持仓，按最后价格平仓
            final_price = data['close'].iloc[-1]
            revenue = position * final_price
            profit = revenue - (position * entry_price)
            capital += revenue
            
            self.logger.info(f"最后平仓! 价格: {final_price:.2f}, 数量: {position}, 收益: {profit:.2f}")
        
        # 计算总收益
        total_return = (capital - initial_capital) / initial_capital * 100
        
        # 记录回测结果
        self.logger.info("=============================================")
        self.logger.info("回测完成，结果如下:")
        self.logger.info(f"初始资金: {initial_capital:.2f}")
        self.logger.info(f"最终资金: {capital:.2f}")
        self.logger.info(f"总收益: {capital - initial_capital:.2f}")
        self.logger.info(f"收益率: {total_return:.2f}%")
        self.logger.info(f"交易次数: {len(self.trades)}")
        
        # 计算其他指标（如果有交易）
        if self.trades:
            wins = sum(1 for trade in self.trades if trade['profit'] > 0)
            win_rate = wins / len(self.trades) * 100
            average_win = sum(trade['profit'] for trade in self.trades if trade['profit'] > 0) / max(1, wins)
            average_loss = sum(trade['profit'] for trade in self.trades if trade['profit'] <= 0) / max(1, len(self.trades) - wins)
            
            self.logger.info(f"胜率: {win_rate:.2f}%")
            self.logger.info(f"平均盈利: {average_win:.2f}")
            self.logger.info(f"平均亏损: {average_loss:.2f}")
            if average_loss != 0:
                self.logger.info(f"盈亏比: {abs(average_win/average_loss):.2f}")
            
        self.logger.info("=============================================")
        
        return {
            'final_capital': capital,
            'total_return': total_return,
            'trades': self.trades
        }


def main():
    """主函数"""
    # 创建回测器实例
    backtester = SimpleBacktester(strategy_name="量子MA交叉策略")
    
    # 生成模拟数据
    data = backtester.generate_sample_data(days=200)
    
    # 运行回测
    results = backtester.run_backtest(data, initial_capital=100000.0)
    
    print(f"回测完成，总收益率: {results['total_return']:.2f}%")


if __name__ == "__main__":
    main() 