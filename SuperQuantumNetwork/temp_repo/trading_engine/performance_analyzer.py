#!/usr/bin/env python3
"""
超神量子共生系统 - 性能分析器
分析交易表现，计算关键指标，提供绩效报告
"""

import logging
import time
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any
import pandas as pd
import numpy as np
from collections import defaultdict

# 设置日志
logger = logging.getLogger("TradingEngine.PerformanceAnalyzer")

class PerformanceAnalyzer:
    """性能分析器 - 分析交易表现和计算指标"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化性能分析器
        
        参数:
            config: 配置参数
        """
        self.config = config or {}
        
        # 设置默认配置
        self.benchmark_symbol = self.config.get("benchmark_symbol", "000001.SH")
        self.risk_free_rate = self.config.get("risk_free_rate", 0.02)  # 年化无风险利率
        self.capital_base = self.config.get("capital_base", 1000000.0)  # 基准资本
        
        # 性能数据
        self.daily_returns = []
        self.benchmark_returns = []
        self.trade_history = []
        self.position_history = []
        self.equity_curve = []
        self.drawdowns = []
        
        # 缓存计算结果
        self._cached_metrics = {}
        self._last_update_time = None
        
        logger.info("性能分析器初始化完成")
    
    def update(self, timestamp: datetime, account_value: float, cash: float, 
               positions: List[Dict], trades: List[Dict], benchmark_value: Optional[float] = None):
        """
        更新性能数据
        
        参数:
            timestamp: 时间戳
            account_value: 账户总价值
            cash: 现金
            positions: 持仓列表
            trades: 交易列表
            benchmark_value: 基准资产价值
        """
        # 记录日期和账户价值
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': account_value,
            'cash': cash,
            'invested': account_value - cash
        })
        
        # 计算每日收益率
        if len(self.equity_curve) >= 2:
            prev_equity = self.equity_curve[-2]['equity']
            if prev_equity > 0:
                daily_return = (account_value / prev_equity) - 1
                self.daily_returns.append({
                    'timestamp': timestamp,
                    'return': daily_return
                })
        
        # 记录基准收益率
        if benchmark_value is not None:
            if len(self.benchmark_returns) > 0:
                prev_benchmark = self.benchmark_returns[-1]['value']
                if prev_benchmark > 0:
                    benchmark_return = (benchmark_value / prev_benchmark) - 1
                    self.benchmark_returns.append({
                        'timestamp': timestamp,
                        'value': benchmark_value,
                        'return': benchmark_return
                    })
            else:
                # 第一个值, 收益率为0
                self.benchmark_returns.append({
                    'timestamp': timestamp,
                    'value': benchmark_value,
                    'return': 0.0
                })
        
        # 记录持仓
        self.position_history.append({
            'timestamp': timestamp,
            'positions': positions
        })
        
        # 记录交易
        for trade in trades:
            if not any(t.get('order_id') == trade.get('order_id') for t in self.trade_history):
                self.trade_history.append(trade)
        
        # 计算回撤
        self._update_drawdowns()
        
        # 清除缓存的计算结果
        self._cached_metrics = {}
        self._last_update_time = timestamp
    
    def _update_drawdowns(self):
        """更新回撤计算"""
        if len(self.equity_curve) == 0:
            return
        
        # 查找历史最高点
        peak = self.equity_curve[0]['equity']
        for point in self.equity_curve:
            equity = point['equity']
            if equity > peak:
                peak = equity
            
            # 计算回撤
            drawdown = 1 - (equity / peak) if peak > 0 else 0
            
            self.drawdowns.append({
                'timestamp': point['timestamp'],
                'equity': equity,
                'peak': peak,
                'drawdown': drawdown
            })
    
    def get_metrics(self) -> Dict:
        """
        获取性能指标
        
        返回:
            Dict: 性能指标字典
        """
        # 如果缓存有效，直接返回
        if self._cached_metrics and self._last_update_time is not None:
            return self._cached_metrics
        
        # 初始化指标
        metrics = {
            'start_date': None,
            'end_date': None,
            'duration_days': 0,
            'total_return': 0.0,
            'annual_return': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'expectancy': 0.0,
            'trade_count': len(self.trade_history),
            'avg_trade_return': 0.0,
            'benchmark_return': 0.0,
            'alpha': 0.0,
            'beta': 0.0,
            'r_squared': 0.0,
            'return_volatility': 0.0,
            'calmar_ratio': 0.0
        }
        
        # 如果没有足够的数据，返回默认指标
        if len(self.equity_curve) < 2:
            self._cached_metrics = metrics
            return metrics
        
        # 计算基本时间和回报指标
        metrics['start_date'] = self.equity_curve[0]['timestamp']
        metrics['end_date'] = self.equity_curve[-1]['timestamp']
        
        # 计算交易天数
        if hasattr(metrics['start_date'], 'date') and hasattr(metrics['end_date'], 'date'):
            metrics['duration_days'] = (metrics['end_date'].date() - metrics['start_date'].date()).days
        else:
            metrics['duration_days'] = len(self.equity_curve)
        
        # 计算总收益率
        start_equity = self.equity_curve[0]['equity']
        end_equity = self.equity_curve[-1]['equity']
        
        if start_equity > 0:
            metrics['total_return'] = (end_equity / start_equity) - 1
            
            # 计算年化收益率
            if metrics['duration_days'] > 0:
                metrics['annual_return'] = pow(1 + metrics['total_return'], 365 / metrics['duration_days']) - 1
        
        # 计算波动率
        if len(self.daily_returns) > 0:
            returns = [r['return'] for r in self.daily_returns]
            metrics['return_volatility'] = np.std(returns) * np.sqrt(252)  # 年化波动率
            
            # 计算Sharpe比率
            if metrics['return_volatility'] > 0:
                excess_return = metrics['annual_return'] - self.risk_free_rate
                metrics['sharpe_ratio'] = excess_return / metrics['return_volatility']
            
            # 计算Sortino比率
            downside_returns = [r for r in returns if r < 0]
            if len(downside_returns) > 0:
                downside_deviation = np.std(downside_returns) * np.sqrt(252)
                if downside_deviation > 0:
                    metrics['sortino_ratio'] = (metrics['annual_return'] - self.risk_free_rate) / downside_deviation
        
        # 计算最大回撤
        if len(self.drawdowns) > 0:
            max_drawdown = max(d['drawdown'] for d in self.drawdowns)
            metrics['max_drawdown'] = max_drawdown
            
            # 计算Calmar比率
            if max_drawdown > 0:
                metrics['calmar_ratio'] = metrics['annual_return'] / max_drawdown
        
        # 计算交易统计
        if len(self.trade_history) > 0:
            # 计算盈亏交易
            winning_trades = [t for t in self.trade_history if t.get('realized_pnl', 0) > 0]
            losing_trades = [t for t in self.trade_history if t.get('realized_pnl', 0) < 0]
            
            # 胜率
            metrics['win_rate'] = len(winning_trades) / len(self.trade_history) if len(self.trade_history) > 0 else 0
            
            # 盈亏比
            total_profit = sum(t.get('realized_pnl', 0) for t in winning_trades)
            total_loss = abs(sum(t.get('realized_pnl', 0) for t in losing_trades))
            metrics['profit_factor'] = total_profit / total_loss if total_loss > 0 else float('inf')
            
            # 期望值
            avg_win = total_profit / len(winning_trades) if len(winning_trades) > 0 else 0
            avg_loss = total_loss / len(losing_trades) if len(losing_trades) > 0 else 0
            metrics['expectancy'] = (metrics['win_rate'] * avg_win) - ((1 - metrics['win_rate']) * avg_loss)
            
            # 平均交易收益
            total_pnl = sum(t.get('realized_pnl', 0) for t in self.trade_history)
            metrics['avg_trade_return'] = total_pnl / len(self.trade_history)
        
        # 计算基准相关指标
        if len(self.benchmark_returns) > 1:
            benchmark_start = self.benchmark_returns[0]['value']
            benchmark_end = self.benchmark_returns[-1]['value']
            
            if benchmark_start > 0:
                metrics['benchmark_return'] = (benchmark_end / benchmark_start) - 1
                
                # 计算Alpha, Beta, R-squared
                if len(self.daily_returns) > 0 and len(self.benchmark_returns) >= len(self.daily_returns):
                    strategy_returns = np.array([r['return'] for r in self.daily_returns])
                    benchmark_daily_returns = np.array([r['return'] for r in self.benchmark_returns[-len(self.daily_returns):]])
                    
                    # 计算Beta
                    covariance = np.cov(strategy_returns, benchmark_daily_returns)[0][1]
                    benchmark_variance = np.var(benchmark_daily_returns)
                    if benchmark_variance > 0:
                        metrics['beta'] = covariance / benchmark_variance
                    
                    # 计算Alpha
                    benchmark_annual_return = pow(1 + metrics['benchmark_return'], 365 / metrics['duration_days']) - 1 if metrics['duration_days'] > 0 else 0
                    metrics['alpha'] = metrics['annual_return'] - (self.risk_free_rate + metrics['beta'] * (benchmark_annual_return - self.risk_free_rate))
                    
                    # 计算R-squared
                    correlation = np.corrcoef(strategy_returns, benchmark_daily_returns)[0][1]
                    metrics['r_squared'] = correlation ** 2
        
        # 缓存计算结果
        self._cached_metrics = metrics
        return metrics
    
    def get_equity_curve(self) -> pd.DataFrame:
        """
        获取权益曲线数据
        
        返回:
            pd.DataFrame: 权益曲线DataFrame
        """
        if not self.equity_curve:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.equity_curve)
        df.set_index('timestamp', inplace=True)
        return df
    
    def get_drawdown_curve(self) -> pd.DataFrame:
        """
        获取回撤曲线数据
        
        返回:
            pd.DataFrame: 回撤曲线DataFrame
        """
        if not self.drawdowns:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.drawdowns)
        df.set_index('timestamp', inplace=True)
        return df
    
    def get_trade_statistics(self) -> Dict:
        """
        获取交易统计
        
        返回:
            Dict: 交易统计字典
        """
        stats = {
            'total_trades': len(self.trade_history),
            'long_trades': 0,
            'short_trades': 0,
            'profitable_trades': 0,
            'losing_trades': 0,
            'avg_profit_per_trade': 0.0,
            'avg_loss_per_trade': 0.0,
            'largest_profit': 0.0,
            'largest_loss': 0.0,
            'avg_trade_duration': timedelta(),
            'symbols_traded': set()
        }
        
        if not self.trade_history:
            return stats
        
        # 分类统计
        profitable_trades = []
        losing_trades = []
        
        for trade in self.trade_history:
            # 交易方向
            if trade.get('direction') == 'BUY':
                stats['long_trades'] += 1
            elif trade.get('direction') == 'SELL':
                stats['short_trades'] += 1
            
            # 盈亏分类
            pnl = trade.get('realized_pnl', 0)
            if pnl > 0:
                stats['profitable_trades'] += 1
                profitable_trades.append(trade)
                stats['largest_profit'] = max(stats['largest_profit'], pnl)
            elif pnl < 0:
                stats['losing_trades'] += 1
                losing_trades.append(trade)
                stats['largest_loss'] = min(stats['largest_loss'], pnl)
            
            # 记录交易的标的
            if 'symbol' in trade:
                stats['symbols_traded'].add(trade['symbol'])
        
        # 计算平均盈亏
        if stats['profitable_trades'] > 0:
            stats['avg_profit_per_trade'] = sum(t.get('realized_pnl', 0) for t in profitable_trades) / stats['profitable_trades']
        
        if stats['losing_trades'] > 0:
            stats['avg_loss_per_trade'] = sum(t.get('realized_pnl', 0) for t in losing_trades) / stats['losing_trades']
        
        # 计算平均交易持续时间
        trades_with_duration = [t for t in self.trade_history if 'created_time' in t and 'execution_time' in t]
        if trades_with_duration:
            total_duration = timedelta()
            for trade in trades_with_duration:
                try:
                    start_time = datetime.fromisoformat(trade['created_time'])
                    end_time = datetime.fromisoformat(trade['execution_time'])
                    total_duration += (end_time - start_time)
                except (ValueError, TypeError):
                    pass
            
            if trades_with_duration:
                stats['avg_trade_duration'] = total_duration / len(trades_with_duration)
        
        # 转换交易的标的为列表
        stats['symbols_traded'] = list(stats['symbols_traded'])
        
        return stats
    
    def get_monthly_returns(self) -> pd.DataFrame:
        """
        获取月度收益率
        
        返回:
            pd.DataFrame: 月度收益率DataFrame
        """
        if not self.equity_curve:
            return pd.DataFrame()
        
        # 创建日期索引的权益曲线
        df = pd.DataFrame(self.equity_curve)
        df['date'] = pd.to_datetime(df['timestamp'])
        df.set_index('date', inplace=True)
        
        # 计算每日收益率
        df['daily_return'] = df['equity'].pct_change()
        
        # 计算月度累计收益率
        monthly_returns = df['daily_return'].resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        return pd.DataFrame(monthly_returns, columns=['monthly_return'])
    
    def get_performance_report(self) -> Dict:
        """
        获取完整性能报告
        
        返回:
            Dict: 性能报告字典
        """
        # 汇总各类指标
        report = {
            'metrics': self.get_metrics(),
            'trade_statistics': self.get_trade_statistics(),
            'last_update': self._last_update_time.isoformat() if self._last_update_time else None
        }
        
        # 添加月度收益
        monthly_returns = self.get_monthly_returns()
        if not monthly_returns.empty:
            report['monthly_returns'] = monthly_returns.to_dict()['monthly_return']
        
        return report
    
    def reset(self):
        """重置分析器"""
        self.daily_returns = []
        self.benchmark_returns = []
        self.trade_history = []
        self.position_history = []
        self.equity_curve = []
        self.drawdowns = []
        self._cached_metrics = {}
        self._last_update_time = None
        logger.info("性能分析器已重置")

# 测试函数
def test_performance_analyzer():
    """测试性能分析器"""
    # 创建性能分析器
    analyzer = PerformanceAnalyzer()
    
    # 模拟一系列的性能数据
    start_date = datetime(2025, 1, 1)
    equity = 1000000.0
    cash = 500000.0
    benchmark = 10000.0
    
    # 模拟数据
    for i in range(30):
        date = start_date + timedelta(days=i)
        
        # 模拟涨跌
        daily_change = 0.005 * np.random.randn() + 0.001
        equity *= (1 + daily_change)
        
        benchmark_change = 0.004 * np.random.randn() + 0.0005
        benchmark *= (1 + benchmark_change)
        
        # 模拟持仓和交易
        positions = [
            {'symbol': '000001.SZ', 'quantity': 10000, 'market_value': equity * 0.3},
            {'symbol': '600000.SH', 'quantity': 5000, 'market_value': equity * 0.2}
        ]
        
        trades = []
        if i % 5 == 0:  # 每5天一笔交易
            trade_pnl = equity * (0.01 * np.random.randn() + 0.002)
            trades = [{
                'order_id': f'order_{i}',
                'symbol': '000001.SZ' if i % 2 == 0 else '600000.SH',
                'direction': 'BUY' if i % 3 == 0 else 'SELL',
                'quantity': 1000,
                'price': 10.0,
                'realized_pnl': trade_pnl,
                'created_time': (date - timedelta(hours=2)).isoformat(),
                'execution_time': date.isoformat()
            }]
        
        # 更新分析器
        analyzer.update(date, equity, cash, positions, trades, benchmark)
        
        # 模拟现金变化
        cash = equity * (0.5 - i * 0.01)
        if cash < 0:
            cash = equity * 0.1
    
    # 获取性能指标
    metrics = analyzer.get_metrics()
    print("性能指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    # 获取交易统计
    trade_stats = analyzer.get_trade_statistics()
    print("\n交易统计:")
    for key, value in trade_stats.items():
        print(f"  {key}: {value}")
    
    # 获取月度收益
    monthly_returns = analyzer.get_monthly_returns()
    print("\n月度收益:")
    print(monthly_returns)
    
    return analyzer

if __name__ == "__main__":
    # 设置日志输出
    logging.basicConfig(level=logging.INFO)
    
    # 测试性能分析器
    test_performance_analyzer() 