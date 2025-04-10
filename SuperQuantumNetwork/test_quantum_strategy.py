#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
量子爆发策略测试脚本 - 扩展版本
"""

import logging
import sys
from event import MarketDataEvent, EventType
from data import TushareDataHandler
from portfolio import SimplePortfolio
from backtest import BacktestEngine
from strategy import QuantumBurstStrategy

# 配置日志
logging.basicConfig(
    level=logging.INFO,  # 修改为INFO级别，减少日志输出量
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main():
    try:
        # 设置回测参数
        symbols = [
            # 沪深300成分股样本 - 扩展为50只股票
            '000001.SZ',  # 平安银行
            '000002.SZ',  # 万科A
            '000063.SZ',  # 中兴通讯
            '000066.SZ',  # 中国长城
            '000333.SZ',  # 美的集团
            '000651.SZ',  # 格力电器
            '000725.SZ',  # 京东方A
            '000776.SZ',  # 广发证券
            '000858.SZ',  # 五粮液
            '000999.SZ',  # 华润三九
            '001979.SZ',  # 招商蛇口
            '002001.SZ',  # 新和成
            '002007.SZ',  # 华兰生物
            '002024.SZ',  # 苏宁易购
            '002027.SZ',  # 分众传媒
            '002032.SZ',  # 苏泊尔
            '002050.SZ',  # 三花智控
            '002064.SZ',  # 华峰氨纶
            '002074.SZ',  # 国轩高科
            '002129.SZ',  # 中环股份
            '002142.SZ',  # 宁波银行
            '002157.SZ',  # 正邦科技
            '002179.SZ',  # 中航光电
            '002202.SZ',  # 金风科技
            '002230.SZ',  # 科大讯飞
            '002236.SZ',  # 大华股份
            '002241.SZ',  # 歌尔股份
            '002252.SZ',  # 上海莱士
            '002304.SZ',  # 洋河股份
            '002311.SZ',  # 海大集团
            # 新增样本
            '600000.SH',  # 浦发银行
            '600009.SH',  # 上海机场
            '600016.SH',  # 民生银行
            '600031.SH',  # 三一重工
            '600036.SH',  # 招商银行
            '600104.SH',  # 上汽集团
            '600196.SH',  # 复星医药
            '600276.SH',  # 恒瑞医药
            '600519.SH',  # 贵州茅台
            '600585.SH',  # 海螺水泥
            '600690.SH',  # 海尔智家
            '600887.SH',  # 伊利股份
            '601012.SH',  # 隆基绿能
            '601088.SH',  # 中国神华
            '601166.SH',  # 兴业银行
            '601318.SH',  # 中国平安
            '601398.SH',  # 工商银行
            '601601.SH',  # 中国太保
            '601688.SH',  # 华泰证券
            '601888.SH'   # 中国中免
        ]  # 增加到50只股票
        
        start_date = '20210101'  # 回测开始日期 (延长回测周期)
        end_date = '20231231'    # 回测结束日期
        initial_capital = 2000000.0  # 增加初始资金

        logger.info("开始初始化回测环境...")
        logger.info(f"回测区间: {start_date} - {end_date}")
        logger.info(f"回测标的: {len(symbols)} 只股票")
        logger.info(f"初始资金: {initial_capital:,.2f}")

        # 初始化回测引擎
        backtest = BacktestEngine(
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
            initial_capital=initial_capital
        )

        # 初始化数据处理器
        logger.info("正在加载市场数据...")
        data_handler = TushareDataHandler(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            token="0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"
        )

        # 初始化策略
        logger.info("正在初始化交易策略...")
        strategy = QuantumBurstStrategy()
        
        # 调优策略参数
        strategy.max_positions = 8         # 提高最大持仓数量
        strategy.lookback_period = 15      # 延长回看期
        strategy.price_threshold = 0.12    # 降低价格变动阈值 (12%)
        strategy.volume_threshold = 1.15   # 降低成交量放大阈值
        strategy.signal_threshold = 0.18   # 适当降低信号强度阈值
        strategy.stop_loss = 0.06          # 提高止损比例
        strategy.take_profit = 0.25        # 提高止盈比例
        strategy.position_hold_days_limit = 8  # 延长最大持仓天数
        
        # 优化量子特征权重
        strategy.quantum_weights = {
            'momentum': 0.65,    # 适当降低动量权重
            'coherence': 0.25,   # 提高相干性权重
            'entropy': 0.05,     # 保持不变
            'resonance': 0.05    # 保持不变
        }
        
        strategy.data_handler = data_handler

        # 初始化投资组合
        logger.info("正在初始化投资组合...")
        portfolio = SimplePortfolio(initial_capital=initial_capital)
        portfolio.data_handler = data_handler

        # 设置回测引擎组件
        backtest.data_handler = data_handler
        backtest.strategy = strategy
        backtest.portfolio = portfolio

        # 运行回测
        logger.info("开始运行回测...")
        results = backtest.run()
        
        # 输出回测结果
        logger.info("\n=== 回测结果 ===")
        logger.info(f"最终资金: {results['metrics'].get('final_equity', 'N/A'):,.2f}")
        logger.info(f"总收益率: {results['metrics'].get('total_return', 0):.2%}")
        logger.info(f"年化收益率: {results['metrics'].get('annual_return', 0):.2%}")
        logger.info(f"最大回撤: {results['metrics'].get('max_drawdown', 0):.2%}")
        logger.info(f"夏普比率: {results['metrics'].get('sharpe_ratio', 0):.2f}")

    except Exception as e:
        logger.error(f"回测过程中发生错误: {str(e)}")
        raise

if __name__ == '__main__':
    main() 