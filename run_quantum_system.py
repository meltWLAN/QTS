#!/usr/bin/env python3
"""
超神量子系统 - 主入口
集成数据连接器并启动回测或图形界面
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入数据连接器
from src.utils.data_connector import get_data_connector, TushareConnector
# 添加日志配置
from src.utils.logger import setup_logger

# 设置日志
log_file = f"quantum_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logger = setup_logger(
    log_file=log_file,
    log_level=logging.INFO,
    log_to_console=True
)
# 设置logger名称
logger.name = "QuantumSystem"

def run_backtest(args):
    """运行回测模式"""
    logger.info("启动回测模式...")
    
    # 获取数据连接器实例
    data_connector = get_data_connector('tushare', token=args.token)
    logger.info(f"已初始化数据连接器: {data_connector.name}")
    
    # 根据参数导入回测模块
    if args.advanced:
        logger.info("使用高级回测模块")
        try:
            from SuperQuantumNetwork.backtest_engine import BacktestEngine
            from SuperQuantumNetwork.quantum_burst_strategy_enhanced import QuantumBurstStrategyEnhanced
            
            # 初始化回测引擎
            engine = BacktestEngine(
                start_date=args.start_date,
                end_date=args.end_date,
                symbols=args.symbols.split(','),
                initial_capital=args.capital,
                benchmark_symbol=args.benchmark
            )
            
            # 创建数据适配器，将统一数据连接器适配到回测引擎需要的接口
            class DataConnectorAdapter:
                def __init__(self, connector):
                    self.connector = connector
                    self.symbols = args.symbols.split(',')
                    if args.benchmark and args.benchmark not in self.symbols:
                        self.symbols.append(args.benchmark)
                    self.start_date = args.start_date
                    self.end_date = args.end_date
                    self._data_cache = {}
                
                def get_latest_data(self, symbol, n=1):
                    # 从缓存获取或加载数据
                    if symbol not in self._data_cache:
                        df = self.connector.get_daily_data(symbol, self.start_date, self.end_date)
                        self._data_cache[symbol] = df
                    
                    # 返回最新的n条数据
                    return self._data_cache[symbol].tail(n)
                
                def update_bars(self):
                    # 简化实现，一次性加载所有数据
                    for symbol in self.symbols:
                        if symbol not in self._data_cache:
                            df = self.connector.get_daily_data(symbol, self.start_date, self.end_date)
                            self._data_cache[symbol] = df
                    
                    # 创建并返回市场数据事件
                    from SuperQuantumNetwork.event import MarketDataEvent, EventType
                    events = []
                    
                    for symbol in self.symbols:
                        event = MarketDataEvent(
                            type=EventType.MARKET_DATA,
                            timestamp=datetime.now(),
                            symbol=symbol,
                            data=self._data_cache[symbol]
                        )
                        events.append(event)
                    
                    return events
            
            # 设置数据适配器
            engine.data_handler = DataConnectorAdapter(data_connector)
            
            # 设置策略
            strategy = QuantumBurstStrategyEnhanced(data_handler=engine.data_handler)
            engine.set_strategy(strategy)
            
            # 运行回测
            results = engine.run()
            
            # 输出结果
            logger.info("回测完成")
            for key, value in results.items():
                logger.info(f"{key}: {value}")
            
            # 显示结果图表
            engine.plot_results()
            
        except ImportError as e:
            logger.error(f"导入高级回测模块失败: {str(e)}")
            logger.info("切换到基础回测模式")
            run_simple_backtest(data_connector, args)
    else:
        # 使用简单回测模式
        run_simple_backtest(data_connector, args)
    
    logger.info("回测结束")

def run_simple_backtest(data_connector, args):
    """运行简单回测模式"""
    logger.info("使用基础回测模块")
    
    try:
        # 首先尝试导入quantum_backtest模块
        from quantum_backtest import run_backtest as run_quantum_backtest
        
        # 运行自定义回测
        run_quantum_backtest()
        
    except ImportError:
        # 如果导入失败，使用内置的简单回测逻辑
        logger.info("使用内置简单回测逻辑")
        
        # 获取股票列表
        symbols = args.symbols.split(',')
        logger.info(f"回测标的: {symbols}")
        
        # 初始资金
        initial_capital = args.capital
        logger.info(f"初始资金: {initial_capital}")
        
        # 回测区间
        start_date = args.start_date
        end_date = args.end_date
        logger.info(f"回测区间: {start_date} - {end_date}")
        
        # 回测结果
        results = {}
        
        # 遍历股票
        for symbol in symbols:
            logger.info(f"获取 {symbol} 的历史数据...")
            
            # 获取历史数据
            df = data_connector.get_daily_data(symbol, start_date, end_date)
            
            if df is None or len(df) == 0:
                logger.warning(f"获取 {symbol} 数据失败或数据为空")
                continue
                
            logger.info(f"成功获取 {symbol} 数据，共 {len(df)} 条记录")
            
            # TODO: 添加简单策略逻辑
            
        logger.info("简单回测完成")
        return results

def run_gui(args):
    """运行图形界面模式"""
    logger.info("启动图形界面模式...")
    
    # 获取数据连接器实例
    data_connector = get_data_connector('tushare', token=args.token)
    logger.info(f"已初始化数据连接器: {data_connector.name}")
    
    try:
        # 初始化PyQt应用
        from PyQt5.QtWidgets import QApplication
        app = QApplication(sys.argv)
        
        # 初始化控制器
        logger.info("初始化系统控制器...")
        controllers = init_controllers(data_connector)
        
        # 初始化量子组件
        logger.info("初始化量子组件...")
        quantum_components = init_quantum_components()
        
        # 初始化量子共生核心
        logger.info("初始化量子共生核心...")
        symbiotic_core = init_symbiotic_core()
        
        # 导入主应用
        from SuperQuantumNetwork.gui.app import SuperGodApp
        
        # 创建超神应用
        logger.info("创建超神应用实例...")
        supergod_app = SuperGodApp(
            controllers=controllers,
            quantum_components=quantum_components,
            symbiotic_core=symbiotic_core
        )
        
        # 显示主窗口
        supergod_app.main_window.show()
        
        # 运行应用
        logger.info("图形界面启动成功，进入事件循环")
        return app.exec_()
        
    except ImportError as e:
        logger.error(f"导入图形界面模块失败: {str(e)}")
        try_fallback_gui(data_connector)
        return 1

def init_controllers(data_connector):
    """初始化系统控制器"""
    controllers = {}
    
    try:
        # 导入控制器模块
        from SuperQuantumNetwork.controllers.data_controller import DataController
        from SuperQuantumNetwork.controllers.trading_controller import TradingController
        from SuperQuantumNetwork.controllers.portfolio_controller import PortfolioController
        from SuperQuantumNetwork.controllers.risk_controller import RiskController
        
        # 创建数据控制器
        controllers['data'] = DataController(data_connector=data_connector)
        logger.info("数据控制器初始化完成")
        
        # 创建交易控制器
        controllers['trading'] = TradingController()
        logger.info("交易控制器初始化完成")
        
        # 创建投资组合控制器
        controllers['portfolio'] = PortfolioController()
        logger.info("投资组合控制器初始化完成")
        
        # 创建风险控制器
        controllers['risk'] = RiskController()
        logger.info("风险控制器初始化完成")
        
    except ImportError as e:
        logger.error(f"控制器初始化失败: {str(e)}")
        # 创建简化的控制器
        from SuperQuantumNetwork.simplified.controllers import (
            SimpleDataController, 
            SimpleTradingController
        )
        
        controllers['data'] = SimpleDataController(data_connector)
        controllers['trading'] = SimpleTradingController()
        logger.info("已使用简化版控制器")
    
    return controllers

def init_quantum_components():
    """初始化量子组件"""
    components = {}
    
    try:
        # 导入量子组件
        from SuperQuantumNetwork.quantum.core import QuantumCore
        from SuperQuantumNetwork.quantum.predictor import QuantumPredictor
        from SuperQuantumNetwork.quantum.optimizer import QuantumOptimizer
        
        # 创建量子核心
        components['core'] = QuantumCore()
        logger.info("量子核心初始化完成")
        
        # 创建量子预测器
        components['predictor'] = QuantumPredictor()
        logger.info("量子预测器初始化完成")
        
        # 创建量子优化器
        components['optimizer'] = QuantumOptimizer()
        logger.info("量子优化器初始化完成")
        
    except ImportError as e:
        logger.error(f"量子组件初始化失败: {str(e)}")
        logger.info("将使用模拟量子组件")
        
        # 创建模拟组件
        components['core'] = object()
        components['predictor'] = object()
        components['optimizer'] = object()
    
    return components

def init_symbiotic_core():
    """初始化量子共生核心"""
    try:
        # 导入共生核心
        from SuperQuantumNetwork.symbiosis.core import SymbioticCore
        
        # 创建共生核心
        symbiotic_core = SymbioticCore()
        logger.info("量子共生核心初始化完成")
        
        return symbiotic_core
        
    except ImportError as e:
        logger.error(f"量子共生核心初始化失败: {str(e)}")
        logger.info("将使用简化版共生核心")
        
        # 返回一个模拟的共生核心
        class SimpleSymbioticCore:
            def __init__(self):
                self.field_state = {"active": True}
                
            def deactivate_field(self):
                self.field_state["active"] = False
                
        return SimpleSymbioticCore()

def try_fallback_gui(data_connector):
    """尝试启动备用GUI"""
    logger.info("尝试启动备用图形界面...")
    
    try:
        # 尝试导入和启动备用GUI
        from SuperQuantumNetwork.simple_gui import run_simple_gui
        run_simple_gui(data_connector)
        
    except ImportError as e:
        logger.error(f"备用图形界面也无法启动: {str(e)}")
        logger.error("无法启动任何图形界面，请检查安装")
        sys.exit(1)

def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="超神量子系统 - 量化交易回测与分析平台")
    
    # 子命令
    subparsers = parser.add_subparsers(dest="command", help="功能模式")
    
    # 回测模式参数
    backtest_parser = subparsers.add_parser("backtest", help="回测模式")
    backtest_parser.add_argument("--symbols", type=str, default="000001.SZ,600519.SH", help="回测标的，逗号分隔")
    backtest_parser.add_argument("--start-date", type=str, default="20230101", help="起始日期")
    backtest_parser.add_argument("--end-date", type=str, default="20231231", help="结束日期")
    backtest_parser.add_argument("--capital", type=float, default=1000000.0, help="初始资金")
    backtest_parser.add_argument("--benchmark", type=str, default="000300.SH", help="业绩基准")
    backtest_parser.add_argument("--advanced", action="store_true", help="使用高级回测模式")
    
    # 图形界面模式参数
    gui_parser = subparsers.add_parser("gui", help="图形界面模式")
    
    # 通用参数
    for subparser in [backtest_parser, gui_parser]:
        subparser.add_argument("--token", type=str, help="Tushare API Token")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 根据命令选择模式
    if args.command == "backtest":
        run_backtest(args)
    elif args.command == "gui":
        run_gui(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("用户中断程序")
        sys.exit(0)
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}", exc_info=True)
        sys.exit(1)