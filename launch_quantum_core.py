#!/usr/bin/env python3
"""
超神量子核心启动器 - 启动量子核心服务
"""

import os
import sys
import logging
import argparse
from datetime import datetime

# 配置日志
log_file = f"quantum_core_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("QuantumCoreLauncher")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="超神量子核心启动器")
    parser.add_argument('--mode', choices=['standalone', 'integrated'], 
                      default='integrated', help='运行模式')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志级别
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("正在启动超神量子核心...")
    
    try:
        # 导入量子核心运行时环境
        from quantum_core.event_driven_coordinator import RuntimeEnvironment
        
        # 创建运行时环境
        runtime = RuntimeEnvironment()
        
        # 注册组件
        logger.info("注册核心组件...")
        
        # 注册事件系统
        from quantum_core.event_system import QuantumEventSystem
        event_system = QuantumEventSystem()
        runtime.register_component("event_system", event_system)
        
        # 注册数据管道
        from quantum_core.data_pipeline import MarketDataPipeline
        data_pipeline = MarketDataPipeline(event_system)
        runtime.register_component("data_pipeline", data_pipeline)
        
        # 注册量子后端
        from quantum_core.quantum_backend import QuantumBackend
        quantum_backend = QuantumBackend(backend_type='simulator')
        runtime.register_component("quantum_backend", quantum_backend)
        
        # 注册市场分析器
        from quantum_core.multidimensional_analysis import MultidimensionalAnalyzer
        market_analyzer = MultidimensionalAnalyzer()
        runtime.register_component("market_analyzer", market_analyzer)
        
        # 注册策略优化器
        from quantum_core.genetic_strategy_optimizer import GeneticStrategyOptimizer
        strategy_optimizer = GeneticStrategyOptimizer()
        runtime.register_component("strategy_optimizer", strategy_optimizer)
        
        # 注册系统监控
        from quantum_core.system_monitor import SystemHealthMonitor
        system_monitor = SystemHealthMonitor()
        runtime.register_component("system_monitor", system_monitor)
        
        # 在集成模式下连接到超神系统
        if args.mode == 'integrated':
            logger.info("以集成模式启动，连接到超神系统...")
            # 这里添加与超神系统集成的代码
        
        # 启动运行时环境
        logger.info("启动量子核心运行时环境...")
        runtime.start()
        
        logger.info("量子核心启动成功，按Ctrl+C退出")
        
        # 等待用户中断
        import time
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("接收到退出信号")
        finally:
            # 停止运行时环境
            logger.info("正在停止量子核心...")
            runtime.stop()
            logger.info("量子核心已停止")
    
    except Exception as e:
        logger.error(f"启动失败: {str(e)}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
