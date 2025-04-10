#!/usr/bin/env python3
"""
超神量子共生系统 - 策略管理器
负责加载、管理和运行交易策略
"""

import os
import sys
import logging
import importlib
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Any, Type, Tuple

# 设置日志
logger = logging.getLogger("TradingEngine.StrategyManager")

class StrategyManager:
    """策略管理器类，负责加载、管理和运行交易策略"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化策略管理器
        
        参数:
            config: 配置参数
        """
        self.config = config or {}
        
        # 策略目录
        self.strategy_dirs = self.config.get("strategy_dirs", ["trading_engine/strategies"])
        
        # 策略注册表
        self.strategy_registry = {}  # name -> class
        
        # 活跃策略实例
        self.active_strategies = {}  # name -> instance
        
        # 策略执行结果
        self.strategy_results = {}  # name -> results
        
        # 策略状态
        self.strategy_status = {}  # name -> status
        
        # 加载所有可用策略
        self._load_strategies()
        
        logger.info(f"策略管理器初始化完成，已加载 {len(self.strategy_registry)} 个策略")
    
    def _load_strategies(self):
        """加载所有可用策略"""
        # 清空当前注册表
        self.strategy_registry = {}
        
        # 扫描策略目录
        for strategy_dir in self.strategy_dirs:
            if not os.path.exists(strategy_dir):
                logger.warning(f"策略目录不存在: {strategy_dir}")
                continue
            
            # 将策略目录添加到 sys.path
            if strategy_dir not in sys.path:
                sys.path.append(os.path.dirname(os.path.abspath(strategy_dir)))
            
            # 如果存在 __init__.py，尝试导入整个包
            if os.path.exists(os.path.join(strategy_dir, "__init__.py")):
                try:
                    package_name = os.path.basename(strategy_dir)
                    strategies_package = importlib.import_module(package_name)
                    
                    # 尝试获取策略注册表
                    if hasattr(strategies_package, "STRATEGY_REGISTRY"):
                        self.strategy_registry.update(strategies_package.STRATEGY_REGISTRY)
                        logger.info(f"从包 {package_name} 加载了 {len(strategies_package.STRATEGY_REGISTRY)} 个策略")
                    
                    # 尝试从包中加载策略
                    if hasattr(strategies_package, "get_available_strategies"):
                        strategy_names = strategies_package.get_available_strategies()
                        for strategy_name in strategy_names:
                            strategy_class = strategies_package.get_strategy_class(strategy_name)
                            if strategy_class:
                                self.strategy_registry[strategy_name] = strategy_class
                except Exception as e:
                    logger.error(f"加载策略包错误: {str(e)}")
                    traceback.print_exc()
            
            # 扫描目录中的 Python 文件
            for filename in os.listdir(strategy_dir):
                if filename.endswith(".py") and not filename.startswith("__"):
                    module_name = filename[:-3]  # 去掉 .py 后缀
                    
                    try:
                        # 导入模块
                        module_path = f"{os.path.basename(strategy_dir)}.{module_name}"
                        module = importlib.import_module(module_path)
                        
                        # 查找策略类
                        for attr_name in dir(module):
                            if attr_name.endswith("Strategy") and attr_name != "Strategy":
                                strategy_class = getattr(module, attr_name)
                                
                                # 确保是类，不是实例或其他对象
                                if isinstance(strategy_class, type):
                                    self.strategy_registry[attr_name] = strategy_class
                                    logger.info(f"从模块 {module_path} 加载策略: {attr_name}")
                    except Exception as e:
                        logger.error(f"加载策略模块 {module_name} 错误: {str(e)}")
    
    def get_available_strategies(self) -> List[str]:
        """
        获取所有可用策略列表
        
        返回:
            List[str]: 策略名称列表
        """
        return list(self.strategy_registry.keys())
    
    def get_strategy_class(self, strategy_name: str) -> Optional[Type]:
        """
        获取策略类
        
        参数:
            strategy_name: 策略名称
            
        返回:
            Type: 策略类
        """
        return self.strategy_registry.get(strategy_name)
    
    def create_strategy(self, strategy_name: str, strategy_config: Optional[Dict] = None) -> Any:
        """
        创建策略实例
        
        参数:
            strategy_name: 策略名称
            strategy_config: 策略配置
            
        返回:
            Any: 策略实例
        """
        strategy_class = self.get_strategy_class(strategy_name)
        
        if not strategy_class:
            logger.error(f"未找到策略: {strategy_name}")
            return None
        
        try:
            # 创建策略实例
            strategy_instance = strategy_class(strategy_config)
            
            # 更新活跃策略
            self.active_strategies[strategy_name] = strategy_instance
            self.strategy_status[strategy_name] = "initialized"
            
            logger.info(f"创建策略 {strategy_name} 成功")
            return strategy_instance
        except Exception as e:
            logger.error(f"创建策略 {strategy_name} 失败: {str(e)}")
            traceback.print_exc()
            return None
    
    def initialize_strategy(self, strategy_name: str, *args, **kwargs) -> bool:
        """
        初始化策略
        
        参数:
            strategy_name: 策略名称
            *args, **kwargs: 传递给策略初始化方法的参数
            
        返回:
            bool: 是否成功
        """
        if strategy_name not in self.active_strategies:
            logger.error(f"策略 {strategy_name} 未激活")
            return False
        
        strategy = self.active_strategies[strategy_name]
        
        try:
            # 调用策略自定义初始化方法（如果存在）
            if hasattr(strategy, "initialize"):
                strategy.initialize(*args, **kwargs)
            
            self.strategy_status[strategy_name] = "ready"
            return True
        except Exception as e:
            logger.error(f"初始化策略 {strategy_name} 失败: {str(e)}")
            self.strategy_status[strategy_name] = "error"
            return False
    
    def run_strategy(self, strategy_name: str, market_data: Dict, account_value: float) -> Dict:
        """
        运行策略
        
        参数:
            strategy_name: 策略名称
            market_data: 市场数据
            account_value: 账户价值
            
        返回:
            Dict: 策略结果
        """
        if strategy_name not in self.active_strategies:
            logger.error(f"策略 {strategy_name} 未激活")
            return {}
        
        strategy = self.active_strategies[strategy_name]
        
        try:
            # 记录状态为运行中
            self.strategy_status[strategy_name] = "running"
            
            # 运行策略
            current_time = datetime.now()
            result = strategy.update(current_time, market_data, account_value)
            
            # 更新策略结果
            self.strategy_results[strategy_name] = {
                "timestamp": current_time,
                "result": result,
                "status": "success"
            }
            
            # 更新状态为就绪
            self.strategy_status[strategy_name] = "ready"
            
            return result
        except Exception as e:
            logger.error(f"运行策略 {strategy_name} 失败: {str(e)}")
            traceback.print_exc()
            
            # 更新策略结果
            self.strategy_results[strategy_name] = {
                "timestamp": datetime.now(),
                "result": {},
                "status": "error",
                "error_message": str(e)
            }
            
            # 更新状态为错误
            self.strategy_status[strategy_name] = "error"
            
            return {}
    
    def get_strategy_status(self, strategy_name: str) -> str:
        """
        获取策略状态
        
        参数:
            strategy_name: 策略名称
            
        返回:
            str: 策略状态
        """
        return self.strategy_status.get(strategy_name, "unknown")
    
    def get_strategy_result(self, strategy_name: str) -> Dict:
        """
        获取策略结果
        
        参数:
            strategy_name: 策略名称
            
        返回:
            Dict: 策略结果
        """
        return self.strategy_results.get(strategy_name, {})
    
    def get_all_strategy_status(self) -> Dict:
        """
        获取所有策略状态
        
        返回:
            Dict: 策略状态 {strategy_name: status}
        """
        return self.strategy_status.copy()
    
    def get_active_strategies(self) -> List[str]:
        """
        获取活跃策略列表
        
        返回:
            List[str]: 活跃策略名称列表
        """
        return list(self.active_strategies.keys())
    
    def deactivate_strategy(self, strategy_name: str) -> bool:
        """
        停用策略
        
        参数:
            strategy_name: 策略名称
            
        返回:
            bool: 是否成功
        """
        if strategy_name not in self.active_strategies:
            logger.warning(f"策略 {strategy_name} 未激活")
            return False
        
        try:
            # 调用策略的清理方法（如果存在）
            strategy = self.active_strategies[strategy_name]
            if hasattr(strategy, "cleanup"):
                strategy.cleanup()
            
            # 从活跃策略中移除
            del self.active_strategies[strategy_name]
            self.strategy_status[strategy_name] = "deactivated"
            
            logger.info(f"停用策略 {strategy_name} 成功")
            return True
        except Exception as e:
            logger.error(f"停用策略 {strategy_name} 失败: {str(e)}")
            return False
    
    def reload_strategies(self) -> bool:
        """
        重新加载所有策略
        
        返回:
            bool: 是否成功
        """
        try:
            # 保存当前活跃策略名称
            active_strategy_names = list(self.active_strategies.keys())
            
            # 重新加载策略
            self._load_strategies()
            
            # 重新创建之前活跃的策略
            for strategy_name in active_strategy_names:
                if strategy_name in self.strategy_registry:
                    # 创建新实例
                    self.create_strategy(strategy_name)
            
            logger.info(f"重新加载策略成功，共加载 {len(self.strategy_registry)} 个策略")
            return True
        except Exception as e:
            logger.error(f"重新加载策略失败: {str(e)}")
            return False

# 测试函数
def test_strategy_manager():
    """测试策略管理器"""
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建策略管理器
    manager = StrategyManager()
    
    # 获取可用策略
    available_strategies = manager.get_available_strategies()
    print(f"可用策略: {available_strategies}")
    
    # 创建量子策略实例
    if "QuantumStrategy" in available_strategies:
        quantum_strategy = manager.create_strategy("QuantumStrategy", {
            "lookback_period": 20,
            "quantum_threshold": 0.6,
            "position_size": 0.1,
            "max_positions": 5
        })
        
        if quantum_strategy:
            print("量子策略创建成功")
            
            # 获取策略状态
            status = manager.get_strategy_status("QuantumStrategy")
            print(f"策略状态: {status}")
            
            # 初始化策略
            success = manager.initialize_strategy("QuantumStrategy")
            print(f"策略初始化: {'成功' if success else '失败'}")
            
            # 获取活跃策略
            active_strategies = manager.get_active_strategies()
            print(f"活跃策略: {active_strategies}")
            
            # 运行策略（创建模拟数据）
            import numpy as np
            import pandas as pd
            from datetime import datetime, timedelta
            
            # 创建模拟数据
            dates = pd.date_range(start='2025-01-01', periods=30)
            
            # 模拟上证指数数据
            sh_index_data = pd.DataFrame({
                'open': np.random.normal(3000, 20, 30),
                'high': np.random.normal(3050, 30, 30),
                'low': np.random.normal(2950, 30, 30),
                'close': np.random.normal(3000, 25, 30),
                'volume': np.random.normal(100000, 10000, 30)
            }, index=dates)
            
            # 调整使价格有一定连续性
            for i in range(1, len(sh_index_data)):
                change = np.random.normal(0, 15)
                sh_index_data.iloc[i, 3] = sh_index_data.iloc[i-1, 3] + change
            
            # 模拟几只股票数据
            stock_data = {
                "000001.SH": sh_index_data
            }
            
            # 添加几只股票
            for code in ["600000.SH", "000001.SZ", "600036.SH"]:
                base = sh_index_data['close'].values * np.random.uniform(0.5, 1.5)
                noise = np.random.normal(0, base.mean() * 0.01, len(base))
                
                stock_data[code] = pd.DataFrame({
                    'open': base * 0.99 + noise,
                    'high': base * 1.02 + noise,
                    'low': base * 0.98 + noise,
                    'close': base + noise,
                    'volume': np.random.normal(1000000, 200000, len(base))
                }, index=dates)
            
            # 运行策略
            account_value = 1000000.0
            result = manager.run_strategy("QuantumStrategy", stock_data, account_value)
            
            # 打印结果
            print("\n策略运行结果:")
            if result:
                print(f"生成信号数量: {len(result)}")
                for symbol, signal in result.items():
                    print(f"  {symbol}: {signal['direction']} (强度: {signal['strength']:.2f})")
            else:
                print("没有生成交易信号")
            
            # 获取策略状态和结果
            status = manager.get_strategy_status("QuantumStrategy")
            strategy_result = manager.get_strategy_result("QuantumStrategy")
            
            print(f"\n策略状态: {status}")
            print(f"策略结果时间戳: {strategy_result.get('timestamp')}")
            print(f"策略结果状态: {strategy_result.get('status')}")
            
            # 停用策略
            manager.deactivate_strategy("QuantumStrategy")
            print(f"策略停用后状态: {manager.get_strategy_status('QuantumStrategy')}")
    else:
        print("量子策略不可用")
    
    return manager

if __name__ == "__main__":
    test_strategy_manager() 