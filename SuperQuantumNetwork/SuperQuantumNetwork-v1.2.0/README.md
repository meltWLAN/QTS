# 超神量子共生系统 - 交易引擎

超神量子共生系统交易引擎是一个基于量子维度分析的高级交易系统，旨在通过多维度分析和量子算法提供更精准的市场预测和交易信号。

## 系统架构

系统主要由以下组件构成：

- **交易核心 (Trading Core)**: 负责订单管理、执行和账户管理
- **风险管理器 (Risk Manager)**: 提供风险控制和限制管理
- **性能分析器 (Performance Analyzer)**: 分析交易表现，计算关键指标
- **策略模块 (Strategies)**: 包含各种交易策略实现
- **数据管理器 (Data Manager)**: 负责获取和管理市场数据
- **市场体制分析 (Market Regime Analysis)**: 分析市场环境和状态

## 特点

- **量子维度分析**: 将市场分析从传统的11个维度扩展到21个维度，提供更深入的市场洞察
- **多数据源支持**: 支持同时连接多个数据源，如Tushare、EastMoney等
- **完整的风险管理**: 提供全面的风险控制策略，包括最大回撤限制、净暴露度控制等
- **高性能回测**: 支持快速回测交易策略，计算各种性能指标
- **实时市场分析**: 支持实时监控市场状态和体制变化
- **多策略组合**: 支持多个交易策略同时运行和组合

## 安装与配置

### 环境要求

- Python 3.7+
- pandas, numpy, matplotlib
- 适用于各种数据源的API客户端

### 安装步骤

1. 克隆代码库:
```bash
git clone https://github.com/your-username/supergod-trading-engine.git
cd supergod-trading-engine
```

2. 安装依赖:
```bash
pip install -r requirements.txt
```

3. 配置数据源:
修改`config.json`文件，配置您的数据源信息和API密钥

## 使用方法

### 初始化交易引擎

```python
from trading_engine.trading_core import TradingEngine
from trading_engine.risk_manager import RiskManager
from trading_engine.performance_analyzer import PerformanceAnalyzer

# 创建交易引擎实例
engine = TradingEngine({
    "initial_cash": 1000000.0,
    "commission_rate": 0.0003,
    "min_commission": 5.0
})

# 连接风险管理器
risk_manager = RiskManager()
engine.risk_manager = risk_manager

# 连接性能分析器
performance_analyzer = PerformanceAnalyzer()
engine.performance_analyzer = performance_analyzer
```

### 使用量子交易策略

```python
from trading_engine.strategy_manager import StrategyManager
from trading_engine.data_manager import DataManager

# 创建策略管理器
strategy_manager = StrategyManager()

# 创建数据管理器
data_manager = DataManager({
    "data_source_type": "tushare",
    "tushare_token": "your_token_here"
})

# 获取市场数据
symbols = ["000001.SH", "600000.SH", "000001.SZ"]
market_data = data_manager.get_market_data(symbols)

# 创建量子策略实例
quantum_strategy = strategy_manager.create_strategy("QuantumStrategy", {
    "lookback_period": 20,
    "quantum_threshold": 0.65,
    "position_size": 0.1
})

# 运行策略
account_value = engine.get_account_summary()["total_value"]
signals = strategy_manager.run_strategy("QuantumStrategy", market_data, account_value)

# 执行交易信号
for symbol, signal in signals.items():
    if signal["direction"] == "long":
        engine.place_order(
            symbol=symbol,
            direction=OrderDirection.BUY,
            quantity=signal["quantity"]
        )
    elif signal["direction"] == "short":
        engine.place_order(
            symbol=symbol,
            direction=OrderDirection.SELL,
            quantity=signal["quantity"]
        )
```

### 分析市场体制

```python
from trading_engine.strategies.market_regime import MarketRegimeAnalyzer

# 创建市场体制分析器
analyzer = MarketRegimeAnalyzer()

# 分析市场体制
result = analyzer.analyze(market_data)
print(f"当前市场体制: {result.regime_type.value}, 置信度: {result.confidence:.2f}")
print(f"子类型: {result.sub_type}")
```

### 性能评估

```python
# 获取性能指标
metrics = engine.get_performance_metrics()
print(f"总收益率: {metrics['total_return_pct']:.2f}%")
print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
print(f"最大回撤: {metrics['max_drawdown']:.2f}")
```

## 进阶功能

### 自定义交易策略

可以通过继承基础策略类来创建自定义交易策略:

```python
from trading_engine.strategies import BaseStrategy

class MyCustomStrategy(BaseStrategy):
    def __init__(self, config=None):
        super().__init__(config)
        # 初始化自定义参数
        
    def update(self, timestamp, market_data, account_value):
        # 实现策略逻辑
        # 生成交易信号
        return signals
```

### 风险参数定制

可以根据不同的交易需求定制风险参数:

```python
risk_manager.adjust_risk_parameters({
    "max_single_order_amount": 50000.0,
    "max_net_exposure": 0.7,
    "max_drawdown_limit": 0.15
})
```

## 演示与示例

系统包含多个示例脚本，展示如何使用各种功能:

- `examples/backtest_demo.py`: 演示如何回测交易策略
- `examples/realtime_trading.py`: 演示如何进行实时交易
- `examples/market_analysis.py`: 演示如何进行市场分析

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 贡献与支持

欢迎贡献代码或提出问题和建议，请通过GitHub Issues提交。

---

超神量子共生系统 - 用量子智能引领交易未来

