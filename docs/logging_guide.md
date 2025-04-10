# 量子共生网络日志系统使用指南

本文档详细介绍了量子共生网络的日志系统功能和使用方法。日志系统提供了统一的日志记录接口，适用于各种场景，特别是针对量子交易和回测进行了优化。

## 1. 基本使用

### 1.1 导入日志模块

```python
# 导入基本日志设置函数
from src.utils.logger import setup_logger

# 或者导入所有日志工具
from src.utils import setup_logger, get_module_logger, log_info, log_error
```

### 1.2 设置全局日志

在应用程序入口点设置全局日志记录器：

```python
import logging
from src.utils.logger import setup_logger

# 设置日志，默认级别为INFO
logger = setup_logger()

# 或者指定日志文件和级别
logger = setup_logger(
    log_file="my_application.log",
    log_level=logging.DEBUG
)
```

### 1.3 在模块中使用日志

在各个模块中获取和使用日志记录器：

```python
from src.utils.logger import get_module_logger

# 获取当前模块的日志记录器
logger = get_module_logger(__name__)

# 记录不同级别的日志
logger.debug("这是调试信息")
logger.info("这是一般信息")
logger.warning("这是警告信息")
logger.error("这是错误信息")
logger.critical("这是严重错误信息")

# 记录带异常堆栈的错误
try:
    # 某些可能引发异常的操作
    result = 1 / 0
except Exception as e:
    logger.exception("操作失败：%s", str(e))
```

### 1.4 便捷日志函数

对于简单的日志记录，可以使用便捷函数：

```python
from src.utils.logger import log_info, log_error, log_warning, log_debug, log_critical

# 直接记录日志，无需先获取日志记录器
log_info("这是一条信息日志")
log_error("这是一条错误日志")
```

## 2. 回测日志

回测系统有专门的日志配置函数，提供了优化的日志格式和文件处理：

```python
from src.utils.logger import setup_backtest_logger

# 创建回测专用日志记录器
logger = setup_backtest_logger(
    strategy_name="量子爆发增强策略",  # 策略名称会体现在日志文件名中
    log_level=logging.INFO
)

# 记录回测信息
logger.info("=============================================")
logger.info("回测启动")
logger.info(f"回测区间: 20210101 - 20210630")
logger.info(f"回测标的: 53 只股票")
logger.info(f"初始资金: 2,000,000.00")
logger.info("=============================================")

# 记录各种回测过程...

# 记录回测结果
logger.info("回测完成，结果如下:")
logger.info(f"最终权益: 2,153,421.32")
logger.info(f"总收益率: 7.67%")
```

回测日志会自动保存为 `quantum_backtest_YYYYMMDD_HHMMSS.log` 格式的文件。

## 3. 高级配置

### 3.1 自定义日志格式

可以自定义日志格式：

```python
logger = setup_logger(
    log_format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
```

### 3.2 设置日志文件滚动

日志系统支持日志文件滚动，当文件达到指定大小时自动创建新文件：

```python
logger = setup_logger(
    max_file_size=50*1024*1024,  # 设置为50MB
    backup_count=10              # 保留10个备份文件
)
```

### 3.3 关闭控制台输出

在某些场景下可能只需要文件日志而不需要控制台输出：

```python
logger = setup_logger(
    log_to_console=False
)
```

## 4. 最佳实践

### 4.1 日志分级使用建议

- **DEBUG**: 详细的调试信息，帮助诊断问题
- **INFO**: 一般操作信息，确认程序按预期运行
- **WARNING**: 表示可能出现的问题，但程序仍能正常运行
- **ERROR**: 由于严重问题，程序无法执行某些功能
- **CRITICAL**: 严重错误，可能导致程序终止

### 4.2 回测日志标准格式

建议在回测日志中使用以下标准格式：

1. 开始部分：记录基本设置和参数
2. 过程部分：记录每笔交易的信号和执行
3. 结束部分：记录汇总结果和性能指标

### 4.3 性能考虑

- 在高频交易或大型回测中，建议将DEBUG级别的日志限制在关键部分
- 对于特别大的回测，可以增加日志文件的最大大小和备份数量

## 5. 示例

完整的回测日志示例：

```python
from src.utils.logger import setup_backtest_logger
import logging

def run_backtest():
    # 设置日志
    logger = setup_backtest_logger("SuperQuantumStrategy")
    
    # 记录基本信息
    logger.info("=============================================")
    logger.info("超量子策略回测启动")
    logger.info("回测区间: 20210101-20220101")
    logger.info("回测标的: 沪深300成分股")
    logger.info("初始资金: 10,000,000.00")
    logger.info("策略参数:")
    logger.info("- 信号阈值: 0.75")
    logger.info("- 止损比例: 0.05")
    logger.info("=============================================")
    
    # ... 回测过程 ...
    
    # 记录结果
    logger.info("=============================================")
    logger.info("回测完成，结果如下:")
    logger.info("最终资金: 12,345,678.90")
    logger.info("总收益率: 23.46%")
    logger.info("年化收益: 18.32%")
    logger.info("最大回撤: 15.78%")
    logger.info("夏普比率: 1.42")
    logger.info("交易次数: 126")
    logger.info("胜率: 58.73%")
    logger.info("=============================================")

if __name__ == "__main__":
    run_backtest()
``` 