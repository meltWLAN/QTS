# 初始化回测引擎
logger.info("初始化回测引擎")
backtest = BacktestEngine(
    start_date=start_date,
    end_date=end_date,
    symbols=SYMBOLS,
    initial_capital=initial_capital
)
# 设置组件
backtest.data_handler = data_handler
backtest.strategy = strategy
backtest.portfolio = portfolio

# 运行回测
logger.info(f"开始回测 {start_date} 到 {end_date}")
results = backtest.run() 