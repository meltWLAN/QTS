"""
data_pipeline - 量子核心组件
高性能数据流水线 - 处理和转换数据流
"""

import logging
import asyncio
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Callable, Optional

logger = logging.getLogger(__name__)

class DataProcessor:
    """数据处理器 - 对数据执行变换操作"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"DataProcessor.{name}")
        
    async def process(self, data: Any) -> Any:
        """处理数据"""
        # 基类方法，子类应重写
        return data
        
    def __str__(self):
        return f"DataProcessor({self.name})"

class DataSource:
    """数据源 - 提供数据"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"DataSource.{name}")
        
    async def fetch_data(self) -> Any:
        """获取数据"""
        # 基类方法，子类应重写
        raise NotImplementedError("必须在子类中实现fetch_data方法")
        
    def __str__(self):
        return f"DataSource({self.name})"

class DataSink:
    """数据接收器 - 接收并处理最终数据"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"DataSink.{name}")
        
    async def receive_data(self, data: Any) -> None:
        """接收数据"""
        # 基类方法，子类应重写
        pass
        
    def __str__(self):
        return f"DataSink({self.name})"

class DataPipeline:
    """数据流水线 - 管理数据流"""
    
    def __init__(self, name: str = "default"):
        self.name = name
        self.processors = []
        self.sources = {}
        self.sinks = {}
        self.is_running = False
        self.pipeline_task = None
        self.logger = logging.getLogger(f"DataPipeline.{name}")
        self.logger.info(f"数据流水线[{name}]初始化完成")
        
    def add_processor(self, processor: DataProcessor) -> 'DataPipeline':
        """添加数据处理器"""
        self.processors.append(processor)
        self.logger.debug(f"已添加处理器: {processor}")
        return self
        
    def add_source(self, name: str, source: DataSource) -> 'DataPipeline':
        """添加数据源"""
        self.sources[name] = source
        self.logger.debug(f"已添加数据源: {source}")
        return self
        
    def add_sink(self, name: str, sink: DataSink) -> 'DataPipeline':
        """添加数据接收器"""
        self.sinks[name] = sink
        self.logger.debug(f"已添加数据接收器: {sink}")
        return self
        
    async def process_data(self, data: Any) -> Any:
        """处理数据通过流水线"""
        result = data
        for processor in self.processors:
            try:
                result = await processor.process(result)
            except Exception as e:
                self.logger.error(f"处理器[{processor.name}]处理数据时出错: {str(e)}")
                raise
        return result
        
    async def run_once(self, source_name: str, sink_name: str = None) -> Any:
        """运行流水线一次"""
        if source_name not in self.sources:
            raise ValueError(f"未找到数据源: {source_name}")
            
        try:
            # 获取数据
            source = self.sources[source_name]
            data = await source.fetch_data()
            
            # 处理数据
            processed_data = await self.process_data(data)
            
            # 发送到接收器
            if sink_name:
                if sink_name not in self.sinks:
                    raise ValueError(f"未找到数据接收器: {sink_name}")
                sink = self.sinks[sink_name]
                await sink.receive_data(processed_data)
                
            return processed_data
        except Exception as e:
            self.logger.error(f"运行流水线时出错: {str(e)}")
            raise
            
    async def run_continuous(self, source_name: str, sink_name: str, interval: float = 1.0):
        """持续运行流水线"""
        if source_name not in self.sources:
            raise ValueError(f"未找到数据源: {source_name}")
            
        if sink_name not in self.sinks:
            raise ValueError(f"未找到数据接收器: {sink_name}")
            
        self.is_running = True
        last_error_time = 0
        error_count = 0
        
        while self.is_running:
            try:
                await self.run_once(source_name, sink_name)
                # 重置错误计数
                error_count = 0
            except Exception as e:
                # 错误处理和退避逻辑
                current_time = time.time()
                if current_time - last_error_time < 5:
                    error_count += 1
                else:
                    error_count = 1
                    
                last_error_time = current_time
                
                # 计算退避时间
                backoff_time = min(30, interval * (2 ** error_count))
                self.logger.warning(f"发生错误，退避 {backoff_time:.1f} 秒后重试: {str(e)}")
                await asyncio.sleep(backoff_time)
                continue
                
            # 等待下一个周期
            await asyncio.sleep(interval)
            
    async def start(self, source_name: str, sink_name: str, interval: float = 1.0):
        """启动流水线"""
        if self.is_running:
            self.logger.warning("流水线已在运行中")
            return
            
        self.logger.info(f"启动流水线，数据源: {source_name}，数据接收器: {sink_name}，间隔: {interval}秒")
        self.pipeline_task = asyncio.create_task(
            self.run_continuous(source_name, sink_name, interval)
        )
        
    async def stop(self):
        """停止流水线"""
        if not self.is_running:
            return
            
        self.logger.info("停止流水线")
        self.is_running = False
        
        if self.pipeline_task:
            # 取消任务
            self.pipeline_task.cancel()
            
            try:
                await self.pipeline_task
            except asyncio.CancelledError:
                pass
                
            self.pipeline_task = None

class TushareDataSource(DataSource):
    """Tushare数据源"""
    
    def __init__(self, token: str, symbols: List[str], start_date: str, end_date: str):
        super().__init__("Tushare")
        self.token = token
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.tushare_api = None
        self._initialize_api()
        
    def _initialize_api(self):
        """初始化Tushare API"""
        try:
            import tushare as ts
            ts.set_token(self.token)
            self.tushare_api = ts.pro_api()
            self.logger.info("Tushare API初始化成功")
        except ImportError:
            self.logger.error("无法导入tushare库")
        except Exception as e:
            self.logger.error(f"初始化Tushare API失败: {str(e)}")
            
    async def fetch_data(self) -> Dict[str, pd.DataFrame]:
        """获取股票数据"""
        if not self.tushare_api:
            raise ValueError("Tushare API未初始化")
            
        result = {}
        loop = asyncio.get_event_loop()
        
        for symbol in self.symbols:
            try:
                # 使用run_in_executor在线程池中执行阻塞调用
                df = await loop.run_in_executor(
                    None,
                    lambda: self.tushare_api.daily(
                        ts_code=symbol,
                        start_date=self.start_date.replace('-', ''),
                        end_date=self.end_date.replace('-', '')
                    )
                )
                
                if df is not None and not df.empty:
                    result[symbol] = df
                    self.logger.debug(f"成功获取{symbol}数据，共{len(df)}条记录")
                else:
                    self.logger.warning(f"获取{symbol}数据为空")
            except Exception as e:
                self.logger.error(f"获取{symbol}数据失败: {str(e)}")
                
        return result

class DatabaseSink(DataSink):
    """数据库接收器"""
    
    def __init__(self, db_connection):
        super().__init__("Database")
        self.db_connection = db_connection
        
    async def receive_data(self, data: Dict[str, pd.DataFrame]) -> None:
        """保存数据到数据库"""
        if not data:
            return
            
        loop = asyncio.get_event_loop()
        
        for symbol, df in data.items():
            try:
                # 使用run_in_executor在线程池中执行阻塞调用
                await loop.run_in_executor(
                    None,
                    lambda: df.to_sql(
                        name=f"stock_{symbol.replace('.', '_')}",
                        con=self.db_connection,
                        if_exists='append',
                        index=False
                    )
                )
                self.logger.info(f"成功保存{symbol}数据到数据库")
            except Exception as e:
                self.logger.error(f"保存{symbol}数据到数据库失败: {str(e)}")

class CleanDataProcessor(DataProcessor):
    """数据清洗处理器"""
    
    def __init__(self):
        super().__init__("CleanData")
        
    async def process(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """清洗数据"""
        if not data:
            return data
            
        result = {}
        
        for symbol, df in data.items():
            try:
                # 复制数据
                clean_df = df.copy()
                
                # 处理缺失值
                clean_df = clean_df.fillna({
                    'open': clean_df['close'],
                    'high': clean_df['close'],
                    'low': clean_df['close'],
                    'vol': 0
                })
                
                # 处理异常值
                # 例如：将成交量小于0的值设为0
                if 'vol' in clean_df.columns:
                    clean_df.loc[clean_df['vol'] < 0, 'vol'] = 0
                
                # 数据排序
                if 'trade_date' in clean_df.columns:
                    clean_df = clean_df.sort_values('trade_date')
                    
                result[symbol] = clean_df
                self.logger.debug(f"成功清洗{symbol}数据")
            except Exception as e:
                self.logger.error(f"清洗{symbol}数据失败: {str(e)}")
                result[symbol] = df  # 使用原始数据
                
        return result

class CalculateIndicatorsProcessor(DataProcessor):
    """计算技术指标处理器"""
    
    def __init__(self):
        super().__init__("CalculateIndicators")
        
    async def process(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """计算技术指标"""
        if not data:
            return data
            
        result = {}
        
        for symbol, df in data.items():
            try:
                # 复制数据
                indicator_df = df.copy()
                
                # 计算移动平均线
                if 'close' in indicator_df.columns:
                    # 5日均线
                    indicator_df['ma5'] = indicator_df['close'].rolling(window=5).mean()
                    # 10日均线
                    indicator_df['ma10'] = indicator_df['close'].rolling(window=10).mean()
                    # 20日均线
                    indicator_df['ma20'] = indicator_df['close'].rolling(window=20).mean()
                    
                # 计算相对强弱指标(RSI)
                if 'close' in indicator_df.columns:
                    # 计算价格变化
                    delta = indicator_df['close'].diff()
                    # 分离上涨和下跌
                    gain = delta.where(delta > 0, 0)
                    loss = -delta.where(delta < 0, 0)
                    
                    # 计算平均上涨和下跌
                    avg_gain = gain.rolling(window=14).mean()
                    avg_loss = loss.rolling(window=14).mean()
                    
                    # 计算相对强度
                    rs = avg_gain / avg_loss
                    # 计算RSI
                    indicator_df['rsi'] = 100 - (100 / (1 + rs))
                    
                # 计算布林带
                if 'close' in indicator_df.columns:
                    # 20日移动平均线
                    indicator_df['boll_mid'] = indicator_df['close'].rolling(window=20).mean()
                    # 标准差
                    indicator_df['boll_std'] = indicator_df['close'].rolling(window=20).std()
                    # 上轨
                    indicator_df['boll_upper'] = indicator_df['boll_mid'] + 2 * indicator_df['boll_std']
                    # 下轨
                    indicator_df['boll_lower'] = indicator_df['boll_mid'] - 2 * indicator_df['boll_std']
                
                result[symbol] = indicator_df
                self.logger.debug(f"成功计算{symbol}的技术指标")
            except Exception as e:
                self.logger.error(f"计算{symbol}的技术指标失败: {str(e)}")
                result[symbol] = df  # 使用原始数据
                
        return result

class EventEmitterSink(DataSink):
    """事件发送器接收器"""
    
    def __init__(self, event_system, event_type: str = "market_data_updated"):
        super().__init__("EventEmitter")
        self.event_system = event_system
        self.event_type = event_type
        
    async def receive_data(self, data: Dict[str, pd.DataFrame]) -> None:
        """将数据发送为事件"""
        if not data or not self.event_system:
            return
            
        # 发送整体市场数据更新事件
        self.event_system.emit_event(self.event_type, {
            'timestamp': time.time(),
            'data_count': len(data),
            'symbols': list(data.keys())
        })
        
        # 为每个股票发送单独的更新事件
        for symbol, df in data.items():
            event_data = {
                'symbol': symbol,
                'timestamp': time.time(),
                'data': df.to_dict('records') if len(df) < 100 else {'record_count': len(df)}
            }
            
            self.event_system.emit_event(f"{self.event_type}.{symbol}", event_data)
            
        self.logger.debug(f"已发送市场数据更新事件，共{len(data)}个股票")

class MarketDataPipeline:
    """市场数据流水线"""
    
    def __init__(self, event_system):
        self.event_system = event_system
        self.pipeline = DataPipeline("MarketData")
        self.logger = logging.getLogger("MarketDataPipeline")
        self.logger.info("市场数据流水线初始化完成")
        
    async def initialize(self, tushare_token: str = None, symbols: List[str] = None):
        """初始化流水线"""
        # 默认的股票列表
        if not symbols:
            symbols = ['000001.SZ', '600000.SH', '000300.SH']
            
        # 获取今天和30天前的日期
        from datetime import datetime, timedelta
        today = datetime.now().strftime("%Y-%m-%d")
        thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        
        # 添加Tushare数据源
        if tushare_token:
            tushare_source = TushareDataSource(
                token=tushare_token,
                symbols=symbols,
                start_date=thirty_days_ago,
                end_date=today
            )
            self.pipeline.add_source("tushare", tushare_source)
            
        # 添加数据处理器
        self.pipeline.add_processor(CleanDataProcessor())
        self.pipeline.add_processor(CalculateIndicatorsProcessor())
        
        # 添加事件发送器接收器
        event_sink = EventEmitterSink(self.event_system)
        self.pipeline.add_sink("event", event_sink)
        
        self.logger.info("市场数据流水线初始化成功")
        
    async def start(self, interval: float = 60.0):
        """启动市场数据流水线"""
        # 检查是否有数据源
        if not self.pipeline.sources:
            self.logger.warning("没有配置数据源，无法启动流水线")
            return False
            
        # 获取第一个数据源的名称
        source_name = next(iter(self.pipeline.sources.keys()))
        
        # 启动流水线
        await self.pipeline.start(source_name, "event", interval)
        self.logger.info(f"市场数据流水线已启动，更新间隔: {interval}秒")
        return True
        
    async def stop(self):
        """停止市场数据流水线"""
        await self.pipeline.stop()
        self.logger.info("市场数据流水线已停止")
        
    def start_sync(self, interval: float = 60.0):
        """同步启动市场数据流水线（用于非异步环境）"""
        loop = asyncio.get_event_loop()
        if not loop.is_running():
            return loop.run_until_complete(self.start(interval))
        else:
            # 创建任务
            task = loop.create_task(self.start(interval))
            return True
            
    def stop_sync(self):
        """同步停止市场数据流水线（用于非异步环境）"""
        loop = asyncio.get_event_loop()
        if not loop.is_running():
            return loop.run_until_complete(self.stop())
        else:
            # 创建任务
            task = loop.create_task(self.stop())
            return True

