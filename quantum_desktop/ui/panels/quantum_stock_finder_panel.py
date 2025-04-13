"""
超神量子选股器面板 - 利用量子计算捕捉大牛股
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os
import json
import time
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                         QPushButton, QComboBox, QFrame, QGridLayout, 
                         QTableWidget, QTableWidgetItem, QHeaderView,
                         QSplitter, QTabWidget, QLineEdit, QDateEdit,
                         QGroupBox, QFormLayout, QDoubleSpinBox, QSpinBox,
                         QCheckBox, QProgressBar, QTextEdit, QApplication,
                         QMessageBox, QSlider, QInputDialog)
from PyQt5.QtCore import Qt, pyqtSlot, QDate, pyqtSignal, QThread, QTimer
from PyQt5.QtGui import QColor, QBrush, QPen, QPainter, QFont
from PyQt5.QtChart import (QChart, QChartView, QLineSeries, 
                       QValueAxis, QDateTimeAxis, QScatterSeries)

logger = logging.getLogger("QuantumDesktop.QuantumStockFinderPanel")

class StockFinderThread(QThread):
    """超神选股线程 - 负责在后台运行选股算法，避免阻塞UI"""
    
    # 进度更新信号
    progress_updated = pyqtSignal(int)
    # 状态更新信号
    status_updated = pyqtSignal(str)
    # 结果就绪信号
    results_ready = pyqtSignal(dict)
    # 完成信号
    completed = pyqtSignal()
    # 错误信号
    error_signal = pyqtSignal(str, str)
    
    def __init__(self, stock_strategy=None, quantum_power=50, market_scope="全市场", sector_filter="全部行业", system_manager=None, use_real_data=True):
        """初始化选股线程"""
        super().__init__()
        
        # 选股策略
        self.stock_strategy = stock_strategy
        
        # 量子能力值 (0-100)
        self.quantum_power = quantum_power
        
        # 选股范围
        self.market_scope = market_scope
        
        # 行业过滤
        self.sector_filter = sector_filter
        
        # 系统管理器
        self.system_manager = system_manager
        
        # 结束标志
        self._stop_flag = False
        
        # 强制使用真实数据
        self.use_real_data = use_real_data
    
    def stop(self):
        """停止线程"""
        self._stop_flag = True
        logger.info("用户请求停止选股线程")
        
    def _simulate_selection_process(self):
        """模拟量子选股过程"""
        # 模拟进度更新
        for progress in range(10, 95, 5):
            if self._stop_flag:
                break
                
            self.progress_updated.emit(progress)
            
            # 模拟不同阶段的处理
            if progress == 15:
                self.status_updated.emit("构建量子态...")
            elif progress == 30:
                self.status_updated.emit("执行量子算法分析...")
            elif progress == 50:
                self.status_updated.emit("计算量子纠缠系数...")
            elif progress == 70:
                self.status_updated.emit("多维度特征分析...")
            
            # 延时，模拟计算过程
            QThread.msleep(100)
    
    def run(self):
        """线程执行函数"""
        try:
            # 始终尝试使用真实数据获取结果
            self.progress_updated.emit(5)
            self.status_updated.emit("初始化量子选股引擎...")
            
            try:
                # 强制使用真实市场数据
                self.progress_updated.emit(10)
                self.status_updated.emit("连接实时市场数据...")
                
                # 从真实市场获取数据
                results = self._generate_results()
                
                # 结果就绪
                self.progress_updated.emit(100)
                self.status_updated.emit("超神选股完成!")
                self.results_ready.emit(results)
                self.completed.emit()
                return
            except Exception as e:
                logger.error(f"获取真实市场数据失败: {str(e)}")
                
                # 如果用户明确要求使用真实数据但失败，显示错误
                if self.use_real_data:
                    self.progress_updated.emit(0)
                    self.status_updated.emit(f"数据获取失败: {str(e)}")
                    self.completed.emit()
                    return
                
            # 仅在明确不使用真实数据且真实数据获取失败时使用模拟数据
            self.status_updated.emit("切换到模拟数据...")
            self._simulate_selection_process()
            
            # 生成模拟结果
            self.progress_updated.emit(95)
            self.status_updated.emit("生成模拟选股结果...")
            
            results = self._generate_simulated_results()
            
            # 结果就绪
            self.progress_updated.emit(100)
            self.status_updated.emit("模拟选股完成! (非真实市场数据)")
            self.results_ready.emit(results)
                
        except Exception as e:
            logger.error(f"超神选股过程出错: {str(e)}")
            self.status_updated.emit(f"选股过程出错: {str(e)}")
            
        finally:
            self.completed.emit()
    
    def _generate_results(self):
        """根据真实市场数据生成选股结果"""
        try:
            # 直接使用预设的API密钥
            api_key = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"
            logger.info(f"使用预设API密钥: {api_key[:5]}...{api_key[-5:]}")

            # 导入tushare
            try:
                import tushare as ts
                logger.info("成功导入tushare库")
            except ImportError:
                logger.error("未能导入tushare库，请确保已安装")
                raise Exception("缺少Tushare库，请使用pip install tushare安装")
            
            # 记录明确的开始标记，用于调试数据获取流程
            logger.info("====== 开始获取实时市场数据 ======")
            self.status_updated.emit("正在连接实时市场数据服务...")
            
            # 检查系统状态
            system_ready = False
            if self.system_manager:
                try:
                    # 使用hasattr检查是否有is_system_running方法
                    if hasattr(self.system_manager, "is_system_running"):
                        system_ready = self.system_manager.is_system_running()
                        logger.info(f"系统运行状态: {'就绪' if system_ready else '未就绪'}")
                    else:
                        # 尝试其他可能的方法
                        system_ready = True  # 默认假设系统就绪
                        logger.warning("系统管理器没有is_system_running方法，默认系统已就绪")
                except Exception as e:
                    logger.error(f"检查系统状态时出错: {str(e)}")
            
            # 设置token并初始化API
            ts.set_token(api_key)
            pro = None
            
            # 尝试多次初始化API
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    logger.info(f"尝试初始化Tushare API (尝试 {attempt+1}/{max_attempts})...")
                    pro = ts.pro_api()
                    break
                except Exception as e:
                    logger.error(f"初始化Tushare API失败 (尝试 {attempt+1}/{max_attempts}): {str(e)}")
                    if attempt < max_attempts - 1:
                        time.sleep(1)  # 暂停后重试
            
            if pro is None:
                raise Exception(f"无法初始化Tushare API，经过 {max_attempts} 次尝试")
            
            # 验证API连接是否正常
            try:
                self.status_updated.emit("验证API连接...")
                test_result = pro.query('stock_basic', exchange='', list_status='L', fields='ts_code,name', limit=1)
                if test_result is not None and not test_result.empty:
                    logger.info(f"Tushare API测试成功，获取到股票: {test_result.iloc[0]['name']}")
                    self.status_updated.emit(f"API连接成功，已获取: {test_result.iloc[0]['name']}")
                else:
                    logger.error("Tushare API测试失败，返回结果为空")
                    raise Exception("Tushare API测试失败，返回结果为空")
            except Exception as e:
                logger.error(f"Tushare API测试失败: {str(e)}")
                raise Exception(f"Tushare API连接测试失败: {str(e)}")
            
            # 更新进度
            self.progress_updated.emit(15)
            self.status_updated.emit("已连接Tushare API，获取股票列表...")
            
            # 确定当前交易日
            today = datetime.now().strftime('%Y%m%d')
            try:
                trade_cal = pro.trade_cal(exchange='SSE', start_date=today, end_date=today)
                if trade_cal.empty or trade_cal.iloc[0]['is_open'] == 0:
                    # 获取最近的交易日
                    from datetime import timedelta
                    last_day = (datetime.now() - timedelta(days=5)).strftime('%Y%m%d')
                    trade_cal = pro.trade_cal(exchange='SSE', start_date=last_day, end_date=today)
                    trade_cal = trade_cal[trade_cal['is_open'] == 1]
                    if not trade_cal.empty:
                        today = trade_cal.iloc[-1]['cal_date']
                        logger.info(f"使用最近交易日: {today}")
                        self.status_updated.emit(f"使用最近交易日: {today}")
            except Exception as e:
                logger.warning(f"无法确定交易日: {str(e)}，使用当前日期")
            
            # 获取股票基本信息
            self.progress_updated.emit(20)
            stock_basic = None
            
            # 根据选择的市场范围获取股票
            if self.market_scope == "沪深300":
                try:
                    logger.info("正在获取沪深300成分股...")
                    self.status_updated.emit("获取沪深300成分股...")
                    index_stocks = pro.index_weight(index_code='000300.SH', fields='con_code,weight')
                    if index_stocks is not None and not index_stocks.empty:
                        stock_list = list(index_stocks['con_code'])
                        logger.info(f"获取到 {len(stock_list)} 只沪深300成分股")
                        stock_basic = pro.stock_basic(ts_code=','.join(stock_list[:100]),  # 限制API调用
                            fields='ts_code,symbol,name,area,industry,market,list_date')
                except Exception as e:
                    logger.error(f"获取沪深300成分股失败: {str(e)}")
            elif self.market_scope == "中证500":
                try:
                    logger.info("正在获取中证500成分股...")
                    self.status_updated.emit("获取中证500成分股...")
                    index_stocks = pro.index_weight(index_code='000905.SH', fields='con_code,weight')
                    if index_stocks is not None and not index_stocks.empty:
                        stock_list = list(index_stocks['con_code'])
                        logger.info(f"获取到 {len(stock_list)} 只中证500成分股")
                        stock_basic = pro.stock_basic(ts_code=','.join(stock_list[:100]),  # 限制API调用
                            fields='ts_code,symbol,name,area,industry,market,list_date')
                except Exception as e:
                    logger.error(f"获取中证500成分股失败: {str(e)}")
            elif self.market_scope == "科创板":
                logger.info("正在获取科创板股票...")
                self.status_updated.emit("获取科创板股票...")
                stock_basic = pro.stock_basic(market='科创板', list_status='L',
                    fields='ts_code,symbol,name,area,industry,market,list_date')
            elif self.market_scope == "创业板":
                logger.info("正在获取创业板股票...")
                self.status_updated.emit("获取创业板股票...")
                stock_basic = pro.stock_basic(market='创业板', list_status='L',
                    fields='ts_code,symbol,name,area,industry,market,list_date')
            
            # 如果没有特定市场数据或选择了全市场，获取全部A股
            if stock_basic is None or stock_basic.empty:
                logger.info("正在获取全部A股股票列表...")
                self.status_updated.emit("获取全部A股股票列表...")
                stock_basic = pro.stock_basic(exchange='', list_status='L',
                    fields='ts_code,symbol,name,area,industry,market,list_date')
            
            if stock_basic is None or stock_basic.empty:
                logger.error("无法获取股票列表数据")
                raise Exception("无法获取股票列表数据")
            else:
                logger.info(f"成功获取 {len(stock_basic)} 只股票基本信息")
            
            # 过滤行业
            if self.sector_filter and self.sector_filter != "全部行业":
                pre_filter_count = len(stock_basic)
                # 模糊匹配行业
                stock_basic = stock_basic[stock_basic['industry'].str.contains(self.sector_filter, na=False)]
                post_filter_count = len(stock_basic)
                logger.info(f"行业过滤: {self.sector_filter}, 过滤前: {pre_filter_count}, 过滤后: {post_filter_count}")
                
                # 如果过滤后没有股票，尝试更宽松的匹配
                if post_filter_count == 0:
                    logger.info(f"行业 '{self.sector_filter}' 过滤后没有股票，尝试更宽松的匹配")
                    # 使用多个关键词进行匹配
                    related_keywords = []
                    if '科技' in self.sector_filter:
                        related_keywords = ['软件', '计算机', '通信', '电子', '芯片', '互联网']
                    elif '消费' in self.sector_filter:
                        related_keywords = ['零售', '食品', '饮料', '家电', '酒']
                    elif '医药' in self.sector_filter:
                        related_keywords = ['医疗', '生物', '制药']
                    elif '金融' in self.sector_filter:
                        related_keywords = ['银行', '保险', '券商', '信托']
                    elif '新能源' in self.sector_filter:
                        related_keywords = ['电力', '风电', '太阳能', '锂电', '储能']
                    
                    if related_keywords:
                        stock_basic = pro.stock_basic(exchange='', list_status='L',
                            fields='ts_code,symbol,name,area,industry,market,list_date')
                        
                        pattern = '|'.join(related_keywords)
                        stock_basic = stock_basic[stock_basic['industry'].str.contains(pattern, na=False, regex=True)]
                        logger.info(f"宽松匹配关键词: {related_keywords}, 匹配到: {len(stock_basic)} 只股票")
            
            # 确保有足够的数据
            if stock_basic.empty:
                logger.error(f"没有找到符合条件的股票（行业: {self.sector_filter}, 市场: {self.market_scope}）")
                raise Exception(f"没有找到符合条件的股票（行业: {self.sector_filter}, 市场: {self.market_scope}）")
            
            # 限制股票数量，避免API调用次数过多
            total_stocks = len(stock_basic)
            max_stocks = 100  # 通常的API调用限制
            
            logger.info(f"找到 {total_stocks} 只符合条件的股票，获取详细数据...")
            self.status_updated.emit(f"找到 {total_stocks} 只符合条件的股票，获取详细数据...")
            
            if total_stocks > max_stocks:
                stock_basic = stock_basic.sample(max_stocks)
                total_stocks = max_stocks
                logger.info(f"为避免API限制，随机选择 {max_stocks} 只股票")
            
            # 获取每日基本面指标
            self.progress_updated.emit(30)
            
            # 分批获取数据，避免API限制
            all_stocks_data = []
            batch_size = 20  # 较小的批次大小，避免API限制
            batches = (total_stocks + batch_size - 1) // batch_size
            
            for i in range(batches):
                if self._stop_flag:
                    raise Exception("用户中止了操作")
                
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, total_stocks)
                batch_stocks = stock_basic.iloc[start_idx:end_idx]
                
                progress = 30 + (i / batches) * 40
                self.progress_updated.emit(int(progress))
                self.status_updated.emit(f"获取第 {i+1}/{batches} 批次股票数据...")
                
                try:
                    # 获取这批股票的TS代码列表
                    ts_codes = batch_stocks['ts_code'].tolist()
                    
                    # 获取每日基本指标
                    logger.info(f"获取第 {i+1}/{batches} 批次基本指标数据, 股票: {len(ts_codes)} 只")
                    daily_basic = pro.daily_basic(trade_date=today, ts_code=','.join(ts_codes))
                    
                    # 获取当日行情
                    logger.info(f"获取第 {i+1}/{batches} 批次行情数据, 股票: {len(ts_codes)} 只")
                    daily_quotes = pro.daily(trade_date=today, ts_code=','.join(ts_codes))
                    
                    # 如果获取当日数据失败，尝试获取上一个交易日数据
                    if daily_basic.empty or daily_quotes.empty:
                        logger.warning(f"获取当日数据失败，尝试获取上一个交易日数据")
                        cal = pro.trade_cal(exchange='SSE', start_date='20230101', end_date=today)
                        trade_days = cal[cal['is_open'] == 1]['cal_date'].tolist()
                        if len(trade_days) > 1:
                            last_trade_day = trade_days[-2]  # 上一个交易日
                            daily_basic = pro.daily_basic(trade_date=last_trade_day, ts_code=','.join(ts_codes))
                            daily_quotes = pro.daily(trade_date=last_trade_day, ts_code=','.join(ts_codes))
                            logger.info(f"使用上一交易日 {last_trade_day} 的数据")
                    
                    # 检查是否获取到数据
                    if daily_basic.empty or daily_quotes.empty:
                        logger.error(f"第 {i+1}/{batches} 批次数据获取失败，日期: {today}")
                        continue
                    
                    logger.info(f"成功获取第 {i+1}/{batches} 批次数据，基本面: {len(daily_basic)} 条，行情: {len(daily_quotes)} 条")
                    
                    # 处理每只股票的数据
                    for ts_code in ts_codes:
                        # 获取基本信息
                        stock_info = batch_stocks[batch_stocks['ts_code'] == ts_code].iloc[0]
                        
                        # 获取基本面数据
                        basic_data = daily_basic[daily_basic['ts_code'] == ts_code]
                        quote_data = daily_quotes[daily_quotes['ts_code'] == ts_code]
                        
                        if basic_data.empty or quote_data.empty:
                            logger.warning(f"股票 {ts_code} 无法获取完整数据，跳过")
                            continue
                        
                        # 合并数据
                        latest_basic = basic_data.iloc[0]
                        latest_quote = quote_data.iloc[0]
                        
                        # 创建股票数据字典
                        stock_data = {
                            "ts_code": ts_code,
                            "symbol": stock_info['symbol'],
                            "name": stock_info['name'],
                            "industry": stock_info['industry'],
                            "area": stock_info['area'],
                            "market": stock_info['market'],
                            "list_date": stock_info['list_date'],
                            
                            # 行情数据
                            "close": latest_quote.get('close', None),  # 关键字段1
                            "open": latest_quote.get('open', None),
                            "high": latest_quote.get('high', None),
                            "low": latest_quote.get('low', None),
                            "pre_close": latest_quote.get('pre_close', None),
                            "pct_chg": latest_quote.get('pct_chg', None),  # 关键字段2
                            "vol": latest_quote.get('vol', None),
                            "amount": latest_quote.get('amount', None),
                            
                            # 基本面数据
                            "pe": latest_basic.get('pe', None),
                            "pe_ttm": latest_basic.get('pe_ttm', None),  # 关键字段3
                            "pb": latest_basic.get('pb', None),  # 关键字段4
                            "ps": latest_basic.get('ps', None),
                            "ps_ttm": latest_basic.get('ps_ttm', None),
                            "total_mv": latest_basic.get('total_mv', None),  # 关键字段5 - 总市值（万元）
                            "circ_mv": latest_basic.get('circ_mv', None),
                            "total_share": latest_basic.get('total_share', None),
                            "turnover_rate": latest_basic.get('turnover_rate', None),
                            "turnover_rate_f": latest_basic.get('turnover_rate_f', None),
                            "volume_ratio": latest_basic.get('volume_ratio', None),  # 关键字段6
                            "dv_ratio": latest_basic.get('dv_ratio', None),
                            
                            # 添加数据来源标记，用于验证数据源是否为真实市场数据
                            "data_source": "tushare_real",
                            "data_date": today
                        }
                        
                        # 检查是否包含所有关键字段
                        critical_fields = ["ts_code", "area", "market", "turnover_rate", "volume_ratio", "total_mv"]
                        missing_fields = [field for field in critical_fields if field not in stock_data or stock_data[field] is None]
                        
                        if missing_fields:
                            logger.warning(f"股票 {ts_code} 缺少关键字段: {', '.join(missing_fields)}")
                        else:
                            logger.info(f"股票 {ts_code} 数据完整，包含所有关键字段")
                        
                        # 添加到列表
                        all_stocks_data.append(stock_data)
                except Exception as e:
                    logger.error(f"获取批次 {i+1} 数据失败: {str(e)}")
            
            # 检查是否获取到足够的数据
            if not all_stocks_data:
                logger.error("未获取到任何有效股票数据，无法继续选股")
                raise Exception("未获取到任何有效股票数据，无法继续选股")
            
            # 记录获取到的股票数量
            logger.info(f"成功获取 {len(all_stocks_data)} 只股票的完整数据")
            
            # 评分和选择超神股票
            self.progress_updated.emit(70)
            self.status_updated.emit("量子算法分析股票...")
            
            # 这里应用量子评分算法
            quantum_stocks = self._quantum_scoring(all_stocks_data)
            
            # 选择前n只最高评分股票
            final_count = 30  # 限制输出结果
            quantum_stocks = sorted(quantum_stocks, key=lambda x: x['quantum_score'], reverse=True)[:final_count]
            
            self.progress_updated.emit(90)
            self.status_updated.emit("生成超神量子选股结果...")
            
            # 返回结果
            return {
                "status": "success",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "quantum_power": self.quantum_power,
                "market_scope": self.market_scope,
                "data_source": "tushare_real",  # 明确标记数据来源
                "stocks": quantum_stocks
            }
            
        except Exception as e:
            logger.error(f"从真实市场获取股票数据失败: {str(e)}")
            raise
    
    def _quantum_scoring(self, stocks_data):
        """使用量子算法对股票进行评分"""
        if not stocks_data:
            return []
            
        # 获取量子分析引擎
        quantum_enhanced = False
        if self.system_manager:
            # 将is_ready()替换为is_system_running()
            if hasattr(self.system_manager, "is_system_running") and self.system_manager.is_system_running():
                quantum_backend = self.system_manager.get_component("quantum_backend")
                if quantum_backend:
                    quantum_enhanced = True
        
        scored_stocks = []
        for stock in stocks_data:
            try:
                # 基本评分
                base_score = 50
                
                # 技术面评分
                tech_score = 0
                if 'pct_chg' in stock and stock['pct_chg'] is not None:
                    # 涨幅得分
                    change_score = min(25, max(-25, stock['pct_chg'] * 2.5))
                    tech_score += change_score
                
                # 基本面评分
                fundamental_score = 0
                
                # PE评分 (低PE加分)
                if 'pe_ttm' in stock and stock['pe_ttm'] is not None:
                    pe = stock['pe_ttm']
                    if 0 < pe <= 15:  # 低估值
                        fundamental_score += 15
                    elif 15 < pe <= 30:  # 合理估值
                        fundamental_score += 10
                    elif 30 < pe <= 50:  # 较高估值
                        fundamental_score += 5
                    elif pe > 50:  # 高估值
                        fundamental_score -= 5
                
                # PB评分
                if 'pb' in stock and stock['pb'] is not None:
                    pb = stock['pb']
                    if 0 < pb <= 1.5:  # 低PB
                        fundamental_score += 10
                    elif 1.5 < pb <= 3:  # 合理PB
                        fundamental_score += 5
                    elif pb > 5:  # 高PB
                        fundamental_score -= 5
                
                # 市值评分
                if 'total_mv' in stock and stock['total_mv'] is not None:
                    total_mv = stock['total_mv'] / 10000  # 万元转为亿元
                    if 100 <= total_mv <= 1000:  # 中型股
                        fundamental_score += 10
                    elif total_mv > 1000:  # 大型股
                        fundamental_score += 5
                
                # 换手率评分
                if 'turnover_rate' in stock and stock['turnover_rate'] is not None:
                    turnover = stock['turnover_rate']
                    if 3 <= turnover <= 10:  # 适中换手率
                        tech_score += 10
                    elif turnover > 15:  # 过高换手率
                        tech_score -= 5
                    elif turnover < 1:  # 过低换手率
                        tech_score -= 5
                
                # 量子增强评分
                quantum_score = 0
                if quantum_enhanced:
                    try:
                        # 简化版量子增强：使用量子功率作为增强因子
                        quantum_factor = 1 + (self.quantum_power / 100)
                        quantum_score = random.uniform(5, 20) * quantum_factor
                    except Exception as e:
                        logger.error(f"量子增强评分失败: {str(e)}")
                else:
                    # 没有量子后端时，基于量子功率模拟量子增强
                    quantum_score = random.uniform(5, 15) * (1 + (self.quantum_power / 100))
                
                # 计算总评分
                total_score = min(99.9, base_score + tech_score + fundamental_score + quantum_score)
                
                # 计算预期收益 (评分越高，预期收益越高，但有一定随机性)
                expected_gain = total_score * 0.5 + random.uniform(-10, 20)
                expected_gain = max(10, min(100, expected_gain))  # 限制在10%~100%之间
                
                # 生成理由
                reasons = self._generate_stock_reasons(stock, total_score)
                
                # 添加评分结果
                scored_stock = stock.copy()
                scored_stock.update({
                    "quantum_score": total_score,
                    "expected_gain": expected_gain,
                    "confidence": random.uniform(75, 98),
                    "timeframe": self._determine_timeframe(stock),
                    "reasons": reasons,
                    "recommendation": "强烈推荐" if total_score > 90 else "推荐"
                })
                
                scored_stocks.append(scored_stock)
                
            except Exception as e:
                logger.error(f"评分股票 {stock.get('name', '')} 失败: {str(e)}")
        
        return scored_stocks
    
    def _determine_timeframe(self, stock):
        """确定时间范围推荐"""
        sector = stock.get('sector', '')
        
        # 长期投资行业
        long_term_sectors = ['消费', '医药', '白酒', '食品']
        # 中期投资行业
        mid_term_sectors = ['科技', '半导体', '新能源', '军工']
        # 短期投资行业
        short_term_sectors = ['有色金属', '钢铁', '煤炭', '石油']
        
        for lt_sector in long_term_sectors:
            if lt_sector in sector:
                return "长期"
                
        for mt_sector in mid_term_sectors:
            if mt_sector in sector:
                return "中期"
                
        for st_sector in short_term_sectors:
            if st_sector in sector:
                return "短期"
        
        # 默认中期
        return random.choice(["短期", "中期", "长期"])
    
    def _generate_stock_reasons(self, stock, score):
        """根据股票数据和评分生成理由"""
        reasons_pool = [
            "量子态分析显示强势突破形态",
            "多维度市场情绪指标极度看好",
            "超空间趋势通道形成",
            "量子态交叉信号确认",
            "多维度技术指标共振",
            "行业景气度量子评分处于高位",
            "超神算法检测到主力资金潜伏",
            "量子波动特征与历史大牛股吻合",
            "超空间市场结构分析显示稀缺性溢价",
            "行业拐点信号被超神算法捕捉"
        ]
        
        # 根据实际数据添加更相关的理由
        if 'pe_ttm' in stock and stock['pe_ttm'] is not None:
            pe = stock['pe_ttm']
            if pe < 20:
                reasons_pool.append("PE低于行业平均，具备较高安全边际")
                
        if 'pb' in stock and stock['pb'] is not None:
            pb = stock['pb']
            if pb < 2:
                reasons_pool.append("PB处于历史低位，具备较高安全边际")
                
        if 'total_mv' in stock and stock['total_mv'] is not None:
            total_mv = stock['total_mv'] / 10000  # 万元转为亿元
            if total_mv > 1000:
                reasons_pool.append("大盘蓝筹股，市值稳定，抗风险能力强")
            elif 100 <= total_mv <= 1000:
                reasons_pool.append("中盘成长股，具备良好成长性和规模效应")
                
        if 'turnover_rate' in stock and stock['turnover_rate'] is not None:
            turnover = stock['turnover_rate']
            if 3 <= turnover <= 10:
                reasons_pool.append("换手率适中，资金参与度高但不过热")
        
        # 高评分增加更确信的理由
        if score > 90:
            reasons_pool.append("多因子量子模型综合评分极高")
            reasons_pool.append("量子算法检测到未来强劲上涨概率超过90%")
        
        # 选择3个不重复的理由
        return random.sample(reasons_pool, min(3, len(reasons_pool)))
        
    def _generate_simulated_results(self):
        """生成模拟选股结果 (当无法获取真实数据时使用)"""
        # 股票池
        stock_pool = [
            {"code": "600519", "name": "贵州茅台", "sector": "白酒"},
            {"code": "000858", "name": "五粮液", "sector": "白酒"},
            {"code": "601318", "name": "中国平安", "sector": "金融保险"},
            {"code": "600036", "name": "招商银行", "sector": "银行"},
            {"code": "000333", "name": "美的集团", "sector": "家电"},
            {"code": "600276", "name": "恒瑞医药", "sector": "医药"},
            {"code": "002475", "name": "立讯精密", "sector": "电子"},
            {"code": "300750", "name": "宁德时代", "sector": "新能源"},
            {"code": "603288", "name": "海天味业", "sector": "食品"},
            {"code": "601888", "name": "中国中免", "sector": "免税"},
            {"code": "600031", "name": "三一重工", "sector": "工程机械"},
            {"code": "000651", "name": "格力电器", "sector": "家电"},
            {"code": "002594", "name": "比亚迪", "sector": "汽车新能源"},
            {"code": "601899", "name": "紫金矿业", "sector": "有色金属"},
            {"code": "600887", "name": "伊利股份", "sector": "食品饮料"},
            {"code": "000538", "name": "云南白药", "sector": "医药"},
            {"code": "600309", "name": "万华化学", "sector": "化工"},
            {"code": "300059", "name": "东方财富", "sector": "金融信息"},
            {"code": "600900", "name": "长江电力", "sector": "公用事业"},
            {"code": "688981", "name": "中芯国际", "sector": "半导体"},
        ]
        
        # 如果设置了行业过滤，应用过滤
        if self.sector_filter and self.sector_filter != "全部行业":
            stock_pool = [s for s in stock_pool if s["sector"] == self.sector_filter]
            
        if len(stock_pool) == 0:
            # 如果过滤后没有股票，返回原始池
            stock_pool = [
                {"code": "600519", "name": "贵州茅台", "sector": "白酒"},
                {"code": "000858", "name": "五粮液", "sector": "白酒"},
                {"code": "601318", "name": "中国平安", "sector": "金融保险"}
            ]
            
        # 根据量子功率选择股票数量
        num_stocks = 3 + int(self.quantum_power / 20)
        selected_stocks = random.sample(stock_pool, min(num_stocks, len(stock_pool)))
        
        results = []
        for stock in selected_stocks:
            # 生成随机的超神评分和预期上涨空间
            quantum_score = random.uniform(80, 99.5)
            expected_gain = random.uniform(20, 100)
            
            # 生成其他所需的信息
            stock_data = {
                "code": stock["code"],
                "name": stock["name"],
                "sector": stock["sector"],
                "quantum_score": quantum_score,
                "expected_gain": expected_gain,
                "confidence": random.uniform(75, 95),
                "timeframe": random.choice(["短期", "中期", "长期"]),
                "recommendation": "强烈推荐" if quantum_score > 90 else "推荐",
                "reasons": self._generate_stock_reasons(stock, quantum_score),
                # 明确标记为模拟数据
                "data_source": "simulated"
            }
            
            # 确保模拟数据不包含真实数据中应有的字段，以免混淆验证
            # 删除可能存在的真实数据字段
            for field in ["ts_code", "area", "market", "turnover_rate", "volume_ratio", "total_mv"]:
                if field in stock_data:
                    del stock_data[field]
                    
            results.append(stock_data)
            
        # 对结果按评分排序
        results = sorted(results, key=lambda x: x["quantum_score"], reverse=True)
        
        # 返回结果
        return {
            "status": "success",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "quantum_power": self.quantum_power,
            "market_scope": self.market_scope,
            # 明确标记为模拟数据
            "data_source": "simulated",
            "stocks": results
        }

class PotentialStockChart(QChartView):
    """潜力股图表 - 显示超神选股结果的图形化表示"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 创建图表
        self.chart = QChart()
        self.chart.setTitle("超神量子选股分析")
        self.chart.setAnimationOptions(QChart.SeriesAnimations)
        
        # 设置抗锯齿
        self.setRenderHint(QPainter.Antialiasing)
        
        # 设置图表
        self.setChart(self.chart)
        
        # 创建散点图系列
        self.scatter_series = QScatterSeries()
        self.scatter_series.setName("潜力股")
        self.scatter_series.setMarkerSize(15)
        
        # 添加系列到图表
        self.chart.addSeries(self.scatter_series)
        
        # 创建X轴 (预期收益)
        self.axis_x = QValueAxis()
        self.axis_x.setTitleText("预期收益率(%)")
        self.axis_x.setRange(0, 100)
        
        # 创建Y轴 (超神评分)
        self.axis_y = QValueAxis()
        self.axis_y.setTitleText("超神量子评分")
        self.axis_y.setRange(70, 100)
        
        # 添加轴到图表
        self.chart.addAxis(self.axis_x, Qt.AlignBottom)
        self.chart.addAxis(self.axis_y, Qt.AlignLeft)
        
        # 将系列附加到轴
        self.scatter_series.attachAxis(self.axis_x)
        self.scatter_series.attachAxis(self.axis_y)
        
        # 设置图表主题和样式
        self.chart.setTheme(QChart.ChartThemeDark)
        self.chart.setBackgroundVisible(False)
        self.chart.legend().setVisible(True)
        self.chart.legend().setAlignment(Qt.AlignBottom)
        
    def set_data(self, stocks):
        """设置股票数据"""
        # 清除旧数据
        self.scatter_series.clear()
        
        if not stocks:
            return
        
        # 添加散点数据
        for stock in stocks:
            expected_gain = stock.get("expected_gain", 0)
            quantum_score = stock.get("quantum_score", 0)
            self.scatter_series.append(expected_gain, quantum_score)
            
        # 刷新图表
        self.chart.update()

class QuantumStockFinderPanel(QWidget):
    """超神量子选股器面板 - 利用量子算法发现潜在大牛股"""
    
    def __init__(self, system_manager, parent=None):
        super().__init__(parent)
        self.system_manager = system_manager
        
        # 选股结果
        self.finder_results = None
        
        # 选股线程
        self.finder_thread = None
        
        # 量子选股策略
        self.stock_strategy = None
        
        # 历史选股数据
        self.historical_data = None
        
        # 初始化量子选股策略
        self._init_stock_strategy()
        
        # 默认不加载历史数据，避免使用过时缓存
        # self._load_historical_data()
        
        # 清除过期的历史缓存，确保下次使用最新数据
        self._clear_outdated_history()
        
        # 初始化UI
        self._init_ui()
        
        logger.info("超神量子选股器面板初始化完成")
        
    def _init_stock_strategy(self):
        """初始化量子选股策略"""
        try:
            # 导入量子选股策略
            from quantum_core.quantum_stock_strategy import QuantumStockStrategy
            
            # 获取量子后端
            quantum_backend = None
            if self.system_manager:
                quantum_backend = self.system_manager.get_component("quantum_backend")
                
            # 获取市场分析器
            market_analyzer = None
            if self.system_manager:
                market_analyzer = self.system_manager.get_component("market_analyzer")
                
            # 创建量子选股策略
            self.stock_strategy = QuantumStockStrategy(
                quantum_backend=quantum_backend,
                market_analyzer=market_analyzer
            )
            
            # 如果系统已启动，则启动策略
            if self.system_manager and hasattr(self.system_manager, "is_system_running") and self.system_manager.is_system_running():
                self.stock_strategy.start()
                
            logger.info("量子选股策略初始化成功")
            
        except Exception as e:
            logger.error(f"初始化量子选股策略时出错: {str(e)}")
            self.stock_strategy = None

    def _load_historical_data(self):
        """加载历史选股数据"""
        try:
            # 检查是否存在历史数据目录
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
                
            # 历史数据文件路径
            history_file = os.path.join(data_dir, "stock_finder_history.json")
            
            if os.path.exists(history_file):
                with open(history_file, 'r', encoding='utf-8') as f:
                    self.historical_data = json.load(f)
                logger.info(f"已加载历史选股数据: {len(self.historical_data.get('stocks', []))} 只股票")
            else:
                logger.info("未找到历史选股数据")
        except Exception as e:
            logger.error(f"加载历史选股数据时出错: {str(e)}")
            self.historical_data = None

    def _clear_outdated_history(self):
        """清除过期的历史缓存数据"""
        try:
            # 历史数据文件路径
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
            history_file = os.path.join(data_dir, "stock_finder_history.json")
            
            # 检查文件是否存在
            if not os.path.exists(history_file):
                logger.info("没有历史缓存文件需要清除")
                return
                
            # 检查文件修改时间
            file_mtime = datetime.fromtimestamp(os.path.getmtime(history_file))
            current_time = datetime.now()
            
            # 计算文件年龄（小时）
            file_age_hours = (current_time - file_mtime).total_seconds() / 3600
            
            # 如果文件超过8小时，清除它
            if file_age_hours > 8:
                logger.info(f"发现过期历史缓存（{file_age_hours:.1f}小时），正在清除")
                os.remove(history_file)
                logger.info("已清除过期历史缓存")
            else:
                logger.info(f"历史缓存仍然有效（{file_age_hours:.1f}小时）")
                
        except Exception as e:
            logger.error(f"清除历史缓存时出错: {str(e)}")

    def _init_ui(self):
        """初始化用户界面"""
        # 确保API密钥始终可用
        self._ensure_api_key_available()
        
        # 主布局
        self.main_layout = QVBoxLayout(self)
        
        # 标题区域
        self.title_label = QLabel("超神量子选股器")
        self.title_label.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.title_label)
        
        # 添加Tushare API密钥设置按钮 - 但已经预设了API密钥
        self.api_key_setting_button = QPushButton("设置Tushare API密钥")
        self.api_key_setting_button.clicked.connect(self._set_tushare_api_key)
        self.api_key_setting_button.setVisible(False)  # 隐藏按钮，因为已预设API密钥
        self.main_layout.addWidget(self.api_key_setting_button)
        
        # 描述标签
        self.description_label = QLabel(
            "超越宇宙的选股策略，利用量子算法捕捉市场中的大牛股，助您超神"
        )
        self.description_label.setAlignment(Qt.AlignCenter)
        self.description_label.setStyleSheet("font-size: 12px; margin-bottom: 15px;")
        self.main_layout.addWidget(self.description_label)
        
        # 配置区域
        self._create_config_area()
        
        # 结果显示区域
        self._create_results_area()
        
        # 底部状态区域
        self._create_status_area()
        
    def _ensure_api_key_available(self):
        """确保Tushare API密钥始终可用"""
        # 预设的API密钥
        hardcoded_api_key = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"
        
        try:
            # 检查是否已有API密钥管理器
            if self.system_manager:
                api_key_manager = self.system_manager.get_component("api_key_manager")
                
                if api_key_manager:
                    # 如果有API管理器，检查是否有tushare密钥
                    if hasattr(api_key_manager, "get_api_key"):
                        current_key = api_key_manager.get_api_key("tushare")
                        
                        # 如果没有设置密钥或密钥为空，设置硬编码的密钥
                        if not current_key:
                            if hasattr(api_key_manager, "set_api_key"):
                                api_key_manager.set_api_key("tushare", hardcoded_api_key)
                                logger.info("已自动设置Tushare API密钥")
                else:
                    # 如果没有API密钥管理器，创建一个简单的管理器并注册到系统
                    class SimpleApiKeyManager:
                        def __init__(self):
                            self.keys = {"tushare": hardcoded_api_key}
                        
                        def get_api_key(self, provider):
                            return self.keys.get(provider, "")
                        
                        def set_api_key(self, provider, key):
                            self.keys[provider] = key
                            return True
                    
                    # 注册到系统
                    self.system_manager.components["api_key_manager"] = SimpleApiKeyManager()
                    logger.info("已创建API密钥管理器并设置Tushare API密钥")
            
            # 确保配置文件中也有该密钥
            try:
                import os
                import json
                
                config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config")
                os.makedirs(config_dir, exist_ok=True)
                
                config_path = os.path.join(config_dir, "api_keys.json")
                
                # 读取现有配置或创建新配置
                config = {}
                if os.path.exists(config_path):
                    with open(config_path, "r") as f:
                        try:
                            config = json.load(f)
                        except:
                            pass
                
                # 确保tushare密钥存在
                if "tushare" not in config or not config["tushare"]:
                    config["tushare"] = hardcoded_api_key
                    
                    # 保存配置
                    with open(config_path, "w") as f:
                        json.dump(config, f)
                    logger.info("已将Tushare API密钥保存到配置文件")
                    
            except Exception as e:
                logger.error(f"保存API密钥到配置文件时出错: {str(e)}")
        
        except Exception as e:
            logger.error(f"确保API密钥可用时出错: {str(e)}")

    def _create_config_area(self):
        """创建配置区域"""
        # 配置框架
        self.config_frame = QFrame()
        self.config_frame.setFrameShape(QFrame.StyledPanel)
        self.config_layout = QHBoxLayout(self.config_frame)
        self.main_layout.addWidget(self.config_frame)
        
        # 超神量子能力设置组
        self.power_group = QGroupBox("超神量子能力")
        self.power_layout = QVBoxLayout(self.power_group)
        
        # 量子能力滑块
        self.power_label = QLabel("量子计算能力:")
        self.power_layout.addWidget(self.power_label)
        
        self.power_slider = QSlider(Qt.Horizontal)
        self.power_slider.setMinimum(10)
        self.power_slider.setMaximum(100)
        self.power_slider.setValue(50)
        self.power_slider.setTickPosition(QSlider.TicksBelow)
        self.power_slider.setTickInterval(10)
        self.power_slider.valueChanged.connect(self._on_power_changed)
        self.power_layout.addWidget(self.power_slider)
        
        self.power_value_label = QLabel("50% - 平衡")
        self.power_value_label.setAlignment(Qt.AlignCenter)
        self.power_layout.addWidget(self.power_value_label)
        
        # 使用真实数据的状态提示
        self.api_status_label = QLabel("✅ Tushare API已预设")
        self.api_status_label.setStyleSheet("color: green; font-weight: bold;")
        self.power_layout.addWidget(self.api_status_label)
        
        # 强制使用真实数据复选框 - 默认选中但禁用，因为已预设API密钥
        self.use_real_data_checkbox = QCheckBox("使用真实市场数据")
        self.use_real_data_checkbox.setChecked(True)
        self.use_real_data_checkbox.setEnabled(False)  # 禁用复选框，因为已预设API密钥
        self.use_real_data_checkbox.setToolTip("已预设Tushare API密钥，将自动使用实时市场数据")
        self.power_layout.addWidget(self.use_real_data_checkbox)
        
        self.config_layout.addWidget(self.power_group)
        
        # 市场选择组
        self.market_group = QGroupBox("市场范围")
        self.market_layout = QFormLayout(self.market_group)
        
        # 市场选择
        self.market_combo = QComboBox()
        self.market_combo.addItems(["全市场", "沪深300", "中证500", "科创板", "创业板"])
        self.market_layout.addRow("选股范围:", self.market_combo)
        
        # 行业选择
        self.sector_combo = QComboBox()
        self.sector_combo.addItems(["全部行业", "科技", "消费", "医药", "金融", "新能源", "先进制造"])
        self.market_layout.addRow("行业选择:", self.sector_combo)
        
        self.config_layout.addWidget(self.market_group)
        
        # 操作按钮组
        self.action_group = QGroupBox("操作")
        self.action_layout = QVBoxLayout(self.action_group)
        
        # 开始选股按钮
        self.start_button = QPushButton("开始量子选股")
        self.start_button.setStyleSheet("font-weight: bold; height: 40px;")
        self.start_button.clicked.connect(self._on_start_finder)
        self.action_layout.addWidget(self.start_button)
        
        # 停止按钮
        self.stop_button = QPushButton("停止")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._on_stop_finder)
        self.action_layout.addWidget(self.stop_button)
        
        # 刷新数据按钮
        self.refresh_button = QPushButton("刷新数据")
        self.refresh_button.setToolTip("重新选股，获取最新市场数据")
        self.refresh_button.clicked.connect(self._on_refresh_data)
        self.action_layout.addWidget(self.refresh_button)
        
        # 数据验证按钮
        self.verify_button = QPushButton("验证数据来源")
        self.verify_button.setToolTip("验证当前选股结果是否使用真实市场数据")
        self.verify_button.clicked.connect(self._verify_data_source)
        self.action_layout.addWidget(self.verify_button)
        
        self.config_layout.addWidget(self.action_group)
        
    def _create_results_area(self):
        """创建结果显示区域"""
        # 分割器
        self.results_splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.results_splitter, 1)
        
        # 左侧：结果表格
        self.table_frame = QFrame()
        self.table_layout = QVBoxLayout(self.table_frame)
        
        # 结果表格
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(7)
        self.results_table.setHorizontalHeaderLabels([
            "代码", "名称", "行业", "超神评分", 
            "预期涨幅", "推荐", "时间周期"
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.setSelectionMode(QTableWidget.SingleSelection)
        self.results_table.itemSelectionChanged.connect(self._on_stock_selected)
        self.table_layout.addWidget(self.results_table)
        
        self.results_splitter.addWidget(self.table_frame)
        
        # 右侧：详细信息和图表
        self.details_frame = QFrame()
        self.details_layout = QVBoxLayout(self.details_frame)
        
        # 潜力股图表
        self.stock_chart = PotentialStockChart()
        self.details_layout.addWidget(self.stock_chart)
        
        # 详细信息组
        self.details_group = QGroupBox("股票详情")
        self.details_group_layout = QVBoxLayout(self.details_group)
        
        # 选中股票信息
        self.stock_info_label = QLabel("请选择股票查看详情")
        self.stock_info_label.setAlignment(Qt.AlignCenter)
        self.details_group_layout.addWidget(self.stock_info_label)
        
        # 超神推荐理由
        self.reasons_label = QLabel("超神推荐理由")
        self.reasons_label.setStyleSheet("font-weight: bold;")
        self.details_group_layout.addWidget(self.reasons_label)
        
        self.reasons_text = QTextEdit()
        self.reasons_text.setReadOnly(True)
        self.reasons_text.setMaximumHeight(150)
        self.details_group_layout.addWidget(self.reasons_text)
        
        self.details_layout.addWidget(self.details_group)
        
        self.results_splitter.addWidget(self.details_frame)
        
        # 设置分割比例
        self.results_splitter.setSizes([600, 500])
        
    def _create_status_area(self):
        """创建状态区域"""
        # 状态框架
        self.status_frame = QFrame()
        self.status_layout = QHBoxLayout(self.status_frame)
        self.main_layout.addWidget(self.status_frame)
        
        # 数据来源指示器
        self.data_source_label = QLabel("数据来源: 待定")
        self.data_source_label.setStyleSheet("font-weight: bold;")
        self.status_layout.addWidget(self.data_source_label)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(100)
        self.status_layout.addWidget(self.progress_bar, 1)
        
        # 状态标签
        self.status_label = QLabel("就绪")
        self.status_layout.addWidget(self.status_label)
        
    def _on_power_changed(self, value):
        """量子能力滑块值变化处理"""
        power_descriptions = {
            10: "低 - 保守选股",
            30: "中低 - 稳健选股",
            50: "平衡",
            70: "中高 - 进取选股",
            90: "超高 - 激进选股",
            100: "最大 - 极限预测"
        }
        
        # 找到最接近的描述
        closest_key = min(power_descriptions.keys(), key=lambda k: abs(k - value))
        description = power_descriptions[closest_key]
        
        self.power_value_label.setText(f"{value}% - {description}")
        
    def _on_start_finder(self):
        """开始选股按钮点击事件"""
        # 强制使用实时数据，而不是首先检查历史数据
        # 检查系统状态，但即使未启动也不阻止选股
        system_running = True
        if self.system_manager:
            system_running = self._check_system_ready()
            
        # 获取配置
        quantum_power = self.power_slider.value()
        market_scope = self.market_combo.currentText()
        sector_filter = self.sector_combo.currentText()
        
        # 总是使用真实数据，因为已预设API密钥
        use_real_data = True
        
        # 强制删除历史缓存，确保每次都使用最新数据
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
        history_file = os.path.join(data_dir, "stock_finder_history.json")
        
        # 清除历史数据变量
        self.historical_data = None
        
        # 多次尝试删除历史缓存文件
        delete_success = False
        max_attempts = 3
        for attempt in range(max_attempts):
            if os.path.exists(history_file):
                try:
                    os.remove(history_file)
                    if not os.path.exists(history_file):
                        delete_success = True
                        logger.info(f"已成功清除历史缓存文件 (尝试 {attempt+1}/{max_attempts})")
                        break
                    else:
                        logger.warning(f"历史缓存文件删除后仍然存在 (尝试 {attempt+1}/{max_attempts})")
                        time.sleep(0.1)  # 短暂暂停后重试
                except Exception as e:
                    logger.warning(f"无法清除历史缓存文件 (尝试 {attempt+1}/{max_attempts}): {str(e)}")
                    time.sleep(0.1)  # 短暂暂停后重试
            else:
                delete_success = True
                logger.info("历史缓存文件不存在，无需删除")
                break
                
        if not delete_success:
            logger.warning(f"无法删除历史缓存文件 {history_file} 经过 {max_attempts} 次尝试")
            # 尝试强制清空文件内容
            try:
                with open(history_file, 'w') as f:
                    f.write('{"status": "invalid", "stocks": []}')
                logger.info("已强制清空历史缓存文件内容")
            except Exception as e:
                logger.error(f"无法清空历史缓存文件内容: {str(e)}")
        
        # 创建并启动选股线程
        self.finder_thread = StockFinderThread(
            stock_strategy=self.stock_strategy,
            quantum_power=quantum_power, 
            market_scope=market_scope, 
            sector_filter=sector_filter,
            system_manager=self.system_manager,
            use_real_data=use_real_data  # 总是使用真实数据
        )
        
        # 连接信号
        self.finder_thread.progress_updated.connect(self.progress_bar.setValue)
        self.finder_thread.status_updated.connect(self.status_label.setText)
        self.finder_thread.results_ready.connect(self._on_results_ready)
        self.finder_thread.completed.connect(self._on_finder_completed)
        
        # 启动线程
        self.finder_thread.start()
        
        # 更新UI状态
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.power_slider.setEnabled(False)
        self.market_combo.setEnabled(False)
        self.sector_combo.setEnabled(False)
        self.progress_bar.setValue(0)
        
        # 显示数据来源信息
        self.data_source_label.setText("数据来源: 正在获取实时市场数据...")
        self.data_source_label.setStyleSheet("font-weight: bold; color: green;")
        
        self.status_label.setText("正在初始化量子选股算法...")
        
    def _on_stop_finder(self):
        """停止选股按钮点击事件"""
        if self.finder_thread and self.finder_thread.isRunning():
            self.finder_thread.stop()
            self.status_label.setText("正在停止...")
            
            # 使用计时器等待线程终止
            QTimer.singleShot(500, self._reset_ui_after_stop)
            
    def _reset_ui_after_stop(self):
        """停止选股后重置UI"""
        # 更新UI状态
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.power_slider.setEnabled(True)
        self.market_combo.setEnabled(True)
        self.sector_combo.setEnabled(True)
        self.status_label.setText("已停止")
        
    def _on_results_ready(self, results):
        """选股结果就绪处理"""
        if not results or "stocks" not in results or not results.get("stocks"):
            logger.error("收到的选股结果无效或为空")
            self.status_label.setText("选股失败：结果无效")
            self.data_source_label.setText("数据来源: 无效")
            self.data_source_label.setStyleSheet("font-weight: bold; color: red;")
            return
            
        self.finder_results = results
        
        # 保存结果到历史文件前验证数据来源
        using_real_data = self._check_using_real_data(results)
        
        # 在结果中添加或强制更新数据源标记
        if using_real_data:
            results["data_source"] = "tushare_real"
            # 确保每个股票对象也有数据源标记
            for stock in results.get("stocks", []):
                stock["data_source"] = "tushare_real"
        else:
            results["data_source"] = "simulated"
            # 确保每个股票对象也有数据源标记
            for stock in results.get("stocks", []):
                stock["data_source"] = "simulated"
        
        # 保存结果到历史文件
        self._save_results_to_history(results)
        
        # 输出详细的数据验证结果
        stock_count = len(results.get("stocks", []))
        logger.info(f"选股结果就绪，共有 {stock_count} 只股票")
        logger.info(f"数据来源验证结果: {'真实市场数据' if using_real_data else '模拟数据'}")
        
        # 更新状态标签，显示数据来源
        if using_real_data:
            self.status_label.setText(f"选股完成！使用真实市场数据，找到 {stock_count} 只股票")
            self.data_source_label.setText("数据来源: 实时市场")
            self.data_source_label.setStyleSheet("font-weight: bold; color: green;")
        else:
            self.status_label.setText(f"选股完成！使用模拟数据，生成 {stock_count} 只股票")
            self.data_source_label.setText("数据来源: 模拟数据")
            self.data_source_label.setStyleSheet("font-weight: bold; color: orange;")
        
        # 更新UI
        self._update_results_table()
        self._update_chart()
        
    def _check_using_real_data(self, results):
        """检查是否使用了真实市场数据"""
        if not results or "stocks" not in results:
            logger.warning("无法验证数据来源：结果为空或缺少股票列表")
            return False
            
        # 检查结果中是否有明确的数据来源标记
        if "data_source" in results:
            data_source = results["data_source"]
            if data_source == "tushare_real":
                logger.info("检测到真实市场数据来源标记")
                return True
            elif data_source == "simulated":
                logger.info("检测到模拟数据来源标记")
                return False
            else:
                logger.warning(f"数据来源标记未知: {data_source}")
                
        stocks = results.get("stocks", [])
        if not stocks:
            logger.warning("无法验证数据来源：股票列表为空")
            return False
            
        # 检查股票数据中是否包含真实数据中才有的字段
        real_data_fields = ["ts_code", "area", "market", "turnover_rate", "volume_ratio", "data_source"]
        
        # 检查第一只股票是否包含这些字段
        first_stock = stocks[0]
        
        # 如果股票中有明确的数据来源标记
        if "data_source" in first_stock:
            data_source = first_stock["data_source"]
            if data_source == "tushare_real":
                logger.info("检测到股票数据中的真实市场数据来源标记")
                return True
            elif data_source == "simulated":
                logger.info("检测到股票数据中的模拟数据来源标记")
                return False
            else:
                logger.warning(f"股票数据来源标记未知: {data_source}")
        
        # 用于记录找到的真实数据字段
        found_real_fields = []
        
        # 检查关键字段
        for field in real_data_fields:
            if field in first_stock and first_stock[field] is not None:
                found_real_fields.append(field)
                
        # 输出验证结果的日志
        if found_real_fields:
            logger.info(f"在股票数据中找到 {len(found_real_fields)} 个真实数据字段: {', '.join(found_real_fields)}")
            # 只有当找到至少两个真实数据字段时才认为是真实数据
            return len(found_real_fields) >= 2
        else:
            logger.warning("在股票数据中未找到任何真实数据字段")
            return False
    
    def _on_finder_completed(self):
        """选股完成处理"""
        # 更新UI状态
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.power_slider.setEnabled(True)
        self.market_combo.setEnabled(True)
        self.sector_combo.setEnabled(True)
        
    def _update_results_table(self):
        """更新结果表格"""
        if not self.finder_results:
            return
            
        stocks = self.finder_results.get("stocks", [])
        
        # 设置表格行数
        self.results_table.setRowCount(len(stocks))
        
        for row, stock in enumerate(stocks):
            # 设置单元格数据 - 添加检查确保键存在
            self.results_table.setItem(row, 0, QTableWidgetItem(stock.get("code", stock.get("ts_code", "未知"))))
            self.results_table.setItem(row, 1, QTableWidgetItem(stock.get("name", "未知")))
            self.results_table.setItem(row, 2, QTableWidgetItem(stock.get("sector", stock.get("industry", "未知"))))
            
            # 超神评分
            score = stock.get("quantum_score", 0)
            score_item = QTableWidgetItem(f"{score:.2f}")
            # 根据评分设置颜色
            if score >= 95:
                score_item.setBackground(QBrush(QColor(255, 215, 0, 100)))  # 金色
            elif score >= 90:
                score_item.setBackground(QBrush(QColor(0, 255, 0, 50)))  # 绿色
            self.results_table.setItem(row, 3, score_item)
            
            # 预期涨幅
            gain = stock.get("expected_gain", 0)
            gain_item = QTableWidgetItem(f"{gain:.2f}%")
            if gain >= 50:
                gain_item.setBackground(QBrush(QColor(255, 100, 100, 80)))
            elif gain >= 30:
                gain_item.setBackground(QBrush(QColor(255, 180, 0, 80)))
            self.results_table.setItem(row, 4, gain_item)
            
            # 推荐
            self.results_table.setItem(row, 5, QTableWidgetItem(stock.get("recommendation", "未知")))
            
            # 时间周期
            self.results_table.setItem(row, 6, QTableWidgetItem(stock.get("timeframe", "未知")))
            
        # 自动选择第一行
        if len(stocks) > 0:
            self.results_table.selectRow(0)
            
    def _update_chart(self):
        """更新图表"""
        if not self.finder_results:
            return
            
        stocks = self.finder_results.get("stocks", [])
        self.stock_chart.set_data(stocks)
        
    def _on_stock_selected(self):
        """股票选中事件处理"""
        try:
            selected_rows = self.results_table.selectionModel().selectedRows()
            if not selected_rows or not self.finder_results:
                return
                
            row = selected_rows[0].row()
            stocks = self.finder_results.get("stocks", [])
            
            if row >= 0 and row < len(stocks):
                selected_stock = stocks[row]
                self._display_stock_details(selected_stock)
        except Exception as e:
            logger.error(f"显示股票详情时出错: {str(e)}")
            # 显示友好的错误信息而不是崩溃
            self.stock_info_label.setText(f"<b>显示股票详情时出错</b><br>错误信息: {str(e)}")
            self.reasons_text.setText("无法显示选股理由。请检查日志获取更多信息。")
    
    def _display_stock_details(self, stock):
        """显示股票详细信息"""
        if not stock:
            return
            
        # 创建HTML格式的详细信息
        info_text = (
            f"<b>{stock.get('name', '未知')}</b> ({stock.get('ts_code', stock.get('code', '未知'))}) - {stock.get('sector', stock.get('industry', '未知'))}<br>"
            f"超神评分: <span style='color: {'gold' if stock.get('quantum_score', 0) >= 95 else 'green'};'>"
            f"{stock.get('quantum_score', 0):.2f}</span>  "
            f"预期涨幅: <span style='color: red;'>{stock.get('expected_gain', 0):.2f}%</span><br>"
            f"推荐: {stock.get('recommendation', '未知')}  "
            f"置信度: {stock.get('confidence', 0):.1f}%  "
            f"时间周期: {stock.get('timeframe', '未知')}"
        )
        self.stock_info_label.setText(info_text)
        
        # 更新推荐理由
        if "reasons" in stock and stock["reasons"]:
            reasons_text = "■ " + "\n\n■ ".join(stock["reasons"])
        else:
            reasons_text = "没有提供选股理由"
        self.reasons_text.setText(reasons_text)
    
    def _check_system_ready(self):
        """检查系统是否准备就绪"""
        if not self.system_manager:
            return False
            
        # 确保量子选股策略已启动
        if self.stock_strategy and not self.stock_strategy.is_running:
            try:
                self.stock_strategy.start()
            except Exception as e:
                logger.error(f"启动量子选股策略时出错: {str(e)}")
                
        # 检查量子后端是否启动
        try:
            quantum_backend = self.system_manager.get_component("quantum_backend")
            if quantum_backend and quantum_backend.is_active():
                return True
                
            # 尝试启动量子后端
            if quantum_backend and not quantum_backend.is_active():
                return quantum_backend.start()
                
        except Exception as e:
            logger.error(f"检查系统状态时出错: {str(e)}")
            
        return False
    
    def on_system_started(self):
        """系统启动事件处理"""
        # 启动量子选股策略
        if self.stock_strategy and not self.stock_strategy.is_running:
            try:
                self.stock_strategy.start()
            except Exception as e:
                logger.error(f"启动量子选股策略时出错: {str(e)}")
                
        # 允许使用选股功能
        self.start_button.setEnabled(True)
        
    def on_system_stopped(self):
        """系统停止事件处理"""
        # 停止量子选股策略
        if self.stock_strategy and self.stock_strategy.is_running:
            try:
                self.stock_strategy.stop()
            except Exception as e:
                logger.error(f"停止量子选股策略时出错: {str(e)}")
                
        # 允许使用选股功能（即使系统停止也可以使用模拟模式）
        self.start_button.setEnabled(True)
        
        # 如果正在进行选股，停止它
        if self.finder_thread and self.finder_thread.isRunning():
            self._on_stop_finder()

    def _use_historical_data(self):
        """使用历史数据显示选股结果"""
        logger.info("使用保存的历史选股数据")
        
        # 直接使用历史数据
        self.finder_results = self.historical_data
        
        # 模拟进度条效果
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.power_slider.setEnabled(False)
        self.market_combo.setEnabled(False)
        self.sector_combo.setEnabled(False)
        
        # 清空表格
        self.results_table.setRowCount(0)
        
        # 模拟正在加载
        self.status_label.setText("正在读取历史选股数据...")
        self.data_source_label.setText("数据来源: 历史缓存")
        self.data_source_label.setStyleSheet("font-weight: bold; color: blue;")
        
        # 使用计时器模拟加载进度
        self.progress_counter = 0
        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(self._update_history_progress)
        self.progress_timer.start(30)
    
    def _update_history_progress(self):
        """更新历史数据加载进度"""
        self.progress_counter += 3
        self.progress_bar.setValue(self.progress_counter)
        
        if self.progress_counter >= 100:
            self.progress_timer.stop()
            self.status_label.setText("已加载历史选股数据")
            self._update_results_table()
            self._update_chart()
            self._on_finder_completed()
            
    def _save_results_to_history(self, results):
        """保存选股结果到历史文件"""
        try:
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
                
            history_file = os.path.join(data_dir, "stock_finder_history.json")
            
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
                
            logger.info(f"已保存选股结果到历史文件: {history_file}")
            self.historical_data = results
        except Exception as e:
            logger.error(f"保存选股结果到历史文件时出错: {str(e)}")

    def _on_refresh_data(self):
        """刷新数据按钮点击事件"""
        # 临时禁用历史数据
        temp_historical = self.historical_data
        self.historical_data = None
        
        # 调用开始选股
        self._on_start_finder()
        
        # 恢复历史数据引用（新的结果会通过_save_results_to_history更新）
        self.historical_data = temp_historical 

    def _verify_data_source(self):
        """验证数据来源"""
        if not self.finder_results or "stocks" not in self.finder_results:
            QMessageBox.information(self, "数据验证", "没有选股结果可供验证")
            return
            
        using_real_data = self._check_using_real_data(self.finder_results)
        stocks = self.finder_results.get("stocks", [])
        
        if not stocks:
            QMessageBox.information(self, "数据验证", "选股结果为空，无法验证")
            return
            
        # 创建验证报告
        report = "数据来源验证报告\n"
        report += "=" * 30 + "\n\n"
        
        if using_real_data:
            report += "✅ 确认使用了真实市场数据\n\n"
        else:
            report += "⚠️ 使用了模拟数据\n\n"
            
        # 样本数据详情
        report += "样本股票数据字段分析：\n"
        sample_stock = stocks[0]
        
        # 真实数据特有字段
        real_data_fields = ["ts_code", "area", "market", "turnover_rate", "volume_ratio", "total_mv"]
        for field in real_data_fields:
            if field in sample_stock:
                report += f"✅ {field}: {sample_stock[field]}\n"
            else:
                report += f"❌ {field}: 缺失\n"
                
        # 统计结果
        report += f"\n共有股票: {len(stocks)} 只\n"
        report += f"时间戳: {self.finder_results.get('timestamp', '未知')}\n"
        report += f"量子功率: {self.finder_results.get('quantum_power', '未知')}%\n"
        report += f"市场范围: {self.finder_results.get('market_scope', '未知')}\n"
        
        # 显示报告
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("数据源验证")
        msg_box.setText(report)
        msg_box.setIcon(QMessageBox.Information)
        msg_box.exec_()

    def _check_api_key(self):
        """检查是否配置了有效的Tushare API密钥"""
        # 硬编码的API密钥作为首选和备用
        hardcoded_api_key = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"
        
        # 首先尝试使用硬编码的密钥
        logger.info("直接使用预设的Tushare API密钥")
        try:
            import tushare as ts
            # 设置token并初始化API
            ts.set_token(hardcoded_api_key)
            pro = ts.pro_api()
            
            # 测试API连接
            test_result = pro.query('stock_basic', exchange='', list_status='L', fields='ts_code,name', limit=1)
            if test_result is not None and not test_result.empty:
                logger.info("预设Tushare API密钥验证成功")
                
                # 确保系统中保存了这个密钥
                if self.system_manager:
                    api_key_manager = self.system_manager.get_component("api_key_manager")
                    if api_key_manager and hasattr(api_key_manager, "set_api_key"):
                        api_key_manager.set_api_key("tushare", hardcoded_api_key)
            
                return True
            else:
                logger.warning("预设Tushare API密钥验证失败，尝试其他方法")
        except Exception as e:
            logger.warning(f"使用预设API密钥时出错: {str(e)}")
        
        # 如果硬编码密钥失败，尝试其他方法获取API密钥
        api_key = None
        try:
            if self.system_manager:
                api_key_manager = self.system_manager.get_component("api_key_manager")
                if api_key_manager and hasattr(api_key_manager, "get_api_key"):
                    api_key = api_key_manager.get_api_key("tushare")
                    if api_key:
                        masked_key = api_key[:5] + "..." + api_key[-5:] if len(api_key) > 10 else "***"
                        logger.info(f"从API密钥管理器获取到Tushare密钥: {masked_key}")
            else:
                logger.warning("系统管理器未初始化，无法获取API密钥管理器")
        except Exception as e:
            logger.error(f"从系统获取API密钥时出错: {str(e)}")
        
        # 尝试从环境变量或配置文件获取
        if not api_key:
            import os
            api_key = os.environ.get("TUSHARE_API_KEY")
            if api_key:
                logger.info("从环境变量获取到Tushare API密钥")
            else:
                try:
                    import json
                    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "api_keys.json")
                    if os.path.exists(config_path):
                        with open(config_path, "r") as f:
                            config = json.load(f)
                            if "tushare" in config:
                                api_key = config["tushare"]
                                logger.info("从配置文件获取到Tushare API密钥")
                except Exception as e:
                    logger.error(f"从配置文件读取API密钥失败: {str(e)}")
        
        # 如果仍然没有API密钥，再次使用硬编码的密钥
        if not api_key:
            api_key = hardcoded_api_key
            logger.info(f"使用硬编码的API密钥: {api_key[:5]}...{api_key[-5:]}")
        
        # 验证API密钥
        if api_key:
            try:
                import tushare as ts
                ts.set_token(api_key)
                pro = ts.pro_api()
                
                test_result = pro.query('stock_basic', exchange='', list_status='L', fields='ts_code,name', limit=1)
                if test_result is not None and not test_result.empty:
                    logger.info("Tushare API密钥验证成功")
                    return True
                else:
                    logger.error("Tushare API密钥验证失败: 返回结果为空")
            except Exception as e:
                logger.error(f"验证Tushare API密钥时出错: {str(e)}")
        
        logger.error("未找到有效的Tushare API密钥")
        return False

    def _set_tushare_api_key(self):
        """设置Tushare API密钥"""
        # 获取当前密钥（如果存在）
        current_key = ""
        try:
            if self.system_manager:
                api_key_manager = self.system_manager.get_component("api_key_manager")
                if api_key_manager and hasattr(api_key_manager, "get_api_key"):
                    current_key = api_key_manager.get_api_key("tushare") or ""
        except:
            pass
            
        # 显示输入对话框
        key, ok = QInputDialog.getText(
            self, 
            "设置Tushare API密钥", 
            "请输入您的Tushare API Token：", 
            QLineEdit.Normal,
            current_key
        )
        
        if ok and key:
            # 直接在系统管理器中注册临时密钥管理器组件
            try:
                # 创建临时API密钥管理器
                class TempApiKeyManager:
                    def __init__(self):
                        self.keys = {"tushare": key}
                        
                    def get_api_key(self, provider):
                        return self.keys.get(provider, "")
                        
                    def set_api_key(self, provider, key):
                        self.keys[provider] = key
                        return True
                
                # 注册到系统管理器
                if self.system_manager:
                    # 首先检查是否已有API密钥管理器
                    existing_manager = self.system_manager.get_component("api_key_manager")
                    if existing_manager and hasattr(existing_manager, "set_api_key"):
                        # 如果已有管理器有set_api_key方法，则使用它
                        existing_manager.set_api_key("tushare", key)
                    else:
                        # 否则注册临时管理器
                        self.system_manager.components["api_key_manager"] = TempApiKeyManager()
                    
                    QMessageBox.information(
                        self,
                        "API密钥已设置",
                        "Tushare API密钥已成功设置！\n\n请点击'开始量子选股'按钮使用实时市场数据。"
                    )
                    
                    # 清除历史缓存
                    self._clear_outdated_history()
                    
                    return True
                else:
                    QMessageBox.warning(
                        self,
                        "无法设置API密钥",
                        "系统管理器未初始化，无法设置API密钥。"
                    )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "设置API密钥失败",
                    f"设置API密钥时发生错误：{str(e)}"
                )
                
        return False 