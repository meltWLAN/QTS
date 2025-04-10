#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - 数据控制器
管理数据获取和预处理，为GUI提供数据支持
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from PyQt5.QtCore import QObject, pyqtSignal, QTimer
from quantum_symbiotic_network.data_sources.tushare_data_source import TushareDataSource
import random  # Added missing import for random module

# 检查量子预测模块是否可用
PREDICTOR_AVAILABLE = True
try:
    from quantum_symbiotic_network.quantum_prediction import QuantumSymbioticPredictor
except ImportError:
    PREDICTOR_AVAILABLE = False
    logging.warning("量子预测模块导入失败，预测功能不可用")

# 导入数据源
try:
    from quantum_symbiotic_network.data_sources.tushare_data_source import TushareDataSource
    TUSHARE_AVAILABLE = True
except ImportError:
    TUSHARE_AVAILABLE = False
    logging.warning("无法导入TushareDataSource，将使用备用数据")

# 导入AKShare数据源（作为备用真实数据源）
try:
    from quantum_symbiotic_network.data_sources.akshare_data_source import AKShareDataSource
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    logging.warning("无法导入AKShareDataSource，可能是因为AKShare未安装")

# 导入其他可能的数据源（将来可扩展）
from quantum_symbiotic_network.data_sources.enhanced_data_source import EnhancedDataSource


class DataController(QObject):
    """数据控制器，管理市场数据和预测"""
    
    # 信号定义
    data_loaded_signal = pyqtSignal(dict)
    loading_progress_signal = pyqtSignal(int, str)
    error_signal = pyqtSignal(str)
    stock_data_updated_signal = pyqtSignal(str, dict)
    prediction_ready_signal = pyqtSignal(str, dict)
    market_insights_ready_signal = pyqtSignal(dict)
    market_status_updated_signal = pyqtSignal(dict)
    market_indices_updated_signal = pyqtSignal(list)
    recommended_stocks_updated_signal = pyqtSignal(list)
    data_updated_signal = pyqtSignal()
    
    def __init__(self, parent=None):
        """初始化数据控制器"""
        super().__init__(parent)
        self.logger = logging.getLogger("DataController")
        
        # Tushare Token - 使用固定的token
        self.tushare_token = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"
        
        # 初始化数据源
        self.tushare_source = None
        self.akshare_source = None
        self.unified_source = None
        
        # 标记数据源是否可用
        self.is_tushare_ready = False
        self.is_akshare_ready = False
        
        # 加载配置，可能会更新token
        self.config = self._load_config()
        
        # 股票数据缓存
        self.stocks_data = {}
        self.cache = {}
        
        # 预测器
        self.predictor = None
        self.initialize_predictor()  # 初始化预测器
        
        # 添加预测增强函数
        self.predictor_enhanced = None  # 将由量子意识控制器提供增强能力
        
        # 添加自动刷新定时器
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_market_data)
    
    def initialize_predictor(self):
        """初始化量子预测器"""
        try:
            if PREDICTOR_AVAILABLE:
                from quantum_symbiotic_network.quantum_prediction import get_predictor
                # 使用全局预测器实例并传入token
                self.predictor = get_predictor(self.tushare_token, force_new=True)
                self.logger.info("量子预测器初始化成功")
                
                # 设置超神级量子参数
                self.predictor.set_quantum_params(
                    coherence=0.95,
                    superposition=0.92,
                    entanglement=0.90
                )
                self.logger.info("量子预测参数已设置为超神级别")
            else:
                self.logger.warning("量子预测模块不可用，预测功能将受限")
        except Exception as e:
            self.logger.error(f"初始化量子预测器失败: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def _load_config(self):
        """加载配置"""
        try:
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    # 如果配置文件中有tushare_token，则使用配置文件中的token
                    if "tushare_token" in config and config["tushare_token"]:
                        self.tushare_token = config["tushare_token"]
                    return config
            return {}
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            return {}
    
    def initialize(self):
        """初始化数据控制器"""
        try:
            # 初始化状态
            self.initialization_success = False
            self.logger.info("开始初始化数据控制器...")
            
            # 检查是否需要加载超神分析结果
            try:
                self._check_supergod_results()
            except Exception as e:
                self.logger.error(f"检查超神分析结果时出错: {str(e)}")
            
            # 初始化统一数据源（整合多个真实数据源）
            try:
                from quantum_symbiotic_network.data_sources.unified_data_source import UnifiedDataSource
                
                # 创建统一数据源配置
                config = {
                    'tushare_token': self.tushare_token,
                    'use_akshare': True
                }
                
                self.logger.info("初始化统一数据源...")
                self.unified_source = UnifiedDataSource(config)
                
                # 检查统一数据源状态
                data_source_status = self.unified_source.get_data_source_status()
                if data_source_status.get('tushare', False):
                    self.is_tushare_ready = True
                    self.logger.info("TuShare数据源可用")
                
                if data_source_status.get('akshare', False):
                    self.is_akshare_ready = True
                    self.logger.info("AKShare数据源可用")
                
                if not any(data_source_status.values()):
                    self.logger.warning("所有数据源均不可用，系统功能将受限")
                    self.error_signal.emit("所有数据源均不可用，系统功能将受限")
                
            except ImportError:
                self.logger.warning("统一数据源未实现，将尝试使用独立数据源")
                # 初始化TuShare和AKShare数据源
                tushare_init_success = self._initialize_tushare()
                akshare_init_success = self._initialize_akshare()
                
                if not tushare_init_success and not akshare_init_success:
                    self.logger.error("所有数据源初始化失败！系统将有限运行")
                    self.error_signal.emit("所有数据源都未就绪，系统功能受限")
            except Exception as e:
                self.logger.error(f"初始化统一数据源失败: {str(e)}")
                # 尝试独立初始化数据源
                tushare_init_success = self._initialize_tushare()
                akshare_init_success = self._initialize_akshare()
            
            # 加载初始数据
            try:
                # 强制刷新一次数据
                self.refresh_market_data()
                
                # 启动自动刷新
                self.start_auto_refresh()
                
                # 设置定期更新缓存的定时器
                self.cache_update_timer = QTimer(self)
                self.cache_update_timer.timeout.connect(self.update_data_cache)
                # 每小时更新一次缓存
                self.cache_update_timer.start(60 * 60 * 1000)
                
                # 标记初始化成功
                self.initialization_success = True
                self.logger.info("数据控制器初始化成功")
                
                # 发出数据加载完成信号
                self.data_loaded_signal.emit({"success": True, "message": "数据加载成功"})
                return True
            except Exception as e:
                self.logger.error(f"加载初始数据失败: {str(e)}")
                self.error_signal.emit(f"加载初始数据失败: {str(e)}")
                
                # 即使初始加载失败，也标记为部分成功，系统可以继续运行
                self.initialization_success = False
                
                # 发出数据加载失败信号
                self.data_loaded_signal.emit({"success": False, "message": f"数据加载失败: {str(e)}"})
                return False
                
        except Exception as e:
            self.logger.error(f"数据控制器初始化失败: {str(e)}")
            self.error_signal.emit(f"数据控制器初始化失败: {str(e)}")
            self.initialization_success = False
            return False
    
    def _initialize_tushare(self):
        """初始化TuShare数据源"""
        if not TUSHARE_AVAILABLE:
            self.logger.error("无法导入TuShare库，TuShare初始化失败")
            return False
            
        try:
            from quantum_symbiotic_network.data_sources.tushare_data_source import TushareDataSource
            
            # 从配置中获取token
            tushare_pro_token = self.config.get('tushare_token', '')
            if not tushare_pro_token or len(tushare_pro_token) < 10:
                # 使用默认token
                tushare_pro_token = self.tushare_token
                
            self.logger.info(f"初始化TuShare Pro数据源，使用token: {tushare_pro_token[:8]}...")
            
            # 初始化TuShare数据源
                self.tushare_source = TushareDataSource(token=tushare_pro_token)
            
                # 检查API是否真的初始化成功
                self.is_tushare_ready = (self.tushare_source.api is not None)
                
                if self.is_tushare_ready:
                    self.logger.info("TuShare Pro API初始化成功")
                return True
                else:
                self.logger.warning("TuShare Pro API初始化失败，将尝试其他真实数据源")
                return False
                
            except Exception as e:
                self.logger.error(f"初始化TuShare Pro数据源失败: {str(e)}")
            return False
    
    def _initialize_akshare(self):
        """初始化AKShare数据源"""
        if not AKSHARE_AVAILABLE:
            self.logger.error("无法导入AKShare库，AKShare初始化失败")
            return False
            
        try:
            self.logger.info("初始化AKShare数据源...")
            
            # 初始化AKShare数据源
            self.akshare_source = AKShareDataSource()
            
            # 检查API是否初始化成功
            self.is_akshare_ready = self.akshare_source.is_ready
            
            if self.is_akshare_ready:
                self.logger.info("AKShare API初始化成功")
        return True
            else:
                self.logger.warning("AKShare API初始化失败")
                return False
                
        except Exception as e:
            self.logger.error(f"初始化AKShare数据源失败: {str(e)}")
            return False

    def update_data_cache(self):
        """更新数据缓存"""
        try:
            # 检查是否有可用的数据源
            data_source = self._get_available_data_source()
            if not data_source:
                self.logger.warning("没有可用的数据源，无法更新缓存")
                return
            
            # 获取市场状态
            self.get_market_status(force_refresh=True)
            
            # 获取指数数据
            self.get_indices_data(force_refresh=True)
            
            # 获取推荐股票
            self.get_recommended_stocks(force_refresh=True)
            
            self.logger.info("数据缓存更新完成")
            
        except Exception as e:
            self.logger.error(f"更新数据缓存失败: {str(e)}")
    
    def _get_available_data_source(self):
        """获取可用的数据源
        
        按优先级返回第一个可用的数据源:
        1. UnifiedDataSource (整合多个真实数据源)
        2. TuShare
        3. AKShare
        
        Returns:
            object: 数据源对象，如果没有可用的数据源则返回None
        """
        if hasattr(self, 'unified_source') and self.unified_source:
            return self.unified_source
        elif self.is_tushare_ready and self.tushare_source:
            return self.tushare_source
        elif self.is_akshare_ready and self.akshare_source:
            return self.akshare_source
        
        return None
    
    def _check_supergod_results(self):
        """检查是否有超神分析结果需要加载"""
        try:
            import os
            import json
            
            # 检查环境变量
            config_path = os.environ.get("SUPERGOD_CONFIG_PATH", "")
            if config_path and os.path.exists(config_path):
                self.logger.info(f"检测到超神配置文件: {config_path}")
                with open(config_path, "r", encoding="utf-8") as f:
                    supergod_config = json.load(f)
                    
                if supergod_config.get("load_results", False):
                    insights_file = supergod_config.get("insights_file", "")
                    predictions_file = supergod_config.get("predictions_file", "")
                    
                    if insights_file and os.path.exists(insights_file):
                        self.logger.info(f"加载超神洞察: {insights_file}")
                        with open(insights_file, "r", encoding="utf-8") as f:
                            self.supergod_insights = json.load(f)
                    
                    if predictions_file and os.path.exists(predictions_file):
                        self.logger.info(f"加载超神预测: {predictions_file}")
                        with open(predictions_file, "r", encoding="utf-8") as f:
                            self.supergod_predictions = json.load(f)
                    
                    self.has_supergod_results = True
                    self.logger.info("超神分析结果加载成功！")
                    
                    # 标记配置文件为已处理
                    try:
                        os.rename(config_path, config_path + ".processed")
                    except:
                        pass
            else:
                # 直接检查环境变量中的文件路径
                insights_file = os.environ.get("SUPERGOD_INSIGHTS_FILE", "")
                predictions_file = os.environ.get("SUPERGOD_PREDICTIONS_FILE", "")
                
                if insights_file and os.path.exists(insights_file) and predictions_file and os.path.exists(predictions_file):
                    self.logger.info(f"直接从环境变量加载超神分析结果")
                    
                    with open(insights_file, "r", encoding="utf-8") as f:
                        self.supergod_insights = json.load(f)
                        
                    with open(predictions_file, "r", encoding="utf-8") as f:
                        self.supergod_predictions = json.load(f)
                        
                    self.has_supergod_results = True
                    self.logger.info("超神分析结果加载成功！")
        except Exception as e:
            self.logger.error(f"检查超神分析结果时出错: {str(e)}")
            self.has_supergod_results = False
    
    def start_auto_refresh(self):
        """启动自动刷新"""
        # 每5分钟刷新一次市场数据
        self.refresh_timer.start(5 * 60 * 1000)
        self.logger.info("启动自动刷新，间隔：5分钟")
    
    def refresh_market_data(self):
        """刷新市场数据，包括市场状态、股指数据、推荐股票和热门股票"""
        self.logger.info("开始刷新市场数据...")
        self.logger.debug("获取市场状态")
        
        success = True
        offline_mode = False
            
            # 获取市场状态
        try:
            market_status = self.get_market_status()
            if market_status:
                if "离线模式" in market_status.get("status", ""):
                    offline_mode = True
                    self.logger.info("系统处于离线模式，将使用缓存数据")
                self.market_status = market_status
            self.market_status_updated_signal.emit(market_status)
                self.logger.debug(f"市场状态已更新: {market_status}")
            else:
                offline_mode = True
                self.logger.warning("获取市场状态失败，将使用本地缓存数据")
                success = False
        except Exception as e:
            offline_mode = True
            self.logger.error(f"获取市场状态时发生错误: {str(e)}")
            success = False
        
        # 刷新股指数据
        try:
            indices_data = self.get_indices_data()
            if indices_data:
                self.indices_data = indices_data
                self.market_indices_updated_signal.emit(indices_data)
                self.logger.debug(f"已获取 {len(indices_data)} 个股指数据")
            elif not offline_mode:
                self.logger.warning("无法获取股指数据")
                success = False
        except Exception as e:
            if not offline_mode:
                self.logger.error(f"获取股指数据时发生错误: {str(e)}")
            success = False
        
        # 刷新推荐股票
        try:
            recommended_stocks = self.get_recommended_stocks()
            if recommended_stocks:
                self.recommended_stocks = recommended_stocks
                self.recommended_stocks_updated_signal.emit(recommended_stocks)
                self.logger.debug(f"已获取 {len(recommended_stocks)} 个推荐股票")
            elif not offline_mode:
                self.logger.warning("无法获取推荐股票")
                success = False
        except Exception as e:
            if not offline_mode:
                self.logger.error(f"获取推荐股票时发生错误: {str(e)}")
            success = False
        
        # 如果是交易时间，刷新热门股票
        if not offline_mode and self.market_status and self.market_status.get("status") == "交易中":
            try:
                hot_stocks = self.get_hot_stocks()
                if hot_stocks:
                    self.hot_stocks = hot_stocks
                    self.hot_stocks_updated_signal.emit(hot_stocks)
                    self.logger.debug(f"已获取 {len(hot_stocks)} 个热门股票")
                else:
                    self.logger.warning("无法获取热门股票")
                    success = False
            except Exception as e:
                self.logger.error(f"获取热门股票时发生错误: {str(e)}")
                success = False
        
        # 检查超神预测结果
        self._check_supergod_results()
        
        # 发出数据更新完成信号
        self.data_updated_signal.emit(success)
        return success
    
    def _is_today_likely_trading_day(self):
        """判断今天是否可能为交易日（简单判断）"""
        # 获取今天的日期
        today = datetime.now()
        
        # 判断是否为周末
        if today.weekday() >= 5:  # 5是星期六，6是星期日
            return False
        
        # 中国常见节假日（简化判断）
        holidays = [
            # 元旦
            f"{today.year}-01-01",
            # 春节（农历，只能粗略估计）
            f"{today.year}-01-21", f"{today.year}-01-22", f"{today.year}-01-23", 
            f"{today.year}-01-24", f"{today.year}-01-25", f"{today.year}-01-26", f"{today.year}-01-27",
            # 清明节
            f"{today.year}-04-05",
            # 劳动节
            f"{today.year}-05-01", f"{today.year}-05-02", f"{today.year}-05-03",
            # 端午节
            f"{today.year}-06-22", 
            # 中秋节
            f"{today.year}-09-29",
            # 国庆节
            f"{today.year}-10-01", f"{today.year}-10-02", f"{today.year}-10-03", 
            f"{today.year}-10-04", f"{today.year}-10-05", f"{today.year}-10-06", f"{today.year}-10-07",
        ]
        
        # 检查是否为节假日
        if today.strftime("%Y-%m-%d") in holidays:
            return False
        
        # 检查时间是否在交易时间内
        now = today.time()
        morning_start = datetime.strptime('09:30', '%H:%M').time()
        morning_end = datetime.strptime('11:30', '%H:%M').time()
        afternoon_start = datetime.strptime('13:00', '%H:%M').time()
        afternoon_end = datetime.strptime('15:00', '%H:%M').time()
        
        if (morning_start <= now <= morning_end) or (afternoon_start <= now <= afternoon_end):
            return True
            
            return False
    
    def load_initial_data(self):
        """加载初始数据"""
        try:
            self.logger.info("开始加载初始数据...")
            
            # 连接数据源
            data_source = self._get_available_data_source()
            if not data_source:
                raise Exception("没有可用的数据源，无法加载初始数据")
            
            # 发送进度信号
            self.loading_progress_signal.emit(10, "连接数据源")
            
            # 获取市场状态
            self.logger.info("获取市场状态...")
            market_status = self.get_market_status()
            self.market_status_updated_signal.emit(market_status)
            
            # 发送进度信号
            self.loading_progress_signal.emit(25, "加载市场状态")
            
            # 获取指数数据
            self.logger.info("获取指数数据...")
            indices = self.get_indices_data()
            self.market_indices_updated_signal.emit(indices)
            
            # 发送进度信号
            self.loading_progress_signal.emit(50, "加载指数数据")
            
            # 获取推荐股票
            self.logger.info("获取推荐股票...")
            recommended_stocks = self.get_recommended_stocks()
            self.recommended_stocks_updated_signal.emit(recommended_stocks)
            
            # 发送进度信号
            self.loading_progress_signal.emit(75, "加载推荐股票")
            
            # 获取热门股票
            self.logger.info("获取热门股票...")
            hot_stocks = self.get_hot_stocks()
            
            # 发送进度信号
            self.loading_progress_signal.emit(100, "数据加载完成")
            
            # 准备数据包
            data_package = {
                "market_status": market_status,
                "indices": indices,
                "recommended_stocks": recommended_stocks,
                "hot_stocks": hot_stocks
            }
            
            # 发送数据加载完成信号
            self.data_loaded_signal.emit({"success": True, "data": data_package})
            
            self.logger.info("初始数据加载完成")
            
            return True
        
        except Exception as e:
            self.logger.error(f"加载初始数据失败: {str(e)}")
            self.error_signal.emit(f"加载初始数据失败: {str(e)}")
            
            # 发送数据加载失败信号
            self.data_loaded_signal.emit({"success": False, "message": str(e)})
            
            return False
    
    def get_market_status(self, force_refresh=False):
        """获取市场状态"""
        self.logger.debug("获取市场状态")
        
        try:
            # 获取可用的数据源
            data_source = self._get_available_data_source()
            
            if data_source:
                # 尝试获取市场状态
                try:
                    # 从数据源获取市场状态
                    market_status = data_source.get_market_status()
                    
                    # 添加数据源信息
                    if isinstance(data_source, TushareDataSource):
                        market_status["source"] = "tushare"
                    elif isinstance(data_source, AKShareDataSource):
                        market_status["source"] = "akshare"
            else:
                        market_status["source"] = "unknown"
                        
                    return market_status
                    
                except Exception as e:
                    self.logger.warning(f"获取市场状态失败: {str(e)}，将尝试使用基础状态")
            
            # 如果数据源不可用或获取失败，使用基础状态
            return {
                "status": "未知",
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "trading_day": self._is_today_likely_trading_day(),
                "next_trading_day": None,
                "last_trading_day": None,
                "source": "basic"
            }
            
        except Exception as e:
            self.logger.error(f"获取市场状态失败: {str(e)}")
            # 返回基本状态而不是抛出异常
            return {
                "status": "错误",
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "trading_day": self._is_today_likely_trading_day(),
                "next_trading_day": None,
                "last_trading_day": None,
                "source": "error",
                "error": str(e)
            }
    
    def get_indices_data(self, force_refresh=False):
        """获取股指数据"""
        try:
            # 获取可用的数据源
            data_source = self._get_available_data_source()
            
            if data_source and hasattr(data_source, 'get_index_data'):
                try:
                # 尝试获取实时指数数据
                    indices = data_source.get_index_data()
                    if indices and len(indices) > 0:
                        return indices
                except Exception as e:
                    self.logger.warning(f"获取股指数据失败: {str(e)}，将尝试其他数据源")
            
            # 尝试备用数据源
            if self.is_tushare_ready and data_source != self.tushare_source and hasattr(self.tushare_source, 'get_index_data'):
                try:
                indices = self.tushare_source.get_index_data()
                    if indices and len(indices) > 0:
                    return indices
                except Exception as e:
                    self.logger.warning(f"从TuShare获取股指数据失败: {str(e)}")
            
            # 尝试AKShare数据源
            if self.is_akshare_ready and data_source != self.akshare_source and hasattr(self.akshare_source, 'get_index_data'):
                try:
                    indices = self.akshare_source.get_index_data()
                    if indices and len(indices) > 0:
                        return indices
                except Exception as e:
                    self.logger.warning(f"从AKShare获取股指数据失败: {str(e)}")
            
            self.logger.error("无法获取股指数据：所有数据源均失败")
            return []
        
        except Exception as e:
            self.logger.error(f"获取股指数据失败: {str(e)}")
            return []
    
    def get_recommended_stocks(self, force_refresh=False):
        """获取推荐股票"""
        try:
            # 获取可用的数据源
            data_source = self._get_available_data_source()
            
            if data_source and hasattr(data_source, 'get_recommended_stocks'):
                try:
                    stocks = data_source.get_recommended_stocks()
                    if stocks:
                        return stocks
                except Exception as e:
                    self.logger.warning(f"获取推荐股票失败: {str(e)}，将尝试其他数据源")
            
            # 尝试备用数据源
            if self.is_tushare_ready and data_source != self.tushare_source and hasattr(self.tushare_source, 'get_recommended_stocks'):
                try:
                stocks = self.tushare_source.get_recommended_stocks()
                if stocks:
                    return stocks
                except Exception as e:
                    self.logger.warning(f"从TuShare获取推荐股票失败: {str(e)}")
            
            # 尝试AKShare数据源
            if self.is_akshare_ready and data_source != self.akshare_source and hasattr(self.akshare_source, 'get_recommended_stocks'):
                try:
                    stocks = self.akshare_source.get_recommended_stocks()
                    if stocks:
                        return stocks
                except Exception as e:
                    self.logger.warning(f"从AKShare获取推荐股票失败: {str(e)}")
            
            self.logger.error("无法获取推荐股票：所有数据源均失败")
            return []
        
        except Exception as e:
            self.logger.error(f"获取推荐股票失败: {str(e)}")
            return []
    
    def get_hot_stocks(self, count=5, force_refresh=False):
        """获取热门股票"""
        try:
            # 获取可用的数据源
            data_source = self._get_available_data_source()
            
            if data_source and hasattr(data_source, 'get_hot_stocks'):
                try:
                    stocks = data_source.get_hot_stocks(count)
                    if stocks:
                        return stocks
                except Exception as e:
                    self.logger.warning(f"获取热门股票失败: {str(e)}，将尝试其他数据源")
            
            # 尝试从推荐股票中获取热门股票
            recommended = self.get_recommended_stocks(force_refresh)
            if recommended:
                # 根据推荐度排序，取前count个
                sorted_stocks = sorted(recommended, key=lambda x: x.get('recommendation', 0), reverse=True)
                return sorted_stocks[:count]
            
            self.logger.error("无法获取热门股票：所有数据源均失败")
            return []
        
        except Exception as e:
            self.logger.error(f"获取热门股票失败: {str(e)}")
            return []
    
    def get_stock_data(self, code, force_refresh=False):
        """获取单个股票数据"""
        try:
            # 检查缓存
            if not force_refresh and code in self.stocks_data:
                return self.stocks_data[code]
            
            # 获取可用的数据源
            data_source = self._get_available_data_source()
            
            if data_source and hasattr(data_source, 'get_stock_data'):
                try:
                    stock_data = data_source.get_stock_data(code, force_refresh)
                if stock_data:
                    # 缓存数据
                    self.stocks_data[code] = stock_data
                    # 发出信号
                    self.stock_data_updated_signal.emit(code, stock_data)
                    return stock_data
                except Exception as e:
                    self.logger.warning(f"获取股票数据失败: {str(e)}，将尝试其他数据源")
            
            # 尝试备用数据源
            if self.is_tushare_ready and data_source != self.tushare_source and hasattr(self.tushare_source, 'get_stock_data'):
                try:
                    stock_data = self.tushare_source.get_stock_data(code, force_refresh)
                    if stock_data:
                        self.stocks_data[code] = stock_data
                        self.stock_data_updated_signal.emit(code, stock_data)
                        return stock_data
                except Exception as e:
                    self.logger.warning(f"从TuShare获取股票数据失败: {str(e)}")
            
            # 尝试AKShare数据源
            if self.is_akshare_ready and data_source != self.akshare_source and hasattr(self.akshare_source, 'get_stock_data'):
                try:
                    stock_data = self.akshare_source.get_stock_data(code, force_refresh)
                    if stock_data:
                        self.stocks_data[code] = stock_data
                        self.stock_data_updated_signal.emit(code, stock_data)
                        return stock_data
                except Exception as e:
                    self.logger.warning(f"从AKShare获取股票数据失败: {str(e)}")
            
            self.logger.error(f"无法获取股票 {code} 数据：所有数据源均失败")
            return {}
        
        except Exception as e:
            self.logger.error(f"获取股票 {code} 数据失败: {str(e)}")
            return {}
    
    def search_stocks(self, keyword):
        """搜索股票"""
        try:
            # 获取可用的数据源
            data_source = self._get_available_data_source()
            
            if data_source and hasattr(data_source, 'search_stocks'):
                try:
                    results = data_source.search_stocks(keyword)
                    if results:
                        return results
                except Exception as e:
                    self.logger.warning(f"搜索股票失败: {str(e)}，将尝试其他数据源")
            
            # 尝试备用数据源
            if self.is_tushare_ready and data_source != self.tushare_source and hasattr(self.tushare_source, 'search_stocks'):
                try:
                results = self.tushare_source.search_stocks(keyword)
                if results:
                    return results
                except Exception as e:
                    self.logger.warning(f"从TuShare搜索股票失败: {str(e)}")
            
            # 尝试AKShare数据源
            if self.is_akshare_ready and data_source != self.akshare_source and hasattr(self.akshare_source, 'search_stocks'):
                try:
                    results = self.akshare_source.search_stocks(keyword)
                    if results:
                        return results
                except Exception as e:
                    self.logger.warning(f"从AKShare搜索股票失败: {str(e)}")
            
            self.logger.error("无法搜索股票：所有数据源均失败")
            return []
        
        except Exception as e:
            self.logger.error(f"搜索股票失败: {str(e)}")
            return []
    
    def get_predictor(self):
        """获取预测器，如果未初始化则重新初始化
        
        Returns:
            object: 预测器实例
        """
        if self.predictor is None:
            self.initialize_predictor()
        
        if self.predictor is None:
            # 尝试再次初始化，使用最基本的方式
            try:
                from quantum_symbiotic_network.quantum_prediction import get_predictor
                self.predictor = get_predictor()
                self.logger.info("使用默认全局预测器作为备用")
            except Exception as e:
                self.logger.error(f"无法获取预测器: {str(e)}")
            try:
                from quantum_symbiotic_network.quantum_prediction import QuantumSymbioticPredictor
                self.predictor = QuantumSymbioticPredictor()
                self.logger.info("创建新预测器实例作为备用")
            except Exception as e:
                self.logger.error(f"无法创建预测器: {str(e)}")
    
        return self.predictor

    def predict_stock(self, stock_code, days=5, use_quantum=True, callback=None):
        """预测股票未来走势
        
        Args:
            stock_code: 股票代码
            days: 预测天数
            use_quantum: 是否使用量子预测
            callback: 预测完成后的回调函数
            
        Returns:
            dict: 预测结果
        """
        # 创建线程进行预测，避免UI卡顿
        def prediction_thread():
            try:
                self.logger.info(f"开始预测股票 {stock_code} 未来 {days} 天走势")
                result = {
                    'stock_code': stock_code,
                    'days': days,
                    'success': False,
                    'message': '',
                    'prediction': None,
                }
                
                # 格式化股票代码
                formatted_code = self._format_stock_code(stock_code)
                
                # 获取历史数据
                history_data = self._get_stock_data(formatted_code, 120)
                if history_data is None or history_data.empty:
                    error_msg = f"无法获取 {formatted_code} 的历史数据"
                    self.logger.error(error_msg)
                    result['message'] = error_msg
                    
                    if callback:
                        callback(result)
                    return result
                
                # 检查数据是否足够
                if len(history_data) < 30:
                    error_msg = f"{formatted_code} 的历史数据不足 (只有 {len(history_data)} 行)"
                    self.logger.warning(error_msg)
                    result['message'] = error_msg + "，但仍将尝试预测"
                
                # 提取最新价格和日期
                latest_date = history_data['trade_date'].iloc[-1]
                latest_price = history_data['close'].iloc[-1]
                self.logger.info(f"{formatted_code} 最新价格 (日期: {latest_date}): {latest_price}")
                
                # 初始化预测结果
                prediction_data = {
                    'dates': [],
                    'prices': [],
                    'lower_bounds': [],
                    'upper_bounds': [],
                    'indicators': {},
                    'insights': [],
                    'latest_price': latest_price,
                    'latest_date': latest_date,
                    'quantum_used': False,
                    'model_type': 'basic'
                }
                
                # 尝试使用量子预测
                quantum_prediction_succeeded = False
                if use_quantum:
                    try:
                        # 检查量子预测器是否可用
                        predictor = self.get_predictor()
                        if predictor:
                            self.logger.info(f"使用量子预测器预测 {formatted_code}")
                            
                            # 检查预测器类型并使用适当的方法
                            if hasattr(predictor, 'predict_stock'):
                                quantum_result = predictor.predict_stock(
                                    stock_code=formatted_code,
                                    history_data=history_data,
                                    days=days
                                )
                            else:
                                # 使用predict方法
                                quantum_result = predictor.predict(
                                    stock_code=formatted_code,
                                    stock_data=history_data,
                                    days=days
                                )
                            
                            if quantum_result and ('prices' in quantum_result or 'predicted_prices' in quantum_result):
                                # 标准化预测结果格式
                                if 'predicted_prices' in quantum_result and 'prices' not in quantum_result:
                                    quantum_result['prices'] = quantum_result['predicted_prices']
                                
                                # 确保结果中有价格列表并且非空
                                if 'prices' in quantum_result and len(quantum_result['prices']) > 0:
                                    prediction_data.update(quantum_result)
                                    prediction_data['quantum_used'] = True
                                    prediction_data['model_type'] = 'quantum'
                                    quantum_prediction_succeeded = True
                                    self.logger.info(f"量子预测 {formatted_code} 成功")
                                    
                                    # 尝试使用超神级增强
                                    try:
                                        from quantum_symbiotic_network.quantum_prediction import get_enhancer
                                        enhancer = get_enhancer(predictor)
                                        if enhancer:
                                            self.logger.info(f"使用超神级增强器进一步增强预测结果")
                                            enhanced_data = enhancer.enhance_prediction(prediction_data, stock_code=formatted_code)
                                            if enhanced_data:
                                                prediction_data = enhanced_data
                                                prediction_data['supergod_enhanced'] = True
                                                prediction_data['model_type'] = 'quantum_supergod'
                                                self.logger.info(f"超神级增强成功应用到预测结果")
                                    except Exception as e:
                                        self.logger.error(f"超神级增强失败: {str(e)}")
                                else:
                                    self.logger.warning(f"量子预测 {formatted_code} 返回无效结果格式")
                            else:
                                self.logger.warning(f"量子预测 {formatted_code} 返回空结果")
                        else:
                            self.logger.warning("量子预测器未初始化")
                    except Exception as e:
                        self.logger.error(f"量子预测失败: {str(e)}")
                
                # 如果量子预测失败，使用传统方法
                if not quantum_prediction_succeeded:
                    self.logger.info(f"使用传统方法预测 {formatted_code}")
                    
                    # 生成预测日期
                    last_date = datetime.strptime(latest_date, '%Y%m%d')
                    prediction_dates = []
                    for i in range(1, days + 1):
                        next_date = last_date + timedelta(days=i)
                        # 跳过周末
                        while next_date.weekday() >= 5:  # 5=周六, 6=周日
                            next_date += timedelta(days=1)
                        prediction_dates.append(next_date.strftime('%Y%m%d'))
                    
                    # 创建预测结果 - 使用备用预测方法
                    try:
                        # 尝试使用标准预测方法
                        prediction_prices = self._predict_with_standard_models(history_data, days)
                        prediction_data['model_type'] = 'standard'
                    except Exception as e:
                        self.logger.error(f"标准预测失败: {str(e)}")
                        # 如果失败，使用备用方法生成预测
                        prediction_prices = self._generate_backup_prediction(days, formatted_code)
                        prediction_data['model_type'] = 'backup'
                    
                    # 计算置信区间
                    volatility = history_data['close'].pct_change().std()
                    confidence_interval = volatility * 1.96  # 95% 置信区间
                    
                    lower_bounds = []
                    upper_bounds = []
                    for price in prediction_prices:
                        lower_bound = price * (1 - confidence_interval)
                        upper_bound = price * (1 + confidence_interval)
                        lower_bounds.append(round(lower_bound, 2))
                        upper_bounds.append(round(upper_bound, 2))
                    
                    # 更新预测数据
                    prediction_data['dates'] = prediction_dates
                    prediction_data['prices'] = [round(price, 2) for price in prediction_prices]
                    prediction_data['lower_bounds'] = lower_bounds
                    prediction_data['upper_bounds'] = upper_bounds
                    
                    # 计算简单技术指标
                    try:
                        # 将历史数据和预测数据合并
                        combined_prices = list(history_data['close'].values[-30:]) + prediction_prices
                        
                        # 计算10日和30日移动平均线
                        ma10 = [sum(combined_prices[i-10:i])/10 for i in range(10, len(combined_prices))][-days:]
                        if len(history_data) >= 30:
                            ma30 = [sum(combined_prices[i-30:i])/30 for i in range(30, len(combined_prices))][-days:]
                        else:
                            ma30 = [sum(combined_prices[i-min(30, len(combined_prices[:i])):i])/min(30, len(combined_prices[:i])) 
                                   for i in range(min(30, len(combined_prices)), len(combined_prices))][-days:]
                        
                        # 添加到预测数据
                        prediction_data['indicators']['ma10'] = [round(val, 2) for val in ma10]
                        prediction_data['indicators']['ma30'] = [round(val, 2) for val in ma30]
                        
                        # 添加简单市场洞察
                        insights = []
                        price_change = (prediction_prices[-1] / latest_price - 1) * 100
                        if price_change > 5:
                            insights.append(f"预测在{days}天内价格可能上涨{price_change:.2f}%，呈现强劲上升趋势")
                        elif price_change > 0:
                            insights.append(f"预测在{days}天内价格可能小幅上涨{price_change:.2f}%，呈现温和上升趋势")
                        elif price_change > -5:
                            insights.append(f"预测在{days}天内价格可能下跌{abs(price_change):.2f}%，呈现小幅下跌趋势")
                        else:
                            insights.append(f"预测在{days}天内价格可能下跌{abs(price_change):.2f}%，呈现明显下跌趋势")
                        
                        # 分析动量
                        if all(prediction_prices[i] > prediction_prices[i-1] for i in range(1, len(prediction_prices))):
                            insights.append("预测价格持续上涨，动量强劲")
                        elif all(prediction_prices[i] < prediction_prices[i-1] for i in range(1, len(prediction_prices))):
                            insights.append("预测价格持续下跌，动量减弱")
                        
                        # 分析MA交叉
                        if ma10[-1] > ma30[-1] and ma10[0] <= ma30[0]:
                            insights.append("预测期内可能出现黄金交叉信号(MA10上穿MA30)，可能是买入机会")
                        elif ma10[-1] < ma30[-1] and ma10[0] >= ma30[0]:
                            insights.append("预测期内可能出现死亡交叉信号(MA10下穿MA30)，可能是卖出机会")
                        
                        prediction_data['insights'] = insights
                    except Exception as e:
                        self.logger.error(f"计算技术指标时出错: {str(e)}")
                
                # 设置结果
                result['success'] = True
                result['prediction'] = prediction_data
                
                # 调用回调函数
                if callback:
                    callback(result)
                
                self.logger.info(f"预测 {formatted_code} 未来 {days} 天走势完成")
                return result
            
            except Exception as e:
                error_msg = f"预测股票 {stock_code} 时发生错误: {str(e)}"
                self.logger.error(error_msg)
                result = {
                    'stock_code': stock_code,
                    'days': days,
                    'success': False,
                    'message': error_msg,
                    'prediction': None
                }
                
                if callback:
                    callback(result)
                
                return result

        # 创建并启动预测线程
        from threading import Thread
        thread_result = [None]  # 用于存储线程结果
        
        def thread_wrapper():
            result = prediction_thread()
            thread_result[0] = result
        
        prediction_thread_obj = Thread(target=thread_wrapper)
        prediction_thread_obj.daemon = True
        prediction_thread_obj.start()
        
        # 如果没有提供回调函数，则需要等待线程完成并返回结果
        if callback is None:
            import time
            # 等待预测完成，但最多等待30秒
            for _ in range(300):  # 300 * 0.1s = 30s
                if not prediction_thread_obj.is_alive():
                    break
                time.sleep(0.1)
                
            # 如果线程仍在运行，提供一个基本的结果
            if prediction_thread_obj.is_alive():
                self.logger.warning(f"预测 {stock_code} 超时，返回基本结果")
                return {
                    'stock_code': stock_code,
                    'days': days,
                    'success': False,
                    'message': '预测超时，请稍后重试',
                    'prediction': None
                }
            
            # 返回线程结果  
            if thread_result[0]:
                return thread_result[0]
                
        # 由于线程异步运行，此函数无法直接返回预测结果
        # 在有回调函数的情况下，结果会通过回调函数返回
        return {
            'stock_code': stock_code,
            'days': days,
            'success': False,
            'message': '预测正在进行中',
            'prediction': None,
            'async': True
        }

    def _format_stock_code(self, code):
        """格式化股票代码
        
        Args:
            code: 原始股票代码
            
        Returns:
            str: 格式化后的股票代码
        """
        # 去除空格
        code = code.strip() if isinstance(code, str) else str(code)
        
        # 记录原始输入
        self.logger.debug(f"格式化股票代码: 输入 {code}")
        
        # 如果已经带有后缀，直接返回
        if code.endswith(('.SH', '.SZ', '.BJ')):
            self.logger.debug(f"代码 {code} 已有后缀，无需处理")
            return code
        
        # 标准化代码长度，确保是6位
        if len(code) != 6:
            self.logger.warning(f"股票代码 {code} 长度不是6位，可能不标准")
            # 尝试左侧补0至6位
            if len(code) < 6 and code.isdigit():
                code = code.zfill(6)
                self.logger.info(f"股票代码补零为: {code}")
        
        # 根据开头判断后缀
        formatted_code = code
        if code.startswith('6'):
            formatted_code = f"{code}.SH"
            self.logger.debug(f"代码 {code} 以6开头，判定为上交所股票: {formatted_code}")
        elif code.startswith('0'):
            formatted_code = f"{code}.SZ"
            self.logger.debug(f"代码 {code} 以0开头，判定为深交所主板股票: {formatted_code}")
        elif code.startswith('3'):
            formatted_code = f"{code}.SZ"
            self.logger.debug(f"代码 {code} 以3开头，判定为创业板股票: {formatted_code}")
        elif code.startswith(('4', '8')):
            formatted_code = f"{code}.BJ"
            self.logger.debug(f"代码 {code} 以4或8开头，判定为北交所股票: {formatted_code}")
        else:
            self.logger.warning(f"无法判断股票代码 {code} 的交易所，保持原样")
        
        # 记录处理结果
        self.logger.info(f"股票代码格式化: {code} -> {formatted_code}")
        return formatted_code

    def _predict_with_standard_models(self, history_data, days=5):
        """使用标准预测模型预测股票价格
        
        Args:
            history_data: 历史数据（DataFrame格式）
            days: 预测天数
            
        Returns:
            list: 预测价格列表
        """
        try:
            if history_data is None or history_data.empty or 'close' not in history_data.columns:
                raise ValueError("历史数据为空或格式不正确")
                
            # 获取收盘价
            prices = history_data['close'].values
            
            # 1. 使用简单移动平均模型
            ma_window = min(5, len(prices))
            avg_change = np.mean(np.diff(prices[-ma_window:]))
            
            # 2. 使用指数平滑模型
            alpha = 0.3  # 平滑因子
            last_ema = prices[-1]
            
            # 3. 使用AR(1)模型
            if len(prices) >= 2:
                ar_coef = np.corrcoef(prices[:-1], prices[1:])[0, 1]
                ar_coef = max(0, min(ar_coef, 0.99))  # 限制系数在合理范围内
            else:
                ar_coef = 0.8  # 默认值
                
            mean_price = np.mean(prices[-10:]) if len(prices) >= 10 else np.mean(prices)
            
            # 综合预测
            predicted_prices = []
            last_price = prices[-1]
            
            for i in range(days):
                # 移动平均预测
                ma_pred = last_price + avg_change
                
                # 指数平滑预测
                ema_pred = alpha * last_price + (1 - alpha) * last_ema
                last_ema = ema_pred
                
                # AR(1)预测
                ar_pred = mean_price + ar_coef * (last_price - mean_price)
                
                # 综合模型（加权平均）
                combined_pred = 0.3 * ma_pred + 0.3 * ema_pred + 0.4 * ar_pred
                
                # 添加随机波动（模拟市场噪声）
                volatility = np.std(prices) / np.mean(prices) if np.mean(prices) > 0 else 0.01
                noise = np.random.normal(0, volatility * last_price * 0.05)
                combined_pred += noise
                
                # 确保预测价格为正
                combined_pred = max(0.01, combined_pred)
                
                # 添加到预测列表
                predicted_prices.append(combined_pred)
                
                # 更新最后价格
                last_price = combined_pred
                
            return predicted_prices
            
        except Exception as e:
            self.logger.error(f"标准模型预测失败: {str(e)}")
            raise e

    def _generate_backup_prediction(self, days=5, stock_code=None):
        """当标准预测方法失败时，生成备用预测数据
        
        Args:
            days: 预测天数
            stock_code: 股票代码
            
        Returns:
            list: 预测价格列表
        """
        self.logger.info(f"为股票 {stock_code} 生成备用预测")
        
        # 确定基准价格
        base_price = 50.0  # 默认基准价格
        
        if stock_code:
            try:
                # 尝试从股票代码获取一个基准价格
                code_digits = int(stock_code.replace('.SH', '').replace('.SZ', '')[-4:])
                base_price = 10.0 + (code_digits % 90)  # 生成10-100之间的价格
            except:
                pass
                
        # 生成偏向上涨的随机走势
        trend = random.uniform(-0.01, 0.02)  # 偏向上涨的趋势
        prices = []
        current_price = base_price
        
        for i in range(days):
            # 添加趋势和随机波动
            current_price *= (1 + trend + random.uniform(-0.02, 0.02))
            current_price = max(0.01, current_price)  # 确保价格为正
            prices.append(current_price)
            
        return prices

    def _traditional_prediction(self, stock_code, days=5):
        """使用传统模型进行预测
        
        Args:
            stock_code: 股票代码
            days: 预测天数
            
        Returns:
            dict: 预测结果
        """
        try:
            # 获取股票数据
            stock_data = self._get_stock_data(stock_code)
            
            if stock_data is None or stock_data.empty:
                self.logger.warning(f"无法获取股票 {stock_code} 的数据")
                return None
            
            # 使用简单的移动平均预测
            if 'close' in stock_data.columns:
                prices = stock_data['close'].values
                if len(prices) < 10:
                    self.logger.warning(f"股票 {stock_code} 的数据不足")
                    return None
                
                # 简单移动平均
                window_size = 5
                avg_change = np.mean(np.diff(prices[-window_size:]))
                last_price = prices[-1]
                
                # 生成预测
                predicted_prices = []
                for i in range(days):
                    next_price = last_price + avg_change
                    predicted_prices.append(round(next_price, 2))
                    last_price = next_price
                
                # 生成日期
                import datetime
                start_date = datetime.datetime.now()
                dates = [(start_date + datetime.timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, days+1)]
                
                # 计算变化率
                changes = []
                for i, price in enumerate(predicted_prices):
                    if i == 0:
                        prev_price = prices[-1]
                    else:
                        prev_price = predicted_prices[i-1]
                    
                    change = (price - prev_price) / prev_price * 100 if prev_price > 0 else 0
                    changes.append(round(change, 2))
                
                # 创建预测结果
                predictions = []
                for date, price, change in zip(dates, predicted_prices, changes):
                    predictions.append({
                        'date': date,
                        'price': price,
                        'change_percent': change
                    })
                
                # 构建结果
                result = {
                    'stock_code': stock_code,
                    'stock_name': stock_code,  # 可以从其他地方获取股票名称
                    'predictions': predictions,
                    'dates': dates,
                    'predicted_prices': predicted_prices,
                    'confidence': 50.0,  # 中等置信度
                    'market_sentiment': 0.5,
                    'market_momentum': 0,
                    'market_trend': 0,
                    'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'is_accurate': False,
                    'method': 'traditional_ma'
                }
                
                return result
            else:
                self.logger.warning(f"股票数据中没有收盘价")
                return None
        except Exception as e:
            self.logger.error(f"传统预测失败: {str(e)}")
            return None

    def _generate_mock_prediction(self, stock_code, days=5):
        """生成模拟预测数据
        
        Args:
            stock_code: 股票代码
            days: 预测天数
            
        Returns:
            dict: 模拟预测结果
        """
        import random
        import datetime
        
        # 生成基础价格
        if isinstance(stock_code, str) and len(stock_code) >= 2:
            try:
                # 使用股票代码后两位作为随机种子
                seed = int(stock_code[-2:])
                random.seed(seed)
                base_price = 15.0 + seed * 0.5
            except:
                base_price = random.uniform(20.0, 60.0)
        else:
            base_price = random.uniform(20.0, 60.0)
        
        # 生成预测价格
        trend = random.uniform(-0.01, 0.02)  # 趋势偏向于上涨
        predicted_prices = []
        current_price = base_price
        
        for i in range(days):
            # 添加趋势和随机波动
            current_price *= (1 + trend + random.uniform(-0.02, 0.02))
            current_price = round(max(0.01, current_price), 2)
            predicted_prices.append(current_price)
        
        # 生成日期
        start_date = datetime.datetime.now()
        dates = [(start_date + datetime.timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, days+1)]
        
        # 计算变化率
        changes = []
        prev_price = base_price
        for price in predicted_prices:
            change = (price - prev_price) / prev_price * 100 if prev_price > 0 else 0
            changes.append(round(change, 2))
            prev_price = price
        
        # 创建预测结果
        predictions = []
        for date, price, change in zip(dates, predicted_prices, changes):
            predictions.append({
                'date': date,
                'price': price,
                'change_percent': change
            })
        
        # 模拟市场洞察
        market_insights = [
            "市场动能显示潜在的稳定趋势",
            "波动性预计将保持在正常范围内",
            "建议关注大盘整体走势以获取额外指引"
        ]
        
        # 构建结果
        result = {
            'stock_code': stock_code,
            'stock_name': f"股票{stock_code}",
            'predictions': predictions,
            'dates': dates,
            'predicted_prices': predicted_prices,
            'confidence': 30.0,  # 较低置信度
            'market_sentiment': round(random.uniform(0.3, 0.7), 2),
            'market_momentum': round(random.uniform(-0.01, 0.01), 4),
            'market_trend': round(random.uniform(-0.005, 0.005), 4),
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'is_accurate': False,
            'method': 'mock',
            'market_insights': market_insights
        }
        
        return result
    
    def generate_market_insights(self):
        """生成市场洞察报告"""
        try:
            # 获取热门股票和推荐股票数据
            stocks_data = {}
            
            # 添加指数数据
            indices = self.get_indices_data()
            for index in indices:
                code = index.get("code", "")
                if code:
                    stocks_data[code] = index
            
            # 添加热门股票数据
            hot_stocks = self.get_hot_stocks(10)
            for stock in hot_stocks:
                code = stock.get("code", "")
                if code:
                    # 获取完整数据
                    stock_data = self.get_stock_data(code)
                    if stock_data:
                        stocks_data[code] = stock_data
            
            # 使用预测器生成市场洞察
            if self.predictor and hasattr(self.predictor, 'generate_market_insights') and stocks_data:
                insights = self.predictor.generate_market_insights(stocks_data)
                if insights:
                    self.market_insights_ready_signal.emit(insights)
                    return insights
            
            # 如果预测器不可用或无法生成洞察，使用模拟数据
            self.logger.info("使用模拟数据生成市场洞察")
            insights = self._generate_mock_market_insights()
            if insights:
                self.market_insights_ready_signal.emit(insights)
            return insights
            
        except Exception as e:
            self.logger.error(f"生成市场洞察失败: {str(e)}")
            # 尝试生成模拟洞察数据
            insights = self._generate_mock_market_insights()
            if insights:
                self.market_insights_ready_signal.emit(insights)
            return insights
    
    def _generate_mock_market_insights(self):
        """生成模拟市场洞察数据"""
        try:
            import random
            from datetime import datetime
            
            # 计算市场情绪
            sentiment_score = random.uniform(-1, 1)
            sentiment = "积极" if sentiment_score > 0.3 else "中性" if sentiment_score > -0.3 else "消极"
            
            # 生成行业表现
            industries = ["金融", "科技", "医药", "消费", "能源", "材料"]
            industry_performance = {industry: round(random.uniform(-3, 5), 2) for industry in industries}
            
            # 排名前3的行业
            top_industries = sorted(industry_performance.items(), key=lambda x: x[1], reverse=True)[:3]
            
            # 生成风险评估
            risk_level = random.choice(["低", "中", "高"])
            risk_factors = []
            if random.random() > 0.6:
                risk_factors.append("市场波动性增加")
            if random.random() > 0.6:
                risk_factors.append("国际形势不确定性")
            if random.random() > 0.6:
                risk_factors.append("流动性压力")
            if random.random() > 0.7:
                risk_factors.append("政策变动预期")
            
            # 生成投资建议
            investment_suggestions = []
            if sentiment_score > 0.3:
                investment_suggestions.append("适度增加权益类资产配置")
                if top_industries:
                    investment_suggestions.append(f"关注{top_industries[0][0]}行业机会")
            elif sentiment_score < -0.3:
                investment_suggestions.append("控制仓位，关注防御性板块")
            else:
                investment_suggestions.append("均衡配置，精选个股")
            
            # 生成市场洞察
            insights = {
                "market_sentiment": {
                    "score": round(sentiment_score, 2),
                    "evaluation": sentiment
                },
                "industry_performance": industry_performance,
                "top_industries": [{"name": ind, "change_pct": val} for ind, val in top_industries],
                "risk_assessment": {
                    "level": risk_level,
                    "factors": risk_factors
                },
                "investment_suggestions": investment_suggestions,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "is_quantum_enhanced": False,
                "source": "模拟数据"
            }
            
            return insights
            
        except Exception as e:
            self.logger.error(f"生成模拟市场洞察失败: {str(e)}")
            return None
    
    def set_tushare_token(self, token):
        """设置TuShare API Token"""
        try:
            self.tushare_token = token
            
            # 重新初始化TuShare数据源
            self.tushare_source = TushareDataSource(token=self.tushare_token)
            
            self.logger.info(f"TuShare API Token已更新")
            return True
        
        except Exception as e:
            self.logger.error(f"设置TuShare API Token失败: {str(e)}")
            return False
    
    def set_quantum_params(self, coherence=None, superposition=None, entanglement=None):
        """设置量子预测参数"""
        try:
            if self.predictor:
                success = self.predictor.set_quantum_params(
                    coherence=coherence,
                    superposition=superposition,
                    entanglement=entanglement
                )
                if success:
                    self.logger.info("量子预测参数已更新")
                return success
            else:
                self.logger.error("预测器未初始化，无法设置量子参数")
                return False
        
        except Exception as e:
            self.logger.error(f"设置量子预测参数失败: {str(e)}")
            return False
    
    def get_account_data(self):
        """获取账户数据"""
        # 模拟账户数据
        return {
            "total_asset": 1000000.0,
            "available_cash": 500000.0,
            "market_value": 500000.0,
            "daily_profit": 50000.0,
            "daily_profit_pct": 0.05,
            "total_profit": 100000.0,
            "total_profit_pct": 0.1,
            "max_drawdown": 0.052,
            "sharpe": 2.35,
            "volatility": 0.158,
            "var": 25000.0
        }
    
    def get_positions(self):
        """获取持仓数据"""
        # 模拟持仓数据
        positions = []
        for i, (code, stock) in enumerate(list(self.mock_data["stocks"].items())[:5]):
            quantity = np.random.randint(1000, 10000) // 100 * 100
            cost = stock["price"] * (1 - np.random.uniform(-0.1, 0.1))
            positions.append({
                "code": code,
                "name": stock["name"],
                "quantity": quantity,
                "available": quantity,
                "cost": cost,
                "price": stock["price"],
                "profit": (stock["price"] - cost) * quantity
            })
        return positions
    
    def get_allocation_data(self):
        """获取资产配置数据"""
        # 模拟资产配置数据
        return [
            {'name': '金融', 'value': 0.25, 'color': (255, 0, 0)},
            {'name': '科技', 'value': 0.30, 'color': (0, 255, 0)},
            {'name': '医药', 'value': 0.15, 'color': (0, 0, 255)},
            {'name': '消费', 'value': 0.20, 'color': (255, 255, 0)},
            {'name': '其他', 'value': 0.10, 'color': (128, 128, 128)}
        ]
    
    def get_performance_data(self):
        """获取绩效数据"""
        # 模拟绩效数据
        days = 100
        dates = pd.date_range(end=pd.Timestamp.now(), periods=days).tolist()
        portfolio_values = [1000000 * (1 + 0.001 * i + 0.002 * np.sin(i/10)) for i in range(days)]
        benchmark_values = [1000000 * (1 + 0.0008 * i) for i in range(days)]
        
        return {
            "dates": dates,
            "portfolio_values": portfolio_values,
            "benchmark_values": benchmark_values,
            "annual_return": 0.158,
            "alpha": 0.052,
            "beta": 0.85,
            "sortino": 1.95,
            "win_rate": 0.652,
            "profit_loss_ratio": 2.5
        }
    
    def get_strategy_data(self):
        """获取策略数据"""
        # 模拟策略数据
        days = 200
        dates = pd.date_range(end=pd.Timestamp.now(), periods=days).tolist()
        
        strategy1 = [100 * (1 + 0.001 * i + 0.003 * np.sin(i/10)) for i in range(days)]
        strategy2 = [100 * (1 + 0.0015 * i + 0.002 * np.cos(i/12)) for i in range(days)]
        strategy3 = [100 * (1 + 0.0012 * i - 0.001 * np.sin(i/15)) for i in range(days)]
        benchmark = [100 * (1 + 0.0008 * i) for i in range(days)]
        
        return {
            'dates': dates,
            'strategies': {
                '量子动量策略': strategy1,
                '分形套利策略': strategy2,
                '波动跟踪策略': strategy3,
                '基准': benchmark
            }
        }
    
    def get_strategy_stats(self):
        """获取策略统计数据"""
        # 模拟策略统计数据
        return [
            {
                'name': '量子动量策略',
                'annual_return': 0.268,
                'sharpe': 2.35,
                'max_drawdown': 0.12,
                'volatility': 0.15,
                'win_rate': 0.68,
                'avg_return': 0.021
            },
            {
                'name': '分形套利策略',
                'annual_return': 0.312,
                'sharpe': 2.56,
                'max_drawdown': 0.15,
                'volatility': 0.18,
                'win_rate': 0.72,
                'avg_return': 0.025
            },
            {
                'name': '波动跟踪策略',
                'annual_return': 0.245,
                'sharpe': 2.18,
                'max_drawdown': 0.10,
                'volatility': 0.13,
                'win_rate': 0.65,
                'avg_return': 0.019
            }
        ]
    
    def get_correlation_data(self):
        """获取相关性数据"""
        # 模拟相关性数据
        stocks = ["工商银行", "茅台", "腾讯", "阿里巴巴", "平安保险", "中国石油", "中国移动", "恒瑞医药", "格力电器", "万科A"]
        n = len(stocks)
        
        # 创建相关性矩阵
        correlation_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    # 生成一个-0.5到0.9之间的随机相关系数
                    correlation_matrix[i, j] = np.random.uniform(-0.5, 0.9)
                    correlation_matrix[j, i] = correlation_matrix[i, j]  # 确保对称
        
        return {
            'stocks': stocks,
            'matrix': correlation_matrix
        }
    
    def get_risk_decomposition(self):
        """获取风险分解数据"""
        # 模拟风险分解数据
        return [
            {'name': '市场风险', 'value': 0.45, 'color': (255, 0, 0)},
            {'name': '特异风险', 'value': 0.25, 'color': (0, 255, 0)},
            {'name': '行业风险', 'value': 0.15, 'color': (0, 0, 255)},
            {'name': '风格风险', 'value': 0.10, 'color': (255, 255, 0)},
            {'name': '其他风险', 'value': 0.05, 'color': (128, 128, 128)}
        ]
    
    def get_risk_metrics(self):
        """获取风险指标数据"""
        # 模拟风险指标数据
        return {
            'var': 0.025,  # 95% VaR
            'cvar': 0.035,  # 95% CVaR
            'volatility': 0.15,  # 年化波动率
            'max_drawdown': 0.12,  # 最大回撤
            'downside_risk': 0.08,  # 下行风险
            'beta': 0.85,  # Beta
            'tracking_error': 0.05  # 跟踪误差
        }
    
    def get_network_status(self):
        """获取网络状态"""
        # 模拟网络状态数据
        return {
            'segments': 5,
            'agents': 25,
            'learning': True,
            'evolution': 3,
            'performance': 85.2
        }
    
    def get_market_data(self):
        """获取市场数据，用于宇宙共振分析
        
        Returns:
            dict: 市场数据，包含指数、热门股票等
        """
        try:
            market_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "indices": [],
                "hot_stocks": [],
                "sectors": {}
            }
            
            # 获取指数数据
            indices = self.get_indices_data()
            market_data["indices"] = indices
            
            # 获取热门股票
            hot_stocks = self.get_hot_stocks(10)
            market_data["hot_stocks"] = hot_stocks
            
            # 提取行业数据
            sectors = {}
            for stock in hot_stocks:
                sector = stock.get("industry", "未知")
                if sector not in sectors:
                    sectors[sector] = []
                sectors[sector].append(stock)
            
            market_data["sectors"] = sectors
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"获取市场数据失败: {str(e)}")
            return {}
    
    def get_supergod_insights(self):
        """获取超神洞察"""
        if hasattr(self, 'supergod_insights') and self.supergod_insights:
            return self.supergod_insights
        
        # 如果没有加载过结果，尝试使用模拟数据
        self.logger.info("使用模拟数据生成市场洞察")
        return self._generate_mock_insights()
    
    def get_supergod_predictions(self):
        """获取超神预测"""
        if hasattr(self, 'supergod_predictions') and self.supergod_predictions:
            return self.supergod_predictions
        
        # 如果没有加载过结果，返回空列表
        return []

    def _get_stock_data(self, stock_code, days=365):
        """获取股票历史数据
        
        Args:
            stock_code: 股票代码
            days: 获取天数
            
        Returns:
            pandas.DataFrame: 股票数据，如果失败则返回None
        """
        try:
            self.logger.info(f"获取股票 {stock_code} 的历史数据")
            
            # 检查缓存
            cache_key = f"{stock_code}_daily_{days}"
            if cache_key in self.cache:
                cache_time, cache_data = self.cache[cache_key]
                # 如果缓存不超过30分钟，直接使用
                if (datetime.now() - cache_time).total_seconds() < 1800:
                    self.logger.info(f"使用缓存的 {stock_code} 历史数据")
                    return cache_data
            
            # 使用新的日线数据获取方法
            df = self.get_daily_data(stock_code, days=days)
            
            if df is not None and not df.empty:
                # 更新缓存
                self.cache[cache_key] = (datetime.now(), df)
                return df
            
            # 如果无法获取数据，返回None
            return None
        
        except Exception as e:
            self.logger.error(f"获取股票数据时出错: {str(e)}")
            return None

    def get_daily_data(self, code, start_date=None, end_date=None, days=365):
        """获取股票日线数据
        
        Args:
            code: 股票代码
            start_date: 开始日期，如果为None则使用end_date往前推days天
            end_date: 结束日期，如果为None则使用当前日期
            days: 天数
            
        Returns:
            pd.DataFrame: 日线数据，包含日期、开盘价、收盘价、最高价、最低价、成交量
        """
        self.logger.debug(f"获取 {code} 日线数据, days={days}")
        
        # 确保TuShare数据源已准备好
        if not self.is_tushare_ready:
            self.logger.error("TuShare数据源未准备好，无法获取日线数据")
            self.error_signal.emit("TuShare数据源未准备好，请检查网络和token有效性")
            raise Exception("TuShare数据源未准备好")
            
        # 获取日线数据
        try:
            df = self.tushare_source.get_daily_data(
                code, 
                start_date=start_date, 
                end_date=end_date, 
                days=days
            )
            
            if df is None or df.empty:
                self.logger.error(f"获取 {code} 日线数据失败")
                self.error_signal.emit(f"获取 {code} 日线数据失败")
                raise Exception(f"获取 {code} 日线数据失败")
                
            return df
            
        except Exception as e:
            self.logger.error(f"获取 {code} 日线数据失败: {str(e)}")
            self.error_signal.emit(f"获取 {code} 日线数据失败: {str(e)}")
            raise Exception(f"获取 {code} 日线数据失败: {str(e)}")

    def on_connect_symbiosis(self, symbiosis_core):
        """连接到共生核心时调用
        
        Args:
            symbiosis_core: 共生核心实例
        """
        self.logger.info("数据控制器已连接到共生核心")
        self.symbiosis_core = symbiosis_core
        
        # 发送一条连接消息
        if hasattr(symbiosis_core, "send_message"):
            try:
                # 尝试新版API
                symbiosis_core.send_message(
                    source="data_controller",
                    target=None,  # 广播给所有模块
                    message_type="connection",
                    data={"status": "ready", "sources": [s.name for s in self.data_sources]}
                )
            except Exception as e:
                try:
                    # 尝试旧版API
                    symbiosis_core.send_message(
                        source_module="data_controller",
                        target_module=None,  # 广播给所有模块
                        message_type="connection",
                        data={"status": "ready", "sources": [s.name for s in self.data_sources]}
                    )
                except Exception as ee:
                    self.logger.error(f"无法发送连接消息: {str(ee)}")
        
    def on_disconnect_symbiosis(self):
        """从共生核心断开时调用"""
        self.logger.info("数据控制器已断开与共生核心的连接")
        self.symbiosis_core = None
        
    def on_message(self, message):
        """处理来自共生核心的消息
        
        Args:
            message: 消息数据
        """
        try:
            message_type = message.get("type", "")
            source = message.get("source", "unknown")
            data = message.get("data", {})
            
            self.logger.debug(f"收到消息 [{message_type}] 来自 {source}")
            
            # 处理请求数据类型的消息
            if message_type == "request_data":
                stock_code = data.get("stock_code")
                days = data.get("days", 30)
                
                if stock_code:
                    # 获取股票数据
                    stock_data = self.get_daily_data(stock_code, days=days)
                    
                    # 返回数据响应
                    if self.symbiosis_core and hasattr(self.symbiosis_core, "send_message"):
                        self.symbiosis_core.send_message(
                            source="data_controller",
                            target=source,  # 回复请求源
                            message_type="data_response",
                            data={
                                "stock_code": stock_code,
                                "data": stock_data.to_dict() if not stock_data.empty else {},
                                "status": "success" if not stock_data.empty else "empty"
                            }
                        )
            
            # 处理数据源状态消息
            elif message_type == "datasource_status":
                source_name = data.get("name")
                status = data.get("status")
                
                if source_name and status:
                    self.logger.info(f"数据源 [{source_name}] 状态更新: {status}")
                    
                    # 如果是TuShare数据源状态变更
                    if source_name == "tushare" and status == "ready":
                        # 通知其他模块TuShare数据源已准备好
                        if self.symbiosis_core and hasattr(self.symbiosis_core, "send_message"):
                            self.symbiosis_core.send_message(
                                source="data_controller",
                                target=None,  # 广播
                                message_type="tushare_ready",
                                data={"status": "ready"}
                            )
                
        except Exception as e:
            self.logger.error(f"处理消息时出错: {str(e)}")