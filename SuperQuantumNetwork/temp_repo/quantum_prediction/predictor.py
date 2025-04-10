#!/usr/bin/env python3
"""
量子预测器 - 超神系统的预测核心

提供基于量子计算的市场预测能力
"""

import logging
import threading
import time
import random
import numpy as np
from datetime import datetime, timedelta
import os
import json

from .prediction_model import PredictionModel
from .prediction_result import PredictionResult
from .feature_extractor import FeatureExtractor
from .prediction_evaluator import PredictionEvaluator

class QuantumPredictor:
    """量子预测器
    
    超神系统的预测核心，提供量子增强的市场预测
    """
    
    def __init__(self, dimensions=12, model_path="models"):
        """初始化量子预测器
        
        Args:
            dimensions: 量子维度
            model_path: 模型路径
        """
        self.logger = logging.getLogger("QuantumPredictor")
        self.logger.info("初始化量子预测器...")
        
        # 预测器状态
        self.predictor_state = {
            "active": False,
            "dimensions": dimensions,
            "quantum_coherence": 0.85,
            "model_count": 0,
            "prediction_accuracy": 0.0,
            "last_prediction": None,
            "last_update": datetime.now()
        }
        
        # 模型路径
        self.model_path = model_path
        
        # 预测模型
        self.models = {}
        
        # 特征提取器
        self.feature_extractor = FeatureExtractor()
        
        # 预测评估器
        self.evaluator = PredictionEvaluator()
        
        # 预测结果缓存
        self.prediction_cache = {}
        self.cache_validity = 3600  # 缓存有效期(秒)
        
        # 量子引擎连接
        self.quantum_engine = None
        
        # 数据源连接
        self.data_source = None
        
        # 运行线程
        self.active = False
        self.prediction_thread = None
        self.lock = threading.RLock()
        
        # 超参数
        self.hyperparameters = {
            "prediction_horizon": [1, 3, 5, 10, 21],  # 预测时间范围(天)
            "confidence_threshold": 0.65,  # 置信度阈值
            "quantum_enhancement": 0.3,  # 量子增强系数
            "feature_importance_threshold": 0.05,  # 特征重要性阈值
            "ensemble_weights": {  # 集成模型权重
                "lstm": 0.4,
                "gru": 0.3,
                "transformer": 0.2,
                "quantum": 0.1
            }
        }
        
    def initialize(self):
        """初始化量子预测器"""
        with self.lock:
            # 初始化特征提取器
            self.feature_extractor.initialize()
            
            # 初始化预测评估器
            self.evaluator.initialize()
            
            # 确保模型目录存在
            os.makedirs(self.model_path, exist_ok=True)
            
            self.predictor_state["active"] = True
            self.active = True
            
            self.logger.info("量子预测器初始化完成")
            return True
            
    def set_quantum_engine(self, engine):
        """设置量子引擎连接
        
        Args:
            engine: 量子引擎实例
        """
        self.quantum_engine = engine
        self.logger.info("量子引擎连接成功")
        
    def set_data_source(self, data_source):
        """设置数据源连接
        
        Args:
            data_source: 数据源实例
        """
        self.data_source = data_source
        self.logger.info("数据源连接成功")
        
    def load_models(self):
        """加载预测模型"""
        with self.lock:
            try:
                # 获取模型文件列表
                model_files = [f for f in os.listdir(self.model_path) if f.endswith('.h5') or f.endswith('.json')]
                
                # 加载每个模型
                loaded_count = 0
                for model_file in model_files:
                    if model_file.endswith('.h5'):
                        # 提取股票代码
                        symbol = os.path.splitext(model_file)[0]
                        
                        # 检查是否有对应的缩放器
                        scaler_file = f"scaler_{symbol}.json"
                        if scaler_file in model_files:
                            # 加载模型
                            model = PredictionModel(symbol)
                            if model.load(os.path.join(self.model_path, model_file), 
                                        os.path.join(self.model_path, scaler_file)):
                                self.models[symbol] = model
                                loaded_count += 1
                
                self.predictor_state["model_count"] = loaded_count
                self.logger.info(f"成功加载 {loaded_count} 个预测模型")
                return True
                
            except Exception as e:
                self.logger.error(f"加载预测模型失败: {str(e)}")
                return False
                
    def start_prediction_service(self):
        """启动预测服务"""
        with self.lock:
            if not self.predictor_state["active"]:
                self.logger.warning("量子预测器未初始化，无法启动预测服务")
                return False
                
            # 启动预测线程
            if not self.prediction_thread or not self.prediction_thread.is_alive():
                self.prediction_thread = threading.Thread(target=self._prediction_processor)
                self.prediction_thread.daemon = True
                self.prediction_thread.start()
                
            self.logger.info("预测服务已启动")
            return True
            
    def stop_prediction_service(self):
        """停止预测服务"""
        with self.lock:
            self.active = False
            
            # 等待预测线程结束
            if self.prediction_thread and self.prediction_thread.is_alive():
                self.prediction_thread.join(timeout=2.0)
                
            self.logger.info("预测服务已停止")
            return True
            
    def predict_stock(self, symbol, horizon=5, use_quantum=True):
        """预测单个股票
        
        Args:
            symbol: 股票代码
            horizon: 预测时间范围(天)
            use_quantum: 是否使用量子增强
            
        Returns:
            PredictionResult: 预测结果
        """
        # 检查缓存
        cache_key = f"{symbol}_{horizon}_{use_quantum}"
        if cache_key in self.prediction_cache:
            cache_item = self.prediction_cache[cache_key]
            if (datetime.now() - cache_item["timestamp"]).total_seconds() < self.cache_validity:
                return cache_item["result"]
        
        # 获取模型
        model = self.models.get(symbol)
        if not model:
            self.logger.warning(f"未找到股票 {symbol} 的预测模型")
            return None
            
        # 获取历史数据
        if not self.data_source:
            self.logger.warning("未设置数据源，无法获取历史数据")
            return None
            
        # 获取数据，假设data_source有get_stock_history方法
        history_days = model.get_input_length() + horizon
        end_date = datetime.now()
        start_date = end_date - timedelta(days=history_days * 2)  # 获取更多数据以防有非交易日
        
        try:
            history_data = self.data_source.get_daily_data(
                symbol, 
                start_date.strftime('%Y%m%d'), 
                end_date.strftime('%Y%m%d')
            )
            
            if history_data is None or history_data.empty:
                self.logger.warning(f"无法获取股票 {symbol} 的历史数据")
                return None
                
            # 提取特征
            features = self.feature_extractor.extract_features(history_data)
            
            # 预测
            prediction = model.predict(features, horizon)
            
            # 量子增强
            if use_quantum and self.quantum_engine:
                try:
                    # 计算量子概率分布
                    quantum_params = {
                        "symbol": symbol,
                        "horizon": horizon,
                        "base_prediction": prediction.to_dict(),
                        "market_state": self._get_market_state()
                    }
                    
                    # 假设quantum_engine有calculate_quantum_probability方法
                    quantum_prob = self.quantum_engine.calculate_quantum_probability(quantum_params)
                    
                    # 应用量子调整
                    prediction.apply_quantum_adjustment(quantum_prob, self.hyperparameters["quantum_enhancement"])
                    
                except Exception as e:
                    self.logger.error(f"应用量子增强失败: {str(e)}")
            
            # 评估预测结果
            self.evaluator.evaluate_prediction(prediction)
            
            # 更新状态
            self.predictor_state["last_prediction"] = datetime.now()
            
            # 缓存结果
            self.prediction_cache[cache_key] = {
                "timestamp": datetime.now(),
                "result": prediction
            }
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"预测股票 {symbol} 失败: {str(e)}")
            return None
            
    def predict_portfolio(self, symbols, horizon=5, use_quantum=True):
        """预测投资组合
        
        Args:
            symbols: 股票代码列表
            horizon: 预测时间范围(天)
            use_quantum: 是否使用量子增强
            
        Returns:
            dict: 预测结果字典，键为股票代码
        """
        results = {}
        
        for symbol in symbols:
            result = self.predict_stock(symbol, horizon, use_quantum)
            if result:
                results[symbol] = result
                
        return results
        
    def predict_market(self, market_indices, horizon=5, use_quantum=True):
        """预测市场
        
        Args:
            market_indices: 市场指数代码列表
            horizon: 预测时间范围(天)
            use_quantum: 是否使用量子增强
            
        Returns:
            dict: 预测结果字典，键为指数代码
        """
        return self.predict_portfolio(market_indices, horizon, use_quantum)
        
    def generate_trading_signals(self, predictions):
        """根据预测生成交易信号
        
        Args:
            predictions: 预测结果字典，键为股票代码
            
        Returns:
            dict: 交易信号字典，键为股票代码
        """
        signals = {}
        
        for symbol, prediction in predictions.items():
            if not prediction:
                continue
                
            # 获取信号
            signal = prediction.get_trading_signal(self.hyperparameters["confidence_threshold"])
            
            signals[symbol] = {
                "signal": signal.signal_type,
                "confidence": signal.confidence,
                "target_price": signal.target_price,
                "current_price": signal.current_price,
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        return signals
        
    def get_predictor_state(self):
        """获取预测器状态
        
        Returns:
            dict: 当前预测器状态
        """
        with self.lock:
            # 计算准确率
            if self.evaluator:
                self.predictor_state["prediction_accuracy"] = self.evaluator.get_overall_accuracy()
                
            # 返回状态副本
            return self.predictor_state.copy()
            
    def _prediction_processor(self):
        """预测处理线程"""
        self.logger.info("预测处理线程已启动")
        
        while self.active:
            try:
                # 休眠
                time.sleep(300)  # 每5分钟运行一次
                
                # 获取市场指数
                if self.data_source:
                    indices = self._get_major_indices()
                    if indices:
                        # 预测市场
                        self.predict_market(indices)
                        
                # 清理过期缓存
                self._clean_expired_cache()
                
            except Exception as e:
                self.logger.error(f"预测处理发生错误: {str(e)}")
                time.sleep(60)
                
        self.logger.info("预测处理线程已结束")
        
    def _get_major_indices(self):
        """获取主要市场指数代码"""
        # 中国市场主要指数
        return ["000001.SH", "399001.SZ", "399006.SZ", "000688.SH"]
        
    def _get_market_state(self):
        """获取当前市场状态"""
        market_state = {
            "volatility": random.uniform(0.01, 0.05),
            "trend": random.choice(["bullish", "bearish", "neutral"]),
            "liquidity": random.uniform(0.6, 1.0),
            "sentiment": random.uniform(-1.0, 1.0)
        }
        
        # 如果有数据源，尝试获取真实市场状态
        if self.data_source:
            try:
                real_state = self.data_source.get_market_state()
                if real_state:
                    market_state.update(real_state)
            except:
                pass
                
        return market_state
        
    def _clean_expired_cache(self):
        """清理过期缓存"""
        now = datetime.now()
        expired_keys = []
        
        for key, cache_item in self.prediction_cache.items():
            if (now - cache_item["timestamp"]).total_seconds() >= self.cache_validity:
                expired_keys.append(key)
                
        for key in expired_keys:
            del self.prediction_cache[key]

def get_quantum_predictor(dimensions=12, model_path="models"):
    """获取量子预测器
    
    全局单例模式
    
    Args:
        dimensions: 量子维度
        model_path: 模型路径
        
    Returns:
        QuantumPredictor: 量子预测器实例
    """
    # 使用全局变量保存实例
    if not hasattr(get_quantum_predictor, "_instance") or not get_quantum_predictor._instance:
        get_quantum_predictor._instance = QuantumPredictor(dimensions, model_path)
        
    return get_quantum_predictor._instance 