#!/usr/bin/env python3
"""
量子预测器 - 超神系统的核心预测引擎

基于量子计算原理的市场预测和趋势分析工具
"""

import logging
import threading
import time
import random
import numpy as np
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

# 尝试导入量子核心
try:
    from quantum_core.quantum_engine import QuantumEngine
    from quantum_core.quantum_state import QuantumState
    QUANTUM_CORE_AVAILABLE = True
except ImportError:
    QUANTUM_CORE_AVAILABLE = False
    
# 尝试导入宇宙共振
try:
    from cosmic_resonance.cosmic_resonator import get_cosmic_resonator
    COSMIC_RESONANCE_AVAILABLE = True
except ImportError:
    COSMIC_RESONANCE_AVAILABLE = False

# 导入预测相关类
from .prediction_model import PredictionModel
from .prediction_result import PredictionResult


class QuantumPredictor:
    """量子预测器类
    
    超神系统的核心预测引擎，使用量子计算和宇宙共振进行市场预测
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化量子预测器
        
        Args:
            config: 预测器配置
        """
        self.logger = logging.getLogger("QuantumPredictor")
        self.logger.info("初始化量子预测器...")
        
        # 设置默认配置
        self.config = {
            "prediction_horizon": [1, 5, 10, 30, 60],  # 预测周期（分钟）
            "quantum_dimensions": 12,                  # 量子维度
            "prediction_confidence_threshold": 0.65,   # 预测置信度阈值
            "cosmic_resonance_factor": 0.3,            # 宇宙共振影响因子
            "auto_calibration": True,                  # 自动校准
            "calibration_interval": 60,                # 校准间隔（分钟）
            "parallel_universes": 7,                   # 并行宇宙数
            "use_quantum_core": QUANTUM_CORE_AVAILABLE,# 使用量子核心
            "use_cosmic_resonance": COSMIC_RESONANCE_AVAILABLE, # 使用宇宙共振
            "prediction_models": ["quantum_wave", "timeline_confluence", "entropic_analysis"]
        }
        
        # 更新配置
        if config:
            self.config.update(config)
            
        # 预测器状态
        self.state = {
            "active": False,
            "initialized": False,
            "calibration_level": 0.0,
            "accuracy": 0.0,
            "predictions_made": 0,
            "successful_predictions": 0,
            "last_calibration": None,
            "current_resonance": 0.0,
            "quantum_stability": 0.8,
            "timeline_coherence": 0.7,
            "last_update": datetime.now()
        }
        
        # 预测模型
        self.models = {}
        
        # 预测历史
        self.prediction_history = []
        self.max_history_size = 1000
        
        # 量子引擎和共振器
        self.quantum_engine = None
        self.cosmic_resonator = None
        
        # 锁和线程
        self.lock = threading.RLock()
        self.calibration_thread = None
        self.prediction_thread = None
        self._stop_event = threading.Event()
        
        # 初始化组件
        self._initialize_components()
        
        self.logger.info("量子预测器初始化完成")
        
    def _initialize_components(self):
        """初始化预测器组件"""
        # 初始化量子引擎
        if self.config["use_quantum_core"] and QUANTUM_CORE_AVAILABLE:
            try:
                self.logger.info("初始化量子引擎...")
                self.quantum_engine = QuantumEngine(
                    dimensions=self.config["quantum_dimensions"],
                    compute_power=0.8
                )
                self.quantum_engine.start()
                self.logger.info("量子引擎初始化成功")
            except Exception as e:
                self.logger.error(f"初始化量子引擎失败: {str(e)}")
                self.quantum_engine = None
        
        # 初始化宇宙共振器
        if self.config["use_cosmic_resonance"] and COSMIC_RESONANCE_AVAILABLE:
            try:
                self.logger.info("初始化宇宙共振器...")
                self.cosmic_resonator = get_cosmic_resonator(
                    resonance_threshold=0.7,
                    cosmic_sensitivity=0.8
                )
                self.logger.info("宇宙共振器初始化成功")
            except Exception as e:
                self.logger.error(f"初始化宇宙共振器失败: {str(e)}")
                self.cosmic_resonator = None
        
        # 初始化预测模型
        self._initialize_prediction_models()
        
        # 标记初始化完成
        self.state["initialized"] = True
        
    def _initialize_prediction_models(self):
        """初始化预测模型"""
        model_configs = {
            "quantum_wave": {
                "dimensions": self.config["quantum_dimensions"],
                "wavefront_sensitivity": 0.8,
                "phase_alignment": 0.7,
                "temporal_resolution": 0.001
            },
            "timeline_confluence": {
                "parallel_timelines": self.config["parallel_universes"],
                "confluence_threshold": 0.65,
                "divergence_weight": 0.4,
                "temporal_momentum": 0.6
            },
            "entropic_analysis": {
                "entropy_sensitivity": 0.75,
                "chaos_damping": 0.3,
                "order_emergence_factor": 0.6,
                "pattern_recognition_threshold": 0.55
            },
            "quantum_field_analysis": {
                "field_dimensions": 5,
                "field_strength": 0.7,
                "interaction_depth": 3,
                "coherence_threshold": 0.6
            },
            "probability_wave_collapse": {
                "wave_count": 12,
                "collapse_threshold": 0.7,
                "observer_effect": 0.2,
                "quantum_tunneling": True
            }
        }
        
        # 创建已配置的模型
        for model_name in self.config["prediction_models"]:
            if model_name in model_configs:
                try:
                    self.logger.info(f"初始化预测模型: {model_name}")
                    model_config = model_configs[model_name]
                    self.models[model_name] = PredictionModel(
                        name=model_name,
                        config=model_config
                    )
                except Exception as e:
                    self.logger.error(f"初始化模型 {model_name} 失败: {str(e)}")
    
    def start(self) -> bool:
        """启动预测器
        
        Returns:
            bool: 启动是否成功
        """
        with self.lock:
            if self.state["active"]:
                self.logger.info("预测器已在运行中")
                return True
                
            if not self.state["initialized"]:
                self.logger.error("预测器未正确初始化")
                return False
                
            # 启动宇宙共振器
            if self.cosmic_resonator:
                try:
                    self.cosmic_resonator.start()
                except Exception as e:
                    self.logger.error(f"启动宇宙共振器失败: {str(e)}")
            
            # 设置为活跃状态
            self.state["active"] = True
            self._stop_event.clear()
            
            # 启动校准线程
            if self.config["auto_calibration"]:
                self.calibration_thread = threading.Thread(target=self._calibration_loop)
                self.calibration_thread.daemon = True
                self.calibration_thread.start()
            
            self.logger.info("量子预测器已启动")
            return True
    
    def stop(self) -> bool:
        """停止预测器
        
        Returns:
            bool: 停止是否成功
        """
        with self.lock:
            if not self.state["active"]:
                return True
                
            # 设置停止标志
            self._stop_event.set()
            self.state["active"] = False
            
            # 停止宇宙共振器
            if self.cosmic_resonator:
                try:
                    self.cosmic_resonator.stop()
                except Exception as e:
                    self.logger.error(f"停止宇宙共振器失败: {str(e)}")
            
            # 等待线程结束
            if self.calibration_thread and self.calibration_thread.is_alive():
                self.calibration_thread.join(timeout=2.0)
                
            if self.prediction_thread and self.prediction_thread.is_alive():
                self.prediction_thread.join(timeout=2.0)
                
            self.logger.info("量子预测器已停止")
            return True
    
    def predict(self, market_data: Dict[str, Any], 
                horizon: int = None, 
                include_details: bool = False) -> PredictionResult:
        """生成市场预测
        
        Args:
            market_data: 市场数据
            horizon: 预测周期（分钟）
            include_details: 是否包含详细信息
            
        Returns:
            PredictionResult: 预测结果
        """
        with self.lock:
            if not self.state["active"]:
                self.logger.warning("预测器未启动")
                return PredictionResult(
                    success=False,
                    error="predictor_not_active",
                    prediction_id=str(uuid.uuid4())
                )
            
            # 使用默认周期（如果未指定）
            if horizon is None:
                horizon = self.config["prediction_horizon"][0]
                
            start_time = datetime.now()
            self.logger.info(f"开始生成预测 (周期: {horizon}分钟)")
            
            # 生成预测ID
            prediction_id = f"pred_{int(time.time())}_{random.randint(1000, 9999)}"
            
            try:
                # 预处理市场数据
                processed_data = self._preprocess_market_data(market_data)
                
                # 收集模型预测
                model_predictions = {}
                model_confidences = {}
                
                for model_name, model in self.models.items():
                    # 使用量子增强进行预测
                    prediction, confidence = self._generate_model_prediction(
                        model=model, 
                        data=processed_data, 
                        horizon=horizon
                    )
                    
                    model_predictions[model_name] = prediction
                    model_confidences[model_name] = confidence
                
                # 整合各模型预测
                final_prediction, confidence = self._integrate_predictions(
                    model_predictions, 
                    model_confidences
                )
                
                # 应用宇宙共振调整
                if self.cosmic_resonator:
                    # 获取当前宇宙共振强度
                    cosmic_energy = self.cosmic_resonator.get_cosmic_energy()
                    
                    # 根据共振调整预测
                    final_prediction, confidence = self._apply_cosmic_adjustment(
                        prediction=final_prediction,
                        confidence=confidence,
                        cosmic_energy=cosmic_energy
                    )
                    
                    # 更新状态
                    self.state["current_resonance"] = cosmic_energy
                
                # 构建预测结果
                result = PredictionResult(
                    prediction_id=prediction_id,
                    timestamp=start_time,
                    horizon=horizon,
                    success=True,
                    prediction=final_prediction,
                    confidence=confidence,
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
                
                # 添加详细信息
                if include_details:
                    result.details = {
                        "model_predictions": model_predictions,
                        "model_confidences": model_confidences,
                        "cosmic_resonance": self.state["current_resonance"] if self.cosmic_resonator else 0.0,
                        "quantum_stability": self.state["quantum_stability"],
                        "calibration_level": self.state["calibration_level"],
                        "dimensions_used": self.config["quantum_dimensions"]
                    }
                
                # 更新状态和历史
                self.state["predictions_made"] += 1
                self._add_to_prediction_history(result)
                
                self.logger.info(f"预测完成: {prediction_id}, 置信度: {confidence:.2f}")
                return result
                
            except Exception as e:
                self.logger.error(f"生成预测失败: {str(e)}")
                
                # 返回错误结果
                return PredictionResult(
                    prediction_id=prediction_id,
                    timestamp=start_time,
                    success=False,
                    error=str(e),
                    execution_time=(datetime.now() - start_time).total_seconds()
                )
    
    def predict_async(self, market_data: Dict[str, Any], 
                      horizon: int = None,
                      callback=None) -> str:
        """异步生成市场预测
        
        Args:
            market_data: 市场数据
            horizon: 预测周期（分钟）
            callback: 完成回调函数
            
        Returns:
            str: 预测ID
        """
        # 生成预测ID
        prediction_id = f"pred_async_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # 创建并启动预测线程
        self.prediction_thread = threading.Thread(
            target=self._async_prediction_worker,
            args=(prediction_id, market_data, horizon, callback)
        )
        self.prediction_thread.daemon = True
        self.prediction_thread.start()
        
        self.logger.info(f"启动异步预测: {prediction_id}")
        return prediction_id
    
    def _async_prediction_worker(self, prediction_id: str, 
                                market_data: Dict[str, Any],
                                horizon: int,
                                callback) -> None:
        """异步预测工作线程
        
        Args:
            prediction_id: 预测ID
            market_data: 市场数据
            horizon: 预测周期
            callback: 回调函数
        """
        try:
            # 执行同步预测
            result = self.predict(market_data, horizon, include_details=True)
            
            # 确保使用传入的ID
            result.prediction_id = prediction_id
            
            # 调用回调函数（如果提供）
            if callback and callable(callback):
                callback(result)
                
        except Exception as e:
            self.logger.error(f"异步预测工作线程错误: {str(e)}")
            
            # 创建错误结果
            error_result = PredictionResult(
                prediction_id=prediction_id,
                success=False,
                error=str(e),
                timestamp=datetime.now()
            )
            
            # 调用回调函数（如果提供）
            if callback and callable(callback):
                callback(error_result)
    
    def calibrate(self) -> Dict[str, Any]:
        """校准预测器
        
        Returns:
            Dict[str, Any]: 校准结果
        """
        with self.lock:
            self.logger.info("开始校准量子预测器...")
            
            start_time = datetime.now()
            
            # 量子校准过程
            if self.quantum_engine:
                try:
                    # 获取量子引擎状态
                    quantum_state = self.quantum_engine.get_state()
                    
                    # 调整量子稳定性
                    self.state["quantum_stability"] = quantum_state.get("stability", 0.8)
                    
                    # 执行量子校准计算
                    calibration_task = {
                        "type": "calibration",
                        "dimensions": self.config["quantum_dimensions"],
                        "time": time.time()
                    }
                    
                    # 提交计算任务
                    task_id = self.quantum_engine.submit_calculation(calibration_task, priority=0.9)
                    
                    # 等待结果
                    result = self.quantum_engine.get_result(task_id, timeout=5.0)
                    
                    if result:
                        # 更新校准状态
                        self.state["calibration_level"] = result.get("calibration", 0.7)
                    else:
                        self.logger.warning("量子校准计算超时")
                        
                except Exception as e:
                    self.logger.error(f"量子校准失败: {str(e)}")
            else:
                # 无量子引擎，使用模拟校准
                self.state["calibration_level"] = random.uniform(0.7, 0.95)
                self.state["quantum_stability"] = random.uniform(0.75, 0.9)
                
            # 模型校准
            for model_name, model in self.models.items():
                try:
                    model.calibrate(self.state["calibration_level"])
                except Exception as e:
                    self.logger.error(f"校准模型 {model_name} 失败: {str(e)}")
            
            # 更新校准时间
            self.state["last_calibration"] = datetime.now()
            
            # 计算校准耗时
            calibration_time = (datetime.now() - start_time).total_seconds()
            
            # 构建校准结果
            calibration_result = {
                "success": True,
                "calibration_level": self.state["calibration_level"],
                "quantum_stability": self.state["quantum_stability"],
                "execution_time": calibration_time,
                "models_calibrated": list(self.models.keys()),
                "timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"校准完成: 级别={self.state['calibration_level']:.2f}, 耗时={calibration_time:.2f}秒")
            return calibration_result
    
    def _calibration_loop(self):
        """校准循环线程"""
        self.logger.info("启动自动校准线程")
        
        while not self._stop_event.is_set():
            try:
                # 检查是否需要校准
                needs_calibration = False
                
                if self.state["last_calibration"] is None:
                    # 首次校准
                    needs_calibration = True
                else:
                    # 检查校准间隔
                    elapsed = (datetime.now() - self.state["last_calibration"]).total_seconds()
                    interval_seconds = self.config["calibration_interval"] * 60
                    
                    if elapsed >= interval_seconds:
                        needs_calibration = True
                
                # 执行校准
                if needs_calibration:
                    self.calibrate()
                
                # 等待一段时间（检查间隔为1分钟）
                for _ in range(60):
                    if self._stop_event.is_set():
                        break
                    time.sleep(1)
                    
            except Exception as e:
                self.logger.error(f"校准线程错误: {str(e)}")
                time.sleep(60)  # 出错后等待较长时间
        
        self.logger.info("校准线程已停止")
    
    def _preprocess_market_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """预处理市场数据
        
        Args:
            market_data: 原始市场数据
            
        Returns:
            Dict[str, Any]: 处理后的市场数据
        """
        # 确保数据格式正确
        if not isinstance(market_data, dict):
            raise ValueError("市场数据必须为字典格式")
            
        # 复制数据避免修改原始数据
        processed_data = market_data.copy()
        
        # 基本数据验证
        required_fields = ["symbol", "timestamp", "price"]
        for field in required_fields:
            if field not in processed_data:
                raise ValueError(f"市场数据缺少必要字段: {field}")
        
        # 添加处理标记
        processed_data["preprocessed"] = True
        processed_data["preprocessed_at"] = datetime.now().isoformat()
        
        # 应用量子噪声过滤（如果有量子引擎）
        if self.quantum_engine:
            try:
                # 如果有价格历史数据，应用量子噪声过滤
                if "price_history" in processed_data and isinstance(processed_data["price_history"], list):
                    # 量子噪声过滤算法
                    noise_task = {
                        "type": "noise_filter",
                        "data": processed_data["price_history"],
                        "filter_level": 0.7
                    }
                    
                    # 提交量子计算
                    task_id = self.quantum_engine.submit_calculation(noise_task, priority=0.6)
                    result = self.quantum_engine.get_result(task_id, timeout=2.0)
                    
                    if result and "filtered_data" in result:
                        processed_data["price_history"] = result["filtered_data"]
                        processed_data["quantum_filtered"] = True
            except Exception as e:
                self.logger.warning(f"量子噪声过滤失败: {str(e)}")
        
        return processed_data
    
    def _generate_model_prediction(self, model: PredictionModel, 
                                 data: Dict[str, Any], 
                                 horizon: int) -> Tuple[Dict[str, Any], float]:
        """使用模型生成预测
        
        Args:
            model: 预测模型
            data: 处理后的市场数据
            horizon: 预测周期（分钟）
            
        Returns:
            Tuple[Dict[str, Any], float]: (预测结果, 置信度)
        """
        # 使用模型生成预测
        prediction_data = model.predict(data, horizon)
        
        # 获取基础预测
        base_prediction = prediction_data.get("prediction", {})
        base_confidence = prediction_data.get("confidence", 0.5)
        
        # 应用量子增强
        if self.quantum_engine:
            try:
                # 创建量子状态
                market_state = {
                    "price": data.get("price", 0),
                    "volume": data.get("volume", 0),
                    "trend": data.get("trend", 0),
                    "volatility": data.get("volatility", 0),
                    "timestamp": time.time()
                }
                
                # 执行量子预测计算
                quantum_task = {
                    "type": "prediction_enhancement",
                    "base_prediction": base_prediction,
                    "market_state": market_state,
                    "horizon": horizon,
                    "dimensions": self.config["quantum_dimensions"]
                }
                
                # 提交量子计算
                task_id = self.quantum_engine.submit_calculation(quantum_task, priority=0.8)
                result = self.quantum_engine.get_result(task_id, timeout=3.0)
                
                if result:
                    # 获取量子增强的预测
                    enhanced_prediction = result.get("enhanced_prediction", base_prediction)
                    enhanced_confidence = result.get("enhanced_confidence", base_confidence)
                    
                    # 返回增强的预测
                    return enhanced_prediction, enhanced_confidence
                    
            except Exception as e:
                self.logger.warning(f"量子预测增强失败: {str(e)}")
        
        # 如果没有量子增强或量子增强失败，返回基础预测
        return base_prediction, base_confidence
    
    def _integrate_predictions(self, 
                               model_predictions: Dict[str, Dict[str, Any]],
                               model_confidences: Dict[str, float]) -> Tuple[Dict[str, Any], float]:
        """整合多个模型的预测
        
        Args:
            model_predictions: 各模型的预测
            model_confidences: 各模型的置信度
            
        Returns:
            Tuple[Dict[str, Any], float]: (整合后的预测, 整合后的置信度)
        """
        if not model_predictions:
            raise ValueError("没有可用的模型预测")
            
        # 计算权重（基于置信度）
        total_confidence = sum(model_confidences.values())
        
        if total_confidence <= 0:
            # 所有模型置信度都为0，使用均等权重
            weights = {model: 1.0 / len(model_confidences) for model in model_confidences}
        else:
            # 使用基于置信度的权重
            weights = {model: conf / total_confidence for model, conf in model_confidences.items()}
        
        # 收集价格预测
        price_predictions = {}
        trend_predictions = {}
        volatility_predictions = {}
        
        # 提取各预测组件
        for model_name, prediction in model_predictions.items():
            if "price" in prediction:
                price_predictions[model_name] = prediction["price"]
                
            if "trend" in prediction:
                trend_predictions[model_name] = prediction["trend"]
                
            if "volatility" in prediction:
                volatility_predictions[model_name] = prediction["volatility"]
        
        # 计算整合结果
        integrated_prediction = {}
        
        # 整合价格预测
        if price_predictions:
            integrated_price = sum(price * weights[model] for model, price in price_predictions.items())
            integrated_prediction["price"] = integrated_price
            
        # 整合趋势预测
        if trend_predictions:
            integrated_trend = sum(trend * weights[model] for model, trend in trend_predictions.items())
            integrated_prediction["trend"] = integrated_trend
            
        # 整合波动性预测
        if volatility_predictions:
            integrated_volatility = sum(vol * weights[model] for model, vol in volatility_predictions.items())
            integrated_prediction["volatility"] = integrated_volatility
            
        # 计算整合置信度
        # 加权平均置信度，并应用校准因子
        raw_confidence = sum(conf * weights[model] for model, conf in model_confidences.items())
        calibration_factor = self.state["calibration_level"]
        
        integrated_confidence = raw_confidence * (0.7 + 0.3 * calibration_factor)
        
        # 应用量子稳定性修正
        integrated_confidence *= (0.5 + 0.5 * self.state["quantum_stability"])
        
        # 确保置信度在有效范围内
        integrated_confidence = max(0.1, min(0.99, integrated_confidence))
        
        return integrated_prediction, integrated_confidence
    
    def _apply_cosmic_adjustment(self, 
                                prediction: Dict[str, Any], 
                                confidence: float,
                                cosmic_energy: float) -> Tuple[Dict[str, Any], float]:
        """应用宇宙共振调整
        
        Args:
            prediction: 预测结果
            confidence: 预测置信度
            cosmic_energy: 宇宙能量水平
            
        Returns:
            Tuple[Dict[str, Any], float]: (调整后的预测, 调整后的置信度)
        """
        # 复制预测避免修改原始数据
        adjusted_prediction = prediction.copy()
        
        # 计算共振影响因子
        cosmic_factor = self.config["cosmic_resonance_factor"]
        
        # 宇宙能量对置信度的影响
        # 高能量状态下提高置信度
        confidence_adjustment = (cosmic_energy - 0.5) * cosmic_factor
        adjusted_confidence = confidence + confidence_adjustment
        
        # 价格预测调整
        if "price" in prediction:
            # 能量波动影响价格预测
            energy_deviation = (cosmic_energy - 0.5) * 0.02
            price_adjustment = prediction["price"] * energy_deviation
            adjusted_prediction["price"] = prediction["price"] + price_adjustment
            
        # 趋势预测调整
        if "trend" in prediction:
            # 能量状态影响趋势方向
            trend_adjustment = (cosmic_energy - 0.5) * cosmic_factor * 0.2
            adjusted_prediction["trend"] = max(-1.0, min(1.0, prediction["trend"] + trend_adjustment))
            
        # 波动性预测调整
        if "volatility" in prediction:
            # 能量状态影响波动性
            vol_adjustment = cosmic_energy * cosmic_factor * 0.15
            adjusted_prediction["volatility"] = max(0.01, prediction["volatility"] + vol_adjustment)
        
        # 确保置信度在有效范围内
        adjusted_confidence = max(0.1, min(0.99, adjusted_confidence))
        
        return adjusted_prediction, adjusted_confidence
    
    def _add_to_prediction_history(self, prediction: PredictionResult) -> None:
        """添加预测到历史记录
        
        Args:
            prediction: 预测结果
        """
        self.prediction_history.append(prediction)
        
        # 限制历史记录大小
        if len(self.prediction_history) > self.max_history_size:
            self.prediction_history = self.prediction_history[-self.max_history_size:]
    
    def evaluate_prediction(self, prediction_id: str, 
                           actual_data: Dict[str, Any]) -> Dict[str, Any]:
        """评估预测准确性
        
        Args:
            prediction_id: 预测ID
            actual_data: 实际市场数据
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        # 查找预测
        target_prediction = None
        for pred in self.prediction_history:
            if pred.prediction_id == prediction_id:
                target_prediction = pred
                break
                
        if not target_prediction:
            return {
                "success": False,
                "error": "prediction_not_found",
                "prediction_id": prediction_id
            }
        
        # 确保预测成功
        if not target_prediction.success:
            return {
                "success": False,
                "error": "original_prediction_failed",
                "prediction_id": prediction_id
            }
        
        try:
            # 计算价格预测误差
            price_error = 0.0
            price_accuracy = 0.0
            
            if "price" in target_prediction.prediction and "price" in actual_data:
                predicted_price = target_prediction.prediction["price"]
                actual_price = actual_data["price"]
                
                # 计算相对误差
                price_error = abs(predicted_price - actual_price) / actual_price
                price_accuracy = max(0.0, 1.0 - price_error)
            
            # 计算趋势预测准确性
            trend_accuracy = 0.0
            
            if "trend" in target_prediction.prediction and "trend" in actual_data:
                predicted_trend = target_prediction.prediction["trend"]
                actual_trend = actual_data["trend"]
                
                # 趋势方向一致则为正确
                if (predicted_trend > 0 and actual_trend > 0) or (predicted_trend < 0 and actual_trend < 0):
                    trend_accuracy = 1.0
                else:
                    trend_accuracy = 0.0
            
            # 计算总体准确性
            overall_accuracy = (price_accuracy * 0.7 + trend_accuracy * 0.3)
            
            # 更新预测器统计
            with self.lock:
                # 根据准确性阈值判断是否为成功预测
                if overall_accuracy >= 0.6:
                    self.state["successful_predictions"] += 1
                
                # 更新总体准确率
                total_predictions = self.state["predictions_made"]
                if total_predictions > 0:
                    self.state["accuracy"] = (
                        (self.state["accuracy"] * (total_predictions - 1) + overall_accuracy) / 
                        total_predictions
                    )
            
            # 构建评估结果
            evaluation = {
                "success": True,
                "prediction_id": prediction_id,
                "overall_accuracy": overall_accuracy,
                "price_accuracy": price_accuracy,
                "trend_accuracy": trend_accuracy,
                "price_error": price_error,
                "evaluation_time": datetime.now().isoformat(),
                "original_prediction": target_prediction.prediction,
                "actual_data": actual_data
            }
            
            self.logger.info(f"预测评估: ID={prediction_id}, 准确度={overall_accuracy:.2f}")
            return evaluation
            
        except Exception as e:
            self.logger.error(f"评估预测失败: {str(e)}")
            
            return {
                "success": False,
                "error": str(e),
                "prediction_id": prediction_id
            }
    
    def get_state(self) -> Dict[str, Any]:
        """获取预测器状态
        
        Returns:
            Dict[str, Any]: 预测器状态
        """
        with self.lock:
            state_copy = self.state.copy()
            
            # 处理datetime对象
            if state_copy["last_calibration"]:
                state_copy["last_calibration"] = state_copy["last_calibration"].isoformat()
                
            state_copy["last_update"] = state_copy["last_update"].isoformat()
            
            # 添加额外信息
            state_copy["config"] = self.config.copy()
            state_copy["model_count"] = len(self.models)
            state_copy["history_count"] = len(self.prediction_history)
            
            # 添加组件状态
            state_copy["components"] = {
                "quantum_engine": self.quantum_engine is not None,
                "cosmic_resonator": self.cosmic_resonator is not None
            }
            
            return state_copy
    
    def get_prediction_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取预测历史
        
        Args:
            limit: 返回的历史记录数量
            
        Returns:
            List[Dict[str, Any]]: 预测历史
        """
        with self.lock:
            # 获取最近的n条记录
            recent_history = self.prediction_history[-limit:]
            
            # 转换为字典
            history_dicts = []
            for pred in recent_history:
                history_dicts.append(pred.to_dict())
                
            return history_dicts
    
    def reset_statistics(self) -> Dict[str, Any]:
        """重置统计信息
        
        Returns:
            Dict[str, Any]: 重置结果
        """
        with self.lock:
            # 保存重置前的统计信息
            old_stats = {
                "predictions_made": self.state["predictions_made"],
                "successful_predictions": self.state["successful_predictions"],
                "accuracy": self.state["accuracy"]
            }
            
            # 重置统计信息
            self.state["predictions_made"] = 0
            self.state["successful_predictions"] = 0
            self.state["accuracy"] = 0.0
            
            self.logger.info("预测器统计信息已重置")
            
            return {
                "success": True,
                "previous_stats": old_stats,
                "reset_time": datetime.now().isoformat()
            }


def create_predictor(config: Dict[str, Any] = None) -> QuantumPredictor:
    """创建量子预测器实例
    
    Args:
        config: 预测器配置
        
    Returns:
        QuantumPredictor: 量子预测器实例
    """
    predictor = QuantumPredictor(config)
    return predictor 