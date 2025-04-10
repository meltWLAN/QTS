#!/usr/bin/env python3
"""
预测结果 - 超神系统量子预测的结果封装

提供预测结果的封装、处理和分析功能
"""

import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

class PredictionResult:
    """预测结果类
    
    封装量子预测器生成的预测结果，提供结果处理和转换功能
    """
    
    def __init__(self, 
                prediction_id: str = None, 
                timestamp: datetime = None,
                horizon: int = None,
                success: bool = False,
                prediction: Dict[str, Any] = None,
                confidence: float = 0.0,
                error: str = None,
                execution_time: float = None,
                details: Dict[str, Any] = None):
        """初始化预测结果
        
        Args:
            prediction_id: 预测ID
            timestamp: 预测时间戳
            horizon: 预测周期（分钟）
            success: 预测是否成功
            prediction: 预测数据
            confidence: 预测置信度
            error: 错误信息（如果预测失败）
            execution_time: 执行时间（秒）
            details: 详细信息
        """
        # 设置预测ID
        self.prediction_id = prediction_id or f"pred_{int(time.time())}_{hash(str(datetime.now()))}"
        
        # 设置时间戳
        self.timestamp = timestamp or datetime.now()
        
        # 预测周期
        self.horizon = horizon
        
        # 预测状态
        self.success = success
        
        # 预测数据
        self.prediction = prediction or {}
        
        # 预测置信度
        self.confidence = confidence
        
        # 错误信息
        self.error = error
        
        # 执行时间
        self.execution_time = execution_time
        
        # 详细信息
        self.details = details or {}
        
        # 预测验证
        self.validated = False
        self.validation_result = None
        
        # 预测分类
        self._classify_prediction()
    
    def _classify_prediction(self):
        """根据预测数据对预测进行分类"""
        # 默认分类
        self.classification = {
            "signal_type": "unknown",
            "strength": "neutral",
            "confidence_level": "low",
            "time_frame": "unknown"
        }
        
        if not self.success or not self.prediction:
            return
            
        # 判断信号类型
        if "trend" in self.prediction:
            trend = self.prediction["trend"]
            if trend > 0.3:
                self.classification["signal_type"] = "strong_buy"
            elif trend > 0.1:
                self.classification["signal_type"] = "buy"
            elif trend < -0.3:
                self.classification["signal_type"] = "strong_sell"
            elif trend < -0.1:
                self.classification["signal_type"] = "sell"
            else:
                self.classification["signal_type"] = "neutral"
        
        # 判断信号强度
        if "volatility" in self.prediction:
            volatility = self.prediction["volatility"]
            if volatility > 0.03:
                self.classification["strength"] = "high"
            elif volatility > 0.01:
                self.classification["strength"] = "medium"
            else:
                self.classification["strength"] = "low"
                
        # 置信度级别
        if self.confidence > 0.8:
            self.classification["confidence_level"] = "very_high"
        elif self.confidence > 0.65:
            self.classification["confidence_level"] = "high"
        elif self.confidence > 0.5:
            self.classification["confidence_level"] = "medium"
        else:
            self.classification["confidence_level"] = "low"
            
        # 时间框架
        if self.horizon:
            if self.horizon <= 5:
                self.classification["time_frame"] = "very_short"
            elif self.horizon <= 15:
                self.classification["time_frame"] = "short"
            elif self.horizon <= 60:
                self.classification["time_frame"] = "medium"
            else:
                self.classification["time_frame"] = "long"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式
        
        Returns:
            Dict[str, Any]: 字典表示
        """
        result = {
            "prediction_id": self.prediction_id,
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            "horizon": self.horizon,
            "success": self.success,
            "confidence": self.confidence,
            "classification": self.classification
        }
        
        # 添加预测数据（如果有）
        if self.prediction:
            result["prediction"] = self.prediction
            
        # 添加错误信息（如果有）
        if self.error:
            result["error"] = self.error
            
        # 添加执行时间（如果有）
        if self.execution_time is not None:
            result["execution_time"] = self.execution_time
            
        # 添加详细信息（如果有）
        if self.details:
            result["details"] = self.details
            
        # 添加验证结果（如果有）
        if self.validated:
            result["validation"] = self.validation_result
            
        return result
    
    def to_json(self) -> str:
        """转换为JSON字符串
        
        Returns:
            str: JSON字符串
        """
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PredictionResult':
        """从字典创建预测结果
        
        Args:
            data: 预测结果字典
            
        Returns:
            PredictionResult: 预测结果对象
        """
        # 处理时间戳
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp)
            except (ValueError, TypeError):
                timestamp = datetime.now()
                
        # 创建实例
        result = cls(
            prediction_id=data.get("prediction_id"),
            timestamp=timestamp,
            horizon=data.get("horizon"),
            success=data.get("success", False),
            prediction=data.get("prediction"),
            confidence=data.get("confidence", 0.0),
            error=data.get("error"),
            execution_time=data.get("execution_time"),
            details=data.get("details")
        )
        
        # 设置验证结果（如果有）
        if "validation" in data:
            result.validated = True
            result.validation_result = data["validation"]
            
        # 设置分类（如果有）
        if "classification" in data:
            result.classification = data["classification"]
            
        return result
    
    @classmethod
    def from_json(cls, json_str: str) -> 'PredictionResult':
        """从JSON字符串创建预测结果
        
        Args:
            json_str: JSON字符串
            
        Returns:
            PredictionResult: 预测结果对象
        """
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except json.JSONDecodeError:
            # 创建错误结果
            return cls(
                success=False,
                error="invalid_json_format"
            )
    
    def validate(self, actual_data: Dict[str, Any]) -> Dict[str, Any]:
        """验证预测结果
        
        Args:
            actual_data: 实际市场数据
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        if not self.success:
            return {
                "validated": True,
                "success": False,
                "error": "cannot_validate_failed_prediction"
            }
            
        if not self.prediction:
            return {
                "validated": True,
                "success": False,
                "error": "empty_prediction"
            }
            
        try:
            # 计算预测准确性
            accuracy_metrics = {}
            
            # 价格准确性
            if "price" in self.prediction and "price" in actual_data:
                predicted_price = self.prediction["price"]
                actual_price = actual_data["price"]
                
                # 计算相对误差
                price_error = abs(predicted_price - actual_price) / actual_price
                accuracy_metrics["price_accuracy"] = max(0.0, 1.0 - price_error)
                accuracy_metrics["price_error"] = price_error
                accuracy_metrics["price_direction_correct"] = (
                    (predicted_price > actual_data.get("reference_price", 0) and 
                     actual_price > actual_data.get("reference_price", 0)) or
                    (predicted_price < actual_data.get("reference_price", 0) and 
                     actual_price < actual_data.get("reference_price", 0))
                )
            
            # 趋势准确性
            if "trend" in self.prediction and "trend" in actual_data:
                predicted_trend = self.prediction["trend"]
                actual_trend = actual_data["trend"]
                
                # 计算趋势准确性
                accuracy_metrics["trend_direction_correct"] = (
                    (predicted_trend > 0 and actual_trend > 0) or
                    (predicted_trend < 0 and actual_trend < 0)
                )
                
                # 计算趋势误差
                trend_error = abs(predicted_trend - actual_trend)
                accuracy_metrics["trend_error"] = trend_error
                accuracy_metrics["trend_accuracy"] = max(0.0, 1.0 - trend_error)
            
            # 波动性准确性
            if "volatility" in self.prediction and "volatility" in actual_data:
                predicted_volatility = self.prediction["volatility"]
                actual_volatility = actual_data["volatility"]
                
                # 计算波动性误差
                volatility_error = abs(predicted_volatility - actual_volatility)
                accuracy_metrics["volatility_error"] = volatility_error
                accuracy_metrics["volatility_accuracy"] = max(0.0, 1.0 - volatility_error / max(0.01, actual_volatility))
            
            # 计算总体准确性
            overall_accuracy = 0.0
            accuracy_count = 0
            
            if "price_accuracy" in accuracy_metrics:
                overall_accuracy += accuracy_metrics["price_accuracy"] * 0.6
                accuracy_count += 0.6
                
            if "trend_accuracy" in accuracy_metrics:
                overall_accuracy += accuracy_metrics["trend_accuracy"] * 0.3
                accuracy_count += 0.3
                
            if "volatility_accuracy" in accuracy_metrics:
                overall_accuracy += accuracy_metrics["volatility_accuracy"] * 0.1
                accuracy_count += 0.1
                
            if accuracy_count > 0:
                accuracy_metrics["overall_accuracy"] = overall_accuracy / accuracy_count
            else:
                accuracy_metrics["overall_accuracy"] = 0.0
                
            # 设置验证结果
            validation_result = {
                "validated": True,
                "success": True,
                "validation_time": datetime.now().isoformat(),
                "metrics": accuracy_metrics,
                "actual_data": {k: v for k, v in actual_data.items() if k != "price_history"}
            }
            
            self.validated = True
            self.validation_result = validation_result
            
            return validation_result
            
        except Exception as e:
            # 验证失败
            validation_result = {
                "validated": True,
                "success": False,
                "error": str(e),
                "validation_time": datetime.now().isoformat()
            }
            
            self.validated = True
            self.validation_result = validation_result
            
            return validation_result
    
    def get_signal(self) -> Dict[str, Any]:
        """获取交易信号
        
        基于预测生成交易信号
        
        Returns:
            Dict[str, Any]: 交易信号
        """
        if not self.success:
            return {
                "signal": "none",
                "reason": "failed_prediction",
                "confidence": 0.0
            }
            
        # 默认信号
        signal = {
            "signal": "none",
            "type": self.classification["signal_type"],
            "strength": self.classification["strength"],
            "confidence": self.confidence,
            "time_frame": self.classification["time_frame"],
            "timestamp": datetime.now().isoformat(),
            "prediction_id": self.prediction_id
        }
        
        # 根据预测生成信号
        if self.classification["signal_type"] in ["strong_buy", "buy"]:
            signal["signal"] = "buy"
        elif self.classification["signal_type"] in ["strong_sell", "sell"]:
            signal["signal"] = "sell"
        else:
            signal["signal"] = "hold"
            
        # 只有高置信度的才给出交易信号
        if self.confidence < 0.6:
            signal["signal"] = "hold"
            signal["reason"] = "low_confidence"
            
        # 添加预测数据
        if self.prediction:
            signal["prediction"] = self.prediction
            
        return signal
    
    def get_summary(self) -> str:
        """获取预测结果摘要
        
        Returns:
            str: 预测结果摘要
        """
        if not self.success:
            return f"预测失败: {self.error or '未知错误'}"
            
        # 格式化时间戳
        if isinstance(self.timestamp, datetime):
            time_str = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        else:
            time_str = str(self.timestamp)
            
        # 预测周期
        horizon_str = f"{self.horizon}分钟" if self.horizon else "未知"
        
        # 信号类型
        signal_types = {
            "strong_buy": "强烈买入",
            "buy": "买入",
            "neutral": "中性",
            "sell": "卖出",
            "strong_sell": "强烈卖出",
            "unknown": "未知"
        }
        signal_type = signal_types.get(self.classification["signal_type"], "未知")
        
        # 预测价格
        price_str = f"{self.prediction.get('price', 'N/A'):.2f}" if "price" in self.prediction else "N/A"
        
        # 构建摘要
        summary = (
            f"预测ID: {self.prediction_id}\n"
            f"时间: {time_str}\n"
            f"周期: {horizon_str}\n"
            f"信号: {signal_type}\n"
            f"价格: {price_str}\n"
            f"置信度: {self.confidence:.2f}"
        )
        
        # 添加验证结果（如果有）
        if self.validated and self.validation_result and self.validation_result.get("success", False):
            metrics = self.validation_result.get("metrics", {})
            accuracy = metrics.get("overall_accuracy", 0.0)
            summary += f"\n验证结果: 准确率 {accuracy:.2f}"
            
        return summary 