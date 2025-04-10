#!/usr/bin/env python3
"""
预测评估器 - 超神系统量子预测的评估工具

提供预测结果的评估、性能分析和优化建议
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import defaultdict

from .prediction_result import PredictionResult

class PredictionEvaluator:
    """预测评估器类
    
    评估预测结果的准确性、可靠性和性能，提供优化建议
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化预测评估器
        
        Args:
            config: 配置参数
        """
        self.logger = logging.getLogger("PredictionEvaluator")
        
        # 默认配置
        self.default_config = {
            "min_predictions_for_analysis": 10,     # 分析所需的最小预测数量
            "confidence_thresholds": [0.5, 0.6, 0.7, 0.8, 0.9],  # 置信度阈值分析
            "time_decay_factor": 0.95,              # 时间衰减因子（新预测权重更高）
            "store_all_evaluations": True,          # 存储所有评估结果
            "comparison_window": 30,                # 比较窗口（天）
        }
        
        # 合并配置
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
            
        # 评估历史
        self.evaluation_history = []
        self.evaluation_summary = {
            "total_predictions": 0,
            "successful_predictions": 0,
            "overall_accuracy": 0.0,
            "price_accuracy": 0.0,
            "trend_accuracy": 0.0,
            "by_confidence": {},
            "by_model": {},
            "by_horizon": {},
            "by_time": {},
            "by_symbol": {}
        }
        
        # 模型性能缓存
        self.model_performance = {}
        
        # 最佳参数建议
        self.parameter_suggestions = {}
        
        self.logger.info("预测评估器初始化完成")
        
    def evaluate_prediction(self, prediction: Union[PredictionResult, Dict[str, Any]], 
                          actual_data: Dict[str, Any]) -> Dict[str, Any]:
        """评估单个预测结果
        
        Args:
            prediction: 预测结果对象或字典
            actual_data: 实际市场数据
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        # 确保我们有PredictionResult对象
        if isinstance(prediction, dict):
            pred_result = PredictionResult.from_dict(prediction)
        else:
            pred_result = prediction
            
        # 验证预测
        validation_result = pred_result.validate(actual_data)
        
        # 提取评估指标
        metrics = validation_result.get("metrics", {})
        
        # 创建评估记录
        evaluation = {
            "prediction_id": pred_result.prediction_id,
            "timestamp": datetime.now().isoformat(),
            "prediction_timestamp": pred_result.timestamp.isoformat() if isinstance(pred_result.timestamp, datetime) else pred_result.timestamp,
            "symbol": actual_data.get("symbol", "unknown"),
            "horizon": pred_result.horizon,
            "confidence": pred_result.confidence,
            "model": pred_result.details.get("model", "unknown") if pred_result.details else "unknown",
            "metrics": metrics,
            "success": validation_result.get("success", False),
            "prediction_data": pred_result.prediction,
            "actual_data": {k: v for k, v in actual_data.items() if k != "price_history"}
        }
        
        # 存储评估结果
        if self.config["store_all_evaluations"]:
            self.evaluation_history.append(evaluation)
            
        # 更新评估摘要
        self._update_evaluation_summary(evaluation)
        
        return evaluation
    
    def _update_evaluation_summary(self, evaluation: Dict[str, Any]) -> None:
        """更新评估摘要统计
        
        Args:
            evaluation: 评估结果
        """
        # 更新总预测数
        self.evaluation_summary["total_predictions"] += 1
        
        # 如果评估成功
        if evaluation["success"]:
            metrics = evaluation["metrics"]
            
            # 更新成功预测数
            successful = False
            if "overall_accuracy" in metrics and metrics["overall_accuracy"] > 0.5:
                successful = True
                self.evaluation_summary["successful_predictions"] += 1
                
            # 更新准确性统计
            if "overall_accuracy" in metrics:
                # 使用衰减因子更新总体准确性
                weight = self.config["time_decay_factor"] ** (self.evaluation_summary["total_predictions"] - 1)
                old_weight = 1 - weight
                
                self.evaluation_summary["overall_accuracy"] = (
                    self.evaluation_summary["overall_accuracy"] * old_weight +
                    metrics["overall_accuracy"] * weight
                )
                
            if "price_accuracy" in metrics:
                # 更新价格准确性
                weight = self.config["time_decay_factor"] ** (self.evaluation_summary["total_predictions"] - 1)
                old_weight = 1 - weight
                
                self.evaluation_summary["price_accuracy"] = (
                    self.evaluation_summary["price_accuracy"] * old_weight +
                    metrics["price_accuracy"] * weight
                )
                
            if "trend_direction_correct" in metrics:
                # 更新趋势准确性
                weight = self.config["time_decay_factor"] ** (self.evaluation_summary["total_predictions"] - 1)
                old_weight = 1 - weight
                
                trend_accuracy = 1.0 if metrics["trend_direction_correct"] else 0.0
                self.evaluation_summary["trend_accuracy"] = (
                    self.evaluation_summary["trend_accuracy"] * old_weight +
                    trend_accuracy * weight
                )
                
            # 按置信度分类
            confidence = evaluation["confidence"]
            conf_key = str(round(confidence * 10) / 10)  # 舍入到一位小数
            
            if conf_key not in self.evaluation_summary["by_confidence"]:
                self.evaluation_summary["by_confidence"][conf_key] = {
                    "total": 0,
                    "successful": 0,
                    "accuracy": 0.0
                }
                
            self.evaluation_summary["by_confidence"][conf_key]["total"] += 1
            if successful:
                self.evaluation_summary["by_confidence"][conf_key]["successful"] += 1
                
            # 更新置信度组的准确性
            conf_group = self.evaluation_summary["by_confidence"][conf_key]
            conf_group["accuracy"] = conf_group["successful"] / conf_group["total"]
            
            # 按模型分类
            model = evaluation["model"]
            if model not in self.evaluation_summary["by_model"]:
                self.evaluation_summary["by_model"][model] = {
                    "total": 0,
                    "successful": 0,
                    "accuracy": 0.0
                }
                
            self.evaluation_summary["by_model"][model]["total"] += 1
            if successful:
                self.evaluation_summary["by_model"][model]["successful"] += 1
                
            # 更新模型组的准确性
            model_group = self.evaluation_summary["by_model"][model]
            model_group["accuracy"] = model_group["successful"] / model_group["total"]
            
            # 按预测周期分类
            horizon = evaluation["horizon"]
            horizon_key = str(horizon)
            
            if horizon_key not in self.evaluation_summary["by_horizon"]:
                self.evaluation_summary["by_horizon"][horizon_key] = {
                    "total": 0,
                    "successful": 0,
                    "accuracy": 0.0
                }
                
            self.evaluation_summary["by_horizon"][horizon_key]["total"] += 1
            if successful:
                self.evaluation_summary["by_horizon"][horizon_key]["successful"] += 1
                
            # 更新预测周期组的准确性
            horizon_group = self.evaluation_summary["by_horizon"][horizon_key]
            horizon_group["accuracy"] = horizon_group["successful"] / horizon_group["total"]
            
            # 按时间分类（每天）
            pred_time = None
            try:
                if isinstance(evaluation["prediction_timestamp"], str):
                    pred_time = datetime.fromisoformat(evaluation["prediction_timestamp"])
                elif isinstance(evaluation["prediction_timestamp"], datetime):
                    pred_time = evaluation["prediction_timestamp"]
            except (ValueError, TypeError):
                pass
                
            if pred_time:
                date_key = pred_time.strftime("%Y-%m-%d")
                
                if date_key not in self.evaluation_summary["by_time"]:
                    self.evaluation_summary["by_time"][date_key] = {
                        "total": 0,
                        "successful": 0,
                        "accuracy": 0.0
                    }
                    
                self.evaluation_summary["by_time"][date_key]["total"] += 1
                if successful:
                    self.evaluation_summary["by_time"][date_key]["successful"] += 1
                    
                # 更新时间组的准确性
                time_group = self.evaluation_summary["by_time"][date_key]
                time_group["accuracy"] = time_group["successful"] / time_group["total"]
                
            # 按交易品种分类
            symbol = evaluation["symbol"]
            
            if symbol not in self.evaluation_summary["by_symbol"]:
                self.evaluation_summary["by_symbol"][symbol] = {
                    "total": 0,
                    "successful": 0,
                    "accuracy": 0.0
                }
                
            self.evaluation_summary["by_symbol"][symbol]["total"] += 1
            if successful:
                self.evaluation_summary["by_symbol"][symbol]["successful"] += 1
                
            # 更新交易品种组的准确性
            symbol_group = self.evaluation_summary["by_symbol"][symbol]
            symbol_group["accuracy"] = symbol_group["successful"] / symbol_group["total"]
    
    def batch_evaluate(self, predictions: List[Union[PredictionResult, Dict[str, Any]]], 
                     actual_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """批量评估多个预测结果
        
        Args:
            predictions: 预测结果列表
            actual_data_list: 实际市场数据列表
            
        Returns:
            Dict[str, Any]: 批量评估结果
        """
        if len(predictions) != len(actual_data_list):
            self.logger.error(f"预测数量 ({len(predictions)}) 与实际数据数量 ({len(actual_data_list)}) 不匹配")
            return {"error": "predictions_and_actual_data_mismatch"}
            
        # 批量评估结果
        batch_results = []
        
        # 执行评估
        for i, (pred, actual) in enumerate(zip(predictions, actual_data_list)):
            try:
                result = self.evaluate_prediction(pred, actual)
                batch_results.append(result)
            except Exception as e:
                self.logger.error(f"评估第 {i} 个预测时发生错误: {str(e)}")
                # 添加错误结果
                batch_results.append({
                    "success": False,
                    "error": str(e),
                    "prediction_id": pred.prediction_id if isinstance(pred, PredictionResult) else pred.get("prediction_id", f"unknown_{i}")
                })
                
        # 统计批量结果
        successful = sum(1 for r in batch_results if r.get("success", False))
        success_rate = successful / len(batch_results) if batch_results else 0
        
        # 计算平均准确性
        accuracies = [r["metrics"].get("overall_accuracy", 0) for r in batch_results if r.get("success", False) and "metrics" in r]
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
        
        # 创建批量报告
        batch_report = {
            "total_evaluated": len(batch_results),
            "successful_evaluations": successful,
            "success_rate": success_rate,
            "average_accuracy": avg_accuracy,
            "timestamp": datetime.now().isoformat(),
            "details": batch_results
        }
        
        return batch_report
    
    def analyze_performance(self) -> Dict[str, Any]:
        """分析预测性能和趋势
        
        Returns:
            Dict[str, Any]: 性能分析结果
        """
        # 确保有足够的数据进行分析
        min_predictions = self.config["min_predictions_for_analysis"]
        if self.evaluation_summary["total_predictions"] < min_predictions:
            self.logger.warning(f"可用预测数量不足: {self.evaluation_summary['total_predictions']} < {min_predictions}")
            return {
                "success": False,
                "error": "insufficient_data",
                "required": min_predictions,
                "available": self.evaluation_summary["total_predictions"]
            }
            
        # 准备分析结果
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "data_points": self.evaluation_summary["total_predictions"],
            "overall_performance": {
                "accuracy": self.evaluation_summary["overall_accuracy"],
                "success_rate": self.evaluation_summary["successful_predictions"] / self.evaluation_summary["total_predictions"],
                "price_accuracy": self.evaluation_summary["price_accuracy"],
                "trend_accuracy": self.evaluation_summary["trend_accuracy"]
            },
            "confidence_analysis": self._analyze_confidence(),
            "model_analysis": self._analyze_models(),
            "horizon_analysis": self._analyze_horizons(),
            "time_analysis": self._analyze_time_performance(),
            "symbol_analysis": self._analyze_symbols(),
            "recommendations": self._generate_recommendations()
        }
        
        return analysis
    
    def _analyze_confidence(self) -> Dict[str, Any]:
        """分析不同置信度级别的性能
        
        Returns:
            Dict[str, Any]: 置信度分析结果
        """
        confidence_data = self.evaluation_summary["by_confidence"]
        
        # 按照置信度排序
        sorted_confidence = sorted(confidence_data.items(), key=lambda x: float(x[0]))
        
        # 准备结果
        confidence_analysis = {
            "thresholds": [],
            "accuracy_by_threshold": [],
            "count_by_threshold": [],
            "success_rate_by_threshold": [],
            "optimal_threshold": 0.5,
            "optimal_accuracy": 0.0
        }
        
        # 分析每个置信度阈值
        for conf, data in sorted_confidence:
            confidence_level = float(conf)
            
            confidence_analysis["thresholds"].append(confidence_level)
            confidence_analysis["accuracy_by_threshold"].append(data["accuracy"])
            confidence_analysis["count_by_threshold"].append(data["total"])
            confidence_analysis["success_rate_by_threshold"].append(data["successful"] / data["total"] if data["total"] > 0 else 0)
            
            # 更新最佳阈值
            if data["accuracy"] > confidence_analysis["optimal_accuracy"] and data["total"] >= 5:
                confidence_analysis["optimal_accuracy"] = data["accuracy"]
                confidence_analysis["optimal_threshold"] = confidence_level
                
        # 计算置信度与准确性的相关性
        if len(confidence_analysis["thresholds"]) > 1:
            try:
                correlation = np.corrcoef(
                    confidence_analysis["thresholds"], 
                    confidence_analysis["accuracy_by_threshold"]
                )[0, 1]
                confidence_analysis["confidence_accuracy_correlation"] = correlation
            except:
                confidence_analysis["confidence_accuracy_correlation"] = 0
                
        # 分析置信度校准
        confidence_analysis["calibration"] = []
        
        for i, conf in enumerate(confidence_analysis["thresholds"]):
            accuracy = confidence_analysis["accuracy_by_threshold"][i]
            calibration_error = abs(conf - accuracy)
            
            calibration_data = {
                "confidence": conf,
                "actual_accuracy": accuracy,
                "calibration_error": calibration_error,
                "is_overconfident": conf > accuracy
            }
            
            confidence_analysis["calibration"].append(calibration_data)
            
        # 计算平均校准误差
        if confidence_analysis["calibration"]:
            avg_error = sum(c["calibration_error"] for c in confidence_analysis["calibration"]) / len(confidence_analysis["calibration"])
            confidence_analysis["average_calibration_error"] = avg_error
            
        return confidence_analysis
    
    def _analyze_models(self) -> Dict[str, Any]:
        """分析不同模型的性能
        
        Returns:
            Dict[str, Any]: 模型分析结果
        """
        model_data = self.evaluation_summary["by_model"]
        
        # 按准确性排序
        sorted_models = sorted(
            model_data.items(), 
            key=lambda x: x[1]["accuracy"], 
            reverse=True
        )
        
        # 准备结果
        model_analysis = {
            "models": [],
            "accuracy_by_model": [],
            "count_by_model": [],
            "best_model": None,
            "worst_model": None,
            "model_details": {}
        }
        
        # 分析每个模型
        for model_name, data in sorted_models:
            model_analysis["models"].append(model_name)
            model_analysis["accuracy_by_model"].append(data["accuracy"])
            model_analysis["count_by_model"].append(data["total"])
            
            # 添加详细信息
            model_analysis["model_details"][model_name] = {
                "total_predictions": data["total"],
                "successful_predictions": data["successful"],
                "accuracy": data["accuracy"],
                "rank": len(model_analysis["models"])
            }
            
        # 设置最佳和最差模型
        if sorted_models:
            model_analysis["best_model"] = sorted_models[0][0]
            model_analysis["worst_model"] = sorted_models[-1][0]
            
        # 更新模型性能缓存
        for model_name, data in model_data.items():
            if model_name not in self.model_performance:
                self.model_performance[model_name] = {
                    "accuracy_history": [],
                    "prediction_count": []
                }
                
            self.model_performance[model_name]["accuracy_history"].append(data["accuracy"])
            self.model_performance[model_name]["prediction_count"].append(data["total"])
            
            # 限制历史长度
            max_history = 100
            if len(self.model_performance[model_name]["accuracy_history"]) > max_history:
                self.model_performance[model_name]["accuracy_history"] = self.model_performance[model_name]["accuracy_history"][-max_history:]
                self.model_performance[model_name]["prediction_count"] = self.model_performance[model_name]["prediction_count"][-max_history:]
                
        # 分析模型性能趋势
        for model_name in model_analysis["models"]:
            if model_name in self.model_performance and len(self.model_performance[model_name]["accuracy_history"]) >= 3:
                # 计算性能趋势（正值表示提高，负值表示下降）
                history = self.model_performance[model_name]["accuracy_history"]
                if len(history) >= 10:
                    # 使用线性回归计算趋势
                    x = np.arange(len(history))
                    slope, _ = np.polyfit(x, history, 1)
                    trend = slope * 100  # 转换为每100个预测的变化
                else:
                    # 简单计算最近几次的平均变化
                    recent = history[-3:]
                    changes = [recent[i] - recent[i-1] for i in range(1, len(recent))]
                    trend = sum(changes) / len(changes) * 100
                
                model_analysis["model_details"][model_name]["trend"] = trend
                
        return model_analysis
    
    def _analyze_horizons(self) -> Dict[str, Any]:
        """分析不同预测周期的性能
        
        Returns:
            Dict[str, Any]: 周期分析结果
        """
        horizon_data = self.evaluation_summary["by_horizon"]
        
        # 按周期排序
        sorted_horizons = sorted(
            horizon_data.items(), 
            key=lambda x: int(x[0])
        )
        
        # 准备结果
        horizon_analysis = {
            "horizons": [],
            "accuracy_by_horizon": [],
            "count_by_horizon": [],
            "optimal_horizon": None,
            "horizon_accuracy_trend": None
        }
        
        # 分析每个周期
        for horizon_str, data in sorted_horizons:
            horizon = int(horizon_str)
            
            horizon_analysis["horizons"].append(horizon)
            horizon_analysis["accuracy_by_horizon"].append(data["accuracy"])
            horizon_analysis["count_by_horizon"].append(data["total"])
            
        # 设置最佳周期（如果有足够数据）
        if sorted_horizons:
            # 找到具有至少5个预测且准确率最高的周期
            valid_horizons = [(h, d) for h, d in sorted_horizons if d["total"] >= 5]
            if valid_horizons:
                best_horizon = max(valid_horizons, key=lambda x: x[1]["accuracy"])
                horizon_analysis["optimal_horizon"] = int(best_horizon[0])
                
        # 计算周期与准确性的关系
        if len(horizon_analysis["horizons"]) > 1:
            try:
                # 使用线性回归分析趋势
                x = np.array(horizon_analysis["horizons"])
                y = np.array(horizon_analysis["accuracy_by_horizon"])
                
                slope, intercept = np.polyfit(x, y, 1)
                horizon_analysis["horizon_accuracy_trend"] = slope
                
                # 分析是短期还是长期预测更准确
                if slope > 0.001:
                    horizon_analysis["accuracy_trend"] = "long_term_better"
                elif slope < -0.001:
                    horizon_analysis["accuracy_trend"] = "short_term_better"
                else:
                    horizon_analysis["accuracy_trend"] = "neutral"
            except:
                horizon_analysis["horizon_accuracy_trend"] = 0
                horizon_analysis["accuracy_trend"] = "unknown"
                
        return horizon_analysis
    
    def _analyze_time_performance(self) -> Dict[str, Any]:
        """分析时间维度的性能变化
        
        Returns:
            Dict[str, Any]: 时间性能分析
        """
        time_data = self.evaluation_summary["by_time"]
        
        # 按日期排序
        sorted_dates = sorted(time_data.items())
        
        # 准备结果
        time_analysis = {
            "dates": [],
            "accuracy_by_date": [],
            "count_by_date": [],
            "recent_trend": None
        }
        
        # 分析每个日期
        for date_str, data in sorted_dates:
            time_analysis["dates"].append(date_str)
            time_analysis["accuracy_by_date"].append(data["accuracy"])
            time_analysis["count_by_date"].append(data["total"])
            
        # 计算最近的趋势
        if len(time_analysis["dates"]) >= 7:
            recent_accuracy = time_analysis["accuracy_by_date"][-7:]
            
            try:
                # 使用线性回归分析趋势
                x = np.arange(len(recent_accuracy))
                slope, _ = np.polyfit(x, recent_accuracy, 1)
                
                # 转换为每天的变化百分比
                trend = slope * 100
                time_analysis["recent_trend"] = trend
                
                # 判断趋势方向
                if trend > 0.5:
                    time_analysis["trend_direction"] = "improving"
                elif trend < -0.5:
                    time_analysis["trend_direction"] = "deteriorating"
                else:
                    time_analysis["trend_direction"] = "stable"
            except:
                time_analysis["recent_trend"] = 0
                time_analysis["trend_direction"] = "unknown"
                
        # 寻找表现最好和最差的日子
        if sorted_dates:
            # 找到至少有5个预测的日子
            valid_dates = [(d, data) for d, data in sorted_dates if data["total"] >= 5]
            
            if valid_dates:
                best_date = max(valid_dates, key=lambda x: x[1]["accuracy"])
                worst_date = min(valid_dates, key=lambda x: x[1]["accuracy"])
                
                time_analysis["best_date"] = {
                    "date": best_date[0],
                    "accuracy": best_date[1]["accuracy"],
                    "predictions": best_date[1]["total"]
                }
                
                time_analysis["worst_date"] = {
                    "date": worst_date[0],
                    "accuracy": worst_date[1]["accuracy"],
                    "predictions": worst_date[1]["total"]
                }
                
        return time_analysis
    
    def _analyze_symbols(self) -> Dict[str, Any]:
        """分析不同交易品种的性能
        
        Returns:
            Dict[str, Any]: 交易品种分析
        """
        symbol_data = self.evaluation_summary["by_symbol"]
        
        # 按准确性排序
        sorted_symbols = sorted(
            symbol_data.items(), 
            key=lambda x: x[1]["accuracy"], 
            reverse=True
        )
        
        # 准备结果
        symbol_analysis = {
            "symbols": [],
            "accuracy_by_symbol": [],
            "count_by_symbol": [],
            "best_symbol": None,
            "worst_symbol": None,
            "symbol_details": {}
        }
        
        # 分析每个交易品种
        for symbol, data in sorted_symbols:
            symbol_analysis["symbols"].append(symbol)
            symbol_analysis["accuracy_by_symbol"].append(data["accuracy"])
            symbol_analysis["count_by_symbol"].append(data["total"])
            
            # 添加详细信息
            symbol_analysis["symbol_details"][symbol] = {
                "total_predictions": data["total"],
                "successful_predictions": data["successful"],
                "accuracy": data["accuracy"],
                "rank": len(symbol_analysis["symbols"])
            }
            
        # 设置最佳和最差交易品种（如果有足够数据）
        if sorted_symbols:
            # 找到至少有5个预测的品种
            valid_symbols = [(s, d) for s, d in sorted_symbols if d["total"] >= 5]
            
            if valid_symbols:
                symbol_analysis["best_symbol"] = valid_symbols[0][0]
                symbol_analysis["worst_symbol"] = valid_symbols[-1][0]
                
        return symbol_analysis
    
    def _generate_recommendations(self) -> Dict[str, Any]:
        """生成改进建议
        
        Returns:
            Dict[str, Any]: 改进建议
        """
        recommendations = {
            "confidence_threshold": None,
            "best_models": [],
            "optimal_horizons": [],
            "areas_for_improvement": [],
            "parameter_suggestions": {}
        }
        
        # 置信度阈值建议
        conf_analysis = self._analyze_confidence()
        if "optimal_threshold" in conf_analysis:
            recommendations["confidence_threshold"] = conf_analysis["optimal_threshold"]
            
        # 模型建议
        model_analysis = self._analyze_models()
        if "model_details" in model_analysis:
            # 找出表现最好的模型
            good_models = [
                name for name, details in model_analysis["model_details"].items()
                if details["total_predictions"] >= 5 and details["accuracy"] >= 0.6
            ]
            recommendations["best_models"] = good_models
            
            # 模型改进建议
            for name, details in model_analysis["model_details"].items():
                if details["total_predictions"] >= 10 and details["accuracy"] < 0.5:
                    recommendations["areas_for_improvement"].append(f"Improve model: {name}")
                    
        # 预测周期建议
        horizon_analysis = self._analyze_horizons()
        if "optimal_horizon" in horizon_analysis and horizon_analysis["optimal_horizon"]:
            recommendations["optimal_horizons"].append(horizon_analysis["optimal_horizon"])
            
        # 校准建议
        if "average_calibration_error" in conf_analysis:
            if conf_analysis["average_calibration_error"] > 0.2:
                recommendations["areas_for_improvement"].append("Improve confidence calibration")
                
                # 判断是过于自信还是过于保守
                overconfident_count = sum(1 for c in conf_analysis["calibration"] if c.get("is_overconfident", False))
                total_calibrations = len(conf_analysis["calibration"])
                
                if total_calibrations > 0:
                    overconfident_ratio = overconfident_count / total_calibrations
                    
                    if overconfident_ratio > 0.7:
                        recommendations["areas_for_improvement"].append("Models are overconfident, reduce confidence values")
                    elif overconfident_ratio < 0.3:
                        recommendations["areas_for_improvement"].append("Models are underconfident, increase confidence values")
                        
        # 生成参数建议
        self.parameter_suggestions = {
            "confidence_threshold": recommendations["confidence_threshold"] if recommendations["confidence_threshold"] else 0.6,
            "preferred_models": recommendations["best_models"],
            "optimal_horizons": recommendations["optimal_horizons"]
        }
        
        recommendations["parameter_suggestions"] = self.parameter_suggestions
        
        return recommendations
    
    def get_summary(self) -> Dict[str, Any]:
        """获取评估摘要
        
        Returns:
            Dict[str, Any]: 评估摘要
        """
        # 基本摘要
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_predictions_evaluated": self.evaluation_summary["total_predictions"],
            "successful_predictions": self.evaluation_summary["successful_predictions"],
            "overall_accuracy": self.evaluation_summary["overall_accuracy"],
            "success_rate": self.evaluation_summary["successful_predictions"] / self.evaluation_summary["total_predictions"] if self.evaluation_summary["total_predictions"] > 0 else 0
        }
        
        # 添加最佳模型信息
        model_analysis = self._analyze_models()
        if "best_model" in model_analysis and model_analysis["best_model"]:
            summary["best_model"] = model_analysis["best_model"]
            summary["best_model_accuracy"] = model_analysis["model_details"][model_analysis["best_model"]]["accuracy"]
            
        # 添加最佳置信度阈值
        conf_analysis = self._analyze_confidence()
        if "optimal_threshold" in conf_analysis:
            summary["optimal_confidence_threshold"] = conf_analysis["optimal_threshold"]
            
        # 添加最近性能趋势
        time_analysis = self._analyze_time_performance()
        if "recent_trend" in time_analysis and time_analysis["recent_trend"] is not None:
            summary["recent_performance_trend"] = time_analysis["recent_trend"]
            summary["trend_direction"] = time_analysis.get("trend_direction", "unknown")
            
        # 添加参数建议
        if self.parameter_suggestions:
            summary["parameter_suggestions"] = self.parameter_suggestions
            
        return summary
    
    def reset_statistics(self) -> Dict[str, Any]:
        """重置评估统计
        
        Returns:
            Dict[str, Any]: 重置结果
        """
        # 保存旧的统计信息
        old_stats = {
            "total_predictions": self.evaluation_summary["total_predictions"],
            "successful_predictions": self.evaluation_summary["successful_predictions"],
            "overall_accuracy": self.evaluation_summary["overall_accuracy"]
        }
        
        # 重置评估历史
        if self.config["store_all_evaluations"]:
            self.evaluation_history = []
            
        # 重置评估摘要
        self.evaluation_summary = {
            "total_predictions": 0,
            "successful_predictions": 0,
            "overall_accuracy": 0.0,
            "price_accuracy": 0.0,
            "trend_accuracy": 0.0,
            "by_confidence": {},
            "by_model": {},
            "by_horizon": {},
            "by_time": {},
            "by_symbol": {}
        }
        
        # 保留模型性能历史
        # self.model_performance = {}
        
        # 重置参数建议
        self.parameter_suggestions = {}
        
        self.logger.info("评估统计已重置")
        
        # 创建重置结果
        reset_result = {
            "success": True,
            "previous_stats": old_stats,
            "reset_time": datetime.now().isoformat()
        }
        
        return reset_result 