import os
import platform
import socket
import traceback
import numpy as np
import pandas as pd

class AdvancedValidator:
    def __init__(self, logger):
        self.logger = logger
        self.test_results = {}
        self.recommendations = []
        self.validation_passed = True

    def run_validation(self):
        """执行全面验证"""
        print("\n" + "=" * 80)
        print(f"{'超神量子共生系统 - 高级验证测试':^80}")
        print("=" * 80 + "\n")
        
        self.logger.info("开始高级系统验证")
        
        try:
            # 1. 进行数据流一致性测试
            self.test_data_consistency()
            
            # 2. 测试核心组件交互
            self.test_component_interactions()
            
            # 3. 压力测试
            self.run_stress_test()
            
            # 4. 安全性测试
            self.test_security()
            
            # 5. 系统兼容性测试
            self.test_system_compatibility()
            
            # 6. 数据一致性深度验证
            self.deep_data_consistency_test()
            
            # 7. 生成报告
            self.generate_report()
            
        except Exception as e:
            self.logger.error(f"验证过程中发生错误: {str(e)}")
            traceback.print_exc()
            
        return self.validation_passed
        
    def test_system_compatibility(self):
        """测试系统兼容性"""
        self.logger.info("测试系统兼容性...")
        print("\n测试系统兼容性...")
        
        # 检测操作系统兼容性
        try:
            system = platform.system()
            
            self.logger.info(f"当前操作系统: {system}")
            self.test_results["operating_system"] = system
            
            if system not in ["Windows", "Linux", "Darwin"]:
                self.logger.warning(f"未知操作系统类型: {system}，可能存在兼容性问题")
                self.test_results["os_compatibility"] = "未知系统类型"
                self.recommendations.append("在主流操作系统(Windows/Linux/MacOS)上测试系统")
            else:
                self.test_results["os_compatibility"] = "兼容"
                
            # 检测Python版本兼容性
            python_version = platform.python_version()
            self.test_results["python_version"] = python_version
            
            major, minor, _ = python_version.split(".")
            if int(major) < 3 or (int(major) == 3 and int(minor) < 7):
                self.logger.warning(f"Python版本过低: {python_version}，推荐使用Python 3.7+")
                self.test_results["python_compatibility"] = "版本过低"
                self.recommendations.append("升级到Python 3.7或更高版本")
            else:
                self.test_results["python_compatibility"] = "兼容"
                
            # 检测显示设备兼容性（对图形界面重要）
            if system == "Linux":
                try:
                    display = os.environ.get('DISPLAY')
                    if not display:
                        self.logger.warning("Linux环境下未检测到DISPLAY环境变量，图形界面可能无法启动")
                        self.test_results["display_compatibility"] = "可能不兼容"
                        self.recommendations.append("在Linux下确保设置了DISPLAY环境变量")
                    else:
                        self.test_results["display_compatibility"] = "兼容"
                except Exception as e:
                    self.logger.error(f"检查显示设备时出错: {str(e)}")
            
            # 检测PyQt版本兼容性
            try:
                from PyQt5.QtCore import QT_VERSION_STR
                qt_version = QT_VERSION_STR
                self.test_results["qt_version"] = qt_version
                
                if qt_version < "5.12":
                    self.logger.warning(f"Qt版本较旧: {qt_version}，可能影响UI渲染")
                    self.test_results["qt_compatibility"] = "版本较旧"
                    self.recommendations.append("考虑升级到Qt 5.12或更高版本以获得更好的UI体验")
                else:
                    self.test_results["qt_compatibility"] = "兼容"
            except ImportError:
                self.logger.warning("未安装PyQt5，图形界面功能将不可用")
                self.test_results["qt_compatibility"] = "未安装"
                self.recommendations.append("安装PyQt5以启用图形界面功能")
            except Exception as e:
                self.logger.error(f"检查Qt版本时出错: {str(e)}")
                
            # 检测网络连接性（对数据获取重要）
            try:
                host = "www.tushare.pro"
                port = 80
                socket.setdefaulttimeout(3)
                socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
                self.test_results["network_connectivity"] = "正常"
                self.logger.info("网络连接正常")
            except Exception as e:
                self.logger.warning(f"网络连接测试失败: {str(e)}")
                self.test_results["network_connectivity"] = "异常"
                self.recommendations.append("检查网络连接，确保能访问数据源服务器")
                
        except Exception as e:
            self.logger.error(f"系统兼容性测试发生错误: {str(e)}")
            self.test_results["system_compatibility_test"] = f"错误: {str(e)}"
            
    def deep_data_consistency_test(self):
        """数据一致性深度验证"""
        self.logger.info("执行数据一致性深度验证...")
        print("\n执行数据一致性深度验证...")
        
        try:
            # 创建一个随机的测试向量
            test_vector = {
                "price": 100.0 + np.random.random() * 10,
                "volume": np.random.random(),
                "momentum": np.random.random() * 0.2,
                "volatility": np.random.random() * 0.1,
                "trend": np.random.random() * 0.3 - 0.15,
                "oscillator": np.random.random() * 0.4,
                "sentiment": np.random.random() * 0.4 - 0.2,
                "liquidity": np.random.random() * 0.7,
                "correlation": np.random.random() * 0.3,
                "divergence": np.random.random() * 0.1,
                "cycle_phase": np.random.random() * 0.2
            }
            
            # 记录原始值用于后续比较
            original_values = test_vector.copy()
            
            # 1. 通过多次循环转换测试数据一致性
            self.logger.info("测试数据转换的一致性...")
            
            # 检查代码中是否存在各种转换器
            transformers = []
            
            # 检查是否存在量子维度扩展器
            if os.path.exists("quantum_dimension_expander.py"):
                try:
                    from quantum_dimension_expander import QuantumDimensionExpander
                    transformers.append(("QuantumDimensionExpander", QuantumDimensionExpander()))
                except ImportError:
                    self.logger.warning("无法导入QuantumDimensionExpander")
            
            # 检查是否存在市场状态预测器
            if os.path.exists("market_state_predictor.py"):
                try:
                    from market_state_predictor import MarketStatePredictor
                    transformers.append(("MarketStatePredictor", MarketStatePredictor()))
                except ImportError:
                    self.logger.warning("无法导入MarketStatePredictor")
            
            # 检查是否存在数据标准化器
            if os.path.exists("data_normalizer.py"):
                try:
                    from data_normalizer import DataNormalizer
                    transformers.append(("DataNormalizer", DataNormalizer()))
                except ImportError:
                    self.logger.warning("无法导入DataNormalizer")
            
            # 对于找到的每个转换器，执行循环测试
            for name, transformer in transformers:
                self.logger.info(f"测试 {name} 的数据一致性...")
                
                # 根据转换器类型调用相应的方法
                try:
                    current_data = test_vector.copy()
                    
                    # 执行5次循环转换
                    for i in range(5):
                        # 根据转换器类型调用不同的方法
                        if name == "QuantumDimensionExpander":
                            expanded = transformer.expand_dimensions(current_data)
                            current_data = transformer.collapse_dimensions(expanded)
                        elif name == "MarketStatePredictor":
                            # 假设predictor有process和revert方法
                            processed = transformer.process(current_data)
                            current_data = transformer.revert(processed)
                        elif name == "DataNormalizer":
                            # 假设normalizer有normalize和denormalize方法
                            normalized = transformer.normalize(current_data)
                            current_data = transformer.denormalize(normalized)
                        else:
                            # 默认假设有transform和inverse_transform方法
                            if hasattr(transformer, "transform") and hasattr(transformer, "inverse_transform"):
                                transformed = transformer.transform(current_data)
                                current_data = transformer.inverse_transform(transformed)
                    
                    # 计算与原始数据的差异
                    deviations = {}
                    max_deviation = 0.0
                    for key in original_values:
                        if key in current_data:
                            deviation = abs(original_values[key] - current_data[key])
                            deviations[key] = deviation
                            max_deviation = max(max_deviation, deviation)
                    
                    if max_deviation > 0.1:  # 允许10%的误差
                        self.logger.warning(f"{name} 循环测试出现较大偏差: {max_deviation:.4f}")
                        self.test_results[f"{name}_consistency"] = f"偏差过大: {max_deviation:.4f}"
                        self.recommendations.append(f"检查 {name} 的正向和反向转换逻辑，减少循环累积误差")
                    else:
                        self.logger.info(f"{name} 循环测试通过，最大偏差: {max_deviation:.4f}")
                        self.test_results[f"{name}_consistency"] = f"通过，偏差: {max_deviation:.4f}"
                
                except Exception as e:
                    self.logger.error(f"测试 {name} 时出错: {str(e)}")
                    self.test_results[f"{name}_consistency"] = f"错误: {str(e)}"
            
            # 2. 测试数据管道的端到端一致性
            self.logger.info("测试数据处理管道的端到端一致性...")
            
            # 读取一个小的真实数据集
            try:
                if os.path.exists("tushare_data_connector.py"):
                    from tushare_data_connector import TushareDataConnector
                    connector = TushareDataConnector()
                    if hasattr(connector, 'get_index_data') or hasattr(connector, 'get_market_data'):
                        # 获取样本数据
                        sample_data = None
                        if hasattr(connector, 'get_index_data'):
                            sample_data = connector.get_index_data("000001.SH", limit=10)
                        else:
                            sample_data = connector.get_market_data("000001.SH", limit=10)
                        
                        if sample_data is not None and not sample_data.empty:
                            # 检查数据处理过程中的一致性
                            try:
                                # 1. 检查复制操作是否安全
                                df_copy = sample_data.copy()
                                is_copy_identical = df_copy.equals(sample_data)
                                self.test_results["dataframe_copy_safety"] = "安全" if is_copy_identical else "不安全"
                                
                                # 2. 检查转换和还原后的数据一致性
                                # 例如日期转换
                                if isinstance(sample_data.index, pd.DatetimeIndex):
                                    str_dates = sample_data.index.strftime('%Y-%m-%d').tolist()
                                    back_to_dates = pd.to_datetime(str_dates)
                                    is_dates_consistent = all(d1 == d2.date() for d1, d2 in zip(sample_data.index.date, back_to_dates))
                                    self.test_results["date_conversion_consistency"] = "一致" if is_dates_consistent else "不一致"
                                
                                # 3. 数值计算精度测试
                                # 计算简单统计并验证是否符合预期
                                sample_mean = sample_data['close'].mean()
                                parts_mean = pd.concat([sample_data['close'].iloc[:5], sample_data['close'].iloc[5:]]).mean()
                                mean_diff = abs(sample_mean - parts_mean)
                                
                                if mean_diff > 1e-10:
                                    self.logger.warning(f"数值计算精度可能存在问题，差异: {mean_diff}")
                                    self.test_results["numerical_precision"] = f"偏差: {mean_diff}"
                                    self.recommendations.append("检查数值计算方法，可能存在精度问题")
                                else:
                                    self.test_results["numerical_precision"] = "良好"
                                
                            except Exception as e:
                                self.logger.error(f"数据处理一致性测试出错: {str(e)}")
                                self.test_results["data_processing_consistency"] = f"错误: {str(e)}"
                        else:
                            self.logger.warning("未能获取样本数据进行深度一致性测试")
                            self.test_results["data_processing_consistency"] = "跳过，无样本数据"
            except ImportError:
                self.logger.warning("无法导入数据连接器进行深度一致性测试")
                self.test_results["data_processing_consistency"] = "跳过，无法导入连接器"
            except Exception as e:
                self.logger.error(f"数据处理一致性测试出错: {str(e)}")
                self.test_results["data_processing_consistency"] = f"错误: {str(e)}"
                
        except Exception as e:
            self.logger.error(f"数据一致性深度验证出错: {str(e)}")
            self.test_results["deep_data_consistency"] = f"错误: {str(e)}"