#!/usr/bin/env python3
"""
超神量子共生系统 - 增强版验证工具
专注于深度数据一致性和系统兼容性测试
"""

import os
import sys
import logging
import platform
import socket
import traceback
import numpy as np
import pandas as pd
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("enhanced_validation.log", 'a')
    ]
)

logger = logging.getLogger("EnhancedValidator")

class EnhancedValidator:
    """增强版系统验证器"""
    
    def __init__(self):
        self.logger = logger
        self.test_results = {}
        self.validation_passed = True
        self.critical_errors = []
        self.recommendations = []
        
    def run_validation(self):
        """执行增强版验证"""
        print("\n" + "=" * 80)
        print(f"{'超神量子共生系统 - 增强版验证测试':^80}")
        print("=" * 80 + "\n")
        
        self.logger.info("开始增强版系统验证")
        
        try:
            # 1. 系统兼容性测试
            self.test_system_compatibility()
            
            # 2. 数据一致性深度验证
            self.test_data_consistency()
            
            # 3. 模块间交互验证
            self.test_module_interactions()
            
            # 4. 生成报告
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
    
    def test_data_consistency(self):
        """测试数据一致性"""
        self.logger.info("测试数据一致性...")
        print("\n测试数据一致性...")
        
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
            
            # 测试量子维度扩展器的数据一致性
            if os.path.exists("quantum_dimension_expander.py"):
                try:
                    from quantum_dimension_expander import QuantumDimensionExpander
                    expander = QuantumDimensionExpander()
                    
                    # 执行5次循环扩展与折叠
                    current_data = test_vector.copy()
                    for i in range(5):
                        expanded = expander.expand_dimensions(current_data)
                        current_data = expander.collapse_dimensions(expanded)
                    
                    # 计算偏差
                    deviations = {}
                    max_deviation = 0.0
                    for key in original_values:
                        if key in current_data:
                            deviation = abs(original_values[key] - current_data[key])
                            deviations[key] = deviation
                            max_deviation = max(max_deviation, deviation)
                    
                    if max_deviation > 0.1:  # 允许10%的误差
                        self.logger.warning(f"量子维度扩展器循环测试出现较大偏差: {max_deviation:.4f}")
                        self.test_results["quantum_dimension_consistency"] = f"偏差过大: {max_deviation:.4f}"
                        self.recommendations.append("检查量子维度扩展器的正向和反向转换逻辑，减少循环累积误差")
                    else:
                        self.logger.info(f"量子维度扩展器循环测试通过，最大偏差: {max_deviation:.4f}")
                        self.test_results["quantum_dimension_consistency"] = f"通过，偏差: {max_deviation:.4f}"
                
                except Exception as e:
                    self.logger.error(f"测试量子维度扩展器时出错: {str(e)}")
                    self.test_results["quantum_dimension_consistency"] = f"错误: {str(e)}"
            
            # 测试数据连接器和数据格式一致性
            if os.path.exists("tushare_data_connector.py"):
                try:
                    from tushare_data_connector import TushareDataConnector
                    connector = TushareDataConnector()
                    
                    # 测试不同获取方法的数据一致性
                    if hasattr(connector, 'get_index_data') and hasattr(connector, 'get_daily_data'):
                        index_data = connector.get_index_data("000001.SH")
                        daily_data = connector.get_daily_data("000001.SH")
                        
                        # 检查两种方法获取的数据是否一致
                        if index_data is not None and not index_data.empty and daily_data is not None and not daily_data.empty:
                            common_columns = set(index_data.columns).intersection(set(daily_data.columns))
                            if common_columns:
                                is_consistent = True
                                for col in common_columns:
                                    if not index_data[col].equals(daily_data[col]):
                                        is_consistent = False
                                        break
                                
                                self.test_results["data_methods_consistency"] = "一致" if is_consistent else "不一致"
                                if not is_consistent:
                                    self.logger.warning("不同数据获取方法返回的结果不一致")
                                    self.recommendations.append("检查不同数据获取方法的实现，确保结果一致性")
                            else:
                                self.logger.warning("数据方法返回的列不一致，无法比较")
                                self.test_results["data_methods_consistency"] = "无法比较"
                    
                    # 测试数据精度
                    if hasattr(connector, 'get_index_data'):
                        data = connector.get_index_data("000001.SH")
                        if data is not None and not data.empty:
                            # 检查数值计算精度
                            sample_mean = data['close'].mean()
                            parts_mean = pd.concat([data['close'].iloc[:5], data['close'].iloc[5:]]).mean()
                            mean_diff = abs(sample_mean - parts_mean)
                            
                            self.test_results["numerical_precision"] = f"偏差: {mean_diff:.10f}"
                            if mean_diff > 1e-10:
                                self.logger.warning(f"数值计算精度可能存在问题，差异: {mean_diff:.10f}")
                                self.recommendations.append("检查数值计算方法，可能存在精度问题")
                            
                            # 测试数据转换和格式化
                            if isinstance(data.index, pd.DatetimeIndex):
                                # 格式化日期后再转回
                                str_dates = data.index.strftime('%Y-%m-%d').tolist()
                                back_to_dates = pd.to_datetime(str_dates)
                                
                                # 检查是否一致（只比较日期部分）
                                is_consistent = all(d1.date() == d2.date() for d1, d2 in zip(data.index, back_to_dates))
                                self.test_results["date_conversion"] = "一致" if is_consistent else "不一致"
                                if not is_consistent:
                                    self.logger.warning("日期转换过程中数据不一致")
                                    self.recommendations.append("检查日期转换逻辑，确保数据一致性")
                
                except Exception as e:
                    self.logger.error(f"测试数据连接器一致性时出错: {str(e)}")
                    self.test_results["data_connector_consistency"] = f"错误: {str(e)}"
        
        except Exception as e:
            self.logger.error(f"数据一致性测试出错: {str(e)}")
            self.test_results["data_consistency"] = f"错误: {str(e)}"
    
    def test_module_interactions(self):
        """测试模块间交互"""
        self.logger.info("测试模块间交互...")
        print("\n测试模块间交互...")
        
        try:
            # 测试统一入口点和数据连接器的交互
            if os.path.exists("run_supergod_unified.py") and os.path.exists("tushare_data_connector.py"):
                try:
                    # 通过importlib动态导入，避免影响当前环境
                    import importlib.util
                    
                    # 导入统一入口点
                    spec_unified = importlib.util.spec_from_file_location("run_supergod_unified", "run_supergod_unified.py")
                    unified = importlib.util.module_from_spec(spec_unified)
                    spec_unified.loader.exec_module(unified)
                    
                    # 导入数据连接器
                    from tushare_data_connector import TushareDataConnector
                    
                    # 检查get_data_connector函数是否能正确返回连接器实例
                    if hasattr(unified, 'get_data_connector'):
                        connector = unified.get_data_connector()
                        
                        # 验证返回的是否为TushareDataConnector实例
                        is_correct_type = isinstance(connector, TushareDataConnector)
                        self.test_results["data_connector_integration"] = "正常" if is_correct_type else "类型不匹配"
                        
                        if not is_correct_type:
                            self.logger.warning("统一入口点返回的数据连接器类型不正确")
                            self.recommendations.append("检查get_data_connector函数的实现")
                        
                        # 测试连接器功能
                        if is_correct_type and hasattr(connector, 'connect'):
                            connection_result = connector.connect()
                            self.test_results["connector_function"] = "正常" if connection_result else "连接失败"
                            
                            if not connection_result:
                                self.logger.warning("数据连接器连接测试失败")
                                self.recommendations.append("检查数据连接器配置和API密钥")
                    else:
                        self.logger.warning("统一入口点缺少get_data_connector函数")
                        self.test_results["data_connector_integration"] = "缺少函数"
                        self.recommendations.append("在统一入口点实现get_data_connector函数")
                
                except Exception as e:
                    self.logger.error(f"测试统一入口点和数据连接器交互时出错: {str(e)}")
                    self.test_results["unified_connector_interaction"] = f"错误: {str(e)}"
            
            # 测试驾驶舱和增强模块的交互
            if os.path.exists("supergod_cockpit.py"):
                try:
                    # 检查是否能设置增强模块
                    import importlib.util
                    
                    # 导入驾驶舱模块
                    spec_cockpit = importlib.util.spec_from_file_location("supergod_cockpit", "supergod_cockpit.py")
                    cockpit_module = importlib.util.module_from_spec(spec_cockpit)
                    spec_cockpit.loader.exec_module(cockpit_module)
                    
                    # 测试是否有相关设置方法
                    if hasattr(cockpit_module, 'SupergodCockpit'):
                        cockpit_class = getattr(cockpit_module, 'SupergodCockpit')
                        
                        # 检查是否有设置增强模块的方法
                        has_enhancement_method = hasattr(cockpit_class, 'set_enhancement_modules')
                        self.test_results["cockpit_enhancement_integration"] = "支持" if has_enhancement_method else "不支持"
                        
                        if not has_enhancement_method:
                            self.logger.warning("驾驶舱不支持设置增强模块")
                            self.recommendations.append("在驾驶舱类中添加set_enhancement_modules方法")
                        
                        # 检查是否有设置数据连接器的方法
                        has_connector_method = hasattr(cockpit_class, 'set_data_connector')
                        self.test_results["cockpit_connector_integration"] = "支持" if has_connector_method else "不支持"
                        
                        if not has_connector_method:
                            self.logger.warning("驾驶舱不支持设置数据连接器")
                            self.recommendations.append("在驾驶舱类中添加set_data_connector方法")
                
                except Exception as e:
                    self.logger.error(f"测试驾驶舱和增强模块交互时出错: {str(e)}")
                    self.test_results["cockpit_enhancement_interaction"] = f"错误: {str(e)}"
        
        except Exception as e:
            self.logger.error(f"模块间交互测试出错: {str(e)}")
            self.test_results["module_interactions"] = f"错误: {str(e)}"
    
    def generate_report(self):
        """生成验证报告"""
        self.logger.info("生成验证报告...")
        print("\n生成验证报告...")
        
        # 汇总验证结果
        failing_tests = [k for k, v in self.test_results.items() if "失败" in str(v) or "错误" in str(v)]
        warning_tests = [k for k, v in self.test_results.items() if "警告" in str(v) or "不一致" in str(v)]
        
        # 更新验证结果
        if self.critical_errors:
            self.validation_passed = False
        elif failing_tests:
            self.validation_passed = False
        
        # 打印报告
        print("\n" + "=" * 80)
        print(f"{'超神量子共生系统 - 增强版验证测试报告':^80}")
        print("=" * 80)
        
        print(f"\n验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"验证结果: {'通过' if self.validation_passed else '失败'}")
        
        # 打印测试结果摘要
        print("\n测试结果摘要:")
        print("-" * 80)
        print(f"{'测试项目':<40}{'结果':<40}")
        print("-" * 80)
        
        for key, value in self.test_results.items():
            print(f"{key:<40}{str(value):<40}")
        
        # 打印关键错误
        if self.critical_errors:
            print("\n关键错误:")
            for error in self.critical_errors:
                print(f"- {error}")
        
        # 打印建议
        if self.recommendations:
            print("\n改进建议:")
            for recommendation in self.recommendations:
                print(f"- {recommendation}")
        
        print("\n" + "=" * 80)
        
        # 保存报告到文件
        report_file = f"supergod_enhanced_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"{'超神量子共生系统 - 增强版验证测试报告':^80}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"验证结果: {'通过' if self.validation_passed else '失败'}\n\n")
            
            f.write("测试结果摘要:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'测试项目':<40}{'结果':<40}\n")
            f.write("-" * 80 + "\n")
            
            for key, value in self.test_results.items():
                f.write(f"{key:<40}{str(value):<40}\n")
            
            if self.critical_errors:
                f.write("\n关键错误:\n")
                for error in self.critical_errors:
                    f.write(f"- {error}\n")
            
            if self.recommendations:
                f.write("\n改进建议:\n")
                for recommendation in self.recommendations:
                    f.write(f"- {recommendation}\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        self.logger.info(f"增强版验证报告已保存至: {report_file}")
        print(f"\n完整报告已保存至: {report_file}")
        
        return report_file

def main():
    """主函数"""
    print("开始执行增强版系统验证...")
    validator = EnhancedValidator()
    result = validator.run_validation()
    return 0 if result else 1

if __name__ == "__main__":
    sys.exit(main()) 