#!/usr/bin/env python3
"""
超神量子共生系统 - 简化版验证测试工具
用于执行系统验证和诊断
"""

import os
import sys
import time
import json
import logging
import importlib
import traceback
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("supergod_validator_fixed.log", 'a')
    ]
)

logger = logging.getLogger("SupergodValidator")

class SupergodValidator:
    """超神系统验证测试工具类"""
    
    def __init__(self):
        """初始化验证测试工具"""
        self.logger = logger
        self.test_results = []
        self.validation_passed = True
        
    def run_validation(self):
        """运行验证测试"""
        self.show_banner()
        self.logger.info("开始系统验证测试")
        
        try:
            # 检查环境依赖
            self.check_dependencies()
            
            # 检查项目结构
            self.check_project_structure()
            
            # 验证数据连接器
            self.check_data_connector()
            
            # 验证核心模块
            self.check_core_modules()
            
            # 验证UI模块
            self.check_ui_modules()
            
            # 输出结果报告
            self.generate_report()
            
            return self.validation_passed
            
        except Exception as e:
            self.logger.error(f"验证过程中发生错误: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.test_results.append(("系统验证", "失败", f"发生错误: {str(e)}"))
            self.validation_passed = False
            return False
    
    def show_banner(self):
        """显示验证测试工具横幅"""
        banner = """
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                                                                      ║
    ║               超神量子共生系统 - 验证测试工具                       ║
    ║                                                                      ║
    ║          系统诊断 · 模块验证 · 功能测试 · 优化建议                  ║
    ║                                                                      ║
    ╚══════════════════════════════════════════════════════════════════════╝
    
    开始时间: %s
    """ % datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        print(banner)
    
    def check_dependencies(self):
        """检查系统依赖"""
        self.logger.info("检查系统依赖...")
        print("检查系统依赖...")
        
        dependencies = {
            "basic": ["numpy", "pandas", "matplotlib"],
            "data": ["tushare", "akshare"],
            "ui": ["PyQt5"]
        }
        
        missing = []
        
        for category, modules in dependencies.items():
            for module in modules:
                try:
                    imported = importlib.import_module(module)
                    version = getattr(imported, "__version__", "未知版本")
                    self.logger.info(f"✓ 依赖 {module} ({version}) 已安装")
                    print(f"✓ 依赖 {module} ({version}) 已安装")
                except ImportError:
                    missing.append(module)
                    self.logger.warning(f"✗ 依赖 {module} 未安装")
                    print(f"✗ 依赖 {module} 未安装")
        
        if missing:
            self.test_results.append(("依赖检查", "警告", f"缺少依赖: {', '.join(missing)}"))
            if "numpy" in missing or "pandas" in missing:
                self.validation_passed = False
                self.logger.error("缺少关键依赖，系统无法运行")
                print("❌ 缺少关键依赖，系统无法运行")
        else:
            self.test_results.append(("依赖检查", "通过", "所有依赖已安装"))
            self.logger.info("所有依赖已安装")
            print("✅ 所有依赖已安装")
    
    def check_project_structure(self):
        """检查项目结构"""
        self.logger.info("检查项目结构...")
        print("\n检查项目结构...")
        
        required_files = [
            "run_supergod_unified.py",
            "supergod_cockpit.py",
            "tushare_data_connector.py"
        ]
        
        required_dirs = [
            "quantum_symbiotic_network",
            "quantum_prediction"
        ]
        
        structure_valid = True
        missing_items = []
        
        for file in required_files:
            if not os.path.isfile(file):
                structure_valid = False
                missing_items.append(f"文件: {file}")
                self.logger.warning(f"缺少必要文件: {file}")
                print(f"✗ 缺少必要文件: {file}")
                
        for directory in required_dirs:
            if not os.path.isdir(directory):
                structure_valid = False
                missing_items.append(f"目录: {directory}")
                self.logger.warning(f"缺少必要目录: {directory}")
                print(f"✗ 缺少必要目录: {directory}")
        
        if structure_valid:
            self.test_results.append(("项目结构", "通过", "所有必要文件和目录存在"))
            self.logger.info("项目结构检查通过")
            print("✅ 项目结构检查通过")
        else:
            self.validation_passed = False
            self.test_results.append(("项目结构", "失败", f"缺少项目文件/目录: {', '.join(missing_items)}"))
            self.logger.error("项目结构不完整")
            print("❌ 项目结构不完整")
    
    def check_data_connector(self):
        """检查数据连接器"""
        self.logger.info("检查数据连接器...")
        print("\n检查数据连接器...")
        
        try:
            # 尝试导入TushareDataConnector
            if os.path.exists("tushare_data_connector.py"):
                try:
                    spec = importlib.util.spec_from_file_location("tushare_data_connector", "tushare_data_connector.py")
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # 检查是否有连接器类
                    if hasattr(module, "TushareDataConnector"):
                        self.logger.info("数据连接器模块加载成功")
                        print("✅ 数据连接器模块加载成功")
                        
                        # 检查必要方法
                        connector_class = getattr(module, "TushareDataConnector")
                        connector = connector_class()
                        
                        required_methods = ["connect", "get_stock_data", "get_index_data", "update_bars"]
                        missing_methods = []
                        
                        for method in required_methods:
                            if not hasattr(connector, method):
                                missing_methods.append(method)
                                self.logger.warning(f"数据连接器缺少方法: {method}")
                                print(f"✗ 数据连接器缺少方法: {method}")
                        
                        if missing_methods:
                            self.test_results.append(("数据连接器", "警告", f"缺少方法: {', '.join(missing_methods)}"))
                        else:
                            self.test_results.append(("数据连接器", "通过", "数据连接器实现完整"))
                            self.logger.info("数据连接器实现完整")
                            print("✅ 数据连接器实现完整")
                    else:
                        self.logger.warning("数据连接器模块中未找到TushareDataConnector类")
                        print("✗ 数据连接器模块中未找到TushareDataConnector类")
                        self.test_results.append(("数据连接器", "失败", "未找到TushareDataConnector类"))
                        self.validation_passed = False
                        
                except Exception as e:
                    self.logger.error(f"导入数据连接器时出错: {str(e)}")
                    print(f"❌ 导入数据连接器时出错: {str(e)}")
                    self.test_results.append(("数据连接器", "失败", f"导入错误: {str(e)}"))
                    self.validation_passed = False
            else:
                self.logger.warning("未找到数据连接器文件 tushare_data_connector.py")
                print("✗ 未找到数据连接器文件 tushare_data_connector.py")
                self.test_results.append(("数据连接器", "失败", "未找到数据连接器文件"))
                self.validation_passed = False
                
        except Exception as e:
            self.logger.error(f"检查数据连接器时出错: {str(e)}")
            print(f"❌ 检查数据连接器时出错: {str(e)}")
            self.test_results.append(("数据连接器", "失败", f"检查错误: {str(e)}"))
            self.validation_passed = False
    
    def check_core_modules(self):
        """检查核心功能模块"""
        self.logger.info("检查核心功能模块...")
        print("\n检查核心功能模块...")
        
        core_modules = [
            ("量子共生网络", "quantum_symbiotic_network"),
            ("量子预测引擎", "quantum_prediction"),
            ("市场分析器", "market_analyzer.py"),
            ("交易信号生成器", "trading_signal_generator.py"),
            ("回测引擎", "backtest_engine.py")
        ]
        
        for module_name, module_path in core_modules:
            try:
                if module_path.endswith('.py'):
                    # 检查文件
                    if os.path.exists(module_path):
                        self.logger.info(f"✓ 核心模块 {module_name} 存在")
                        print(f"✓ 核心模块 {module_name} 存在")
                        self.test_results.append((f"核心模块 - {module_name}", "通过", "模块文件存在"))
                    else:
                        # 检查该文件是否在quantum_symbiotic_network目录下
                        alt_path = os.path.join("quantum_symbiotic_network", module_path)
                        if os.path.exists(alt_path):
                            self.logger.info(f"✓ 核心模块 {module_name} 存在于quantum_symbiotic_network目录")
                            print(f"✓ 核心模块 {module_name} 存在于quantum_symbiotic_network目录")
                            self.test_results.append((f"核心模块 - {module_name}", "通过", "模块文件存在"))
                        else:
                            self.logger.warning(f"✗ 核心模块 {module_name} 文件不存在")
                            print(f"✗ 核心模块 {module_name} 文件不存在")
                            self.test_results.append((f"核心模块 - {module_name}", "警告", "模块文件不存在"))
                else:
                    # 检查目录
                    if os.path.isdir(module_path):
                        self.logger.info(f"✓ 核心模块目录 {module_name} 存在")
                        print(f"✓ 核心模块目录 {module_name} 存在")
                        self.test_results.append((f"核心模块 - {module_name}", "通过", "模块目录存在"))
                    else:
                        self.logger.warning(f"✗ 核心模块目录 {module_name} 不存在")
                        print(f"✗ 核心模块目录 {module_name} 不存在")
                        self.test_results.append((f"核心模块 - {module_name}", "警告", "模块目录不存在"))
            except Exception as e:
                self.logger.error(f"检查核心模块 {module_name} 时出错: {str(e)}")
                print(f"❌ 检查核心模块 {module_name} 时出错: {str(e)}")
                self.test_results.append((f"核心模块 - {module_name}", "失败", f"检查错误: {str(e)}"))
                
        # 检测模块间的集成
        print("\n检查模块间集成...")
        try:
            # 尝试导入run_supergod_unified.py进行整体检测
            if os.path.exists("run_supergod_unified.py"):
                self.logger.info("开始验证统一入口点集成...")
                print("开始验证统一入口点集成...")
                
                spec = importlib.util.spec_from_file_location("run_supergod_unified", "run_supergod_unified.py")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # 检查是否有主函数和关键函数
                has_main = hasattr(module, "main")
                has_check_deps = hasattr(module, "check_dependencies")
                
                if has_main and has_check_deps:
                    self.logger.info("✓ 统一入口点接口完整")
                    print("✓ 统一入口点接口完整")
                    self.test_results.append(("模块集成 - 统一入口", "通过", "接口完整"))
                else:
                    missing = []
                    if not has_main: missing.append("main")
                    if not has_check_deps: missing.append("check_dependencies")
                    
                    self.logger.warning(f"✗ 统一入口点缺少关键函数: {', '.join(missing)}")
                    print(f"✗ 统一入口点缺少关键函数: {', '.join(missing)}")
                    self.test_results.append(("模块集成 - 统一入口", "警告", f"缺少关键函数: {', '.join(missing)}"))
            else:
                self.logger.warning("✗ 未找到统一入口点文件 run_supergod_unified.py")
                print("✗ 未找到统一入口点文件 run_supergod_unified.py")
                self.test_results.append(("模块集成 - 统一入口", "警告", "未找到统一入口点文件"))
                
        except Exception as e:
            self.logger.error(f"验证模块集成时出错: {str(e)}")
            print(f"❌ 验证模块集成时出错: {str(e)}")
            self.test_results.append(("模块集成", "失败", f"验证错误: {str(e)}"))
            
        # 检查量子维度扩展模块
        if os.path.exists("quantum_dimension_expander.py") or os.path.exists("quantum_symbiotic_network/quantum_dimension_expander.py"):
            self.logger.info("✓ 量子维度扩展模块存在")
            print("✓ 量子维度扩展模块存在")
            self.test_results.append(("增强模块 - 量子维度扩展", "通过", "模块存在"))
        else:
            self.logger.warning("⚠ 未找到量子维度扩展模块")
            print("⚠ 未找到量子维度扩展模块")
            self.test_results.append(("增强模块 - 量子维度扩展", "警告", "模块不存在"))
    
    def check_ui_modules(self):
        """检查UI模块"""
        self.logger.info("检查UI模块...")
        print("\n检查UI模块...")
        
        try:
            # 检查超神驾驶舱UI
            if os.path.exists("supergod_cockpit.py"):
                try:
                    spec = importlib.util.spec_from_file_location("supergod_cockpit", "supergod_cockpit.py")
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # 检查是否有UI类
                    if hasattr(module, "SupergodCockpit"):
                        self.logger.info("驾驶舱UI模块加载成功")
                        print("✅ 驾驶舱UI模块加载成功")
                        self.test_results.append(("驾驶舱UI", "通过", "模块加载正常"))
                    else:
                        self.logger.warning("驾驶舱UI模块中未找到SupergodCockpit类")
                        print("✗ 驾驶舱UI模块中未找到SupergodCockpit类")
                        self.test_results.append(("驾驶舱UI", "警告", "未找到SupergodCockpit类"))
                        
                except Exception as e:
                    self.logger.warning(f"导入驾驶舱UI模块时出错: {str(e)}")
                    print(f"✗ 导入驾驶舱UI模块时出错: {str(e)}")
                    self.test_results.append(("驾驶舱UI", "警告", f"导入错误: {str(e)}"))
            else:
                self.logger.warning("未找到驾驶舱UI文件 supergod_cockpit.py")
                print("✗ 未找到驾驶舱UI文件 supergod_cockpit.py")
                self.test_results.append(("驾驶舱UI", "警告", "未找到驾驶舱UI文件"))
                
        except Exception as e:
            self.logger.error(f"检查UI模块时出错: {str(e)}")
            print(f"❌ 检查UI模块时出错: {str(e)}")
            self.test_results.append(("UI模块", "失败", f"检查错误: {str(e)}"))
    
    def generate_report(self):
        """生成结果报告"""
        self.logger.info("生成结果报告...")
        print("\n生成结果报告...")
        
        # 格式化测试结果
        print("\n" + "=" * 80)
        print(f"{'超神量子共生系统验证测试报告':^80}")
        print("=" * 80)
        
        print(f"\n测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"测试结果: {'通过' if self.validation_passed else '失败'}")
        
        print("\n" + "-" * 80)
        print(f"{'测试项目':<30}{'结果':<10}{'说明':<40}")
        print("-" * 80)
        
        for item, result, message in self.test_results:
            result_symbol = "✓" if result == "通过" else "✗" if result == "失败" else "⚠"
            print(f"{item:<30}{result_symbol} {result:<8}{message:<40}")
        
        print("\n" + "=" * 80)
        
        # 保存报告到文件
        report_file = f"supergod_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "validation_passed": self.validation_passed,
            "test_results": [
                {"item": item, "result": result, "message": message}
                for item, result, message in self.test_results
            ]
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"验证测试报告已保存至: {report_file}")
        print(f"\n完整报告已保存至: {report_file}")


def main():
    """主函数"""
    print("开始执行简化版超神系统验证...")
    
    try:
        # 创建验证工具实例
        validator = SupergodValidator()
        
        # 执行验证测试
        result = validator.run_validation()
        
        # 返回测试结果
        return 0 if result else 1
        
    except Exception as e:
        print(f"验证器执行出错: {str(e)}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 