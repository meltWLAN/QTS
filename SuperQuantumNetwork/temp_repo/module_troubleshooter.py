#!/usr/bin/env python3
"""
超神系统 - 模块诊断和修复工具
用于检测和修复系统各个模块的问题
"""

import os
import sys
import time
import logging
import argparse
import importlib
import json
from datetime import datetime
import traceback

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("troubleshooter.log", 'a')
    ]
)

logger = logging.getLogger("ModuleTroubleshooter")

class ModuleTroubleshooter:
    """模块诊断和修复工具"""
    
    def __init__(self, args):
        """初始化诊断工具"""
        self.args = args
        self.modules = {}
        self.fixes = []
        self.errors = []
        
    def run(self):
        """运行诊断"""
        logger.info("开始超神系统诊断...")
        
        try:
            # 检查项目结构
            self.check_project_structure()
            
            # 检查Python环境
            self.check_python_environment()
            
            # 检查TuShare连接
            self.check_tushare()
            
            # 检查缓存目录
            self.check_cache_directories()
            
            # 检查核心模块
            self.check_core_modules()
            
            # 检查UI模块
            if not self.args.no_ui:
                self.check_ui_modules()
            
            # 应用修复
            if not self.args.no_fix and self.fixes:
                self.apply_fixes()
                
            # 打印结果
            self.print_result()
            
        except Exception as e:
            logger.error(f"诊断过程中出错: {str(e)}")
            logger.error(traceback.format_exc())
            print(f"\n❌ 诊断失败: {str(e)}")
            
    def check_project_structure(self):
        """检查项目结构"""
        logger.info("正在检查项目结构...")
        
        # 检查根目录
        required_dirs = ["gui", "quantum_symbiotic_network", "trading_signals", "market_symbiosis"]
        missing_dirs = []
        
        for dir_name in required_dirs:
            if not os.path.exists(dir_name):
                missing_dirs.append(dir_name)
                
        if missing_dirs:
            self.errors.append(f"缺少必要目录: {', '.join(missing_dirs)}")
            
        # 检查关键文件
        required_files = ["launch_supergod.py", "config.json"]
        missing_files = []
        
        for file_name in required_files:
            if not os.path.exists(file_name):
                missing_files.append(file_name)
                
        if missing_files:
            self.errors.append(f"缺少必要文件: {', '.join(missing_files)}")
            
        if not missing_dirs and not missing_files:
            logger.info("✅ 项目结构检查通过")
        else:
            logger.warning("⚠️ 项目结构检查未通过")
            
    def check_python_environment(self):
        """检查Python环境"""
        logger.info("正在检查Python环境...")
        
        # 检查必要的依赖库
        required_packages = ["numpy", "pandas", "tushare", "PyQt5"]
        missing_packages = []
        
        for package in required_packages:
            try:
                importlib.import_module(package)
            except ImportError:
                missing_packages.append(package)
                
        if missing_packages:
            self.errors.append(f"缺少必要的Python库: {', '.join(missing_packages)}")
            self.fixes.append({
                "type": "pip_install",
                "packages": missing_packages,
                "description": f"安装缺少的Python库: {', '.join(missing_packages)}"
            })
            logger.warning(f"⚠️ Python环境检查未通过, 缺少库: {', '.join(missing_packages)}")
        else:
            logger.info("✅ Python环境检查通过")
            
    def check_tushare(self):
        """检查TuShare连接"""
        logger.info("正在检查TuShare连接...")
        
        try:
            import tushare as ts
            
            # 从配置文件中读取token
            token = self._get_tushare_token()
            
            if not token:
                logger.warning("未找到TuShare token")
                self.errors.append("未找到有效的TuShare token")
                return
                
            # 设置token
            ts.set_token(token)
            
            # 尝试连接
            pro = ts.pro_api()
            
            # 测试查询
            df = pro.query('trade_cal', exchange='', start_date='20210101', end_date='20210110')
            
            if df is not None and not df.empty:
                logger.info("✅ TuShare连接成功")
                
                # 检查API权限
                try:
                    df = pro.query('tushare_token', token=token)
                    interfaces = list(df['interface_name']) if df is not None and not df.empty else []
                    logger.info(f"TuShare接口数量: {len(interfaces)}")
                except Exception as e:
                    logger.warning(f"无法获取TuShare接口列表: {str(e)}")
            else:
                self.errors.append("TuShare连接失败, API没有返回有效数据")
                logger.warning("⚠️ TuShare连接检查未通过")
        except Exception as e:
            logger.error(f"TuShare连接失败: {str(e)}")
            self.errors.append(f"TuShare连接失败: {str(e)}")
            
            # 添加修复方案
            self.fixes.append({
                "type": "fix_tushare",
                "description": "修复TuShare连接问题"
            })
            
    def check_cache_directories(self):
        """检查缓存目录"""
        logger.info("正在检查缓存目录...")
        
        # 确保缓存目录存在
        cache_dirs = [
            "quantum_symbiotic_network/cache",
            "trading_signals/cache",
            "market_symbiosis/cache"
        ]
        
        for cache_dir in cache_dirs:
            if not os.path.exists(cache_dir):
                logger.info(f"创建缓存目录: {cache_dir}")
                os.makedirs(cache_dir, exist_ok=True)
                
        logger.info("✅ 缓存目录检查完成")
        
    def check_core_modules(self):
        """检查核心模块"""
        logger.info("正在检查核心模块...")
        
        # 检查数据控制器
        try:
            from gui.controllers.data_controller import DataController
            self.modules["data_controller"] = True
            logger.info("✅ 数据控制器模块存在")
        except ImportError as e:
            self.modules["data_controller"] = False
            self.errors.append(f"数据控制器模块导入失败: {str(e)}")
            logger.error(f"❌ 数据控制器模块导入失败: {str(e)}")
            
        # 检查交易控制器
        try:
            from gui.controllers.trading_controller import TradingController
            self.modules["trading_controller"] = True
            logger.info("✅ 交易控制器模块存在")
        except ImportError as e:
            self.modules["trading_controller"] = False
            self.errors.append(f"交易控制器模块导入失败: {str(e)}")
            logger.error(f"❌ 交易控制器模块导入失败: {str(e)}")
            
        # 检查量子预测模块
        try:
            from quantum_symbiotic_network.quantum_prediction import QuantumSymbioticPredictor
            self.modules["quantum_predictor"] = True
            logger.info("✅ 量子预测模块存在")
        except ImportError as e:
            self.modules["quantum_predictor"] = False
            self.errors.append(f"量子预测模块导入失败: {str(e)}")
            logger.error(f"❌ 量子预测模块导入失败: {str(e)}")
            
        # 检查宇宙共振模块
        try:
            from cosmic_resonance import CosmicEngine
            self.modules["cosmic_engine"] = True
            logger.info("✅ 宇宙共振模块存在")
        except ImportError as e:
            self.modules["cosmic_engine"] = False
            self.errors.append(f"宇宙共振模块导入失败: {str(e)}")
            logger.error(f"❌ 宇宙共振模块导入失败: {str(e)}")
            
    def check_ui_modules(self):
        """检查UI模块"""
        logger.info("正在检查UI模块...")
        
        # 检查PyQt5
        try:
            from PyQt5.QtWidgets import QApplication, QMainWindow
            self.modules["pyqt5"] = True
            logger.info("✅ PyQt5模块存在")
        except ImportError as e:
            self.modules["pyqt5"] = False
            self.errors.append(f"PyQt5模块导入失败: {str(e)}")
            logger.error(f"❌ PyQt5模块导入失败: {str(e)}")
            
            # 添加修复方案
            self.fixes.append({
                "type": "pip_install",
                "packages": ["PyQt5"],
                "description": "安装PyQt5"
            })
            
        # 检查主窗口
        try:
            from gui.views.main_window import SuperTradingMainWindow
            self.modules["main_window"] = True
            logger.info("✅ 主窗口模块存在")
        except ImportError as e:
            self.modules["main_window"] = False
            self.errors.append(f"主窗口模块导入失败: {str(e)}")
            logger.error(f"❌ 主窗口模块导入失败: {str(e)}")
            
        # 检查市场视图
        try:
            from gui.views.market_view import RealTimeMarketView
            self.modules["market_view"] = True
            logger.info("✅ 市场视图模块存在")
        except ImportError as e:
            self.modules["market_view"] = False
            self.errors.append(f"市场视图模块导入失败: {str(e)}")
            logger.error(f"❌ 市场视图模块导入失败: {str(e)}")
            
    def apply_fixes(self):
        """应用修复"""
        logger.info(f"开始应用 {len(self.fixes)} 个修复方案...")
        
        for i, fix in enumerate(self.fixes):
            fix_type = fix.get("type")
            description = fix.get("description", "未知修复")
            
            logger.info(f"应用修复 [{i+1}/{len(self.fixes)}]: {description}")
            print(f"\n正在应用修复: {description}")
            
            try:
                if fix_type == "pip_install":
                    packages = fix.get("packages", [])
                    if packages:
                        self._fix_pip_install(packages)
                elif fix_type == "fix_tushare":
                    self._fix_tushare()
                else:
                    logger.warning(f"未知的修复类型: {fix_type}")
            except Exception as e:
                logger.error(f"应用修复 '{description}' 失败: {str(e)}")
                print(f"❌ 修复失败: {str(e)}")
                
        print("\n修复应用完成")
        
    def _fix_pip_install(self, packages):
        """安装Python包"""
        import subprocess
        
        for package in packages:
            print(f"正在安装 {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ {package} 安装成功")
            except subprocess.CalledProcessError as e:
                print(f"❌ {package} 安装失败: {str(e)}")
                
    def _fix_tushare(self):
        """修复TuShare连接问题"""
        # 获取token
        token = self._get_tushare_token()
        
        if not token:
            token = input("请输入有效的TuShare token: ")
            if token:
                # 保存到配置文件
                self._save_tushare_token(token)
                print(f"✅ Token已保存")
            else:
                print("❌ 未提供有效token")
                return
        
        # 测试连接
        try:
            import tushare as ts
            ts.set_token(token)
            pro = ts.pro_api()
            df = pro.query('trade_cal', exchange='', start_date='20210101', end_date='20210110')
            
            if df is not None and not df.empty:
                print("✅ TuShare连接测试成功")
            else:
                print("❌ TuShare连接测试失败")
        except Exception as e:
            print(f"❌ TuShare连接测试失败: {str(e)}")
            
    def _get_tushare_token(self):
        """从配置文件中获取TuShare token"""
        try:
            # 读取配置文件
            if os.path.exists("config.json"):
                with open("config.json", "r", encoding="utf-8") as f:
                    config = json.load(f)
                    return config.get("tushare_token", "")
        except Exception as e:
            logger.error(f"读取配置文件失败: {str(e)}")
            
        return ""
        
    def _save_tushare_token(self, token):
        """保存TuShare token到配置文件"""
        try:
            # 读取现有配置
            config = {}
            if os.path.exists("config.json"):
                with open("config.json", "r", encoding="utf-8") as f:
                    config = json.load(f)
                    
            # 更新token
            config["tushare_token"] = token
            
            # 保存配置
            with open("config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"保存配置文件失败: {str(e)}")
        
    def print_result(self):
        """打印诊断结果"""
        print("\n=============================================")
        print("             超神系统诊断结果                ")
        print("=============================================")
        
        # 打印模块状态
        print("\n模块状态:")
        for module, status in self.modules.items():
            status_str = "✅ 正常" if status else "❌ 异常"
            print(f"  {module}: {status_str}")
            
        # 打印错误
        if self.errors:
            print("\n发现问题:")
            for i, error in enumerate(self.errors):
                print(f"  {i+1}. {error}")
        else:
            print("\n✅ 未发现问题")
            
        # 打印修复建议
        if self.fixes:
            print("\n修复建议:")
            for i, fix in enumerate(self.fixes):
                print(f"  {i+1}. {fix.get('description', '未知修复')}")
                
        print("\n=============================================")
        
        if not self.errors:
            print("🎉 诊断完成，系统状态良好！")
        else:
            print(f"⚠️ 诊断完成，发现 {len(self.errors)} 个问题。")
            
            if not self.args.no_fix and not self.fixes:
                print("❗ 无法自动修复所有问题，请手动解决。")
                
        print("=============================================\n")
        
def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="超神系统模块诊断和修复工具")
    parser.add_argument("--no-fix", action="store_true", help="不应用自动修复")
    parser.add_argument("--no-ui", action="store_true", help="不检查UI模块")
    parser.add_argument("--verbose", action="store_true", help="显示详细日志")
    
    args = parser.parse_args()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # 运行诊断
    troubleshooter = ModuleTroubleshooter(args)
    troubleshooter.run()
    
if __name__ == "__main__":
    main() 