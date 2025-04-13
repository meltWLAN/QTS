#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
量子选股验证工具 - 对超神量子选股器进行完整测试验证
"""

import os
import sys
import json
import logging
import time
from datetime import datetime
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QProgressBar, QTextEdit, QPushButton
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

# 导入量子选股相关模块
from quantum_desktop.ui.panels.quantum_stock_finder_panel import StockFinderThread
from quantum_desktop.system_manager import SystemManager

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(project_root, "quantum_stock_validation.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("QuantumStockValidation")

class StockFinderValidator(QThread):
    """量子选股验证线程"""
    
    status_updated = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    validation_complete = pyqtSignal(dict)
    
    def __init__(self, system_manager, parent=None):
        super().__init__(parent)
        self.system_manager = system_manager
        self.is_running = False
        self.results = {}
        
    def run(self):
        """运行验证测试"""
        self.is_running = True
        try:
            # 开始验证过程
            self.status_updated.emit("开始量子选股验证...")
            self.progress_updated.emit(5)
            
            # 步骤1: 系统状态检查
            self.status_updated.emit("步骤1/5: 系统状态检查...")
            system_check_results = self._check_system_status()
            self.progress_updated.emit(20)
            
            # 步骤2: 模拟数据验证
            self.status_updated.emit("步骤2/5: 量子选股器模拟数据验证...")
            sim_results = self._validate_simulated_mode()
            self.progress_updated.emit(40)
            
            # 步骤3: 实时数据验证 (如果可用)
            self.status_updated.emit("步骤3/5: 实时数据API验证...")
            api_results = self._validate_api_connection()
            self.progress_updated.emit(60)
            
            # 步骤4: 实时量子选股验证
            self.status_updated.emit("步骤4/5: 实时量子选股验证...")
            real_results = self._validate_real_mode()
            self.progress_updated.emit(80)
            
            # 步骤5: 结果一致性分析
            self.status_updated.emit("步骤5/5: 结果一致性分析...")
            consistency_results = self._analyze_consistency(sim_results, real_results)
            self.progress_updated.emit(95)
            
            # 汇总全部结果
            self.results = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "system_status": system_check_results,
                "simulated_test": sim_results,
                "api_test": api_results,
                "real_test": real_results,
                "consistency": consistency_results,
                "overall_status": self._calculate_overall_status(
                    system_check_results, sim_results, api_results, real_results, consistency_results
                )
            }
            
            # 验证完成
            self.status_updated.emit("量子选股验证完成！")
            self.progress_updated.emit(100)
            self.validation_complete.emit(self.results)
            
        except Exception as e:
            logger.error(f"验证过程中发生错误: {str(e)}")
            self.status_updated.emit(f"验证失败: {str(e)}")
        finally:
            self.is_running = False
    
    def _check_system_status(self):
        """检查系统状态"""
        results = {
            "components": {},
            "overall": "未就绪"
        }
        
        try:
            # 检查系统是否已启动
            system_running = False
            if hasattr(self.system_manager, "is_system_running"):
                system_running = self.system_manager.is_system_running()
            
            results["system_running"] = system_running
            
            # 获取组件状态
            critical_components = ["quantum_backend", "converter", "market_data_provider"]
            all_critical_ready = True
            
            for component_name in critical_components:
                status = "未就绪"
                is_ready = False
                
                component = self.system_manager.get_component(component_name)
                if component:
                    comp_status = self.system_manager.get_component_status(component_name).get(component_name)
                    status = comp_status
                    is_ready = comp_status == "running"
                    
                results["components"][component_name] = {
                    "status": status,
                    "is_ready": is_ready
                }
                
                if not is_ready:
                    all_critical_ready = False
            
            # 判断整体状态
            if system_running and all_critical_ready:
                results["overall"] = "就绪"
            
            return results
        
        except Exception as e:
            logger.error(f"检查系统状态时出错: {str(e)}")
            results["error"] = str(e)
            return results
    
    def _validate_simulated_mode(self):
        """验证模拟数据模式"""
        results = {
            "success": False,
            "error": None,
            "stocks_count": 0,
            "data_source": None,
            "execution_time_ms": 0
        }
        
        try:
            # 创建模拟模式线程
            start_time = time.time()
            sim_thread = StockFinderThread(None)
            sim_thread.quantum_power = 50
            sim_thread.market_scope = "全市场"
            sim_thread.sector_filter = "全部行业"
            sim_thread.use_real_data = False  # 强制使用模拟数据
            
            # 运行选股
            finder_results = sim_thread.run()
            end_time = time.time()
            
            # 记录结果
            if finder_results and finder_results.get("status") == "success":
                results["success"] = True
                results["stocks_count"] = len(finder_results.get("stocks", []))
                results["data_source"] = finder_results.get("data_source")
                results["execution_time_ms"] = int((end_time - start_time) * 1000)
                
                # 记录前5只股票
                top_stocks = []
                for i, stock in enumerate(finder_results.get("stocks", [])[:5]):
                    top_stocks.append({
                        "rank": i + 1,
                        "code": stock.get("code", "未知"),
                        "name": stock.get("name", "未知"),
                        "score": stock.get("quantum_score", 0)
                    })
                results["top_stocks"] = top_stocks
            
            return results
        
        except Exception as e:
            logger.error(f"验证模拟模式时出错: {str(e)}")
            results["error"] = str(e)
            return results
    
    def _validate_api_connection(self):
        """验证API连接"""
        results = {
            "success": False,
            "provider": "tushare",
            "error": None,
            "connection_status": "未连接"
        }
        
        try:
            # 尝试导入和连接tushare
            try:
                import tushare as ts
                
                # 尝试使用API
                api_key = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"
                ts.set_token(api_key)
                pro = ts.pro_api()
                
                # 尝试一个简单查询
                df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
                
                # 记录结果
                if df is not None and len(df) > 0:
                    results["success"] = True
                    results["connection_status"] = "连接成功"
                    results["stocks_count"] = len(df)
                    results["sample_data"] = df.head(3).to_dict('records')
            
            except ImportError:
                results["error"] = "Tushare库未安装"
            except Exception as e:
                results["error"] = f"API连接错误: {str(e)}"
            
            return results
        
        except Exception as e:
            logger.error(f"验证API连接时出错: {str(e)}")
            results["error"] = str(e)
            return results
    
    def _validate_real_mode(self):
        """验证实时数据模式"""
        results = {
            "success": False,
            "error": None,
            "stocks_count": 0,
            "data_source": None,
            "execution_time_ms": 0
        }
        
        try:
            # 尝试实时模式 (如果API测试成功)
            api_config = self._validate_api_connection()
            if not api_config["success"]:
                results["error"] = "API连接失败，无法进行实时验证"
                results["api_error"] = api_config["error"]
                return results
            
            # 创建实时模式线程
            start_time = time.time()
            real_thread = StockFinderThread(None)
            real_thread.quantum_power = 50
            real_thread.market_scope = "科创板"  # 限制范围，加快测试
            real_thread.sector_filter = "全部行业"
            real_thread.use_real_data = True  # 强制使用实时数据
            
            try:
                # 尝试运行选股
                finder_results = real_thread.run()
                end_time = time.time()
                
                # 记录结果
                if finder_results and finder_results.get("status") == "success":
                    results["success"] = True
                    results["stocks_count"] = len(finder_results.get("stocks", []))
                    results["data_source"] = finder_results.get("data_source")
                    results["execution_time_ms"] = int((end_time - start_time) * 1000)
                    
                    # 记录前5只股票
                    top_stocks = []
                    for i, stock in enumerate(finder_results.get("stocks", [])[:5]):
                        top_stocks.append({
                            "rank": i + 1,
                            "code": stock.get("ts_code", stock.get("code", "未知")),
                            "name": stock.get("name", "未知"),
                            "score": stock.get("quantum_score", 0)
                        })
                    results["top_stocks"] = top_stocks
            
            except Exception as e:
                # 失败回退到模拟模式
                results["error"] = f"实时选股失败: {str(e)}"
                results["fallback"] = "已回退到模拟模式"
                
                # 记录回退到模拟模式的结果
                sim_results = self._validate_simulated_mode()
                results["fallback_results"] = sim_results
            
            return results
        
        except Exception as e:
            logger.error(f"验证实时模式时出错: {str(e)}")
            results["error"] = str(e)
            return results
    
    def _analyze_consistency(self, sim_results, real_results):
        """分析模拟和实时结果的一致性"""
        results = {
            "score_ranges_consistent": False,
            "performance_analysis": {},
            "data_structure_consistent": False,
            "notes": []
        }
        
        try:
            # 检查两种模式是否都成功
            if not sim_results["success"]:
                results["notes"].append("模拟模式测试失败，无法进行一致性分析")
                return results
                
            if not real_results["success"]:
                results["notes"].append("实时模式测试失败，使用回退结果进行分析")
                
                # 检查是否有回退结果
                if "fallback_results" in real_results and real_results["fallback_results"]["success"]:
                    real_results = real_results["fallback_results"]
                else:
                    results["notes"].append("无可用的实时或回退结果，无法进行一致性分析")
                    return results
            
            # 分析数据结构一致性
            sim_top = sim_results.get("top_stocks", [])
            real_top = real_results.get("top_stocks", [])
            
            if sim_top and real_top:
                # 检查关键字段是否存在
                sim_keys = set()
                real_keys = set()
                
                for stock in sim_top:
                    sim_keys.update(stock.keys())
                
                for stock in real_top:
                    real_keys.update(stock.keys())
                
                # 检查必要字段
                required_keys = {"rank", "code", "name", "score"}
                sim_has_required = required_keys.issubset(sim_keys)
                real_has_required = required_keys.issubset(real_keys)
                
                results["data_structure_consistent"] = sim_has_required and real_has_required
                
                if not results["data_structure_consistent"]:
                    results["notes"].append(f"数据结构不一致。模拟模式: {sim_keys}, 实时模式: {real_keys}")
            
            # 分析评分范围一致性
            sim_scores = [stock.get("score", 0) for stock in sim_top]
            real_scores = [stock.get("score", 0) for stock in real_top]
            
            if sim_scores and real_scores:
                sim_min, sim_max = min(sim_scores), max(sim_scores)
                real_min, real_max = min(real_scores), max(real_scores)
                
                # 检查评分范围是否大致相同
                min_diff = abs(sim_min - real_min)
                max_diff = abs(sim_max - real_max)
                
                results["score_ranges_consistent"] = (min_diff < 20) and (max_diff < 20)
                results["score_comparison"] = {
                    "simulated": {"min": sim_min, "max": sim_max},
                    "real": {"min": real_min, "max": real_max}
                }
                
                if not results["score_ranges_consistent"]:
                    results["notes"].append(f"评分范围差异较大。模拟: {sim_min}-{sim_max}, 实时: {real_min}-{real_max}")
            
            # 性能分析
            results["performance_analysis"] = {
                "simulated_ms": sim_results.get("execution_time_ms", 0),
                "real_ms": real_results.get("execution_time_ms", 0),
                "speedup_factor": round(real_results.get("execution_time_ms", 1) / max(sim_results.get("execution_time_ms", 1), 1), 2)
            }
            
            return results
        
        except Exception as e:
            logger.error(f"分析一致性时出错: {str(e)}")
            results["error"] = str(e)
            return results
    
    def _calculate_overall_status(self, system, sim, api, real, consistency):
        """计算整体状态"""
        status = {
            "status": "失败",
            "score": 0,
            "issues": []
        }
        
        # 计算评分 (满分100)
        score = 0
        
        # 系统状态 (25分)
        if system["overall"] == "就绪":
            score += 25
        else:
            status["issues"].append("系统未就绪")
            # 部分分数
            ready_components = sum(1 for comp in system["components"].values() if comp["is_ready"])
            total_components = len(system["components"])
            if total_components > 0:
                score += int(25 * (ready_components / total_components))
        
        # 模拟模式 (25分)
        if sim["success"]:
            score += 25
        else:
            status["issues"].append("模拟模式测试失败")
        
        # API连接 (15分)
        if api["success"]:
            score += 15
        else:
            status["issues"].append(f"API连接失败: {api.get('error')}")
        
        # 实时模式 (25分)
        if real["success"]:
            score += 25
        elif "fallback_results" in real and real["fallback_results"]["success"]:
            score += 10
            status["issues"].append("实时模式测试失败，但成功回退到模拟模式")
        else:
            status["issues"].append(f"实时模式测试失败: {real.get('error')}")
        
        # 一致性 (10分)
        consistency_score = 0
        if consistency.get("data_structure_consistent", False):
            consistency_score += 5
        if consistency.get("score_ranges_consistent", False):
            consistency_score += 5
        score += consistency_score
        
        if consistency_score < 10:
            status["issues"].append("模拟和实时模式的结果一致性较差")
        
        # 设置最终状态
        status["score"] = score
        if score >= 90:
            status["status"] = "优秀"
        elif score >= 75:
            status["status"] = "良好"
        elif score >= 60:
            status["status"] = "通过"
        elif score >= 40:
            status["status"] = "需要改进"
        else:
            status["status"] = "失败"
        
        return status

class ValidationWindow(QMainWindow):
    """验证工具窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("超神量子选股验证工具")
        self.resize(800, 600)
        
        # 创建系统管理器
        self.system_manager = SystemManager()
        
        # 创建验证线程
        self.validator = StockFinderValidator(self.system_manager)
        self.validator.status_updated.connect(self.update_status)
        self.validator.progress_updated.connect(self.update_progress)
        self.validator.validation_complete.connect(self.show_results)
        
        # 初始化UI
        self._init_ui()
        
    def _init_ui(self):
        """初始化UI"""
        # 中央窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        layout = QVBoxLayout(central_widget)
        
        # 标题
        title_label = QLabel("超神量子选股验证工具")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 16pt; font-weight: bold;")
        layout.addWidget(title_label)
        
        # 状态标签
        self.status_label = QLabel("准备就绪")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        # 结果文本框
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        layout.addWidget(self.results_text)
        
        # 按钮区域
        button_layout = QVBoxLayout()
        
        # 开始验证按钮
        self.start_button = QPushButton("开始验证")
        self.start_button.clicked.connect(self.start_validation)
        button_layout.addWidget(self.start_button)
        
        # 保存结果按钮
        self.save_button = QPushButton("保存结果")
        self.save_button.clicked.connect(self.save_results)
        self.save_button.setEnabled(False)
        button_layout.addWidget(self.save_button)
        
        layout.addLayout(button_layout)
    
    def start_validation(self):
        """开始验证"""
        # 清理旧结果
        self.results_text.clear()
        self.progress_bar.setValue(0)
        
        # 禁用开始按钮
        self.start_button.setEnabled(False)
        self.save_button.setEnabled(False)
        
        # 启动系统
        if not self.system_manager.is_system_running():
            self.update_status("启动量子核心系统...")
            self.system_manager.start_system()
            time.sleep(2)  # 给系统一些启动时间
        
        # 开始验证
        self.validator.start()
    
    def update_status(self, status):
        """更新状态"""
        self.status_label.setText(status)
        self.results_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] {status}")
    
    def update_progress(self, value):
        """更新进度"""
        self.progress_bar.setValue(value)
    
    def show_results(self, results):
        """显示结果"""
        self.start_button.setEnabled(True)
        self.save_button.setEnabled(True)
        
        # 格式化JSON
        results_json = json.dumps(results, indent=4, ensure_ascii=False)
        
        # 显示结果
        self.results_text.clear()
        self.results_text.append("====================== 验证结果 ======================\n")
        
        # 显示总体状态
        overall = results.get("overall_status", {})
        status_str = overall.get("status", "未知")
        score = overall.get("score", 0)
        
        self.results_text.append(f"总体评价: {status_str} (得分: {score}/100)")
        
        if "issues" in overall and overall["issues"]:
            self.results_text.append("\n发现的问题:")
            for issue in overall["issues"]:
                self.results_text.append(f"- {issue}")
        
        # 显示系统状态
        system = results.get("system_status", {})
        self.results_text.append(f"\n系统状态: {system.get('overall', '未知')}")
        
        # 显示API状态
        api = results.get("api_test", {})
        api_status = "成功" if api.get("success", False) else "失败"
        self.results_text.append(f"\nAPI连接: {api_status}")
        if not api.get("success", False) and "error" in api:
            self.results_text.append(f"错误: {api['error']}")
        
        # 显示模拟和实时结果
        sim = results.get("simulated_test", {})
        real = results.get("real_test", {})
        
        self.results_text.append(f"\n模拟选股: {'成功' if sim.get('success', False) else '失败'}")
        self.results_text.append(f"实时选股: {'成功' if real.get('success', False) else '失败'}")
        
        # 详细JSON结果
        self.results_text.append("\n\n完整结果 (JSON):\n")
        self.results_text.append(results_json)
        
        # 保存结果到对象
        self.validation_results = results
    
    def save_results(self):
        """保存结果"""
        try:
            # 确保结果目录存在
            results_dir = os.path.join(project_root, "validation_results")
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
                
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(results_dir, f"validation_{timestamp}.json")
            
            # 保存结果
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.validation_results, f, indent=4, ensure_ascii=False)
                
            self.update_status(f"结果已保存到: {filename}")
            
        except Exception as e:
            self.update_status(f"保存结果失败: {str(e)}")

def main():
    """主函数"""
    app = QApplication(sys.argv)
    window = ValidationWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 