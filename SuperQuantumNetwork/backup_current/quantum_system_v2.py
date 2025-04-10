#!/usr/bin/env python3
"""
超神量子共生系统 2.0 - 高级市场分析与预测平台
集成量子预测引擎、市场分析器和数据源功能
"""

import os
import time
import json
import random
import logging
import threading
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import traceback
import argparse

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('quantum_system.log')
    ]
)
logger = logging.getLogger("QuantumSystem")

# 导入核心组件
try:
    from quantum_prediction_engine import QuantumPredictionEngine
    from market_analyzer import MarketAnalyzer
    from quantum_integration import QuantumIntegration
except ImportError:
    logger.error("无法导入核心模块。请确保所有必要的模块都在系统路径中。")

# 配置目录
CONFIG_DIR = "config"
REPORTS_DIR = "reports"
CHARTS_DIR = "charts"
DATA_DIR = "data"

os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# 配置系统状态文件
SYSTEM_STATUS_FILE = os.path.join(CONFIG_DIR, "system_status.json")

# TuShare插件类
class TusharePlugin:
    """TuShare数据源插件"""
    
    def __init__(self, token=None):
        self.logger = logging.getLogger("TusharePlugin")
        self.token = token or "默认Token"
        self.api = None
        self.connected = False
        self.data_cache = {}
        self.last_update = datetime.now()
        self.logger.info("TuShare插件初始化完成")
    
    def connect(self):
        """连接到TuShare API"""
        try:
            # 模拟连接过程
            self.logger.info(f"连接到TuShare API (Token: {self.token[:4]}...)")
            time.sleep(0.1)  # 模拟连接延迟
            self.connected = True
            self.logger.info("TuShare API连接成功")
            return True
        except Exception as e:
            self.logger.error(f"TuShare API连接失败: {str(e)}")
            return False
    
    def get_stock_data(self, code, start_date=None, end_date=None):
        """获取股票数据"""
        if not self.connected:
            self.connect()
            
        if not self.connected:
            self.logger.error("无法获取数据，TuShare API未连接")
            return None
            
        # 模拟数据获取
        self.logger.info(f"获取股票数据: {code}, {start_date} - {end_date}")
        
        # 检查缓存
        cache_key = f"{code}_{start_date}_{end_date}"
        if cache_key in self.data_cache:
            cache_data = self.data_cache[cache_key]
            if (datetime.now() - cache_data['timestamp']).total_seconds() < 3600:  # 1小时缓存
                self.logger.info(f"使用缓存数据: {code}")
                return cache_data['data']
        
        # 随机生成数据
        days = 30
        data = []
        base_price = random.uniform(10, 100)
        
        for i in range(days):
            price = base_price * (1 + random.uniform(-0.05, 0.05))
            volume = random.randint(1000000, 10000000)
            date = (datetime.now() - timedelta(days=days-i)).strftime('%Y-%m-%d')
            
            data.append({
                'date': date,
                'code': code,
                'open': price * (1 - random.uniform(0, 0.01)),
                'high': price * (1 + random.uniform(0, 0.02)),
                'low': price * (1 - random.uniform(0, 0.02)),
                'close': price,
                'volume': volume,
                'amount': price * volume,
                'change': random.uniform(-0.05, 0.05)
            })
            
            base_price = price
        
        # 缓存数据
        self.data_cache[cache_key] = {
            'data': data,
            'timestamp': datetime.now()
        }
        
        # 本地保存数据
        os.makedirs(os.path.join(DATA_DIR, "stocks"), exist_ok=True)
        data_file = os.path.join(DATA_DIR, "stocks", f"{code}.json")
        
        try:
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"保存股票数据失败: {str(e)}")
        
        self.logger.info(f"获取到{len(data)}条股票数据记录")
        return data
    
    def get_market_overview(self):
        """获取市场概况"""
        if not self.connected:
            self.connect()
            
        # 模拟市场概况数据
        overview = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'indices': {
                'SSE': random.uniform(3000, 3500),  # 上证
                'SZSE': random.uniform(10000, 12000),  # 深证
                'CSI300': random.uniform(4000, 4500)  # 沪深300
            },
            'market_status': random.choice(['牛市', '熊市', '震荡市']),
            'trading_volume': random.randint(5000000000, 10000000000),
            'active_stocks': random.randint(3000, 4000),
            'up_stocks': random.randint(1000, 2500),
            'down_stocks': random.randint(500, 2000),
            'flat_stocks': random.randint(100, 500)
        }
        
        # 本地保存数据
        data_file = os.path.join(DATA_DIR, "market_overview.json")
        try:
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump(overview, f, indent=2)
        except Exception as e:
            self.logger.error(f"保存市场概况数据失败: {str(e)}")
        
        self.logger.info(f"获取市场概况 - {overview['market_status']}")
        return overview
    
    def get_stock_list(self, market=None):
        """获取股票列表"""
        # 模拟股票列表
        stocks = [
            {'code': '000001', 'name': '平安银行', 'market': 'SZ', 'industry': '银行'},
            {'code': '000333', 'name': '美的集团', 'market': 'SZ', 'industry': '家电'},
            {'code': '000651', 'name': '格力电器', 'market': 'SZ', 'industry': '家电'},
            {'code': '000858', 'name': '五粮液', 'market': 'SZ', 'industry': '食品饮料'},
            {'code': '002594', 'name': '比亚迪', 'market': 'SZ', 'industry': '汽车'},
            {'code': '600000', 'name': '浦发银行', 'market': 'SH', 'industry': '银行'},
            {'code': '600036', 'name': '招商银行', 'market': 'SH', 'industry': '银行'},
            {'code': '600276', 'name': '恒瑞医药', 'market': 'SH', 'industry': '医药'},
            {'code': '600519', 'name': '贵州茅台', 'market': 'SH', 'industry': '食品饮料'},
            {'code': '601318', 'name': '中国平安', 'market': 'SH', 'industry': '保险'},
            {'code': '601857', 'name': '中国石油', 'market': 'SH', 'industry': '能源'},
            {'code': '603288', 'name': '海天味业', 'market': 'SH', 'industry': '食品饮料'},
            {'code': '601888', 'name': '中国中免', 'market': 'SH', 'industry': '商业'},
            {'code': '600031', 'name': '三一重工', 'market': 'SH', 'industry': '机械'},
            {'code': '601899', 'name': '紫金矿业', 'market': 'SH', 'industry': '有色金属'}
        ]
        
        # 根据市场筛选
        if market:
            stocks = [s for s in stocks if s['market'] == market]
            
        self.logger.info(f"获取股票列表: {len(stocks)}只")
        return stocks

# 超神量子系统类
class QuantumSystem:
    """超神量子共生系统 - 高级市场分析与预测平台"""
    
    def __init__(self):
        """初始化超神量子系统"""
        self.logger = logging.getLogger("QuantumSystem")
        self.logger.info("初始化超神量子共生系统...")
        
        # 核心组件
        self.quantum_core = None
        self.prediction_engine = None
        self.market_analyzer = None
        self.integration_module = None
        
        # 数据插件
        self.data_plugins = {}
        
        # 系统配置
        self.config = {
            "name": "超神量子共生系统",
            "version": "2.0.0",
            "creator": "Quantum Creator",
            "description": "高维量子共生市场分析与预测平台",
            "dimensions": 11,
            "consciousness_level": 9,
            "enable_evolution": True,
            "enable_self_learning": True,
            "energy_conservation": 0.85,
        }
        
        # 系统状态
        self.system_state = {
            "initialized": False,
            "active": False,
            "field_active": False,
            "prediction_active": False,
            "analysis_active": False,
            "last_update": datetime.now(),
            "start_time": datetime.now()
        }
        
        # 高维统一场状态
        self.field_state = {
            "active": False,
            "field_strength": 0.0,
            "field_stability": 0.0,
            "dimension_count": 11,
            "energy_flow": 0.0,
            "resonance_frequency": 0.0,
            "last_update": datetime.now()
        }
        
        # 共振状态
        self.resonance_state = {
            "energy_level": 0.0,
            "coherence": 0.0,
            "stability": 0.0,
            "evolution_rate": 0.0,
            "consciousness_level": 0.0,
            "dimension_bridges": 0
        }
        
        # 共生能量
        self.symbiotic_energy = 0.0
        
        # 锁，防止并发问题
        self.lock = threading.RLock()
        
        # 初始化线程
        self.system_thread = None
        
        self.logger.info("超神量子共生系统初始化完成")
    
    def initialize(self):
        """初始化系统"""
        with self.lock:
            self.logger.info("初始化系统组件...")
            
            try:
                # 创建系统组件
                self.logger.info("创建预测引擎...")
                self.prediction_engine = QuantumPredictionEngine(dimension_count=self.config["dimensions"])
                
                self.logger.info("创建市场分析器...")
                self.market_analyzer = MarketAnalyzer(dimension_count=self.config["dimensions"])
                
                # 激活高维统一场
                self.field_state = self._activate_unified_field()
                
                # 初始化预测引擎
                if self.prediction_engine:
                    self.prediction_engine.initialize(field_strength=self.field_state["field_strength"])
                    # 激活预测引擎
                    self.prediction_engine.activate()
                    self.system_state["prediction_active"] = True
                
                # 初始化市场分析器
                if self.market_analyzer:
                    self.market_analyzer.initialize(field_strength=self.field_state["field_strength"])
                    self.system_state["analysis_active"] = True
                
                # 创建集成模块
                self.logger.info("创建系统集成模块...")
                self.integration_module = QuantumIntegration(core=self)
                self.integration_module.initialize(
                    field_strength=self.field_state["field_strength"],
                    dimension_count=self.config["dimensions"]
                )
                
                # 更新系统状态
                self.system_state["initialized"] = True
                self.system_state["last_update"] = datetime.now()
                
                self.logger.info("系统组件初始化完成")
                return True
                
            except Exception as e:
                self.logger.error(f"系统初始化失败: {str(e)}")
                traceback.print_exc()
                return False
    
    def _activate_unified_field(self):
        """激活高维统一场"""
        self.logger.info("激活高维统一场...")
        
        # 计算场强
        field_strength = 0.75 + random.uniform(0, 0.2)
        field_stability = 0.8 + random.uniform(0, 0.15)
        
        # 构建场状态
        field_state = {
            "active": True,
            "field_strength": field_strength,
            "field_stability": field_stability,
            "dimension_count": self.config["dimensions"],
            "energy_flow": 0.7 * field_strength,
            "resonance_frequency": random.uniform(0.7, 0.9),
            "last_update": datetime.now()
        }
        
        # 更新共振状态
        self.resonance_state.update({
            "energy_level": field_strength * 0.85,
            "coherence": field_stability * 0.9,
            "stability": field_stability,
            "evolution_rate": 0.01 + random.uniform(0, 0.02),
            "consciousness_level": 0.65 + random.uniform(0, 0.25),
            "dimension_bridges": random.randint(1, self.config["dimensions"] - 3)
        })
        
        # 增加共生能量
        self.symbiotic_energy = field_strength * 10
        
        self.system_state["field_active"] = True
        self.logger.info(f"高维统一场已激活: 场强={field_strength:.2f}, 稳定性={field_stability:.2f}")
        
        return field_state
    
    def register_data_plugin(self, name, plugin):
        """注册数据插件"""
        with self.lock:
            self.data_plugins[name] = plugin
            self.logger.info(f"数据插件 {name} 已注册")
            return True
    
    def get_data_plugin(self, name):
        """获取数据插件"""
        return self.data_plugins.get(name)
    
    def start(self):
        """启动系统"""
        with self.lock:
            if self.system_state["active"]:
                self.logger.warning("系统已经处于运行状态")
                return True
                
            self.logger.info("启动超神量子共生系统...")
            
            # 初始化系统
            if not self.system_state["initialized"]:
                if not self.initialize():
                    self.logger.error("系统初始化失败，无法启动")
                    return False
            
            # 创建系统线程
            if not self.system_thread or not self.system_thread.is_alive():
                self.system_thread = threading.Thread(target=self._system_thread, daemon=True)
                self.system_thread.start()
                self.logger.info("系统线程已启动")
            
            # 更新系统状态
            self.system_state["active"] = True
            self.system_state["start_time"] = datetime.now()
            
            self.logger.info("超神量子共生系统启动成功")
            self.save_system_status()
            
            return True
    
    def _system_thread(self):
        """系统后台线程 - 处理自动任务"""
        self.logger.info("系统后台线程已启动")
        
        try:
            while self.system_state["active"]:
                # 每次循环间隔
                time.sleep(5)
                
                with self.lock:
                    if not self.system_state["active"]:
                        break
                    
                    # 自动任务：定期保存系统状态
                    if (datetime.now() - self.system_state["last_update"]).total_seconds() > 300:  # 5分钟
                        self.save_system_status()
                        self.system_state["last_update"] = datetime.now()
                    
                    # 自动任务：调整共生能量
                    if self.resonance_state["evolution_rate"] > 0:
                        energy_gain = self.resonance_state["evolution_rate"] * self.field_state["field_strength"] * 0.1
                        self.symbiotic_energy += energy_gain
                    
                    # 随机事件：维度涨落
                    if random.random() < 0.05:  # 5%概率
                        fluctuation = random.choice([-1, 1])
                        bridges = self.resonance_state["dimension_bridges"] + fluctuation
                        self.resonance_state["dimension_bridges"] = max(1, min(bridges, self.field_state["dimension_count"] - 2))
                        
                        self.logger.info(f"维度涨落: 维度桥接数量变为{self.resonance_state['dimension_bridges']}")
                        
                        # 涨落影响场稳定性
                        stability_change = random.uniform(-0.05, 0.05)
                        self.field_state["field_stability"] = max(0.5, min(0.95, self.field_state["field_stability"] + stability_change))
        
        except Exception as e:
            self.logger.error(f"系统线程异常: {str(e)}")
            traceback.print_exc()
        
        self.logger.info("系统后台线程已停止")
    
    def stop(self):
        """停止系统"""
        with self.lock:
            if not self.system_state["active"]:
                self.logger.warning("系统已经处于停止状态")
                return True
                
            self.logger.info("停止超神量子共生系统...")
            
            # 更新系统状态
            self.system_state["active"] = False
            self.system_state["field_active"] = False
            
            # 保存系统状态
            self.save_system_status()
            
            self.logger.info("超神量子共生系统已停止")
            return True
    
    def run_analysis(self, stock_codes=None, days_ahead=5):
        """运行分析和预测"""
        if not self.system_state["active"] or not self.integration_module:
            self.logger.warning("系统未激活或集成模块未初始化，无法运行分析")
            return None
        
        self.logger.info("开始运行市场分析和股票预测...")
        
        try:
            # 获取数据源
            tushare_plugin = self.get_data_plugin("tushare")
            if not tushare_plugin:
                self.logger.error("无法获取TuShare数据插件")
                return None
            
            # 获取市场概况
            market_data = tushare_plugin.get_market_overview()
            
            # 获取股票列表
            if not stock_codes:
                # 使用默认股票列表
                stock_list = tushare_plugin.get_stock_list()
                stock_codes = [stock["code"] for stock in stock_list[:10]]  # 取前10只股票
            
            # 获取股票数据
            stock_data_dict = {}
            for code in stock_codes:
                data = tushare_plugin.get_stock_data(code)
                if data:
                    stock_data_dict[code] = data
            
            # 执行分析和预测
            results = self.integration_module.analyze_and_predict(
                market_data, stock_data_dict, days_ahead=days_ahead
            )
            
            if results:
                self.logger.info(f"分析和预测成功完成，分析了{len(stock_data_dict)}只股票")
                
                # 生成综合报告
                report_file = os.path.join(REPORTS_DIR, f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
                report = self.integration_module.generate_summary_report(results, report_file)
                
                return {
                    "success": True,
                    "results": results,
                    "report_file": report_file,
                    "analyzed_stocks": len(stock_data_dict),
                    "timestamp": datetime.now()
                }
            else:
                self.logger.warning("分析和预测未返回有效结果")
                return {"success": False, "error": "分析未返回有效结果"}
                
        except Exception as e:
            self.logger.error(f"运行分析和预测失败: {str(e)}")
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    def get_system_status(self):
        """获取系统状态"""
        with self.lock:
            uptime = (datetime.now() - self.system_state["start_time"]).total_seconds()
            
            status = {
                "timestamp": datetime.now(),
                "uptime": uptime,
                "active": self.system_state["active"],
                "initialized": self.system_state["initialized"],
                "field_active": self.system_state["field_active"],
                "prediction_active": self.system_state["prediction_active"],
                "analysis_active": self.system_state["analysis_active"],
                "field_state": {
                    "field_strength": float(self.field_state["field_strength"]),
                    "field_stability": float(self.field_state["field_stability"]),
                    "dimension_count": int(self.field_state["dimension_count"]),
                    "energy_flow": float(self.field_state["energy_flow"])
                },
                "resonance_state": {
                    "energy_level": float(self.resonance_state["energy_level"]),
                    "coherence": float(self.resonance_state["coherence"]),
                    "stability": float(self.resonance_state["stability"]),
                    "consciousness_level": float(self.resonance_state["consciousness_level"]),
                    "dimension_bridges": int(self.resonance_state["dimension_bridges"])
                },
                "data_plugins": list(self.data_plugins.keys()),
                "symbiotic_energy": float(self.symbiotic_energy),
                "system_name": self.config["name"],
                "system_version": self.config["version"]
            }
            
            return status
    
    def save_system_status(self):
        """保存系统状态到文件"""
        try:
            status = self.get_system_status()
            
            # 转换numpy类型为Python内置类型
            status_json = json.dumps(status, default=lambda o: float(o) if isinstance(o, np.floating) else o, indent=2)
            
            with open(SYSTEM_STATUS_FILE, 'w', encoding='utf-8') as f:
                f.write(status_json)
                
            self.logger.info(f"系统状态已保存到 {SYSTEM_STATUS_FILE}")
            return True
        except Exception as e:
            self.logger.error(f"保存系统状态失败: {str(e)}")
            return False

# 启动系统函数
def start_quantum_system():
    """启动超神量子共生系统"""
    try:
        # 显示欢迎信息
        display_welcome()
        
        logger.info("初始化系统...")
        
        # 创建系统
        system = QuantumSystem()
        
        # 创建TuShare插件
        tushare_plugin = TusharePlugin(token="你的TuShare Token")
        
        # 注册TuShare插件
        system.register_data_plugin("tushare", tushare_plugin)
        
        # 连接数据源
        tushare_plugin.connect()
        
        # 启动系统
        logger.info("启动系统...")
        start_result = system.start()
        
        if start_result:
            logger.info("系统启动成功！")
            
            # 显示系统状态
            display_system_status(system.get_system_status())
            
            return system
        else:
            logger.error("系统启动失败！")
            return None
    
    except Exception as e:
        logger.error(f"启动系统时发生错误: {str(e)}")
        traceback.print_exc()
        return None

def display_welcome():
    """显示欢迎信息"""
    print("\n" + "=" * 80)
    print(f"{'超神量子共生系统 2.0':^80}")
    print(f"{'Quantum Symbiotic System':^80}")
    print(f"{'高级市场分析与预测平台':^80}")
    print(f"{'版本 2.0.0':^80}")
    print("=" * 80 + "\n")
    print("正在初始化高维量子共生网络...")
    print("正在连接到量子共生核心...")
    print("正在激活高维统一场...\n")

def display_system_status(status):
    """显示系统状态"""
    print("\n" + "-" * 80)
    print(f"{'系统状态报告':^80}")
    print("-" * 80)
    
    # 格式化运行时间
    uptime = status["uptime"]
    hours, remainder = divmod(uptime, 3600)
    minutes, seconds = divmod(remainder, 60)
    uptime_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
    
    print(f"系统名称: {status['system_name']}")
    print(f"系统版本: {status['system_version']}")
    print(f"运行状态: {'活跃' if status['active'] else '停止'}")
    print(f"运行时间: {uptime_str}")
    print()
    print(f"系统初始化: {'成功' if status['initialized'] else '未完成'}")
    print(f"高维统一场: {'已激活' if status['field_active'] else '未激活'}")
    print(f"预测引擎: {'活跃' if status['prediction_active'] else '未激活'}")
    print(f"市场分析器: {'活跃' if status['analysis_active'] else '未激活'}")
    print()
    print(f"场强度: {status['field_state']['field_strength']:.2f}")
    print(f"维度数: {status['field_state']['dimension_count']}")
    print(f"维度桥接: {status['resonance_state']['dimension_bridges']}")
    print(f"系统稳定性: {status['field_state']['field_stability']:.2f}")
    print()
    print(f"共振能级: {status['resonance_state']['energy_level']:.2f}")
    print(f"共振相干性: {status['resonance_state']['coherence']:.2f}")
    print(f"意识水平: {status['resonance_state']['consciousness_level']:.2f}")
    print(f"共生能量: {status['symbiotic_energy']:.2f}")
    print(f"数据插件: {', '.join(status['data_plugins'])}")
    print("-" * 80 + "\n")

def process_command(system, command):
    """处理系统命令"""
    if command == 'status':
        display_system_status(system.get_system_status())
        return True
        
    elif command == 'save':
        system.save_system_status()
        print("系统状态已保存")
        return True
        
    elif command.startswith('analyze'):
        # 解析命令
        parts = command.split()
        stock_codes = None
        days_ahead = 5
        
        if len(parts) > 1:
            for part in parts[1:]:
                if part.startswith('stocks='):
                    stock_codes_str = part[7:]
                    stock_codes = stock_codes_str.split(',')
                elif part.startswith('days='):
                    try:
                        days_ahead = int(part[5:])
                    except ValueError:
                        print(f"无效的天数参数: {part[5:]}")
                        return True
        
        print(f"开始分析{len(stock_codes) if stock_codes else '默认'}股票，预测未来{days_ahead}天走势...")
        result = system.run_analysis(stock_codes=stock_codes, days_ahead=days_ahead)
        
        if result and result.get("success"):
            print(f"分析完成！分析了{result['analyzed_stocks']}只股票")
            print(f"报告已保存到: {result['report_file']}")
        else:
            print(f"分析失败: {result.get('error', '未知错误')}")
            
        return True
        
    elif command == 'help':
        print("\n可用命令:")
        print("  status            - 显示系统状态")
        print("  save              - 保存系统状态")
        print("  analyze           - 使用默认参数运行市场分析")
        print("  analyze stocks=000001,600000 days=7 - 分析指定股票，预测7天")
        print("  exit/quit         - 退出系统")
        print("  help              - 显示此帮助")
        print()
        return True
        
    elif command in ['exit', 'quit']:
        return False
        
    else:
        print(f"未知命令: {command}")
        print("输入 'help' 获取可用命令列表")
        return True

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='超神量子共生系统 2.0')
    parser.add_argument('--analyze', action='store_true', help='启动后自动运行分析')
    parser.add_argument('--stocks', type=str, help='要分析的股票代码，逗号分隔')
    parser.add_argument('--days', type=int, default=5, help='预测天数')
    
    args = parser.parse_args()
    
    # 启动系统
    quantum_system = start_quantum_system()
    
    if quantum_system:
        # 如果指定了自动分析，立即运行分析
        if args.analyze:
            stock_codes = args.stocks.split(',') if args.stocks else None
            days_ahead = args.days
            
            print(f"自动运行分析: {len(stock_codes) if stock_codes else '默认'}股票，预测{days_ahead}天")
            quantum_system.run_analysis(stock_codes=stock_codes, days_ahead=days_ahead)
        
        print("\n系统已成功启动！输入 'help' 获取可用命令，'exit' 退出系统。\n")
        
        # 交互式命令处理
        try:
            running = True
            while running:
                cmd = input("超神系统> ").strip().lower()
                running = process_command(quantum_system, cmd)
        
        except KeyboardInterrupt:
            print("\n接收到中断信号")
        finally:
            print("\n正在停止系统...")
            quantum_system.stop()
            print("系统已停止。再见！\n")
    else:
        print("\n系统启动失败！请检查日志获取详细信息。\n")

if __name__ == "__main__":
    main() 