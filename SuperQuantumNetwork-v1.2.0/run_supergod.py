#!/usr/bin/env python3
"""
超神量子共生系统 - 统一驾驶舱入口
整合所有功能的单一启动入口
"""

import os
import sys
import time
import logging
import argparse
import traceback
from datetime import datetime
import signal
import platform

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("supergod_unified.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SupergodUnified")

# 显示横幅
def show_banner():
    """显示统一超神系统横幅"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                                                                       ║
    ║                  超神量子共生系统 - 统一驾驶舱                       ║
    ║                                                                       ║
    ║                 SUPERGOD QUANTUM SYMBIOTIC SYSTEM                     ║
    ║                       UNIFIED COCKPIT EDITION                         ║
    ║                                                                       ║
    ║           实时数据 · 增强分析 · 量子扩展 · 智能交互                  ║
    ║                   集成一体 · 尽在掌握                                 ║
    ║                                                                       ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

# 检查依赖
def check_dependencies():
    """检查系统依赖和必要模块"""
    logger.info("检查系统依赖...")
    
    # 检查Python版本
    py_version = sys.version_info
    logger.info(f"Python版本: {py_version.major}.{py_version.minor}.{py_version.micro}")
    
    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 7):
        logger.warning("建议使用Python 3.7或更高版本以获得最佳性能")
    
    # 基础依赖
    basic_deps = {
        'numpy': '数值计算',
        'pandas': '数据处理',
        'matplotlib': '图表绘制'
    }
    
    # 数据源依赖
    data_deps = {
        'tushare': 'A股数据源',
        'akshare': '备选数据源'
    }
    
    # UI依赖
    ui_deps = {
        'PyQt5': '图形界面'
    }
    
    # 高级功能依赖
    advanced_deps = {
        'scipy': '科学计算',
        'sklearn': '机器学习',
        'statsmodels': '统计模型',
        'torch': '深度学习',
        'numba': '高性能计算',
        'mplfinance': '金融图表'
    }
    
    # 检查基础依赖
    logger.info("检查基础依赖...")
    missing_basic = []
    for mod, desc in basic_deps.items():
        try:
            __import__(mod)
            logger.info(f"✓ {mod} - {desc}")
        except ImportError:
            logger.error(f"✗ {mod} - {desc} [缺失!]")
            missing_basic.append(mod)
    
    # 检查数据源依赖
    logger.info("检查数据源依赖...")
    missing_data = []
    has_datasource = False
    for mod, desc in data_deps.items():
        try:
            __import__(mod)
            logger.info(f"✓ {mod} - {desc}")
            has_datasource = True
        except ImportError:
            logger.warning(f"△ {mod} - {desc} [不可用]")
            missing_data.append(mod)
    
    # 检查UI依赖
    logger.info("检查UI依赖...")
    missing_ui = []
    for mod, desc in ui_deps.items():
        try:
            __import__(mod)
            logger.info(f"✓ {mod} - {desc}")
        except ImportError:
            logger.error(f"✗ {mod} - {desc} [缺失!]")
            missing_ui.append(mod)
    
    # 检查高级功能依赖
    logger.info("检查高级功能依赖...")
    missing_advanced = []
    for mod, desc in advanced_deps.items():
        try:
            __import__(mod)
            logger.info(f"✓ {mod} - {desc}")
        except ImportError:
            logger.warning(f"△ {mod} - {desc} [可选]")
            missing_advanced.append(mod)
    
    # 检查核心模块
    logger.info("检查超神系统核心模块...")
    core_modules = [
        ('china_market_core', '中国市场分析核心'),
        ('policy_analyzer', '政策分析器'),
        ('sector_rotation_tracker', '板块轮动跟踪器')
    ]
    
    missing_core = []
    for mod, desc in core_modules:
        try:
            __import__(mod)
            logger.info(f"✓ {mod} - {desc}")
        except ImportError:
            logger.warning(f"✗ {mod} - {desc} [缺失]")
            missing_core.append(mod)
    
    # 检查增强模块
    logger.info("检查超神系统增强模块...")
    enhancement_modules = [
        ('supergod_enhancer', '超神增强器'),
        ('chaos_theory_framework', '混沌理论框架'),
        ('quantum_dimension_enhancer', '量子维度扩展器')
    ]
    
    missing_enhancement = []
    for mod, desc in enhancement_modules:
        try:
            __import__(mod)
            logger.info(f"✓ {mod} - {desc}")
        except ImportError:
            logger.warning(f"△ {mod} - {desc} [可选]")
            missing_enhancement.append(mod)
    
    # 汇总结果
    if missing_basic:
        logger.error(f"缺少基础依赖: {', '.join(missing_basic)}")
        logger.error(f"请使用以下命令安装: pip install {' '.join(missing_basic)}")
        return False
    
    if missing_ui:
        logger.error(f"缺少UI依赖: {', '.join(missing_ui)}")
        logger.error(f"请使用以下命令安装: pip install {' '.join(missing_ui)}")
        return False
    
    if not has_datasource:
        logger.warning("未安装任何数据源模块，将使用模拟数据")
        print("\n建议安装以下数据源之一以获取真实市场数据:")
        print("pip install tushare")
        print("pip install akshare==1.12.24")
    
    return True

# 获取可用的数据连接器
def get_data_connector(token=None):
    """创建并返回可用的数据连接器"""
    logger.info("初始化数据连接器...")
    
    # 首先尝试Tushare
    try:
        import tushare as ts
        
        # 设置token
        if token:
            ts.set_token(token)
        else:
            # 尝试从环境变量获取
            env_token = os.environ.get('TUSHARE_TOKEN')
            if env_token:
                ts.set_token(env_token)
            else:
                # 使用默认token
                default_token = '0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10'
                ts.set_token(default_token)
        
        # 测试连接
        pro = ts.pro_api()
        test_data = pro.trade_cal(exchange='SSE', start_date='20250401', end_date='20250407')
        
        if test_data is not None and len(test_data) > 0:
            logger.info("Tushare连接成功，使用Tushare数据源")
            from tushare_data_connector import TushareDataConnector
            return TushareDataConnector(token=ts.get_token())
    except Exception as e:
        logger.warning(f"Tushare初始化失败: {str(e)}")
    
    # 尝试AKShare
    try:
        import akshare as ak
        
        # 测试连接
        test_data = ak.stock_zh_index_daily(symbol="sh000001")
        if test_data is not None and not test_data.empty:
            logger.info("AKShare连接成功，使用AKShare数据源")
            from akshare_data_connector import AKShareDataConnector
            return AKShareDataConnector()
    except Exception as e:
        logger.warning(f"AKShare初始化失败: {str(e)}")
    
    # 如果都失败，使用演示数据
    logger.warning("无法连接到任何数据源，将使用演示数据")
    
    try:
        # 使用内置模拟数据连接器
        from tushare_data_connector import TushareDataConnector
        return TushareDataConnector()  # 将使用其内置的模拟数据模式
    except Exception as e:
        logger.error(f"无法创建模拟数据连接器: {str(e)}")
        return None

# 加载增强模块
def load_enhancement_modules():
    """加载超神增强模块"""
    logger.info("加载超神增强模块...")
    
    enhancement_modules = {}
    
    try:
        # 导入增强器
        try:
            from supergod_enhancer import get_enhancer
            enhancement_modules['enhancer'] = get_enhancer()
            logger.info("✓ 超神增强器已加载")
        except ImportError:
            logger.warning("△ 超神增强器不可用")
        
        # 导入混沌理论框架
        try:
            from chaos_theory_framework import get_chaos_analyzer
            enhancement_modules['chaos'] = get_chaos_analyzer()
            logger.info("✓ 混沌理论分析框架已加载")
        except ImportError:
            logger.warning("△ 混沌理论分析框架不可用")
        
        # 导入量子维度扩展器
        try:
            from quantum_dimension_enhancer import get_dimension_enhancer
            enhancement_modules['dimensions'] = get_dimension_enhancer()
            logger.info("✓ 量子维度扩展器已加载")
        except ImportError:
            logger.warning("△ 量子维度扩展器不可用")
        
        # 检查增强模块数量
        if len(enhancement_modules) < 1:
            logger.warning("没有找到任何增强模块，将使用基础版功能")
            return None
            
        logger.info(f"成功加载 {len(enhancement_modules)} 个增强模块")
        return enhancement_modules
        
    except Exception as e:
        logger.error(f"加载增强模块失败: {str(e)}")
        logger.error(traceback.format_exc())
        return None

# 终端模式入口
def run_terminal_mode(data_connector, enhancement_modules):
    """在终端运行超神系统分析"""
    logger.info("启动终端分析模式...")
    
    try:
        # 加载核心模块
        from china_market_core import ChinaMarketCore
        market_core = ChinaMarketCore()
        
        # 获取测试数据
        symbol = "000001.SH"
        logger.info(f"获取市场数据: {symbol}")
        market_data = data_connector.get_market_data(symbol)
        
        if market_data is None or market_data.empty:
            logger.error("无法获取市场数据，终止分析")
            return False
        
        logger.info(f"成功获取市场数据，共{len(market_data)}条记录")
        print(f"\n最近市场数据:")
        print(market_data.tail(3))
        
        # 分析市场数据
        logger.info("执行市场分析...")
        analysis_result = market_core.analyze_market(market_data)
        
        # 应用增强分析
        if enhancement_modules:
            logger.info("应用超神增强...")
            
            # 混沌分析
            if 'chaos' in enhancement_modules:
                chaos = enhancement_modules['chaos']
                logger.info("执行混沌理论分析...")
                try:
                    chaos_results = chaos.analyze(market_data['close'].values)
                    analysis_result['chaos'] = chaos_results
                    
                    # 生成混沌吸引子图表
                    try:
                        chaos.plot_phase_space("market_chaos_attractor.png")
                        logger.info("已生成混沌吸引子图表: market_chaos_attractor.png")
                    except Exception as e:
                        logger.warning(f"生成混沌吸引子图表失败: {str(e)}")
                except Exception as e:
                    logger.warning(f"混沌理论分析失败: {str(e)}")
            
            # 量子维度分析
            if 'dimensions' in enhancement_modules:
                dimensions = enhancement_modules['dimensions']
                logger.info("执行量子维度分析...")
                try:
                    dimensions_data = dimensions.enhance_dimensions(market_data)
                    dimension_state = dimensions.get_dimension_state()
                    analysis_result['quantum_dimensions'] = {
                        'data': dimensions_data,
                        'state': dimension_state
                    }
                except Exception as e:
                    logger.warning(f"量子维度分析失败: {str(e)}")
            
            # 最终增强
            if 'enhancer' in enhancement_modules:
                enhancer = enhancement_modules['enhancer']
                logger.info("应用超神增强...")
                try:
                    enhanced_results = enhancer.enhance(market_data, analysis_result)
                    analysis_result = enhanced_results
                except Exception as e:
                    logger.warning(f"应用超神增强失败: {str(e)}")
        
        # 输出分析结果
        print("\n======== 市场分析结果 ========")
        
        # 输出市场情绪
        if 'market_sentiment' in analysis_result:
            sentiment = analysis_result['market_sentiment']
            print(f"市场情绪指数: {sentiment:.2f}")
            if sentiment > 0.3:
                print("市场状态: 看涨")
            elif sentiment < -0.3:
                print("市场状态: 看跌") 
            else:
                print("市场状态: 盘整")
        
        # 输出市场周期
        if 'current_cycle' in analysis_result:
            print(f"市场周期: {analysis_result['current_cycle']}")
            
        # 输出周期置信度
        if 'cycle_confidence' in analysis_result:
            print(f"周期置信度: {analysis_result['cycle_confidence']:.2f}")
        
        # 输出混沌分析结果
        if 'chaos' in analysis_result:
            chaos = analysis_result['chaos']
            print("\n===== 混沌理论分析 =====")
            for key, value in chaos.items():
                if isinstance(value, (int, float)):
                    print(f"{key}: {value:.3f}")
                else:
                    print(f"{key}: {value}")
        
        # 输出量子维度结果
        if 'quantum_dimensions' in analysis_result:
            qd = analysis_result['quantum_dimensions']
            print("\n===== 量子维度分析 =====")
            if 'state' in qd:
                state = qd['state']
                print("维度状态:")
                for dim_name, dim_data in state.items():
                    if isinstance(dim_data, dict) and 'value' in dim_data:
                        print(f"  {dim_name}: {dim_data['value']:.3f}")
        
        # 输出预测
        if 'predictions' in analysis_result:
            predictions = analysis_result['predictions']
            print("\n===== 市场预测 =====")
            if 'short_term' in predictions:
                short = predictions['short_term']
                print(f"短期(1-3天): {short['direction']} 置信度: {short['confidence']:.2f}")
            if 'medium_term' in predictions:
                medium = predictions['medium_term']
                print(f"中期(1-2周): {medium['direction']} 置信度: {medium['confidence']:.2f}")
            if 'long_term' in predictions:
                long = predictions['long_term'] 
                print(f"长期(1-3月): {long['direction']} 置信度: {long['confidence']:.2f}")
        
        logger.info("终端分析完成")
        return True
    
    except Exception as e:
        logger.error(f"终端分析模式失败: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"\n分析过程中发生错误: {str(e)}")
        return False

# 驾驶舱模式入口
def run_cockpit_mode(data_connector, enhancement_modules):
    """启动超神驾驶舱"""
    logger.info("启动超神驾驶舱模式...")
    
    try:
        # 导入驾驶舱
        from PyQt5.QtWidgets import QApplication
        from supergod_cockpit import SupergodCockpit
        import signal
        
        # 捕获Ctrl+C信号，确保程序能够正常退出
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        
        # 创建应用
        app = QApplication(sys.argv)
        
        # 创建主窗口
        cockpit = SupergodCockpit()
        
        # 设置数据连接器
        if hasattr(cockpit, 'set_data_connector'):
            cockpit.set_data_connector(data_connector)
            logger.info("已设置数据连接器")
        
        # 设置增强模块
        if enhancement_modules and hasattr(cockpit, 'set_enhancement_modules'):
            cockpit.set_enhancement_modules(enhancement_modules)
            logger.info("已设置增强模块")
        
        # 显示驾驶舱
        cockpit.show()
        
        # 设置退出处理
        app.aboutToQuit.connect(lambda: cleanup_resources(cockpit))
        
        # 运行应用
        logger.info("超神驾驶舱启动完成")
        return app.exec_()
    
    except Exception as e:
        logger.error(f"启动驾驶舱失败: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"\n启动驾驶舱失败: {str(e)}")
        # 确保异常情况下也能正确清理资源
        cleanup_and_exit()
        return 1

def cleanup_resources(cockpit=None):
    """清理资源，确保程序干净退出"""
    logger.info("正在清理资源...")
    
    # 停止任何仍在运行的线程或计时器
    if cockpit:
        if hasattr(cockpit, 'update_timer') and cockpit.update_timer.isActive():
            cockpit.update_timer.stop()
        if hasattr(cockpit, 'special_effects_timer') and cockpit.special_effects_timer.isActive():
            cockpit.special_effects_timer.stop()
    
    # 关闭所有可能的资源连接
    logger.info("资源清理完成")

def cleanup_and_exit():
    """紧急清理资源并退出"""
    logger.info("紧急清理资源...")
    # 如果有数据库连接等，在这里关闭
    # 强制结束可能还在运行的线程
    import threading
    for thread in threading.enumerate():
        if thread is not threading.current_thread() and not thread.daemon:
            logger.info(f"强制终止线程: {thread.name}")
            # 线程无法直接终止，但可以设置标志
            if hasattr(thread, "stop"):
                thread.stop()
    logger.info("紧急清理完成")
    sys.exit(1)

# 桌面模式入口
def run_desktop_mode(data_connector, enhancement_modules):
    """启动超神桌面应用"""
    logger.info("启动超神桌面模式...")
    
    try:
        # 导入桌面应用
        from PyQt5.QtWidgets import QApplication
        from supergod_desktop import SupergodDesktop
        
        # 捕获Ctrl+C信号，确保程序能够正常退出
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        
        # 创建应用
        app = QApplication(sys.argv)
        
        # 创建主窗口
        desktop = SupergodDesktop()
        
        # 设置数据连接器
        if data_connector:
            desktop.set_data_connector(data_connector)
            logger.info("已设置数据连接器")
        
        # 设置增强模块
        if enhancement_modules:
            desktop.set_enhancement_modules(enhancement_modules)
            logger.info("已设置增强模块")
        
        # 显示桌面
        desktop.show()
        
        # 运行应用
        logger.info("超神桌面启动完成")
        try:
            return app.exec_()
        except KeyboardInterrupt:
            logger.info("程序被用户中断")
            return 0
    
    except Exception as e:
        logger.error(f"启动桌面应用失败: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"\n启动桌面应用失败: {str(e)}")
        return 1

# 主函数
def main():
    """主函数 - 解析命令行参数并启动相应模式"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='超神量子共生系统统一入口')
    parser.add_argument('--mode', type=str, default='cockpit', 
                        choices=['cockpit', 'desktop', 'terminal'],
                        help='运行模式：cockpit(驾驶舱)、desktop(桌面)、terminal(终端)')
    parser.add_argument('--token', type=str, default=None, 
                        help='Tushare API Token，可选')
    parser.add_argument('--enhanced', action='store_true', default=False,
                        help='启用增强模式，加载所有增强模块')
    parser.add_argument('--symbol', type=str, default="000001.SH", 
                        help="分析的股票或指数代码")
    parser.add_argument('--no-banner', action='store_true', default=False,
                        help='不显示启动横幅')
    parser.add_argument('--test', action='store_true', default=False,
                        help='测试模式，仅验证组件可用性，不实际启动系统')
    args = parser.parse_args()
    
    # 测试模式处理 - 仅验证组件可用性
    if args.test:
        logger.info("执行测试模式验证...")
        try:
            # 检查依赖项
            check_dependencies()
            
            # 尝试初始化数据连接器
            data_connector = get_data_connector(args.token)
            
            # 加载增强模块（如果启用）
            if args.enhanced:
                enhancement_modules = load_enhancement_modules()
                if not enhancement_modules:
                    logger.warning("无法加载增强模块")
            
            logger.info("测试模式验证通过")
            return 0
        except Exception as e:
            logger.error(f"测试模式验证失败: {str(e)}")
            return 1
    
    # 显示横幅
    if not args.no_banner:
        show_banner()
    
    # 检查依赖
    if not check_dependencies():
        print("\n启动失败: 缺少必要依赖")
        sys.exit(1)
    
    # 获取系统信息
    sysinfo = {
        'platform': platform.system(),
        'release': platform.release(),
        'python': platform.python_version(),
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    logger.info(f"系统平台: {sysinfo['platform']} {sysinfo['release']}")
    logger.info(f"Python版本: {sysinfo['python']}")
    logger.info(f"启动时间: {sysinfo['time']}")
    logger.info(f"运行模式: {args.mode}")
    
    # 设置TuShare Token
    if args.token:
        os.environ['TUSHARE_TOKEN'] = args.token
        logger.info(f"已设置自定义Tushare Token")
    
    # 获取数据连接器
    data_connector = get_data_connector(args.token)
    
    # 加载增强模块
    enhancement_modules = None
    if args.enhanced:
        logger.info("启用超神增强模式")
        enhancement_modules = load_enhancement_modules()
    
    # 根据模式运行
    if args.mode == "terminal":
        success = run_terminal_mode(data_connector, enhancement_modules)
        sys.exit(0 if success else 1)
    elif args.mode == "desktop":
        sys.exit(run_desktop_mode(data_connector, enhancement_modules))
    else:  # cockpit模式
        sys.exit(run_cockpit_mode(data_connector, enhancement_modules))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        sys.exit(0)
    except Exception as e:
        logger.error(f"程序出现未捕获异常: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"\n程序出现未捕获异常: {str(e)}")
        sys.exit(1) 