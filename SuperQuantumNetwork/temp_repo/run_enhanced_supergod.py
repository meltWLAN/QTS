#!/usr/bin/env python3
"""
超神量子共生系统 - 增强版运行脚本
运行真正超神级别的系统
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import traceback
import matplotlib.pyplot as plt

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("supergod_enhanced.log")
    ]
)

logger = logging.getLogger("SupergodEnhanced")

def show_banner():
    """显示超神系统横幅"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                                                                       ║
    ║                  超神量子共生系统 - 真·超神增强版                    ║
    ║                                                                       ║
    ║               SUPERGOD QUANTUM SYMBIOTIC SYSTEM                       ║
    ║                   TRANSCENDENT ENHANCED EDITION                       ║
    ║                                                                       ║
    ║                       高维量子分析 · 混沌预测                         ║
    ║                       真实超神 · 尽在眼前                             ║
    ║                                                                       ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_requirements():
    """检查运行要求"""
    logger.info("检查系统运行要求...")
    
    # 检查Python版本
    py_version = sys.version_info
    logger.info(f"Python版本: {py_version.major}.{py_version.minor}.{py_version.micro}")
    
    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 7):
        logger.warning("建议使用Python 3.7或更高版本以获得最佳性能")
    
    # 检查必要模块
    required_modules = {
        'numpy': '高效数值计算',
        'pandas': '数据处理',
        'matplotlib': '可视化',
        'scipy': '科学计算'
    }
    
    advanced_modules = {
        'numba': '即时编译加速',
        'sklearn': '机器学习功能',
        'torch': '深度学习功能',
        'statsmodels': '统计模型',
        'mplfinance': '金融图表'
    }
    
    logger.info("检查基础模块...")
    missing_required = []
    for module, desc in required_modules.items():
        try:
            __import__(module)
            logger.info(f"✓ {module} - {desc}")
        except ImportError:
            logger.error(f"✗ {module} - {desc} [缺失!]")
            missing_required.append(module)
    
    logger.info("检查高级模块...")
    missing_advanced = []
    for module, desc in advanced_modules.items():
        try:
            __import__(module)
            logger.info(f"✓ {module} - {desc}")
        except ImportError:
            logger.warning(f"△ {module} - {desc} [可选]")
            missing_advanced.append(module)
    
    if missing_required:
        logger.error(f"缺少必要模块: {', '.join(missing_required)}")
        logger.error("请使用以下命令安装: pip install " + " ".join(missing_required))
        return False
        
    if missing_advanced:
        logger.warning(f"缺少高级模块: {', '.join(missing_advanced)}")
        logger.warning("部分高级功能将不可用，建议安装: pip install " + " ".join(missing_advanced))
    
    return True

def load_core_system():
    """加载超神核心系统"""
    logger.info("加载超神核心系统...")
    
    try:
        # 导入核心模块
        from china_market_core import ChinaMarketCore
        from policy_analyzer import PolicyAnalyzer
        from sector_rotation_tracker import SectorRotationTracker
        
        # 初始化核心组件
        market_core = ChinaMarketCore()
        policy_analyzer = PolicyAnalyzer()
        sector_tracker = SectorRotationTracker()
        
        logger.info("超神核心系统加载完成")
        return market_core, policy_analyzer, sector_tracker
    
    except ImportError as e:
        logger.error(f"导入核心模块失败: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None, None
    except Exception as e:
        logger.error(f"加载核心系统失败: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None, None

def load_enhancement_modules():
    """加载增强模块"""
    logger.info("加载超神增强模块...")
    
    enhancement_modules = {}
    
    try:
        # 导入增强器
        from supergod_enhancer import get_enhancer
        enhancement_modules['enhancer'] = get_enhancer()
        logger.info("✓ 超神增强器")
        
        # 导入混沌理论框架
        try:
            from chaos_theory_framework import get_chaos_analyzer
            enhancement_modules['chaos'] = get_chaos_analyzer()
            logger.info("✓ 混沌理论分析框架")
        except ImportError:
            logger.warning("△ 混沌理论分析框架不可用")
        
        # 导入量子维度扩展器
        try:
            from quantum_dimension_enhancer import get_dimension_enhancer
            enhancement_modules['dimensions'] = get_dimension_enhancer()
            logger.info("✓ 量子维度扩展器")
        except ImportError:
            logger.warning("△ 量子维度扩展器不可用")
        
        # 检查增强模块数量
        if len(enhancement_modules) < 1:
            logger.warning("没有找到任何增强模块")
            return None
            
        logger.info(f"成功加载 {len(enhancement_modules)} 个增强模块")
        return enhancement_modules
        
    except Exception as e:
        logger.error(f"加载增强模块失败: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def generate_test_data(days=100):
    """生成测试数据"""
    logger.info(f"生成测试数据 ({days}天)...")
    
    try:
        # 生成日期序列
        end_date = datetime.now()
        start_date = end_date - pd.Timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # 创建价格序列 - 添加趋势、循环和噪声
        base = 3000
        trend = np.linspace(0, 0.15, len(dates))
        cycle1 = 0.05 * np.sin(np.linspace(0, 4*np.pi, len(dates)))
        cycle2 = 0.03 * np.sin(np.linspace(0, 12*np.pi, len(dates)))
        noise = np.random.normal(0, 0.01, len(dates))
        
        # 计算每日涨跌幅
        changes = np.diff(np.concatenate([[0], trend])) + cycle1 + cycle2 + noise
        
        # 计算价格
        prices = base * np.cumprod(1 + changes)
        
        # 生成成交量
        volume_base = 1e9
        volume_trend = np.linspace(0, 0.3, len(dates))
        volume_cycle = 0.2 * np.sin(np.linspace(0, 6*np.pi, len(dates)))
        volume_noise = np.random.normal(0, 0.15, len(dates))
        volumes = volume_base * (1 + volume_trend + volume_cycle + volume_noise)
        volumes = np.abs(volumes)  # 确保成交量为正
        
        # 创建数据框
        data = pd.DataFrame({
            'date': dates,
            'open': prices * (1 - 0.005 * np.random.random(len(dates))),
            'high': prices * (1 + 0.01 * np.random.random(len(dates))),
            'low': prices * (1 - 0.01 * np.random.random(len(dates))),
            'close': prices,
            'volume': volumes,
            'turnover_rate': 0.5 * (1 + 0.5 * np.random.random(len(dates)))
        })
        
        logger.info(f"测试数据生成完成，包含 {len(data)} 个交易日")
        return data
    
    except Exception as e:
        logger.error(f"生成测试数据失败: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def run_enhanced_analysis(market_data, core_modules, enhancement_modules):
    """运行增强分析"""
    logger.info("开始超神增强分析...")
    
    results = {
        'market_analysis': None,
        'chaos_analysis': None,
        'quantum_dimensions': None,
        'enhanced_predictions': None
    }
    
    try:
        market_core, policy_analyzer, sector_tracker = core_modules
        
        # 运行市场分析
        if market_core:
            logger.info("执行市场核心分析...")
            market_analysis = market_core.analyze_market(market_data)
            results['market_analysis'] = market_analysis
            logger.info("市场核心分析完成")
        
        # 运行混沌分析
        if 'chaos' in enhancement_modules:
            logger.info("执行混沌理论分析...")
            chaos_results = enhancement_modules['chaos'].analyze(market_data['close'])
            results['chaos_analysis'] = chaos_results
            
            # 生成混沌吸引子图表
            enhancement_modules['chaos'].plot_phase_space("market_chaos_attractor.png")
            logger.info("混沌理论分析完成")
        
        # 运行量子维度分析
        if 'dimensions' in enhancement_modules:
            logger.info("执行量子维度分析...")
            dimensions_data = enhancement_modules['dimensions'].enhance_dimensions(market_data)
            dimension_state = enhancement_modules['dimensions'].get_dimension_state()
            
            results['quantum_dimensions'] = {
                'data': dimensions_data,
                'state': dimension_state
            }
            logger.info("量子维度分析完成")
        
        # 结合所有分析结果生成增强预测
        logger.info("生成增强预测...")
        enhanced_predictions = generate_enhanced_predictions(results)
        results['enhanced_predictions'] = enhanced_predictions
        logger.info("增强预测完成")
        
        return results
    
    except Exception as e:
        logger.error(f"超神增强分析失败: {str(e)}")
        logger.error(traceback.format_exc())
        return results

def generate_enhanced_predictions(analysis_results):
    """根据分析结果生成增强预测"""
    predictions = {
        'short_term': {
            'direction': None,
            'confidence': 0.0,
            'targets': [],
            'time_frame': '1-3天'
        },
        'medium_term': {
            'direction': None,
            'confidence': 0.0,
            'targets': [],
            'time_frame': '1-2周'
        },
        'long_term': {
            'direction': None,
            'confidence': 0.0,
            'targets': [],
            'time_frame': '1-3月'
        },
        'market_state': {
            'current_phase': None,
            'next_phase': None,
            'transition_probability': 0.0
        },
        'critical_points': [],
        'anomalies': []
    }
    
    try:
        # 市场分析结果
        market_analysis = analysis_results.get('market_analysis')
        
        # 混沌分析结果
        chaos_analysis = analysis_results.get('chaos_analysis')
        
        # 量子维度结果
        quantum_dimensions = analysis_results.get('quantum_dimensions')
        
        # 实现简单预测逻辑
        if market_analysis:
            # 从市场分析提取当前周期
            if 'current_cycle' in market_analysis:
                predictions['market_state']['current_phase'] = str(market_analysis['current_cycle'])
        
        if chaos_analysis:
            # 从混沌分析提取状态和稳定性
            if 'market_regime' in chaos_analysis:
                regime = chaos_analysis['market_regime']
                predictions['market_state']['current_phase'] = regime
                
                # 根据混沌分析预测方向
                if regime in ['trending', 'complex_trending']:
                    predictions['short_term']['direction'] = 'bullish'
                    predictions['medium_term']['direction'] = 'bullish'
                elif regime in ['mean_reverting', 'chaotic_reverting']:
                    predictions['short_term']['direction'] = 'bearish'
                elif regime in ['edge_of_chaos']:
                    predictions['short_term']['direction'] = 'volatile'
                else:
                    predictions['short_term']['direction'] = 'sideways'
                
                # 设置置信度
                predictions['short_term']['confidence'] = max(0.3, min(0.9, chaos_analysis.get('stability', 0.5)))
                predictions['medium_term']['confidence'] = max(0.2, min(0.8, chaos_analysis.get('stability', 0.5) * 0.9))
                
                # 提取临界点
                if 'critical_points' in chaos_analysis and chaos_analysis['critical_points']:
                    for point, score in chaos_analysis['critical_points']:
                        predictions['critical_points'].append({
                            'position': int(point),
                            'score': float(score),
                            'type': 'chaos_transition'
                        })
        
        if quantum_dimensions and 'state' in quantum_dimensions:
            dimension_state = quantum_dimensions['state']
            
            # 使用量子维度状态调整预测
            energy_potential = dimension_state.get('energy_potential', {}).get('value', 0.5)
            temporal_coherence = dimension_state.get('temporal_coherence', {}).get('value', 0.5)
            chaos_degree = dimension_state.get('chaos_degree', {}).get('value', 0.5)
            
            # 使用能量势能预测长期方向
            if energy_potential > 0.7:
                predictions['long_term']['direction'] = 'bullish'
                predictions['long_term']['confidence'] = min(0.8, energy_potential)
            elif energy_potential < 0.3:
                predictions['long_term']['direction'] = 'bearish'
                predictions['long_term']['confidence'] = min(0.8, 1 - energy_potential)
            else:
                predictions['long_term']['direction'] = 'sideways'
                predictions['long_term']['confidence'] = 0.5
                
            # 时间相干性影响转变概率
            predictions['market_state']['transition_probability'] = 1 - temporal_coherence
            
            # 混沌度影响异常检测
            if chaos_degree > 0.75:
                predictions['anomalies'].append({
                    'type': 'high_chaos',
                    'severity': chaos_degree,
                    'description': '市场混沌度异常高，可能出现剧烈波动'
                })
        
        return predictions
        
    except Exception as e:
        logger.error(f"生成增强预测失败: {str(e)}")
        logger.error(traceback.format_exc())
        return predictions

def display_results(analysis_results):
    """显示分析结果"""
    logger.info("显示超神分析结果...")
    
    # 市场分析结果
    market_analysis = analysis_results.get('market_analysis')
    if market_analysis:
        print("\n========== 市场分析结果 ==========")
        if 'current_cycle' in market_analysis:
            print(f"市场周期: {market_analysis['current_cycle']}")
        if 'cycle_confidence' in market_analysis:
            print(f"周期置信度: {market_analysis['cycle_confidence']:.2f}")
        if 'market_sentiment' in market_analysis:
            print(f"市场情绪: {market_analysis['market_sentiment']:.2f}")
    
    # 混沌分析结果
    chaos_analysis = analysis_results.get('chaos_analysis')
    if chaos_analysis:
        print("\n========== 混沌理论分析 ==========")
        print(f"赫斯特指数: {chaos_analysis.get('hurst_exponent', 0):.3f} (>0.5趋势性, <0.5反转性)")
        print(f"分形维度: {chaos_analysis.get('fractal_dimension', 0):.3f}")
        print(f"莱雅普诺夫指数: {chaos_analysis.get('lyapunov_exponent', 0):.6f}")
        print(f"熵值: {chaos_analysis.get('entropy', 0):.3f}")
        print(f"复杂度: {chaos_analysis.get('complexity', 0):.3f}")
        print(f"市场状态: {chaos_analysis.get('market_regime', 'unknown')}")
        print(f"稳定性: {chaos_analysis.get('stability', 0):.3f}")
        
        if 'critical_points' in chaos_analysis:
            print(f"检测到的临界点数量: {len(chaos_analysis['critical_points'])}")
            
        if 'fractal_patterns' in chaos_analysis:
            print(f"识别的分形模式数量: {len(chaos_analysis['fractal_patterns'])}")
            if chaos_analysis['fractal_patterns']:
                for i, pattern in enumerate(chaos_analysis['fractal_patterns'][:3]):
                    print(f"  模式{i+1}: {pattern['name']}, 置信度: {pattern['confidence']:.2f}")
    
    # 量子维度结果
    quantum_dimensions = analysis_results.get('quantum_dimensions')
    if quantum_dimensions and 'state' in quantum_dimensions:
        dimension_state = quantum_dimensions['state']
        print("\n========== 量子维度分析 ==========")
        
        # 基础维度
        print("基础维度状态:")
        basic_dims = ['price', 'volume', 'momentum', 'volatility', 'sentiment']
        for dim in basic_dims:
            if dim in dimension_state:
                value = dimension_state[dim]['value']
                trend = dimension_state[dim]['trend']
                trend_symbol = "↗" if trend > 0.01 else "↘" if trend < -0.01 else "→"
                print(f"  {dim}: {value:.3f} {trend_symbol}")
        
        # 扩展维度
        print("\n扩展维度状态:")
        extended_dims = ['fractal', 'entropy', 'chaos_degree', 'energy_potential', 'temporal_coherence']
        for dim in extended_dims:
            if dim in dimension_state:
                value = dimension_state[dim]['value']
                trend = dimension_state[dim]['trend']
                trend_symbol = "↗" if trend > 0.01 else "↘" if trend < -0.01 else "→"
                print(f"  {dim}: {value:.3f} {trend_symbol}")
    
    # 增强预测
    predictions = analysis_results.get('enhanced_predictions')
    if predictions:
        print("\n========== 超神增强预测 ==========")
        
        # 市场状态
        market_state = predictions.get('market_state', {})
        print(f"当前市场阶段: {market_state.get('current_phase', 'unknown')}")
        if market_state.get('next_phase'):
            print(f"下一阶段预测: {market_state.get('next_phase')}")
            print(f"转变概率: {market_state.get('transition_probability', 0):.2f}")
        
        # 方向预测
        print("\n方向预测:")
        for term in ['short_term', 'medium_term', 'long_term']:
            if term in predictions:
                prediction = predictions[term]
                direction = prediction.get('direction', 'unknown')
                confidence = prediction.get('confidence', 0)
                time_frame = prediction.get('time_frame', '')
                
                direction_symbol = "↗" if direction == 'bullish' else "↘" if direction == 'bearish' else "↔"
                print(f"  {term} ({time_frame}): {direction} {direction_symbol} 置信度: {confidence:.2f}")
        
        # 异常
        if 'anomalies' in predictions and predictions['anomalies']:
            print("\n检测到的异常:")
            for anomaly in predictions['anomalies']:
                print(f"  * {anomaly.get('type')}: {anomaly.get('description')} (严重度: {anomaly.get('severity', 0):.2f})")
        
        # 临界点
        if 'critical_points' in predictions and predictions['critical_points']:
            print(f"\n检测到 {len(predictions['critical_points'])} 个潜在临界点")

def main():
    """主函数"""
    # 显示横幅
    show_banner()
    
    # 检查要求
    if not check_requirements():
        logger.error("不满足基本要求，无法继续运行")
        return
    
    # 加载核心系统
    core_modules = load_core_system()
    if not all(core_modules):
        logger.error("加载核心系统失败，无法继续运行")
        return
    
    # 加载增强模块
    enhancement_modules = load_enhancement_modules()
    if not enhancement_modules:
        logger.warning("加载增强模块失败，将使用有限功能继续")
    
    # 生成测试数据
    market_data = generate_test_data(120)
    if market_data is None:
        logger.error("生成测试数据失败，无法继续运行")
        return
    
    # 运行增强分析
    analysis_results = run_enhanced_analysis(market_data, core_modules, enhancement_modules)
    
    # 显示结果
    display_results(analysis_results)
    
    logger.info("超神增强系统运行完成")
    print("\n图表已保存，可查看市场混沌吸引子图表: market_chaos_attractor.png")
    print("\n超神增强系统运行完成!")

if __name__ == "__main__":
    main() 