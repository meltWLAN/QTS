#!/usr/bin/env python3
"""
量子核心演示 - 展示量子核心系统的主要功能
"""

import sys
import time
import logging
import pandas as pd
import numpy as np

from quantum_core.quantum_backend import QuantumBackend
from quantum_core.quantum_interpreter import QuantumInterpreter
from quantum_core.market_to_quantum import MarketToQuantumConverter
from quantum_core.multidimensional_analysis import MultidimensionalAnalyzer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("QuantumDemo")

def create_sample_market_data():
    """创建示例市场数据"""
    # 创建一个简单的价格时间序列
    dates = pd.date_range(start='2025-01-01', periods=30)
    
    # 为几支股票创建模拟数据
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    market_data = {}
    
    for symbol in symbols:
        # 生成随机价格数据
        np.random.seed(hash(symbol) % 10000)  # 不同股票使用不同的种子
        
        # 生成基础价格趋势
        base_price = np.random.uniform(50, 200)
        trend = np.random.uniform(-0.3, 0.3)
        
        # 生成价格时间序列
        closes = [base_price]
        for i in range(1, len(dates)):
            # 添加趋势和一些随机波动
            next_price = closes[-1] * (1 + trend/100 + np.random.normal(0, 0.02))
            closes.append(next_price)
            
        # 从收盘价生成其他价格数据
        df = pd.DataFrame({
            'date': dates,
            'open': [close * np.random.uniform(0.98, 1.0) for close in closes],
            'high': [close * np.random.uniform(1.0, 1.03) for close in closes],
            'low': [close * np.random.uniform(0.97, 1.0) for close in closes],
            'close': closes,
            'volume': [int(np.random.uniform(100000, 1000000)) for _ in closes]
        })
        
        df.set_index('date', inplace=True)
        market_data[symbol] = df
        
    return market_data

def demo_quantum_circuit(backend: QuantumBackend):
    """演示量子电路创建和执行"""
    logger.info("=" * 50)
    logger.info("量子电路演示")
    logger.info("=" * 50)
    
    # 创建Bell态电路
    circuit_id = backend.create_circuit(2, "Bell状态")
    logger.info(f"创建电路 ID: {circuit_id}")
    
    # 添加量子门
    backend.add_gate(circuit_id, 'H', [0])  # 在第一个量子比特上应用Hadamard门
    backend.add_gate(circuit_id, 'CX', [0, 1])  # CNOT门，控制位为0，目标位为1
    
    # 添加测量
    backend.add_measurement(circuit_id, 0)
    backend.add_measurement(circuit_id, 1)
    
    # 执行电路
    logger.info("执行Bell状态电路...")
    job_id = backend.execute_circuit(circuit_id, shots=1024)
    logger.info(f"作业 ID: {job_id}")
    
    # 等待执行完成
    while True:
        status = backend.get_job_status(job_id)
        if status['status'] in ['completed', 'failed', 'error']:
            break
        time.sleep(0.1)
    
    # 获取结果
    result = backend.get_result(job_id)
    logger.info(f"执行结果:")
    for state, count in result['counts'].items():
        logger.info(f"  状态 |{state}⟩: {count} 次 ({count/1024*100:.1f}%)")
    
    return result

def demo_market_to_quantum(converter: MarketToQuantumConverter, market_data):
    """演示市场数据到量子态的转换"""
    logger.info("\n" + "=" * 50)
    logger.info("市场数据到量子表示转换演示")
    logger.info("=" * 50)
    
    # 获取可用的编码方法
    methods = converter.get_encoding_methods()
    logger.info(f"可用的编码方法: {', '.join(methods.keys())}")
    
    # 使用振幅编码
    logger.info("\n使用振幅编码方法:")
    amp_result = converter.convert(market_data, method='amplitude', 
                                feature='close', num_qubits=3, lookback=8)
    
    for symbol, data in amp_result['quantum_data'].items():
        logger.info(f"  {symbol} 振幅编码:")
        logger.info(f"    量子比特数: {data['num_qubits']}")
        logger.info(f"    原始数据: {[round(x, 2) for x in data['original_data']]}")
        logger.info(f"    归一化范围: [{data['min_val']:.2f}, {data['max_val']:.2f}]")
    
    # 使用角度编码
    logger.info("\n使用角度编码方法:")
    angle_result = converter.convert(market_data, method='angle')
    
    for symbol, data in angle_result['quantum_data'].items():
        logger.info(f"  {symbol} 角度编码:")
        for feature, angle in data['angles'].items():
            logger.info(f"    {feature}: {angle:.2f} 弧度")
    
    return amp_result, angle_result

def demo_quantum_interpreter(interpreter: QuantumInterpreter, quantum_result):
    """演示量子结果解释"""
    logger.info("\n" + "=" * 50)
    logger.info("量子结果解释演示")
    logger.info("=" * 50)
    
    # 获取可用的解释方法
    methods = interpreter.get_interpretation_methods()
    logger.info(f"可用的解释方法: {', '.join(methods.keys())}")
    
    # 使用概率解释
    logger.info("\n使用概率解释方法:")
    prob_interp = interpreter.interpret(quantum_result, method='probability')
    
    if prob_interp['status'] == 'success':
        interp = prob_interp['interpretation']
        logger.info(f"  信号: {interp['signal']}")
        logger.info(f"  强度: {interp['strength']:.1f}%")
        logger.info(f"  上升概率: {interp['up_probability']*100:.1f}%")
        logger.info(f"  下降概率: {interp['down_probability']*100:.1f}%")
        logger.info(f"  最可能状态: |{interp['most_probable_state']}⟩")
    
    # 使用阈值解释
    logger.info("\n使用阈值解释方法:")
    threshold_interp = interpreter.interpret(quantum_result, method='threshold')
    
    if threshold_interp['status'] == 'success':
        interp = threshold_interp['interpretation']
        logger.info(f"  信号: {interp['signal']}")
        logger.info(f"  强度: {interp['strength']:.1f}%")
        logger.info(f"  上升比率: {interp['up_ratio']*100:.1f}%")
        logger.info(f"  下降比率: {interp['down_ratio']*100:.1f}%")
    
    return prob_interp, threshold_interp

def demo_market_analysis(analyzer: MultidimensionalAnalyzer, market_data):
    """演示市场多维分析"""
    logger.info("\n" + "=" * 50)
    logger.info("市场多维分析演示")
    logger.info("=" * 50)
    
    # 添加一些分析维度
    analyzer.add_dimension('price_trend', description='价格趋势分析')
    analyzer.add_dimension('volatility', description='波动性分析')
    
    # 执行分析
    analysis_results = analyzer.analyze(market_data)
    
    # 显示分析结果
    for symbol in market_data.keys():
        if 'default_analysis' in analysis_results and 'data' in analysis_results['default_analysis']:
            if symbol in analysis_results['default_analysis']['data']:
                data = analysis_results['default_analysis']['data'][symbol]
                logger.info(f"\n{symbol} 分析结果:")
                logger.info(f"  平均价格: {data['mean']:.2f}")
                logger.info(f"  标准差: {data['std']:.2f}")
                logger.info(f"  最新价格: {data['latest']:.2f}")
                
                if 'trend' in data:
                    trend = data['trend']
                    logger.info(f"  趋势方向: {trend['direction']}")
                    logger.info(f"  趋势强度: {trend['strength']*100:.1f}%")
                    logger.info(f"  波动性: {trend['volatility']*100:.1f}%")
    
    # 显示综合结果
    if 'combined' in analysis_results:
        logger.info("\n综合分析结果:")
        for symbol, score in analysis_results['combined'].get('symbols', {}).items():
            logger.info(f"  {symbol} 评分: {score:.1f}/100")
    
    return analysis_results

def main():
    """主函数"""
    logger.info("=" * 50)
    logger.info("量子核心系统演示")
    logger.info("=" * 50)
    
    try:
        # 创建示例市场数据
        logger.info("创建示例市场数据...")
        market_data = create_sample_market_data()
        
        # 初始化组件
        logger.info("初始化量子核心组件...")
        
        # 初始化量子后端
        backend = QuantumBackend(backend_type='simulator')
        backend.start()
        
        # 初始化市场到量子转换器
        converter = MarketToQuantumConverter()
        converter.start()
        
        # 初始化量子解释器
        interpreter = QuantumInterpreter()
        interpreter.start()
        
        # 初始化市场分析器
        analyzer = MultidimensionalAnalyzer()
        analyzer.start()
        
        # 运行演示
        quantum_result = demo_quantum_circuit(backend)
        amp_result, angle_result = demo_market_to_quantum(converter, market_data)
        prob_interp, threshold_interp = demo_quantum_interpreter(interpreter, quantum_result)
        analysis_results = demo_market_analysis(analyzer, market_data)
        
        logger.info("\n" + "=" * 50)
        logger.info("演示完成")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"演示过程中出错: {str(e)}", exc_info=True)
        return 1
    finally:
        # 停止所有组件
        logger.info("停止量子核心组件...")
        if 'analyzer' in locals():
            analyzer.stop()
        if 'interpreter' in locals():
            interpreter.stop()
        if 'converter' in locals():
            converter.stop()
        if 'backend' in locals():
            backend.stop()
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 