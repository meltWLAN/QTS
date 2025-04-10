#!/usr/bin/env python3
"""
量子核心系统仪表盘 - 通过Web界面可视化展示量子核心系统功能
"""

from flask import Flask, render_template, jsonify, request
import logging
import pandas as pd
import numpy as np
import json
import os
import time
import threading

# 导入量子核心组件
from quantum_core.quantum_backend import QuantumBackend
from quantum_core.quantum_interpreter import QuantumInterpreter
from quantum_core.market_to_quantum import MarketToQuantumConverter
from quantum_core.multidimensional_analysis import MultidimensionalAnalyzer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("QuantumDashboard")

# 创建Flask应用
app = Flask(__name__)
app.config['SECRET_KEY'] = 'quantum-dashboard-secret-key'

# 全局状态
system_state = {
    'backend': None,
    'converter': None,
    'interpreter': None,
    'analyzer': None,
    'market_data': None,
    'quantum_results': {},
    'analysis_results': {},
    'latest_circuit_id': None,
    'latest_job_id': None,
    'components_status': {
        'backend': 'stopped',
        'converter': 'stopped',
        'interpreter': 'stopped',
        'analyzer': 'stopped'
    }
}

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

@app.route('/')
def index():
    """首页"""
    return render_template('index.html')

@app.route('/api/system/start', methods=['POST'])
def start_system():
    """启动量子核心系统"""
    try:
        # 初始化量子后端
        system_state['backend'] = QuantumBackend(backend_type='simulator')
        system_state['backend'].start()
        system_state['components_status']['backend'] = 'running'
        
        # 初始化市场到量子转换器
        system_state['converter'] = MarketToQuantumConverter()
        system_state['converter'].start()
        system_state['components_status']['converter'] = 'running'
        
        # 初始化量子解释器
        system_state['interpreter'] = QuantumInterpreter()
        system_state['interpreter'].start()
        system_state['components_status']['interpreter'] = 'running'
        
        # 初始化市场分析器
        system_state['analyzer'] = MultidimensionalAnalyzer()
        system_state['analyzer'].start()
        system_state['components_status']['analyzer'] = 'running'
        
        # 创建示例市场数据
        system_state['market_data'] = create_sample_market_data()
        
        return jsonify({'status': 'success', 'message': '量子核心系统已启动'})
    except Exception as e:
        logger.error(f"启动系统时出错: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'启动失败: {str(e)}'})

@app.route('/api/system/stop', methods=['POST'])
def stop_system():
    """停止量子核心系统"""
    try:
        # 停止所有组件
        for component in ['analyzer', 'interpreter', 'converter', 'backend']:
            if system_state[component]:
                system_state[component].stop()
                system_state['components_status'][component] = 'stopped'
        
        return jsonify({'status': 'success', 'message': '量子核心系统已停止'})
    except Exception as e:
        logger.error(f"停止系统时出错: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'停止失败: {str(e)}'})

@app.route('/api/system/status', methods=['GET'])
def get_system_status():
    """获取系统状态"""
    return jsonify({
        'status': 'success',
        'components': system_state['components_status'],
        'has_market_data': system_state['market_data'] is not None
    })

@app.route('/api/quantum/create_circuit', methods=['POST'])
def create_quantum_circuit():
    """创建量子电路"""
    if not system_state['backend'] or system_state['components_status']['backend'] != 'running':
        return jsonify({'status': 'error', 'message': '量子后端未运行'})
        
    try:
        # 创建Bell态电路
        circuit_id = system_state['backend'].create_circuit(2, "Bell状态")
        
        # 添加量子门
        system_state['backend'].add_gate(circuit_id, 'H', [0])  # Hadamard门
        system_state['backend'].add_gate(circuit_id, 'CX', [0, 1])  # CNOT门
        
        # 添加测量
        system_state['backend'].add_measurement(circuit_id, 0)
        system_state['backend'].add_measurement(circuit_id, 1)
        
        system_state['latest_circuit_id'] = circuit_id
        
        return jsonify({
            'status': 'success', 
            'circuit_id': circuit_id,
            'message': 'Bell状态量子电路已创建'
        })
    except Exception as e:
        logger.error(f"创建量子电路时出错: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'创建失败: {str(e)}'})

@app.route('/api/quantum/execute_circuit', methods=['POST'])
def execute_quantum_circuit():
    """执行量子电路"""
    if not system_state['backend'] or system_state['components_status']['backend'] != 'running':
        return jsonify({'status': 'error', 'message': '量子后端未运行'})
        
    try:
        circuit_id = system_state['latest_circuit_id']
        if not circuit_id:
            return jsonify({'status': 'error', 'message': '没有可用的电路'})
            
        # 执行电路
        job_id = system_state['backend'].execute_circuit(circuit_id, shots=1024)
        system_state['latest_job_id'] = job_id
        
        # 等待执行完成
        def wait_for_completion():
            while True:
                status = system_state['backend'].get_job_status(job_id)
                if status['status'] in ['completed', 'failed', 'error']:
                    break
                time.sleep(0.1)
                
            if status['status'] == 'completed':
                # 获取结果
                result = system_state['backend'].get_result(job_id)
                system_state['quantum_results'][job_id] = result
        
        # 在后台线程等待完成
        thread = threading.Thread(target=wait_for_completion)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'success', 
            'job_id': job_id,
            'message': '量子电路执行中'
        })
    except Exception as e:
        logger.error(f"执行量子电路时出错: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'执行失败: {str(e)}'})

@app.route('/api/quantum/job_status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """获取作业状态"""
    if not system_state['backend'] or system_state['components_status']['backend'] != 'running':
        return jsonify({'status': 'error', 'message': '量子后端未运行'})
        
    try:
        status = system_state['backend'].get_job_status(job_id)
        
        # 如果已完成，添加结果
        result = None
        if status['status'] == 'completed' and job_id in system_state['quantum_results']:
            result = system_state['quantum_results'][job_id]
            
        return jsonify({
            'status': 'success',
            'job_status': status,
            'result': result
        })
    except Exception as e:
        logger.error(f"获取作业状态时出错: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'获取状态失败: {str(e)}'})

@app.route('/api/market/convert', methods=['POST'])
def convert_market_data():
    """转换市场数据"""
    if (not system_state['converter'] or 
        system_state['components_status']['converter'] != 'running' or
        not system_state['market_data']):
        return jsonify({'status': 'error', 'message': '转换器未运行或没有市场数据'})
        
    try:
        method = request.json.get('method', 'amplitude')
        
        # 执行转换
        result = system_state['converter'].convert(
            system_state['market_data'], 
            method=method
        )
        
        return jsonify({
            'status': 'success',
            'conversion_result': result
        })
    except Exception as e:
        logger.error(f"转换市场数据时出错: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'转换失败: {str(e)}'})

@app.route('/api/quantum/interpret', methods=['POST'])
def interpret_quantum_result():
    """解释量子结果"""
    if (not system_state['interpreter'] or 
        system_state['components_status']['interpreter'] != 'running'):
        return jsonify({'status': 'error', 'message': '解释器未运行'})
        
    try:
        job_id = system_state['latest_job_id']
        if not job_id or job_id not in system_state['quantum_results']:
            return jsonify({'status': 'error', 'message': '没有可用的量子结果'})
            
        method = request.json.get('method', 'probability')
        
        # 执行解释
        result = system_state['interpreter'].interpret(
            system_state['quantum_results'][job_id],
            method=method
        )
        
        return jsonify({
            'status': 'success',
            'interpretation': result
        })
    except Exception as e:
        logger.error(f"解释量子结果时出错: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'解释失败: {str(e)}'})

@app.route('/api/market/analyze', methods=['POST'])
def analyze_market_data():
    """分析市场数据"""
    if (not system_state['analyzer'] or 
        system_state['components_status']['analyzer'] != 'running' or
        not system_state['market_data']):
        return jsonify({'status': 'error', 'message': '分析器未运行或没有市场数据'})
        
    try:
        # 添加分析维度
        system_state['analyzer'].add_dimension('price_trend', description='价格趋势分析')
        system_state['analyzer'].add_dimension('volatility', description='波动性分析')
        
        # 执行分析
        results = system_state['analyzer'].analyze(system_state['market_data'])
        system_state['analysis_results'] = results
        
        # 转换结果为可序列化格式
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable_results[key] = value
                
        return jsonify({
            'status': 'success',
            'analysis_results': serializable_results
        })
    except Exception as e:
        logger.error(f"分析市场数据时出错: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'分析失败: {str(e)}'})

@app.route('/api/market/data', methods=['GET'])
def get_market_data():
    """获取市场数据"""
    if not system_state['market_data']:
        return jsonify({'status': 'error', 'message': '没有市场数据'})
        
    try:
        # 转换市场数据为可JSON序列化格式
        serialized_data = {}
        for symbol, df in system_state['market_data'].items():
            serialized_data[symbol] = {
                'dates': df.index.strftime('%Y-%m-%d').tolist(),
                'open': df['open'].tolist(),
                'high': df['high'].tolist(),
                'low': df['low'].tolist(),
                'close': df['close'].tolist(),
                'volume': df['volume'].tolist()
            }
            
        return jsonify({
            'status': 'success',
            'market_data': serialized_data
        })
    except Exception as e:
        logger.error(f"获取市场数据时出错: {str(e)}", exc_info=True)
        return jsonify({'status': 'error', 'message': f'获取失败: {str(e)}'})

if __name__ == '__main__':
    # 确保模板目录存在
    if not os.path.exists('templates'):
        os.makedirs('templates')
        
    # 创建静态资源目录
    if not os.path.exists('static'):
        os.makedirs('static')
        if not os.path.exists('static/js'):
            os.makedirs('static/js')
        if not os.path.exists('static/css'):
            os.makedirs('static/css')
            
    app.run(debug=True, host='0.0.0.0', port=5000) 