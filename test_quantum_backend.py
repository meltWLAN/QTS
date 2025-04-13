#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试脚本，验证QuantumBackend类的run方法
"""

import sys
import time
from qiskit import QuantumCircuit

# 导入量子后端
sys.path.append('/Users/mac/Desktop/QTS')
from quantum_core.quantum_backend import QuantumBackend

def test_run_method():
    """测试QuantumBackend类的run方法"""
    # 创建量子后端
    backend = QuantumBackend()
    backend.start()
    
    print("量子后端已创建并启动")
    
    try:
        # 创建一个简单的Bell态电路
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure([0, 1], [0, 1])
        
        print("创建了Bell态电路")
        
        # 运行电路
        print("运行电路...")
        job = backend.run(circuit, shots=1024)
        
        print(f"作业已提交，ID: {job.job_id()}")
        print(f"作业状态: {job.status()}")
        
        # 获取结果
        result = job.result()
        
        if result is not None:
            print("成功获取到结果")
            # 如果有counts，打印它们
            if hasattr(result, 'get_counts'):
                counts = result.get_counts()
                print(f"测量结果计数: {counts}")
            else:
                print("结果对象没有get_counts方法")
        else:
            print("结果为空")
            
        print("测试完成")
        return True
        
    except Exception as e:
        print(f"测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 停止后端
        backend.stop()
        print("量子后端已停止")

if __name__ == "__main__":
    print("开始测试QuantumBackend.run()方法...")
    success = test_run_method()
    print(f"测试结果: {'成功' if success else '失败'}") 