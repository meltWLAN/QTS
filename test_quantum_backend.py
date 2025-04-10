#!/usr/bin/env python3
"""
量子后端测试脚本 - 测试量子后端核心功能
"""

import sys
import time
import logging
from quantum_core.quantum_backend import QuantumBackend

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("QuantumBackendTest")

def test_basic_circuit():
    """测试基本量子电路"""
    logger.info("测试基本量子电路...")
    
    # 创建量子后端
    backend = QuantumBackend(backend_type='simulator')
    
    # 启动后端
    backend.start()
    
    try:
        # 创建一个2量子比特的电路
        circuit_id = backend.create_circuit(2, "Bell状态电路")
        logger.info(f"创建电路ID: {circuit_id}")
        
        # 添加量子门操作
        # 将第一个量子比特置于叠加态
        backend.add_gate(circuit_id, 'H', [0])
        # 添加CNOT门，将两个量子比特纠缠
        backend.add_gate(circuit_id, 'CX', [0, 1])
        # 添加测量
        backend.add_measurement(circuit_id, 0)
        backend.add_measurement(circuit_id, 1)
        
        # 执行电路
        logger.info("执行量子电路...")
        job_id = backend.execute_circuit(circuit_id, shots=1024)
        logger.info(f"作业ID: {job_id}")
        
        # 等待执行完成
        status = backend.get_job_status(job_id)
        while status['status'] not in ['completed', 'failed', 'error']:
            time.sleep(0.1)
            status = backend.get_job_status(job_id)
            
        # 获取结果
        result = backend.get_result(job_id)
        logger.info(f"执行结果: {result}")
        
        # 验证结果
        if 'counts' in result:
            counts = result['counts']
            # Bell状态应该只有 |00> 和 |11> 两种结果
            expected_results = {'00', '11'}
            actual_results = set(counts.keys())
            
            if actual_results.issubset(expected_results) and len(actual_results) > 0:
                logger.info("Bell状态测试通过!")
                return True
            else:
                logger.error(f"测量结果不符合预期: {actual_results}")
                return False
        else:
            logger.error(f"执行失败: {result}")
            return False
    finally:
        # 停止后端
        backend.stop()

def test_rz_gate():
    """测试RZ门"""
    logger.info("测试RZ门...")
    
    # 创建量子后端
    backend = QuantumBackend(backend_type='simulator')
    
    # 启动后端
    backend.start()
    
    try:
        # 创建一个1量子比特的电路
        circuit_id = backend.create_circuit(1, "RZ测试电路")
        
        # 添加量子门操作
        # 将量子比特置于叠加态
        backend.add_gate(circuit_id, 'H', [0])
        # 添加RZ门
        backend.add_gate(circuit_id, 'RZ', [0], {'theta': 3.14159})
        # 添加测量
        backend.add_measurement(circuit_id, 0)
        
        # 执行电路
        logger.info("执行RZ测试电路...")
        job_id = backend.execute_circuit(circuit_id, shots=1024)
        
        # 等待执行完成
        status = backend.get_job_status(job_id)
        while status['status'] not in ['completed', 'failed', 'error']:
            time.sleep(0.1)
            status = backend.get_job_status(job_id)
            
        # 获取结果
        result = backend.get_result(job_id)
        logger.info(f"RZ测试结果: {result}")
        
        # 结果不需要严格验证，因为这取决于相位，但应该有结果
        if 'counts' in result and len(result['counts']) > 0:
            logger.info("RZ门测试通过!")
            return True
        else:
            logger.error(f"执行失败: {result}")
            return False
    finally:
        # 停止后端
        backend.stop()

def test_backend_info():
    """测试后端信息获取"""
    logger.info("测试后端信息获取...")
    
    # 创建量子后端
    backend = QuantumBackend(backend_type='simulator')
    
    # 获取未启动状态的信息
    info_before = backend.get_backend_info()
    logger.info(f"启动前后端信息: {info_before}")
    
    # 启动后端
    backend.start()
    
    try:
        # 获取启动后的信息
        info_after = backend.get_backend_info()
        logger.info(f"启动后后端信息: {info_after}")
        
        # 验证信息
        if (info_after['is_running'] and
            info_after['type'] == 'simulator' and
            info_after['max_qubits'] > 0):
            logger.info("后端信息测试通过!")
            return True
        else:
            logger.error("后端信息不符合预期")
            return False
    finally:
        # 停止后端
        backend.stop()

def main():
    """主函数"""
    logger.info("=" * 50)
    logger.info("量子后端测试 - 开始")
    logger.info("=" * 50)
    
    # 执行测试
    tests = [
        ("基本电路测试", test_basic_circuit),
        ("RZ门测试", test_rz_gate),
        ("后端信息测试", test_backend_info)
    ]
    
    results = {}
    all_passed = True
    
    for name, test_func in tests:
        logger.info(f"\n{'-' * 30}\n测试 {name}\n{'-' * 30}")
        try:
            result = test_func()
            results[name] = result
            if not result:
                all_passed = False
        except Exception as e:
            logger.error(f"测试 {name} 时出错: {str(e)}", exc_info=True)
            results[name] = False
            all_passed = False
    
    # 显示结果摘要
    logger.info("\n" + "=" * 50)
    logger.info("测试结果摘要")
    logger.info("=" * 50)
    
    for name, result in results.items():
        status = "通过" if result else "失败"
        logger.info(f"{name}: {status}")
    
    if all_passed:
        logger.info("\n所有测试通过，量子后端运行正常！")
        return 0
    else:
        logger.warning("\n部分测试失败，请检查日志了解详情。")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 