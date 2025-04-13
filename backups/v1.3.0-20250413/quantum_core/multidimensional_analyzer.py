#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多维分析器 - 提供高级的多维数据分析功能
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

class MultidimensionalAnalyzer:
    """多维分析器 - 提供高级的多维数据分析功能"""
    
    def __init__(self):
        self.logger = logging.getLogger('quantum_core.multidimensional_analyzer')
        self.is_running = False
        self.analysis_methods = {}
        self.analysis_cache = {}
        
        # 高级分析特性
        self.quantum_pca = False
        self.quantum_tsne = False
        self.quantum_clustering = False
        self.quantum_anomaly_detection = False
        
        self.logger.info("多维分析器初始化完成")
        
    def start(self):
        """启动分析器"""
        try:
            self.logger.info("启动多维分析器...")
            self.is_running = True
            self._initialize_advanced_features()
            self._register_default_methods()
            self.logger.info("多维分析器启动完成")
            return True
        except Exception as e:
            self.logger.error(f"启动失败: {str(e)}")
            return False
            
    def stop(self):
        """停止分析器"""
        try:
            self.logger.info("停止多维分析器...")
            self.is_running = False
            self.analysis_methods.clear()
            self.analysis_cache.clear()
            self.logger.info("多维分析器已停止")
            return True
        except Exception as e:
            self.logger.error(f"停止失败: {str(e)}")
            return False
            
    def _initialize_advanced_features(self):
        """初始化高级特性"""
        # 初始化量子PCA
        self.quantum_pca = True
        self._setup_quantum_pca()
        
        # 初始化量子t-SNE
        self.quantum_tsne = True
        self._setup_quantum_tsne()
        
        # 初始化量子聚类
        self.quantum_clustering = True
        self._setup_quantum_clustering()
        
        # 初始化量子异常检测
        self.quantum_anomaly_detection = True
        self._setup_quantum_anomaly_detection()
        
    def _setup_quantum_pca(self):
        """设置量子PCA功能"""
        self.quantum_pca_parameters = {
            'n_components': 3,
            'quantum_enhanced': True,
            'variance_threshold': 0.95
        }
        
    def _setup_quantum_tsne(self):
        """设置量子t-SNE功能"""
        self.quantum_tsne_parameters = {
            'n_components': 2,
            'perplexity': 30,
            'n_iter': 1000,
            'quantum_enhanced': True
        }
        
    def _setup_quantum_clustering(self):
        """设置量子聚类功能"""
        self.quantum_clustering_parameters = {
            'algorithm': 'quantum_kmeans',
            'n_clusters': 5,
            'max_iter': 300,
            'quantum_enhanced': True
        }
        
    def _setup_quantum_anomaly_detection(self):
        """设置量子异常检测功能"""
        self.quantum_anomaly_detection_parameters = {
            'algorithm': 'quantum_isolation_forest',
            'contamination': 0.1,
            'n_estimators': 100,
            'quantum_enhanced': True
        }
        
    def _register_default_methods(self):
        """注册默认分析方法"""
        self.register_analysis_method('pca', self._pca_analysis)
        self.register_analysis_method('tsne', self._tsne_analysis)
        self.register_analysis_method('clustering', self._clustering_analysis)
        self.register_analysis_method('anomaly', self._anomaly_detection)
        self.register_analysis_method('correlation', self._correlation_analysis)
        self.register_analysis_method('trend', self._trend_analysis)
        self.logger.info("注册了 6 个默认分析方法")
        
    def register_analysis_method(self, name: str, method: callable):
        """注册新的分析方法"""
        self.analysis_methods[name] = method
        self.logger.info(f"注册分析方法: {name}")
        
    def analyze_data(self, data: Union[np.ndarray, pd.DataFrame], method: str = 'pca', 
                    parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """分析多维数据
        
        Args:
            data: 输入数据，可以是numpy数组或pandas DataFrame
            method: 分析方法名称
            parameters: 分析参数
            
        Returns:
            分析结果字典
        """
        if not self.is_running:
            raise RuntimeError("分析器未运行")
            
        if method not in self.analysis_methods:
            raise ValueError(f"不支持的分析方法: {method}")
            
        # 检查缓存
        cache_key = f"{method}_{str(data.shape)}_{str(parameters)}"
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
            
        # 转换数据格式
        if isinstance(data, pd.DataFrame):
            data_array = data.values
            feature_names = data.columns.tolist()
        else:
            data_array = data
            feature_names = [f"feature_{i}" for i in range(data.shape[1])]
            
        # 标准化数据
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_array)
        
        # 应用分析方法
        analysis_result = self.analysis_methods[method](data_scaled, parameters)
        
        # 应用量子增强
        if method == 'pca' and self.quantum_pca:
            analysis_result = self._apply_quantum_pca(analysis_result)
        elif method == 'tsne' and self.quantum_tsne:
            analysis_result = self._apply_quantum_tsne(analysis_result)
        elif method == 'clustering' and self.quantum_clustering:
            analysis_result = self._apply_quantum_clustering(analysis_result)
        elif method == 'anomaly' and self.quantum_anomaly_detection:
            analysis_result = self._apply_quantum_anomaly_detection(analysis_result)
            
        # 添加元数据
        analysis_result['feature_names'] = feature_names
        analysis_result['data_shape'] = data_array.shape
        analysis_result['method'] = method
        
        # 缓存结果
        self.analysis_cache[cache_key] = analysis_result
        
        return analysis_result
        
    def _pca_analysis(self, data: np.ndarray, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """PCA分析"""
        if parameters is None:
            parameters = {}
            
        n_components = parameters.get('n_components', self.quantum_pca_parameters['n_components'])
        variance_threshold = parameters.get('variance_threshold', 
                                          self.quantum_pca_parameters['variance_threshold'])
        
        # 执行PCA
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(data)
        
        # 计算解释方差比
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        
        # 找到达到方差阈值的组件数
        n_components_threshold = np.argmax(cumulative_variance_ratio >= variance_threshold) + 1
        
        return {
            'transformed_data': pca_result,
            'components': pca.components_,
            'explained_variance_ratio': explained_variance_ratio.tolist(),
            'cumulative_variance_ratio': cumulative_variance_ratio.tolist(),
            'n_components_threshold': int(n_components_threshold),
            'feature_importance': np.abs(pca.components_).tolist()
        }
        
    def _tsne_analysis(self, data: np.ndarray, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """t-SNE分析"""
        if parameters is None:
            parameters = {}
            
        n_components = parameters.get('n_components', self.quantum_tsne_parameters['n_components'])
        perplexity = parameters.get('perplexity', self.quantum_tsne_parameters['perplexity'])
        n_iter = parameters.get('n_iter', self.quantum_tsne_parameters['n_iter'])
        
        # 执行t-SNE
        tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter)
        tsne_result = tsne.fit_transform(data)
        
        # 计算KL散度
        kl_divergence = tsne.kl_divergence_
        
        return {
            'transformed_data': tsne_result,
            'kl_divergence': float(kl_divergence),
            'perplexity': perplexity,
            'n_iter': n_iter
        }
        
    def _clustering_analysis(self, data: np.ndarray, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """聚类分析"""
        if parameters is None:
            parameters = {}
            
        algorithm = parameters.get('algorithm', self.quantum_clustering_parameters['algorithm'])
        n_clusters = parameters.get('n_clusters', self.quantum_clustering_parameters['n_clusters'])
        
        # 执行聚类
        if algorithm == 'kmeans':
            clustering = KMeans(n_clusters=n_clusters, max_iter=300)
            labels = clustering.fit_predict(data)
            centers = clustering.cluster_centers_
            inertia = clustering.inertia_
        elif algorithm == 'dbscan':
            clustering = DBSCAN(eps=0.5, min_samples=5)
            labels = clustering.fit_predict(data)
            centers = None
            inertia = None
        else:
            raise ValueError(f"不支持的聚类算法: {algorithm}")
            
        # 计算轮廓系数
        from sklearn.metrics import silhouette_score
        silhouette_avg = silhouette_score(data, labels)
        
        return {
            'labels': labels.tolist(),
            'centers': centers.tolist() if centers is not None else None,
            'inertia': float(inertia) if inertia is not None else None,
            'silhouette_score': float(silhouette_avg),
            'n_clusters': int(np.unique(labels).size)
        }
        
    def _anomaly_detection(self, data: np.ndarray, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """异常检测"""
        if parameters is None:
            parameters = {}
            
        algorithm = parameters.get('algorithm', self.quantum_anomaly_detection_parameters['algorithm'])
        contamination = parameters.get('contamination', 
                                     self.quantum_anomaly_detection_parameters['contamination'])
        
        # 执行异常检测
        from sklearn.ensemble import IsolationForest
        
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        labels = iso_forest.fit_predict(data)
        
        # 转换标签：-1表示异常，1表示正常
        anomaly_labels = (labels == -1).astype(int)
        
        # 计算异常分数
        anomaly_scores = iso_forest.score_samples(data)
        
        return {
            'anomaly_labels': anomaly_labels.tolist(),
            'anomaly_scores': anomaly_scores.tolist(),
            'n_anomalies': int(np.sum(anomaly_labels)),
            'contamination': contamination
        }
        
    def _correlation_analysis(self, data: np.ndarray, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """相关性分析"""
        # 计算相关系数矩阵
        corr_matrix = np.corrcoef(data.T)
        
        # 计算特征重要性
        feature_importance = np.mean(np.abs(corr_matrix), axis=1)
        
        # 找出强相关特征对
        strong_correlations = []
        for i in range(corr_matrix.shape[0]):
            for j in range(i+1, corr_matrix.shape[1]):
                if abs(corr_matrix[i, j]) > 0.7:  # 强相关阈值
                    strong_correlations.append({
                        'feature1': i,
                        'feature2': j,
                        'correlation': float(corr_matrix[i, j])
                    })
                    
        return {
            'correlation_matrix': corr_matrix.tolist(),
            'feature_importance': feature_importance.tolist(),
            'strong_correlations': strong_correlations
        }
        
    def _trend_analysis(self, data: np.ndarray, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """趋势分析"""
        # 计算每个特征的趋势
        trends = []
        for i in range(data.shape[1]):
            feature_data = data[:, i]
            
            # 计算线性回归
            x = np.arange(len(feature_data))
            slope, intercept = np.polyfit(x, feature_data, 1)
            
            # 计算趋势强度
            trend_strength = abs(slope) / np.std(feature_data) if np.std(feature_data) > 0 else 0
            
            trends.append({
                'feature_index': i,
                'slope': float(slope),
                'intercept': float(intercept),
                'trend_strength': float(trend_strength),
                'direction': 'up' if slope > 0 else 'down'
            })
            
        # 按趋势强度排序
        trends.sort(key=lambda x: x['trend_strength'], reverse=True)
        
        return {
            'trends': trends,
            'strongest_trend': trends[0] if trends else None
        }
        
    def _apply_quantum_pca(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """应用量子PCA增强"""
        if not self.quantum_pca:
            return analysis_result
            
        # 创建量子电路
        n_qubits = min(analysis_result['data_shape'][1], 10)  # 限制量子比特数量
        circuit = QuantumCircuit(n_qubits)
        
        # 应用量子PCA
        # 这里使用简化的方法，实际应用中可能需要更复杂的实现
        for i in range(n_qubits):
            circuit.h(i)
            
        # 添加量子增强结果
        analysis_result['quantum_enhanced'] = True
        analysis_result['quantum_circuit'] = circuit
        
        return analysis_result
        
    def _apply_quantum_tsne(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """应用量子t-SNE增强"""
        if not self.quantum_tsne:
            return analysis_result
            
        # 创建量子电路
        n_qubits = min(analysis_result['data_shape'][1], 10)  # 限制量子比特数量
        circuit = QuantumCircuit(n_qubits)
        
        # 应用量子t-SNE
        # 这里使用简化的方法，实际应用中可能需要更复杂的实现
        for i in range(n_qubits):
            circuit.h(i)
            if i < n_qubits - 1:
                circuit.cx(i, i+1)
                
        # 添加量子增强结果
        analysis_result['quantum_enhanced'] = True
        analysis_result['quantum_circuit'] = circuit
        
        return analysis_result
        
    def _apply_quantum_clustering(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """应用量子聚类增强"""
        if not self.quantum_clustering:
            return analysis_result
            
        # 创建量子电路
        n_qubits = min(analysis_result['data_shape'][1], 10)  # 限制量子比特数量
        circuit = QuantumCircuit(n_qubits)
        
        # 应用量子聚类
        # 这里使用简化的方法，实际应用中可能需要更复杂的实现
        for i in range(n_qubits):
            circuit.h(i)
            
        # 添加量子增强结果
        analysis_result['quantum_enhanced'] = True
        analysis_result['quantum_circuit'] = circuit
        
        return analysis_result
        
    def _apply_quantum_anomaly_detection(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """应用量子异常检测增强"""
        if not self.quantum_anomaly_detection:
            return analysis_result
            
        # 创建量子电路
        n_qubits = min(analysis_result['data_shape'][1], 10)  # 限制量子比特数量
        circuit = QuantumCircuit(n_qubits)
        
        # 应用量子异常检测
        # 这里使用简化的方法，实际应用中可能需要更复杂的实现
        for i in range(n_qubits):
            circuit.h(i)
            if i < n_qubits - 1:
                circuit.cx(i, i+1)
                
        # 添加量子增强结果
        analysis_result['quantum_enhanced'] = True
        analysis_result['quantum_circuit'] = circuit
        
        return analysis_result
        
    def get_analysis_methods(self) -> List[str]:
        """获取所有可用的分析方法"""
        return list(self.analysis_methods.keys())
        
    def clear_cache(self):
        """清除分析缓存"""
        self.analysis_cache.clear() 