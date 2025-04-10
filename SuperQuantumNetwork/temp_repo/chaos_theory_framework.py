#!/usr/bin/env python3
"""
超神量子共生系统 - 混沌理论分析框架
用于识别市场的非线性特征、吸引子和分形模式
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import traceback

logger = logging.getLogger("ChaosTheory")

class ChaosTheoryAnalyzer:
    """混沌理论分析引擎"""
    
    def __init__(self):
        """初始化混沌理论分析器"""
        self.logger = logging.getLogger("ChaosTheory")
        
        # 混沌分析参数
        self.params = {
            'embedding_dimension': 3,  # 嵌入维度
            'time_delay': 2,           # 时间延迟
            'lyapunov_max_steps': 100, # 计算Lyapunov指数的最大步数
            'hurst_min_lag': 2,        # 计算Hurst指数的最小滞后
            'hurst_max_lag': 20,       # 计算Hurst指数的最大滞后
            'entropy_bins': 50,        # 计算熵值的分箱数
            'fractal_min_scale': 5,    # 分形分析的最小尺度
            'fractal_max_scale': 50,   # 分形分析的最大尺度
            'critical_point_window': 10, # 临界点检测窗口
            'attractor_sample_limit': 5000, # 吸引子采样限制
        }
        
        # 分析结果
        self.results = {
            'hurst_exponent': 0.5,
            'fractal_dimension': 0.0,
            'lyapunov_exponent': 0.0,
            'entropy': 0.0,
            'complexity': 0.0,
            'attractors': [],
            'critical_points': [],
            'fractal_patterns': [],
            'market_regime': 'unknown',
            'stability': 0.5,
            'last_update': None
        }
        
        self.logger.info("混沌理论分析器初始化完成")
    
    def analyze(self, price_data, additional_data=None):
        """
        执行混沌理论全面分析
        
        参数:
            price_data: numpy数组或DataFrame，价格时间序列
            additional_data: 可选，附加数据
            
        返回:
            dict: 分析结果
        """
        self.logger.info("开始混沌理论全面分析...")
        
        try:
            # 预处理数据
            if isinstance(price_data, pd.DataFrame):
                if 'close' in price_data.columns:
                    series = price_data['close'].values
                else:
                    series = price_data.iloc[:, 0].values
            else:
                series = np.array(price_data)
            
            # 计算Hurst指数
            self.results['hurst_exponent'] = self._calculate_hurst_exponent(series)
            
            # 计算分形维度
            self.results['fractal_dimension'] = self._calculate_fractal_dimension(series)
            
            # 计算Lyapunov指数
            self.results['lyapunov_exponent'] = self._calculate_lyapunov_exponent(series)
            
            # 计算熵和复杂度
            self.results['entropy'] = self._calculate_entropy(series)
            self.results['complexity'] = self._calculate_complexity(series)
            
            # 重构相空间
            self.results['attractors'] = self._reconstruct_phase_space(
                series, 
                dimension=self.params['embedding_dimension'], 
                tau=self.params['time_delay']
            )
            
            # 检测临界点
            self.results['critical_points'] = self._detect_critical_points(series)
            
            # 识别分形模式
            self.results['fractal_patterns'] = self._identify_fractal_patterns(series)
            
            # 确定市场状态
            self.results['market_regime'] = self._determine_market_regime(
                self.results['hurst_exponent'],
                self.results['lyapunov_exponent'],
                self.results['entropy']
            )
            
            # 计算系统稳定性
            self.results['stability'] = self._calculate_stability(
                self.results['hurst_exponent'],
                self.results['lyapunov_exponent'],
                self.results['entropy'],
                self.results['fractal_dimension']
            )
            
            # 更新时间戳
            self.results['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            self.logger.info("混沌理论分析完成")
            return self.results
            
        except Exception as e:
            self.logger.error(f"混沌理论分析错误: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None
    
    def _calculate_hurst_exponent(self, series):
        """
        计算Hurst指数
        H < 0.5: 反持续序列，反转趋势
        H = 0.5: 随机游走，无记忆特性
        H > 0.5: 持续序列，趋势持续
        """
        try:
            # 确保输入有效
            if len(series) < self.params['hurst_max_lag'] * 2:
                return 0.5
            
            # 计算不同滞后下的R/S统计量
            lags = range(self.params['hurst_min_lag'], self.params['hurst_max_lag'])
            tau = []; # R/S统计量
            
            # 计算每个滞后的R/S值
            for lag in lags:
                # 将序列分割
                segments = len(series) // lag
                if segments < 1:
                    continue
                    
                rs_values = []
                for i in range(segments):
                    segment = series[i*lag:(i+1)*lag]
                    
                    # 计算均值和标准差
                    mean = np.mean(segment)
                    std = np.std(segment)
                    if std == 0:
                        continue
                        
                    # 均值调整
                    adjusted = segment - mean
                    
                    # 累积偏差
                    cumulative = np.cumsum(adjusted)
                    
                    # 计算范围R
                    r = max(cumulative) - min(cumulative)
                    
                    # 计算R/S值
                    rs = r / std if std > 0 else 0
                    rs_values.append(rs)
                
                if rs_values:
                    tau.append([lag, np.mean(rs_values)])
            
            # 回归分析
            if len(tau) < 4:
                return 0.5
                
            x = np.log10([t[0] for t in tau])
            y = np.log10([t[1] for t in tau])
            
            slope, _, _, _, _ = stats.linregress(x, y)
            
            return slope
        except Exception as e:
            self.logger.error(f"计算Hurst指数错误: {str(e)}")
            return 0.5
    
    def _calculate_fractal_dimension(self, series):
        """计算分形维度，使用盒计数法"""
        try:
            # 确保输入有效
            if len(series) < self.params['fractal_max_scale']:
                return 1.0
            
            # 标准化数据到[0,1]区间
            min_val = min(series)
            max_val = max(series)
            if max_val == min_val:
                return 1.0
                
            normalized = (series - min_val) / (max_val - min_val)
            
            # 计算不同尺度下的盒计数
            scales = np.logspace(
                np.log10(self.params['fractal_min_scale']), 
                np.log10(self.params['fractal_max_scale']), 
                num=10
            )
            counts = []
            
            for scale in scales:
                scale = int(scale)
                if scale < 1:
                    continue
                    
                # 对时间序列进行分箱
                bins = np.linspace(0, 1, scale)
                histogram, _ = np.histogram(normalized, bins=bins)
                
                # 计算非空盒数
                count = np.sum(histogram > 0)
                counts.append(count)
            
            # 回归分析
            if len(counts) < 4:
                return 1.0
                
            scales = scales[:len(counts)]
            x = np.log(1/scales)
            y = np.log(counts)
            
            slope, _, _, _, _ = stats.linregress(x, y)
            
            return slope
        except Exception as e:
            self.logger.error(f"计算分形维度错误: {str(e)}")
            return 1.0
    
    def _calculate_lyapunov_exponent(self, series):
        """计算莱雅普诺夫指数，衡量系统对初始条件的敏感性"""
        try:
            # 确保输入有效
            if len(series) < 100:
                return 0.0
                
            # 获取参数
            max_steps = min(self.params['lyapunov_max_steps'], len(series) // 2)
            
            # 计算时间序列的差分
            d = np.diff(series)
            
            # 初始化变量
            n = len(d)
            epsilon = np.std(d) / 10
            
            # 随机选择参考点
            lyapunov_sum = 0.0
            count = 0
            
            for i in range(0, n-max_steps, max_steps):
                x0 = d[i]
                # 寻找最近邻点
                distances = np.abs(d - x0)
                distances[i] = np.inf  # 排除自身
                nearest_idx = np.argmin(distances)
                
                if nearest_idx + max_steps >= n:
                    continue
                    
                # 初始距离
                d0 = abs(d[nearest_idx] - x0)
                if d0 < epsilon and d0 > 0:
                    # 跟踪轨迹分离
                    path1 = d[i:i+max_steps]
                    path2 = d[nearest_idx:nearest_idx+max_steps]
                    
                    # 计算每一步的距离
                    step_distances = np.abs(path1 - path2)
                    valid_distances = step_distances[step_distances > 0]
                    
                    if len(valid_distances) == 0:
                        continue
                        
                    # 计算轨迹分离的对数
                    lyapunov = np.log(np.mean(valid_distances) / d0) / max_steps
                    lyapunov_sum += lyapunov
                    count += 1
            
            # 计算平均值
            if count > 0:
                return lyapunov_sum / count
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"计算莱雅普诺夫指数错误: {str(e)}")
            return 0.0
    
    def _calculate_entropy(self, series):
        """计算信息熵，量化系统复杂度"""
        try:
            # 确保输入有效
            if len(series) < 50:
                return 0.0
                
            # 对数据进行分箱
            hist, _ = np.histogram(series, bins=self.params['entropy_bins'])
            
            # 计算概率分布
            distribution = hist / np.sum(hist)
            
            # 去除零概率
            distribution = distribution[distribution > 0]
            
            # 计算香农熵
            entropy = -np.sum(distribution * np.log2(distribution))
            
            # 归一化到[0,1]
            max_entropy = np.log2(self.params['entropy_bins'])
            if max_entropy == 0:
                return 0.0
                
            return entropy / max_entropy
            
        except Exception as e:
            self.logger.error(f"计算熵错误: {str(e)}")
            return 0.0
    
    def _calculate_complexity(self, series):
        """计算系统复杂度，基于熵和自组织临界性"""
        try:
            # 使用熵和Hurst指数的组合
            entropy = self._calculate_entropy(series)
            hurst = self._calculate_hurst_exponent(series)
            
            # 计算复杂度 - 最大复杂度在Hurst = 0.5(边缘混沌)和高熵的情况下
            # 复杂度 = 4 * H * (1-H) * Entropy
            complexity = 4 * hurst * (1 - hurst) * entropy
            
            return complexity
            
        except Exception as e:
            self.logger.error(f"计算复杂度错误: {str(e)}")
            return 0.0
    
    def _reconstruct_phase_space(self, series, dimension=3, tau=2):
        """
        重构相空间，用于识别吸引子
        
        参数:
            series: 时间序列
            dimension: 嵌入维度
            tau: 时间延迟
            
        返回:
            list: 相空间中的点
        """
        try:
            # 确保输入有效
            if len(series) < dimension * tau:
                return []
                
            # 限制点数以避免内存问题
            limit = self.params['attractor_sample_limit']
            if len(series) > limit:
                # 均匀采样
                indices = np.linspace(0, len(series)-1, limit, dtype=int)
                series = series[indices]
            
            # 重构相空间
            points = []
            for i in range(len(series) - (dimension-1) * tau):
                point = [series[i + j*tau] for j in range(dimension)]
                points.append(point)
                
            return points
            
        except Exception as e:
            self.logger.error(f"重构相空间错误: {str(e)}")
            return []
    
    def _detect_critical_points(self, series):
        """
        检测临界点(可能的相变点)
        
        参数:
            series: 时间序列
            
        返回:
            list: 临界点的索引和得分
        """
        try:
            # 确保输入有效
            if len(series) < 3 * self.params['critical_point_window']:
                return []
                
            # 使用滑动窗口计算局部熵和复杂度变化
            window = self.params['critical_point_window']
            critical_points = []
            
            for i in range(window, len(series) - window):
                # 计算前后窗口的特性
                pre_window = series[i-window:i]
                post_window = series[i:i+window]
                
                # 计算特征
                pre_entropy = self._calculate_entropy(pre_window)
                post_entropy = self._calculate_entropy(post_window)
                
                pre_hurst = self._calculate_hurst_exponent(pre_window)
                post_hurst = self._calculate_hurst_exponent(post_window)
                
                # 计算变化
                entropy_change = abs(post_entropy - pre_entropy)
                hurst_change = abs(post_hurst - pre_hurst)
                
                # 临界点得分 (熵变化和Hurst指数变化的加权和)
                score = 0.7 * entropy_change + 0.3 * hurst_change
                
                # 如果得分高于阈值，标记为临界点
                if score > 0.15:  # 阈值可调
                    critical_points.append((i, score))
            
            # 按得分排序
            critical_points.sort(key=lambda x: x[1], reverse=True)
            
            # 合并接近的临界点
            merged_points = []
            if critical_points:
                current_point = critical_points[0]
                for i in range(1, len(critical_points)):
                    if critical_points[i][0] - current_point[0] < window // 2:
                        # 如果接近，保留得分更高的
                        if critical_points[i][1] > current_point[1]:
                            current_point = critical_points[i]
                    else:
                        merged_points.append(current_point)
                        current_point = critical_points[i]
                
                merged_points.append(current_point)
            
            return merged_points
            
        except Exception as e:
            self.logger.error(f"检测临界点错误: {str(e)}")
            return []
    
    def _identify_fractal_patterns(self, series):
        """
        识别分形模式
        
        参数:
            series: 时间序列
            
        返回:
            list: 识别的分形模式及其位置
        """
        try:
            # 确保输入有效
            if len(series) < 50:
                return []
                
            # 主要分形模式
            patterns = {
                'w_bottom': '底部W形',
                'm_top': '顶部M形',
                'head_shoulders_top': '头肩顶',
                'head_shoulders_bottom': '头肩底',
                'v_bottom': 'V形底',
                'v_top': 'V形顶',
                'fractal_triangle': '分形三角形',
                'fractal_channel': '分形通道',
                'elliott_5waves': '艾略特五浪',
                'self_similarity': '自相似区域'
            }
            
            # 这里实现简单的模式识别
            # 注意：完整的分形模式识别相当复杂，这里只实现简化版
            fractal_patterns = []
            
            # 检测W形底部
            w_bottoms = self._detect_w_bottom(series)
            for position, confidence in w_bottoms:
                fractal_patterns.append({
                    'type': 'w_bottom',
                    'name': patterns['w_bottom'],
                    'position': position,
                    'confidence': confidence
                })
            
            # 检测M形顶部
            m_tops = self._detect_m_top(series)
            for position, confidence in m_tops:
                fractal_patterns.append({
                    'type': 'm_top',
                    'name': patterns['m_top'],
                    'position': position,
                    'confidence': confidence
                })
            
            # 添加其他模式检测...
            
            return fractal_patterns
            
        except Exception as e:
            self.logger.error(f"识别分形模式错误: {str(e)}")
            return []
    
    def _detect_w_bottom(self, series):
        """检测W形底部模式"""
        # 简化实现
        patterns = []
        window = 15  # 检测窗口大小
        
        for i in range(window, len(series) - window):
            left_window = series[i-window:i]
            right_window = series[i:i+window]
            
            if len(left_window) < 5 or len(right_window) < 5:
                continue
                
            # 查找左窗口和右窗口的局部最小值
            left_min_idx = np.argmin(left_window)
            right_min_idx = np.argmin(right_window) + i
            
            # 获取中间点位置和值
            mid_point = series[i]
            
            # 检查是否形成W形状
            if left_min_idx > 0 and right_min_idx < len(series) - 1:
                left_min = left_window[left_min_idx]
                right_min = right_window[np.argmin(right_window)]
                
                # 检查W形状条件: 两个低点和中间的反弹
                if mid_point > left_min and mid_point > right_min and \
                   abs(left_min - right_min) / max(abs(left_min), abs(right_min)) < 0.1:
                    
                    # 计算模式置信度
                    confidence = 0.6 + 0.4 * (mid_point - min(left_min, right_min)) / mid_point
                    patterns.append((i, min(1.0, confidence)))
        
        return patterns
    
    def _detect_m_top(self, series):
        """检测M形顶部模式"""
        # 与W形底部类似，但寻找的是局部最大值
        patterns = []
        window = 15  # 检测窗口大小
        
        for i in range(window, len(series) - window):
            left_window = series[i-window:i]
            right_window = series[i:i+window]
            
            if len(left_window) < 5 or len(right_window) < 5:
                continue
                
            # 查找左窗口和右窗口的局部最大值
            left_max_idx = np.argmax(left_window)
            right_max_idx = np.argmax(right_window) + i
            
            # 获取中间点位置和值
            mid_point = series[i]
            
            # 检查是否形成M形状
            if left_max_idx > 0 and right_max_idx < len(series) - 1:
                left_max = left_window[left_max_idx]
                right_max = right_window[np.argmax(right_window)]
                
                # 检查M形状条件: 两个高点和中间的回落
                if mid_point < left_max and mid_point < right_max and \
                   abs(left_max - right_max) / max(abs(left_max), abs(right_max)) < 0.1:
                    
                    # 计算模式置信度
                    confidence = 0.6 + 0.4 * (min(left_max, right_max) - mid_point) / min(left_max, right_max)
                    patterns.append((i, min(1.0, confidence)))
        
        return patterns
    
    def _determine_market_regime(self, hurst, lyapunov, entropy):
        """
        根据混沌理论指标确定市场状态
        
        参数:
            hurst: Hurst指数
            lyapunov: 莱雅普诺夫指数
            entropy: 信息熵
            
        返回:
            str: 市场状态描述
        """
        # 趋势状态 (Hurst > 0.6)
        if hurst > 0.6:
            if entropy > 0.7:
                return "complex_trending"  # 复杂趋势
            else:
                return "trending"  # 简单趋势
                
        # 反转状态 (Hurst < 0.4)
        elif hurst < 0.4:
            if lyapunov > 0.01:
                return "chaotic_reverting"  # 混沌反转
            else:
                return "mean_reverting"  # 均值回归
                
        # 随机游走或混沌区域 (0.4 <= Hurst <= 0.6)
        else:
            if lyapunov > 0.01:
                if entropy > 0.6:
                    return "edge_of_chaos"  # 混沌边缘
                else:
                    return "mildly_chaotic"  # 轻度混沌
            else:
                return "random_walk"  # 随机游走
    
    def _calculate_stability(self, hurst, lyapunov, entropy, fractal_dim):
        """
        计算市场的稳定性指标
        
        参数:
            hurst: Hurst指数
            lyapunov: 莱雅普诺夫指数
            entropy: 信息熵
            fractal_dim: 分形维度
            
        返回:
            float: 市场稳定性 (0-1)，1表示最稳定
        """
        # 稳定市场的特征:
        # - 高Hurst指数(趋势持续)或非常低(强均值回归)
        # - 低莱雅普诺夫指数(低敏感性)
        # - 中等或低熵(低复杂度)
        # - 低分形维度(简单结构)
        
        # Hurst指数对稳定性的贡献 (U形函数)
        hurst_factor = 1 - 4 * (hurst - 0.5)**2
        
        # 莱雅普诺夫指数对稳定性的贡献 (负相关)
        lyapunov_factor = max(0, 1 - 50 * abs(lyapunov))
        
        # 熵对稳定性的贡献 (负相关)
        entropy_factor = 1 - entropy
        
        # 分形维度对稳定性的贡献 (负相关)
        fractal_factor = max(0, 2 - fractal_dim)
        
        # 计算加权平均
        stability = (
            0.3 * hurst_factor + 
            0.3 * lyapunov_factor + 
            0.2 * entropy_factor + 
            0.2 * fractal_factor
        )
        
        return max(0, min(1, stability))
    
    def plot_phase_space(self, filename=None):
        """绘制相空间（吸引子）图"""
        if not self.results['attractors'] or len(self.results['attractors'][0]) < 3:
            self.logger.warning("没有足够的数据绘制相空间图")
            return False
            
        try:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # 提取坐标
            x = [pt[0] for pt in self.results['attractors']]
            y = [pt[1] for pt in self.results['attractors']]
            z = [pt[2] for pt in self.results['attractors']]
            
            # 绘制散点图
            sc = ax.scatter(x, y, z, c=z, cmap='viridis', s=5, alpha=0.6)
            
            # 添加信息
            ax.set_title('市场混沌吸引子', fontsize=14)
            ax.set_xlabel('X(t)')
            ax.set_ylabel('X(t+τ)')
            ax.set_zlabel('X(t+2τ)')
            
            # 添加颜色条
            plt.colorbar(sc, ax=ax, label='值')
            
            # 保存或显示图表
            if filename:
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
                return True
            else:
                plt.show()
                return True
                
        except Exception as e:
            self.logger.error(f"绘制相空间图错误: {str(e)}")
            return False

# 全局混沌分析器
_chaos_analyzer = None

def get_chaos_analyzer():
    """获取全局混沌理论分析器实例"""
    global _chaos_analyzer
    if _chaos_analyzer is None:
        _chaos_analyzer = ChaosTheoryAnalyzer()
    return _chaos_analyzer

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    # 测试混沌理论分析器
    analyzer = get_chaos_analyzer()
    
    # 生成示例数据 - 混沌Logistic映射
    r = 3.9  # 参数值，r>3.57时系统进入混沌
    n = 1000
    x = np.zeros(n)
    x[0] = 0.1  # 初始值
    
    for i in range(1, n):
        x[i] = r * x[i-1] * (1 - x[i-1])
    
    # 执行分析
    results = analyzer.analyze(x)
    
    # 打印结果
    print(f"Hurst指数: {results['hurst_exponent']:.3f}")
    print(f"分形维度: {results['fractal_dimension']:.3f}")
    print(f"莱雅普诺夫指数: {results['lyapunov_exponent']:.6f}")
    print(f"熵值: {results['entropy']:.3f}")
    print(f"复杂度: {results['complexity']:.3f}")
    print(f"市场状态: {results['market_regime']}")
    print(f"稳定性: {results['stability']:.3f}")
    print(f"检测到的临界点数量: {len(results['critical_points'])}")
    print(f"识别的分形模式数量: {len(results['fractal_patterns'])}")
    
    # 绘制相空间
    analyzer.plot_phase_space("chaos_attractor_test.png") 