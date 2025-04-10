import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import talib
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from .tushare_data_connector import TushareDataConnector

class QuantumPredictor:
    """超神量子预测引擎核心"""
    
    def __init__(self):
        self.data_connector = TushareDataConnector()
        self.scaler = MinMaxScaler()
        self.quantum_params = {
            'coherence': 0.7,  # 量子相干性
            'superposition': 0.5,  # 量子叠加态
            'entanglement': 0.8,  # 量子纠缠度
        }
        self.setup_quantum_network()
        
    def setup_quantum_network(self):
        """初始化量子神经网络"""
        self.quantum_model = tf.keras.Sequential([
            tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(60, 12)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(64, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(3)  # 预测未来3个时间点
        ])
        self.quantum_model.compile(optimizer='adam', loss='mse')
        
    def get_top_recommendations(self, top_n=5):
        """获取最优推荐股票
        
        Args:
            top_n: 返回推荐数量
            
        Returns:
            list: 推荐股票列表
        """
        try:
            # 获取沪深300成分股
            stocks = self.data_connector.get_hs300_stocks()
            recommendations = []
            
            for stock in stocks[:50]:  # 为了性能，先分析前50只股票
                analysis = self.analyze_stock(stock['code'])
                if analysis['quantum_score'] > 0.7:  # 量子评分阈值
                    recommendations.append({
                        'code': stock['code'],
                        'name': stock['name'],
                        'quantum_score': analysis['quantum_score'],
                        'current_price': analysis['current_price'],
                        'prediction': analysis['prediction']
                    })
            
            # 按量子评分排序
            recommendations.sort(key=lambda x: x['quantum_score'], reverse=True)
            return recommendations[:top_n]
            
        except Exception as e:
            print(f"获取推荐股票失败: {str(e)}")
            return []
            
    def analyze_stock(self, stock_code):
        """深度分析单只股票
        
        Args:
            stock_code: 股票代码
            
        Returns:
            dict: 分析结果
        """
        try:
            # 获取历史数据
            hist_data = self.data_connector.get_stock_history(stock_code, days=120)
            
            # 技术指标计算
            close_prices = hist_data['close'].values
            volumes = hist_data['volume'].values
            
            # 计算技术指标
            macd, signal, hist = talib.MACD(close_prices)
            rsi = talib.RSI(close_prices)
            upper, middle, lower = talib.BBANDS(close_prices)
            
            # 量子特征提取
            quantum_features = self.extract_quantum_features(
                close_prices, volumes, macd, rsi, hist
            )
            
            # 价格预测
            future_prices = self.predict_prices(quantum_features)
            
            # 计算量子评分
            quantum_score = self.calculate_quantum_score(
                quantum_features, 
                future_prices,
                self.quantum_params
            )
            
            current_price = close_prices[-1]
            prediction = {
                '7d': future_prices[0],
                '30d': future_prices[1],
                '90d': future_prices[2]
            }
            
            return {
                'quantum_score': quantum_score,
                'current_price': current_price,
                'prediction': prediction,
                'features': quantum_features
            }
            
        except Exception as e:
            print(f"股票分析失败 {stock_code}: {str(e)}")
            return None
            
    def extract_quantum_features(self, prices, volumes, macd, rsi, hist):
        """提取量子特征
        
        Args:
            prices: 价格序列
            volumes: 成交量序列
            macd: MACD指标
            rsi: RSI指标
            hist: MACD柱状图
            
        Returns:
            dict: 量子特征
        """
        # 价格动量
        returns = np.diff(prices) / prices[:-1]
        momentum = np.mean(returns[-5:])
        
        # 波动率
        volatility = np.std(returns) * np.sqrt(252)
        
        # 成交量趋势
        volume_ma5 = np.mean(volumes[-5:])
        volume_ma20 = np.mean(volumes[-20:])
        volume_trend = volume_ma5 / volume_ma20 - 1
        
        # 技术指标趋势
        macd_trend = macd[-1] - macd[-5]
        rsi_trend = rsi[-1] - rsi[-5]
        
        # 量子态特征
        quantum_state = {
            'momentum': momentum,
            'volatility': volatility,
            'volume_trend': volume_trend,
            'macd_trend': macd_trend,
            'rsi_trend': rsi_trend,
            'price_position': (prices[-1] - np.min(prices)) / (np.max(prices) - np.min(prices)),
            'volume_position': (volumes[-1] - np.min(volumes)) / (np.max(volumes) - np.min(volumes))
        }
        
        return quantum_state
        
    def calculate_quantum_score(self, features, predictions, params):
        """计算量子评分
        
        Args:
            features: 量子特征
            predictions: 预测价格
            params: 量子参数
            
        Returns:
            float: 量子评分 (0-1)
        """
        # 趋势得分
        trend_score = (
            features['momentum'] * 0.3 +
            features['macd_trend'] * 0.2 +
            features['rsi_trend'] * 0.2 +
            features['volume_trend'] * 0.3
        )
        
        # 量子态得分
        quantum_state_score = (
            (1 - features['volatility']) * params['coherence'] +
            features['price_position'] * params['superposition'] +
            features['volume_position'] * params['entanglement']
        ) / (params['coherence'] + params['superposition'] + params['entanglement'])
        
        # 预测得分
        pred_returns = [
            (pred - predictions['7d']) / predictions['7d']
            for pred in [predictions['30d'], predictions['90d']]
        ]
        prediction_score = np.mean(pred_returns) if all(pred_returns) else 0
        
        # 综合评分
        final_score = (
            trend_score * 0.4 +
            quantum_state_score * 0.4 +
            prediction_score * 0.2
        )
        
        # 归一化到0-1区间
        return max(0, min(1, (final_score + 1) / 2))
        
    def predict_prices(self, features):
        """预测未来价格
        
        Args:
            features: 量子特征
            
        Returns:
            dict: 不同时间周期的预测价格
        """
        # 准备预测数据
        X = np.array([list(features.values())])
        X = self.scaler.fit_transform(X)
        
        # 使用量子模型预测
        predictions = self.quantum_model.predict(X)
        
        # 反归一化
        predictions = self.scaler.inverse_transform(predictions)
        
        return {
            '7d': predictions[0][0],
            '30d': predictions[0][1],
            '90d': predictions[0][2]
        }
        
    def analyze_quantum_state(self, stock_code):
        """分析股票量子态
        
        Args:
            stock_code: 股票代码
            
        Returns:
            dict: 量子态分析结果
        """
        try:
            analysis = self.analyze_stock(stock_code)
            if not analysis:
                return {'strength': 0}
                
            # 计算量子态强度
            features = analysis['features']
            strength = (
                abs(features['momentum']) * 0.3 +
                (1 - features['volatility']) * 0.2 +
                abs(features['volume_trend']) * 0.2 +
                features['price_position'] * 0.15 +
                features['volume_position'] * 0.15
            )
            
            return {
                'strength': strength,
                'stability': 1 - features['volatility'],
                'momentum': features['momentum'],
                'volume_trend': features['volume_trend']
            }
            
        except Exception as e:
            print(f"量子态分析失败 {stock_code}: {str(e)}")
            return {'strength': 0}
            
    def predict_multi_dimensional(self, stock_code, period='短期(7天)'):
        """多维度预测
        
        Args:
            stock_code: 股票代码
            period: 预测周期
            
        Returns:
            dict: 预测结果
        """
        try:
            analysis = self.analyze_stock(stock_code)
            if not analysis:
                return {}
                
            predictions = analysis['prediction']
            current_price = analysis['current_price']
            
            # 计算预测置信度
            confidence = min(1.0, analysis['quantum_score'] * 1.2)
            
            # 计算不同周期的预测
            results = {
                '7天': {
                    'price': predictions['7d'],
                    'change': (predictions['7d'] - current_price) / current_price * 100
                },
                '30天': {
                    'price': predictions['30d'],
                    'change': (predictions['30d'] - current_price) / current_price * 100
                },
                '90天': {
                    'price': predictions['90d'],
                    'change': (predictions['90d'] - current_price) / current_price * 100
                },
                'confidence': confidence
            }
            
            return results
            
        except Exception as e:
            print(f"多维度预测失败 {stock_code}: {str(e)}")
            return {} 