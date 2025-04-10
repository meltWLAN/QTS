import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger("QuantumAnalysis")

class QuantumAnalysisEngine:
    """量子分析引擎 - 核心分析模块"""
    
    def __init__(self, data_connector):
        """
        初始化量子分析引擎
        
        参数:
            data_connector: 数据连接器实例
        """
        self.data_connector = data_connector
        self.analysis_cache = {}
        self.technical_indicators = {}
        self.fundamental_indicators = {}
        self.market_sentiment = {}
        
    def analyze_stock(self, code: str, start_date: str = None, end_date: str = None) -> Dict:
        """
        全面分析股票
        
        参数:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        返回:
            Dict: 分析结果
        """
        # 1. 获取基础数据
        daily_data = self.data_connector.get_market_data(code, start_date, end_date)
        daily_basic = self.data_connector.get_daily_basic(code, start_date, end_date)
        
        # 2. 技术分析
        technical_analysis = self._analyze_technical(daily_data)
        
        # 3. 基本面分析
        fundamental_analysis = self._analyze_fundamental(daily_basic)
        
        # 4. 资金面分析
        money_flow_analysis = self._analyze_money_flow(code, start_date, end_date)
        
        # 5. 市场情绪分析
        sentiment_analysis = self._analyze_market_sentiment(code, start_date, end_date)
        
        # 6. 生成综合评分
        overall_score = self._calculate_overall_score(
            technical_analysis,
            fundamental_analysis,
            money_flow_analysis,
            sentiment_analysis
        )
        
        return {
            'code': code,
            'technical_analysis': technical_analysis,
            'fundamental_analysis': fundamental_analysis,
            'money_flow_analysis': money_flow_analysis,
            'sentiment_analysis': sentiment_analysis,
            'overall_score': overall_score,
            'recommendation': self._generate_recommendation(overall_score),
            'risk_level': self._calculate_risk_level(overall_score),
            'update_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _analyze_technical(self, data: pd.DataFrame) -> Dict:
        """技术分析"""
        if data is None or data.empty:
            return {}
            
        # 计算技术指标
        indicators = {}
        
        # 1. 趋势指标
        indicators['ma'] = self._calculate_ma(data)
        indicators['macd'] = self._calculate_macd(data)
        indicators['kdj'] = self._calculate_kdj(data)
        
        # 2. 波动指标
        indicators['boll'] = self._calculate_boll(data)
        indicators['atr'] = self._calculate_atr(data)
        
        # 3. 成交量指标
        indicators['volume_ma'] = self._calculate_volume_ma(data)
        indicators['obv'] = self._calculate_obv(data)
        
        # 4. 动量指标
        indicators['rsi'] = self._calculate_rsi(data)
        indicators['cci'] = self._calculate_cci(data)
        
        return indicators
    
    def _analyze_fundamental(self, data: pd.DataFrame) -> Dict:
        """基本面分析"""
        if data is None or data.empty:
            return {}
            
        # 计算基本面指标
        indicators = {}
        
        # 1. 估值指标
        indicators['pe'] = data['pe'].iloc[-1] if 'pe' in data.columns else None
        indicators['pb'] = data['pb'].iloc[-1] if 'pb' in data.columns else None
        indicators['ps'] = data['ps'].iloc[-1] if 'ps' in data.columns else None
        
        # 2. 成长指标
        indicators['revenue_yoy'] = data['revenue_yoy'].iloc[-1] if 'revenue_yoy' in data.columns else None
        indicators['profit_yoy'] = data['profit_yoy'].iloc[-1] if 'profit_yoy' in data.columns else None
        
        # 3. 质量指标
        indicators['roe'] = data['roe'].iloc[-1] if 'roe' in data.columns else None
        indicators['roa'] = data['roa'].iloc[-1] if 'roa' in data.columns else None
        
        return indicators
    
    def _analyze_money_flow(self, code: str, start_date: str, end_date: str) -> Dict:
        """资金面分析"""
        try:
            # 获取资金流向数据
            money_flow = self.data_connector.pro.moneyflow(
                ts_code=code,
                start_date=start_date,
                end_date=end_date
            )
            
            if money_flow is None or money_flow.empty:
                return {}
                
            # 计算资金指标
            indicators = {}
            
            # 1. 主力资金
            indicators['net_mf_amount'] = money_flow['net_mf_amount'].iloc[-1]
            indicators['net_mf_vol'] = money_flow['net_mf_vol'].iloc[-1]
            
            # 2. 超大单资金
            indicators['net_xl_amount'] = money_flow['net_xl_amount'].iloc[-1]
            indicators['net_xl_vol'] = money_flow['net_xl_vol'].iloc[-1]
            
            # 3. 大单资金
            indicators['net_l_amount'] = money_flow['net_l_amount'].iloc[-1]
            indicators['net_l_vol'] = money_flow['net_l_vol'].iloc[-1]
            
            # 4. 中单资金
            indicators['net_m_amount'] = money_flow['net_m_amount'].iloc[-1]
            indicators['net_m_vol'] = money_flow['net_m_vol'].iloc[-1]
            
            # 5. 小单资金
            indicators['net_s_amount'] = money_flow['net_s_amount'].iloc[-1]
            indicators['net_s_vol'] = money_flow['net_s_vol'].iloc[-1]
            
            return indicators
            
        except Exception as e:
            logger.error(f"资金面分析出错: {str(e)}")
            return {}
    
    def _analyze_market_sentiment(self, code: str, start_date: str, end_date: str) -> Dict:
        """市场情绪分析"""
        try:
            # 获取龙虎榜数据
            top_list = self.data_connector.pro.top_list(
                ts_code=code,
                start_date=start_date,
                end_date=end_date
            )
            
            # 获取大宗交易数据
            block_trade = self.data_connector.pro.block_trade(
                ts_code=code,
                start_date=start_date,
                end_date=end_date
            )
            
            # 获取融资融券数据
            margin = self.data_connector.pro.margin(
                ts_code=code,
                start_date=start_date,
                end_date=end_date
            )
            
            sentiment = {
                'top_list': self._analyze_top_list(top_list) if top_list is not None else {},
                'block_trade': self._analyze_block_trade(block_trade) if block_trade is not None else {},
                'margin': self._analyze_margin(margin) if margin is not None else {}
            }
            
            return sentiment
            
        except Exception as e:
            logger.error(f"市场情绪分析出错: {str(e)}")
            return {}
    
    def _calculate_overall_score(self, technical: Dict, fundamental: Dict, 
                               money_flow: Dict, sentiment: Dict) -> float:
        """计算综合评分"""
        scores = []
        
        # 1. 技术面评分 (30%)
        if technical:
            tech_score = self._calculate_technical_score(technical)
            scores.append(('technical', tech_score, 0.3))
        
        # 2. 基本面评分 (30%)
        if fundamental:
            fund_score = self._calculate_fundamental_score(fundamental)
            scores.append(('fundamental', fund_score, 0.3))
        
        # 3. 资金面评分 (20%)
        if money_flow:
            money_score = self._calculate_money_flow_score(money_flow)
            scores.append(('money_flow', money_score, 0.2))
        
        # 4. 市场情绪评分 (20%)
        if sentiment:
            sent_score = self._calculate_sentiment_score(sentiment)
            scores.append(('sentiment', sent_score, 0.2))
        
        # 计算加权平均分
        if not scores:
            return 0.0
            
        total_score = sum(score * weight for _, score, weight in scores)
        return round(total_score, 2)
    
    def _generate_recommendation(self, score: float) -> str:
        """生成投资建议"""
        if score >= 80:
            return "强烈买入"
        elif score >= 60:
            return "买入"
        elif score >= 40:
            return "持有"
        elif score >= 20:
            return "卖出"
        else:
            return "强烈卖出"
    
    def _calculate_risk_level(self, score: float) -> str:
        """计算风险等级"""
        if score >= 80:
            return "低风险"
        elif score >= 60:
            return "中等风险"
        elif score >= 40:
            return "高风险"
        else:
            return "极高风险"
    
    # 技术指标计算方法
    def _calculate_ma(self, data: pd.DataFrame) -> Dict:
        """计算移动平均线"""
        return {
            'ma5': data['close'].rolling(window=5).mean().iloc[-1],
            'ma10': data['close'].rolling(window=10).mean().iloc[-1],
            'ma20': data['close'].rolling(window=20).mean().iloc[-1],
            'ma60': data['close'].rolling(window=60).mean().iloc[-1]
        }
    
    def _calculate_macd(self, data: pd.DataFrame) -> Dict:
        """计算MACD指标"""
        exp1 = data['close'].ewm(span=12, adjust=False).mean()
        exp2 = data['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        
        return {
            'macd': macd.iloc[-1],
            'signal': signal.iloc[-1],
            'hist': hist.iloc[-1]
        }
    
    def _calculate_kdj(self, data: pd.DataFrame) -> Dict:
        """计算KDJ指标"""
        low_list = data['low'].rolling(window=9, min_periods=9).min()
        high_list = data['high'].rolling(window=9, min_periods=9).max()
        rsv = (data['close'] - low_list) / (high_list - low_list) * 100
        
        k = pd.DataFrame(rsv).ewm(com=2).mean()
        d = k.ewm(com=2).mean()
        j = 3 * k - 2 * d
        
        return {
            'k': k.iloc[-1][0],
            'd': d.iloc[-1][0],
            'j': j.iloc[-1][0]
        }
    
    def _calculate_boll(self, data: pd.DataFrame) -> Dict:
        """计算布林带"""
        ma20 = data['close'].rolling(window=20).mean()
        std20 = data['close'].rolling(window=20).std()
        
        return {
            'upper': (ma20 + 2 * std20).iloc[-1],
            'middle': ma20.iloc[-1],
            'lower': (ma20 - 2 * std20).iloc[-1]
        }
    
    def _calculate_atr(self, data: pd.DataFrame) -> Dict:
        """计算ATR指标"""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(14).mean()
        
        return {'atr': atr.iloc[-1]}
    
    def _calculate_volume_ma(self, data: pd.DataFrame) -> Dict:
        """计算成交量均线"""
        return {
            'volume_ma5': data['vol'].rolling(window=5).mean().iloc[-1],
            'volume_ma10': data['vol'].rolling(window=10).mean().iloc[-1]
        }
    
    def _calculate_obv(self, data: pd.DataFrame) -> Dict:
        """计算OBV指标"""
        obv = (np.sign(data['close'].diff()) * data['vol']).fillna(0).cumsum()
        return {'obv': obv.iloc[-1]}
    
    def _calculate_rsi(self, data: pd.DataFrame) -> Dict:
        """计算RSI指标"""
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return {'rsi': rsi.iloc[-1]}
    
    def _calculate_cci(self, data: pd.DataFrame) -> Dict:
        """计算CCI指标"""
        tp = (data['high'] + data['low'] + data['close']) / 3
        ma_tp = tp.rolling(window=20).mean()
        md_tp = tp.rolling(window=20).apply(lambda x: pd.Series(x).mad())
        
        cci = (tp - ma_tp) / (0.015 * md_tp)
        return {'cci': cci.iloc[-1]}
    
    # 评分计算方法
    def _calculate_technical_score(self, technical: Dict) -> float:
        """计算技术面评分"""
        score = 0
        weights = {
            'ma': 0.2,
            'macd': 0.2,
            'kdj': 0.15,
            'boll': 0.15,
            'rsi': 0.15,
            'volume': 0.15
        }
        
        # MA评分
        if 'ma' in technical:
            ma_data = technical['ma']
            if ma_data['ma5'] > ma_data['ma10'] > ma_data['ma20']:
                score += weights['ma'] * 100
            elif ma_data['ma5'] > ma_data['ma10']:
                score += weights['ma'] * 60
            else:
                score += weights['ma'] * 20
        
        # MACD评分
        if 'macd' in technical:
            macd_data = technical['macd']
            if macd_data['hist'] > 0 and macd_data['macd'] > 0:
                score += weights['macd'] * 100
            elif macd_data['hist'] > 0:
                score += weights['macd'] * 60
            else:
                score += weights['macd'] * 20
        
        # 其他指标评分...
        
        return round(score, 2)
    
    def _calculate_fundamental_score(self, fundamental: Dict) -> float:
        """计算基本面评分"""
        score = 0
        weights = {
            'valuation': 0.3,
            'growth': 0.3,
            'quality': 0.4
        }
        
        # 估值评分
        if all(k in fundamental for k in ['pe', 'pb', 'ps']):
            pe_score = min(100, max(0, 50 - fundamental['pe']))
            pb_score = min(100, max(0, 30 - fundamental['pb']))
            ps_score = min(100, max(0, 20 - fundamental['ps']))
            score += weights['valuation'] * (pe_score + pb_score + ps_score) / 3
        
        # 成长性评分
        if all(k in fundamental for k in ['revenue_yoy', 'profit_yoy']):
            growth_score = (fundamental['revenue_yoy'] + fundamental['profit_yoy']) / 2
            score += weights['growth'] * min(100, max(0, growth_score))
        
        # 质量评分
        if all(k in fundamental for k in ['roe', 'roa']):
            quality_score = (fundamental['roe'] + fundamental['roa']) / 2
            score += weights['quality'] * min(100, max(0, quality_score))
        
        return round(score, 2)
    
    def _calculate_money_flow_score(self, money_flow: Dict) -> float:
        """计算资金面评分"""
        score = 0
        weights = {
            'net_mf_amount': 0.4,
            'net_xl_amount': 0.3,
            'net_l_amount': 0.3
        }
        
        # 主力资金评分
        if 'net_mf_amount' in money_flow:
            mf_score = min(100, max(0, money_flow['net_mf_amount'] / 1000000))
            score += weights['net_mf_amount'] * mf_score
        
        # 超大单资金评分
        if 'net_xl_amount' in money_flow:
            xl_score = min(100, max(0, money_flow['net_xl_amount'] / 500000))
            score += weights['net_xl_amount'] * xl_score
        
        # 大单资金评分
        if 'net_l_amount' in money_flow:
            l_score = min(100, max(0, money_flow['net_l_amount'] / 200000))
            score += weights['net_l_amount'] * l_score
        
        return round(score, 2)
    
    def _calculate_sentiment_score(self, sentiment: Dict) -> float:
        """计算市场情绪评分"""
        score = 0
        weights = {
            'top_list': 0.4,
            'block_trade': 0.3,
            'margin': 0.3
        }
        
        # 龙虎榜评分
        if 'top_list' in sentiment and sentiment['top_list']:
            top_score = min(100, max(0, sentiment['top_list'].get('net_amount', 0) / 1000000))
            score += weights['top_list'] * top_score
        
        # 大宗交易评分
        if 'block_trade' in sentiment and sentiment['block_trade']:
            block_score = min(100, max(0, sentiment['block_trade'].get('net_amount', 0) / 500000))
            score += weights['block_trade'] * block_score
        
        # 融资融券评分
        if 'margin' in sentiment and sentiment['margin']:
            margin_score = min(100, max(0, sentiment['margin'].get('net_amount', 0) / 200000))
            score += weights['margin'] * margin_score
        
        return round(score, 2)
    
    def _analyze_top_list(self, data: pd.DataFrame) -> Dict:
        """分析龙虎榜数据"""
        if data is None or data.empty:
            return {}
            
        latest = data.iloc[-1]
        return {
            'net_amount': latest['net_amount'] if 'net_amount' in latest else 0,
            'buy_amount': latest['buy_amount'] if 'buy_amount' in latest else 0,
            'sell_amount': latest['sell_amount'] if 'sell_amount' in latest else 0,
            'net_rate': latest['net_rate'] if 'net_rate' in latest else 0
        }
    
    def _analyze_block_trade(self, data: pd.DataFrame) -> Dict:
        """分析大宗交易数据"""
        if data is None or data.empty:
            return {}
            
        latest = data.iloc[-1]
        return {
            'net_amount': latest['net_amount'] if 'net_amount' in latest else 0,
            'price': latest['price'] if 'price' in latest else 0,
            'vol': latest['vol'] if 'vol' in latest else 0
        }
    
    def _analyze_margin(self, data: pd.DataFrame) -> Dict:
        """分析融资融券数据"""
        if data is None or data.empty:
            return {}
            
        latest = data.iloc[-1]
        return {
            'net_amount': latest['net_amount'] if 'net_amount' in latest else 0,
            'rzye': latest['rzye'] if 'rzye' in latest else 0,
            'rqye': latest['rqye'] if 'rqye' in latest else 0
        } 