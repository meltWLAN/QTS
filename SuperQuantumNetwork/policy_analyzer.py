#!/usr/bin/env python3
"""
超神量子共生系统 - 政策分析器
专门分析中国政策环境对A股市场的影响
"""

import logging
import re
import json
import numpy as np
from datetime import datetime, timedelta
import jieba
import jieba.analyse
from collections import Counter, defaultdict

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

logger = logging.getLogger("PolicyAnalyzer")

class PolicyAnalyzer:
    """政策分析器，专门分析政策对A股市场的影响"""
    
    def __init__(self, config=None):
        """
        初始化政策分析器
        
        参数:
            config: 配置信息
        """
        self.logger = logging.getLogger("PolicyAnalyzer")
        self.logger.info("初始化政策分析器...")
        
        # 默认配置
        self.config = {
            'keyword_weights': {
                '货币政策': 2.0,
                '降息': 1.8,
                '降准': 1.8,
                '流动性': 1.5,
                '扩张': 1.3,
                '刺激': 1.3,
                '宽松': 1.5,
                '支持': 1.0,
                '利好': 1.2,
                '减税': 1.4,
                '财政政策': 1.6,
                '加息': -1.8,
                '紧缩': -1.5,
                '收紧': -1.5,
                '调控': -1.2,
                '整顿': -1.4,
                '监管': -1.3,
                '处罚': -1.7,
                '加强': -1.0,
                '严格': -1.1
            },
            'sector_keywords': {
                '房地产': ['房地产', '楼市', '地产', '房价', '住房', '二手房', '商品房'],
                '银行': ['银行', '信贷', '存款', '贷款', '理财', '金融机构'],
                '医药': ['医药', '医疗', '药品', '疫苗', '医保', '生物医药'],
                '科技': ['科技', '互联网', '人工智能', '软件', '半导体', '芯片', '5G', '数字经济'],
                '新能源': ['新能源', '光伏', '风电', '氢能', '储能', '电动汽车', '新能源汽车'],
                '消费': ['消费', '零售', '电商', '服务业', '旅游', '餐饮'],
                '制造业': ['制造', '工业', '机械', '设备', '原材料', '钢铁', '有色金属'],
                '基建': ['基建', '基础设施', '铁路', '公路', '建筑', '城市建设', '水利']
            },
            'policy_memory_days': 30,
            'high_impact_threshold': 0.7,
            'policy_uncertainty_baseline': 0.5,
            'news_recency_decay': 0.9  # 新闻时效性衰减系数
        }
        
        # 更新配置
        if config:
            self.config.update(config)
            
        # 加载扩展词典
        self._init_jieba()
        
        # 初始化政策记忆库，用于追踪最近的政策新闻
        self.policy_memory = []
        
        # 政策周期状态跟踪
        self.policy_cycle = {
            'monetary': 0.0,  # -1到1，负表示紧缩，正表示宽松
            'fiscal': 0.0,    # -1到1，负表示紧缩，正表示宽松
            'regulatory': 0.0,  # -1到1，负表示严格，正表示宽松
            'industry_support': {}  # 行业政策支持度
        }
        
        # 政策变化检测
        self.last_policy_direction = 0.0
        self.policy_shift_detected = False
        self.policy_shift_magnitude = 0.0
        
        # 政策不确定性指标
        self.policy_uncertainty = self.config['policy_uncertainty_baseline']
        
        self.logger.info("政策分析器初始化完成")
    
    def _init_jieba(self):
        """初始化分词器"""
        # 加载扩展词典
        policy_terms = list(self.config['keyword_weights'].keys())
        for sector, keywords in self.config['sector_keywords'].items():
            policy_terms.extend(keywords)
        
        # 添加自定义词典
        for term in policy_terms:
            jieba.add_word(term)
        
        self.logger.info("分词器初始化完成")
    
    def analyze(self, data=None):
        """
        分析政策环境
        
        参数:
            data: 包含政策新闻和其他信息的字典
            
        返回:
            dict: 政策分析结果
        """
        self.logger.info("开始分析政策环境...")
        
        if not data:
            self.logger.warning("未提供政策数据，使用历史状态")
            return self._get_current_policy_state()
        
        # 处理新政策数据
        if 'policy_news' in data:
            self._process_policy_news(data['policy_news'])
        
        # 处理政策事件
        if 'policy_events' in data:
            self._process_policy_events(data['policy_events'])
        
        # 计算政策方向
        policy_direction = self._calculate_policy_direction()
        
        # 检测政策转变
        self._detect_policy_shift(policy_direction)
        
        # 更新政策不确定性
        self._update_policy_uncertainty()
        
        # 识别政策影响的行业
        impacted_sectors = self._identify_impacted_sectors()
        
        # 更新政策周期状态
        self._update_policy_cycle()
        
        # 构建分析结果
        analysis_result = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'policy_direction': policy_direction,
            'monetary_policy': self.policy_cycle['monetary'],
            'fiscal_policy': self.policy_cycle['fiscal'],
            'regulatory_policy': self.policy_cycle['regulatory'],
            'policy_uncertainty': self.policy_uncertainty,
            'policy_shift_detected': self.policy_shift_detected,
            'policy_shift_magnitude': self.policy_shift_magnitude,
            'policy_impact_sectors': impacted_sectors,
            'recent_policy_keywords': self._extract_top_keywords(10),
            'policy_news_count': len(self.policy_memory),
            'policy_recommendations': self._generate_policy_recommendations()
        }
        
        self.logger.info(f"政策分析完成，当前政策方向: {policy_direction:.2f}")
        return analysis_result
    
    def _process_policy_news(self, news_items):
        """处理政策新闻"""
        if not news_items:
            return
            
        current_time = datetime.now()
        
        for news in news_items:
            # 确保新闻项目格式正确
            if not isinstance(news, dict) or 'content' not in news:
                continue
                
            # 新闻基本信息
            news_content = news['content']
            news_time = news.get('time', current_time.strftime('%Y-%m-%d'))
            news_source = news.get('source', '未知来源')
            news_title = news.get('title', '')
            
            try:
                news_datetime = datetime.strptime(news_time, '%Y-%m-%d')
            except:
                news_datetime = current_time
            
            # 计算新闻时效性权重 (越近越重要)
            days_diff = (current_time - news_datetime).days
            if days_diff < 0:
                days_diff = 0
                
            recency_weight = self.config['news_recency_decay'] ** days_diff
            
            # 提取关键词并计算政策倾向
            keywords = jieba.analyse.extract_tags(news_content, topK=20, withWeight=True)
            policy_score = 0.0
            matched_sectors = Counter()
            
            # 分析政策关键词
            for word, weight in keywords:
                if word in self.config['keyword_weights']:
                    keyword_impact = self.config['keyword_weights'][word]
                    policy_score += keyword_impact * weight * recency_weight
                
                # 分析行业关键词
                for sector, sector_keywords in self.config['sector_keywords'].items():
                    if word in sector_keywords:
                        matched_sectors[sector] += weight * recency_weight
            
            # 计算标题的额外权重 (标题通常更重要)
            if news_title:
                title_keywords = jieba.analyse.extract_tags(news_title, topK=10, withWeight=True)
                for word, weight in title_keywords:
                    if word in self.config['keyword_weights']:
                        keyword_impact = self.config['keyword_weights'][word]
                        # 标题关键词权重加倍
                        policy_score += keyword_impact * weight * recency_weight * 2.0
                    
                    # 分析行业关键词
                    for sector, sector_keywords in self.config['sector_keywords'].items():
                        if word in sector_keywords:
                            matched_sectors[sector] += weight * recency_weight * 2.0
            
            # 归一化政策评分 (-1到1)
            policy_score = max(-1.0, min(1.0, policy_score / 5.0))
            
            # 提取主要影响行业
            top_sectors = [sector for sector, _ in matched_sectors.most_common(3)]
            
            # 保存到政策记忆
            self.policy_memory.append({
                'content': news_content,
                'title': news_title,
                'time': news_time,
                'source': news_source,
                'policy_score': policy_score,
                'impacted_sectors': top_sectors,
                'recency_weight': recency_weight,
                'keywords': dict(keywords[:10])
            })
        
        # 保持记忆大小在设定范围内
        memory_days = self.config['policy_memory_days']
        cutoff_date = (current_time - timedelta(days=memory_days)).strftime('%Y-%m-%d')
        
        self.policy_memory = [
            item for item in self.policy_memory 
            if item['time'] >= cutoff_date
        ]
        
        # 按时间排序
        self.policy_memory = sorted(self.policy_memory, key=lambda x: x['time'], reverse=True)
        
        self.logger.info(f"处理了 {len(news_items)} 条政策新闻，记忆库现有 {len(self.policy_memory)} 条")
    
    def _process_policy_events(self, events):
        """处理政策事件"""
        if not events:
            return
            
        for event in events:
            if not isinstance(event, dict):
                continue
                
            event_type = event.get('type', '')
            event_value = event.get('value', 0.0)
            
            # 处理货币政策事件
            if event_type == 'interest_rate_change':
                self.policy_cycle['monetary'] += event_value * 0.5
                
            elif event_type == 'reserve_ratio_change':
                self.policy_cycle['monetary'] += event_value * 0.6
                
            # 处理财政政策事件
            elif event_type == 'fiscal_stimulus':
                self.policy_cycle['fiscal'] += event_value * 0.7
                
            elif event_type == 'tax_policy_change':
                self.policy_cycle['fiscal'] += event_value * 0.4
                
            # 处理监管政策事件
            elif event_type == 'regulatory_change':
                self.policy_cycle['regulatory'] += event_value * 0.5
                
            # 处理行业政策事件
            elif event_type == 'industry_policy' and 'sector' in event:
                sector = event['sector']
                if sector not in self.policy_cycle['industry_support']:
                    self.policy_cycle['industry_support'][sector] = 0.0
                    
                self.policy_cycle['industry_support'][sector] += event_value * 0.6
        
        # 确保值在合理范围内
        self.policy_cycle['monetary'] = max(-1.0, min(1.0, self.policy_cycle['monetary']))
        self.policy_cycle['fiscal'] = max(-1.0, min(1.0, self.policy_cycle['fiscal']))
        self.policy_cycle['regulatory'] = max(-1.0, min(1.0, self.policy_cycle['regulatory']))
        
        for sector in self.policy_cycle['industry_support']:
            self.policy_cycle['industry_support'][sector] = max(
                -1.0, 
                min(1.0, self.policy_cycle['industry_support'][sector])
            )
            
        self.logger.info(f"处理了 {len(events)} 个政策事件")
    
    def _calculate_policy_direction(self):
        """计算整体政策方向"""
        if not self.policy_memory:
            # 使用现有政策周期状态
            monetary_weight = 0.4
            fiscal_weight = 0.3
            regulatory_weight = 0.3
            
            policy_direction = (
                self.policy_cycle['monetary'] * monetary_weight +
                self.policy_cycle['fiscal'] * fiscal_weight +
                self.policy_cycle['regulatory'] * regulatory_weight
            )
            
            return policy_direction
        
        # 基于记忆库计算政策方向
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for item in self.policy_memory:
            recency_weight = item['recency_weight']
            weighted_sum += item['policy_score'] * recency_weight
            weight_sum += recency_weight
        
        if weight_sum > 0:
            raw_direction = weighted_sum / weight_sum
        else:
            raw_direction = 0.0
        
        # 平滑处理，考虑已有政策周期状态
        current_state_weight = 0.3
        news_weight = 0.7
        
        current_state = (
            self.policy_cycle['monetary'] * 0.4 +
            self.policy_cycle['fiscal'] * 0.3 +
            self.policy_cycle['regulatory'] * 0.3
        )
        
        policy_direction = (
            current_state * current_state_weight +
            raw_direction * news_weight
        )
        
        return max(-1.0, min(1.0, policy_direction))
    
    def _detect_policy_shift(self, current_direction):
        """检测政策转变"""
        if self.last_policy_direction == 0.0:
            # 首次计算，无法检测变化
            self.last_policy_direction = current_direction
            return
        
        # 计算变化幅度
        direction_change = current_direction - self.last_policy_direction
        change_magnitude = abs(direction_change)
        
        # 检测是否达到转变阈值
        threshold = self.config['high_impact_threshold']
        if change_magnitude > threshold:
            self.policy_shift_detected = True
            self.policy_shift_magnitude = change_magnitude
            self.logger.info(f"检测到政策转变！幅度: {change_magnitude:.2f}")
        else:
            self.policy_shift_detected = False
            self.policy_shift_magnitude = change_magnitude
        
        # 更新上一次方向
        self.last_policy_direction = current_direction
    
    def _update_policy_uncertainty(self):
        """更新政策不确定性指标"""
        if not self.policy_memory:
            # 没有新数据，维持当前不确定性水平
            return
        
        # 计算政策评分的标准差
        if len(self.policy_memory) > 1:
            scores = [item['policy_score'] for item in self.policy_memory]
            std_dev = np.std(scores)
            
            # 计算政策方向的一致性
            recent_items = self.policy_memory[:min(10, len(self.policy_memory))]
            positive_count = sum(1 for item in recent_items if item['policy_score'] > 0.1)
            negative_count = sum(1 for item in recent_items if item['policy_score'] < -0.1)
            total_count = len(recent_items)
            
            # 方向一致性越高，不确定性越低
            if total_count > 0:
                consistency = max(positive_count, negative_count) / total_count
                inconsistency = 1 - consistency
            else:
                inconsistency = 0.5
            
            # 政策变化幅度也影响不确定性
            shift_impact = self.policy_shift_magnitude * 0.5
            
            # 更新不确定性指标
            raw_uncertainty = (std_dev * 0.4) + (inconsistency * 0.4) + (shift_impact * 0.2)
            
            # 平滑处理
            smooth_factor = 0.3
            self.policy_uncertainty = (
                self.policy_uncertainty * (1 - smooth_factor) +
                raw_uncertainty * smooth_factor
            )
            
            # 确保在有效范围内
            self.policy_uncertainty = max(0.1, min(1.0, self.policy_uncertainty))
    
    def _update_policy_cycle(self):
        """更新政策周期状态"""
        if not self.policy_memory:
            return
            
        # 计算近期政策对各类政策类型的影响
        monetary_signals = []
        fiscal_signals = []
        regulatory_signals = []
        
        # 货币政策关键词
        monetary_keywords = {'货币政策', '降息', '降准', '流动性', '加息', '央行', 'MLF', 'LPR', '公开市场操作'}
        
        # 财政政策关键词
        fiscal_keywords = {'财政政策', '减税', '财政支出', '补贴', '专项债', '赤字率', '政府债券', '扩张'}
        
        # 监管政策关键词
        regulatory_keywords = {'监管', '整顿', '调控', '处罚', '规范', '严格', '约束', '禁止', '限制'}
        
        for item in self.policy_memory:
            # 提取政策类型信号
            monetary_signal = 0.0
            fiscal_signal = 0.0
            regulatory_signal = 0.0
            
            # 根据关键词判断政策类型
            for word in item['keywords']:
                if word in monetary_keywords:
                    monetary_signal += item['policy_score'] * item['keywords'][word] * item['recency_weight']
                
                if word in fiscal_keywords:
                    fiscal_signal += item['policy_score'] * item['keywords'][word] * item['recency_weight']
                
                if word in regulatory_keywords:
                    regulatory_signal += item['policy_score'] * item['keywords'][word] * item['recency_weight']
            
            if abs(monetary_signal) > 0.01:
                monetary_signals.append(monetary_signal)
                
            if abs(fiscal_signal) > 0.01:
                fiscal_signals.append(fiscal_signal)
                
            if abs(regulatory_signal) > 0.01:
                regulatory_signals.append(regulatory_signal)
        
        # 计算各类政策的方向
        def calculate_direction(signals):
            if not signals:
                return 0.0
                
            # 使用加权平均
            weighted_sum = sum(signals)
            return max(-1.0, min(1.0, weighted_sum / len(signals)))
        
        # 更新政策周期状态，考虑平滑处理
        smooth_factor = 0.3
        
        if monetary_signals:
            new_monetary = calculate_direction(monetary_signals)
            self.policy_cycle['monetary'] = (
                self.policy_cycle['monetary'] * (1 - smooth_factor) +
                new_monetary * smooth_factor
            )
        
        if fiscal_signals:
            new_fiscal = calculate_direction(fiscal_signals)
            self.policy_cycle['fiscal'] = (
                self.policy_cycle['fiscal'] * (1 - smooth_factor) +
                new_fiscal * smooth_factor
            )
        
        if regulatory_signals:
            new_regulatory = calculate_direction(regulatory_signals)
            self.policy_cycle['regulatory'] = (
                self.policy_cycle['regulatory'] * (1 - smooth_factor) +
                new_regulatory * smooth_factor
            )
    
    def _identify_impacted_sectors(self):
        """识别政策影响的行业"""
        if not self.policy_memory:
            return []
            
        # 统计各行业受影响频率
        sector_mentions = Counter()
        sector_impact = defaultdict(float)
        
        # 分析最近的政策新闻
        recent_items = self.policy_memory[:min(20, len(self.policy_memory))]
        
        for item in recent_items:
            for sector in item['impacted_sectors']:
                sector_mentions[sector] += 1
                sector_impact[sector] += item['policy_score'] * item['recency_weight']
                
        # 更新行业政策支持度
        for sector, impact in sector_impact.items():
            if sector not in self.policy_cycle['industry_support']:
                self.policy_cycle['industry_support'][sector] = 0.0
                
            # 平滑处理
            smooth_factor = 0.3
            normalized_impact = max(-1.0, min(1.0, impact / max(1, sector_mentions[sector])))
            
            self.policy_cycle['industry_support'][sector] = (
                self.policy_cycle['industry_support'][sector] * (1 - smooth_factor) +
                normalized_impact * smooth_factor
            )
                
        # 构建影响行业列表
        impacted_sectors = []
        
        for sector, count in sector_mentions.most_common(5):
            if count >= 2:  # 至少被提及2次
                impact_direction = self.policy_cycle['industry_support'].get(sector, 0.0)
                impacted_sectors.append({
                    'sector': sector,
                    'mention_count': count,
                    'impact_direction': impact_direction,
                    'is_positive': impact_direction > 0.1,
                    'is_negative': impact_direction < -0.1
                })
                
        return impacted_sectors
    
    def _extract_top_keywords(self, topk=10):
        """提取政策新闻中的热门关键词"""
        if not self.policy_memory:
            return []
            
        # 合并所有关键词
        all_keywords = Counter()
        
        for item in self.policy_memory:
            for word, weight in item['keywords'].items():
                all_keywords[word] += weight * item['recency_weight']
                
        # 返回top关键词
        return [{'word': word, 'weight': round(weight, 2)} 
                for word, weight in all_keywords.most_common(topk)]
    
    def _generate_policy_recommendations(self):
        """基于政策分析生成建议"""
        recommendations = []
        
        # 基于货币政策的建议
        monetary_value = self.policy_cycle['monetary']
        if monetary_value > 0.5:
            recommendations.append({
                'type': 'monetary',
                'content': "货币政策宽松，有利于估值提升，关注受益于低利率环境的板块如地产、基建、金融等"
            })
        elif monetary_value < -0.5:
            recommendations.append({
                'type': 'monetary',
                'content': "货币政策收紧，关注现金流充裕、负债率低的优质企业，减持高估值高杠杆资产"
            })
        
        # 基于财政政策的建议
        fiscal_value = self.policy_cycle['fiscal']
        if fiscal_value > 0.5:
            recommendations.append({
                'type': 'fiscal',
                'content': "财政政策积极，关注基建、新基建等政府投资领域，以及减税降费受益行业"
            })
        elif fiscal_value < -0.5:
            recommendations.append({
                'type': 'fiscal',
                'content': "财政政策趋紧，政府投资可能减速，关注必需消费、医疗健康等防御性板块"
            })
        
        # 基于监管政策的建议
        regulatory_value = self.policy_cycle['regulatory']
        if regulatory_value < -0.7:
            recommendations.append({
                'type': 'regulatory',
                'content': "监管政策趋严，规避监管风险较高的行业，关注合规性强的龙头企业"
            })
        
        # 基于政策不确定性的建议
        if self.policy_uncertainty > 0.7:
            recommendations.append({
                'type': 'uncertainty',
                'content': "政策不确定性高，建议控制仓位，降低组合波动性，增加确定性较强的品种"
            })
        elif self.policy_uncertainty < 0.3:
            recommendations.append({
                'type': 'uncertainty',
                'content': "政策环境明确，可根据政策方向积极布局受益板块"
            })
        
        # 基于政策方向的建议
        policy_direction = self._calculate_policy_direction()
        if policy_direction > 0.7:
            recommendations.append({
                'type': 'direction',
                'content': "政策整体偏宽松，市场风险偏好有望提升，可关注高弹性板块"
            })
        elif policy_direction < -0.7:
            recommendations.append({
                'type': 'direction',
                'content': "政策整体偏紧，市场风险偏好可能下降，建议降低仓位，防御为主"
            })
        
        # 基于行业政策的建议
        sector_recommendations = []
        for sector, support in self.policy_cycle['industry_support'].items():
            if support > 0.6:
                sector_recommendations.append({
                    'sector': sector,
                    'is_positive': True,
                    'content': f"政策支持力度高，是中长期布局重点"
                })
            elif support < -0.6:
                sector_recommendations.append({
                    'sector': sector,
                    'is_positive': False,
                    'content': f"政策压力较大，建议减持规避"
                })
                
        # 添加行业建议（最多3个）
        for rec in sector_recommendations[:3]:
            recommendations.append({
                'type': 'sector',
                'sector': rec['sector'],
                'content': f"{rec['sector']}行业{rec['content']}"
            })
        
        return recommendations
    
    def _get_current_policy_state(self):
        """返回当前的政策状态"""
        policy_direction = self._calculate_policy_direction()
        
        impacted_sectors = []
        for sector, impact in self.policy_cycle['industry_support'].items():
            if abs(impact) > 0.3:
                impacted_sectors.append({
                    'sector': sector,
                    'mention_count': 0,
                    'impact_direction': impact,
                    'is_positive': impact > 0.1,
                    'is_negative': impact < -0.1
                })
        
        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'policy_direction': policy_direction,
            'monetary_policy': self.policy_cycle['monetary'],
            'fiscal_policy': self.policy_cycle['fiscal'],
            'regulatory_policy': self.policy_cycle['regulatory'],
            'policy_uncertainty': self.policy_uncertainty,
            'policy_shift_detected': self.policy_shift_detected,
            'policy_shift_magnitude': self.policy_shift_magnitude,
            'policy_impact_sectors': impacted_sectors,
            'recent_policy_keywords': [],
            'policy_news_count': len(self.policy_memory),
            'policy_recommendations': self._generate_policy_recommendations()
        }
    
    def inject_policy_event(self, event_type, value, sectors=None):
        """
        注入政策事件
        
        参数:
            event_type: 事件类型 ('interest_rate', 'reserve_ratio', 'fiscal', 'regulatory')
            value: 事件值 (-1到1)
            sectors: 影响的行业列表
        """
        event = {
            'type': event_type,
            'value': value
        }
        
        if sectors:
            event['sectors'] = sectors
            
        self._process_policy_events([event])
        self.logger.info(f"注入政策事件: {event_type}, 值: {value}")
        
        return True
    
    def add_policy_news(self, news_items):
        """
        添加政策新闻
        
        参数:
            news_items: 新闻列表
            
        返回:
            bool: 是否成功
        """
        if not news_items:
            return False
            
        # 确保输入格式正确
        if not isinstance(news_items, list):
            news_items = [news_items]
            
        self._process_policy_news(news_items)
        
        return True
    
    def clear_policy_memory(self):
        """清空政策记忆"""
        self.policy_memory = []
        self.logger.info("政策记忆已清空")
        
        return True
    
    def add_sector_keywords(self, sector, keywords):
        """
        为行业添加关键词
        
        参数:
            sector: 行业名称
            keywords: 关键词列表
        """
        if sector not in self.config['sector_keywords']:
            self.config['sector_keywords'][sector] = []
            
        for keyword in keywords:
            if keyword not in self.config['sector_keywords'][sector]:
                self.config['sector_keywords'][sector].append(keyword)
                jieba.add_word(keyword)
                
        self.logger.info(f"为 {sector} 行业添加了 {len(keywords)} 个关键词")
        
        return True
    
    def add_policy_keywords(self, keyword, weight):
        """
        添加政策关键词及权重
        
        参数:
            keyword: 关键词
            weight: 权重值 (-2到2)
        """
        self.config['keyword_weights'][keyword] = max(-2.0, min(2.0, weight))
        jieba.add_word(keyword)
        
        self.logger.info(f"添加政策关键词: {keyword}, 权重: {weight}")
        
        return True
    
    def analyze_policy_environment(self, data=None):
        """
        分析政策环境 (主要方法)
        
        参数:
            data: 包含政策新闻和其他信息的字典
            
        返回:
            dict: 政策分析结果
        """
        self.logger.info("开始全面分析政策环境...")
        
        # 调用基础分析方法
        result = self.analyze(data)
        
        # 增加额外分析内容
        result.update({
            'analysis_type': 'comprehensive',
            'policy_cycle_phase': self._determine_policy_cycle_phase(),
            'market_impact_prediction': self._predict_market_impact(),
            'policy_analysis_confidence': self._calculate_analysis_confidence()
        })
        
        self.logger.info(f"政策环境全面分析完成，周期阶段: {result['policy_cycle_phase']}")
        return result
    
    def _determine_policy_cycle_phase(self):
        """确定当前政策周期阶段"""
        monetary = self.policy_cycle['monetary']
        fiscal = self.policy_cycle['fiscal']
        regulatory = self.policy_cycle['regulatory']
        
        # 政策周期阶段判断逻辑
        if monetary > 0.3 and fiscal > 0.2:
            return "POLICY_EASING"  # 政策宽松期
        elif monetary < -0.3 and regulatory < -0.2:
            return "POLICY_TIGHTENING"  # 政策收紧期
        elif abs(monetary) < 0.2 and abs(fiscal) < 0.2:
            return "POLICY_NEUTRAL"  # 政策中性期
        elif monetary > 0.3 and regulatory < -0.3:
            return "MIXED_POLICY"  # 混合政策期
        else:
            return "POLICY_TRANSITION"  # 政策转换期
    
    def _predict_market_impact(self):
        """预测政策对市场的影响"""
        impact = {
            'short_term': 0.0,  # -1 到 1, 负数表示负面影响
            'medium_term': 0.0,
            'long_term': 0.0,
            'sectors': {}
        }
        
        # 短期影响主要受货币政策影响
        impact['short_term'] = self.policy_cycle['monetary'] * 0.7
        
        # 中期影响受财政政策和货币政策共同影响
        impact['medium_term'] = (self.policy_cycle['monetary'] * 0.4 + 
                              self.policy_cycle['fiscal'] * 0.6)
        
        # 长期影响受监管政策和财政政策影响更大
        impact['long_term'] = (self.policy_cycle['fiscal'] * 0.5 + 
                            self.policy_cycle['regulatory'] * 0.3 +
                            self.policy_cycle['monetary'] * 0.2)
        
        # 各行业影响
        for sector, support in self.policy_cycle.get('industry_support', {}).items():
            impact['sectors'][sector] = support
        
        return impact
    
    def _calculate_analysis_confidence(self):
        """计算分析的置信度"""
        # 置信度与政策不确定性成反比
        base_confidence = 1.0 - self.policy_uncertainty
        
        # 政策数据量会影响置信度
        news_count = len(self.policy_memory)
        data_factor = min(1.0, news_count / 20.0)  # 至少需要20条新闻达到满置信度
        
        # 政策转变会降低置信度
        shift_penalty = self.policy_shift_magnitude * 0.3 if self.policy_shift_detected else 0
        
        # 计算最终置信度
        confidence = base_confidence * data_factor * (1 - shift_penalty)
        
        # 约束到0.1-1.0范围
        return max(0.1, min(1.0, confidence))

# 如果直接运行此脚本，则执行示例
if __name__ == "__main__":
    # 创建政策分析器
    policy_analyzer = PolicyAnalyzer()
    
    # 示例政策新闻
    example_news = [
        {
            'title': '央行宣布下调存款准备金率0.5个百分点',
            'content': '中国人民银行今日宣布，下调金融机构存款准备金率0.5个百分点，释放长期资金约1万亿元。此举旨在优化金融机构资金结构，增强金融服务实体经济的能力。',
            'time': datetime.now().strftime('%Y-%m-%d'),
            'source': '央行官网'
        },
        {
            'title': '发改委批复多项重大基建项目',
            'content': '国家发展改革委近日批复了多项重大基础设施建设项目，总投资超过5000亿元，涉及高速铁路、城市轨道交通、能源等多个领域，旨在扩大有效投资，促进经济稳定增长。',
            'time': datetime.now().strftime('%Y-%m-%d'),
            'source': '发改委网站'
        }
    ]
    
    # 添加测试新闻
    policy_analyzer.add_policy_news(example_news)
    
    # 注入政策事件
    policy_analyzer.inject_policy_event('interest_rate_change', -0.5)  # 降息
    
    # 执行分析
    result = policy_analyzer.analyze()
    
    # 输出分析结果
    print("\n" + "="*60)
    print("超神量子共生系统 - 政策分析器")
    print("="*60 + "\n")
    
    print(f"政策方向: {result['policy_direction']:.2f}")
    print(f"货币政策: {result['monetary_policy']:.2f}")
    print(f"财政政策: {result['fiscal_policy']:.2f}")
    print(f"监管政策: {result['regulatory_policy']:.2f}")
    print(f"政策不确定性: {result['policy_uncertainty']:.2f}")
    
    print("\n政策影响行业:")
    for sector in result['policy_impact_sectors']:
        direction = "正面" if sector['is_positive'] else "负面" if sector['is_negative'] else "中性"
        print(f"- {sector['sector']}: {direction} (影响程度: {sector['impact_direction']:.2f})")
    
    print("\n政策关键词:")
    for keyword in result['recent_policy_keywords']:
        print(f"- {keyword['word']}: {keyword['weight']:.2f}")
    
    print("\n政策建议:")
    for rec in result['policy_recommendations']:
        print(f"- [{rec['type']}] {rec['content']}")
    
    print("\n系统已准备就绪，可进行中国政策环境分析。")
    print("="*60) 