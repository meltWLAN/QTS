#!/usr/bin/env python3
"""
市场情绪分析器 - 超神量子共生系统情绪分析模块
分析财经新闻和社交媒体数据，提取市场情绪和事件影响
"""

import os
import re
import time
import json
import random
import logging
import numpy as np
import requests
from datetime import datetime, timedelta
from collections import defaultdict
import traceback

# 尝试导入NLP相关库
try:
    import jieba
    import jieba.analyse
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('sentiment_analyzer.log')
    ]
)
logger = logging.getLogger("SentimentAnalyzer")

# 配置目录
DATA_DIR = "data"
SENTIMENT_DATA_DIR = os.path.join(DATA_DIR, "sentiment")
os.makedirs(SENTIMENT_DATA_DIR, exist_ok=True)

class MarketSentimentAnalyzer:
    """市场情绪分析器 - 捕捉市场参与者集体情绪与新闻事件影响"""
    
    def __init__(self, dimension_count=11):
        """初始化市场情绪分析器
        
        参数:
            dimension_count (int): 情绪张量维度
        """
        self.logger = logging.getLogger("SentimentAnalyzer")
        self.logger.info("初始化市场情绪分析器...")
        
        # 基本配置
        self.dimension_count = dimension_count
        self.initialized = False
        self.active = False
        self.last_update = datetime.now()
        
        # 情绪分析状态
        self.sentiment_state = {
            "market_sentiment": 0.5,  # 0-极度悲观, 1-极度乐观
            "fear_greed_index": 50,   # 0-100 恐惧与贪婪指数
            "news_impact": 0.0,       # 新闻影响强度
            "social_resonance": 0.0,  # 社交媒体共振强度
            "event_significance": 0.0, # 事件重要性
            "volatility_expectation": 0.5, # 波动性预期
            "last_analysis": None     # 上次分析时间
        }
        
        # 高维数据结构
        self.sentiment_matrix = np.zeros((dimension_count, dimension_count))
        self.news_impact_tensor = np.zeros((dimension_count, 3))  # 影响强度、持续时间、传播速度
        self.social_resonance_field = np.zeros(dimension_count)
        
        # 情绪词典
        self.sentiment_dict = {
            "positive": self._load_word_list("positive_words.txt"),
            "negative": self._load_word_list("negative_words.txt"),
            "financial_positive": self._load_word_list("financial_positive_words.txt"),
            "financial_negative": self._load_word_list("financial_negative_words.txt"),
            "intensifiers": self._load_word_list("intensifier_words.txt"),
            "diminishers": self._load_word_list("diminisher_words.txt")
        }
        
        # 情绪历史
        self.sentiment_history = []
        
        # 新闻缓存
        self.news_cache = []
        self.news_cache_expiry = datetime.now()
        
        # 社交媒体数据缓存
        self.social_data_cache = []
        self.social_cache_expiry = datetime.now()
        
        # 关键事件库
        self.key_events = []
        
        # 情绪异常检测阈值
        self.anomaly_thresholds = {
            "sentiment_change": 0.2,  # 情绪突变阈值
            "fear_greed_extremes": 20, # 恐贪指数极端区间
            "news_impact_spike": 0.7,  # 新闻影响峰值阈值
            "social_resonance_spike": 0.8 # 社交共振峰值阈值
        }
        
        self.logger.info(f"市场情绪分析器初始化完成，维度: {dimension_count}")
    
    def _load_word_list(self, filename):
        """从文件加载词语列表，如果文件不存在则返回默认列表"""
        try:
            file_path = os.path.join(SENTIMENT_DATA_DIR, filename)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return [line.strip() for line in f if line.strip()]
            else:
                # 如果文件不存在，返回一个小的默认列表并创建文件
                default_lists = {
                    "positive_words.txt": ["上涨", "增长", "利好", "突破", "向好", "反弹", "牛市", "获利"],
                    "negative_words.txt": ["下跌", "下滑", "利空", "跌破", "走弱", "回调", "熊市", "亏损"],
                    "financial_positive_words.txt": ["盈利", "扩张", "收购", "增持", "分红", "扭亏", "创新高"],
                    "financial_negative_words.txt": ["亏损", "缩减", "出售", "减持", "停牌", "创新低", "违规"],
                    "intensifier_words.txt": ["大幅", "显著", "剧烈", "极度", "强劲", "猛烈", "迅速"],
                    "diminisher_words.txt": ["略微", "轻微", "稍稍", "小幅", "缓慢", "温和", "平稳"]
                }
                
                default_list = default_lists.get(filename, [])
                
                # 创建目录并保存默认列表
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(default_list))
                
                return default_list
        except Exception as e:
            self.logger.error(f"加载词语列表失败 {filename}: {str(e)}")
            return []
    
    def initialize(self, field_strength=0.75):
        """初始化情绪分析器
        
        参数:
            field_strength (float): 初始场强度
            
        返回:
            bool: 初始化是否成功
        """
        self.logger.info(f"初始化情绪分析器: 场强={field_strength}")
        
        try:
            # 初始化情绪矩阵
            self.sentiment_matrix = np.random.randn(self.dimension_count, self.dimension_count) * field_strength * 0.1
            
            # 初始化新闻影响张量
            self.news_impact_tensor = np.random.randn(self.dimension_count, 3) * field_strength * 0.2
            
            # 初始化社交共振场
            self.social_resonance_field = np.random.randn(self.dimension_count) * field_strength * 0.15
            
            # 初始化情绪状态
            self.sentiment_state["market_sentiment"] = 0.45 + random.random() * 0.1
            self.sentiment_state["fear_greed_index"] = 40 + random.randint(0, 20)
            self.sentiment_state["news_impact"] = random.random() * 0.3
            self.sentiment_state["social_resonance"] = random.random() * 0.3
            self.sentiment_state["event_significance"] = random.random() * 0.2
            
            # 检查NLP库可用性
            if not JIEBA_AVAILABLE:
                self.logger.warning("jieba库未安装，中文分词功能将受限")
            
            if not TEXTBLOB_AVAILABLE:
                self.logger.warning("textblob库未安装，英文情感分析功能将受限")
            
            # 预加载一些历史情绪数据
            self._preload_sentiment_history()
            
            self.initialized = True
            self.active = True
            self.logger.info("情绪分析器初始化完成")
            
            return True
            
        except Exception as e:
            self.logger.error(f"情绪分析器初始化失败: {str(e)}")
            traceback.print_exc()
            return False
            
    def _preload_sentiment_history(self):
        """预加载一些历史情绪数据，用于初始化"""
        self.logger.info("预加载历史情绪数据...")
        
        # 生成30天的模拟历史情绪数据
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        current_date = start_date
        
        # 设置基础情绪值和趋势
        sentiment = 0.5 + random.uniform(-0.1, 0.1)
        fear_greed = 50 + random.randint(-10, 10)
        trend = random.choice([-1, 1]) * 0.02
        
        while current_date < end_date:
            # 添加随机波动
            noise = random.uniform(-0.03, 0.03)
            sentiment += trend + noise
            sentiment = max(0.1, min(0.9, sentiment))  # 限制在合理范围内
            
            fear_greed += trend * 50 + random.randint(-3, 3)
            fear_greed = max(10, min(90, fear_greed))  # 限制在合理范围内
            
            # 偶尔改变趋势
            if random.random() < 0.1:
                trend = random.choice([-1, 1]) * 0.02
            
            # 记录历史数据
            self.sentiment_history.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "market_sentiment": sentiment,
                "fear_greed_index": fear_greed,
                "news_impact": random.uniform(0, 0.5),
                "social_resonance": random.uniform(0, 0.4)
            })
            
            current_date += timedelta(days=1)
        
        self.logger.info(f"预加载了{len(self.sentiment_history)}天的历史情绪数据")

    def fetch_financial_news(self, keywords=None, max_items=50):
        """获取财经新闻数据
        
        参数:
            keywords (list): 关键词列表
            max_items (int): 最大新闻条数
            
        返回:
            list: 新闻列表
        """
        # 检查缓存是否有效
        if self.news_cache and self.news_cache_expiry > datetime.now():
            self.logger.info(f"使用缓存的新闻数据 ({len(self.news_cache)}条)")
            return self.news_cache
            
        self.logger.info("获取财经新闻数据...")
        
        # 在实际应用中，这里应该调用新闻API
        # 例如：新浪财经、东方财富、华尔街见闻等
        # 这里我们生成模拟数据
        
        # 定义一些常见的财经新闻标题模板
        title_templates = [
            "{公司}发布{季度}财报，{业绩}{方向}超出市场预期",
            "{公司}{方向}收购{目标公司}，交易价值{金额}亿",
            "{机构}发布研报{看多|看空}{行业}，预计将{上涨|下跌}{幅度}%",
            "{国家}宣布{政策}政策，{行业}板块{上涨|下跌}",
            "央行宣布{加息|降息}{幅度}个基点，市场{方向}反应",
            "{指数}创{时间段}{新高|新低}，{原因}成为主要推动力",
            "{公司}发布{产品}，市场反应{积极|平淡|消极}",
            "{行业}需求{增长|下滑}，{公司}股价{上涨|下跌}{幅度}%",
            "{机构}调整对{公司}的评级至{评级}，目标价{上调|下调}至{价格}",
            "{国家}经济数据{好于|差于}预期，{指数}{上涨|下跌}{幅度}%"
        ]
        
        # 定义替换变量
        companies = ["阿里巴巴", "腾讯控股", "茅台", "平安保险", "工商银行", "比亚迪", "华为", "小米", "京东", "美团"]
        target_companies = ["某科技初创公司", "某互联网平台", "某制造业企业", "某金融机构", "某海外资产"]
        quarters = ["Q1", "Q2", "Q3", "Q4", "年度", "中期"]
        performances = ["营收", "利润", "市场份额", "用户增长", "海外业务"]
        directions_pos = ["上升", "增长", "提升", "扩大", "强劲增长", "显著提高"]
        directions_neg = ["下降", "收缩", "减少", "下滑", "大幅下跌", "轻微下调"]
        institutions = ["高盛", "摩根士丹利", "中金公司", "中信证券", "瑞银", "花旗", "野村证券"]
        industries = ["科技", "金融", "医疗", "消费品", "新能源", "制造业", "互联网", "房地产"]
        countries = ["中国", "美国", "欧盟", "日本", "印度", "俄罗斯", "英国"]
        policies = ["财政刺激", "减税", "产业支持", "监管", "改革", "开放", "限制"]
        indices = ["上证指数", "深证成指", "创业板指", "恒生指数", "道琼斯", "纳斯达克", "标普500"]
        time_periods = ["年内", "3年来", "历史", "近5年", "本季度", "本月"]
        reasons = ["政策利好", "业绩增长", "外资流入", "市场信心", "技术突破", "行业整合"]
        products = ["新产品", "战略升级", "服务创新", "技术突破", "商业模式"]
        reactions = ["积极", "热烈", "平淡", "谨慎", "负面", "复杂"]
        ratings = ["买入", "增持", "中性", "减持", "卖出", "跑赢大市", "跟随大市", "跑输大市"]
        
        # 生成模拟新闻
        news_list = []
        for _ in range(max_items):
            # 随机选择一个模板
            template = random.choice(title_templates)
            
            # 替换模板中的变量
            title = template
            
            if "{公司}" in title:
                title = title.replace("{公司}", random.choice(companies))
            
            if "{目标公司}" in title:
                title = title.replace("{目标公司}", random.choice(target_companies))
                
            if "{季度}" in title:
                title = title.replace("{季度}", random.choice(quarters))
                
            if "{业绩}" in title:
                title = title.replace("{业绩}", random.choice(performances))
                
            if "{方向}" in title:
                # 有60%的概率是正面的
                if random.random() < 0.6:
                    title = title.replace("{方向}", random.choice(directions_pos))
                else:
                    title = title.replace("{方向}", random.choice(directions_neg))
                    
            if "{机构}" in title:
                title = title.replace("{机构}", random.choice(institutions))
                
            if "{行业}" in title:
                title = title.replace("{行业}", random.choice(industries))
                
            if "{国家}" in title:
                title = title.replace("{国家}", random.choice(countries))
                
            if "{政策}" in title:
                title = title.replace("{政策}", random.choice(policies))
                
            if "{指数}" in title:
                title = title.replace("{指数}", random.choice(indices))
                
            if "{时间段}" in title:
                title = title.replace("{时间段}", random.choice(time_periods))
                
            if "{原因}" in title:
                title = title.replace("{原因}", random.choice(reasons))
                
            if "{产品}" in title:
                title = title.replace("{产品}", random.choice(products))
                
            if "{积极|平淡|消极}" in title:
                title = title.replace("{积极|平淡|消极}", random.choice(reactions))
                
            if "{评级}" in title:
                title = title.replace("{评级}", random.choice(ratings))
                
            if "{金额}" in title:
                title = title.replace("{金额}", str(random.randint(1, 500)))
                
            if "{幅度}" in title:
                title = title.replace("{幅度}", str(random.randint(1, 20)))
                
            if "{价格}" in title:
                title = title.replace("{价格}", str(random.randint(10, 1000)))
                
            # 处理可选模式 {上涨|下跌}
            for pattern in ["{上涨|下跌}", "{新高|新低}", "{好于|差于}", "{上调|下调}", "{加息|降息}"]:
                if pattern in title:
                    options = pattern[1:-1].split('|')
                    replacement = random.choice(options)
                    title = title.replace(pattern, replacement)
                    
            # 处理看多看空
            if "{看多|看空}" in title:
                if random.random() < 0.6:  # 60%概率看多
                    title = title.replace("{看多|看空}", "看多")
                else:
                    title = title.replace("{看多|看空}", "看空")
                    
            # 创建新闻对象
            news_date = datetime.now() - timedelta(hours=random.randint(0, 72))
            
            # 计算情感分数（后面会用真实算法替代）
            sentiment_score = random.uniform(-1, 1)
            
            # 为关键词筛选
            contains_keyword = True
            if keywords:
                contains_keyword = any(keyword in title for keyword in keywords)
                
            if contains_keyword:
                news_list.append({
                    "title": title,
                    "source": random.choice(["新浪财经", "东方财富", "华尔街见闻", "财联社", "Bloomberg", "路透社", "第一财经"]),
                    "url": f"https://example.com/news/{random.randint(10000, 99999)}",
                    "date": news_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "summary": f"{title}。" + "相关分析表明，这一事件可能对市场产生重要影响。" * random.randint(1, 3),
                    "sentiment_score": sentiment_score,
                    "relevance": random.uniform(0.5, 1.0)
                })
        
        # 按日期排序，最新的在前
        news_list.sort(key=lambda x: x["date"], reverse=True)
        
        # 更新缓存
        self.news_cache = news_list
        self.news_cache_expiry = datetime.now() + timedelta(hours=1)
        
        self.logger.info(f"获取了{len(news_list)}条财经新闻")
        return news_list 

    def fetch_social_media_data(self, keywords=None, max_items=100):
        """获取社交媒体数据
        
        参数:
            keywords (list): 关键词列表
            max_items (int): 最大条目数
            
        返回:
            list: 社交媒体数据列表
        """
        # 检查缓存是否有效
        if self.social_data_cache and self.social_cache_expiry > datetime.now():
            self.logger.info(f"使用缓存的社交媒体数据 ({len(self.social_data_cache)}条)")
            return self.social_data_cache
            
        self.logger.info("获取社交媒体数据...")
        
        # 在实际应用中，这里应该调用社交媒体API
        # 例如：微博、Twitter、Reddit等
        # 这里我们生成模拟数据
        
        # 生成一些模拟的社交媒体内容模板
        content_templates = [
            "今天{指数}又{涨跌了}，{心情}！",
            "{看好|看空}{股票}，理由是{理由}。",
            "刚刚{买入|卖出}了一些{股票}，希望能{赚|亏}！",
            "{机构}发布的{股票}研报{看多|看空}，目标价{价格}，大家怎么看？",
            "最近{行业}行情{不错|一般|很差}，建议{操作}。",
            "今日大盘{涨跌}，个人认为是因为{原因}，后市将{预期}。",
            "财报季要来了，{股票}业绩应该会{预期}，{买入|持有|卖出}！",
            "听说{政策}要出台了，对{行业}是{利好|利空}啊！",
            "美股昨晚{涨跌}，A股今天开盘应该会{预期}。",
            "{支持|反对}现在入场，市场情绪{过热|正常|低迷}。"
        ]
        
        # 定义替换变量
        indices = ["大盘", "上证", "创业板", "科创板", "50ETF", "A股", "港股", "美股"]
        up_actions = ["涨停", "大涨", "暴涨", "反弹", "走强", "突破", "创新高"]
        down_actions = ["跌停", "大跌", "暴跌", "回调", "走弱", "跌破", "创新低"]
        positive_moods = ["开心", "高兴", "兴奋", "满意", "期待", "激动"]
        negative_moods = ["难过", "失望", "郁闷", "焦虑", "担忧", "后悔"]
        stocks = ["茅台", "腾讯", "阿里", "平安", "工商银行", "兴业银行", "比亚迪", "宁德时代", "中国石油", "中国移动"]
        reasons = ["业绩超预期", "估值合理", "技术面看好", "有政策利好", "行业前景好", "机构持续买入", 
                   "业绩不达预期", "估值过高", "技术面看淡", "有政策利空", "行业下滑", "机构持续卖出"]
        institutions = ["高盛", "摩根", "中金", "中信证券", "国泰君安", "华泰证券"]
        industries = ["新能源", "医药", "消费", "科技", "金融", "地产", "白酒", "军工"]
        operations = ["抄底", "满仓", "加仓", "减仓", "观望", "空仓"]
        reasons_market = ["外盘影响", "政策消息", "资金流向", "技术面因素", "情绪面因素", "基本面表现"]
        expectations = ["继续上涨", "震荡整理", "高位盘整", "触底反弹", "继续下跌", "筑底企稳"]
        policies = ["货币政策", "财政政策", "产业政策", "监管政策", "改革政策"]
        
        # 生成模拟社交媒体数据
        social_data = []
        for _ in range(max_items):
            # 随机选择一个模板
            template = random.choice(content_templates)
            
            # 替换模板中的变量
            content = template
            
            if "{指数}" in content:
                content = content.replace("{指数}", random.choice(indices))
                
            if "{涨跌了}" in content:
                if random.random() < 0.6:  # 60%概率是上涨
                    action = random.choice(up_actions)
                    sentiment_bias = 0.6  # 积极情绪偏向
                else:
                    action = random.choice(down_actions)
                    sentiment_bias = -0.6  # 消极情绪偏向
                content = content.replace("{涨跌了}", action)
            else:
                sentiment_bias = 0.0
                
            if "{心情}" in content:
                if sentiment_bias > 0:
                    mood = random.choice(positive_moods)
                elif sentiment_bias < 0:
                    mood = random.choice(negative_moods)
                else:
                    mood = random.choice(positive_moods + negative_moods)
                content = content.replace("{心情}", mood)
                
            if "{股票}" in content:
                content = content.replace("{股票}", random.choice(stocks))
                
            if "{理由}" in content:
                content = content.replace("{理由}", random.choice(reasons))
                
            if "{机构}" in content:
                content = content.replace("{机构}", random.choice(institutions))
                
            if "{价格}" in content:
                content = content.replace("{价格}", str(random.randint(10, 1000)))
                
            if "{行业}" in content:
                content = content.replace("{行业}", random.choice(industries))
                
            if "{操作}" in content:
                content = content.replace("{操作}", random.choice(operations))
                
            if "{涨跌}" in content:
                if random.random() < 0.6:  # 60%概率是上涨
                    content = content.replace("{涨跌}", random.choice(up_actions))
                else:
                    content = content.replace("{涨跌}", random.choice(down_actions))
                    
            if "{原因}" in content:
                content = content.replace("{原因}", random.choice(reasons_market))
                
            if "{预期}" in content:
                content = content.replace("{预期}", random.choice(expectations))
                
            if "{政策}" in content:
                content = content.replace("{政策}", random.choice(policies))
                
            # 处理可选模式
            for pattern in ["{看好|看空}", "{买入|卖出}", "{赚|亏}", "{不错|一般|很差}", 
                            "{利好|利空}", "{买入|持有|卖出}", "{支持|反对}"]:
                if pattern in content:
                    options = pattern[1:-1].split('|')
                    replacement = random.choice(options)
                    content = content.replace(pattern, replacement)
                    
                    # 调整情感偏向
                    if replacement in ["看好", "买入", "赚", "不错", "利好", "支持"]:
                        sentiment_bias += 0.3
                    elif replacement in ["看空", "卖出", "亏", "很差", "利空", "反对"]:
                        sentiment_bias -= 0.3
            
            # 计算情感分数（-1到1之间）
            base_sentiment = random.uniform(-0.3, 0.3)  # 基础随机性
            sentiment_score = max(-1.0, min(1.0, base_sentiment + sentiment_bias))  # 限制在-1到1之间
            
            # 为关键词筛选
            contains_keyword = True
            if keywords:
                contains_keyword = any(keyword in content for keyword in keywords)
                
            if contains_keyword:
                # 创建发布时间（过去48小时内随机时间）
                post_time = datetime.now() - timedelta(hours=random.randint(0, 48), 
                                                       minutes=random.randint(0, 59))
                
                # 添加到数据列表
                social_data.append({
                    "content": content,
                    "platform": random.choice(["微博", "雪球", "Twitter", "Reddit", "股吧", "投资论坛"]),
                    "user_type": random.choice(["散户", "专业投资者", "分析师", "机构", "媒体", "意见领袖"]),
                    "followers": random.randint(10, 1000000),
                    "engagement": random.randint(0, 10000),
                    "time": post_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "sentiment_score": sentiment_score,
                    "relevance": random.uniform(0.5, 1.0)
                })
        
        # 按时间排序，最新的在前
        social_data.sort(key=lambda x: x["time"], reverse=True)
        
        # 更新缓存
        self.social_data_cache = social_data
        self.social_cache_expiry = datetime.now() + timedelta(minutes=30)
        
        self.logger.info(f"获取了{len(social_data)}条社交媒体数据")
        return social_data
    
    def analyze_text_sentiment(self, text, lang="zh"):
        """分析文本情感
        
        参数:
            text (str): 文本内容
            lang (str): 语言，zh-中文，en-英文
            
        返回:
            float: 情感分数(-1到1，负面到正面)
        """
        if not text:
            return 0.0
            
        try:
            if lang == "en" and TEXTBLOB_AVAILABLE:
                # 使用TextBlob进行英文情感分析
                blob = TextBlob(text)
                # TextBlob的情感极性范围是-1到1
                return blob.sentiment.polarity
                
            elif lang == "zh":
                # 中文情感分析
                if JIEBA_AVAILABLE:
                    # 使用结巴分词进行中文分词
                    words = jieba.lcut(text)
                else:
                    # 简单的按字符分割
                    words = list(text)
                
                # 基于词典的简单情感分析
                positive_score = 0
                negative_score = 0
                intensity = 1.0  # 强度修饰词的影响
                
                # 合并金融领域和通用情感词典
                positive_words = set(self.sentiment_dict["positive"] + self.sentiment_dict["financial_positive"])
                negative_words = set(self.sentiment_dict["negative"] + self.sentiment_dict["financial_negative"])
                intensifiers = set(self.sentiment_dict["intensifiers"])
                diminishers = set(self.sentiment_dict["diminishers"])
                
                for word in words:
                    if word in intensifiers:
                        intensity = 1.5  # 强度词会增强下一个情感词的效果
                        continue
                        
                    if word in diminishers:
                        intensity = 0.5  # 减弱词会减弱下一个情感词的效果
                        continue
                        
                    if word in positive_words:
                        positive_score += 1 * intensity
                        intensity = 1.0  # 重置强度
                        
                    if word in negative_words:
                        negative_score += 1 * intensity
                        intensity = 1.0  # 重置强度
                
                # 计算总分数并归一化到-1到1
                total_words = len(words) if len(words) > 0 else 1
                score = (positive_score - negative_score) / total_words * 2
                return max(-1.0, min(1.0, score))  # 限制在-1到1之间
                
            else:
                self.logger.warning(f"不支持的语言: {lang}")
                return 0.0
                
        except Exception as e:
            self.logger.error(f"情感分析失败: {str(e)}")
            return 0.0
    
    def analyze_news_sentiment(self, news_data):
        """分析新闻情绪影响
        
        参数:
            news_data (list): 新闻数据列表
            
        返回:
            dict: 情绪分析结果
        """
        self.logger.info(f"分析新闻情绪数据 ({len(news_data)}条)")
        
        if not news_data:
            return {
                "overall_sentiment": 0.0,
                "impact_score": 0.0,
                "sentiment_distribution": {"positive": 0, "neutral": 0, "negative": 0},
                "key_topics": [],
                "recent_trend": "稳定"
            }
        
        try:
            # 情感分数和影响力总和
            total_sentiment = 0.0
            total_impact = 0.0
            sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
            topic_sentiment = defaultdict(list)
            
            # 对新闻进行情感分析
            for news in news_data:
                # 如果已有情感分数，直接使用
                if "sentiment_score" in news:
                    sentiment = news["sentiment_score"]
                else:
                    # 否则分析标题和摘要的情感
                    title_sentiment = self.analyze_text_sentiment(news["title"])
                    summary_sentiment = self.analyze_text_sentiment(news.get("summary", ""))
                    # 标题的情感权重更大
                    sentiment = title_sentiment * 0.7 + summary_sentiment * 0.3
                    news["sentiment_score"] = sentiment
                
                # 计算新闻的影响力分数
                # 考虑因素：新闻来源权威性、时效性、相关性
                source_weight = 1.0  # 可以为不同的来源设置不同权重
                
                # 时效性权重，越新的新闻权重越大
                time_diff = datetime.now() - datetime.strptime(news["date"], "%Y-%m-%d %H:%M:%S")
                time_weight = max(0.1, 1.0 - (time_diff.total_seconds() / (3 * 24 * 3600)))  # 3天内线性衰减
                
                # 相关性权重
                relevance = news.get("relevance", 0.8)
                
                # 计算总影响力
                impact = source_weight * time_weight * relevance
                
                # 累加情感和影响力
                total_sentiment += sentiment * impact
                total_impact += impact
                
                # 统计情感分布
                if sentiment > 0.2:
                    sentiment_counts["positive"] += 1
                elif sentiment < -0.2:
                    sentiment_counts["negative"] += 1
                else:
                    sentiment_counts["neutral"] += 1
                
                # 提取关键词和主题
                if JIEBA_AVAILABLE:
                    try:
                        # 使用jieba提取关键词
                        keywords = jieba.analyse.extract_tags(news["title"] + news.get("summary", ""), topK=3)
                        for keyword in keywords:
                            if len(keyword) > 1:  # 过滤单字关键词
                                topic_sentiment[keyword].append(sentiment)
                    except:
                        pass
            
            # 计算加权平均情感分数
            overall_sentiment = total_sentiment / total_impact if total_impact > 0 else 0.0
            
            # 计算情感分布百分比
            total_news = len(news_data)
            sentiment_distribution = {
                "positive": sentiment_counts["positive"] / total_news,
                "neutral": sentiment_counts["neutral"] / total_news,
                "negative": sentiment_counts["negative"] / total_news
            }
            
            # 影响力分数归一化
            impact_score = min(1.0, total_impact / (len(news_data) * 0.5))
            
            # 提取热门话题及其情感
            top_topics = []
            for topic, sentiments in topic_sentiment.items():
                if len(sentiments) >= 2:  # 至少出现在两篇新闻中
                    avg_sentiment = sum(sentiments) / len(sentiments)
                    top_topics.append({
                        "topic": topic,
                        "count": len(sentiments),
                        "sentiment": avg_sentiment
                    })
            
            # 按出现次数排序
            top_topics.sort(key=lambda x: x["count"], reverse=True)
            top_topics = top_topics[:10]  # 取前10个热门话题
            
            # 分析近期趋势
            # 将新闻按时间分组，分析情感变化趋势
            recent_news = sorted(news_data, key=lambda x: x["date"])
            if len(recent_news) >= 6:
                # 分为两半比较
                mid_point = len(recent_news) // 2
                first_half = recent_news[:mid_point]
                second_half = recent_news[mid_point:]
                
                first_sentiment = sum(n["sentiment_score"] for n in first_half) / len(first_half)
                second_sentiment = sum(n["sentiment_score"] for n in second_half) / len(second_half)
                
                sentiment_change = second_sentiment - first_sentiment
                if sentiment_change > 0.15:
                    recent_trend = "转为乐观"
                elif sentiment_change < -0.15:
                    recent_trend = "转为悲观"
                elif sentiment_change > 0.05:
                    recent_trend = "略微乐观"
                elif sentiment_change < -0.05:
                    recent_trend = "略微悲观"
                else:
                    recent_trend = "稳定"
            else:
                recent_trend = "数据不足"
            
            # 构造结果
            result = {
                "overall_sentiment": float(overall_sentiment),
                "impact_score": float(impact_score),
                "sentiment_distribution": sentiment_distribution,
                "key_topics": top_topics,
                "recent_trend": recent_trend,
                "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"分析新闻情绪失败: {str(e)}")
            traceback.print_exc()
            return {"overall_sentiment": 0.0, "impact_score": 0.0, "error": str(e)}
    
    def analyze_social_sentiment(self, social_data):
        """分析社交媒体情绪影响
        
        参数:
            social_data (list): 社交媒体数据列表
            
        返回:
            dict: 情绪分析结果
        """
        self.logger.info(f"分析社交媒体情绪数据 ({len(social_data)}条)")
        
        if not social_data:
            return {
                "overall_sentiment": 0.0,
                "resonance_score": 0.0,
                "sentiment_distribution": {"positive": 0, "neutral": 0, "negative": 0},
                "influencer_sentiment": 0.0,
                "crowd_sentiment": 0.0,
                "key_topics": [],
                "recent_trend": "稳定"
            }
        
        try:
            # 情感分数和影响力总和
            total_sentiment = 0.0
            total_impact = 0.0
            sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
            
            # 意见领袖和普通用户的情感
            influencer_sentiments = []
            crowd_sentiments = []
            
            # 话题情感
            topic_sentiment = defaultdict(list)
            
            # 对社交媒体内容进行情感分析
            for post in social_data:
                # 如果已有情感分数，直接使用
                if "sentiment_score" in post:
                    sentiment = post["sentiment_score"]
                else:
                    # 否则分析内容的情感
                    sentiment = self.analyze_text_sentiment(post["content"])
                    post["sentiment_score"] = sentiment
                
                # 计算帖子的影响力分数
                # 考虑因素：用户影响力、互动度、时效性
                follower_count = post.get("followers", 100)
                follower_weight = min(1.0, 0.3 + (0.7 * follower_count / 10000))  # 粉丝数影响，但有上限
                
                engagement = post.get("engagement", 0)
                engagement_weight = min(1.0, 0.1 + (0.9 * engagement / 1000))  # 互动数影响，但有上限
                
                # 时效性权重，越新的帖子权重越大
                time_diff = datetime.now() - datetime.strptime(post["time"], "%Y-%m-%d %H:%M:%S")
                time_weight = max(0.1, 1.0 - (time_diff.total_seconds() / (2 * 24 * 3600)))  # 2天内线性衰减
                
                # 计算总影响力
                impact = follower_weight * 0.5 + engagement_weight * 0.3 + time_weight * 0.2
                
                # 累加情感和影响力
                total_sentiment += sentiment * impact
                total_impact += impact
                
                # 统计情感分布
                if sentiment > 0.2:
                    sentiment_counts["positive"] += 1
                elif sentiment < -0.2:
                    sentiment_counts["negative"] += 1
                else:
                    sentiment_counts["neutral"] += 1
                
                # 分离意见领袖和普通用户的情感
                user_type = post.get("user_type", "散户")
                if user_type in ["专业投资者", "分析师", "机构", "意见领袖"] or follower_count > 10000:
                    influencer_sentiments.append(sentiment)
                else:
                    crowd_sentiments.append(sentiment)
                
                # 提取关键词和主题
                if JIEBA_AVAILABLE:
                    try:
                        # 使用jieba提取关键词
                        keywords = jieba.analyse.extract_tags(post["content"], topK=3)
                        for keyword in keywords:
                            if len(keyword) > 1:  # 过滤单字关键词
                                topic_sentiment[keyword].append(sentiment)
                    except:
                        pass
            
            # 计算加权平均情感分数
            overall_sentiment = total_sentiment / total_impact if total_impact > 0 else 0.0
            
            # 计算情感分布百分比
            total_posts = len(social_data)
            sentiment_distribution = {
                "positive": sentiment_counts["positive"] / total_posts,
                "neutral": sentiment_counts["neutral"] / total_posts,
                "negative": sentiment_counts["negative"] / total_posts
            }
            
            # 计算意见领袖和普通用户的平均情感
            influencer_sentiment = sum(influencer_sentiments) / len(influencer_sentiments) if influencer_sentiments else 0.0
            crowd_sentiment = sum(crowd_sentiments) / len(crowd_sentiments) if crowd_sentiments else 0.0
            
            # 计算共振分数（意见领袖和普通用户情感的一致性）
            if influencer_sentiments and crowd_sentiments:
                # 情感方向一致时共振强
                sentiment_alignment = 1.0 - abs(influencer_sentiment - crowd_sentiment)
                # 情感强度的加权平均
                sentiment_intensity = (abs(influencer_sentiment) * 0.6 + abs(crowd_sentiment) * 0.4)
                resonance_score = sentiment_alignment * sentiment_intensity
            else:
                resonance_score = 0.5  # 默认中等共振
            
            # 提取热门话题及其情感
            top_topics = []
            for topic, sentiments in topic_sentiment.items():
                if len(sentiments) >= 3:  # 至少出现在三个帖子中
                    avg_sentiment = sum(sentiments) / len(sentiments)
                    top_topics.append({
                        "topic": topic,
                        "count": len(sentiments),
                        "sentiment": avg_sentiment
                    })
            
            # 按出现次数排序
            top_topics.sort(key=lambda x: x["count"], reverse=True)
            top_topics = top_topics[:10]  # 取前10个热门话题
            
            # 分析近期趋势
            # 将帖子按时间分组，分析情感变化趋势
            recent_posts = sorted(social_data, key=lambda x: x["time"])
            if len(recent_posts) >= 10:
                # 分为两半比较
                mid_point = len(recent_posts) // 2
                first_half = recent_posts[:mid_point]
                second_half = recent_posts[mid_point:]
                
                first_sentiment = sum(p["sentiment_score"] for p in first_half) / len(first_half)
                second_sentiment = sum(p["sentiment_score"] for p in second_half) / len(second_half)
                
                sentiment_change = second_sentiment - first_sentiment
                if sentiment_change > 0.2:
                    recent_trend = "明显乐观"
                elif sentiment_change < -0.2:
                    recent_trend = "明显悲观"
                elif sentiment_change > 0.05:
                    recent_trend = "略微乐观"
                elif sentiment_change < -0.05:
                    recent_trend = "略微悲观"
                else:
                    recent_trend = "情绪稳定"
            else:
                recent_trend = "数据不足"
            
            # 构造结果
            result = {
                "overall_sentiment": float(overall_sentiment),
                "resonance_score": float(resonance_score),
                "sentiment_distribution": sentiment_distribution,
                "influencer_sentiment": float(influencer_sentiment),
                "crowd_sentiment": float(crowd_sentiment),
                "key_topics": top_topics,
                "recent_trend": recent_trend,
                "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"分析社交媒体情绪失败: {str(e)}")
            traceback.print_exc()
            return {"overall_sentiment": 0.0, "resonance_score": 0.0, "error": str(e)}
    
    def detect_sentiment_anomalies(self, news_sentiment, social_sentiment, market_data=None):
        """检测情绪异常模式
        
        参数:
            news_sentiment (dict): 新闻情绪分析结果
            social_sentiment (dict): 社交媒体情绪分析结果
            market_data (dict): 市场数据，可选
            
        返回:
            list: 异常情况列表
        """
        self.logger.info("检测情绪异常模式...")
        
        anomalies = []
        
        try:
            # 提取关键指标
            news_sentiment_score = news_sentiment.get("overall_sentiment", 0)
            news_impact = news_sentiment.get("impact_score", 0)
            news_trend = news_sentiment.get("recent_trend", "稳定")
            
            social_sentiment_score = social_sentiment.get("overall_sentiment", 0)
            social_resonance = social_sentiment.get("resonance_score", 0)
            social_trend = social_sentiment.get("recent_trend", "稳定")
            
            influencer_sentiment = social_sentiment.get("influencer_sentiment", 0)
            crowd_sentiment = social_sentiment.get("crowd_sentiment", 0)
            
            # 1. 检测情绪极端值
            if abs(news_sentiment_score) > 0.7:
                direction = "乐观" if news_sentiment_score > 0 else "悲观"
                anomalies.append({
                    "type": "extreme_news_sentiment",
                    "description": f"新闻情绪极度{direction}",
                    "severity": "高" if abs(news_sentiment_score) > 0.8 else "中",
                    "value": news_sentiment_score
                })
                
            if abs(social_sentiment_score) > 0.65:
                direction = "乐观" if social_sentiment_score > 0 else "悲观"
                anomalies.append({
                    "type": "extreme_social_sentiment",
                    "description": f"社交媒体情绪极度{direction}",
                    "severity": "高" if abs(social_sentiment_score) > 0.75 else "中",
                    "value": social_sentiment_score
                })
            
            # 2. 检测新闻与社交媒体情绪差异
            sentiment_gap = abs(news_sentiment_score - social_sentiment_score)
            if sentiment_gap > 0.4:
                news_direction = "乐观" if news_sentiment_score > 0 else "悲观"
                social_direction = "乐观" if social_sentiment_score > 0 else "悲观"
                anomalies.append({
                    "type": "sentiment_divergence",
                    "description": f"新闻情绪偏{news_direction}而社交媒体偏{social_direction}，出现明显分歧",
                    "severity": "高" if sentiment_gap > 0.6 else "中",
                    "value": sentiment_gap
                })
            
            # 3. 检测意见领袖与普通投资者情绪差异
            influencer_gap = abs(influencer_sentiment - crowd_sentiment)
            if influencer_gap > 0.5:
                leader_direction = "乐观" if influencer_sentiment > 0 else "悲观"
                crowd_direction = "乐观" if crowd_sentiment > 0 else "悲观"
                anomalies.append({
                    "type": "sentiment_leadership_gap",
                    "description": f"意见领袖情绪偏{leader_direction}而普通投资者偏{crowd_direction}，市场分歧明显",
                    "severity": "高" if influencer_gap > 0.7 else "中",
                    "value": influencer_gap
                })
            
            # 4. 检测情绪共振异常
            if social_resonance > 0.8:
                direction = "乐观" if social_sentiment_score > 0 else "悲观"
                anomalies.append({
                    "type": "high_resonance",
                    "description": f"社交媒体情绪高度共振，一致{direction}",
                    "severity": "高" if social_resonance > 0.9 else "中",
                    "value": social_resonance
                })
            
            # 5. 检测情绪与新闻影响力的不匹配
            if abs(news_sentiment_score) < 0.2 and news_impact > 0.7:
                anomalies.append({
                    "type": "high_impact_neutral_sentiment",
                    "description": "高影响力新闻但市场情绪反应平淡",
                    "severity": "中",
                    "value": news_impact
                })
            
            # 6. 检测情绪趋势变化
            sentiment_change_terms = ["转为乐观", "转为悲观", "明显乐观", "明显悲观"]
            if any(term in news_trend for term in sentiment_change_terms) and any(term in social_trend for term in sentiment_change_terms):
                anomalies.append({
                    "type": "rapid_sentiment_shift",
                    "description": f"市场情绪快速转变：新闻{news_trend}，社交媒体{social_trend}",
                    "severity": "高",
                    "value": max(abs(news_sentiment_score), abs(social_sentiment_score))
                })
            
            # 7. 检测与市场数据的不匹配（如果提供了市场数据）
            if market_data:
                market_direction = None
                if "indices" in market_data:
                    # 简单判断大盘方向，可根据实际市场数据结构调整
                    for index_name, index_value in market_data["indices"].items():
                        if "change" in market_data["indices"][index_name]:
                            change = market_data["indices"][index_name]["change"]
                            if change > 0.01:
                                market_direction = "上涨"
                            elif change < -0.01:
                                market_direction = "下跌"
                            else:
                                market_direction = "横盘"
                            break
                
                if market_direction:
                    # 情绪与市场方向不一致
                    sentiment_avg = (news_sentiment_score + social_sentiment_score) / 2
                    if (sentiment_avg > 0.4 and market_direction == "下跌") or (sentiment_avg < -0.4 and market_direction == "上涨"):
                        anomalies.append({
                            "type": "sentiment_market_mismatch",
                            "description": f"市场情绪与市场走势不匹配：情绪{'乐观' if sentiment_avg > 0 else '悲观'}但市场{market_direction}",
                            "severity": "高" if abs(sentiment_avg) > 0.6 else "中",
                            "value": sentiment_avg
                        })
            
            self.logger.info(f"检测到{len(anomalies)}个情绪异常模式")
            return anomalies
            
        except Exception as e:
            self.logger.error(f"检测情绪异常模式失败: {str(e)}")
            traceback.print_exc()
            return [] 