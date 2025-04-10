#!/usr/bin/env python3
"""
新闻爬虫模块 - 获取财经新闻和社交媒体数据
为情绪分析提供数据源
"""

import os
import time
import json
import random
import logging
import requests
from datetime import datetime, timedelta
import traceback

# 配置日志
logger = logging.getLogger("NewsCrawler")

class NewsCrawler:
    """新闻爬虫 - 获取财经新闻和社交媒体数据"""
    
    def __init__(self):
        """初始化新闻爬虫"""
        self.logger = logging.getLogger("NewsCrawler")
        self.logger.info("初始化新闻爬虫...")
        
        # 基本配置
        self.initialized = False
        self.active = False
        self.last_update = datetime.now()
        
        # 新闻源配置
        self.news_sources = {
            "financial_news": ["东方财富网", "新浪财经", "华尔街见闻", "金融时报", "Bloomberg", "Reuters"],
            "social_media": ["微博财经", "股吧", "Reddit/wallstreetbets", "Twitter财经"],
            "official_releases": ["央行公告", "证监会公告", "公司公告", "美联储声明"]
        }
        
        # 缓存
        self.news_cache = {}
        self.cache_expiry = 3600  # 缓存1小时过期
        
        # 新闻主题词典
        self.topics = [
            "货币政策", "财政政策", "利率调整", "通胀数据", "经济增长", "失业率",
            "贸易战", "国际关系", "地缘政治", "产业政策", "科技革新", "能源转型",
            "IPO", "并购", "业绩预告", "财报发布", "股票回购", "分红派息",
            "行业监管", "公司治理", "数据泄露", "环境问题", "供应链危机", "芯片短缺"
        ]
        
        self.logger.info("新闻爬虫初始化完成")
    
    def initialize(self):
        """初始化爬虫"""
        self.logger.info("初始化新闻爬虫连接...")
        
        # 模拟初始化连接
        time.sleep(0.5)
        
        self.initialized = True
        self.active = True
        
        self.logger.info("新闻爬虫初始化完成")
        return True
    
    def fetch_financial_news(self, keywords=None, days=1, limit=50):
        """获取财经新闻
        
        参数:
            keywords (list): 关键词列表
            days (int): 获取最近几天的新闻
            limit (int): 最大新闻数量
            
        返回:
            list: 新闻列表
        """
        if not self.active:
            self.logger.warning("新闻爬虫未激活，无法获取新闻")
            return []
        
        # 缓存键
        cache_key = f"financial_{'-'.join(keywords) if keywords else 'all'}_{days}_{limit}"
        
        # 检查缓存
        if cache_key in self.news_cache:
            cache_time, news_data = self.news_cache[cache_key]
            if (datetime.now() - cache_time).total_seconds() < self.cache_expiry:
                self.logger.info(f"使用缓存的财经新闻数据: {len(news_data)}条")
                return news_data
        
        self.logger.info(f"获取财经新闻: 关键词={keywords}, 天数={days}, 限制={limit}")
        
        try:
            # 模拟获取新闻数据
            news_list = self._generate_mock_financial_news(keywords, days, limit)
            
            # 更新缓存
            self.news_cache[cache_key] = (datetime.now(), news_list)
            
            self.logger.info(f"获取到{len(news_list)}条财经新闻")
            return news_list
            
        except Exception as e:
            self.logger.error(f"获取财经新闻失败: {str(e)}")
            traceback.print_exc()
            return []
    
    def fetch_social_sentiment(self, stock_codes=None, days=1, limit=100):
        """获取社交媒体情绪数据
        
        参数:
            stock_codes (list): 股票代码列表
            days (int): 获取最近几天的数据
            limit (int): 最大数据量
            
        返回:
            list: 社交媒体情绪数据
        """
        if not self.active:
            self.logger.warning("新闻爬虫未激活，无法获取社交媒体数据")
            return []
        
        # 缓存键
        cache_key = f"social_{'-'.join(stock_codes) if stock_codes else 'market'}_{days}_{limit}"
        
        # 检查缓存
        if cache_key in self.news_cache:
            cache_time, data = self.news_cache[cache_key]
            if (datetime.now() - cache_time).total_seconds() < self.cache_expiry:
                self.logger.info(f"使用缓存的社交媒体数据: {len(data)}条")
                return data
        
        self.logger.info(f"获取社交媒体情绪: 股票={stock_codes}, 天数={days}, 限制={limit}")
        
        try:
            # 模拟获取社交媒体数据
            social_data = self._generate_mock_social_sentiment(stock_codes, days, limit)
            
            # 更新缓存
            self.news_cache[cache_key] = (datetime.now(), social_data)
            
            self.logger.info(f"获取到{len(social_data)}条社交媒体情绪数据")
            return social_data
            
        except Exception as e:
            self.logger.error(f"获取社交媒体情绪失败: {str(e)}")
            traceback.print_exc()
            return []
    
    def fetch_official_announcements(self, days=7, limit=20):
        """获取官方公告
        
        参数:
            days (int): 获取最近几天的公告
            limit (int): 最大公告数量
            
        返回:
            list: 公告列表
        """
        if not self.active:
            self.logger.warning("新闻爬虫未激活，无法获取官方公告")
            return []
        
        # 缓存键
        cache_key = f"official_{days}_{limit}"
        
        # 检查缓存
        if cache_key in self.news_cache:
            cache_time, data = self.news_cache[cache_key]
            if (datetime.now() - cache_time).total_seconds() < self.cache_expiry:
                self.logger.info(f"使用缓存的官方公告: {len(data)}条")
                return data
        
        self.logger.info(f"获取官方公告: 天数={days}, 限制={limit}")
        
        try:
            # 模拟获取官方公告
            announcements = self._generate_mock_announcements(days, limit)
            
            # 更新缓存
            self.news_cache[cache_key] = (datetime.now(), announcements)
            
            self.logger.info(f"获取到{len(announcements)}条官方公告")
            return announcements
            
        except Exception as e:
            self.logger.error(f"获取官方公告失败: {str(e)}")
            traceback.print_exc()
            return []
    
    def get_market_news_summary(self, days=3):
        """获取市场新闻摘要
        
        参数:
            days (int): 获取最近几天的新闻摘要
            
        返回:
            dict: 新闻摘要
        """
        if not self.active:
            self.logger.warning("新闻爬虫未激活，无法获取市场新闻摘要")
            return {}
        
        self.logger.info(f"获取市场新闻摘要: 天数={days}")
        
        try:
            # 获取各类新闻数据
            financial_news = self.fetch_financial_news(days=days, limit=100)
            social_data = self.fetch_social_sentiment(days=days, limit=150)
            announcements = self.fetch_official_announcements(days=days, limit=30)
            
            # 计算情绪分布
            sentiment_distribution = {
                "positive": 0,
                "neutral": 0,
                "negative": 0
            }
            
            for news in financial_news:
                sentiment = news.get("sentiment", "neutral")
                sentiment_distribution[sentiment] = sentiment_distribution.get(sentiment, 0) + 1
            
            for post in social_data:
                sentiment = post.get("sentiment", "neutral")
                sentiment_distribution[sentiment] = sentiment_distribution.get(sentiment, 0) + 1
            
            # 计算主题分布
            topic_distribution = {}
            
            for news in financial_news + announcements:
                topics = news.get("topics", [])
                for topic in topics:
                    topic_distribution[topic] = topic_distribution.get(topic, 0) + 1
            
            # 主题排序
            sorted_topics = sorted(topic_distribution.items(), key=lambda x: x[1], reverse=True)
            top_topics = sorted_topics[:10]
            
            # 计算热门股票
            stock_mentions = {}
            
            for post in social_data:
                stocks = post.get("mentioned_stocks", [])
                for stock in stocks:
                    stock_mentions[stock] = stock_mentions.get(stock, 0) + 1
            
            # 股票排序
            sorted_stocks = sorted(stock_mentions.items(), key=lambda x: x[1], reverse=True)
            hot_stocks = sorted_stocks[:10]
            
            # 构建摘要
            summary = {
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "period_days": days,
                "data_sources": {
                    "financial_news": len(financial_news),
                    "social_posts": len(social_data),
                    "official_announcements": len(announcements)
                },
                "sentiment_distribution": sentiment_distribution,
                "top_topics": [{
                    "topic": topic,
                    "mentions": count
                } for topic, count in top_topics],
                "hot_stocks": [{
                    "stock": stock,
                    "mentions": count
                } for stock, count in hot_stocks],
                "important_announcements": [a for a in announcements if a.get("importance", 0) > 7][:5]
            }
            
            self.logger.info("市场新闻摘要生成完成")
            return summary
            
        except Exception as e:
            self.logger.error(f"获取市场新闻摘要失败: {str(e)}")
            traceback.print_exc()
            return {}
    
    def _generate_mock_financial_news(self, keywords=None, days=1, limit=50):
        """生成模拟财经新闻数据"""
        news_list = []
        
        # 随机生成新闻
        for _ in range(limit):
            # 随机选择新闻源
            source = random.choice(self.news_sources["financial_news"])
            
            # 随机选择主题
            if keywords:
                topics = random.sample(keywords, min(3, len(keywords)))
            else:
                topics = random.sample(self.topics, random.randint(1, 3))
            
            # 随机生成标题
            title_templates = [
                "{topic}最新进展：{detail}",
                "突发！{topic}出现重大变化",
                "{source}独家：{topic}分析",
                "每日要闻：{topic}与市场影响",
                "{topic}：{detail}，投资者需关注",
                "警惕！{topic}可能带来的风险",
                "利好！{topic}带来新机遇",
                "专家解读：{topic}背后的逻辑",
                "{topic}走势分析：{detail}",
                "重磅报道：{topic}新政策解读"
            ]
            
            title_template = random.choice(title_templates)
            topic = random.choice(topics)
            
            details = [
                "市场反应积极", "引发投资者担忧", "专家持谨慎态度", 
                "或将改变行业格局", "机构投资者纷纷布局", "分析师预期乐观",
                "短期波动加剧", "长期影响值得关注", "政策导向明确",
                "数据表现超预期"
            ]
            
            title = title_template.format(
                topic=topic,
                source=source,
                detail=random.choice(details)
            )
            
            # 随机发布时间
            published_time = datetime.now() - timedelta(
                days=random.uniform(0, days),
                hours=random.uniform(0, 24)
            )
            
            # 随机情感倾向
            sentiment_weights = {"positive": 0.35, "neutral": 0.4, "negative": 0.25}
            sentiment = random.choices(
                list(sentiment_weights.keys()),
                weights=list(sentiment_weights.values())
            )[0]
            
            # 随机重要性和影响
            importance = random.randint(1, 10)
            market_impact = random.uniform(0, 1) * importance / 10
            
            # 构建新闻对象
            news = {
                "id": f"news_{int(time.time())}_{_}",
                "title": title,
                "source": source,
                "url": f"https://example.com/news/{int(time.time())}-{_}",
                "published_time": published_time.strftime('%Y-%m-%d %H:%M:%S'),
                "topics": topics,
                "sentiment": sentiment,
                "importance": importance,
                "market_impact": market_impact,
                "mentioned_stocks": []
            }
            
            # 随机添加提到的股票
            if random.random() < 0.7:  # 70%的新闻会提到股票
                stock_count = random.randint(1, 5)
                stock_pool = [
                    {"code": "000001", "name": "平安银行"},
                    {"code": "000333", "name": "美的集团"},
                    {"code": "000651", "name": "格力电器"},
                    {"code": "000858", "name": "五粮液"},
                    {"code": "002594", "name": "比亚迪"},
                    {"code": "600000", "name": "浦发银行"},
                    {"code": "600036", "name": "招商银行"},
                    {"code": "600276", "name": "恒瑞医药"},
                    {"code": "600519", "name": "贵州茅台"},
                    {"code": "601318", "name": "中国平安"}
                ]
                
                mentioned_stocks = random.sample(stock_pool, stock_count)
                news["mentioned_stocks"] = [f"{s['code']}({s['name']})" for s in mentioned_stocks]
            
            news_list.append(news)
        
        # 按发布时间排序
        news_list.sort(key=lambda x: x["published_time"], reverse=True)
        
        return news_list
    
    def _generate_mock_social_sentiment(self, stock_codes=None, days=1, limit=100):
        """生成模拟社交媒体情绪数据"""
        social_data = []
        
        # 社交媒体平台
        platforms = self.news_sources["social_media"]
        
        # 情绪词典
        sentiment_terms = {
            "positive": ["看涨", "买入", "利好", "强势", "突破", "机会", "牛市", "上涨", "潜力", "推荐"],
            "negative": ["看跌", "卖出", "利空", "弱势", "跌破", "风险", "熊市", "下跌", "套牢", "警惕"],
            "neutral": ["观望", "持仓", "震荡", "盘整", "分歧", "谨慎", "波动", "反复", "调整", "等待"]
        }
        
        # 支持的股票池
        stock_pool = [
            {"code": "000001", "name": "平安银行"},
            {"code": "000333", "name": "美的集团"},
            {"code": "000651", "name": "格力电器"},
            {"code": "000858", "name": "五粮液"},
            {"code": "002594", "name": "比亚迪"},
            {"code": "600000", "name": "浦发银行"},
            {"code": "600036", "name": "招商银行"},
            {"code": "600276", "name": "恒瑞医药"},
            {"code": "600519", "name": "贵州茅台"},
            {"code": "601318", "name": "中国平安"}
        ]
        
        # 过滤股票池
        if stock_codes:
            filtered_pool = [s for s in stock_pool if s["code"] in stock_codes]
            if filtered_pool:
                stock_pool = filtered_pool
        
        # 生成社交媒体帖子
        for _ in range(limit):
            # 随机选择平台
            platform = random.choice(platforms)
            
            # 随机选择股票
            mentioned_stocks = []
            stock_count = random.randint(1, 3)
            for _ in range(stock_count):
                stock = random.choice(stock_pool)
                mentioned_stocks.append(f"{stock['code']}({stock['name']})")
            
            # 随机情感倾向
            sentiment = random.choice(list(sentiment_terms.keys()))
            
            # 随机生成内容
            terms = sentiment_terms[sentiment]
            term1 = random.choice(terms)
            term2 = random.choice(terms)
            
            content_templates = [
                "{stock}今天表现不错，{term1}！",
                "分析{stock}走势，个人认为{term1}，{term2}",
                "{stock}要{term1}了吗？大家怎么看？",
                "今日操作：{stock} {term1}",
                "{term1}信号出现，{stock}值得关注",
                "{stock}技术面{term1}，基本面{term2}",
                "重点推荐{stock}，理由是{term1}",
                "{stock}要注意风险，{term1}",
                "行业趋势向好，{stock}有望{term1}",
                "经过详细分析，{stock}建议{term1}"
            ]
            
            content_template = random.choice(content_templates)
            stock_mention = random.choice(mentioned_stocks) if mentioned_stocks else "大盘"
            
            content = content_template.format(
                stock=stock_mention,
                term1=term1,
                term2=term2
            )
            
            # 随机发布时间
            published_time = datetime.now() - timedelta(
                days=random.uniform(0, days),
                hours=random.uniform(0, 24)
            )
            
            # 随机影响力和互动
            influence = random.uniform(0, 1)
            likes = int(influence * random.randint(10, 1000))
            comments = int(influence * random.randint(5, 200))
            
            # 构建社交媒体数据对象
            post = {
                "id": f"social_{int(time.time())}_{_}",
                "platform": platform,
                "content": content,
                "published_time": published_time.strftime('%Y-%m-%d %H:%M:%S'),
                "sentiment": sentiment,
                "mentioned_stocks": mentioned_stocks,
                "likes": likes,
                "comments": comments,
                "influence_score": influence,
                "topics": random.sample(self.topics, random.randint(0, 2))
            }
            
            social_data.append(post)
        
        # 按发布时间排序
        social_data.sort(key=lambda x: x["published_time"], reverse=True)
        
        return social_data
    
    def _generate_mock_announcements(self, days=7, limit=20):
        """生成模拟官方公告"""
        announcements = []
        
        # 公告来源
        sources = self.news_sources["official_releases"]
        
        # 公告类型
        announcement_types = [
            "政策发布", "监管措施", "数据发布", "会议纪要", "重要讲话",
            "业绩预告", "分红公告", "董事会决议", "股东变动", "重大交易"
        ]
        
        # 生成公告
        for _ in range(limit):
            # 随机选择来源和类型
            source = random.choice(sources)
            announcement_type = random.choice(announcement_types)
            
            # 随机主题
            topics = random.sample(self.topics, random.randint(1, 2))
            
            # 随机标题
            if source in ["央行公告", "证监会公告", "美联储声明"]:
                title_templates = [
                    "{source}发布{topic}相关政策",
                    "{source}关于{topic}的指导意见",
                    "{source}公布{date}{topic}数据",
                    "{source}召开{topic}专题会议",
                    "{source}主席就{topic}发表讲话"
                ]
            else:  # 公司公告
                title_templates = [
                    "{公司}发布{year}年{quarter}季度财报",
                    "{公司}关于{topic}的公告",
                    "{公司}董事会决议公告",
                    "{公司}{year}年度分红方案",
                    "{公司}重大资产重组进展公告",
                    "{公司}关于股东减持计划的公告",
                    "{公司}新产品发布公告"
                ]
            
            title_template = random.choice(title_templates)
            
            # 随机日期
            date = (datetime.now() - timedelta(days=random.randint(0, days-1))).strftime('%Y年%m月%d日')
            year = date[:4]
            quarter = random.choice(["一", "二", "三", "四"])
            
            # 随机公司
            companies = ["贵州茅台", "中国平安", "招商银行", "格力电器", "美的集团", "比亚迪", "恒瑞医药", "五粮液"]
            company = random.choice(companies)
            
            # 生成标题
            title = title_template.format(
                source=source,
                topic=random.choice(topics),
                date=date,
                公司=company,
                year=year,
                quarter=quarter
            )
            
            # 随机发布时间
            published_time = datetime.now() - timedelta(
                days=random.uniform(0, days),
                hours=random.uniform(0, 24)
            )
            
            # 随机重要性
            importance = random.randint(1, 10)
            
            # 构建公告对象
            announcement = {
                "id": f"announcement_{int(time.time())}_{_}",
                "title": title,
                "source": source,
                "type": announcement_type,
                "url": f"https://example.com/announcement/{int(time.time())}-{_}",
                "published_time": published_time.strftime('%Y-%m-%d %H:%M:%S'),
                "topics": topics,
                "importance": importance,
                "mentioned_stocks": []
            }
            
            # 对于公司公告，添加对应的股票
            if source == "公司公告":
                if company == "贵州茅台":
                    announcement["mentioned_stocks"] = ["600519(贵州茅台)"]
                elif company == "中国平安":
                    announcement["mentioned_stocks"] = ["601318(中国平安)"]
                elif company == "招商银行":
                    announcement["mentioned_stocks"] = ["600036(招商银行)"]
                elif company == "格力电器":
                    announcement["mentioned_stocks"] = ["000651(格力电器)"]
                elif company == "美的集团":
                    announcement["mentioned_stocks"] = ["000333(美的集团)"]
                elif company == "比亚迪":
                    announcement["mentioned_stocks"] = ["002594(比亚迪)"]
                elif company == "恒瑞医药":
                    announcement["mentioned_stocks"] = ["600276(恒瑞医药)"]
                elif company == "五粮液":
                    announcement["mentioned_stocks"] = ["000858(五粮液)"]
            
            announcements.append(announcement)
        
        # 按发布时间排序
        announcements.sort(key=lambda x: x["published_time"], reverse=True)
        
        return announcements 