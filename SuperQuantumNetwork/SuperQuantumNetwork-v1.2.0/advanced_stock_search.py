#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - 高级股票搜索组件
支持多种格式股票代码输入、模糊搜索、智能匹配
"""

import os
import logging
import re
import time
import pandas as pd
import difflib
from fuzzywuzzy import fuzz, process

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AdvancedStockSearch")

class StockCodeConverter:
    """股票代码格式转换器"""
    
    @staticmethod
    def to_tushare_format(code):
        """将股票代码转换为Tushare格式
        
        Args:
            code: 输入的股票代码
            
        Returns:
            str: Tushare格式的股票代码
        """
        # 去除空格和特殊字符
        code = re.sub(r'\s+', '', code)
        
        # 已经是标准格式的情况
        if re.match(r'\d{6}\.(SH|SZ|BJ)', code, re.I):
            return code.upper()
        
        # 处理带交易所字母前缀的情况: sh600000, sz000001等
        prefix_match = re.match(r'(sh|sz|bj)(\d{6})', code, re.I)
        if prefix_match:
            exchange = prefix_match.group(1).upper()
            number = prefix_match.group(2)
            return f"{number}.{exchange}"
        
        # 处理纯数字的情况
        if re.match(r'\d{6}', code):
            # 上交所股票: 6开头
            if code.startswith('6'):
                return f"{code}.SH"
            # 科创板股票: 688开头
            elif code.startswith('688'):
                return f"{code}.SH"
            # 深交所股票: 0开头或3开头
            elif code.startswith('0') or code.startswith('3'):
                return f"{code}.SZ"
            # 北交所股票: 4、8开头
            elif code.startswith('4') or code.startswith('8'):
                return f"{code}.BJ"
        
        # 无法识别的格式，返回原始代码
        return code
    
    @staticmethod
    def get_market_by_code(code):
        """根据股票代码获取市场名称
        
        Args:
            code: 股票代码
            
        Returns:
            str: 市场名称
        """
        # 转换为标准格式
        ts_code = StockCodeConverter.to_tushare_format(code)
        
        if ts_code.endswith('.SH'):
            return "上海证券交易所"
        elif ts_code.endswith('.SZ'):
            return "深圳证券交易所"
        elif ts_code.endswith('.BJ'):
            return "北京证券交易所"
        else:
            return "未知市场"
    
    @staticmethod
    def normalize_code(code):
        """标准化股票代码（去除交易所后缀，只保留6位数字）
        
        Args:
            code: 股票代码
            
        Returns:
            str: 标准化后的股票代码
        """
        # 去除空格和特殊字符
        code = re.sub(r'\s+', '', code)
        
        # 处理带交易所后缀的情况: 600000.SH, 000001.SZ等
        suffix_match = re.match(r'(\d{6})\.(SH|SZ|BJ)', code, re.I)
        if suffix_match:
            return suffix_match.group(1)
        
        # 处理带交易所字母前缀的情况: sh600000, sz000001等
        prefix_match = re.match(r'(sh|sz|bj)(\d{6})', code, re.I)
        if prefix_match:
            return prefix_match.group(2)
        
        # 处理纯数字的情况
        if re.match(r'\d{6}', code):
            return code
        
        # 无法识别的格式，返回原始代码
        return code


class AdvancedStockSearch:
    """高级股票搜索类
    支持模糊搜索、拼音匹配、股票代码智能识别等功能
    """
    
    def __init__(self, data_source=None):
        """初始化高级股票搜索
        
        Args:
            data_source: 数据源对象，需要实现get_stock_list方法
        """
        self.logger = logging.getLogger("AdvancedStockSearch")
        self.data_source = data_source
        self.stocks = []
        self.stock_map = {}  # 股票代码 -> 股票信息
        self.name_map = {}   # 股票名称 -> 股票信息
        self.industry_map = {}  # 行业 -> 股票列表
        self.area_map = {}      # 地区 -> 股票列表
        
        # 加载拼音模块
        try:
            from pypinyin import lazy_pinyin
            self.pinyin_func = lazy_pinyin
            self.pinyin_available = True
        except ImportError:
            self.pinyin_available = False
            self.logger.warning("未安装pypinyin模块，拼音搜索功能不可用")

        # 初始化股票列表
        self.update_stock_list()
    
    def update_stock_list(self):
        """更新股票列表"""
        if self.data_source is None:
            self.logger.warning("未提供数据源，无法获取股票列表")
            return
        
        try:
            # 获取股票列表
            stocks = self.data_source.get_stock_list()
            if not stocks:
                self.logger.warning("获取股票列表为空")
                return
            
            self.stocks = stocks
            
            # 更新索引
            self._update_indices()
            
            self.logger.info(f"股票列表更新成功，共 {len(stocks)} 只股票")
        except Exception as e:
            self.logger.error(f"更新股票列表失败: {str(e)}")
    
    def _update_indices(self):
        """更新各种索引"""
        self.stock_map = {}
        self.name_map = {}
        self.industry_map = {}
        self.area_map = {}
        
        for stock in self.stocks:
            # 更新代码映射
            ts_code = stock.get('ts_code', '')
            code = stock.get('code', '')
            if ts_code:
                self.stock_map[ts_code] = stock
            if code:
                self.stock_map[code] = stock
            
            # 更新名称映射
            name = stock.get('name', '')
            if name:
                self.name_map[name] = stock
                
                # 更新拼音映射
                if self.pinyin_available:
                    pinyin = ''.join(self.pinyin_func(name))
                    first_letters = ''.join([p[0] for p in self.pinyin_func(name)])
                    self.name_map[pinyin] = stock
                    self.name_map[first_letters] = stock
            
            # 更新行业映射
            industry = stock.get('industry', '')
            if industry:
                if industry not in self.industry_map:
                    self.industry_map[industry] = []
                self.industry_map[industry].append(stock)
            
            # 更新地区映射
            area = stock.get('area', '')
            if area:
                if area not in self.area_map:
                    self.area_map[area] = []
                self.area_map[area].append(stock)
    
    def find_stock(self, query, max_results=10, threshold=70):
        """查找股票
        
        Args:
            query: 查询字符串，可以是股票代码、名称或拼音
            max_results: 最大返回结果数量
            threshold: 相似度阈值，0-100
            
        Returns:
            list: 匹配的股票列表
        """
        if not query or not self.stocks:
            return []
        
        # 规范化查询字符串
        query = query.strip().upper()
        
        # 直接匹配结果
        direct_matches = []
        
        # 1. 尝试精确匹配股票代码
        normalized_code = StockCodeConverter.normalize_code(query)
        ts_code = StockCodeConverter.to_tushare_format(query)
        
        for code in [query, normalized_code, ts_code]:
            if code in self.stock_map:
                direct_matches.append(self.stock_map[code])
        
        # 2. 尝试精确匹配股票名称
        if query in self.name_map:
            direct_matches.append(self.name_map[query])
        
        # 去重
        direct_matches = [dict(t) for t in {tuple(d.items()) for d in direct_matches}]
        
        # 如果有精确匹配，直接返回
        if direct_matches:
            return direct_matches[:max_results]
        
        # 3. 尝试模糊匹配
        fuzzy_matches = []
        
        # 代码模糊匹配
        for code, stock in self.stock_map.items():
            if code and (code.startswith(query) or query in code):
                fuzzy_matches.append((stock, 95))  # 代码部分匹配，给高分
        
        # 名称模糊匹配
        for name, stock in self.name_map.items():
            # 使用fuzzywuzzy计算相似度
            similarity = fuzz.partial_ratio(query, name)
            if similarity >= threshold:
                fuzzy_matches.append((stock, similarity))
            
            # 包含关系
            elif query in name:
                fuzzy_matches.append((stock, 90))  # 名称包含查询词，给高分
        
        # 行业匹配
        for industry, stocks in self.industry_map.items():
            if query in industry:
                for stock in stocks:
                    fuzzy_matches.append((stock, 80))  # 行业匹配，给较低分
        
        # 排序、去重并返回结果
        fuzzy_matches.sort(key=lambda x: x[1], reverse=True)
        
        # 去重
        unique_results = []
        seen_codes = set()
        
        for stock, score in fuzzy_matches:
            ts_code = stock.get('ts_code', '')
            if ts_code and ts_code not in seen_codes:
                seen_codes.add(ts_code)
                stock['match_score'] = score  # 添加匹配分数
                unique_results.append(stock)
                
                if len(unique_results) >= max_results:
                    break
        
        return unique_results
    
    def search_by_industry(self, industry, limit=50):
        """按行业搜索股票
        
        Args:
            industry: 行业名称
            limit: 返回数量限制
            
        Returns:
            list: 匹配的股票列表
        """
        if not industry or not self.industry_map:
            return []
        
        # 精确匹配
        if industry in self.industry_map:
            return self.industry_map[industry][:limit]
        
        # 模糊匹配
        matches = []
        for ind, stocks in self.industry_map.items():
            if industry in ind:
                matches.extend(stocks)
        
        # 去重
        unique_results = []
        seen_codes = set()
        
        for stock in matches:
            ts_code = stock.get('ts_code', '')
            if ts_code and ts_code not in seen_codes:
                seen_codes.add(ts_code)
                unique_results.append(stock)
                
                if len(unique_results) >= limit:
                    break
        
        return unique_results
    
    def search_by_area(self, area, limit=50):
        """按地区搜索股票
        
        Args:
            area: 地区名称
            limit: 返回数量限制
            
        Returns:
            list: 匹配的股票列表
        """
        if not area or not self.area_map:
            return []
        
        # 精确匹配
        if area in self.area_map:
            return self.area_map[area][:limit]
        
        # 模糊匹配
        matches = []
        for a, stocks in self.area_map.items():
            if area in a:
                matches.extend(stocks)
        
        # 去重
        unique_results = []
        seen_codes = set()
        
        for stock in matches:
            ts_code = stock.get('ts_code', '')
            if ts_code and ts_code not in seen_codes:
                seen_codes.add(ts_code)
                unique_results.append(stock)
                
                if len(unique_results) >= limit:
                    break
        
        return unique_results


def example_usage():
    """示例用法"""
    # 导入数据源
    try:
        from quantum_symbiotic_network.data_sources.tushare_data_source import TushareDataSource
        data_source = TushareDataSource()
        
        # 创建搜索器
        searcher = AdvancedStockSearch(data_source)
        
        # 测试搜索
        test_queries = [
            "600000",       # 代码
            "浦发银行",      # 名称
            "sh600000",     # 带前缀代码
            "600000.SH",    # 带后缀代码
            "浦发",         # 部分名称
            "银行",         # 部分名称
            "pufa",         # 拼音
            "上海",         # 地区
            "金融",         # 行业
        ]
        
        for query in test_queries:
            print(f"\n搜索: {query}")
            results = searcher.find_stock(query)
            
            if results:
                print(f"找到 {len(results)} 个结果:")
                for i, stock in enumerate(results):
                    score = stock.get('match_score', 100)
                    print(f"{i+1}. {stock['name']}({stock['ts_code']}) - 行业: {stock.get('industry', '未知')} - 匹配度: {score}%")
            else:
                print("未找到匹配结果")
                
        # 测试行业搜索
        industry = "银行"
        print(f"\n按行业搜索: {industry}")
        results = searcher.search_by_industry(industry, 5)
        
        if results:
            print(f"找到 {len(results)} 个结果:")
            for i, stock in enumerate(results):
                print(f"{i+1}. {stock['name']}({stock['ts_code']}) - 行业: {stock.get('industry', '未知')}")
    
    except Exception as e:
        print(f"示例运行失败: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 运行示例
    example_usage() 