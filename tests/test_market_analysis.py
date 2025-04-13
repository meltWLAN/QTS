#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
市场分析面板单元测试
测试数据获取、API设置以及数据验证功能
"""

import sys
import os
import unittest
import logging
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入测试目标
from quantum_desktop.ui.panels.market_analysis_panel import MarketAnalysisPanel

# 配置日志记录
logging.basicConfig(level=logging.DEBUG,
                  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('test_market_analysis')

class TestMarketAnalysisPanel(unittest.TestCase):
    """市场分析面板测试类"""
    
    @classmethod
    def setUpClass(cls):
        """创建QApplication实例，只需要一次"""
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication(sys.argv)
            
    def setUp(self):
        """每个测试方法前执行"""
        # 创建模拟的SystemManager
        self.mock_system_manager = MagicMock()
        self.mock_system_manager.get_component.return_value = MagicMock()
        self.mock_system_manager.get_component_status.return_value = {'analyzer': 'running'}
        
        # 创建测试对象
        self.panel = MarketAnalysisPanel(self.mock_system_manager)
        
    def test_init(self):
        """测试初始化"""
        self.assertIsNotNone(self.panel)
        self.assertEqual(self.panel.system_manager, self.mock_system_manager)
        self.assertEqual(self.panel.current_symbol, None)
        self.assertTrue(self.panel.use_real_data)
        
    def test_api_key_saving(self):
        """测试API密钥保存功能"""
        # 模拟API密钥输入和保存
        test_api_key = "test_api_key_123456"
        self.panel.api_key_input.setText(test_api_key)
        
        # 模拟测试Tushare API密钥的方法
        with patch.object(self.panel, '_test_tushare_api_key', return_value=None) as mock_test:
            # 保存API密钥
            self.panel._save_api_key()
            
            # 验证API密钥已保存到panel.api_keys中
            self.assertEqual(self.panel.api_keys['tushare'], test_api_key)
            
            # 验证测试方法被调用
            mock_test.assert_called_once_with(test_api_key)
        
    def test_dataframe_conversion(self):
        """测试DataFrame转换为内部字典格式"""
        # 创建测试数据帧
        test_dates = pd.date_range(start=datetime.now() - timedelta(days=10), 
                                 periods=10, freq='D')
        test_df = pd.DataFrame({
            'Open': [100 + i for i in range(10)],
            'High': [110 + i for i in range(10)],
            'Low': [90 + i for i in range(10)],
            'Close': [105 + i for i in range(10)],
            'Volume': [1000000 + i*1000 for i in range(10)]
        }, index=test_dates)
        
        # 转换为内部字典
        result = self.panel._convert_dataframe_to_dict(test_df, 'TEST')
        
        # 验证结果
        self.assertEqual(result['symbol'], 'TEST')
        self.assertEqual(len(result['dates']), 10)
        self.assertEqual(len(result['open']), 10)
        self.assertEqual(len(result['high']), 10)
        self.assertEqual(len(result['low']), 10)
        self.assertEqual(len(result['close']), 10)
        self.assertEqual(len(result['volume']), 10)
        
    def test_create_sample_data(self):
        """测试创建样本数据"""
        sample_data = self.panel._create_sample_data('SAMPLE')
        
        # 验证样本数据结构
        self.assertEqual(sample_data['symbol'], 'SAMPLE')
        self.assertEqual(len(sample_data['dates']), 90)
        self.assertEqual(len(sample_data['open']), 90)
        self.assertEqual(len(sample_data['high']), 90)
        self.assertEqual(len(sample_data['low']), 90)
        self.assertEqual(len(sample_data['close']), 90)
        self.assertEqual(len(sample_data['volume']), 90)
        
    def test_data_authenticity(self):
        """测试数据真实性验证"""
        # 创建有效的测试数据
        valid_df = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [103, 104, 105],
            'Volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range(start=datetime.now() - timedelta(days=3), periods=3))
        
        # 验证有效数据
        self.assertTrue(self.panel._verify_data_authenticity(valid_df, 'TEST', 'test_source'))
        
        # 创建无效数据（价格不一致）
        invalid_df = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [110, 111, 112],  # 收盘价高于最高价
            'Volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range(start=datetime.now() - timedelta(days=3), periods=3))
        
        # 验证无效数据
        self.assertFalse(self.panel._verify_data_authenticity(invalid_df, 'TEST', 'test_source'))
        
    def test_update_market_view(self):
        """测试更新市场视图"""
        # 设置测试数据
        self.panel.current_symbol = "TEST"
        self.panel.market_data = {
            'symbol': 'TEST',
            'dates': [(datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(5, 0, -1)],
            'open': [100, 101, 102, 103, 104],
            'high': [105, 106, 107, 108, 109],
            'low': [95, 96, 97, 98, 99],
            'close': [102, 103, 104, 105, 106],
            'volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        }
        
        # 执行更新
        self.panel._update_market_view()
        
        # 验证表格更新
        self.assertEqual(self.panel.market_table.rowCount(), 5)
        
        # 验证信息标签更新
        self.assertIn("TEST", self.panel.data_info_label.text())
        
    def tearDown(self):
        """每个测试方法后执行"""
        self.panel.deleteLater()
        
if __name__ == '__main__':
    unittest.main() 