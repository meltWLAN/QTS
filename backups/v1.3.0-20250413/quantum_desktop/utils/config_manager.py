"""
配置管理器 - 管理应用程序配置
"""

import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("QuantumDesktop.ConfigManager")

class ConfigManager:
    """配置管理器 - 加载和保存应用程序配置"""
    
    def __init__(self):
        self.config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
        self.config_file = os.path.join(self.config_dir, "config.json")
        self.config = self.load_config()
        
    def load_config(self):
        """加载配置"""
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)
            
        if not os.path.exists(self.config_file):
            return self.create_default_config()
            
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return self.create_default_config()
            
    def create_default_config(self):
        """创建默认配置"""
        default_config = {
            "theme": "dark",
            "auto_start": True,
            "window": {
                "width": 1200,
                "height": 800,
                "maximized": False
            },
            "system": {
                "log_level": "INFO",
                "data_dir": "data",
                "backup_dir": "backup"
            }
        }
        
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"创建默认配置文件失败: {e}")
            
        return default_config
        
    def save_config(self):
        """保存配置"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")
            return False
            
    def get(self, key, default=None):
        """获取配置项"""
        return self.config.get(key, default)
        
    def set(self, key, value):
        """设置配置项"""
        self.config[key] = value
        return self.save_config()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "app": {
                "theme": "dark",
                "language": "zh_CN",
                "first_run": True,
                "window_size": [1200, 800],
                "window_position": [100, 100]
            },
            "system": {
                "auto_start": False,
                "data_cache_size": 500,
                "log_level": "INFO",
                "update_check": True
            },
            "quantum": {
                "default_backend": "simulator",
                "max_qubits": 16,
                "shots": 1024
            },
            "market": {
                "data_sources": ["local", "online"],
                "default_symbols": ["AAPL", "MSFT", "GOOGL"],
                "update_interval": 60
            }
        }
        
    def reset_config(self) -> bool:
        """重置为默认配置"""
        self.config = self._get_default_config()
        return self.save_config()
        
    def _merge_config(self, target: Dict, source: Dict) -> None:
        """合并配置（递归）"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._merge_config(target[key], value)
            else:
                target[key] = value 