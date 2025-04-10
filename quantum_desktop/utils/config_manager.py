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
    
    def __init__(self, config_path: str = None):
        # 默认配置路径
        if config_path is None:
            self.config_dir = os.path.join(os.path.expanduser("~"), ".quantum_desktop")
            self.config_path = os.path.join(self.config_dir, "config.json")
        else:
            self.config_path = config_path
            self.config_dir = os.path.dirname(config_path)
            
        # 确保配置目录存在
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)
            
        # 初始化默认配置
        self.config = self._get_default_config()
        
        # 加载配置
        self.load_config()
        
        logger.info(f"配置管理器初始化完成，配置路径: {self.config_path}")
        
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
        
    def load_config(self) -> bool:
        """加载配置文件"""
        if not os.path.exists(self.config_path):
            logger.info("配置文件不存在，使用默认配置")
            self.save_config()  # 创建默认配置文件
            return True
            
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                
            # 合并加载的配置与默认配置
            self._merge_config(self.config, loaded_config)
            logger.info("配置加载成功")
            return True
            
        except Exception as e:
            logger.error(f"加载配置文件时出错: {str(e)}")
            return False
            
    def save_config(self) -> bool:
        """保存配置到文件"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
                
            logger.info("配置保存成功")
            return True
            
        except Exception as e:
            logger.error(f"保存配置文件时出错: {str(e)}")
            return False
            
    def get_config(self, section: str = None, key: str = None) -> Any:
        """获取配置值"""
        if section is None:
            return self.config
            
        if key is None:
            return self.config.get(section, {})
            
        return self.config.get(section, {}).get(key)
        
    def set_config(self, section: str, key: str, value: Any) -> bool:
        """设置配置值"""
        if section not in self.config:
            self.config[section] = {}
            
        self.config[section][key] = value
        return self.save_config()
        
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