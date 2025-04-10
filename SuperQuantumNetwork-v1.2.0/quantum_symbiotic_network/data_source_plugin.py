#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - 数据源插件架构
提供数据源插件的基础类和注册机制，实现数据源的可扩展性
"""

import os
import abc
import json
import logging
import importlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable, Type, Tuple

logger = logging.getLogger("DataSourcePlugin")

class DataSourcePlugin(abc.ABC):
    """数据源插件基类
    
    所有数据源必须继承此基类并实现所需方法
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """初始化数据源插件
        
        Args:
            config: 数据源配置
        """
        self.name = "基础数据源"
        self.version = "1.0.0"
        self.description = "数据源插件基类"
        self.priority = 0  # 优先级，值越大优先级越高
        self.config = config or {}
        self.cache_dir = ""
        self.logger = logging.getLogger(f"DataSource.{self.__class__.__name__}")
        
        # 状态
        self.is_ready = False
        self.last_error = None
        self.last_update_time = None
    
    def initialize(self) -> bool:
        """初始化数据源，检查连接状态
        
        Returns:
            bool: 是否初始化成功
        """
        try:
            self.logger.info(f"初始化数据源: {self.name}")
            self._setup_cache_dir()
            self.is_ready = self._initialize()
            if self.is_ready:
                self.logger.info(f"数据源 {self.name} 初始化成功")
            else:
                self.logger.warning(f"数据源 {self.name} 初始化失败")
            return self.is_ready
        except Exception as e:
            self.is_ready = False
            self.last_error = str(e)
            self.logger.error(f"数据源 {self.name} 初始化出错: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return False
    
    @abc.abstractmethod
    def _initialize(self) -> bool:
        """实际初始化过程，子类必须实现
        
        Returns:
            bool: 是否初始化成功
        """
        pass
    
    def _setup_cache_dir(self) -> None:
        """设置缓存目录"""
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.cache_dir = os.path.join(base_dir, "cache", self.__class__.__name__.lower())
            os.makedirs(self.cache_dir, exist_ok=True)
            self.logger.debug(f"设置缓存目录: {self.cache_dir}")
        except Exception as e:
            self.logger.warning(f"设置缓存目录失败: {str(e)}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取数据源状态
        
        Returns:
            Dict: 状态信息
        """
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "is_ready": self.is_ready,
            "last_error": self.last_error,
            "last_update_time": self.last_update_time.strftime("%Y-%m-%d %H:%M:%S") if self.last_update_time else None,
            "priority": self.priority
        }
    
    @abc.abstractmethod
    def is_available(self) -> bool:
        """检查数据源当前是否可用
        
        Returns:
            bool: 是否可用
        """
        pass
    
    @abc.abstractmethod
    def get_market_status(self) -> Dict[str, Any]:
        """获取市场状态
        
        Returns:
            Dict: 市场状态数据
        """
        pass
    
    @abc.abstractmethod
    def get_indices_data(self) -> List[Dict[str, Any]]:
        """获取指数数据
        
        Returns:
            List[Dict]: 指数数据列表
        """
        pass
    
    @abc.abstractmethod
    def get_stock_data(self, code: str, force_refresh: bool = False) -> Dict[str, Any]:
        """获取单个股票数据
        
        Args:
            code: 股票代码
            force_refresh: 是否强制刷新
        
        Returns:
            Dict: 股票数据
        """
        pass
    
    @abc.abstractmethod
    def get_recommended_stocks(self, count: int = 10) -> List[Dict[str, Any]]:
        """获取推荐股票
        
        Args:
            count: 推荐数量
        
        Returns:
            List[Dict]: 推荐股票列表
        """
        pass
    
    @abc.abstractmethod
    def get_hot_stocks(self, count: int = 5) -> List[Dict[str, Any]]:
        """获取热门股票
        
        Args:
            count: 数量
        
        Returns:
            List[Dict]: 热门股票列表
        """
        pass
    
    @abc.abstractmethod
    def search_stocks(self, keyword: str) -> List[Dict[str, Any]]:
        """搜索股票
        
        Args:
            keyword: 搜索关键词
        
        Returns:
            List[Dict]: 搜索结果列表
        """
        pass
    
    def get_historical_data(self, code: str, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """获取历史数据
        
        Args:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Dict: 历史数据
        """
        raise NotImplementedError("此数据源不支持历史数据查询")
    
    def get_realtime_quote(self, code: str) -> Dict[str, Any]:
        """获取实时行情
        
        Args:
            code: 股票代码
            
        Returns:
            Dict: 实时行情
        """
        raise NotImplementedError("此数据源不支持实时行情查询")
    
    # 缓存操作方法
    def _load_cached_data(self, cache_name: str) -> Any:
        """从缓存加载数据
        
        Args:
            cache_name: 缓存名称
            
        Returns:
            缓存数据，或None
        """
        cache_file = os.path.join(self.cache_dir, f"{cache_name}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.logger.debug(f"从缓存加载{cache_name}数据成功")
                return data
            except Exception as e:
                self.logger.error(f"从缓存加载{cache_name}数据失败: {str(e)}")
        return None
    
    def _save_cached_data(self, cache_name: str, data: Any) -> bool:
        """保存数据到缓存
        
        Args:
            cache_name: 缓存名称
            data: 要缓存的数据
            
        Returns:
            bool: 是否保存成功
        """
        cache_file = os.path.join(self.cache_dir, f"{cache_name}.json")
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.logger.debug(f"保存{cache_name}数据到缓存成功")
            return True
        except Exception as e:
            self.logger.error(f"保存{cache_name}数据到缓存失败: {str(e)}")
            return False
    
    def _get_cache_age(self, cache_name: str) -> Optional[float]:
        """获取缓存的年龄（小时）
        
        Args:
            cache_name: 缓存名称
            
        Returns:
            float: 缓存年龄（小时），或None表示缓存不存在
        """
        cache_file = os.path.join(self.cache_dir, f"{cache_name}.json")
        if os.path.exists(cache_file):
            try:
                modified_time = os.path.getmtime(cache_file)
                current_time = datetime.now().timestamp()
                age_hours = (current_time - modified_time) / 3600
                return age_hours
            except Exception:
                return None
        return None


class DataSourceRegistry:
    """数据源注册表，管理所有可用的数据源插件"""
    
    def __init__(self):
        """初始化数据源注册表"""
        self.plugins = {}
        self.plugin_classes = {}
        self.logger = logging.getLogger("DataSourceRegistry")
        self.logger.info("初始化数据源注册表")
    
    def register_plugin(self, plugin_class: Type[DataSourcePlugin], name: str = None) -> None:
        """注册数据源插件类
        
        Args:
            plugin_class: 插件类
            name: 插件名称，如果为None则使用类名
        """
        if not issubclass(plugin_class, DataSourcePlugin):
            self.logger.error(f"无法注册插件: {plugin_class.__name__} 不是DataSourcePlugin的子类")
            return
        
        plugin_name = name or plugin_class.__name__
        self.plugin_classes[plugin_name] = plugin_class
        self.logger.info(f"注册数据源插件: {plugin_name}")
    
    def create_plugin(self, plugin_name: str, config: Dict[str, Any] = None) -> Optional[DataSourcePlugin]:
        """创建数据源插件实例
        
        Args:
            plugin_name: 插件名称
            config: 插件配置
            
        Returns:
            DataSourcePlugin: 插件实例，或None
        """
        if plugin_name not in self.plugin_classes:
            self.logger.error(f"未找到数据源插件: {plugin_name}")
            return None
        
        try:
            plugin_class = self.plugin_classes[plugin_name]
            plugin = plugin_class(config)
            
            # 初始化插件
            if plugin.initialize():
                self.plugins[plugin_name] = plugin
                self.logger.info(f"创建并初始化数据源插件: {plugin_name} 成功")
                return plugin
            else:
                self.logger.warning(f"数据源插件: {plugin_name} 初始化失败")
                return None
        except Exception as e:
            self.logger.error(f"创建数据源插件失败: {str(e)}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return None
    
    def get_plugin(self, plugin_name: str) -> Optional[DataSourcePlugin]:
        """获取插件实例
        
        Args:
            plugin_name: 插件名称
            
        Returns:
            DataSourcePlugin: 插件实例，或None
        """
        return self.plugins.get(plugin_name)
    
    def get_all_plugins(self) -> Dict[str, DataSourcePlugin]:
        """获取所有已创建的插件
        
        Returns:
            Dict: 所有插件
        """
        return self.plugins
    
    def get_available_plugins(self) -> Dict[str, DataSourcePlugin]:
        """获取所有可用的插件
        
        Returns:
            Dict: 可用插件
        """
        return {name: plugin for name, plugin in self.plugins.items() if plugin.is_available()}
    
    def get_best_plugin(self) -> Optional[DataSourcePlugin]:
        """获取最佳可用插件（优先级最高）
        
        Returns:
            DataSourcePlugin: 最佳插件，或None
        """
        available_plugins = self.get_available_plugins()
        if not available_plugins:
            return None
        
        # 按优先级排序
        sorted_plugins = sorted(available_plugins.values(), key=lambda p: p.priority, reverse=True)
        return sorted_plugins[0] if sorted_plugins else None
    
    def discover_plugins(self, plugins_dir: str = None) -> int:
        """发现并加载插件
        
        Args:
            plugins_dir: 插件目录，默认为当前目录下的plugins
            
        Returns:
            int: 发现的插件数量
        """
        if plugins_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            plugins_dir = os.path.join(base_dir, "plugins")
        
        if not os.path.exists(plugins_dir):
            self.logger.warning(f"插件目录不存在: {plugins_dir}")
            return 0
        
        count = 0
        for file in os.listdir(plugins_dir):
            if file.endswith(".py") and not file.startswith("__"):
                module_name = file[:-3]
                try:
                    spec = importlib.util.spec_from_file_location(
                        f"quantum_symbiotic_network.plugins.{module_name}",
                        os.path.join(plugins_dir, file)
                    )
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        # 查找模块中的DataSourcePlugin子类
                        for attr_name in dir(module):
                            attr = getattr(module, attr_name)
                            if (isinstance(attr, type) and 
                                issubclass(attr, DataSourcePlugin) and 
                                attr is not DataSourcePlugin):
                                self.register_plugin(attr)
                                count += 1
                except Exception as e:
                    self.logger.error(f"加载插件模块失败: {module_name}, 错误: {str(e)}")
        
        self.logger.info(f"共发现 {count} 个插件")
        return count


# 全局注册表实例
_registry = DataSourceRegistry()

def get_registry() -> DataSourceRegistry:
    """获取数据源注册表
    
    Returns:
        DataSourceRegistry: 数据源注册表
    """
    global _registry
    return _registry

def register_plugin(plugin_class: Type[DataSourcePlugin], name: str = None) -> None:
    """注册数据源插件
    
    Args:
        plugin_class: 插件类
        name: 插件名称
    """
    registry = get_registry()
    registry.register_plugin(plugin_class, name)

def get_plugin(name: str) -> Optional[DataSourcePlugin]:
    """获取插件实例
    
    Args:
        name: 插件名称
        
    Returns:
        DataSourcePlugin: 插件实例，或None
    """
    registry = get_registry()
    return registry.get_plugin(name)

def discover_plugins() -> int:
    """发现并加载插件
    
    Returns:
        int: 发现的插件数量
    """
    registry = get_registry()
    return registry.discover_plugins()

def get_best_data_source() -> Optional[DataSourcePlugin]:
    """获取最佳可用数据源
    
    Returns:
        DataSourcePlugin: 最佳数据源，或None
    """
    registry = get_registry()
    return registry.get_best_plugin() 