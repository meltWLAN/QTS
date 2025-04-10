#!/usr/bin/env python3
"""
超神量子共生网络交易系统 - 依赖注入容器
管理系统组件依赖，提高代码模块化和可测试性
"""

import logging
import inspect
from typing import Dict, Any, Callable, Optional, List, Type

logger = logging.getLogger("DependencyContainer")

class DependencyContainer:
    """依赖注入容器，管理系统各组件的依赖关系"""
    
    def __init__(self):
        """初始化依赖注入容器"""
        self._services = {}  # 保存注册的服务
        self._factories = {}  # 保存服务工厂函数
        self._singletons = {}  # 保存单例服务实例
        self._config = {}  # 配置参数
        self.logger = logging.getLogger("DependencyContainer")
        self.logger.info("依赖注入容器已初始化")
    
    def register(self, service_name: str, implementation: Any) -> None:
        """注册服务
        
        Args:
            service_name: 服务名称
            implementation: 服务实现类或对象
        """
        self._services[service_name] = implementation
        self.logger.debug(f"已注册服务: {service_name}")
    
    def register_factory(self, service_name: str, factory: Callable[..., Any]) -> None:
        """注册服务工厂函数
        
        Args:
            service_name: 服务名称
            factory: 服务工厂函数，将用于创建服务实例
        """
        self._factories[service_name] = factory
        self.logger.debug(f"已注册服务工厂: {service_name}")
    
    def register_singleton(self, service_name: str, implementation: Any = None, factory: Callable[..., Any] = None) -> None:
        """注册单例服务
        
        Args:
            service_name: 服务名称
            implementation: 服务实现类
            factory: 可选的工厂函数，如果提供将用于创建单例
        """
        if factory:
            self._factories[service_name] = factory
            self._singletons[service_name] = None  # 标记为懒加载单例
        else:
            self._singletons[service_name] = implementation
        
        self.logger.debug(f"已注册单例服务: {service_name}")
    
    def configure(self, config: Dict[str, Any]) -> None:
        """配置容器
        
        Args:
            config: 配置字典
        """
        self._config.update(config)
        self.logger.debug(f"已更新容器配置: {len(config)} 个项目")
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """获取配置值
        
        Args:
            key: 配置键
            default: 默认值，如果键不存在则返回
            
        Returns:
            配置值
        """
        return self._config.get(key, default)
    
    def resolve(self, service_name: str) -> Any:
        """解析服务，获取服务实例
        
        Args:
            service_name: 服务名称
            
        Returns:
            服务实例
            
        Raises:
            KeyError: 如果服务未注册
        """
        # 检查是否是单例，且已创建
        if service_name in self._singletons:
            singleton = self._singletons[service_name]
            if singleton is not None:
                return singleton
            
            # 如果是懒加载单例，则创建实例
            if service_name in self._factories:
                singleton = self._create_instance_from_factory(service_name)
                self._singletons[service_name] = singleton
                return singleton
        
        # 检查是否有注册的服务实现
        if service_name in self._services:
            implementation = self._services[service_name]
            
            # 如果是类（需要实例化）
            if inspect.isclass(implementation):
                return self._create_instance(implementation)
            
            # 如果是已经实例化的对象
            return implementation
        
        # 检查是否有注册的工厂函数
        if service_name in self._factories:
            return self._create_instance_from_factory(service_name)
        
        raise KeyError(f"服务未注册: {service_name}")
    
    def _create_instance_from_factory(self, service_name: str) -> Any:
        """从工厂函数创建实例
        
        Args:
            service_name: 服务名称
            
        Returns:
            服务实例
        """
        factory = self._factories[service_name]
        
        # 检查工厂函数的参数
        sig = inspect.signature(factory)
        params = {}
        
        for param_name, param in sig.parameters.items():
            if param_name == 'config':
                params[param_name] = self._config
            elif param_name == 'container':
                params[param_name] = self
            else:
                # 尝试解析依赖参数
                try:
                    params[param_name] = self.resolve(param_name)
                except KeyError:
                    # 如果有默认值，使用默认值
                    if param.default is not inspect.Parameter.empty:
                        params[param_name] = param.default
                    else:
                        self.logger.warning(f"无法解析工厂函数 {service_name} 的参数: {param_name}")
        
        try:
            return factory(**params)
        except Exception as e:
            self.logger.error(f"从工厂创建 {service_name} 实例失败: {str(e)}")
            raise
    
    def _create_instance(self, implementation: Type) -> Any:
        """创建类实例，自动解析构造函数依赖
        
        Args:
            implementation: 实现类
            
        Returns:
            类实例
        """
        # 检查构造函数参数
        sig = inspect.signature(implementation.__init__)
        params = {}
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            
            if param_name == 'config':
                params[param_name] = self._config
            elif param_name == 'container':
                params[param_name] = self
            else:
                # 尝试解析依赖参数
                try:
                    params[param_name] = self.resolve(param_name)
                except KeyError:
                    # 如果有默认值，使用默认值
                    if param.default is not inspect.Parameter.empty:
                        params[param_name] = param.default
                    else:
                        self.logger.warning(f"无法解析 {implementation.__name__} 的构造参数: {param_name}")
        
        try:
            return implementation(**params)
        except Exception as e:
            self.logger.error(f"创建 {implementation.__name__} 实例失败: {str(e)}")
            raise
    
    def resolve_all(self, base_class: Type) -> List[Any]:
        """解析指定基类的所有服务实例
        
        Args:
            base_class: 基类或接口类
            
        Returns:
            服务实例列表
        """
        instances = []
        
        # 查找所有继承自base_class的服务
        for service_name, implementation in self._services.items():
            try:
                if inspect.isclass(implementation) and issubclass(implementation, base_class):
                    instances.append(self.resolve(service_name))
                elif isinstance(implementation, base_class):
                    instances.append(implementation)
            except (TypeError, Exception) as e:
                self.logger.debug(f"检查 {service_name} 是否为 {base_class.__name__} 类型时出错: {str(e)}")
                continue
        
        return instances

# 全局容器实例
_container = DependencyContainer()

def get_container() -> DependencyContainer:
    """获取全局依赖注入容器
    
    Returns:
        DependencyContainer: 依赖注入容器实例
    """
    global _container
    return _container

def register(service_name: str, implementation: Any) -> None:
    """注册服务到全局容器
    
    Args:
        service_name: 服务名称
        implementation: 服务实现
    """
    container = get_container()
    container.register(service_name, implementation)

def register_singleton(service_name: str, implementation: Any = None, factory: Callable[..., Any] = None) -> None:
    """注册单例服务到全局容器
    
    Args:
        service_name: 服务名称
        implementation: 服务实现
        factory: 可选的工厂函数
    """
    container = get_container()
    container.register_singleton(service_name, implementation, factory)

def resolve(service_name: str) -> Any:
    """从全局容器解析服务
    
    Args:
        service_name: 服务名称
        
    Returns:
        服务实例
    """
    container = get_container()
    return container.resolve(service_name)

def configure(config: Dict[str, Any]) -> None:
    """配置全局容器
    
    Args:
        config: 配置字典
    """
    container = get_container()
    container.configure(config) 