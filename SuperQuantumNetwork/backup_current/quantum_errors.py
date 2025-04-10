"""
超神量子共生系统 - 错误类定义
"""

class QuantumError(Exception):
    """量子计算相关错误"""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

class ModuleError(Exception):
    """模块相关错误"""
    def __init__(self, message: str, module_name: str = None):
        super().__init__(message)
        self.message = message
        self.module_name = module_name

class DataError(Exception):
    """数据相关错误"""
    def __init__(self, message: str, data_id: str = None):
        super().__init__(message)
        self.message = message
        self.data_id = data_id

class SystemError(Exception):
    """系统级错误"""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message 