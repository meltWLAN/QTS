# TuShare Pro API 使用指南

> **重要提示**: 超神系统现在统一使用`launch_supergod.py`作为唯一启动入口。
> 运行方式: `python launch_supergod.py [选项]`

## 重要更新：超神系统只能使用TuShare真实数据

超神系统已经更新为**只使用真实数据**，不再支持模拟数据。这意味着您必须提供有效的TuShare Pro API token才能使用系统。

## 获取TuShare Pro API Token

当前系统显示TuShare Pro API token无效或已过期。要使用超神系统，您需要获取有效的token：

1. 访问 [TuShare Pro官网](https://tushare.pro/)
2. 注册一个账号（如果已有账号则直接登录）
3. 登录后，在个人中心或API页面可以查看您的token
4. 复制该token，更新到系统中

## 更新TuShare Pro API Token

有两种方式可以更新token：

### 方法1：更新配置文件（推荐）

1. 打开项目根目录下的`config.json`文件
2. 找到`"tushare_token"`字段，替换为您的新token
3. 保存文件后重启应用

```json
{
  // ... 其他配置 ...
  "tushare_token": "您的新token",
  // ... 其他配置 ...
}
```

### 方法2：使用API更新

在应用运行时，您可以通过以下方式更新token：

```python
from gui.controllers.data_controller import DataController

controller = DataController()
controller.set_tushare_token("您的新token")
```

## 免费版与Pro版API区别

TuShare提供了两个版本的API：

1. **基础版API**：数据有限，更新不及时，即将停止服务
2. **Pro版API**：实时数据更新，接口稳定，功能更丰富

本系统使用Pro版API。您必须拥有有效的Pro版token才能使用超神系统。

## 常见问题

### 无法获取实时数据？

- 检查token是否正确
- 检查网络连接
- 查看日志中的具体错误信息

### token正确但仍无法使用？

某些Pro API功能可能需要积分或付费才能使用。请登录TuShare Pro官网查看您的账户权限和积分情况。

### 为什么不能使用模拟数据？

超神系统的量子意识共生核心需要真实数据进行深度分析和预测。模拟数据无法提供足够的市场信息熵和量子共振特征，因此系统已完全禁用模拟数据功能。

## 技术支持

如有其他问题，请联系技术支持或查阅[TuShare官方文档](https://tushare.pro/document/1)。 