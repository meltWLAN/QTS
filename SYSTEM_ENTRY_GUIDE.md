# 量子交易系统 - 统一启动入口指南

## 关于统一启动入口

为了提高系统的稳定性、可维护性和安全性，量子交易系统现在采用**统一启动入口**机制。这意味着系统只能通过官方指定的启动脚本启动，而不能通过其他方式直接启动系统组件。

## 推荐的启动方式

从现在起，请使用以下命令启动量子交易系统：

```bash
python start_quantum.py [参数]
```

## 为什么使用统一启动入口？

1. **系统稳定性** - 确保所有系统组件按照正确的顺序和配置启动
2. **防止重复启动** - 避免多个系统实例同时运行造成的资源冲突
3. **配置一致性** - 确保使用统一的配置设置启动系统
4. **简化故障排除** - 集中管理启动日志，方便排查问题
5. **系统完整性验证** - 在启动前验证关键系统文件的完整性
6. **启动安全性** - 防止未授权的系统启动和配置修改

## 系统启动参数

统一启动入口支持以下命令和参数：

```bash
# 基本启动
python start_quantum.py

# 指定启动模式
python start_quantum.py --mode desktop
python start_quantum.py --mode console
python start_quantum.py --mode server

# 启用调试模式
python start_quantum.py --debug

# 使用特定配置文件
python start_quantum.py --config config/custom_config.json

# 查看系统状态
python start_quantum.py status

# 停止系统
python start_quantum.py stop
```

所有这些参数都会被传递给官方启动器 `launch_quantum_core.py`，保持与原有功能的兼容性。

## 常见问题

### 如何检查系统是否正在运行？

```bash
python start_quantum.py status
```

### 如何安全地停止系统？

```bash
python start_quantum.py stop
```

### 为什么不能直接使用 launch_quantum_core.py？

直接使用 `launch_quantum_core.py` 会绕过系统完整性检查、运行状态检查和锁机制，可能导致多个实例同时运行或系统启动不完整。

### 如果遇到"系统已经在运行中"的提示，但系统实际没有运行怎么办？

这可能是由于之前的系统实例异常退出导致的。可以尝试：

```bash
python start_quantum.py stop
python start_quantum.py start
```

### 统一入口如何与现有脚本兼容？

所有命令行参数都会被传递给底层的 `launch_quantum_core.py`，因此与现有功能完全兼容。

## 开发者注意事项

如果你是系统开发者：

1. 不要创建新的启动入口点，始终通过扩展 `launch_quantum_core.py` 添加新功能
2. 不要修改 `quantum_system_entry.py` 中的锁机制和完整性验证逻辑
3. 在创建新组件时，确保它们能够被系统管理器正确初始化和关闭
4. 添加新的关键系统文件时，请更新 `quantum_system_entry.py` 中的完整性检查列表

## 技术支持

如有任何关于系统启动的问题，请联系技术支持团队或查阅详细的系统文档。 