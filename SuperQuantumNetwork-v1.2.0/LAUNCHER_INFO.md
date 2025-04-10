# 超神系统统一启动入口说明

## 关于启动入口的统一

为了提高系统稳定性、减少混淆并确保所有功能保持一致性，超神系统现在只使用**单一官方启动入口**：

```bash
python launch_supergod.py
```

所有其他启动脚本（如`run.py`、`main.py`、`run_enhanced_system.py`等）已被废弃，并修改为重定向脚本。

## 为什么使用单一启动入口？

1. **系统一致性** - 确保所有用户体验到相同的超神系统功能和行为
2. **简化维护** - 只需维护一个启动脚本，减少错误和不一致
3. **功能整合** - 所有系统功能（包括高级功能）都通过参数传递给单一启动脚本
4. **依赖管理** - 更好地管理系统依赖和初始化顺序
5. **数据整合** - 确保整个系统使用相同的数据源和处理机制

## 启动参数

官方启动脚本支持多种参数，可根据需要进行组合：

```bash
# 基本启动
python launch_supergod.py

# 调试模式
python launch_supergod.py --debug

# 仅控制台模式（不启动GUI）
python launch_supergod.py --console-only

# 激活高维统一场（增强预测能力）
python launch_supergod.py --activate-field

# 提升意识水平（增强自适应能力）
python launch_supergod.py --consciousness-boost

# 查看所有可用选项
python launch_supergod.py --help
```

## 处理旧启动脚本

如果您使用过旧的启动脚本，可以使用我们提供的清理工具将它们备份并修改为重定向脚本：

```bash
python cleanup_old_launchers.py
```

此工具会：
1. 将旧脚本备份到`backup_launchers`目录
2. 将原脚本修改为简单的重定向脚本
3. 确保用户知道应该使用官方启动脚本

## 常见问题

### 旧脚本和新脚本有什么区别？

新的统一启动脚本`launch_supergod.py`整合了所有其他脚本的功能，并提供了更多高级参数，同时确保只使用真实市场数据。

### 如何获取之前脚本的功能？

所有功能已整合到`launch_supergod.py`中：
- `run_enhanced_system.py`的增强功能 → 使用`--activate-field --consciousness-boost`参数
- `main.py`的回测功能 → 直接在GUI中使用回测功能
- `gui_app.py`的GUI功能 → 默认包含在`launch_supergod.py`中

### 系统数据会不会丢失？

不会。统一启动入口使用相同的数据目录和配置文件，所有历史数据和设置都会保留。

## 技术支持

如有任何疑问，请参考系统文档或联系技术支持团队。 