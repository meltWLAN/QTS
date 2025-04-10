def main():
    """主函数"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="超神量子共生系统全面验证测试工具")
    parser.add_argument("--level", type=int, default=3, choices=range(1, 6),
                        help="测试级别 (1-5): 1=基础, 2=集成, 3=系统, 4=性能, 5=量子增强")
    parser.add_argument("--no-ui", action="store_true", 
                        help="跳过UI组件测试")
    parser.add_argument("--auto-fix", action="store_true",
                        help="自动修复发现的问题")
    parser.add_argument("--detailed", action="store_true",
                        help="输出详细测试信息")
    
    args = parser.parse_args()
    
    print("DEBUG: 开始执行验证...")
    print(f"DEBUG: 参数: {vars(args)}")
    
    try:
        # 创建验证工具实例
        validator = SupergodSystemValidator(vars(args))
        
        # 执行验证测试
        print("DEBUG: 运行测试...")
        result = validator.run_all_tests()
        
        print(f"DEBUG: 测试完成, 结果: {'通过' if result else '失败'}")
        
        # 返回测试结果
        return 0 if result else 1
    except Exception as e:
        print(f"验证器执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1 