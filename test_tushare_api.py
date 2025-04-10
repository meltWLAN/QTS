#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试Tushare API获取市场指数数据
"""

import tushare as ts
import pandas as pd
import traceback

def main():
    # 使用测试脚本中的token
    token = "0e65a5c636112dc9d9af5ccc93ef06c55987805b9467db0866185a10"
    ts.set_token(token)
    pro = ts.pro_api()
    
    # 获取账户信息
    print("="*50)
    print("获取Tushare账户信息:")
    try:
        info = pro.query('token')
        print(info)
        print(f"当前账户积分: {info.iloc[0]['remain_point'] if 'remain_point' in info.columns else '未知'}")
    except Exception as e:
        print(f"获取账户信息出错: {e}")
        traceback.print_exc()
    
    print("\n" + "="*50)
    print("尝试获取上证指数数据:")
    try:
        df = pro.index_daily(ts_code='000001.SH', start_date='20230101', end_date='20230105')
        if df is not None and not df.empty:
            print(f"成功获取上证指数数据，数据条数: {len(df)}")
            print(df.head())
        else:
            print("获取指数数据失败，返回为空")
    except Exception as e:
        print(f"获取指数数据出错: {e}")
        traceback.print_exc()
    
    print("\n" + "="*50)
    print("尝试获取深证成指数据:")
    try:
        df = pro.index_daily(ts_code='399001.SZ', start_date='20230101', end_date='20230105')
        if df is not None and not df.empty:
            print(f"成功获取深证成指数据，数据条数: {len(df)}")
            print(df.head())
        else:
            print("获取指数数据失败，返回为空")
    except Exception as e:
        print(f"获取指数数据出错: {e}")
        traceback.print_exc()
    
    print("\n" + "="*50)
    print("尝试获取沪深300指数据:")
    try:
        df = pro.index_daily(ts_code='000300.SH', start_date='20230101', end_date='20230105')
        if df is not None and not df.empty:
            print(f"成功获取沪深300指数据，数据条数: {len(df)}")
            print(df.head())
        else:
            print("获取指数数据失败，返回为空")
    except Exception as e:
        print(f"获取指数数据出错: {e}")
        traceback.print_exc()
    
    # 测试是否可以获取股票数据作为对比
    print("\n" + "="*50)
    print("尝试获取股票数据(作为对比测试):")
    try:
        df = pro.daily(ts_code='000001.SZ', start_date='20230101', end_date='20230105')
        if df is not None and not df.empty:
            print(f"成功获取股票数据，数据条数: {len(df)}")
            print(df.head())
        else:
            print("获取股票数据失败，返回为空")
    except Exception as e:
        print(f"获取股票数据出错: {e}")
        traceback.print_exc()
    
if __name__ == "__main__":
    main() 