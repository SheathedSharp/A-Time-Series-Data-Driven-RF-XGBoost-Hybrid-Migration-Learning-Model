'''
Author: hiddenSharp429 z404878860@163.com
Date: 2024-07-04 16:47:37
LastEditors: hiddenSharp429 z404878860@163.com
LastEditTime: 2024-07-04 23:05:15
FilePath: /Pipeline-Failure-Prediction/load_data.py
Description: 读取数据
'''
import pandas as pd

# 读取数据并且获取测试集和训练集
def read_data(file_path):
    # 读取CSV文件
    data = pd.read_csv(file_path)

    # 处理缺失值（如果有）
    data.fillna(0, inplace=True)

    return data