'''
Author: hiddenSharp429 z404878860@163.com
Date: 2024-07-04 16:59:16
LastEditors: hiddenSharp429 z404878860@163.com
LastEditTime: 2024-07-04 17:00:09
FilePath: /Pipeline-Failure-Prediction/utils/split_train_test_datasets.py
Description: 划分训练集和测试集
'''
import pandas as pd
from sklearn.model_selection import train_test_split

def split_train_test_datasets(data):
    """
    划分训练集和测试集
    
    参数：
    data: DataFrame, 包含特征和标签的数据集

    返回值：
    train_data: DataFrame, 训练集
    test_data: DataFrame, 测试集
    """

    # 获取所有日期
    all_dates = data['日期'].unique()

    # 划分训练集和测试集的日期
    train_dates, test_dates = train_test_split(
        all_dates, test_size=0.2)

    # 根据训练集和测试集的日期划分数据集
    train_data = data[data['日期'].isin(train_dates)]
    test_data = data[data['日期'].isin(test_dates)]

    print("train_data.shape:", train_data.shape)
    print("test_data.shape:", test_data.shape)

    return train_data, test_data