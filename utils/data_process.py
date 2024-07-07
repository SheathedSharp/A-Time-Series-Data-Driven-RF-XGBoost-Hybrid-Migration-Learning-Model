'''
Author: hiddenSharp429 z404878860@163.com
Date: 2024-07-04 16:48:00
LastEditors: hiddenSharp429 z404878860@163.com
LastEditTime: 2024-07-06 21:08:58
FilePath: /Pipeline-Failure-Prediction/utils/data_process.py
Description: 对数据进行处理，添加差值特征列和滞后特征列
'''

import pandas as pd
import numpy as np

def remove_irrelevant_features(data):
    """
    移除无关特征列

    参数：
    data: DataFrame, 包含特征和标签的数据集

    返回值：
    data: DataFrame, 移除无关特征列后的数据集
    """
    exclude_columns = ['日期', '时间', '生产线编号', '物料推送装置故障1001',
                       '物料检测装置故障2001', '填装装置检测故障4001', '填装装置定位故障4002', '填装装置填装故障4003',
                       '加盖装置定位故障5001', '加盖装置加盖故障5002', '拧盖装置定位故障6001', '拧盖装置拧盖故障6002']
    feature_columns = [col for col in data.columns if col not in exclude_columns]
    return data[feature_columns], feature_columns

def add_difference_and_lag_features(data, feature_columns):
    """
    添加差值和滞后特征列
    
    参数：
    data: DataFrame, 包含特征和标签的数据集
    feature_columns: list, 特征列名列表

    返回值：
    DataFrame
    """
    new_features_dict = {}
    for i in range(len(feature_columns)):
        for j in range(i + 1, len(feature_columns)):
            col1, col2 = feature_columns[i], feature_columns[j]
            diff_col_name = f'{col1}_{col2}_diff'
            new_features_dict[diff_col_name] = data[col1] - data[col2]
            
    # 计算滞后特征    
    for col in feature_columns:
        lag_col_name = f'{col}_lag'
        values = data[col].values
        lag_values = np.zeros(len(values), dtype=int)
        lag_duration = 0

        for idx in range(1, len(values)):
            if values[idx] == values[idx - 1]:
                lag_duration += 1
            else:
                lag_duration = 0
            lag_values[idx] = lag_duration

        new_features_dict[lag_col_name] = lag_values


    return pd.DataFrame(new_features_dict)

def data_process(data):
    """
    对数据进行处理

    参数：
    data: DataFrame, 包含特征和标签的数据集

    返回值：
    DataFrame
    """
    data, feature_columns = remove_irrelevant_features(data)
    difference_and_lag_features = add_difference_and_lag_features(data, feature_columns)
    data = pd.concat([data, difference_and_lag_features], axis=1)
    print("after data process data.shape is:", data.shape)
    return data