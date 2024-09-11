'''
Author: hiddenSharp429 z404878860@163.com
Date: 2024-09-11 16:32:36
LastEditors: hiddenSharp429 z404878860@163.com
LastEditTime: 2024-09-11 16:32:47
FilePath: /Application of Time Series-Driven XGBoost Model in Pipeline Fault Prediction/utils/count_zero_nonzero_attributes.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import pandas as pd


def count_zero_nonzero_attributes(file_path):
    """
    读取M101的数据，并统计后九个属性列中每个属性列为0和非零的数据

    参数：
    file_path: str, 数据文件的路径

    返回值：
    result: dict, 每个属性列为0和非零的计数
    """
    df = pd.read_csv(file_path)  # 读取数据
    result = {}

    # 统计后九个属性列
    for column in df.columns[-9:]:
        zero_count = (df[column] == 0).sum()
        nonzero_count = (df[column] != 0).sum()
        result[column] = {'zero': zero_count, 'nonzero': nonzero_count}

    return result



result = count_zero_nonzero_attributes('data/M102.csv')
print(result)