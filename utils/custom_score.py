'''
Author: hiddenSharp429 z404878860@163.com
Date: 2024-07-04 16:47:26
LastEditors: hiddenSharp429 z404878860@163.com
LastEditTime: 2024-07-04 17:09:10
FilePath: /Pipeline-Failure-Prediction/utils/custom_score.py
Description: 自定义评分函数
'''
import numpy as np


# 定义评分函数
def custom_score(y_true, y_pred):
    """
    自定义评分函数

    参数：
    y_true: np.ndarray, 真实标签
    y_pred: np.ndarray, 预测标签

    返回值：
    score: float, 评分
    """
    # 统计 True 样本的数量
    true_count = np.sum(y_true)

    # 计算预测为 True 的数量
    pred_true_count = np.sum(y_pred)

    # 如果没有预测到 True 样本，则返回 0
    if pred_true_count == 0:
        return 0

    # 计算预测正确的 True 样本数量
    correct_true_count = np.sum(y_true & y_pred)

    # 计算 Precision，并引入奖励和惩罚
    reward = 1.2  # 奖励倍数
    penalty = 0.8  # 惩罚倍数
    precision = correct_true_count / pred_true_count
    precision = precision * reward if precision > 0.5 else precision * penalty
    return precision