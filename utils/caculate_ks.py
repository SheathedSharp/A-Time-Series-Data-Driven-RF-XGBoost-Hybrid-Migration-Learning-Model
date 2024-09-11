'''
Author: hiddenSharp429 z404878860@163.com
Date: 2024-09-11 16:35:13
LastEditors: hiddenSharp429 z404878860@163.com
LastEditTime: 2024-09-11 16:35:19
FilePath: /Application of Time Series-Driven XGBoost Model in Pipeline Fault Prediction/utils/caculate_ks.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import numpy as np
# 计算KS统计量
def calculate_ks(y_true, y_scores):
    # 计算正负样本的累积分布
    pos_cdf = np.cumsum(y_true) / np.sum(y_true)
    neg_cdf = np.cumsum(1 - y_true) / np.sum(1 - y_true)
    ks_statistic = np.max(pos_cdf - neg_cdf)
    return ks_statistic