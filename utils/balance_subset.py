'''
Author: hiddenSharp429 z404878860@163.com
Date: 2024-07-04 16:47:12
LastEditors: hiddenSharp429 z404878860@163.com
LastEditTime: 2024-07-05 11:09:53
FilePath: /Pipeline-Failure-Prediction/utils/balance_subset.py
Description: 正负样本均衡
'''
import pandas as pd

def balance_subset(X_train, X_test, y_train, y_test, rate=4, is_balance=True):
    if not is_balance:
        return X_train, X_test, y_train, y_test

    X_train['label'] = y_train
    X_test['label'] = y_test

    # 划分训练子集，确保正负样本相对均衡
    train_subsets = balance_slice(X_train, rate)
    test_subsets = balance_slice(X_test, rate)

    # 合并所有训练子集成一个大的训练集
    df_train_all = pd.concat(train_subsets)
    df_test_all = pd.concat(test_subsets)

    # 将特征和标签拆分回来
    X_train_after_balance = df_train_all.iloc[:, :-1]
    y_train_after_balance = df_train_all['label']
    X_test_after_balance = df_test_all.iloc[:, :-1]
    y_test_after_balance = df_test_all['label']

    return X_train_after_balance, X_test_after_balance, y_train_after_balance, y_test_after_balance


def balance_slice(df, rate=1.0):
    """
    划分训练子集，确保正负样本相对均衡

    参数：
    df: DataFrame, 训练集的特征和标签，包含多数False和少数True
    rate: float, 训练子集的正负样本比例，默认为1.0(负样本是正样本的rate倍)

    返回值：
    train_subsets: list of DataFrames, 均衡的训练子集列表，每个子集包含正负样本均衡的特征和标签
    """
    train_subsets = []

    while df['label'].sum() > 0:
        # 找到第一个连续为True的索引
        first_true_index = df[df['label']].index[0]

        # 找到连续 True 区间的结束索引
        last_true_index = first_true_index
        while last_true_index < df.index.max() and df['label'][last_true_index + 1]:
            last_true_index += 1

        # 获取连续 True 区间的长度
        true_length = last_true_index - first_true_index + 1

        # 形成训练子集并添加到列表中
        train_subset = df.loc[first_true_index -
                              rate*true_length:last_true_index].copy()
        train_subsets.append(train_subset)

        # 更新 DataFrame，去除已经形成的训练子集
        df = df.drop(train_subset.index)

    return train_subsets