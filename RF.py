'''
Author: hiddenSharp429 z404878860@163.com
Date: 2024-06-21 21:51:41
LastEditors: hiddenSharp429 z404878860@163.com
LastEditTime: 2024-10-17 22:29:51
FilePath: /JUPYTER/RF.py
Description: 随机森林进行特征选取
'''
import numpy as np
import time
import pandas as pd
import random

from sklearn.model_selection import  RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from utils.data_process import data_process
from utils.get_fault_description import get_fault_description
from utils.balance_subset import balance_subset

def select_important_feature(train_data, test_data, fault_code, fault_description, model_exist=False, need_select=False, need_temporal_features=False):
    """
    使用随机森林进行特征选取
    :param train_data: 训练数据
    :param test_data: 测试数据
    :param fault_code: 故障代码
    :param fault_description: 故障描述
    :param model_exist: 是否存在预训练模型
    :param need_select: 是否需要选择特征
    :param need_temporal_features: 是否需要时序化数据
    """
    print("start select_important_feature......")


    y_train = (train_data[f'{fault_description}'] == fault_code)
    y_test = (test_data[f'{fault_description}'] == fault_code)

    if model_exist:
        need_temporal_features = True
        need_select = True

    # 数据处理
    X_train = data_process(train_data, need_temporal_features=need_temporal_features)
    X_test = data_process(test_data, need_temporal_features=need_temporal_features)

    X_train_original = X_train.copy()
    y_train_original = y_train.copy()
    X_test_original = X_test.copy()
    y_test_original = y_test.copy()

    # 如果不需要选择特征，直接返回原始数据
    if not need_select:
        return X_train_original, X_test_original, y_train_original, y_test_original

    if model_exist:
        print("model exist! loading selected features......")
        # 加载预训练模型所选的特征
        selected_features = load_important_feature(fault_code)
        X_train_selected = X_train_original[selected_features]
        X_test_selected = X_test_original[selected_features]

    
    if not model_exist:
        print("model not exist! using RF to select important features......")
        # 使用随机森林选择重要特征
        X_train_selected, X_test_selected = use_RF(X_train_original, X_test_original, y_train_original, y_test_original, fault_code)


    print("X_train_selected.shape:", X_train_selected.shape)
    print("X_test_selected.shape:", X_test_selected.shape)

    return X_train_selected, X_test_selected, y_train_original, y_test_original

def use_RF(X_train_original, X_test_original, y_train_original, y_test_original, fault_code):
    """
    使用随机森林选择重要特征
    :param X_train_original: 训练数据
    :param X_test_original: 测试数据
    :param y_train_original: 训练标签
    :param y_test_original: 测试标签
    :param fault_code: 故障代码

    :return: X_train_selected, X_test_selected
    """
    # 输入是否需要进行连续均衡切片采样
    is_balance = input("Do you want to balance the ratio of positive and negative samples ? (yes/no): ").lower().strip() == 'yes'

    rate = 4
    if is_balance:
        rate = int(input("Please input the ratio of positive and negative samples: "))

    # 使用 balance_subset 函数来均衡正负样本数量
    X_train, X_test, y_train, y_test= balance_subset(X_train_original, X_test_original, y_train_original, y_test_original, rate=rate, is_balance=is_balance)

    # 确保特征和标签的形状正确
    print("after balance X_train.shape:", X_train.shape)
    print("after balance X_test.shape:", X_test.shape)

    initial_param_space = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    param_space = initial_param_space.copy()

    precision_threshold = 0.1
    max_iterations = 10
    reset_interval = 3

    best_precision = 0
    best_params = None

    # 定义评分函数
    scoring = {'precision': 'precision', 'auc': 'roc_auc'}

    start_time = time.time()

    for iteration in range(max_iterations):
        if iteration % reset_interval == 0 and iteration > 0:
            param_space = initial_param_space.copy()
            print("重置参数空间")

        random_search = RandomizedSearchCV(
            estimator=RandomForestClassifier(random_state=42), 
            param_distributions=param_space, 
            n_iter=1, scoring=scoring, cv=2, verbose=2, n_jobs=-1, refit='precision'
        )

        random_search.fit(X_train, y_train)

        print(f"迭代 {iteration + 1} - 最佳参数: ", random_search.best_params_)

        y_pred = random_search.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=1)
        print(f"迭代 {iteration + 1} - 精确度:", precision)

        if precision > best_precision:
            best_precision = precision
            best_params = random_search.best_params_

        if precision >= precision_threshold:
            break

        param_space = update_param_space(param_space, random_search.best_params_)

    end_time = time.time()
    print(f"随机森林模型训练时间: {end_time - start_time:.2f} 秒")

    best_model = RandomForestClassifier(**best_params, random_state=42)
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=1)
    recall = recall_score(y_test, y_pred, zero_division=1)
    f1 = f1_score(y_test, y_pred, zero_division=1)
    report = classification_report(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Classification Report:\n", report)

    # 获取最佳模型
    best_model = random_search.best_estimator_

    # 获取特征重要性并排序
    feature_importances = best_model.feature_importances_

    # 计算总贡献度
    total_importance = sum(feature_importances)
    # 设置阈值
    threshold = 0.90

    # 特征贡献度排序
    sorted_indices = np.argsort(feature_importances)[::-1]

    # 使用贪心算法选择前threshold的特征
    accumulated_importance = 0
    selected_feature_indices = []
    for index in sorted_indices:
        accumulated_importance += feature_importances[index]
        selected_feature_indices.append(index)
        if accumulated_importance >= total_importance * threshold:
            break

    # 打印所选特征及其贡献度
    print("选取的特征及其贡献度：")
    for index in selected_feature_indices:
        print(f"特征 {index} ({X_train.columns[index]}): 贡献度 = {feature_importances[index]:.4f}")

    # 获取所选特征的名称
    selected_features = X_train.columns[selected_feature_indices]
    print("selected_features:", selected_features)

    # 保存所选特征
    save_important_feature(selected_features, fault_code)

    # 使用选定的特征进行 XGBoost 训练
    X_train_selected = X_train_original[selected_features]
    X_test_selected = X_test_original[selected_features]

    return X_train_selected, X_test_selected

def update_param_space(param_space, best_params):
    new_param_space = {}
    for param, values in param_space.items():
        if param in best_params:
            best_value = best_params[param]
            if isinstance(values, list) and best_value in values:
                index = values.index(best_value)
                if index == 0:
                    new_values = [values[0], values[1]]
                elif index == len(values) - 1:
                    new_values = [values[-2], values[-1]]
                else:
                    new_values = [values[index-1], best_value, values[index+1]]
            else:
                new_values = values

            if isinstance(values[0], int):
                new_values = [int(v) for v in new_values if isinstance(v, (int, float))]
            
            if random.random() < 0.2:
                if isinstance(values[0], int):
                    new_values.append(random.randint(min(values), max(values)))
                elif isinstance(values[0], float):
                    new_values.append(random.uniform(min(values), max(values)))
        else:
            new_values = values
        
        new_param_space[param] = new_values
    
    return new_param_space

def save_important_feature(selected_features, fault_code):
    """
    保存所选特征
    :param selected_features: 所选特征
    :param fault_code: 故障代码
    """
    # 将所选特征从Index转换为dataframe
    selected_features_pd = pd.DataFrame(selected_features, columns=['feature_name'])

    # 保存所选特征的名称到CSV文件
    selected_features_pd.to_csv(f'./feature/{fault_code}_selected_features.csv', index=False)

def load_important_feature(fault_code):
    """
    加载所选特征
    :param fault_code: 故障代码
    :return: 所选特征
    """
    # 加载所选特征的名称
    selected_features = pd.read_csv(f'./feature/{fault_code}_selected_features.csv')
    # 将所选特征转换为list
    selected_features = selected_features['feature_name'].tolist()
    return selected_features


if __name__ == "__main__":
    result = load_important_feature(4003)
    # print(result)