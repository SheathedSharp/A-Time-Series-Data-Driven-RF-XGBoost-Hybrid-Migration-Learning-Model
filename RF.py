'''
Author: hiddenSharp429 z404878860@163.com
Date: 2024-06-21 21:51:41
LastEditors: hiddenSharp429 z404878860@163.com
LastEditTime: 2024-09-11 17:14:07
FilePath: /JUPYTER/RF.py
Description: 随机森林进行特征选取
'''
import numpy as np
import time
import pandas as pd

from sklearn.model_selection import  RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from utils.data_process import data_process
from utils.get_fault_description import get_fault_description
from utils.balance_subset import balance_subset

def select_important_feature(train_data, test_data, fault_code, fault_description, model_exist=False, need_select=True):
    """
    使用随机森林进行特征选取
    :param train_data: 训练数据
    :param test_data: 测试数据
    :param fault_code: 故障代码
    :param fault_description: 故障描述
    :param model_exist: 是否存在预训练模型
    :param need_select: 是否需要选择特征
    """
    print("start select_important_feature......")


    y_train = (train_data[f'{fault_description}'] == fault_code)
    y_test = (test_data[f'{fault_description}'] == fault_code)

    X_train = data_process(train_data)
    X_test = data_process(test_data)

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
    # 使用 balance_subset 函数来均衡正负样本数量
    X_train, X_test, y_train, y_test= balance_subset(X_train_original, X_test_original, y_train_original, y_test_original, rate=4, is_balance=True)

    # 确保特征和标签的形状正确
    print("after balance X_train.shape:", X_train.shape)
    print("after balance X_test.shape:", X_test.shape)

    # 定义参数空间
    param_space = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [3, 5],
    }

    # 记录开始时间
    start_time = time.time()

    # 初始化随机森林模型
    rf_model = RandomForestClassifier()

    # 定义评分函数
    scoring = {'precision': 'precision', 'auc': 'roc_auc'}

    # 使用随机参数搜索
    random_search = RandomizedSearchCV(
        estimator=rf_model, param_distributions=param_space, n_iter=10, scoring=scoring, cv=4, verbose=2, n_jobs=-1, refit='precision'
    )

    # 拟合模型
    random_search.fit(X_train, y_train)

    # 输出最佳参数
    print("Best parameters found: ", random_search.best_params_)

    # 在测试集上评估模型
    y_pred = random_search.predict(X_test)

    # 记录结束时间
    end_time = time.time()

    # 打印模型训练时间
    print(f"随机森林模型训练时间: {end_time - start_time:.2f} 秒")

    # 打印其他性能指标
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

def save_important_feature(selected_features, fault_code):
    """
    保存所选特征
    :param selected_features: 所选特征
    :param fault_code: 故障代码
    """
    # 将所选特征从Index转换为dataframe
    selected_features_pd = pd.DataFrame(selected_features, columns=['feature_name'])

    # 保存所选特征的名称到CSV文件
    selected_features_pd.to_csv(f'./model/{fault_code}_selected_features.csv', index=False)

def load_important_feature(fault_code):
    """
    加载所选特征
    :param fault_code: 故障代码
    :return: 所选特征
    """
    # 加载所选特征的名称
    selected_features = pd.read_csv(f'./model/{fault_code}_selected_features.csv')

    # 将所选特征转换为list
    selected_features = selected_features['feature_name'].tolist()
    return selected_features


if __name__ == "__main__":
    result = load_important_feature(4003)
    # print(result)
