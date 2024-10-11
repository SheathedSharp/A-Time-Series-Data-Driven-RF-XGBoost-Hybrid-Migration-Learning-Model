'''
Author: hiddenSharp429 z404878860@163.com
Date: 2024-07-05 11:41:16
LastEditors: hiddenSharp429 z404878860@163.com
LastEditTime: 2024-10-10 20:58:03
FilePath: /Application of Time Series-Driven XGBoost Model in Pipeline Fault Prediction/XGBoost.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import time
import joblib
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score

from RF import select_important_feature
from utils.load_data import get_input_and_prepare_data
from utils.load_pre_trained_model import load_pre_trained_model
from utils.model_evaluation import evaluate_model

def pipeline_failure_prediction():
    pipeline_code, fault_code, fault_description, train_data, test_data = get_input_and_prepare_data()
    
    # 创建结果保存的文件夹
    model_output_folder = 'model/'
    report_output_folder = 'report/'
    os.makedirs(model_output_folder, exist_ok=True)

    # 设置模型名称
    model_name = f'xgboost{fault_code}_{pipeline_code}'

    # 输入是否需要加载预训练模型
    need_load = input("Do you want to load a pre-trained model? (yes/no): ").lower().strip() == 'yes'
    
    # 加载预训练模型
    model, model_exist = load_pre_trained_model(need_load=need_load)

    # 输入是否需要时序化数据
    need_temporal_features = input("Do you want to load a temporal features? (yes/no): ").lower().strip() == 'yes'

    # 输入是否需要进行RF特征选择
    need_select = input("Do you want to use RF to select important features? (yes/no): ").lower().strip() == 'yes'

    X_train_selected, X_test_selected, y_train_original, y_test_original = select_important_feature(
        train_data, test_data, fault_code, fault_description, model_exist, need_select=need_select, need_temporal_features=need_temporal_features)
    
    # 记录开始时间
    start_time = time.time()

    # 没有预训练模型时，训练新模型
    if model is None:
        # 特征标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)

        initial_param_space = {
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200, 300, 400, 500],
            'subsample': [0.5, 0.7, 0.8, 1.0],
            'colsample_bytree': [0.5, 0.7, 0.8, 1.0],
            'max_depth': [5, 7, 10, 15],
        }

        # 定义评分函数
        scoring = {'precision': 'precision', 'auc': 'roc_auc'}
        
        if need_select:
            param_space = initial_param_space.copy()

            precision_threshold = 0.7
            max_iterations = 10  # 增加最大迭代次数
            reset_interval = 3   # 每3次迭代重置一次参数空间

            best_precision = 0
            best_params = None

            for iteration in range(max_iterations):
                if iteration % reset_interval == 0 and iteration > 0:
                    param_space = initial_param_space.copy()
                    print("重置参数空间")

                random_search = RandomizedSearchCV(
                    estimator=XGBClassifier(), param_distributions=param_space, 
                    n_iter=5, scoring=scoring, cv=5, verbose=2, n_jobs=-1, refit='precision'
                )

                random_search.fit(X_train_scaled, y_train_original)

                print(f"迭代 {iteration + 1} - 最佳参数: ", random_search.best_params_)

                # 在测试集上评估模型
                y_pred = random_search.predict(X_test_scaled)

                # 计算性能指标
                accuracy = accuracy_score(y_test_original, y_pred)
                precision = precision_score(y_test_original, y_pred)
                print(f"迭代 {iteration + 1} - 精确度:", precision)

                if precision > best_precision:
                    best_precision = precision
                    best_params = random_search.best_params_

                # 判断 precision 是否达到阈值
                if precision >= precision_threshold:
                    break

                # 更新参数空间
                param_space = update_param_space(param_space, random_search.best_params_)

        else:
            # 执行简单的随机参数搜索
            print("执行简单的随机参数搜索")
            random_search = RandomizedSearchCV(
                estimator=XGBClassifier(), param_distributions=initial_param_space, 
                n_iter=10, scoring=scoring, cv=3, verbose=2, n_jobs=-1, refit='precision'
            )

            random_search.fit(X_train_scaled, y_train_original)
            best_params = random_search.best_params_
            print("最佳参数: ", best_params)

        model = XGBClassifier(**best_params)
        model.fit(X_train_scaled, y_train_original)

    # 有预训练模型时，使用迁移学习方法来调整模型参数
    else:
        model = transfer_learning(model, X_train_selected, y_train_original)

    # 在测试集上评估模型
    y_pred = model.predict(X_test_scaled)
    y_scores = model.predict_proba(X_test_scaled)[:, 1]  # 获取预测概率

    print("Unique values in y_true:", np.unique(y_test_original, return_counts=True))
    print("y_scores statistics:")
    print("Min:", np.min(y_scores))
    print("Max:", np.max(y_scores))
    print("Mean:", np.mean(y_scores))
    print("Median:", np.median(y_scores))

    plt.figure(figsize=(10, 5))
    plt.hist(y_scores, bins=50)
    plt.title("Distribution of Prediction Probabilities")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.show()

    # 使用新的评估函数
    results = evaluate_model(y_test_original, y_pred, y_scores, model_name, report_output_folder)

    # 打印一些关键指标
    # print("Precision:", results["precision"])
    # print("AUC:", results["auc"])
    # print("KS Statistic:", results["ks_statistic"])


    # 记录结束时间
    end_time = time.time()

    # 打印模型训练时间
    print(f"XGBoost模型训练时间: {end_time - start_time:.2f} 秒")

    # 保存模型
    model_save_path = os.path.join(model_output_folder, f"{model_name}.model")
    joblib.dump(model, f'{model_save_path}.pkl')


    
def transfer_learning(model, target_X_train_selected, target_y_train_original):
    # 读取目标数据
    target_X_train_selected = target_X_train_selected
    target_y_train = target_y_train_original

    # 微调模型
    model.fit(target_X_train_selected, target_y_train, xgb_model=model)

    return model

def update_param_space(param_space, best_params):
    new_param_space = {}
    for param, values in param_space.items():
        best_value = best_params[param]
        index = values.index(best_value)
        
        if index == 0:
            new_values = [values[0], (values[0] + values[1]) / 2, values[1]]
        elif index == len(values) - 1:
            new_values = [values[-2], (values[-2] + values[-1]) / 2, values[-1]]
        else:
            new_values = [
                (values[index-1] + best_value) / 2,
                best_value,
                (best_value + values[index+1]) / 2
            ]
        
        if isinstance(values[0], int):
            new_values = [int(v) for v in new_values]
        
        # 添加随机性
        if random.random() < 0.2:  # 20%的概率
            if isinstance(values[0], int):
                new_values.append(random.randint(min(values), max(values)))
            else:
                new_values.append(random.uniform(min(values), max(values)))
        
        new_param_space[param] = new_values
    
    return new_param_space

if __name__ == "__main__":
    pipeline_failure_prediction()