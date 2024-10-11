'''
Author: hiddenSharp429 z404878860@163.com
Date: 2024-10-08 11:05:59
LastEditors: hiddenSharp429 z404878860@163.com
LastEditTime: 2024-10-08 20:34:25
FilePath: /Application of Time Series-Driven XGBoost Model in Pipeline Fault Prediction/MLP.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from utils.data_process import data_process
from utils.model_evaluation import evaluate_model
from utils.load_data import get_input_and_prepare_data

def pipeline_failure_prediction():
    pipeline_code, fault_code, fault_description, train_data, test_data = get_input_and_prepare_data()
    
    # 创建结果保存的文件夹
    model_output_folder = 'model/'
    report_output_folder = 'report/'
    os.makedirs(model_output_folder, exist_ok=True)

    # 设置模型名称
    model_name = f'mlp{fault_code}_{pipeline_code}'

    y_train = (train_data[f'{fault_description}'] == fault_code)
    y_test = (test_data[f'{fault_description}'] == fault_code)

    # 输入是否需要时序化数据
    need_temporal_features = input("Do you want to load a temporal features? (yes/no): ").lower().strip() == 'yes'

    X_train = data_process(train_data, need_temporal_features=need_temporal_features)
    X_test = data_process(test_data, need_temporal_features=need_temporal_features)

    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    print("start training...")
    # 记录开始时间
    start_time = time.time()

    # 定义参数空间
    param_space = {
        'hidden_layer_sizes': [(64, 32), (128, 64), (64, 32, 16)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.01],
        'max_iter': [200, 300, 400]
    }

    # 初始化 MLP 模型
    mlp = MLPClassifier(random_state=42)

    # 使用随机搜索进行超参数调优
    random_search = RandomizedSearchCV(
        mlp, param_distributions=param_space, n_iter=10, cv=3, n_jobs=-1, verbose=2
    )

    # 训练模型
    random_search.fit(X_train_scaled, y_train)

    # 获取最佳模型
    model = random_search.best_estimator_

    # 记录结束时间
    end_time = time.time()
    print(f"MLP模型训练时间: {end_time - start_time:.2f} 秒")

    # 保存模型
    joblib.dump(model, os.path.join(model_output_folder, f"{model_name}.pkl"))

    # 在测试集上评估模型
    y_pred = model.predict(X_test_scaled)
    y_scores = model.predict_proba(X_test_scaled)[:, 1]

    # 使用评估函数
    results = evaluate_model(y_test, y_pred, y_scores, model_name, report_output_folder)

    # 打印一些关键指标
    # .....

    return model, results

if __name__ == "__main__":
    model, results = pipeline_failure_prediction()