'''
Author: hiddenSharp429 z404878860@163.com
Date: 2024-10-09 09:15:00
LastEditors: hiddenSharp429 z404878860@163.com
LastEditTime: 2024-10-10 15:56:10
FilePath: /Application of Time Series-Driven XGBoost Model in Pipeline Fault Prediction/LightGBM.py
Description: LightGBM implementation for pipeline failure prediction
'''
import os
import time
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
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
    model_name = f'lightgbm_{fault_code}_{pipeline_code}'

    y_train = (train_data[f'{fault_description}'] == fault_code)
    y_test = (test_data[f'{fault_description}'] == fault_code)

    # 输入是否需要时序化数据
    need_temporal_features = input("Do you want to load temporal features? (yes/no): ").lower().strip() == 'yes'

    X_train = data_process(train_data, need_temporal_features=need_temporal_features)
    X_test = data_process(test_data, need_temporal_features=need_temporal_features)

    print("Start training...")
    # 记录开始时间
    start_time = time.time()

    # 定义参数空间
    param_space = {
        'num_leaves': [31, 63, 127],
        'max_depth': [-1, 5, 10],
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [100, 200, 300],
        'min_child_samples': [20, 50, 100],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }

    # 初始化 LightGBM 模型
    lgbm = LGBMClassifier(random_state=42)

    # 使用随机搜索进行超参数调优
    random_search = RandomizedSearchCV(
        lgbm, param_distributions=param_space, n_iter=10, cv=3, n_jobs=-1, verbose=2
    )

    # 训练模型
    random_search.fit(X_train, y_train)

    # 获取最佳模型
    model = random_search.best_estimator_

    # 记录结束时间
    end_time = time.time()
    print(f"LightGBM模型训练时间: {end_time - start_time:.2f} 秒")

    # 保存模型
    joblib.dump(model, os.path.join(model_output_folder, f"{model_name}.pkl"))

    # 在测试集上评估模型
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]

    # 使用评估函数
    results = evaluate_model(y_test, y_pred, y_scores, model_name, report_output_folder)


    return model, results

if __name__ == "__main__":
    model, results = pipeline_failure_prediction()