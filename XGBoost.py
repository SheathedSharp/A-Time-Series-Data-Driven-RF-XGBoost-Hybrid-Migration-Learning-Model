import os
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, classification_report

from RF import select_important_feature
from load_data import read_data
from utils.caculate_ks import calculate_ks
from utils.split_train_test_datasets import split_train_test_datasets
from utils.get_fault_description import get_fault_description

def pipeline_failure_prediction():
    pipeline_code = int(input("请输入流水线代码："))
    data = read_data(file_path=f'./data/M{pipeline_code}.csv')
    train_data, test_data = split_train_test_datasets(data)

    fault_code = int(input("请输入故障代码："))
    fault_description = get_fault_description(fault_code)

    # 创建结果保存的文件夹
    output_folder = 'model/'
    os.makedirs(output_folder, exist_ok=True)

    # 设置模型名称
    model_name = f'xgboost{fault_code}_{pipeline_code}'
    
    # 加载预训练模型
    model, model_exist = load_pre_trained_model(model_name, need_load=False)

    
    # 使用随机森林选择重要特征
    X_train_selected, X_test_selected, y_train_original, y_test_original = select_important_feature(
        train_data, test_data, fault_code, fault_description, model_exist, need_select=True)

    # 记录开始时间
    start_time = time.time()

    # 没有预训练模型时，训练新模型
    if model is None:
        # 特征标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)

        # 定义初始参数空间
        param_space = {
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200, 300, 400, 500],
            'subsample': [0.5, 0.7, 0.8, 1.0],
            'colsample_bytree': [0.5, 0.7, 0.8, 1.0],
            'max_depth': [5, 7, 10, 15],
        }

        # 定义评分函数
        scoring = {'precision': 'precision', 'auc': 'roc_auc'}

        # 初始化模型
        model = XGBClassifier()

        # 初始化随机参数搜索
        random_search = RandomizedSearchCV(
            estimator=model, param_distributions=param_space, n_iter=3, scoring=scoring, cv=5, verbose=2, n_jobs=-1, refit='precision'
        )

        # 初始化 precision 的阈值和迭代次数
        precision_threshold = 0.7
        iteration = 0

        while True:
            # 训练模型
            random_search.fit(X_train_scaled, y_train_original)

            # 输出最佳参数
            print("Best parameters found: ", random_search.best_params_)

            # 在测试集上评估模型
            y_pred = random_search.predict(X_test_scaled)

            # 计算性能指标
            accuracy = accuracy_score(y_test_original, y_pred)
            precision = precision_score(y_test_original, y_pred)
            report = classification_report(y_test_original, y_pred, output_dict=True)

            print("Precision:", precision)

            # 判断 precision 是否达到阈值
            if precision >= precision_threshold:
                break

            # 更新参数空间
            for param in param_space:
                if isinstance(param_space[param][0], int):
                    param_space[param] += [param_space[param]
                                            [-1] + step for step in [1, 5, 10]]
                elif isinstance(param_space[param][0], float):
                    param_space[param] += [param_space[param]
                                            [-1] + step for step in [0.01, 0.05, 0.1]]

            iteration += 1



            # 输出最佳参数
            print("Best parameters found: ", random_search.best_params_)
        
        model = random_search.best_estimator_

    # 有预训练模型时，使用迁移学习方法来调整模型参数
    else:
        model = transfer_learning(model, X_train_selected, y_train_original)

    # 在测试集上评估模型
    y_pred = model.predict(X_test_scaled)
    y_scores = model.predict_proba(X_test_scaled)[:, 1]  # 获取预测概率

    # 计算性能指标
    accuracy = accuracy_score(y_test_original, y_pred)
    precision = precision_score(y_test_original, y_pred)
    ks_statistic = calculate_ks(y_test_original, y_scores)  # 计算KS统计量
    report = classification_report(
        y_test_original, y_pred, output_dict=True)

    print("Precision:", precision)

    # 记录结束时间
    end_time = time.time()

    # 打印模型训练时间
    print(f"XGBoost模型训练时间: {end_time - start_time:.2f} 秒")

    # 将结果保存到文件
    output_file_path = os.path.join(output_folder, f"{model_name}_results.txt")
    with open(output_file_path, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"KS Statistic: {ks_statistic}\n") 
        f.write("Classification Report:\n")
        for metric, score in report.items():
            f.write(f"{metric}: {score}\n")
        f.write("\n\n")

    # 保存模型
    model_save_path = os.path.join(output_folder, f"{model_name}.model")
    joblib.dump(model, f'{model_save_path}.pkl')

    # 打印结果
    print(f"Results for model {model_name} saved to {output_file_path}")    

def load_pre_trained_model(model_name, need_load=True):
    # 是否需要加载预训练模型
    if not need_load:
        return None, False

    # 使用split方法分割字符串，并选择第一个元素
    model_prefix = model_name.split("_")[0]
    model_path = os.path.join('model/', f'{model_prefix}_101.model.pkl')
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}")
        model = joblib.load(model_path)
        return model, True
    else:
        print(f"No pre-trained model found at {model_path}")
        return None, False

def transfer_learning(model, target_X_train_selected, target_y_train_original):
    # 读取目标数据
    target_X_train_selected = target_X_train_selected
    target_y_train = target_y_train_original

    # 微调模型
    model.fit(target_X_train_selected, target_y_train, xgb_model=model)

    return model



if __name__ == "__main__":
    pipeline_failure_prediction()
