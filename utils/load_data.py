'''
Author: hiddenSharp429 z404878860@163.com
Date: 2024-07-04 16:47:37
LastEditors: hiddenSharp429 z404878860@163.com
LastEditTime: 2024-10-08 20:24:47
FilePath: /Pipeline-Failure-Prediction/load_data.py
Description: 读取数据
'''
import os
import pandas as pd

from utils.get_fault_description import get_fault_description
from utils.split_train_test_datasets import split_train_test_datasets

# 读取数据并且获取测试集和训练集
def read_data(file_path):
    # 读取CSV文件
    data = pd.read_csv(file_path)

    # 处理缺失值（如果有）
    data.fillna(0, inplace=True)

    return data

def get_input_and_prepare_data():
    print("\n" + "="*50)
    print("Data Preparation".center(50))
    print("="*50 + "\n")

    while True:
        try:
            pipeline_code = int(input("Please input pipeline code: "))
            file_path = f'./data/M{pipeline_code}.csv'
            if os.path.exists(file_path):
                data = read_data(file_path)
                break
            else:
                print(f"Error: File not found at {file_path}")
        except ValueError:
            print("Error: Please enter a valid integer for pipeline code.")

    print("\n" + "-"*50 + "\n")

    while True:
        try:
            fault_code = int(input("Please input fault code: "))
            fault_description = get_fault_description(fault_code)
            if fault_description:
                break
            else:
                print("Error: Invalid fault code. Please try again.")
        except ValueError:
            print("Error: Please enter a valid integer for fault code.")

    print(f"\nFault Description: {fault_description}")

    print("\n" + "-"*50)
    print("Splitting data into train and test sets...")
    train_data, test_data = split_train_test_datasets(data, fault_code, fault_description)
    print("Data split complete.")

    print("\n" + "="*50 + "\n")

    return pipeline_code, fault_code, fault_description, train_data, test_data