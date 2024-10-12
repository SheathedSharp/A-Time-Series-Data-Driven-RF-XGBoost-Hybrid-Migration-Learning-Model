'''
Author: hiddenSharp429 z404878860@163.com
Date: 2024-10-12 17:13:13
LastEditors: hiddenSharp429 z404878860@163.com
LastEditTime: 2024-10-12 17:28:20
FilePath: /Application of Time Series-Driven XGBoost Model in Pipeline Fault Prediction/count_each_pipeline_error.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import os
import pandas as pd


def calculate_error_rate(df):
    total_seconds = len(df)
    total_days = len(df['日期'].unique())
    day_average_second = total_seconds / total_days if total_days != 0 else 0

    print("Total Seconds:", total_seconds)
    print("Total Days:", total_days)
    print("Day Average Second:", day_average_second)

    # 故障列
    error_columns = df.iloc[:, -9:].columns.tolist()
    print("Error Columns:", error_columns)

    error_results = []

    for column in error_columns:
        print(column)
        error_count = df[column].sum() / int(column[-4:])
        error_day_count = error_count / total_days if total_days != 0 else 0
        error_day_rate = error_day_count / \
            day_average_second if day_average_second != 0 else 0

        print("Error Count:", error_count)
        print("Error Day_Count: {:.2f}".format(error_day_count))
        print("Error Day Rate: {:.2f}".format(error_day_rate))

        error_results.append({column: {'error_count': error_count,
                             'error_day_count': error_day_count, 'error_day_rate': error_day_rate}})

    print("Error Results:", error_results)

    return error_results


def write_results_to_txt(file_path, error_results):
    output_file_path = os.path.splitext(file_path)[0] + '_results.txt'

    with open(output_file_path, 'w') as file:
        for error_dict in error_results:
            for error, results in error_dict.items():
                file.write("{}:\n".format(error))
                file.write("Error Count: {:.10f}\n".format(
                    results['error_count']))
                file.write("Error Day Count: {:.10f}\n".format(
                    results['error_day_count']))
                file.write("Error Day Rate: {:.10f}\n".format(
                    results['error_day_rate']))
                file.write("\n")


def main():
    file_head = './data/'
    file_names = ['M101.csv', 'M102.csv', 'M103.csv']

    for file_name in file_names:
        file_path = os.path.join(file_head, file_name)
        df = pd.read_csv(file_path)
        error_results = calculate_error_rate(df)
        write_results_to_txt(file_path, error_results)


if __name__ == "__main__":
    main()