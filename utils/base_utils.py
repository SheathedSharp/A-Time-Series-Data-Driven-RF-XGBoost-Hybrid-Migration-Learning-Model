from config import EXCLUDE_COLUMNS
import pandas as pd

def get_feature_columns(columns_list):
    """Get feature column names list

    Args:
        data:

    Returns:
        list: List of feature column names
    """
    return [col for col in columns_list if col not in EXCLUDE_COLUMNS]

def calculate_error_rate(df):
    """Calculate error rate for each error column

    Args:
        df (pd.DataFrame): Input dataset

    Returns:
        list: List of dictionaries containing error rate information for each error column
    """
    total_seconds = len(df)
    total_days = len(df['Date'].unique())
    day_average_second = total_seconds / total_days if total_days != 0 else 0

    print("Total Seconds:", total_seconds)
    print("Total Days:", total_days)
    print("Day Average Second:", day_average_second)

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

def count_zero_nonzero_attributes(file_path):
    """Count zero and nonzero values for each error attribute

    Args:
        file_path:

    Returns:

    """
    df = pd.read_csv(file_path)
    result = {}

    for column in df.columns[-9:]:
        zero_count = (df[column] == 0).sum()
        nonzero_count = (df[column] != 0).sum()
        result[column] = {'zero': zero_count, 'nonzero': nonzero_count}

    return result

