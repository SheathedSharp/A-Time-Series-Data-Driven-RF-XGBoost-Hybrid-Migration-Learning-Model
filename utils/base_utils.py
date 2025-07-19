import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def set_global_random_state(random_state=42):
    """
    设置全局随机种子，确保实验的可重复性
    
    Args:
        random_state (int): 随机种子，默认为42
    """
    random.seed(random_state)
    np.random.seed(random_state)
    
    # 设置pandas的随机种子（如果有的话）
    try:
        pd.set_option('mode.chained_assignment', None)
    except:
        pass
    
    print(f"全局随机种子已设置为: {random_state}")


def get_feature_columns(feature_columns):
    """Get the list of feature columns."""
    return [col for col in feature_columns if col not in ['Date', 'Time']]


def print_data_summary(data, title="Data Summary"):
    """Print a summary of the dataset."""
    print(f"\n{title}")
    print("-" * 50)
    print(f"数据形状: {data.shape}")
    print(f"缺失值数量: {data.isnull().sum().sum()}")
    if 'label' in data.columns:
        print(f"正例数量: {data['label'].sum()}")
        print(f"负例数量: {len(data) - data['label'].sum()}")
        print(f"正例比例: {data['label'].mean():.4f}")
    print("-" * 50)


def validate_data_integrity(data, required_columns=None):
    """
    验证数据完整性
    
    Args:
        data (pd.DataFrame): 要验证的数据
        required_columns (list): 必需的列名列表
        
    Returns:
        bool: 数据是否有效
    """
    if data is None or data.empty:
        print("错误: 数据为空")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            print(f"错误: 缺少必需的列: {missing_cols}")
            return False
    
    # 检查是否有全部为NaN的列
    nan_columns = [col for col in data.columns if data[col].isna().all()]
    if nan_columns:
        print(f"警告: 发现全部为NaN的列: {nan_columns}")
    
    return True

