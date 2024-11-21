import pandas as pd
import numpy as np

class TemporalFeatureProcessor:
    """Process raw data to generate temporal features.
    
    This class implements two-dimensional temporal feature engineering:
    1. Horizontal: Feature differences between attributes
    2. Vertical: Time-lagged features for each attribute
    """


    @staticmethod
    def generate_difference_features(data, feature_columns):
        """Generate horizontal difference features between attributes.

        Args:
            data (pd.DataFrame): Input dataset
            feature_columns (list): List of feature column names

        Returns:
            pd.DataFrame: DataFrame containing difference features
        """
        difference_features = {}

        for i in range(len(feature_columns)):
            for j in range(i + 1, len(feature_columns)):
                col1, col2 = feature_columns[i], feature_columns[j]
                diff_col_name = f'{col1}_{col2}_diff'
                difference_features[diff_col_name] = data[col1] - data[col2]

        return pd.DataFrame(difference_features)

    @staticmethod
    def generate_lag_features(data, feature_columns):
        """Generate vertical time-lagged features.
        
        Args:
            data (pd.DataFrame): Input dataset
            feature_columns (list): List of feature column names
            
        Returns:
            pd.DataFrame: DataFrame containing lag features
        """
        lag_features = {}

        for col in feature_columns:
            values = data[col].values
            lag_values = []
            lag_duration = 0
            current_value = values[0]

            lag_values.append(0)

            for idx in range(1, len(values)):
                if values[idx] == 0:
                    lag_duration = 0
                elif values[idx] == current_value:
                    lag_duration += 1
                else:
                    current_value = values[idx]
                    lag_duration = 0
                lag_values.append(lag_duration)

            lag_features[f'{col}_lag'] = lag_values

        return pd.DataFrame(lag_features)
    
    def process(self, data, base_feature_columns, need_temporal_features=True):
        """Process the input data with temporal feature engineering.
        
        Args:
            data (pd.DataFrame): Input dataset
            base_feature_columns (list): List of base feature column names
            need_temporal_features (bool): Whether to add temporal features
            
        Returns:
            pd.DataFrame: Processed dataset with temporal features
        """

        if not need_temporal_features:
            return data

        # Generate temporal features
        difference_features = self.generate_difference_features(data, base_feature_columns)
        lag_features = self.generate_lag_features(data, base_feature_columns)

        # 确保索引一致性
        difference_features.index = data.index
        lag_features.index = data.index

        # Combine all features
        final_data = pd.concat([
            data,
            difference_features,
            lag_features
        ], axis=1)

        return final_data
