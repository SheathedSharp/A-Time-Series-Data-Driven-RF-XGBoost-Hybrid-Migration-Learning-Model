# A Time Series Data-Driven RF-XGBoost Hybrid Migration Learning Model

## Overview
This repository contains the implementation of a novel hybrid fault prediction model that combines Random Forest (RF) and XGBoost with temporal feature engineering. The model is designed for pipeline fault prediction in industrial production lines, utilizing time series data to enhance prediction accuracy.

## Method
Our approach consists of four main components:

### 1. Temporal Feature Engineering
- **Horizontal Processing**: Analyzes relationships between different attributes at each time point
- **Vertical Processing**: Incorporates time-lagged features to capture temporal patterns
- Transforms raw production line data into time series-aware features

### 2. Sampling Methods for Imbalanced Data
- **CBSS (Continuous Balanced Slice Sampler)**: Custom temporal-aware sampling method
- **Random Under Sampling**: Random reduction of majority class samples (fast)
- **NearMiss**: Distance-based intelligent undersampling (boundary-aware)
- **Cluster Centroids**: K-means clustering-based sophisticated undersampling (centroid-based)
- Supports ablation studies to evaluate sampling method effectiveness

### 3. Feature Migration
- Uses Random Forest to evaluate feature importance
- Maps selected features from balanced dataset to original imbalanced dataset
- Preserves key temporal and attribute information
- Reduces feature dimensionality while maintaining prediction power
- Employs greedy selection strategy to identify features contributing to 90% importance

### 4. XGBoost Prediction
- Utilizes selected features for final fault prediction
- Trained on original imbalanced dataset
- Provides robust fault prediction performance

## Project Structure
```angular2html
....
```

## Requirements
we use uv to manage the project.
### Install project dependencies
```bash
uv sync
```

## Usage
### 1. Temporalize Features
To generate time series-aware features, run:
``` 
uv run python scripts/temporalize_dataset.py --production_line 1
```

Required parameter:

`--production_line` specifies the production line number.


### 2. Random Forest Feature Selection
To select features using Random Forest, run:
```
uv run python scripts/select_features.py --production_line 1 --fault_code 1001
```
Required parameters:

`--production_line` specifies the production line number.

`--fault_code` specifies the fault code.

Optional parameters:

`--threshold` specifies the feature importance threshold (default: 0.9).

`--sampling-method` specifies the sampling method ('cbss', 'random_under', 'nearmiss', 'cluster_centroids', 'none').

`--no-temporal` disables temporal feature engineering.

### 3. XGBoost Prediction
To train the XGBoost model, run:
```
uv run python scripts/train_xgboost.py --production_line 1 --fault_code 1001
```

#### Sampling Ablation Experiments
To conduct sampling ablation studies, use different sampling methods:
```bash
# CBSS sampling (default - temporal-aware)
uv run python scripts/train_xgboost.py --production_line 1 --fault_code 1001 --sampling-method cbss

# Random undersampling (majority class reduction - fast)
uv run python scripts/train_xgboost.py --production_line 1 --fault_code 1001 --sampling-method random_under

# NearMiss undersampling (distance-based intelligent undersampling)
uv run python scripts/train_xgboost.py --production_line 1 --fault_code 1001 --sampling-method nearmiss

# Cluster Centroids undersampling (intelligent K-means based)
uv run python scripts/train_xgboost.py --production_line 1 --fault_code 1001 --sampling-method cluster_centroids

# No sampling (raw imbalanced data)
uv run python scripts/train_xgboost.py --production_line 1 --fault_code 1001 --sampling-method none
```
Required parameters:

`--production_line` specifies the production line number.

`--fault_code` specifies the fault code.

Optional parameters:

`--no-temporal` disables temporal feature engineering.

`--parameter-opt` enables hyperparameter optimization.

`--no-rf` disables RF feature selection.

`--sampling-method` specifies the sampling method ('cbss', 'random_under', 'nearmiss', 'cluster_centroids', 'none') (default: cbss).

`--rf-threshold` specifies the feature importance threshold for RF (default: 0.9).

## Citation
....

## Contact
....