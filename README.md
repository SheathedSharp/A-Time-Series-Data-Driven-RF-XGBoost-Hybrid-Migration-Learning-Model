# A Time Series Data-Driven RF-XGBoost Hybrid Migration Learning Model

## Overview
This repository contains the implementation of a novel hybrid fault prediction model that combines Random Forest (RF) and XGBoost with temporal feature engineering. The model is designed for pipeline fault prediction in industrial production lines, utilizing time series data to enhance prediction accuracy.

## Method
Our approach consists of four main components:

### 1. Temporal Feature Engineering
- **Horizontal Processing**: Analyzes relationships between different attributes at each time point
- **Vertical Processing**: Incorporates time-lagged features to capture temporal patterns
- Transforms raw production line data into time series-aware features

### 2. Balanced Random Forest Feature Selection
- Custom sampling method to handle imbalanced fault data
- Maintains temporal characteristics during sampling

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
....

## Usage
....

## Citation
....

## Contact
....