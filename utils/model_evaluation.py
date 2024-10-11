'''
Author: hiddenSharp429 z404878860@163.com
Date: 2024-10-08 10:40:02
LastEditors: hiddenSharp429 z404878860@163.com
LastEditTime: 2024-10-08 20:35:04
FilePath: /Application of Time Series-Driven XGBoost Model in Pipeline Fault Prediction/utils/model_evaluation.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, average_precision_score, precision_recall_curve

def evaluate_model(y_true, y_pred, y_scores, model_name, output_folder):
    """
    Evaluate model performance and generate relevant metrics and charts.

    Parameters:
    y_true: True labels
    y_pred: Predicted labels
    y_scores: Prediction probabilities (for positive class)
    model_name: Model name
    output_folder: Output folder path

    Returns:
    dict: Dictionary containing various performance metrics
    """
    print("Number of predicted positive samples:", np.sum(y_pred))
    print("Number of predicted negative samples:", len(y_pred) - np.sum(y_pred))
    print("Mean of prediction probabilities:", np.mean(y_scores))
    print("Median of prediction probabilities:", np.median(y_scores))
    print("Number of true positive samples:", np.sum(y_true))
    print("Number of true negative samples:", len(y_true) - np.sum(y_true))

    if np.sum(y_pred) == 0:
        print("Warning: No positive samples predicted, returning worst results.")
        results = {
            "accuracy": np.mean(y_true == 0),
            "precision": 0,
            "recall": 0,
            "f1_score": 0,
            "average_precision": 0,
            "best_threshold": 0,
            "precision_best": 0,
            "recall_best": 0,
            "f1_best": 0,
            "confusion_matrix": np.array([[len(y_true), 0], [0, 0]])
        }
    else:
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        average_precision = average_precision_score(y_true, y_scores)

        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_scores)
        f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-10)
        best_threshold_index = np.argmax(f1_scores)
        best_threshold = pr_thresholds[best_threshold_index]

        y_pred_best = (y_scores >= best_threshold).astype(int)
        precision_best = precision_score(y_true, y_pred_best, zero_division=0)
        recall_best = recall_score(y_true, y_pred_best, zero_division=0)
        f1_best = f1_score(y_true, y_pred_best, zero_division=0)

        conf_matrix = confusion_matrix(y_true, y_pred)

        results = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "average_precision": average_precision,
            "best_threshold": best_threshold,
            "precision_best": precision_best,
            "recall_best": recall_best,
            "f1_best": f1_best,
            "confusion_matrix": conf_matrix
        }

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()

        classes = ['Negative', 'Positive']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = 'd'
        thresh = conf_matrix.max() / 2.
        for i, j in np.ndindex(conf_matrix.shape):
            plt.text(j, i, format(conf_matrix[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()

        cm_path = os.path.join(output_folder,"pic", f"{model_name}_confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()

    output_file_path = os.path.join(output_folder, f"{model_name}_results.txt")
    with open(output_file_path, 'w') as f:
        f.write(f"Model: {model_name}\n")
        for key, value in results.items():
            if key != "confusion_matrix":
                f.write(f"{key}: {value}\n")
            else:
                f.write(f"{key}:\n{value}\n")

    print(f"Results for model {model_name} saved to {output_file_path}")
    return results