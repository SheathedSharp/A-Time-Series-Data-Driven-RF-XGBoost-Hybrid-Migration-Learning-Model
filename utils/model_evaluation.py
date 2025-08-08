from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np
from tabulate import tabulate
from utils.progress_display import create_progress_display


def evaluate_model(model, x_test, y_test, optimize_threshold=True):
    """Evaluate model performance with optional threshold optimization."""
    progress = create_progress_display()
    
    with progress.model_evaluation_status() as status:
        status.update("Making predictions...")
        
        if optimize_threshold and hasattr(model, 'predict_proba'):
            # Get prediction probabilities for threshold optimization
            y_pred_proba = model.predict_proba(x_test)[:, 1]
            
            status.update("Optimizing decision threshold...")
            best_threshold, best_f1 = find_optimal_threshold(y_test, y_pred_proba)
            y_pred = (y_pred_proba >= best_threshold).astype(int)
            
            threshold_info = f" (optimized threshold: {best_threshold:.3f})"
        else:
            y_pred = model.predict(x_test)
            threshold_info = " (default threshold: 0.500)"

        # Basic metrics
        status.update("Calculating performance metrics...")
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, zero_division=1),
            'Recall': recall_score(y_test, y_pred, zero_division=1),
            'F1 Score': f1_score(y_test, y_pred, zero_division=1)
        }

        # Display basic metrics table
        progress.display_model_metrics(metrics, "Model Performance Metrics")

        # Get detailed classification report data
        status.update("Generating detailed classification report...")
        report = classification_report(y_test, y_pred, output_dict=True)

        # Prepare classification report table data
        report_data = []
        for label in ['False', 'True']:
            report_data.append([
                label,
                f"{report[label]['precision']:.4f}",
                f"{report[label]['recall']:.4f}",
                f"{report[label]['f1-score']:.4f}",
                report[label]['support']
            ])

        # Add average values
        for avg in ['macro avg', 'weighted avg']:
            report_data.append([
                avg,
                f"{report[avg]['precision']:.4f}",
                f"{report[avg]['recall']:.4f}",
                f"{report[avg]['f1-score']:.4f}",
                report[avg]['support']
            ])

        # Display detailed classification report
        detailed_report = {}
        for i, row in enumerate(report_data):
            if i < 2:  # Class-specific metrics
                class_name = row[0]
                detailed_report[f"{class_name} - Precision"] = row[1]
                detailed_report[f"{class_name} - Recall"] = row[2]
                detailed_report[f"{class_name} - F1-Score"] = row[3]
                detailed_report[f"{class_name} - Support"] = row[4]
            else:  # Average metrics
                avg_name = row[0]
                detailed_report[f"{avg_name} - Precision"] = row[1]
                detailed_report[f"{avg_name} - Recall"] = row[2]
                detailed_report[f"{avg_name} - F1-Score"] = row[3]
                detailed_report[f"{avg_name} - Support"] = row[4]

        progress.display_results_table("Detailed Classification Report", detailed_report)
        status.complete(f"Model evaluation completed{threshold_info}")


def find_optimal_threshold(y_true, y_pred_proba, metric='f1'):
    """Find optimal threshold that maximizes the specified metric."""
    thresholds = np.arange(0.1, 0.9, 0.01)  # Search from 0.1 to 0.9
    best_score = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred_thresh = (y_pred_proba >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred_thresh, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred_thresh, zero_division=0)
        else:
            score = f1_score(y_true, y_pred_thresh, zero_division=0)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score