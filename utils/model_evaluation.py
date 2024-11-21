from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tabulate import tabulate


def evaluate_model(model, x_test, y_test):
    """Evaluate model performance."""
    y_pred = model.predict(x_test)

    # Basic metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=1),
        'Recall': recall_score(y_test, y_pred, zero_division=1),
        'F1 Score': f1_score(y_test, y_pred, zero_division=1)
    }

    # Print basic metrics table
    headers = ["Metric", "Value"]
    metrics_data = [[metric, f"{value:.4f}"] for metric, value in metrics.items()]

    print("\nModel Performance Metrics:")
    print(tabulate(metrics_data, headers=headers, tablefmt="grid"))

    # Get detailed classification report data
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

    print("\nDetailed Classification Report:")
    print(tabulate(report_data,
                   headers=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'],
                   tablefmt="grid"))