import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
TEMPORAL_DATA_DIR = os.path.join(DATA_DIR, 'temporal_features')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'trained_models')
FEATURE_DIR = os.path.join(DATA_DIR, 'selected_features')
EXPERIMENT_DIR = os.path.join(PROJECT_ROOT, 'experiments')
RESULT_DIR = os.path.join(EXPERIMENT_DIR, 'results')
REPORT_DIR = os.path.join(EXPERIMENT_DIR, 'reports')
PIC_DIR = os.path.join(EXPERIMENT_DIR, 'pic')

for directory in [DATA_DIR, RAW_DATA_DIR, MODEL_DIR, FEATURE_DIR, RESULT_DIR, EXPERIMENT_DIR, REPORT_DIR, PIC_DIR]:
    os.makedirs(directory, exist_ok=True)

FAULT_DESCRIPTIONS = {
    1001: 'Material Push Device Fault 1001',
    2001: 'Material Detection Device Fault 2001',
    4001: 'Filling Device Detection Fault 4001',
    4002: 'Filling Device Locator Fault 4002',
    4003: 'Filling Device Filling Fault 4003',
    5001: 'Capping Device Locator Fault 5001',
    5002: 'Capping Device Capping Fault 5002',
    6001: 'Cap Screwing Device Locator Fault 6001',
    6002: 'Cap Screwing Device Fault 6002'
}
EXCLUDE_COLUMNS = ['Date', 'Time', 'Production Line Number', 'Material Push Device Fault 1001',
                     'Material Detection Device Fault 2001', 'Filling Device Detection Fault 4001',
                     'Filling Device Locator Fault 4002', 'Filling Device Filling Fault 4003',
                     'Capping Device Locator Fault 5001', 'Capping Device Capping Fault 5002',
                     'Cap Screwing Device Locator Fault 6001', 'Cap Screwing Device Fault 6002']

MODEL_CONFIGS = {
    'xgboost': {
        'script': 'scripts/train_xgboost.py',
        'description': 'RF-XGBoost hybrid model (temporal features + CBSS + RF selection)',
        'name': 'RF-XGBOOST HYBRID'
    },
    'lightgbm': {
        'script': 'scripts/train_lightgbm.py',
        'description': 'Default LightGBM parameters (no optimization)',
        'name': 'LightGBM BASELINE'
    },
    'svm': {
        'script': 'scripts/train_svm.py', 
        'description': 'Default SVM parameters (Linear kernel, balanced class weights)',
        'name': 'SVM BASELINE'
    },
    'mlp': {
        'script': 'scripts/train_mlp.py',
        'description': 'Default MLP parameters (100-50 hidden layers, early stopping)', 
        'name': 'MLP BASELINE'
    },
    'lstm': {
        'script': 'scripts/train_lstm.py',
        'description': 'Default LSTM parameters (sequence modeling)',
        'name': 'LSTM BASELINE'
    }
}

def get_fault_description(fault_code):
    return FAULT_DESCRIPTIONS.get(fault_code, "")

def get_production_line_data_path(production_line_code, temporal=False):
    if temporal:
        return os.path.join(TEMPORAL_DATA_DIR, f'production_line_{production_line_code}.csv')
    else:
        return os.path.join(RAW_DATA_DIR, f'production_line_{production_line_code}.csv')

def get_model_path(model_name):
    return os.path.join(MODEL_DIR, f'{model_name}.pkl')

def get_feature_path(fault_code):
    return os.path.join(FEATURE_DIR, f'{fault_code}_selected_features.csv')

PLOT_CONFIG = {
    'rcparams': {
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 12,
        'axes.linewidth': 1.2,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'legend.frameon': True,
        'legend.fancybox': False,
        'legend.edgecolor': 'black',
        'grid.linewidth': 0.8,
        'lines.linewidth': 2.0,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    },
    
    # Professional color palette for different models
    'model_colors': {
        'xgboost': '#1f77b4',     # Professional blue
        'lightgbm': '#ff7f0e',    # Professional orange
        'lstm': '#2ca02c',        # Professional green
        'svm': '#d62728',         # Professional red
        'mlp': '#9467bd'          # Professional purple
    },
    
    # Distinguishable colors for fault types (works in both color and grayscale)
    'fault_colors': [
        '#1f77b4',  # Professional blue
        '#ff7f0e',  # Professional orange
        '#2ca02c',  # Professional green
        '#d62728',  # Professional red
        '#9467bd',  # Professional purple
        '#8c564b',  # Professional brown
        '#e377c2',  # Professional pink
        '#7f7f7f',  # Professional gray
        '#bcbd22',  # Professional olive
        '#17becf'   # Professional cyan
    ],
    
    # Line styles for better distinction in grayscale
    'line_styles': ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--'],
    
    # Figure dimensions for different plot types
    'figure_sizes': {
        'roc_single': (8.5, 7),           # Single column width for ROC
        'confusion_matrix': (3, 3.5),    # Per subplot for confusion matrix (increased)
        'performance_comparison': (12, 10),  # Multi-metric comparison
        'single_plot': (10, 8)            # General single plot
    },
    
    # Output formats
    'output_formats': ['png', 'pdf', 'eps'],
    
    # Common plot styling
    'common_style': {
        'grid_alpha': 0.3,
        'line_alpha': 0.85,
        'line_width': 2.2,
        'reference_line_width': 1.0,
        'reference_line_alpha': 0.7,
        'edge_color': 'black',
        'edge_width': 0.8
    }
}


def apply_publication_style():
    """Apply publication-quality matplotlib style."""
    import matplotlib.pyplot as plt
    plt.style.use('default')
    plt.rcParams.update(PLOT_CONFIG['rcparams'])


def get_model_color(model_name):
    """Get color for a specific model."""
    return PLOT_CONFIG['model_colors'].get(model_name, '#7f7f7f')


def get_fault_colors():
    """Get list of colors for fault types."""
    return PLOT_CONFIG['fault_colors']


def get_line_styles():
    """Get list of line styles for fault types."""
    return PLOT_CONFIG['line_styles']