import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
TEMPORAL_DATA_DIR = os.path.join(DATA_DIR, 'temporal_features')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'trained_models')
FEATURE_DIR = os.path.join(DATA_DIR, 'selected_features')
REPORT_DIR = os.path.join(PROJECT_ROOT, 'report')

for directory in [DATA_DIR, RAW_DATA_DIR, MODEL_DIR, FEATURE_DIR, REPORT_DIR]:
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