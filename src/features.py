import numpy as np
from scipy.stats import skew, kurtosis

def calculate_snapshot_features(signal_data, set_name):
    """
    Extracts 20 features from raw signal data.
    Handles the column mismatch between Set 1 (8 cols) and Sets 2/3 (4 cols).
    """
    stats = {}
    
    # 1. Hardware Mapping
    if set_name == 'Set1':
        # Use columns 0, 2, 4, 6 (X-axis)
        bearings_map = {
            'B1': signal_data.iloc[:, 0], 
            'B2': signal_data.iloc[:, 2], 
            'B3': signal_data.iloc[:, 4], 
            'B4': signal_data.iloc[:, 6]
        }
    else:
        # Use columns 0, 1, 2, 3
        bearings_map = {
            'B1': signal_data.iloc[:, 0], 
            'B2': signal_data.iloc[:, 1], 
            'B3': signal_data.iloc[:, 2], 
            'B4': signal_data.iloc[:, 3]
        }

    # 2. Physics Calculation
    for b_name, signal in bearings_map.items():
        abs_signal = np.abs(signal)
        rms = np.sqrt(np.mean(signal**2))
        
        stats[f'{b_name}_rms'] = rms
        stats[f'{b_name}_kurtosis'] = kurtosis(signal)
        stats[f'{b_name}_skew'] = skew(signal)
        stats[f'{b_name}_peak'] = np.max(abs_signal)
        # Avoid division by zero
        stats[f'{b_name}_crest'] = (stats[f'{b_name}_peak'] / rms) if rms != 0 else 0
        
    return stats