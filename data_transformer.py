"""
Data Transformer Module
Converts any input sensor data format to the standard 20-feature format
(5 metrics × 4 bearings: RMS, Kurtosis, Skewness, Peak, Crest Factor)
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Dict, List, Optional


# Standard feature names for the 20-feature format
STANDARD_FEATURES = [
    'B1_rms', 'B1_kurtosis', 'B1_skew', 'B1_peak', 'B1_crest',
    'B2_rms', 'B2_kurtosis', 'B2_skew', 'B2_peak', 'B2_crest',
    'B3_rms', 'B3_kurtosis', 'B3_skew', 'B3_peak', 'B3_crest',
    'B4_rms', 'B4_kurtosis', 'B4_skew', 'B4_peak', 'B4_crest'
]

# Feature keywords for detection
FEATURE_KEYWORDS = ['rms', 'kurtosis', 'kurt', 'skew', 'peak', 'crest']
BEARING_KEYWORDS = ['b1', 'b2', 'b3', 'b4', 'bearing1', 'bearing2', 'bearing3', 'bearing4', 
                    'sensor1', 'sensor2', 'sensor3', 'sensor4', 'ch1', 'ch2', 'ch3', 'ch4']


class DataTransformer:
    """
    Intelligent data transformer that converts any input format to standard 20-feature format.
    
    Supported input formats:
    - raw_signal: Time-series vibration data (requires feature extraction)
    - partial_features: Less than 20 features (will be padded/interpolated)
    - full_features: Exactly 20 features (direct mapping)
    - extra_features: More than 20 features (will map to first 4 bearings)
    """
    
    def __init__(self, window_size: int = 1024):
        self.window_size = window_size
        self.detected_format = None
        self.transformation_info = {}
    
    def detect_format(self, df: pd.DataFrame) -> str:
        """
        Detect the format of input data based on column count and names.
        
        Returns:
            str: One of 'raw_signal', 'partial_features', 'full_features', 'extra_features'
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = len(numeric_cols)
        col_names_lower = [col.lower() for col in numeric_cols]
        
        # Check if columns have feature keywords (suggests pre-extracted features)
        has_feature_keywords = any(
            any(kw in col for kw in FEATURE_KEYWORDS) 
            for col in col_names_lower
        )
        
        # Check if this looks like raw signal data (many rows, few columns, no feature keywords)
        rows_per_col_ratio = len(df) / max(num_cols, 1)
        looks_like_raw_signal = (
            not has_feature_keywords and 
            rows_per_col_ratio > 100 and  # Many samples per column
            num_cols <= 8  # Typically 1-4 channels for raw signals
        )
        
        if looks_like_raw_signal:
            self.detected_format = 'raw_signal'
        elif num_cols < 20:
            self.detected_format = 'partial_features'
        elif num_cols == 20:
            self.detected_format = 'full_features'
        else:
            self.detected_format = 'extra_features'
        
        self.transformation_info = {
            'detected_format': self.detected_format,
            'original_columns': num_cols,
            'original_rows': len(df),
            'has_feature_keywords': has_feature_keywords
        }
        
        return self.detected_format
    
    def extract_features(self, signal_data: np.ndarray, window_size: Optional[int] = None) -> Dict[str, float]:
        """
        Extract statistical features from raw vibration signal.
        
        Args:
            signal_data: 1D array of raw vibration samples
            window_size: Number of samples per window (default: self.window_size)
        
        Returns:
            Dict with keys: rms, kurtosis, skew, peak, crest
        """
        if window_size is None:
            window_size = self.window_size
        
        # Use the full signal if it's smaller than window size
        if len(signal_data) < window_size:
            data = signal_data
        else:
            data = signal_data[:window_size]
        
        # Handle edge cases
        if len(data) == 0:
            return {'rms': 0, 'kurtosis': 0, 'skew': 0, 'peak': 0, 'crest': 0}
        
        # Calculate features
        rms = np.sqrt(np.mean(data ** 2))
        peak = np.max(np.abs(data))
        crest = peak / rms if rms > 0 else 0
        
        # Kurtosis and skewness need enough samples
        if len(data) > 3:
            kurtosis = stats.kurtosis(data, fisher=True)  # Excess kurtosis
            skewness = stats.skew(data)
        else:
            kurtosis = 0
            skewness = 0
        
        return {
            'rms': float(rms),
            'kurtosis': float(kurtosis),
            'skew': float(skewness),
            'peak': float(peak),
            'crest': float(crest)
        }
    
    def extract_features_windowed(self, signal_data: np.ndarray, window_size: Optional[int] = None) -> List[Dict[str, float]]:
        """
        Extract features from signal using sliding windows.
        Returns a list of feature dictionaries, one per window.
        """
        if window_size is None:
            window_size = self.window_size
        
        features_list = []
        num_windows = max(1, len(signal_data) // window_size)
        
        for i in range(num_windows):
            start = i * window_size
            end = start + window_size
            window_data = signal_data[start:end]
            features = self.extract_features(window_data, window_size)
            features_list.append(features)
        
        return features_list
    
    def transform_raw_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw signal data to 20-feature format.
        Each numeric column is treated as a separate bearing/sensor channel.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        num_channels = min(len(numeric_cols), 4)  # Max 4 bearings
        
        # Determine number of output rows based on windowing
        sample_col = numeric_cols[0]
        num_windows = max(1, len(df) // self.window_size)
        
        result_rows = []
        
        for window_idx in range(num_windows):
            start = window_idx * self.window_size
            end = start + self.window_size
            
            row_features = {}
            
            for bearing_idx in range(4):
                bearing_prefix = f'B{bearing_idx + 1}'
                
                if bearing_idx < num_channels:
                    # Extract features from this channel
                    col = numeric_cols[bearing_idx]
                    signal_data = df[col].iloc[start:end].values
                    features = self.extract_features(signal_data)
                else:
                    # Pad with zeros for missing bearings
                    features = {'rms': 0, 'kurtosis': 0, 'skew': 0, 'peak': 0, 'crest': 0}
                
                row_features[f'{bearing_prefix}_rms'] = features['rms']
                row_features[f'{bearing_prefix}_kurtosis'] = features['kurtosis']
                row_features[f'{bearing_prefix}_skew'] = features['skew']
                row_features[f'{bearing_prefix}_peak'] = features['peak']
                row_features[f'{bearing_prefix}_crest'] = features['crest']
            
            result_rows.append(row_features)
        
        self.transformation_info['output_rows'] = len(result_rows)
        self.transformation_info['window_size'] = self.window_size
        self.transformation_info['channels_used'] = num_channels
        
        return pd.DataFrame(result_rows, columns=STANDARD_FEATURES)
    
    def transform_partial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform partial feature data (< 20 columns) to 20-feature format.
        Maps existing columns and pads missing features with zeros.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        result = pd.DataFrame(0.0, index=df.index, columns=STANDARD_FEATURES)
        
        # Try to intelligently map columns
        mapped_cols = 0
        
        # First, try exact name matching
        for col in numeric_cols:
            col_lower = col.lower().replace('_', '').replace('-', '')
            for std_col in STANDARD_FEATURES:
                std_lower = std_col.lower().replace('_', '')
                if col_lower == std_lower or col_lower in std_lower or std_lower in col_lower:
                    result[std_col] = df[col].values
                    mapped_cols += 1
                    break
        
        # If no matches, map sequentially
        if mapped_cols == 0:
            for i, col in enumerate(numeric_cols[:20]):
                result[STANDARD_FEATURES[i]] = df[col].values
            mapped_cols = min(len(numeric_cols), 20)
        
        self.transformation_info['mapped_columns'] = mapped_cols
        self.transformation_info['padded_columns'] = 20 - mapped_cols
        
        return result
    
    def transform_full_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data with exactly 20 features (direct mapping).
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        result = pd.DataFrame(index=df.index, columns=STANDARD_FEATURES)
        
        # Map the first 20 numeric columns to standard features
        for i, std_col in enumerate(STANDARD_FEATURES):
            result[std_col] = df[numeric_cols[i]].values
        
        self.transformation_info['mapping'] = 'direct'
        
        return result
    
    def transform_extra_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data with more than 20 features.
        Maps to first 4 bearings (20 features) and discards excess.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        result = pd.DataFrame(index=df.index, columns=STANDARD_FEATURES)
        
        # Group columns by bearing if possible
        bearing_groups = {1: [], 2: [], 3: [], 4: []}
        ungrouped = []
        
        for col in numeric_cols:
            col_lower = col.lower()
            assigned = False
            for b_num in [1, 2, 3, 4]:
                if f'b{b_num}' in col_lower or f'bearing{b_num}' in col_lower or f'sensor{b_num}' in col_lower:
                    bearing_groups[b_num].append(col)
                    assigned = True
                    break
            if not assigned:
                ungrouped.append(col)
        
        # If grouping worked, use grouped columns
        total_grouped = sum(len(g) for g in bearing_groups.values())
        
        if total_grouped >= 20:
            # Use grouped columns
            for b_num in range(1, 5):
                prefix = f'B{b_num}'
                group_cols = bearing_groups[b_num][:5]  # Max 5 features per bearing
                for i, feat_name in enumerate(['rms', 'kurtosis', 'skew', 'peak', 'crest']):
                    if i < len(group_cols):
                        result[f'{prefix}_{feat_name}'] = df[group_cols[i]].values
                    else:
                        result[f'{prefix}_{feat_name}'] = 0.0
            self.transformation_info['mapping'] = 'grouped'
        else:
            # Sequential mapping
            for i, std_col in enumerate(STANDARD_FEATURES):
                result[std_col] = df[numeric_cols[i]].values
            self.transformation_info['mapping'] = 'sequential'
        
        self.transformation_info['discarded_columns'] = len(numeric_cols) - 20
        
        return result
    
    def transform_to_standard(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Main transformation method. Detects format and applies appropriate transformation.
        
        Args:
            df: Input DataFrame with any format
        
        Returns:
            Tuple of (transformed DataFrame with 20 features, transformation info dict)
        """
        format_type = self.detect_format(df)
        
        if format_type == 'raw_signal':
            result = self.transform_raw_signal(df)
        elif format_type == 'partial_features':
            result = self.transform_partial_features(df)
        elif format_type == 'full_features':
            result = self.transform_full_features(df)
        else:  # extra_features
            result = self.transform_extra_features(df)
        
        self.transformation_info['output_columns'] = 20
        self.transformation_info['output_rows'] = len(result)
        
        return result, self.transformation_info



# Utility function for quick transformation
def transform_data(df: pd.DataFrame, window_size: int = 1024) -> Tuple[pd.DataFrame, Dict]:
    """
    Quick utility function to transform any data to standard 20-feature format.
    
    Args:
        df: Input DataFrame
        window_size: Window size for raw signal feature extraction
    
    Returns:
        Tuple of (transformed DataFrame, transformation info)
    """
    transformer = DataTransformer(window_size=window_size)
    return transformer.transform_to_standard(df)
