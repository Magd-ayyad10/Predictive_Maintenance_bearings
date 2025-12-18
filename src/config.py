import os
"""This keeps the setting in one place.if the path is changed, it changed here
not in five different files"""
# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)

data_path = r"F:\IMS\IMS" 
model_save_path = os.path.join(project_root, 'models', 'universal_model.joblib')

# Hyperparameters
pca_components = 10
threshold_buffer = 1.10  # 10% safety buffer
train_split_ratio = 0.15 # Use first 15% as healthy