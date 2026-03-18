import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import os

# Define paths
DATA_DIR = 'archive'
PROCESSED_DIR = 'processed_data'
TRAIN_FILE = os.path.join(DATA_DIR, 'UNSW_NB15_training-set.csv')
TEST_FILE = os.path.join(DATA_DIR, 'UNSW_NB15_testing-set.csv')

# create processed dir if it doesn't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Define features
# Traffic Load: sload, dload, rate
# Latency/Jitter: sjit, djit, tcprtt, synack, ackdat
# Identity: proto, service
# Target: label
FEATURES = [
    'sload', 'dload', 'rate',
    'sjit', 'djit', 'tcprtt', 'synack', 'ackdat',
    'proto', 'service',
    'label'
]

NUMERICAL_FEATURES = [
    'sload', 'dload', 'rate',
    'sjit', 'djit', 'tcprtt', 'synack', 'ackdat'
]

CATEGORICAL_FEATURES = ['proto', 'service']

def load_and_process():
    print("Loading data...")
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)

    print(f"Training shape: {train_df.shape}")
    print(f"Testing shape: {test_df.shape}")

    # Select only relevant columns
    print("Selecting features...")
    train_df = train_df[FEATURES]
    test_df = test_df[FEATURES]

    # Handle Categorical Features (Label Encoding)
    print("Encoding categorical features...")
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        # Fit on both train and test to ensure all categories are covered
        unique_values = pd.concat([train_df[col], test_df[col]]).unique()
        le.fit(unique_values)
        
        train_df[col] = le.transform(train_df[col])
        test_df[col] = le.transform(test_df[col])

    # Normalize (MinMaxScaler)
    # Fit scaler ONLY on training data to avoid data leakage
    print("Normalizing data...")
    scaler = MinMaxScaler()
    
    # We want to scale everything except the label
    features_to_scale = [col for col in FEATURES if col != 'label']
    
    train_df[features_to_scale] = scaler.fit_transform(train_df[features_to_scale])
    test_df[features_to_scale] = scaler.transform(test_df[features_to_scale])

    # Split Training Data into 5 chunks (Clients)
    print("Splitting training data for 5 clients...")
    train_chunks = np.array_split(train_df, 5)

    # Save processed files
    print("Saving processed files...")
    for i, chunk in enumerate(train_chunks):
        output_file = os.path.join(PROCESSED_DIR, f'train_client_{i}.csv')
        chunk.to_csv(output_file, index=False)
        print(f"Saved {output_file} (shape: {chunk.shape})")

    test_output_file = os.path.join(PROCESSED_DIR, 'test_global.csv')
    test_df.to_csv(test_output_file, index=False)
    print(f"Saved {test_output_file} (shape: {test_df.shape})")

    print("\nData preparation complete!")

if __name__ == "__main__":
    load_and_process()
