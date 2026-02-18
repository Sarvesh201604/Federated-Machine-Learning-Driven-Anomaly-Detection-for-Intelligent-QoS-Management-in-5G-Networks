import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import os
import joblib

# Configuration
PROCESSED_DIR = 'processed_data'
HIDDEN_LAYER_SIZES = (50, 25) # 2 hidden layers
MAX_ITER = 1  # For simulation speed, and to mimic "epochs per round"
RANDOM_STATE = 42

def create_model(warm_start=True):
    """
    Creates a standard MLPClassifier with fixed architecture.
    """
    clf = MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYER_SIZES,
        activation='relu',
        solver='adam',
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        warm_start=warm_start
    )
    return clf

def train_local_node(node_id, data_chunk):
    """
    Simulates training on a local node.
    """
    print(f"[Node {node_id}] Starting training on {len(data_chunk)} samples...")
    
    # Separate features and target
    X = data_chunk.drop('label', axis=1)
    y = data_chunk['label']
    
    # Initialize and train model
    model = create_model(warm_start=True)
    
    # Train using fit()
    model.fit(X, y)
    
    print(f"[Node {node_id}] Training complete. Score: {model.score(X, y):.4f}")
    
    return {
        'node_id': node_id,
        'coefs': model.coefs_,
        'intercepts': model.intercepts_,
        'classes': model.classes_ 
    }
def aggregate_models(local_updates):
    """
    Aggregates model weights using FedAvg (Federated Averaging).
    
    Args:
        local_updates (list): List of dicts returned by train_local_node.
        
    Returns:
        tuple: (avg_coefs, avg_intercepts) representing the global model weights.
    """
    print(f"Aggregating {len(local_updates)} local models...")
    
    n_models = len(local_updates)
    
    # Initialize sums with the first model's weights
    base_coefs = local_updates[0]['coefs']
    base_intercepts = local_updates[0]['intercepts']
    
    # Create structures to hold sums, mimicking the shape of base weights
    # Note: coefs_ is a list of numpy arrays.
    sum_coefs = [np.zeros_like(c) for c in base_coefs]
    sum_intercepts = [np.zeros_like(i) for i in base_intercepts]
    
    # Sum up all weights
    for update in local_updates:
        curr_coefs = update['coefs']
        curr_intercepts = update['intercepts']
        
        for i in range(len(sum_coefs)):
            sum_coefs[i] += curr_coefs[i]
            sum_intercepts[i] += curr_intercepts[i]
            
    # Calculate average
    avg_coefs = [c / n_models for c in sum_coefs]
    avg_intercepts = [i / n_models for i in sum_intercepts]
    
    print("Aggregation complete.")
    return avg_coefs, avg_intercepts

def test_global_model(avg_coefs, avg_intercepts, test_file):
    """
    Constructs a global model from aggregated weights and evaluates it.
    """
    print("Evaluating Global Model...")
    
    # Load Test Data
    test_df = pd.read_csv(test_file)
    X_test = test_df.drop('label', axis=1)
    y_test = test_df['label']
    
    # Create a dummy model to gain structure
    # We use warm_start=False to ensure clean initialization and avoid validation issues.
    # We pass classes explicitly to partial_fit to verify structure.
    global_model = create_model(warm_start=False)
    
    # We need to initialize the model structure (layers, weights arrays)
    # partial_fit on a dummy sample with all classes is a robust way.
    # We construct a dummy input
    dummy_X = X_test.iloc[:5]
    # We ensure classes are correct.
    all_classes = np.unique(y_test) # Should be [0, 1]
    
    # partial_fit initializes the weights randomly.
    global_model.partial_fit(dummy_X, y_test.iloc[:5], classes=all_classes)
    
    # Now OVERWRITE weights with our aggregated ones
    global_model.coefs_ = avg_coefs
    global_model.intercepts_ = avg_intercepts
    
    # Evaluate
    y_pred = global_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Global Model Accuracy: {acc:.4f}")
    return global_model

if __name__ == "__main__":
    # 1. Load Data Chunks for Clients
    local_updates = []
    
    # Simulate 5 nodes
    for i in range(5):
        filename = os.path.join(PROCESSED_DIR, f'train_client_{i}.csv')
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            # Train
            weights = train_local_node(i, df)
            local_updates.append(weights)
        else:
            print(f"Warning: {filename} not found.")

    # 2. Aggregate
    if local_updates:
        avg_coefs, avg_intercepts = aggregate_models(local_updates)
        
        # 3. Test Global Model
        test_file = os.path.join(PROCESSED_DIR, 'test_global.csv')
        if os.path.exists(test_file):
            test_global_model(avg_coefs, avg_intercepts, test_file)
        else:
            print("Test file not found.")
