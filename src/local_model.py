"""
Local FL Model Module
Represents the Machine Learning model running inside each 5G Base Station.
Uses sklearn MLPClassifier for anomaly detection with incremental learning.
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
from typing import Tuple, Optional

class LocalFLModel:
    """
    Local Federated Learning Model for anomaly detection.
    Each base station runs its own instance of this model.
    """
    
    def __init__(self, node_id, hidden_layers=(10,), random_state=42):
        """
        Initialize the Local FL Model.
        
        Args:
            node_id: Identifier for the base station
            hidden_layers: Tuple defining hidden layer architecture
            random_state: Random seed for reproducibility
        """
        self.node_id = node_id
        self.hidden_layers = hidden_layers
        self.random_state = random_state
        
        # Initialize MLPClassifier with crucial settings for incremental learning
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            max_iter=1,  # Single iteration for incremental learning
            warm_start=True,  # CRUCIAL: Allows incremental learning
            random_state=random_state,
            learning_rate_init=0.001
        )
        
        # Track if model has been fitted
        self.is_fitted = False
        
        # Feature scaler (optional but recommended)
        self.scaler = StandardScaler()
        self.scaler_fitted = False
        
        # Training history
        self.training_history = {
            'rounds': [],
            'accuracy': [],
            'loss': []
        }
    
    def train(self, X_data, y_labels):
        """
        Train the model on local data using incremental learning.
        
        Args:
            X_data: Feature matrix (numpy array or pandas DataFrame)
            y_labels: Labels (0=Normal, 1=Anomaly)
            
        Returns:
            Dictionary with training metrics
        """
        # Convert to numpy arrays if needed
        if isinstance(X_data, pd.DataFrame):
            X_data = X_data.values
        if isinstance(y_labels, pd.Series):
            y_labels = y_labels.values
        
        # Ensure we have data
        if len(X_data) == 0:
            return {'error': 'No training data provided'}
        
        # Scale features
        if not self.scaler_fitted:
            X_scaled = self.scaler.fit_transform(X_data)
            self.scaler_fitted = True
        else:
            X_scaled = self.scaler.transform(X_data)
        
        # First time training
        if not self.is_fitted:
            # Use partial_fit with all classes specified
            self.model.partial_fit(X_scaled, y_labels, classes=[0, 1])
            self.is_fitted = True
        else:
            # Incremental training
            self.model.partial_fit(X_scaled, y_labels)
        
        # Calculate training accuracy
        y_pred = self.model.predict(X_scaled)
        accuracy = accuracy_score(y_labels, y_pred)
        
        # Update history
        self.training_history['rounds'].append(len(self.training_history['rounds']) + 1)
        self.training_history['accuracy'].append(accuracy)
        
        return {
            'node_id': self.node_id,
            'samples': len(X_data),
            'accuracy': accuracy,
            'fitted': self.is_fitted
        }
    
    def get_weights(self):
        """
        Extract model weights (coefficients and intercepts).
        Required for Federated Averaging.
        
        Returns:
            Dictionary containing model weights
        """
        if not self.is_fitted:
            return None
        
        return {
            'node_id': self.node_id,
            'coefs': self.model.coefs_,
            'intercepts': self.model.intercepts_,
            'classes': self.model.classes_
        }
    
    def set_weights(self, coefs, intercepts):
        """
        Update model with new weights (from global aggregation).
        
        Args:
            coefs: List of coefficient matrices
            intercepts: List of intercept vectors
        """
        if not self.is_fitted:
            # Need to fit at least once before we can set weights
            # Create dummy data to initialize the model
            dummy_X = np.random.randn(10, len(coefs[0][0]))
            dummy_y = np.random.randint(0, 2, 10)
            self.model.partial_fit(dummy_X, dummy_y, classes=[0, 1])
            self.is_fitted = True
        
        # Update weights
        self.model.coefs_ = coefs
        self.model.intercepts_ = intercepts
    
    def predict_anomaly_score(self, features):
        """
        Predict the probability of traffic being an anomaly.
        
        Args:
            features: Feature vector (single sample or batch)
            
        Returns:
            Anomaly probability (0-1, higher means more likely anomaly)
        """
        if not self.is_fitted:
            # Model not trained yet, return neutral probability
            return 0.5
        
        # Convert to numpy array if needed
        if isinstance(features, pd.DataFrame):
            features = features.values
        elif isinstance(features, list):
            features = np.array(features)
        
        # Ensure 2D shape
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Scale features
        if self.scaler_fitted:
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = features
        
        # Get probability of class 1 (Anomaly)
        try:
            proba = self.model.predict_proba(features_scaled)
            # Return probability of anomaly class (index 1)
            if proba.shape[1] > 1:
                return proba[:, 1] if len(features) > 1 else proba[0, 1]
            else:
                return proba[:, 0] if len(features) > 1 else proba[0, 0]
        except Exception as e:
            print(f"Warning: Prediction failed - {e}")
            return 0.5
    
    def predict(self, features):
        """
        Predict class labels (0=Normal, 1=Anomaly).
        
        Args:
            features: Feature vector or matrix
            
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            return None
        
        # Convert to numpy array if needed
        if isinstance(features, pd.DataFrame):
            features = features.values
        elif isinstance(features, list):
            features = np.array(features)
        
        # Ensure 2D shape
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Scale features
        if self.scaler_fitted:
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = features
        
        return self.model.predict(features_scaled)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            return {'error': 'Model not fitted yet'}
        
        # Convert to numpy
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values
        if isinstance(y_test, pd.Series):
            y_test = y_test.values
        
        # Scale
        if self.scaler_fitted:
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_test_scaled = X_test
        
        # Predict
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0)
        }
        
        return metrics
    
    def save(self, filepath):
        """Save model to disk."""
        if not self.is_fitted:
            print("Warning: Saving unfitted model")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'scaler_fitted': self.scaler_fitted,
            'is_fitted': self.is_fitted,
            'node_id': self.node_id,
            'history': self.training_history
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model from disk."""
        if not os.path.exists(filepath):
            print(f"Error: File {filepath} not found")
            return False
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.scaler_fitted = model_data['scaler_fitted']
        self.is_fitted = model_data['is_fitted']
        self.training_history = model_data.get('history', self.training_history)
        
        print(f"Model loaded from {filepath}")
        return True


# Example usage and testing
if __name__ == "__main__":
    print("Testing LocalFLModel...")
    
    # Create synthetic training data
    np.random.seed(42)
    
    # Normal traffic (label 0)
    X_normal = np.random.randn(100, 8) * 0.5 + np.array([15, 50, 0.01, 2, 10, 0.3, 1, 0])
    y_normal = np.zeros(100)
    
    # Anomalous traffic (label 1)
    X_anomaly = np.random.randn(50, 8) * 2 + np.array([200, 10, 0.5, 80, 900, 0.9, 3, 1])
    y_anomaly = np.ones(50)
    
    # Combine
    X_train = np.vstack([X_normal, X_anomaly])
    y_train = np.hstack([y_normal, y_anomaly])
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X_train))
    X_train = X_train[shuffle_idx]
    y_train = y_train[shuffle_idx]
    
    # Create model
    model = LocalFLModel(node_id='enb_1', hidden_layers=(10, 5))
    
    # Train
    print("\nTraining model...")
    result = model.train(X_train, y_train)
    print(f"Training result: {result}")
    
    # Get weights
    weights = model.get_weights()
    print(f"\nWeights extracted: {len(weights['coefs'])} layers")
    
    # Test prediction
    test_sample = X_train[0:1]
    anomaly_score = model.predict_anomaly_score(test_sample)
    print(f"\nAnomaly score for test sample: {anomaly_score:.4f}")
    
    # Evaluate
    metrics = model.evaluate(X_train[:30], y_train[:30])
    print(f"\nEvaluation metrics: {metrics}")
    
    print("\n✅ LocalFLModel test complete!")
