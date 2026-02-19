"""
Federated Server Module
Represents the Central Cloud that aggregates local models using FedAvg algorithm.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import copy

class FedServer:
    """
    Federated Learning Server implementing the FedAvg algorithm.
    Aggregates weights from multiple local models to create a global model.
    """
    
    def __init__(self, num_clients=5):
        """
        Initialize the Federated Server.
        
        Args:
            num_clients: Expected number of participating clients
        """
        self.num_clients = num_clients
        self.global_weights = None
        self.aggregation_history = []
        self.round_number = 0
    
    def aggregate(self, list_of_local_weights):
        """
        Aggregate local model weights using FedAvg algorithm.
        
        FedAvg (Federated Averaging):
        Global_Weight = (1/N) * Σ(Local_Weight_i) for i=1 to N
        
        Args:
            list_of_local_weights: List of dictionaries containing:
                - 'node_id': Identifier for the node
                - 'coefs': List of weight matrices
                - 'intercepts': List of bias vectors
                
        Returns:
            Dictionary containing:
                - 'coefs': Aggregated coefficient matrices
                - 'intercepts': Aggregated intercept vectors
        """
        # Validate input
        if not list_of_local_weights or len(list_of_local_weights) == 0:
            print("Error: No local weights provided for aggregation")
            return None
        
        print(f"\n--- FedAvg Aggregation (Round {self.round_number + 1}) ---")
        print(f"Aggregating weights from {len(list_of_local_weights)} clients...")
        
        try:
            n_models = len(list_of_local_weights)
            
            # Get the structure from the first model
            base_coefs = list_of_local_weights[0]['coefs']
            base_intercepts = list_of_local_weights[0]['intercepts']
            
            # Initialize sum arrays (deep copy to avoid reference issues)
            sum_coefs = [np.zeros_like(c, dtype=np.float64) for c in base_coefs]
            sum_intercepts = [np.zeros_like(i, dtype=np.float64) for i in base_intercepts]
            
            # Sum all weights from participating clients
            for idx, weights_dict in enumerate(list_of_local_weights):
                node_id = weights_dict.get('node_id', f'client_{idx}')
                curr_coefs = weights_dict['coefs']
                curr_intercepts = weights_dict['intercepts']
                
                # Validate structure consistency
                if len(curr_coefs) != len(sum_coefs):
                    print(f"Warning: Inconsistent model structure from {node_id}")
                    continue
                
                # Add to sum
                for layer_idx in range(len(sum_coefs)):
                    sum_coefs[layer_idx] += curr_coefs[layer_idx]
                    sum_intercepts[layer_idx] += curr_intercepts[layer_idx]
                
                print(f"  ✓ Added weights from {node_id}")
            
            # Calculate average (FedAvg formula)
            avg_coefs = [c / n_models for c in sum_coefs]
            avg_intercepts = [i / n_models for i in sum_intercepts]
            
            # Store global weights
            self.global_weights = {
                'coefs': avg_coefs,
                'intercepts': avg_intercepts,
                'round': self.round_number + 1,
                'num_clients': n_models
            }
            
            # Record in history
            self.aggregation_history.append({
                'round': self.round_number + 1,
                'num_clients': n_models,
                'client_ids': [w.get('node_id', f'client_{i}') for i, w in enumerate(list_of_local_weights)]
            })
            
            self.round_number += 1
            
            print(f"✅ Aggregation complete! Global model updated.")
            print(f"   Layers: {len(avg_coefs)}")
            print(f"   Total parameters: {sum(c.size for c in avg_coefs) + sum(i.size for i in avg_intercepts)}")
            
            return {
                'coefs': avg_coefs,
                'intercepts': avg_intercepts
            }
            
        except Exception as e:
            print(f"❌ Error during aggregation: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def weighted_aggregate(self, list_of_local_weights, sample_counts):
        """
        Weighted aggregation based on number of samples at each client.
        More realistic for non-IID data distributions.
        
        Formula: Global_Weight = Σ(n_i/N * Weight_i) where n_i is samples at client i
        
        Args:
            list_of_local_weights: List of local weight dictionaries
            sample_counts: List of integers representing number of samples at each client
            
        Returns:
            Dictionary with aggregated weights
        """
        if not list_of_local_weights or len(list_of_local_weights) == 0:
            print("Error: No local weights provided")
            return None
        
        if len(list_of_local_weights) != len(sample_counts):
            print("Error: Mismatch between weights and sample counts")
            return None
        
        print(f"\n--- Weighted FedAvg Aggregation (Round {self.round_number + 1}) ---")
        
        try:
            # Calculate total samples
            total_samples = sum(sample_counts)
            print(f"Total samples across all clients: {total_samples}")
            
            # Get base structure
            base_coefs = list_of_local_weights[0]['coefs']
            base_intercepts = list_of_local_weights[0]['intercepts']
            
            # Initialize weighted sum
            weighted_coefs = [np.zeros_like(c, dtype=np.float64) for c in base_coefs]
            weighted_intercepts = [np.zeros_like(i, dtype=np.float64) for i in base_intercepts]
            
            # Weighted sum
            for idx, (weights_dict, n_samples) in enumerate(zip(list_of_local_weights, sample_counts)):
                node_id = weights_dict.get('node_id', f'client_{idx}')
                weight_factor = n_samples / total_samples
                
                curr_coefs = weights_dict['coefs']
                curr_intercepts = weights_dict['intercepts']
                
                for layer_idx in range(len(weighted_coefs)):
                    weighted_coefs[layer_idx] += weight_factor * curr_coefs[layer_idx]
                    weighted_intercepts[layer_idx] += weight_factor * curr_intercepts[layer_idx]
                
                print(f"  ✓ {node_id}: {n_samples} samples (weight: {weight_factor:.3f})")
            
            # Store global weights
            self.global_weights = {
                'coefs': weighted_coefs,
                'intercepts': weighted_intercepts,
                'round': self.round_number + 1,
                'num_clients': len(list_of_local_weights)
            }
            
            self.round_number += 1
            
            print(f"✅ Weighted aggregation complete!")
            
            return {
                'coefs': weighted_coefs,
                'intercepts': weighted_intercepts
            }
            
        except Exception as e:
            print(f"❌ Error during weighted aggregation: {e}")
            return None
    
    def get_global_weights(self):
        """
        Get the current global model weights.
        
        Returns:
            Dictionary with global weights or None if not yet aggregated
        """
        if self.global_weights is None:
            print("Warning: No global weights available yet. Run aggregate() first.")
        return self.global_weights
    
    def get_stats(self):
        """
        Get statistics about the federated learning process.
        
        Returns:
            Dictionary with FL statistics
        """
        return {
            'total_rounds': self.round_number,
            'num_clients': self.num_clients,
            'aggregation_history': self.aggregation_history,
            'has_global_model': self.global_weights is not None
        }
    
    def reset(self):
        """Reset the server state."""
        self.global_weights = None
        self.aggregation_history = []
        self.round_number = 0
        print("Server state reset")
    
    def save_global_weights(self, filepath):
        """
        Save global weights to file.
        
        Args:
            filepath: Path to save the weights
        """
        if self.global_weights is None:
            print("Error: No global weights to save")
            return False
        
        import joblib
        joblib.dump(self.global_weights, filepath)
        print(f"Global weights saved to {filepath}")
        return True
    
    def load_global_weights(self, filepath):
        """
        Load global weights from file.
        
        Args:
            filepath: Path to load the weights from
        """
        import os
        import joblib
        
        if not os.path.exists(filepath):
            print(f"Error: File {filepath} not found")
            return False
        
        self.global_weights = joblib.load(filepath)
        print(f"Global weights loaded from {filepath}")
        return True


# Example usage and testing
if __name__ == "__main__":
    print("Testing FedServer with FedAvg algorithm...\n")
    
    # Simulate 3 local models with different weights
    np.random.seed(42)
    
    # Create dummy weight structures
    local_weights_1 = {
        'node_id': 'enb_1',
        'coefs': [
            np.random.randn(8, 10) * 0.1,  # Input to hidden
            np.random.randn(10, 2) * 0.1   # Hidden to output
        ],
        'intercepts': [
            np.random.randn(10) * 0.1,
            np.random.randn(2) * 0.1
        ]
    }
    
    local_weights_2 = {
        'node_id': 'enb_2',
        'coefs': [
            np.random.randn(8, 10) * 0.1 + 0.5,
            np.random.randn(10, 2) * 0.1 + 0.5
        ],
        'intercepts': [
            np.random.randn(10) * 0.1 + 0.5,
            np.random.randn(2) * 0.1 + 0.5
        ]
    }
    
    local_weights_3 = {
        'node_id': 'enb_3',
        'coefs': [
            np.random.randn(8, 10) * 0.1 - 0.5,
            np.random.randn(10, 2) * 0.1 - 0.5
        ],
        'intercepts': [
            np.random.randn(10) * 0.1 - 0.5,
            np.random.randn(2) * 0.1 - 0.5
        ]
    }
    
    # Test 1: Simple FedAvg
    print("=" * 60)
    print("TEST 1: Simple FedAvg (Equal Weights)")
    print("=" * 60)
    
    server = FedServer(num_clients=3)
    
    global_weights = server.aggregate([local_weights_1, local_weights_2, local_weights_3])
    
    if global_weights:
        print(f"\n📊 Global model statistics:")
        print(f"   First layer shape: {global_weights['coefs'][0].shape}")
        print(f"   Second layer shape: {global_weights['coefs'][1].shape}")
        print(f"   First layer mean: {np.mean(global_weights['coefs'][0]):.4f}")
    
    # Test 2: Weighted FedAvg
    print("\n" + "=" * 60)
    print("TEST 2: Weighted FedAvg (Different Sample Sizes)")
    print("=" * 60)
    
    server2 = FedServer(num_clients=3)
    sample_counts = [1000, 500, 200]  # Different sample sizes
    
    global_weights_weighted = server2.weighted_aggregate(
        [local_weights_1, local_weights_2, local_weights_3],
        sample_counts
    )
    
    # Test 3: Server statistics
    print("\n" + "=" * 60)
    print("TEST 3: Server Statistics")
    print("=" * 60)
    
    stats = server.get_stats()
    print(f"\n📈 Server stats:")
    for key, value in stats.items():
        if key != 'aggregation_history':
            print(f"   {key}: {value}")
    
    print("\n✅ All FedServer tests complete!")
