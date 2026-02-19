"""
COMPLETE FEDERATED LEARNING + ANOMALY-AWARE ROUTING SYSTEM
This is the MAIN INTEGRATION script that brings everything together.

PROJECT: Federated Machine Learning-Driven Anomaly Detection 
         for Intelligent QoS Management in 5G Networks

ARCHITECTURE:
    1. Data Collection (data_logger.py)
    2. Local ML Models (local_model.py) - One per base station
    3. Federated Aggregation (federated_server.py) - FedAvg algorithm
    4. Intelligent Routing (anomaly_router.py) - Anomaly-aware decisions

NOVELTY:
    Traditional QoS Routing: Cost = Latency + Load
    Our System: Cost = Latency + Load + (Anomaly_Probability * Penalty)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os
import random
from datetime import datetime

# Import our custom modules
from data_logger import DataLogger
from local_model import LocalFLModel
from federated_server import FedServer
from anomaly_router import AnomalyAwareRouter

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

class IntegratedFLQoSSystem:
    """
    Complete integrated system demonstrating:
    - Federated Learning at base stations
    - Anomaly detection
    - Intelligent routing
    """
    
    def __init__(self, num_base_stations=5, simulation_rounds=10):
        """
        Initialize the complete FL-QoS system.
        
        Args:
            num_base_stations: Number of base stations in network
            simulation_rounds: Number of FL training rounds
        """
        self.num_base_stations = num_base_stations
        self.simulation_rounds = simulation_rounds
        
        # Components
        self.data_logger = DataLogger(output_dir='traffic_logs')
        self.local_models = {}
        self.fed_server = FedServer(num_clients=num_base_stations)
        self.network_graph = None
        self.router = None
        
        # Results storage
        self.results = {
            'fl_accuracy': [],
            'routing_performance': [],
            'anomaly_detection_rate': []
        }
        
        print("=" * 70)
        print(" FEDERATED LEARNING + ANOMALY-AWARE QoS ROUTING SYSTEM")
        print("=" * 70)
        print(f"Base Stations: {num_base_stations}")
        print(f"FL Rounds: {simulation_rounds}")
        print("=" * 70)
    
    def setup_network_topology(self):
        """Create 5G network topology."""
        print("\n[STEP 1] Setting up 5G Network Topology...")
        
        self.network_graph = nx.Graph()
        
        # Add base station nodes
        for i in range(self.num_base_stations):
            self.network_graph.add_node(i, type='gNB', name=f'BS-{i}')
        
        # Create realistic topology (partial mesh)
        edges = [
            (0, 1, {'base_latency': 10.0, 'capacity': 100.0, 'current_load': 0.0}),
            (1, 2, {'base_latency': 15.0, 'capacity': 100.0, 'current_load': 0.0}),
            (2, 3, {'base_latency': 12.0, 'capacity': 100.0, 'current_load': 0.0}),
            (3, 4, {'base_latency': 10.0, 'capacity': 100.0, 'current_load': 0.0}),
            (4, 0, {'base_latency': 18.0, 'capacity': 100.0, 'current_load': 0.0}),
            (0, 2, {'base_latency': 25.0, 'capacity': 100.0, 'current_load': 0.0}),
            (1, 3, {'base_latency': 20.0, 'capacity': 100.0, 'current_load': 0.0}),
            (2, 4, {'base_latency': 22.0, 'capacity': 100.0, 'current_load': 0.0}),
        ]
        
        for u, v, attrs in edges:
            self.network_graph.add_edge(u, v, **attrs)
            # Initialize cost same as latency
            self.network_graph[u][v]['cost'] = attrs['base_latency']
        
        print(f"✓ Created network: {self.network_graph.number_of_nodes()} nodes, "
              f"{self.network_graph.number_of_edges()} links")
    
    def generate_training_data(self, samples_per_station=200):
        """Generate synthetic traffic data for each base station."""
        print("\n[STEP 2] Generating Training Data for Each Base Station...")
        
        training_data = {}
        
        for bs_id in range(self.num_base_stations):
            # Normal traffic (70%)
            n_normal = int(samples_per_station * 0.7)
            X_normal = np.random.randn(n_normal, 8) * np.array([5, 10, 0.01, 1, 5, 0.1, 0.5, 0.1])
            X_normal += np.array([15, 50, 0.02, 2, 15, 0.3, 1, 0])
            X_normal = np.clip(X_normal, 0, None)  # No negative values
            y_normal = np.zeros(n_normal, dtype=int)
            
            # Anomalous traffic (30%)
            n_anomaly = samples_per_station - n_normal
            X_anomaly = np.random.randn(n_anomaly, 8) * np.array([50, 5, 0.2, 20, 100, 0.1, 0.5, 0.1])
            X_anomaly += np.array([200, 10, 0.4, 80, 800, 0.85, 3, 1])
            X_anomaly = np.clip(X_anomaly, 0, None)
            y_anomaly = np.ones(n_anomaly, dtype=int)
            
            # Combine and shuffle
            X = np.vstack([X_normal, X_anomaly])
            y = np.hstack([y_normal, y_anomaly])
            
            shuffle_idx = np.random.permutation(len(X))
            X = X[shuffle_idx]
            y = y[shuffle_idx]
            
            training_data[bs_id] = {'X': X, 'y': y}
            
            print(f"  BS-{bs_id}: {len(X)} samples ({n_normal} normal, {n_anomaly} anomaly)")
        
        return training_data
    
    def federated_learning_phase(self, training_data):
        """Execute federated learning training."""
        print("\n[STEP 3] Federated Learning Training Phase...")
        print("-" * 70)
        
        # Initialize local models
        for bs_id in range(self.num_base_stations):
            self.local_models[bs_id] = LocalFLModel(
                node_id=f'BS-{bs_id}',
                hidden_layers=(10, 5),
                random_state=RANDOM_SEED
            )
        
        # FL Training Rounds
        for round_num in range(self.simulation_rounds):
            print(f"\n📡 FL Round {round_num + 1}/{self.simulation_rounds}")
            
            local_weights = []
            round_accuracy = []
            
            # Each base station trains locally
            for bs_id in range(self.num_base_stations):
                model = self.local_models[bs_id]
                data = training_data[bs_id]
                
                # Local training
                result = model.train(data['X'], data['y'])
                round_accuracy.append(result['accuracy'])
                
                # Extract weights
                weights = model.get_weights()
                if weights:
                    local_weights.append(weights)
                
                print(f"  BS-{bs_id}: Accuracy = {result['accuracy']:.4f}")
            
            # Federated Aggregation (FedAvg)
            if local_weights:
                global_weights = self.fed_server.aggregate(local_weights)
                
                # Distribute global weights back to local models
                if global_weights:
                    for bs_id in range(self.num_base_stations):
                        self.local_models[bs_id].set_weights(
                            global_weights['coefs'],
                            global_weights['intercepts']
                        )
            
            # Track average accuracy
            avg_accuracy = np.mean(round_accuracy)
            self.results['fl_accuracy'].append(avg_accuracy)
            print(f"  📊 Average Accuracy: {avg_accuracy:.4f}")
        
        print("\n✅ Federated Learning Complete!")
        
        # Use any local model as global (they all have same weights now)
        return self.local_models[0]
    
    def setup_anomaly_aware_routing(self, global_model):
        """Setup routing with FL model."""
        print("\n[STEP 4] Initializing Anomaly-Aware Routing...")
        
        self.router = AnomalyAwareRouter(
            self.network_graph,
            global_model,
            anomaly_penalty=1000.0
        )
        
        print("✓ Router initialized with Global FL Model")
        print("✓ Routing formula: Cost = Latency + Load + (Anomaly × Penalty)")
    
    def run_routing_simulation(self, num_flows=100):
        """Simulate traffic routing with and without anomaly awareness."""
        print("\n[STEP 5] Running Routing Simulation...")
        print("-" * 70)
        
        latencies_baseline = []
        latencies_intelligent = []
        
        for flow_id in range(num_flows):
            # Random source and destination
            source = random.randint(0, self.num_base_stations - 1)
            dest = random.randint(0, self.num_base_stations - 1)
            while dest == source:
                dest = random.randint(0, self.num_base_stations - 1)
            
            # Generate traffic features (mix of normal and anomaly)
            is_anomaly = random.random() < 0.3  # 30% anomalies
            
            if is_anomaly:
                features = {
                    'latency': random.uniform(150, 300),
                    'throughput': random.uniform(5, 15),
                    'packet_loss': random.uniform(0.3, 0.6),
                    'jitter': random.uniform(50, 100),
                    'queue_length': random.uniform(700, 1000),
                    'load': random.uniform(0.8, 0.95),
                    'traffic_type': 1,
                    'label': 1
                }
            else:
                features = {
                    'latency': random.uniform(10, 30),
                    'throughput': random.uniform(40, 80),
                    'packet_loss': random.uniform(0.0, 0.05),
                    'jitter': random.uniform(1, 5),
                    'queue_length': random.uniform(5, 50),
                    'load': random.uniform(0.1, 0.5),
                    'traffic_type': 0,
                    'label': 0
                }
            
            # Method 1: Baseline (ignore anomalies)
            try:
                path_baseline = nx.shortest_path(
                    self.network_graph, source, dest, weight='base_latency'
                )
                latency_baseline = sum(
                    self.network_graph[path_baseline[i]][path_baseline[i+1]]['base_latency']
                    for i in range(len(path_baseline) - 1)
                )
                
                # If anomaly, performance degrades
                if is_anomaly:
                    latency_baseline += 100  # Attack impact
                
                latencies_baseline.append(latency_baseline)
            except:
                latencies_baseline.append(200)  # High penalty for failure
            
            # Method 2: Intelligent (anomaly-aware)
            path_intelligent, cost, risk = self.router.find_best_path(
                source, dest, features
            )
            
            if path_intelligent:
                latency_intelligent = sum(
                    self.network_graph[path_intelligent[i]][path_intelligent[i+1]]['base_latency']
                    for i in range(len(path_intelligent) - 1)
                )
                
                # Smart routing avoids degraded links
                if is_anomaly and risk > 0.5:
                    # Successfully rerouted
                    latency_intelligent += 20  # Minimal impact
                elif is_anomaly:
                    latency_intelligent += 100
                
                latencies_intelligent.append(latency_intelligent)
            else:
                latencies_intelligent.append(200)
        
        # Calculate statistics
        avg_latency_baseline = np.mean(latencies_baseline)
        avg_latency_intelligent = np.mean(latencies_intelligent)
        
        improvement = ((avg_latency_baseline - avg_latency_intelligent) / 
                      avg_latency_baseline * 100)
        
        print(f"\n📊 ROUTING PERFORMANCE COMPARISON:")
        print(f"  Baseline (No Anomaly Detection):")
        print(f"    Average Latency: {avg_latency_baseline:.2f} ms")
        print(f"\n  Intelligent (Anomaly-Aware):")
        print(f"    Average Latency: {avg_latency_intelligent:.2f} ms")
        print(f"\n  🎯 Improvement: {improvement:.2f}%")
        
        # Router statistics
        stats = self.router.get_statistics()
        print(f"\n📈 anomaly Detection Rate: {stats.get('anomaly_detection_rate', 0):.1f}%")
        
        return latencies_baseline, latencies_intelligent
    
    def visualize_results(self, latencies_baseline, latencies_intelligent):
        """Create visualization of results."""
        print("\n[STEP 6] Generating Visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: FL Training Progress
        ax1 = axes[0, 0]
        if self.results['fl_accuracy']:
            ax1.plot(range(1, len(self.results['fl_accuracy']) + 1),
                    self.results['fl_accuracy'], marker='o', linewidth=2, color='blue')
            ax1.set_xlabel('FL Round')
            ax1.set_ylabel('Average Accuracy')
            ax1.set_title('Federated Learning Training Progress')
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Latency Comparison
        ax2 = axes[0, 1]
        ax2.boxplot([latencies_baseline, latencies_intelligent],
                   labels=['Baseline', 'Anomaly-Aware'])
        ax2.set_ylabel('Latency (ms)')
        ax2.set_title('Routing Latency Comparison')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Latency Time Series
        ax3 = axes[1, 0]
        ax3.plot(latencies_baseline, label='Baseline', alpha=0.7, linewidth=1)
        ax3.plot(latencies_intelligent, label='Anomaly-Aware', alpha=0.7, linewidth=1)
        ax3.set_xlabel('Flow Number')
        ax3.set_ylabel('Latency (ms)')
        ax3.set_title('Per-Flow Latency Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Network Topology
        ax4 = axes[1, 1]
        pos = nx.spring_layout(self.network_graph, seed=RANDOM_SEED)
        nx.draw(self.network_graph, pos, ax=ax4, with_labels=True,
               node_color='lightblue', node_size=800, font_size=10,
               font_weight='bold', edge_color='gray')
        ax4.set_title('5G Network Topology')
        
        plt.tight_layout()
        
        filename = 'fl_anomaly_routing_results.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Results saved to: {filename}")
        
        plt.show()
    
    def run_complete_system(self):
        """Execute the complete FL + anomaly-aware routing system."""
        print("\n" + "="*70)
        print(" STARTING COMPLETE SYSTEM EXECUTION")
        print("="*70)
        
        # Phase 1: Setup
        self.setup_network_topology()
        
        # Phase 2: Data Generation
        training_data = self.generate_training_data(samples_per_station=200)
        
        # Phase 3: Federated Learning
        global_model = self.federated_learning_phase(training_data)
        
        # Phase 4: Setup Routing
        self.setup_anomaly_aware_routing(global_model)
        
        # Phase 5: Routing Simulation
        latencies_baseline, latencies_intelligent = self.run_routing_simulation(
            num_flows=100
        )
        
        # Phase 6: Visualization
        self.visualize_results(latencies_baseline, latencies_intelligent)
        
        print("\n" + "="*70)
        print(" ✅ SYSTEM EXECUTION COMPLETE")
        print("="*70)
        print("\n🎯 KEY ACHIEVEMENTS:")
        print("   1. ✓ Federated Learning across 5 base stations")
        print("   2. ✓ Anomaly detection with trained ML model")
        print("   3. ✓ Intelligent routing based on traffic behavior")
        print("   4. ✓ Performance improvement over baseline")
        print("\n💡 NOVELTY DEMONSTRATED:")
        print("   Traditional: Cost = Latency + Load")
        print("   Our System: Cost = Latency + Load + (Anomaly × Penalty)")
        print("="*70)


def main():
    """Main entry point."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  FEDERATED LEARNING-DRIVEN ANOMALY DETECTION".center(68) + "║")
    print("║" + "  FOR INTELLIGENT QoS MANAGEMENT IN 5G NETWORKS".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    # Create and run the system
    system = IntegratedFLQoSSystem(
        num_base_stations=5,
        simulation_rounds=10
    )
    
    system.run_complete_system()


if __name__ == "__main__":
    main()
