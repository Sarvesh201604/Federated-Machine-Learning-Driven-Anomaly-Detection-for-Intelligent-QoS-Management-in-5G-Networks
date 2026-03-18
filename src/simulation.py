import networkx as nx
import pandas as pd
import random
import time
import os
import numpy as np
import matplotlib.pyplot as plt

# Import FL components
from fl_model import create_model, train_local_node, aggregate_models

# Configuration
PROCESSED_DIR = 'processed_data'
NUM_NODES = 5
SIMULATION_STEPS = 50  # Increased for better plotting
ALPHA_PROPOSED = 50
ALPHA_BASELINE = 0

class NetworkSimulation:
    def __init__(self):
        self.graph = nx.Graph()
        self.node_data = {}  # Store loaded CSV data for each node
        self.global_model = None
        
        # 1. Initialize Network
        self.setup_topology()
        self.load_traffic_data()
        
        # 2. Train and Aggregrate Global FL Model
        self.initialize_fl_model()

    def setup_topology(self):
        """
        Sets up the network topology with 5 Base Stations (gNB).
        Edges represent links with capacity, load, latency, and anomaly_score.
        """
        print("Setting up network topology...")
        self.graph.clear()
        for i in range(NUM_NODES):
            self.graph.add_node(i, type='gNB')

        # Create a simple mesh or ring topology
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2), (2, 4)]
        
        for u, v in edges:
            self.graph.add_edge(u, v,
                                capacity=1000.0,
                                current_load=0.0,
                                base_latency=5.0, # Fixed physical latency
                                anomaly_score=0.0,
                                cost=5.0)
        
        print(f"Topology created: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges.")

    def load_traffic_data(self):
        """
        Loads the pre-processed CSV chunks for each node.
        """
        print("Loading traffic data...")
        for i in range(NUM_NODES):
            filename = os.path.join(PROCESSED_DIR, f'train_client_{i}.csv')
            if os.path.exists(filename):
                self.node_data[i] = pd.read_csv(filename)
            else:
                self.node_data[i] = pd.DataFrame()

    def initialize_fl_model(self):
        """
        Simulates the Federated Learning process to get a Global Model.
        """
        print("\n--- Initializing Federated Learning ---")
        local_updates = []
        for i in range(NUM_NODES):
            if not self.node_data[i].empty:
                weights = train_local_node(i, self.node_data[i])
                local_updates.append(weights)
        
        if local_updates:
            avg_coefs, avg_intercepts = aggregate_models(local_updates)
            self.global_model = create_model(warm_start=False)
            dummy_X = self.node_data[0].drop('label', axis=1).iloc[:1]
            dummy_y = self.node_data[0]['label'].iloc[:1]
            self.global_model.partial_fit(dummy_X, dummy_y, classes=[0, 1])
            self.global_model.coefs_ = avg_coefs
            self.global_model.intercepts_ = avg_intercepts
            print("Global FL Model successfully initialized.")
        else:
            print("Error: Could not initialize FL model.")

    def generate_traffic(self, node_id, time_step):
        if node_id not in self.node_data or self.node_data[node_id].empty:
            return None
        data_index = time_step % len(self.node_data[node_id])
        row = self.node_data[node_id].iloc[data_index]
        return row.to_dict()

    def update_network_state(self, source_node, flow_features, alpha):
        """
        Predicts anomaly score and updates link costs.
        Returns: anomaly_probability
        """
        if self.global_model is None:
            return 0.0

        features_df = pd.DataFrame([flow_features])
        if 'label' in features_df.columns:
            features_df = features_df.drop('label', axis=1)
            
        try:
            prob = self.global_model.predict_proba(features_df)[0][1]
        except Exception:
            prob = 0.0
            
        # Update Outgoing Links cost
        for neighbor in self.graph.neighbors(source_node):
            edge = self.graph[source_node][neighbor]
            edge['anomaly_score'] = prob
            
            # Recalculate Cost
            # Cost = Latency + Alpha * AnomalyProb
            # In Baseline (Alpha=0), cost is just latency, so it stays low even for attacks.
            base_lat = edge['base_latency']
            new_cost = base_lat + (alpha * prob)
            edge['cost'] = new_cost
            
        return prob

    def calculate_path_metrics(self, path, anomaly_prob):
        """
        Simulates actual network performance for a chosen path.
        If the path uses links with high anomaly_score (which we assume corresponds to attack traffic),
        the physical performance degrades.
        """
        total_latency = 0
        dropped = False
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            edge = self.graph[u][v]
            
            # Simulated Impact of Attack on Link
            # If the flow itself is an attack (high anomaly prob), AND the link blindly accepts it,
            # the link gets congested.
            # We use the edge's anomaly_score to represent the link's condition.
            # In simulation, we just updated the edge's anomaly_score with the flow's score.
            
            current_link_risk = edge.get('anomaly_score', 0)
            
            # Penalty Logic:
            # If link is carrying attack traffic (risk > 0.5), it degrades.
            actual_latency = edge['base_latency']
            
            if current_link_risk > 0.5:
                # Attack Impact!
                actual_latency += 50.0 # Huge spike
                
                # Packet Drop Probability
                if random.random() < 0.3:
                    dropped = True
            
            total_latency += actual_latency
            
        return total_latency, dropped

    def run_scenario(self, mode="Baseline"):
        """
        Runs the simulation loop for a specific mode.
        """
        print(f"\n--- Starting Scenario: {mode} ---")
        
        alpha = ALPHA_PROPOSED if mode == "Proposed" else ALPHA_BASELINE
        
        # Reset Topology for fairness
        self.setup_topology()
        
        latencies = []
        pdr_stats = {'sent': 0, 'received': 0}
        
        for t in range(SIMULATION_STEPS):
            step_latency = []
            
            for node_id in range(NUM_NODES):
                flow = self.generate_traffic(node_id, t)
                if not flow: continue
                
                dest_id = random.choice([n for n in range(NUM_NODES) if n != node_id])
                
                # 1. Update State & Get Cost
                anomaly_prob = self.update_network_state(node_id, flow, alpha)
                
                # 2. Route
                try:
                    path = nx.dijkstra_path(self.graph, source=node_id, target=dest_id, weight='cost')
                except nx.NetworkXNoPath:
                    continue
                
                # 3. Measure Performance
                latency, dropped = self.calculate_path_metrics(path, anomaly_prob)
                
                pdr_stats['sent'] += 1
                if not dropped:
                    pdr_stats['received'] += 1
                    step_latency.append(latency)
                
            # Log average latency for this time step
            avg_lat = np.mean(step_latency) if step_latency else 0
            latencies.append(avg_lat)
            
            # Optional console log
            if t % 10 == 0:
                print(f"  Step {t}: Avg Latency={avg_lat:.2f}ms")
                
        pdr = (pdr_stats['received'] / pdr_stats['sent']) * 100 if pdr_stats['sent'] > 0 else 0
        print(f"[{mode}] Completed. PDR: {pdr:.2f}%, Avg Latency: {np.mean(latencies):.2f}ms")
        
        return latencies, pdr

    def run_evaluation(self):
        # Run Baseline
        lat_baseline, pdr_baseline = self.run_scenario("Baseline")
        
        # Run Proposed
        lat_proposed, pdr_proposed = self.run_scenario("Proposed")
        
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(lat_baseline, label=f'Baseline (PDR={pdr_baseline:.1f}%)', color='red', linestyle='--')
        plt.plot(lat_proposed, label=f'Proposed FL-Routing (PDR={pdr_proposed:.1f}%)', color='green', linewidth=2)
        
        plt.xlabel('Simulation Time Steps')
        plt.ylabel('Average Packet Latency (ms)')
        plt.title('Performance Comparison: Standard vs Anomaly-Aware Routing')
        plt.legend()
        plt.grid(True)
        
        filename = 'evaluation_results.png'
        plt.savefig(filename)
        print(f"\nEvaluation graph saved as {filename}")

if __name__ == "__main__":
    sim = NetworkSimulation()
    sim.run_evaluation()
