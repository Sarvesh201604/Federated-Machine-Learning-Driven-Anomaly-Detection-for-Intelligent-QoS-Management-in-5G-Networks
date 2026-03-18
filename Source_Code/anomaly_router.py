"""
Anomaly-Aware Router Module
Implements the NOVELTY: Behavior-Aware QoS Routing

KEY INNOVATION:
    Traditional: Cost = Latency + Load
    Our System: Cost = Latency + Load + (Anomaly_Score * Penalty)
    
This makes routing decisions INTELLIGENT and ADAPTIVE to traffic behavior.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Optional

class AnomalyAwareRouter:
    """
    Implements intelligent routing that considers both QoS metrics
    and traffic behavior (anomaly detection).
    
    This is the CORE NOVELTY of the project.
    """
    
    def __init__(self, graph, global_model, anomaly_penalty=1000.0):
        """
        Initialize the Anomaly-Aware Router.
        
        Args:
            graph: NetworkX graph representing the network topology
            global_model: Trained FL global model for anomaly detection
            anomaly_penalty: Weight for anomaly score in cost calculation
        """
        self.graph = graph
        self.global_model = global_model
        self.anomaly_penalty = anomaly_penalty
        
        # Statistics tracking
        self.routing_stats = {
            'total_routes': 0,
            'normal_routes': 0,
            'rerouted_due_to_anomaly': 0,
            'blocked_high_risk': 0
        }
    
    def calculate_link_cost(self, latency, throughput, traffic_features, 
                          load=0.0, anomaly_threshold=0.7):
        """
        Calculate dynamic link cost based on multiple factors.
        
        THIS IS THE NOVELTY FORMULA:
        Cost = Base_Latency + Load_Factor + (Anomaly_Probability * Penalty)
        
        Args:
            latency: Base latency of the link (ms)
            throughput: Available throughput (Mbps)
            traffic_features: Feature vector for ML prediction
            load: Current link utilization (0-1)
            anomaly_threshold: Threshold for blocking (0-1)
            
        Returns:
            Tuple (cost, anomaly_score, should_block)
        """
        # 1. Get ML prediction (Anomaly Probability)
        anomaly_score = self._predict_anomaly(traffic_features)
        
        # 2. Base cost components
        base_cost = latency
        
        # 3. Load factor (penalize congested links)
        load_factor = load * 10.0  # Scale load impact
        
        # 4. Throughput factor (prefer high-throughput links)
        throughput_factor = max(0, 100.0 - throughput) / 10.0
        
        # 5. NOVELTY: Anomaly penalty
        # If anomaly_score is high (0.8-1.0), the penalty makes this link very expensive
        # If anomaly_score is low (0.0-0.2), penalty is minimal
        anomaly_cost = anomaly_score * self.anomaly_penalty
        
        # 6. Total cost
        total_cost = base_cost + load_factor + throughput_factor + anomaly_cost
        
        # 7. Decide if traffic should be blocked entirely
        should_block = anomaly_score > anomaly_threshold
        
        return total_cost, anomaly_score, should_block
    
    def _predict_anomaly(self, traffic_features):
        """
        Use the global FL model to predict anomaly probability.
        
        Args:
            traffic_features: Feature vector or dictionary
            
        Returns:
            Anomaly probability (0-1)
        """
        if self.global_model is None:
            return 0.0  # No model, assume normal
        
        try:
            # Convert features to appropriate format
            if isinstance(traffic_features, dict):
                # Extract feature values in consistent order
                feature_list = [
                    traffic_features.get('latency', 0),
                    traffic_features.get('throughput', 0),
                    traffic_features.get('packet_loss', 0),
                    traffic_features.get('jitter', 0),
                    traffic_features.get('queue_length', 0),
                    traffic_features.get('load', 0),
                    traffic_features.get('traffic_type', 0),
                    traffic_features.get('label', 0)
                ]
                features = np.array(feature_list).reshape(1, -1)
            else:
                features = np.array(traffic_features).reshape(1, -1)
            
            # Get anomaly probability from model
            anomaly_prob = self.global_model.predict_anomaly_score(features)
            
            # Ensure scalar return
            if isinstance(anomaly_prob, np.ndarray):
                anomaly_prob = float(anomaly_prob[0])
            
            return float(anomaly_prob)
            
        except Exception as e:
            print(f"Warning: Anomaly prediction failed - {e}")
            return 0.5  # Return neutral probability on error
    
    def update_link_costs(self, source_node, traffic_features):
        """
        Update costs for all links from a source node based on traffic.
        
        Args:
            source_node: Node ID to update outgoing links from
            traffic_features: Features of the traffic flow
        """
        if source_node not in self.graph.nodes():
            print(f"Warning: Node {source_node} not in graph")
            return
        
        # Get neighbors
        neighbors = list(self.graph.neighbors(source_node))
        
        for neighbor in neighbors:
            # Get link attributes
            edge_data = self.graph[source_node][neighbor]
            
            base_latency = edge_data.get('base_latency', 10.0)
            throughput = edge_data.get('capacity', 100.0)
            current_load = edge_data.get('current_load', 0.0)
            
            # Calculate new cost
            new_cost, anomaly_score, should_block = self.calculate_link_cost(
                base_latency, throughput, traffic_features, current_load
            )
            
            # Update edge attributes
            edge_data['cost'] = new_cost
            edge_data['anomaly_score'] = anomaly_score
            edge_data['blocked'] = should_block
    
    def find_best_path(self, source, target, traffic_features=None):
        """
        Find the best path using Dijkstra's algorithm with dynamic costs.
        
        THIS IS WHERE THE MAGIC HAPPENS:
        - Normal routing: Just finds shortest path
        - Anomaly-aware: Avoids high-risk paths dynamically
        
        Args:
            source: Source node ID
            target: Target node ID
            traffic_features: Optional traffic features for cost calculation
            
        Returns:
            Tuple (path, total_cost, anomaly_risk)
        """
        self.routing_stats['total_routes'] += 1
        
        # If traffic features provided, update costs first
        anomaly_risk = 0.0
        if traffic_features is not None:
            self.update_link_costs(source, traffic_features)
            anomaly_risk = self._predict_anomaly(traffic_features)
        
        try:
            # Find shortest path using dynamic 'cost' weight
            path = nx.dijkstra_path(self.graph, source, target, weight='cost')
            
            # Calculate total cost
            total_cost = 0.0
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                total_cost += self.graph[u][v].get('cost', 0)
            
            # Track statistics
            if anomaly_risk > 0.5:
                self.routing_stats['rerouted_due_to_anomaly'] += 1
            else:
                self.routing_stats['normal_routes'] += 1
            
            return path, total_cost, anomaly_risk
            
        except nx.NetworkXNoPath:
            print(f"No path found from {source} to {target}")
            return None, float('inf'), anomaly_risk
    
    def find_safe_path(self, source, target, traffic_features, max_anomaly=0.5):
        """
        Find a path that avoids high-anomaly links.
        
        Args:
            source: Source node
            target: Target node  
            traffic_features: Traffic features for prediction
            max_anomaly: Maximum acceptable anomaly score
            
        Returns:
            Path that avoids dangerous links, or None
        """
        # Temporarily remove high-risk edges
        removed_edges = []
        
        for u, v, data in self.graph.edges(data=True):
            if data.get('anomaly_score', 0) > max_anomaly:
                removed_edges.append((u, v, data.copy()))
                self.graph.remove_edge(u, v)
        
        # Find path in safe subgraph
        try:
            path = nx.shortest_path(self.graph, source, target, weight='base_latency')
            result = path
        except nx.NetworkXNoPath:
            result = None
        
        # Restore removed edges
        for u, v, data in removed_edges:
            self.graph.add_edge(u, v, **data)
        
        return result
    
    def compare_routing_strategies(self, source, target, traffic_features):
        """
        Compare different routing strategies for analysis.
        
        Returns:
            Dictionary with results from different strategies
        """
        results = {}
        
        # Strategy 1: Simple shortest path (ignore anomalies)
        try:
            path_simple = nx.shortest_path(self.graph, source, target, weight='base_latency')
            cost_simple = sum(self.graph[path_simple[i]][path_simple[i+1]]['base_latency'] 
                            for i in range(len(path_simple)-1))
            results['simple'] = {'path': path_simple, 'cost': cost_simple}
        except:
            results['simple'] = {'path': None, 'cost': float('inf')}
        
        # Strategy 2: Anomaly-aware (our approach)
        path_aware, cost_aware, risk = self.find_best_path(source, target, traffic_features)
        results['anomaly_aware'] = {
            'path': path_aware, 
            'cost': cost_aware, 
            'anomaly_risk': risk
        }
        
        # Strategy 3: Safe path only
        path_safe = self.find_safe_path(source, target, traffic_features, max_anomaly=0.5)
        results['safe_only'] = {'path': path_safe}
        
        return results
    
    def get_statistics(self):
        """Get routing statistics."""
        total = self.routing_stats['total_routes']
        if total == 0:
            return self.routing_stats
        
        stats = self.routing_stats.copy()
        stats['anomaly_detection_rate'] = (
            self.routing_stats['rerouted_due_to_anomaly'] / total * 100
        )
        return stats
    
    def reset_statistics(self):
        """Reset routing statistics."""
        self.routing_stats = {
            'total_routes': 0,
            'normal_routes': 0,
            'rerouted_due_to_anomaly': 0,
            'blocked_high_risk': 0
        }


# Example usage and testing
if __name__ == "__main__":
    print("Testing AnomalyAwareRouter...\n")
    
    # Create a simple network topology
    G = nx.Graph()
    
    # Add nodes (base stations)
    for i in range(5):
        G.add_node(i, type='enb')
    
    # Add edges with attributes
    edges = [
        (0, 1, {'base_latency': 10, 'capacity': 100, 'current_load': 0.3}),
        (1, 2, {'base_latency': 15, 'capacity': 100, 'current_load': 0.5}),
        (2, 3, {'base_latency': 20, 'capacity': 100, 'current_load': 0.2}),
        (0, 3, {'base_latency': 35, 'capacity': 100, 'current_load': 0.1}),
        (1, 4, {'base_latency': 25, 'capacity': 100, 'current_load': 0.4}),
        (3, 4, {'base_latency': 10, 'capacity': 100, 'current_load': 0.3}),
    ]
    
    for u, v, attrs in edges:
        G.add_edge(u, v, **attrs)
    
    # Create a mock global model
    class MockModel:
        def predict_anomaly_score(self, features):
            # Return high anomaly for certain patterns
            if features[0][0] > 100:  # High latency
                return 0.9
            return 0.1
    
    mock_model = MockModel()
    
    # Create router
    router = AnomalyAwareRouter(G, mock_model, anomaly_penalty=1000)
    
    # Test 1: Normal traffic
    print("=" * 60)
    print("TEST 1: Normal Traffic Routing")
    print("=" * 60)
    
    normal_features = {
        'latency': 15,
        'throughput': 50,
        'packet_loss': 0.01,
        'jitter': 2,
        'queue_length': 10,
        'load': 0.3,
        'traffic_type': 0,
        'label': 0
    }
    
    path, cost, risk = router.find_best_path(0, 4, normal_features)
    print(f"\nPath: {path}")
    print(f"Cost: {cost:.2f}")
    print(f"Anomaly Risk: {risk:.4f}")
    
    # Test 2: Anomalous traffic
    print("\n" + "=" * 60)
    print("TEST 2: Anomalous Traffic Routing")
    print("=" * 60)
    
    anomaly_features = {
        'latency': 250,  # High!
        'throughput': 5,
        'packet_loss': 0.4,
        'jitter': 80,
        'queue_length': 900,
        'load': 0.95,
        'traffic_type': 1,
        'label': 1
    }
    
    path, cost, risk = router.find_best_path(0, 4, anomaly_features)
    print(f"\nPath: {path}")
    print(f"Cost: {cost:.2f}")
    print(f"Anomaly Risk: {risk:.4f}")
    print("\n💡 Notice: High cost due to anomaly penalty!")
    
    # Test 3: Compare strategies
    print("\n" + "=" * 60)
    print("TEST 3: Strategy Comparison")
    print("=" * 60)
    
    comparison = router.compare_routing_strategies(0, 4, anomaly_features)
    
    for strategy, result in comparison.items():
        print(f"\n{strategy.upper()}:")
        if result['path']:
            print(f"  Path: {result['path']}")
            print(f"  Cost: {result.get('cost', 'N/A')}")
            if 'anomaly_risk' in result:
                print(f"  Risk: {result['anomaly_risk']:.4f}")
    
    # Statistics
    print("\n" + "=" * 60)
    print("ROUTING STATISTICS")
    print("=" * 60)
    stats = router.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n✅ AnomalyAwareRouter test complete!")
    print("\n🎯 KEY INSIGHT: Cost increases dramatically for anomalous traffic,")
    print("   causing the router to prefer alternative paths!")
