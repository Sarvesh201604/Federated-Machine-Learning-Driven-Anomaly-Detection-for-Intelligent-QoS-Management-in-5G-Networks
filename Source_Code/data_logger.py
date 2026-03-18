"""
Data Logger Module for 5G Network Traffic
Responsible for collecting and saving network metrics for ML training
"""

import csv
import os
from datetime import datetime
from typing import Dict, List, Optional

class DataLogger:
    """
    Handles data collection for federated learning.
    Saves traffic metrics from each base station to separate CSV files.
    """
    
    def __init__(self, output_dir='traffic_logs'):
        """
        Initialize the DataLogger.
        
        Args:
            output_dir: Directory to save traffic log files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Track which files have been initialized
        self.initialized_files = set()
        
        # Define the features we're logging
        self.feature_names = [
            'timestamp',
            'node_id',
            'latency',
            'throughput',
            'packet_loss',
            'jitter',
            'queue_length',
            'load',
            'traffic_type',
            'label'  # 0=Normal, 1=Anomaly
        ]
    
    def log_traffic(self, node_id, latency, throughput, packet_loss, jitter, 
                   queue_length=0, load=0.0, traffic_type='normal', label=0):
        """
        Log traffic data for a specific node.
        
        Args:
            node_id: Identifier for the base station/node
            latency: Packet latency in milliseconds
            throughput: Throughput in Mbps
            packet_loss: Packet loss rate (0-1)
            jitter: Jitter in milliseconds
            queue_length: Current queue length
            load: Current load (0-1)
            traffic_type: Type of traffic (video, iot, emergency, etc.)
            label: 0 for Normal, 1 for Anomaly
        """
        filename = os.path.join(self.output_dir, f'traffic_data_{node_id}.csv')
        
        # Create file with headers if it doesn't exist
        if filename not in self.initialized_files:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.feature_names)
            self.initialized_files.add(filename)
        
        # Append data
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                node_id,
                latency,
                throughput,
                packet_loss,
                jitter,
                queue_length,
                load,
                traffic_type,
                label
            ])
    
    def log_batch(self, node_id, data_list: List[Dict]):
        """
        Log multiple traffic records at once (batch logging).
        
        Args:
            node_id: Identifier for the base station/node
            data_list: List of dictionaries containing traffic metrics
        """
        filename = os.path.join(self.output_dir, f'traffic_data_{node_id}.csv')
        
        # Create file with headers if it doesn't exist
        if filename not in self.initialized_files:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.feature_names)
            self.initialized_files.add(filename)
        
        # Append all records
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            for data in data_list:
                writer.writerow([
                    data.get('timestamp', datetime.now().isoformat()),
                    node_id,
                    data.get('latency', 0),
                    data.get('throughput', 0),
                    data.get('packet_loss', 0),
                    data.get('jitter', 0),
                    data.get('queue_length', 0),
                    data.get('load', 0.0),
                    data.get('traffic_type', 'normal'),
                    data.get('label', 0)
                ])
    
    def get_log_file_path(self, node_id):
        """Get the path to a node's log file."""
        return os.path.join(self.output_dir, f'traffic_data_{node_id}.csv')
    
    def clear_logs(self, node_id=None):
        """
        Clear log files.
        
        Args:
            node_id: If provided, clear only that node's log. Otherwise clear all.
        """
        if node_id is not None:
            filename = os.path.join(self.output_dir, f'traffic_data_{node_id}.csv')
            if os.path.exists(filename):
                os.remove(filename)
                if filename in self.initialized_files:
                    self.initialized_files.remove(filename)
        else:
            # Clear all logs
            for filename in os.listdir(self.output_dir):
                if filename.startswith('traffic_data_') and filename.endswith('.csv'):
                    os.remove(os.path.join(self.output_dir, filename))
            self.initialized_files.clear()
    
    def get_stats(self, node_id=None):
        """
        Get statistics about logged data.
        
        Args:
            node_id: If provided, get stats for that node. Otherwise all nodes.
            
        Returns:
            Dictionary with statistics
        """
        stats = {}
        
        if node_id is not None:
            filename = os.path.join(self.output_dir, f'traffic_data_{node_id}.csv')
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    lines = sum(1 for line in f) - 1  # Subtract header
                    stats[node_id] = lines
        else:
            # Get stats for all nodes
            for filename in os.listdir(self.output_dir):
                if filename.startswith('traffic_data_') and filename.endswith('.csv'):
                    node_id = filename.replace('traffic_data_', '').replace('.csv', '')
                    with open(os.path.join(self.output_dir, filename), 'r') as f:
                        lines = sum(1 for line in f) - 1
                        stats[node_id] = lines
        
        return stats


# Example usage
if __name__ == "__main__":
    # Create logger
    logger = DataLogger()
    
    # Simulate logging some traffic data
    print("Logging sample traffic data...")
    
    # Node 1 - Normal traffic
    logger.log_traffic(
        node_id='enb_1',
        latency=15.5,
        throughput=50.2,
        packet_loss=0.01,
        jitter=2.3,
        queue_length=10,
        load=0.35,
        traffic_type='video',
        label=0  # Normal
    )
    
    # Node 2 - Anomalous traffic
    logger.log_traffic(
        node_id='enb_2',
        latency=250.0,  # High latency
        throughput=5.1,  # Low throughput
        packet_loss=0.35,  # High loss
        jitter=85.2,  # High jitter
        queue_length=950,  # Nearly full queue
        load=0.95,  # High load
        traffic_type='flood',
        label=1  # Anomaly
    )
    
    # Get statistics
    stats = logger.get_stats()
    print(f"\nLogged data statistics: {stats}")
    print(f"Files created in '{logger.output_dir}/' directory")
