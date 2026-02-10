import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import networkx as nx
from enum import Enum
import random
import math
import time
import argparse
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import os
import csv
from matplotlib.animation import FuncAnimation
from IPython.display import clear_output

# Define constants
SIMULATION_SEED = 42
np.random.seed(SIMULATION_SEED)
random.seed(SIMULATION_SEED)

# Set up fancy plots
plt.style.use('ggplot')
sns.set(style="whitegrid")

class TrafficType(Enum):
    VIDEO_STREAMING = 1  # High bandwidth priority
    MESSAGING = 2        # Low latency priority
    IOT_SENSOR = 3       # Reliability priority
    EMERGENCY = 4        # Highest priority

class NodeType(Enum):
    UE = 1       # User Equipment (smartphones)
    ENB = 2      # eNodeB (base stations)
    IOT = 3      # IoT devices
    CORE = 4     # Core network components

@dataclass
class QoSRequirements:
    """Quality of Service requirements for different traffic types"""
    min_bandwidth: float  # Mbps
    max_latency: float    # ms
    reliability: float    # 0-1 (packet delivery ratio)
    priority: int         # 1-10 (10 highest)

# Define QoS requirements for different traffic types
QOS_REQUIREMENTS = {
    TrafficType.VIDEO_STREAMING: QoSRequirements(min_bandwidth=5.0, max_latency=200.0, reliability=0.95, priority=7),
    TrafficType.MESSAGING: QoSRequirements(min_bandwidth=0.1, max_latency=50.0, reliability=0.99, priority=5),
    TrafficType.IOT_SENSOR: QoSRequirements(min_bandwidth=0.05, max_latency=100.0, reliability=0.999, priority=6),
    TrafficType.EMERGENCY: QoSRequirements(min_bandwidth=1.0, max_latency=20.0, reliability=0.9999, priority=10)
}

@dataclass
class Position:
    """2D position"""
    x: float
    y: float
    
    def distance_to(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

class Packet:
    """Network packet class"""
    next_id = 0
    
    def __init__(self, source_id, dest_id, size, traffic_type, creation_time):
        self.id = Packet.next_id
        Packet.next_id += 1
        self.source_id = source_id
        self.dest_id = dest_id
        self.size = size  # In KB
        self.traffic_type = traffic_type
        self.creation_time = creation_time
        self.delivery_time = None
        self.current_node = source_id
        self.path = [source_id]
        self.dropped = False
    
    @property
    def latency(self):
        if self.delivery_time is None:
            return None
        return self.delivery_time - self.creation_time

class Node:
    """Base node class for network elements"""
    def __init__(self, node_id, node_type, position, capacity=100):
        self.id = node_id
        self.type = node_type
        self.position = position
        self.capacity = capacity  # Mbps
        self.current_load = 0.0
        self.queue = []
        self.connected_nodes = []
        self.buffer_size = 1000  # Max packets in queue
        self.stats = {
            'packets_received': 0,
            'packets_sent': 0,
            'packets_dropped': 0,
            'current_load_history': []
        }
    
    def connect_to(self, node):
        if node not in self.connected_nodes:
            self.connected_nodes.append(node)
    
    def receive_packet(self, packet, current_time):
        self.stats['packets_received'] += 1
        
        # Update packet path
        packet.current_node = self.id
        packet.path.append(self.id)
        
        # If this node is the destination, mark as delivered
        if packet.dest_id == self.id:
            packet.delivery_time = current_time
            return True
        
        # If queue is full, drop packet based on priority
        if len(self.queue) >= self.buffer_size:
            if self._should_drop_packet(packet):
                packet.dropped = True
                self.stats['packets_dropped'] += 1
                return False
        
        # Add to queue
        self.queue.append(packet)
        return True
    
    def _should_drop_packet(self, new_packet):
        """Determines if a new packet should be dropped when buffer is full"""
        # If there's a lower priority packet in queue, drop that instead
        lowest_priority = QOS_REQUIREMENTS[new_packet.traffic_type].priority
        lowest_idx = -1
        
        for i, p in enumerate(self.queue):
            p_priority = QOS_REQUIREMENTS[p.traffic_type].priority
            if p_priority < lowest_priority:
                lowest_priority = p_priority
                lowest_idx = i
        
        # If we found a lower priority packet, remove it and accept the new one
        if lowest_idx >= 0:
            dropped_packet = self.queue.pop(lowest_idx)
            dropped_packet.dropped = True
            self.stats['packets_dropped'] += 1
            return False
        
        # Otherwise, drop the new packet
        return True
    
    def process_queue(self, current_time, network):
        """Process packets in the queue"""
        if not self.queue:
            return
        
        # Calculate how many packets we can process based on capacity
        packets_to_process = min(len(self.queue), max(1, int(self.capacity / 10)))
        
        for _ in range(packets_to_process):
            if not self.queue:
                break
                
            # Get next packet using priority scheduling
            packet_idx = self._get_next_packet_idx()
            if packet_idx == -1:
                break
                
            packet = self.queue.pop(packet_idx)
            
            # Forward packet to next hop
            next_node = self._get_next_hop(packet, network)
            if next_node:
                next_node.receive_packet(packet, current_time)
                self.stats['packets_sent'] += 1
    
    def _get_next_packet_idx(self):
        """Get index of next packet to process based on priority"""
        if not self.queue:
            return -1
            
        highest_priority = -1
        selected_idx = -1
        
        for i, packet in enumerate(self.queue):
            priority = QOS_REQUIREMENTS[packet.traffic_type].priority
            if priority > highest_priority:
                highest_priority = priority
                selected_idx = i
        
        return selected_idx
    
    def _get_next_hop(self, packet, network):
        """Determine next hop for packet routing"""
        # Default implementation - override in subclasses
        if not self.connected_nodes:
            return None
            
        # Simple shortest path routing based on destination
        destination_node = network.get_node_by_id(packet.dest_id)
        if not destination_node:
            return None
            
        # Find the connected node that is closest to the destination
        min_distance = float('inf')
        next_hop = None
        
        for node in self.connected_nodes:
            distance = node.position.distance_to(destination_node.position)
            if distance < min_distance:
                min_distance = distance
                next_hop = node
                
        return next_hop
    
    def update_load(self, current_time):
        """Update the current load of the node"""
        # Calculate load as a function of queue size and processing capacity
        self.current_load = min(1.0, len(self.queue) / self.buffer_size)
        self.stats['current_load_history'].append((current_time, self.current_load))

class EnbNode(Node):
    """eNodeB (Base Station) node"""
    def __init__(self, node_id, position, capacity=500, coverage_radius=500):
        super().__init__(node_id, NodeType.ENB, position, capacity)
        self.coverage_radius = coverage_radius  # meters
        self.buffer_size = 10000  # Larger buffer for base stations
        self.connected_ue_nodes = []
        self.failed = False
    
    def is_in_coverage(self, node):
        """Check if a node is within coverage radius"""
        return self.position.distance_to(node.position) <= self.coverage_radius
    
    def _get_next_hop(self, packet, network):
        """Override to implement adaptive QoS routing"""
        if self.failed:
            return None
            
        # Get destination node
        dest_node = network.get_node_by_id(packet.dest_id)
        if not dest_node:
            return None
            
        # If destination is directly connected, send to it
        if dest_node in self.connected_nodes:
            return dest_node
            
        # Get the QoS requirements for this traffic type
        qos_req = QOS_REQUIREMENTS[packet.traffic_type]
        
        # For adaptive routing, we'll select the next hop based on:
        # 1. QoS requirements of the traffic type
        # 2. Current load of neighboring nodes
        # 3. Distance to destination
        
        best_node = None
        best_score = float('-inf')
        
        for node in self.connected_nodes:
            if node.type == NodeType.ENB and node.failed:
                continue
                
            # Calculate distance factor (normalized)
            if dest_node.position is not None:
                distance = node.position.distance_to(dest_node.position)
                distance_factor = 1.0 - min(1.0, distance / 2000.0)  # Normalize to 0-1
            else:
                distance_factor = 0.5  # Default if no position
                
            # Calculate load factor (lower is better)
            load_factor = 1.0 - node.current_load
            
            # Calculate score based on traffic type and QoS requirements
            if packet.traffic_type == TrafficType.VIDEO_STREAMING:
                # Video prioritizes bandwidth
                score = (0.7 * load_factor) + (0.3 * distance_factor)
            elif packet.traffic_type == TrafficType.MESSAGING:
                # Messaging prioritizes latency
                score = (0.3 * load_factor) + (0.7 * distance_factor)
            elif packet.traffic_type == TrafficType.IOT_SENSOR:
                # IoT prioritizes reliability
                score = (0.5 * load_factor) + (0.5 * distance_factor)
            elif packet.traffic_type == TrafficType.EMERGENCY:
                # Emergency prioritizes latency over everything
                score = (0.2 * load_factor) + (0.8 * distance_factor)
            else:
                score = (0.5 * load_factor) + (0.5 * distance_factor)
                
            # Adjust score by priority
            score *= (0.5 + 0.5 * qos_req.priority / 10.0)
            
            if score > best_score:
                best_score = score
                best_node = node
                
        return best_node
    
    def fail(self):
        """Simulate a base station failure"""
        self.failed = True
        self.current_load = 1.0  # Full load
        
    def recover(self):
        """Recover from failure"""
        self.failed = False
        # Calculate load based on queue
        self.update_load(0)

class UeNode(Node):
    """User Equipment node (smartphone, etc.)"""
    def __init__(self, node_id, position, mobility_model="random_walk"):
        super().__init__(node_id, NodeType.UE, position, capacity=10)
        self.serving_enb = None
        self.mobility_model = mobility_model
        self.speed = random.uniform(0.5, 2.0)  # m/s
        self.direction = random.uniform(0, 2 * math.pi)
        self.buffer_size = 100  # Smaller buffer for UE
        self.traffic_pattern = {
            TrafficType.VIDEO_STREAMING: random.random() < 0.3,  # 30% chance
            TrafficType.MESSAGING: random.random() < 0.7,        # 70% chance
            TrafficType.EMERGENCY: random.random() < 0.05        # 5% chance
        }
    
    def move(self, stadium_radius, time_step):
        """Move the UE node according to mobility model"""
        if self.mobility_model == "random_walk":
            # Random walk model - change direction occasionally
            if random.random() < 0.1:
                self.direction = random.uniform(0, 2 * math.pi)
                
            # Calculate new position
            dx = self.speed * time_step * math.cos(self.direction)
            dy = self.speed * time_step * math.sin(self.direction)
            
            new_x = self.position.x + dx
            new_y = self.position.y + dy
            
            # Check if new position is within stadium bounds
            distance_from_center = math.sqrt(new_x**2 + new_y**2)
            if distance_from_center > stadium_radius:
                # Reflect back if hitting boundary
                self.direction = (self.direction + math.pi) % (2 * math.pi)
                new_x = self.position.x
                new_y = self.position.y
            
            self.position = Position(new_x, new_y)
    
    def connect_to_enb(self, enb_node):
        """Connect to an eNodeB"""
        self.serving_enb = enb_node
        self.connected_nodes = [enb_node]
        enb_node.connect_to(self)
        
    def generate_traffic(self, current_time, all_nodes, core_nodes):
        """Generate network traffic based on traffic pattern"""
        packets = []
        
        # Determine which traffic types to generate
        for traffic_type, is_active in self.traffic_pattern.items():
            if not is_active:
                continue
                
            # Probabilistic packet generation
            if traffic_type == TrafficType.VIDEO_STREAMING and random.random() < 0.3:
                # Video traffic: larger packets
                size = random.uniform(500, 2000)  # KB
                # Send to a random core node (representing video server)
                dest_node = random.choice(core_nodes)
                packets.append(Packet(self.id, dest_node.id, size, traffic_type, current_time))
                
            elif traffic_type == TrafficType.MESSAGING and random.random() < 0.1:
                # Messaging traffic: smaller packets
                size = random.uniform(1, 50)  # KB
                # Send to another random UE node
                ue_nodes = [n for n in all_nodes if n.type == NodeType.UE and n.id != self.id]
                if ue_nodes:
                    dest_node = random.choice(ue_nodes)
                    packets.append(Packet(self.id, dest_node.id, size, traffic_type, current_time))
                    
            elif traffic_type == TrafficType.EMERGENCY and random.random() < 0.02:
                # Emergency traffic: medium packets
                size = random.uniform(100, 300)  # KB
                # Send to a random core node (representing emergency services)
                dest_node = random.choice(core_nodes)
                packets.append(Packet(self.id, dest_node.id, size, traffic_type, current_time))
                
        return packets

class IotNode(Node):
    """IoT sensor node"""
    def __init__(self, node_id, position, sensor_type="crowd_density"):
        super().__init__(node_id, NodeType.IOT, position, capacity=1)
        self.sensor_type = sensor_type
        self.serving_enb = None
        self.buffer_size = 50  # Small buffer for IoT
        self.reporting_interval = random.uniform(1.0, 5.0)  # seconds
        self.last_report_time = 0
    
    def connect_to_enb(self, enb_node):
        """Connect to an eNodeB"""
        self.serving_enb = enb_node
        self.connected_nodes = [enb_node]
        enb_node.connect_to(self)
    
    def generate_traffic(self, current_time, all_nodes, core_nodes):
        """Generate IoT sensor data"""
        packets = []
        
        # Check if it's time to report
        if current_time - self.last_report_time >= self.reporting_interval:
            # Generate sensor data packet
            size = random.uniform(0.5, 5.0)  # KB
            # Send to a random core node (representing IoT platform)
            dest_node = random.choice(core_nodes)
            packets.append(Packet(self.id, dest_node.id, size, TrafficType.IOT_SENSOR, current_time))
            self.last_report_time = current_time
            
        return packets

class CoreNode(Node):
    """Core network node"""
    def __init__(self, node_id, position):
        super().__init__(node_id, NodeType.CORE, position, capacity=1000)
        self.buffer_size = 100000  # Very large buffer
    
    def _get_next_hop(self, packet, network):
        """Core nodes have direct connections to all eNodeBs"""
        dest_node = network.get_node_by_id(packet.dest_id)
        if not dest_node:
            return None
            
        # If destination is directly connected, send to it
        if dest_node in self.connected_nodes:
            return dest_node
            
        # If destination is a UE or IoT, find its serving eNodeB
        if dest_node.type in [NodeType.UE, NodeType.IOT]:
            return dest_node.serving_enb
            
        # Otherwise, find closest eNodeB to destination
        enb_nodes = [n for n in self.connected_nodes if n.type == NodeType.ENB and not n.failed]
        if not enb_nodes:
            return None
            
        return min(enb_nodes, key=lambda n: n.position.distance_to(dest_node.position))

class Network:
    """5G network simulation"""
    def __init__(self, stadium_radius=1000, use_adaptive_qos=True):
        self.nodes = []
        self.packets = []
        self.delivered_packets = []
        self.dropped_packets = []
        self.current_time = 0.0
        self.stadium_radius = stadium_radius
        self.use_adaptive_qos = use_adaptive_qos
        self.stats = {
            'total_packets_generated': 0,
            'total_packets_delivered': 0,
            'total_packets_dropped': 0,
            'avg_latency': 0.0,
            'throughput': 0.0,
            'reliability': 0.0
        }
        self.stats_history = []
        
        # Initialize network graph for visualization
        self.graph = nx.Graph()
        
    def add_node(self, node):
        """Add a node to the network"""
        self.nodes.append(node)
        self.graph.add_node(node.id, 
                          pos=(node.position.x, node.position.y),
                          node_type=node.type)
    
    def get_node_by_id(self, node_id):
        """Get a node by its ID"""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None
    
    def connect_nodes(self, node1, node2):
        """Connect two nodes"""
        node1.connect_to(node2)
        node2.connect_to(node1)
        self.graph.add_edge(node1.id, node2.id)
    
    def setup_topology(self, num_ue_nodes, num_enb_nodes, num_iot_nodes):
        """Set up the network topology"""
        # Create core nodes
        core1 = CoreNode(1000, Position(0, 0))
        core2 = CoreNode(1001, Position(100, 100))
        self.add_node(core1)
        self.add_node(core2)
        self.connect_nodes(core1, core2)
        
        # Create eNodeB nodes (base stations) in a circular pattern
        enb_nodes = []
        for i in range(num_enb_nodes):
            angle = 2 * math.pi * i / num_enb_nodes
            r = self.stadium_radius * 0.7  # Place base stations at 70% of radius
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            
            enb = EnbNode(2000 + i, Position(x, y))
            self.add_node(enb)
            enb_nodes.append(enb)
            
            # Connect to core nodes
            self.connect_nodes(enb, core1)
            self.connect_nodes(enb, core2)
            
            # Connect to adjacent eNodeBs
            if i > 0:
                self.connect_nodes(enb, enb_nodes[i-1])
            if i == num_enb_nodes - 1:
                self.connect_nodes(enb, enb_nodes[0])  # Close the circle
        
        # Create UE nodes (smartphones) randomly in the stadium
        for i in range(num_ue_nodes):
            # Random position within stadium
            angle = random.uniform(0, 2 * math.pi)
            r = random.uniform(0, self.stadium_radius * 0.9)
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            
            ue = UeNode(3000 + i, Position(x, y))
            self.add_node(ue)
            
            # Connect to nearest eNodeB
            nearest_enb = min(enb_nodes, key=lambda n: n.position.distance_to(ue.position))
            if nearest_enb.is_in_coverage(ue):
                ue.connect_to_enb(nearest_enb)
                nearest_enb.connected_ue_nodes.append(ue)
        
        # Create IoT nodes
        for i in range(num_iot_nodes):
            # Random position within stadium
            angle = random.uniform(0, 2 * math.pi)
            r = random.uniform(0, self.stadium_radius * 0.95)
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            
            iot = IotNode(4000 + i, Position(x, y))
            self.add_node(iot)
            
            # Connect to nearest eNodeB
            nearest_enb = min(enb_nodes, key=lambda n: n.position.distance_to(iot.position))
            if nearest_enb.is_in_coverage(iot):
                iot.connect_to_enb(nearest_enb)
    
    def run_simulation_step(self, time_step=0.1):
        """Run one step of the simulation"""
        self.current_time += time_step
        
        # 1. Move UE nodes
        ue_nodes = [n for n in self.nodes if n.type == NodeType.UE]
        for ue in ue_nodes:
            old_position = Position(ue.position.x, ue.position.y)
            ue.move(self.stadium_radius, time_step)
            
            # Check if UE needs to handover to another eNodeB
            if ue.serving_enb:
                if not ue.serving_enb.is_in_coverage(ue) or ue.serving_enb.failed:
                    # Find new eNodeB
                    enb_nodes = [n for n in self.nodes if n.type == NodeType.ENB and not n.failed]
                    in_coverage_enbs = [e for e in enb_nodes if e.is_in_coverage(ue)]
                    
                    if in_coverage_enbs:
                        # Handover to nearest eNodeB
                        new_enb = min(in_coverage_enbs, key=lambda e: e.position.distance_to(ue.position))
                        
                        # Remove from old eNodeB
                        if ue in ue.serving_enb.connected_ue_nodes:
                            ue.serving_enb.connected_ue_nodes.remove(ue)
                        
                        # Connect to new eNodeB
                        ue.connect_to_enb(new_enb)
                        new_enb.connected_ue_nodes.append(ue)
        
        # 2. Generate traffic
        core_nodes = [n for n in self.nodes if n.type == NodeType.CORE]
        
        # UE traffic
        for ue in ue_nodes:
            if ue.serving_enb and not ue.serving_enb.failed:
                new_packets = ue.generate_traffic(self.current_time, self.nodes, core_nodes)
                for packet in new_packets:
                    ue.receive_packet(packet, self.current_time)
                    self.packets.append(packet)
                    self.stats['total_packets_generated'] += 1
        
        # IoT traffic
        iot_nodes = [n for n in self.nodes if n.type == NodeType.IOT]
        for iot in iot_nodes:
            if iot.serving_enb and not iot.serving_enb.failed:
                new_packets = iot.generate_traffic(self.current_time, self.nodes, core_nodes)
                for packet in new_packets:
                    iot.receive_packet(packet, self.current_time)
                    self.packets.append(packet)
                    self.stats['total_packets_generated'] += 1
        
        # 3. Process packets at each node
        for node in self.nodes:
            node.process_queue(self.current_time, self)
            node.update_load(self.current_time)
        
        # 4. Update packet status
        to_remove = []
        for packet in self.packets:
            if packet.delivery_time is not None:
                to_remove.append(packet)
                self.delivered_packets.append(packet)
                self.stats['total_packets_delivered'] += 1
            elif packet.dropped:
                to_remove.append(packet)
                self.dropped_packets.append(packet)
                self.stats['total_packets_dropped'] += 1
        
        for packet in to_remove:
            self.packets.remove(packet)
        
        # 5. Update network statistics
        self._update_stats()
    
    def _update_stats(self):
        """Update network statistics"""
        # Calculate average latency
        if self.delivered_packets:
            self.stats['avg_latency'] = sum(p.latency for p in self.delivered_packets) / len(self.delivered_packets)
        else:
            self.stats['avg_latency'] = 0.0
        
        # Calculate throughput (delivered packets per second)
        window_size = 5.0  # 5 second window
        recent_packets = [p for p in self.delivered_packets if p.delivery_time > self.current_time - window_size]
        self.stats['throughput'] = len(recent_packets) / window_size if window_size > 0 else 0
        
        # Calculate reliability (delivery ratio)
        total_completed = len(self.delivered_packets) + len(self.dropped_packets)
        if total_completed > 0:
            self.stats['reliability'] = len(self.delivered_packets) / total_completed
        else:
            self.stats['reliability'] = 1.0
            
        # Save stats history
        self.stats_history.append({
            'time': self.current_time,
            'avg_latency': self.stats['avg_latency'],
            'throughput': self.stats['throughput'],
            'reliability': self.stats['reliability'],
            'total_packets_delivered': self.stats['total_packets_delivered'],
            'total_packets_dropped': self.stats['total_packets_dropped']
        })
    
    def run_scenario(self, scenario_num, sim_time=30):
        """Run a specific scenario"""
        print(f"Running Scenario {scenario_num}...")
        
        # Reset network
        self.packets = []
        self.delivered_packets = []
        self.dropped_packets = []
        
        # Reset stats
        self.stats = {
            'total_packets_generated': 0,
            'total_packets_delivered': 0,
            'total_packets_dropped': 0,
            'avg_latency': 0.0,
            'throughput': 0.0,
            'reliability': 0.0
        }
        self.stats_history = []
        
        # Reset node states
        for node in self.nodes:
            node.queue = []
            node.current_load = 0.0
            if isinstance(node, EnbNode):
                node.failed = False
            node.stats = {
                'packets_received': 0,
                'packets_sent': 0,
                'packets_dropped': 0,
                'current_load_history': []
            }
        
        # Time step
        time_step = 0.1
        
        # Run simulation for specified time
        for t in range(int(sim_time / time_step)):
            # Apply scenario-specific conditions
            if scenario_num == 2:  # Base Station Overload
                if t * time_step > 5.0 and t * time_step < 20.0:
                    # Overload one base station
                    enb_nodes = [n for n in self.nodes if n.type == NodeType.ENB]
                    if enb_nodes:
                        overloaded_enb = enb_nodes[0]
                        # Generate excessive video traffic
                        for _ in range(50):
                            if random.random() < 0.3:
                                ue_nodes = overloaded_enb.connected_ue_nodes
                                if ue_nodes:
                                    ue = random.choice(ue_nodes)
                                    core_nodes = [n for n in self.nodes if n.type == NodeType.CORE]
                                    dest_node = random.choice(core_nodes)
                                    packet = Packet(ue.id, dest_node.id, random.uniform(1000, 3000), 
                                                  TrafficType.VIDEO_STREAMING, self.current_time)
                                    ue.receive_packet(packet, self.current_time)
                                    self.packets.append(packet)
                                    self.stats['total_packets_generated'] += 1
            
            elif scenario_num == 3:  # Emergency Traffic Priority
                if t * time_step > 10.0 and t * time_step < 15.0:
                    # Generate emergency traffic
                    ue_nodes = [n for n in self.nodes if n.type == NodeType.UE]
                    core_nodes = [n for n in self.nodes if n.type == NodeType.CORE]
                    
                    for _ in range(20):
                        if ue_nodes and core_nodes:
                            ue = random.choice(ue_nodes)
                            dest_node = random.choice(core_nodes)
                            packet = Packet(ue.id, dest_node.id, random.uniform(100, 300),
                                           TrafficType.EMERGENCY, self.current_time)
                            if ue.serving_enb and not ue.serving_enb.failed:
                                ue.receive_packet(packet, self.current_time)
                                self.packets.append(packet)
                                self.stats['total_packets_generated'] += 1
            
            elif scenario_num == 4:  # IoT Sensor Data Surge
                if t * time_step > 12.0 and t * time_step < 18.0:
                    # Generate surge in IoT traffic
                    iot_nodes = [n for n in self.nodes if n.type == NodeType.IOT]
                    core_nodes = [n for n in self.nodes if n.type == NodeType.CORE]
                    
                    for iot in iot_nodes:
                        if random.random() < 0.5 and iot.serving_enb and not iot.serving_enb.failed:
                            if core_nodes:
                                dest_node = random.choice(core_nodes)
                                packet = Packet(iot.id, dest_node.id, random.uniform(1, 10),
                                               TrafficType.IOT_SENSOR, self.current_time)
                                iot.receive_packet(packet, self.current_time)
                                self.packets.append(packet)
                                self.stats['total_packets_generated'] += 1
            
            elif scenario_num == 5:  # Device Mobility
                if t * time_step > 5.0:
                    # Increase UE movement speed
                    ue_nodes = [n for n in self.nodes if n.type == NodeType.UE]
                    for ue in ue_nodes:
                        ue.speed = random.uniform(5.0, 10.0)  # Faster movement
            
            elif scenario_num == 6:  # Base Station Failure
                if t * time_step > 10.0 and t * time_step < 20.0:
                    # Fail a random base station
                    if t * time_step == 10.1:  # Only do this once
                        enb_nodes = [n for n in self.nodes if n.type == NodeType.ENB and not n.failed]
                        if enb_nodes:
                            failed_enb = random.choice(enb_nodes)
                            failed_enb.fail()
                            print(f"Base station {failed_enb.id} has failed at time {self.current_time}")
                
                if t * time_step >= 20.0:
                    # Recover all failed base stations
                    enb_nodes = [n for n in self.nodes if n.type == NodeType.ENB and n.failed]
                    for enb in enb_nodes:
                        enb.recover()
                        print(f"Base station {enb.id} has recovered at time {self.current_time}")
            
            # Run one simulation step
            self.run_simulation_step(time_step)
            
            # Print progress
            if t % 50 == 0:
                print(f"Simulation time: {self.current_time:.1f}s / {sim_time:.1f}s")
        
        print(f"Scenario {scenario_num} completed.")
        return self.stats_history
    
    def visualize_network(self):
        """Visualize the network topology"""
        plt.figure(figsize=(12, 10))
        
        # Get positions for all nodes
        pos = {node.id: (node.position.x, node.position.y) for node in self.nodes}
        
        # Draw nodes by type
        node_colors = {
            NodeType.UE: 'blue',
            NodeType.ENB: 'red',
            NodeType.IOT: 'green',
            NodeType.CORE: 'purple'
        }
        
        node_sizes = {
            NodeType.UE: 50,
            NodeType.ENB: 200,
            NodeType.IOT: 30,
            NodeType.CORE: 300
        }
        
        for node_type in NodeType:
            node_list = [n.id for n in self.nodes if n.type == node_type]
            if node_list:
                nx.draw_networkx_nodes(self.graph, pos, 
                                     nodelist=node_list,
                                     node_color=node_colors[node_type],
                                     node_size=node_sizes[node_type],
                                     label=node_type.name)
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, alpha=0.5)
        
        # Draw labels for eNodeB and Core nodes only
        labels = {n.id: f"{n.id}" for n in self.nodes if n.type in [NodeType.ENB, NodeType.CORE]}
        nx.draw_networkx_labels(self.graph, pos, labels=labels, font_size=8)
        
        # Draw circle for stadium boundary
        stadium_circle = plt.Circle((0, 0), self.stadium_radius, fill=False, color='black', linestyle='--')
        plt.gca().add_patch(stadium_circle)
        
        plt.title("5G Network Topology for Event Management")
        plt.legend()
        plt.axis('equal')
        plt.tight_layout()
        
        # Save figure
        plt.savefig("network_topology.png", dpi=300)
        plt.close()
    
    def visualize_results(self, scenario_histories):
        """Visualize simulation results across scenarios"""
        scenarios = list(scenario_histories.keys())
        
        # Plot latency across scenarios
        plt.figure(figsize=(15, 10))
        
        # 1. Average Latency Over Time
        plt.subplot(2, 2, 1)
        for scenario in scenarios:
            data = scenario_histories[scenario]
            times = [entry['time'] for entry in data]
            latencies = [entry['avg_latency'] for entry in data]
            plt.plot(times, latencies, label=f"Scenario {scenario}")
        
        plt.title("Average Latency Over Time")
        plt.xlabel("Simulation Time (s)")
        plt.ylabel("Latency (s)")
        plt.legend()
        plt.grid(True)
        
        # 2. Throughput Over Time
        plt.subplot(2, 2, 2)
        for scenario in scenarios:
            data = scenario_histories[scenario]
            times = [entry['time'] for entry in data]
            throughputs = [entry['throughput'] for entry in data]
            plt.plot(times, throughputs, label=f"Scenario {scenario}")
        
        plt.title("Network Throughput Over Time")
        plt.xlabel("Simulation Time (s)")
        plt.ylabel("Throughput (packets/s)")
        plt.legend()
        plt.grid(True)
        
        # 3. Reliability Over Time
        plt.subplot(2, 2, 3)
        for scenario in scenarios:
            data = scenario_histories[scenario]
            times = [entry['time'] for entry in data]
            reliability = [entry['reliability'] for entry in data]
            plt.plot(times, reliability, label=f"Scenario {scenario}")
        
        plt.title("Network Reliability Over Time")
        plt.xlabel("Simulation Time (s)")
        plt.ylabel("Reliability (delivery ratio)")
        plt.legend()
        plt.grid(True)
        
        # 4. Cumulative Packet Delivery
        plt.subplot(2, 2, 4)
        for scenario in scenarios:
            data = scenario_histories[scenario]
            times = [entry['time'] for entry in data]
            delivered = [entry['total_packets_delivered'] for entry in data]
            plt.plot(times, delivered, label=f"Scenario {scenario}")
        
        plt.title("Cumulative Packet Delivery")
        plt.xlabel("Simulation Time (s)")
        plt.ylabel("Total Packets Delivered")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("scenario_comparison.png", dpi=300)
        plt.close()
        
        # Create a separate fig
        # 
        # ure for traffic type performance
        self.visualize_traffic_type_performance()
    
    def visualize_traffic_type_performance(self):
        """Visualize performance metrics by traffic type"""
        # Analyze performance by traffic type
        traffic_stats = {}
        
        for traffic_type in TrafficType:
            delivered = [p for p in self.delivered_packets if p.traffic_type == traffic_type]
            dropped = [p for p in self.dropped_packets if p.traffic_type == traffic_type]
            
            if delivered or dropped:
                total = len(delivered) + len(dropped)
                delivery_ratio = len(delivered) / total if total > 0 else 0
                avg_latency = sum(p.latency for p in delivered) / len(delivered) if delivered else 0
                
                traffic_stats[traffic_type.name] = {
                    'delivery_ratio': delivery_ratio,
                    'avg_latency': avg_latency,
                    'total_packets': total
                }
        
        # Create bar charts
        plt.figure(figsize=(15, 10))
        
        # Delivery ratio by traffic type
        plt.subplot(2, 2, 1)
        traffic_types = list(traffic_stats.keys())
        delivery_ratios = [traffic_stats[t]['delivery_ratio'] for t in traffic_types]
        
        bars = plt.bar(traffic_types, delivery_ratios)
        plt.title("Delivery Ratio by Traffic Type")
        plt.ylabel("Delivery Ratio")
        plt.ylim(0, 1.1)
        
        # Add value labels above bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'{height:.2f}', ha='center', va='bottom')
        
        # Average latency by traffic type
        plt.subplot(2, 2, 2)
        latencies = [traffic_stats[t]['avg_latency'] for t in traffic_types]
        
        bars = plt.bar(traffic_types, latencies)
        plt.title("Average Latency by Traffic Type")
        plt.ylabel("Latency (s)")
        
        # Add value labels above bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}s', ha='center', va='bottom')
        
        # Total packets by traffic type
        plt.subplot(2, 2, 3)
        packets = [traffic_stats[t]['total_packets'] for t in traffic_types]
        
        bars = plt.bar(traffic_types, packets)
        plt.title("Total Packets by Traffic Type")
        plt.ylabel("Number of Packets")
        
        # Add value labels above bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig("traffic_type_performance.png", dpi=300)
        plt.close()
    
    def visualize_node_loads(self):
        """Visualize load on network nodes over time"""
        plt.figure(figsize=(15, 10))
        
        # Plot load on eNodeB nodes
        enb_nodes = [n for n in self.nodes if n.type == NodeType.ENB]
        
        for node in enb_nodes:
            times = [t for t, _ in node.stats['current_load_history']]
            loads = [load for _, load in node.stats['current_load_history']]
            plt.plot(times, loads, label=f"eNodeB {node.id}")
        
        plt.title("eNodeB Load Over Time")
        plt.xlabel("Simulation Time (s)")
        plt.ylabel("Load (0-1)")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("enb_loads.png", dpi=300)
        plt.close()
    
    def generate_report(self, scenario_histories):
        """Generate a CSV report with simulation results"""
        # Create results directory if it doesn't exist
        if not os.path.exists("results"):
            os.makedirs("results")
        
        # Generate summary report
        with open("results/simulation_summary.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Scenario", "Avg Latency (s)", "Throughput (pkts/s)", "Reliability", 
                           "Total Delivered", "Total Dropped"])
            
            for scenario, history in scenario_histories.items():
                # Get final stats
                final_stats = history[-1]
                writer.writerow([
                    scenario,
                    round(final_stats['avg_latency'], 4),
                    round(final_stats['throughput'], 2),
                    round(final_stats['reliability'], 4),
                    final_stats['total_packets_delivered'],
                    final_stats['total_packets_dropped']
                ])
        
        # Generate detailed time series data
        for scenario, history in scenario_histories.items():
            with open(f"results/scenario_{scenario}_timeseries.csv", "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Time", "Avg Latency", "Throughput", "Reliability", 
                               "Total Delivered", "Total Dropped"])
                
                for entry in history:
                    writer.writerow([
                        round(entry['time'], 1),
                        round(entry['avg_latency'], 4),
                        round(entry['throughput'], 2),
                        round(entry['reliability'], 4),
                        entry['total_packets_delivered'],
                        entry['total_packets_dropped']
                    ])
        
        print("Report generated in 'results' directory")


def run_simulation(args):
    """Run the full simulation with all scenarios"""
    # Create network
    network = Network(stadium_radius=1000, use_adaptive_qos=args.adaptive_qos)
    
    # Set up topology
    network.setup_topology(num_ue_nodes=args.num_ue_nodes, 
                         num_enb_nodes=args.num_enb_nodes, 
                         num_iot_nodes=args.num_iot_nodes)
    
    # Visualize network topology
    network.visualize_network()
    
    scenario_histories = {}
    scenarios_to_run = []
    
    if args.scenario == 0:
        # Run all scenarios
        scenarios_to_run = list(range(1, 7))
    else:
        # Run specific scenario
        scenarios_to_run = [args.scenario]
    
    # Run scenarios
    for scenario in scenarios_to_run:
        scenario_histories[scenario] = network.run_scenario(scenario, sim_time=args.sim_time)
    
    # Visualize results
    network.visualize_results(scenario_histories)
    network.visualize_node_loads()
    
    # Generate report
    network.generate_report(scenario_histories)
    
    print("Simulation completed successfully")


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='5G Adaptive QoS Routing Simulation')
    parser.add_argument('--scenario', type=int, default=0, 
                      help='Scenario to run (0=all, 1-6=specific scenario)')
    parser.add_argument('--sim_time', type=float, default=30.0,
                      help='Simulation time in seconds')
    parser.add_argument('--num_ue_nodes', type=int, default=500,
                      help='Number of UE nodes (smartphones)')
    parser.add_argument('--num_enb_nodes', type=int, default=6,
                      help='Number of eNodeB nodes (base stations)')
    parser.add_argument('--num_iot_nodes', type=int, default=100,
                      help='Number of IoT nodes')
    parser.add_argument('--adaptive_qos', action='store_true',
                      help='Enable adaptive QoS routing')
    return parser.parse_args()


def create_animated_visualization(network, scenario_num, output_file="network_animation.mp4"):
    """Create an animated visualization of the network"""
    try:
        import matplotlib.animation as animation
        from matplotlib.animation import FuncAnimation
        from IPython.display import HTML
    except ImportError:
        print("Animation libraries not available. Please install ffmpeg and matplotlib.")
        return
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    def update(frame):
        ax.clear()
        
        # Run simulation step
        network.run_simulation_step(0.1)
        
        # Draw nodes
        node_colors = {
            NodeType.UE: 'blue',
            NodeType.ENB: 'red',
            NodeType.IOT: 'green',
            NodeType.CORE: 'purple'
        }
        
        node_sizes = {
            NodeType.UE: 50,
            NodeType.ENB: 200,
            NodeType.IOT: 30,
            NodeType.CORE: 300
        }
        
        # Draw nodes by type
        for node_type in NodeType:
            nodes = [n for n in network.nodes if n.type == node_type]
            if nodes:
                x = [n.position.x for n in nodes]
                y = [n.position.y for n in nodes]
                
                # Special handling for failed base stations
                if node_type == NodeType.ENB:
                    colors = ['darkred' if isinstance(n, EnbNode) and n.failed else 'red' for n in nodes]
                    ax.scatter(x, y, c=colors, s=node_sizes[node_type], alpha=0.8, label=node_type.name)
                else:
                    ax.scatter(x, y, c=node_colors[node_type], s=node_sizes[node_type], alpha=0.8, label=node_type.name)
        
        # Draw connections
        for node in network.nodes:
            for connected_node in node.connected_nodes:
                if node.id < connected_node.id:  # Draw each connection only once
                    ax.plot([node.position.x, connected_node.position.x],
                          [node.position.y, connected_node.position.y],
                          'k-', alpha=0.2)
        
        # Draw stadium boundary
        stadium_circle = plt.Circle((0, 0), network.stadium_radius, fill=False, color='black', linestyle='--')
        ax.add_patch(stadium_circle)
        
        # Draw in-flight packets
        for packet in network.packets[:100]:  # Limit to visualize only 100 packets to avoid clutter
            source_node = network.get_node_by_id(packet.source_id)
            current_node = network.get_node_by_id(packet.current_node)
            
            if source_node and current_node:
                color = 'green'
                if packet.traffic_type == TrafficType.VIDEO_STREAMING:
                    color = 'blue'
                elif packet.traffic_type == TrafficType.MESSAGING:
                    color = 'orange'
                elif packet.traffic_type == TrafficType.EMERGENCY:
                    color = 'red'
                    
                ax.plot([source_node.position.x, current_node.position.x],
                      [source_node.position.y, current_node.position.y],
                      color=color, alpha=0.5, linewidth=0.5)
        
        # Set axis limits
        ax.set_xlim(-network.stadium_radius*1.2, network.stadium_radius*1.2)
        ax.set_ylim(-network.stadium_radius*1.2, network.stadium_radius*1.2)
        
        # Add title and stats
        ax.set_title(f"Scenario {scenario_num} - Time: {network.current_time:.1f}s\n"
                   f"Packets in flight: {len(network.packets)}, "
                   f"Delivered: {network.stats['total_packets_delivered']}, "
                   f"Dropped: {network.stats['total_packets_dropped']}")
        
        # Add legend (only once)
        if frame == 0:
            ax.legend(loc='upper right')
        
        return ax
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=300, interval=50, blit=False)
    
    # Save animation
    ani.save(output_file, writer='ffmpeg', fps=20, dpi=100)
    plt.close()


def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Run simulation
    run_simulation(args)


if __name__ == "__main__":
    main()