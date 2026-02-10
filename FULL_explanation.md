# 5G Network Simulation with Adaptive QoS Routing - Project Explanation

## 1. Project Overview
This project simulates a **5G Network** environment designed for event management (e.g., in a stadium). Its primary goal is to demonstrate an **Adaptive Quality of Service (QoS) Routing Algorithm** that optimizes network performance under various challenging conditions such as congestion, emergency situations, IoT data surges, and hardware failures.

The simulation models key network components including **User Equipment (UE)** (smartphones), **eNodeBs** (Base Stations), **IoT Sensors**, and **Core Network** nodes. It runs multiple scenarios to test the resilience and efficiency of the routing logic.

## 2. File Explanations

### `mainapp.py`
This is the **core simulation engine**. It contains all the definitions and logic for the network simulation.
*   **Classes**:
    *   `TrafficType`: Enum for different traffic classes (Video, Messaging, IoT, Emergency).
    *   `NodeType`: Enum for node types (UE, ENB, IOT, CORE).
    *   `QoSRequirements`: Dataclass defining bandwidth, latency, and reliability needs for each traffic type.
    *   `Packet`: Represents a data packet moving through the network.
    *   `Node` (and subclasses `EnbNode`, `UeNode`, `IotNode`, `CoreNode`): Models the network devices.
    *   `Network`: Manages the simulation state, topology, and time steps.
*   **Key Responsibilities**:
    *   **Traffic Generation**: `UeNode` and `IotNode` generate packets based on probabilistic patterns.
    *   **Routing Logic**: `EnbNode` contains the intelligent routing decision-making.
    *   **Simulation Loop**: Moves nodes, processes queues, and updates statistics.
    *   **Visualization**: Generates the static output images (`network_topology.png`, `enb_loads.png`, etc.).

### `simulation.py`
This file defines the **Dashboard Application** (`AdaptiveQoSDashboard`).
*   **Purpose**: To provide an interactive or report-based visualization of the simulation results *after* the simulation has run (or processing generated CSVs).
*   **Key Responsibilities**:
    *   Loads simulation data from the `results/` directory (CSV files).
    *   Creates an interactive `matplotlib` dashboard with sliders and buttons to explore different scenarios.
    *   Generates the **HTML Report** (`qos_report.html`) which summarizes findings and recommendations.

### `create_report.py`
This is a **Helper Script** to automate report generation.
*   **Purpose**: To execute the reporting functionality defined in `simulation.py`.
*   **Key Responsibilities**:
    *   Reads the summary CSV.
    *   Patches data (calculating a 'QoS Score') if necessary.
    *   Calls `dashboard.generate_report()` to output the HTML report.

## 3. The Routing Algorithm
The core of the project is the **Adaptive QoS Routing Algorithm**, located in the `EnbNode._get_next_hop` method in `mainapp.py`.

### How it works:
When a packet arrives at an eNodeB (Base Station), the node decides the next hop (where to send the packet next) based on a computed **Score**. The neighbor with the highest score is chosen.

**The Score Formula involves:**
1.  **Distance Factor**: How close is the neighbor to the final destination? (Shorter distance = Higher score).
2.  **Load Factor**: How busy is the neighbor? (Lower load = Higher score).
3.  **Traffic Type Weights**: Different traffic types value Distance vs. Load differently.

### Traffic Type Priorities:
*   **Video Streaming**: Prioritizes **Load** (70%) over Distance (30%). It needs bandwidth, so it avoids congested nodes even if the path is longer.
*   **Messaging**: Prioritizes **Distance** (70%) over Load (30%). It needs low latency, so it prefers the shortest path.
*   **IoT Sensor**: Balanced approach (50% Load, 50% Distance).
*   **Emergency**: Heavily prioritizes **Distance** (80%) to get the message delivered as fast as possible, but also gets a massive priority boost in the final score calculation.

**Final Score Calculation:**
`Score = (Weight_Load * Load_Factor + Weight_Dist * Distance_Factor) * Priority_Multiplier`

## 4. Output Image Identification

### `network_topology.png`
*   **What it shows**: A map of the network within the stadium (circular boundary).
*   **Key Elements**:
    *   **Red Nodes**: eNodeBs (Base Stations) arranged in a ring.
    *   **Blue Dots**: UE (Smartphones) scattered randomly.
    *   **Green Dots**: IoT Sensors.
    *   **Purple Nodes**: Core Network nodes.
    *   **Lines**: Connectivity between nodes.

### `enb_loads.png`
*   **What it shows**: A line graph tracking the **Load (Congestion Level)** of each Base Station (eNodeB) over time.
*   **Interpretation**:
    *   Y-axis is Load (0.0 to 1.0, where 1.0 is 100% full).
    *   X-axis is Simulation Time.
    *   Spikes indicate congestion events (e.g., Scenario 2 "Base Station Overload" or Scenario 6 "Failure" where traffic shifts).

### `scenario_comparison.png`
*   **What it shows**: A 4-panel comparison of the network's performance across all tested scenarios.
*   **Key Metrics**:
    1.  **Average Latency**: Lower is better. Shows delay.
    2.  **Throughput**: Higher is better. Shows data volume processed.
    3.  **Reliability**: Higher is better (closer to 1.0). Shows packet success rate.
    4.  **Cumulative Packet Delivery**: Total packets successfully sent.

### `traffic_type_performance.png`
*   **What it shows**: Bar charts breaking down performance by **Traffic Type** (Video, Messaging, IoT, Emergency).
*   **Key Insights**:
    *   **Delivery Ratio**: Did Emergency packets get 100% delivery? (They should).
    *   **Latency**: Is Messaging/Emergency latency lower than Video?
    *   **Packet Count**: Volume of each traffic type.
