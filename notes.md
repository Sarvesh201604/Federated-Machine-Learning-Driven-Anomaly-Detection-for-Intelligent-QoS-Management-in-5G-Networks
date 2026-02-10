# Simulation Notes & Output Explanation

This document provides a detailed explanation of the simulation "final run" outputs, including the generated images and the underlying scenarios of the 5G Adaptive QoS Routing Simulation.

## 1. Output Images Explained

The simulation generates four key visualization images in the root directory. Here is what each one signifies:

### 1. `network_topology.png`
*   **Significance**: Visualizes the physical layout of the simulated 5G network within the "stadium" environment.
*   **What it shows**:
    *   **Red Dots (Large)**: eNodeBs (Base Stations) arraged in a circle.
    *   **Purple Dots (Large)**: Core Network nodes.
    *   **Blue Dots (Small)**: User Equipment (UE) - smartphones, etc.
    *   **Green Dots (Small)**: IoT Sensors.
    *   **Lines**: Connections between nodes (wireless or wired backhaul).
    *   **Dotted Circle**: The stadium boundary.
*   **Purpose**: To verify the spatial distribution of nodes and the connectivity graph before running scenarios.

### 2. `scenario_comparison.png`
*   **Significance**: A comparative analysis of network performance across all tested scenarios.
*   **What it shows (4 Subplots)**:
    *   **Average Latency Over Time**: How delay varies during the simulation for each scenario. Lower is better.
    *   **Network Throughput Over Time**: The rate of successful packet delivery (packets/second).
    *   **Network Reliability Over Time**: The ratio of delivered packets to total generated packets (0.0 to 1.0).
    *   **Cumulative Packet Delivery**: The total number of packets successfully delivered as time progresses.
*   **Purpose**: To clearly see how different conditions (like base station failure or emergency traffic) impact global network metrics compared to the baseline "Normal" scenario.

### 3. `traffic_type_performance.png`
*   **Significance**: Break down of network performance by the specific type of application traffic.
*   **What it shows (3 Subplots)**:
    *   **Delivery Ratio**: The success rate for each traffic type. You should see **Emergency** and **IoT** traffic having High reliability (close to 1.0) due to QoS prioritization, while **Video** might be lower.
    *   **Average Latency**: The average time it took for packets of that type to be delivered. **Emergency** traffic should have the lowest latency.
    *   **Total Packets**: The volume of traffic generated for each type.
*   **Purpose**: To validate that the Adaptive QoS algorithm is correctly prioritizing critical traffic (Emergency/IoT) over bandwidth-heavy traffic (Video) during congestion.

### 4. `enb_loads.png`
*   **Significance**: A time-series view of the processing load on each Base Station (eNodeB).
*   **What it shows**:
    *   The Y-axis represents Load (0.0 to 1.0, where 1.0 is 100% capacity).
    *   Traces for each eNodeB (e.g., eNodeB 2000, 2001, etc.).
*   **Purpose**: To identify bottlenecks. For example, in **Scenario 2 (Base Station Overload)**, you should see one line spike to 1.0. In **Scenario 6 (Base Station Failure)**, you might see a line drop to 0 or spike depending on failure simulation, while neighbors take up the load.

---

## 2. Simulation Scenarios Overview

The simulation runs through 6 distinct scenarios to test the network's robustness:

1.  **Normal Traffic**: Baseline operation with balanced traffic.
2.  **Base Station Overload**: One base station is flooded with video traffic (t=5s to 20s) to test load balancing.
3.  **Emergency Traffic Priority**: A surge of high-priority emergency messages (t=10s to 15s) to test QoS prioritization.
4.  **IoT Sensor Data Surge**: A spike in IoT sensor readings (t=12s to 18s) to test massive machine-type communication handling.
5.  **Device Mobility**: UE devices move at high speeds (starts t=5s) to test handover efficiency between base stations.
6.  **Base Station Failure**: A random base station fails completely (t=10s) and recovers (t=20s) to test failover and self-healing.

## 3. Key Concepts

### Traffic Classes (QoS)
The simulation handles 4 distict traffic types with different priorities:
*   **EMERGENCY (Priority 10)**: Highest priority. Low latency required. (e.g., Safety alerts)
*   **VIDEO_STREAMING (Priority 7)**: High bandwidth, generally tolerant of some latency but heavy load.
*   **IOT_SENSOR (Priority 6)**: High reliability required, small packets.
*   **MESSAGING (Priority 5)**: Best effort, lower priority.

### Adaptive Routing Algorithm
The network uses an "Adaptive QoS" routing strategy. Instead of just picking the closest node, eNodeBs select the next hop based on a score calculated from:
1.  **Load Factor**: How busy the neighbor is (send to less busy nodes).
2.  **Distance Factor**: How close the neighbor is to the destination.
3.  **QoS Priority**: Higher priority packets weight these factors differently (e.g., Emergency cares more about speed/distance, Video cares more about available bandwidth/load).
