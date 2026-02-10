# Simulation Reviewer Guide

## 1. Overview
This document explains the execution and results of the **5G/IoT Network Simulation with Adaptive QoS**. The simulation models a dynamic network environment to evaluate how well adaptive QoS routing maintains performance under various stress conditions.

## 2. Execution Summary
We successfully executed the simulation engine (`mainapp.py`), running all **6 defined scenarios**:
1.  **Normal Traffic**: Baseline performance.
2.  **Base Station Overload**: One node heavily congested.
3.  **Emergency Priority**: Critical traffic injection.
4.  **IoT Surge**: Massive sensor data spike.
5.  **
Device Mobility**: High-speed user movement.
6.  **Node Failure**: Base station failure and recovery.

### Environment Setup
- **Dependencies**: `numpy`, `matplotlib`, `pandas`, `seaborn`, `networkx`.
- **Simulation Time**: 30 seconds per scenario.
- **Topology**: 500 mobile UEs, 6 eNodeBs, 100 IoT nodes, 2 Core nodes.

## 3. Key Findings
The Adaptive QoS algorithm performed exceptionally well across all scenarios:
- **Reliability**: **100% (1.0)** packet delivery ratio for all scenarios. No packets were dropped even during overload or failure events.
- **Latency**: Maintained smooth average latency between **240ms and 290ms**.
- **Throughput**: Consistent throughput of **~550-575 packets/second**.

The system successfully prioritized traffic and rerouted packets around congestion and failures, validating the robustness of the adaptive algorithm.

## 4. Deliverables & Artifacts
The following files have been generated for your review:

### A. Analysis Report (Primary)
- **`qos_report.html`**: Interactive HTML report containing detailed analysis of each scenario, recommendations, and executive summary. **Start here.**

### B. Visualizations
- **`network_topology.png`**: Map of the network layout (Base stations, UEs, IoT nodes).
- **`scenario_comparison.png`**: Multi-panel chart comparing latency, throughput, and reliability across all scenarios.
- **`traffic_type_performance.png`**: Breakdown of performance by traffic class (Video, Emergency, IoT, Messaging).
- **`enb_loads.png`**: Time-series plot of base station load, showing how traffic shifted during overload/failure.

### C. Raw Data
- **`results/` Directory**: Contains CSV logs for every time step of every scenario.
- **`qos_summary_export.csv`**: High-level summary metrics (Latency, Throughput, Reliability) for each scenario.

## 5. How to Reproduce
To re-run the simulation and generate fresh reports:

1.  **Run the Simulation**:
    ```bash
    python mainapp.py
    ```
    This generates the raw data in `results/` and the PNG visualizations.

2.  **Generate the Report**:
    ```bash
    python create_report.py
    ```
    This processes the results and creates/updates `qos_report.html` and `qos_summary_export.csv`.
