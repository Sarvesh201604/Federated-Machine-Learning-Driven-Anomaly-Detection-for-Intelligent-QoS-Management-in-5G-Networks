# Federated Machine Learning Driven Anomaly Detection for Intelligent QoS Management in 5G Networks

This repository contains the code and resources for the "Federated Machine Learning Driven Anomaly Detection for Intelligent QoS Management in 5G Networks" project.

## Project Overview

This project aims to leverage Federated Learning (FL) to detect anomalies in 5G network traffic and intelligently manage Quality of Service (QoS).

## Directory Structure

*   `archive/`: Contains the raw dataset.
*   `processed_data/`: Contains the preprocessed data.
*   `results/`: Stores simulation results.
*   `data_loader.py`: Script for data preprocessing.
*   `fl_model.py`: Federated Learning model implementation.
*   `simulation.py`: Network simulation script.
*   `project.md`: Project documentation and task tracking.
*   `README.md`: This file.

## Getting Started

1.  **Install Dependencies**:
    ```bash
    pip install pandas numpy scikit-learn networkx matplotlib
    ```

2.  **Run Simulation**:
    ```bash
    python simulation.py
    ```

## Interpreting Output

### 1. Terminal Logs
When you run `simulation.py`, you will see:

*   **FL Initialization**:
    ```text
    Aggregating 5 local models...
    Global FL Model successfully initialized.
    ```
*   **Simulation Steps**:
    *   **Normal Flow**: `[Node 0 -> 2] Type: Final Normal Flow. Path: [0, 2] (Cost: 5.00)`
    *   **Attack Detected**:
        ```text
        [Node 2 -> 1] Type: ATTACK | Anomaly Score: 0.8923
           >>> Rerouting Path: [2, 0, 1] (Cost: 55.40)
           [!] ALERT: Traffic rerouted via longer path to avoid anomaly!
        ```
        *   **Interpretation**: The system detected an attack (Score > 0.5) on the direct link. It calculated that the direct link is too "expensive" (risky) and triggered a reroute via an intermediate node, even though it's physically longer.

### 2. Evaluation Graph (`evaluation_results.png`)
The script generates this image to prove performance.

*   **Red Dashed Line (Baseline)**: Represents standard routing.
    *   *Interpretation*: You will see spikes in latency. This is because standard routing sends traffic blindly through attacked/congested links.
*   **Green Solid Line (Proposed)**: Represents your FL-driven routing.
    *   *Interpretation*: The line should be lower and flatter. Even during attacks, the system finds clean paths, keeping latency low (QoS high).

### 3. Processed Data
*   `processed_data/test_global.csv`: The held-out dataset used to test the global model's raw accuracy.
