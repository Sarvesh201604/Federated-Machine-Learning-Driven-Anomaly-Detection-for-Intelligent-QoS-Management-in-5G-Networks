# Project: Federated Machine Learning for Anomaly Detection in 5G Networks

## Phase 1: Data Preparation

**Goal**: Convert the raw UNSW-NB15 dataset into a format suitable for training a federated learning model.

### Steps Completed:

1.  **Data Verification**:
    *   Verified the existence of `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv` in the `archive/` directory.

2.  **Feature Selection**:
    *   Selected a subset of 11 features relevant to QoS and Flow analysis, as well as Identity and the Target label.
    *   **Chosen Features**:
        *   *Traffic Load*: `sload`, `dload`, `rate`
        *   *Latency/Jitter*: `sjit`, `djit`, `tcprtt`, `synack`, `ackdat`
        *   *Identity*: `proto`, `service`
        *   *Target*: `label` (0=Normal, 1=Attack)

3.  **Data Processing Script (`data_loader.py`)**:
    *   Created a Python script to automate the data pipeline.
    *   **Categorical Encoding**: Applied `LabelEncoder` to `proto` and `service` columns to convert string categories into numerical format. The encoder was fit on the combined training and testing unique values to handle all categories.
    *   **Normalization**: Applied `MinMaxScaler` to all feature columns (excluding `label`).
        *   *Important*: The scaler was fit **only** on the training data to prevent data leakage. The same scaler was then used to transform the testing data.
        *   All feature values are now scaled between 0 and 1.

4.  **Data Splitting (Federated Learning Setup)**:
    *   Split the processed `training-set` into **5 equal chunks**.
    *   Each chunk represents a local dataset for a simulated "Client" (e.g., a Base Station) in the federated learning environment.
    *   The `testing-set` was kept separate and processed as a single global test set for model evaluation.

5.  **Output**:
    *   Generated processed CSV files in the `processed_data/` directory:
        *   `train_client_0.csv`
        *   `train_client_1.csv`
        *   `train_client_2.csv`
        *   `train_client_3.csv`
        *   `train_client_4.csv`
        *   `test_global.csv`

### Next Steps:
*   Phase 2: Model Architecture (Designing the Neural Network).

## Phase 2: Federated Learning Core

**Goal**: Implement the local training and global aggregation mechanism (FedAvg).

### Steps Completed:

1.  **Federated Model Script (`fl_model.py`)**:
    *   **Model Architecture**: Used `sklearn.neural_network.MLPClassifier` as the base model.
        *   Hidden Layers: (50, 25)
        *   Activation: ReLU
        *   Solver: Adam
    *   **Local Training**: Implemented `train_local_node(node_id, data_chunk)`.
        *   Simulates a client training a local model on its specific data chunk.
        *   Returns the model's weights (`coefs_` and `intercepts_`).
    *   **Aggregation (FedAvg)**: Implemented `aggregate_models(local_updates)`.
        *   Collects weights from all participating nodes.
        *   Calculates the arithmetic mean of the weights layer-by-layer.
        *   `New_Weight = Sum(Node_Weights) / N`
    *   **Global Model Evaluation**:
        *   Constructed a global model using the aggregated weights.
        *   Tested against the global testing set (`test_global.csv`).

### Next Steps:
*   Phase 3: Simulation Loop (Orchestrating the Rounds).

## Phase 3: Network Simulation

**Goal**: Create a virtual 5G network environment to simulate traffic flow and test anomaly detection.

### Steps Completed:

1.  **Network Simulation Script (`simulation.py`)**:
    *   **Topology Setup**:
        *   Used `networkx` to build a graph with 5 nodes (Base Stations).
        *   Defined edges with attributes: `capacity`, `current_load`, `latency`, and `anomaly_score`.
    *   **Traffic Generation**:
        *   Implemented `load_traffic_data` to read the processed training data (`train_client_X.csv`) for each node.
        *   Implemented `generate_traffic(node_id, time_step)` to inject real rows from the UNSW-NB15 dataset as traffic flows.
    *   **Simulation Loop**:
        *   Created a time-stepped loop.
        *   At each step, nodes generate traffic (Normal or Attack based on `label`).
        *   Simulated basic packet movement and logging of Attack flows.

### Next Steps:
*   Phase 4: Integration (Connecting FL to Simulation).

## Phase 4: Anomaly-Aware Routing

**Goal**: Integrate the Global FL Model to enable dynamic, anomaly-aware routing.

### Steps Completed:

1.  **FL Model Integration**:
    *   Imported FL components (`create_model`, `train_local_node`, `aggregate_models`) into `simulation.py`.
    *   **Initialization**: At simulation start, the system now runs a full FL round (Local Training -> Aggregation) to initialize the `self.global_model`.

2.  **Real-Time Anomaly Scoring**:
    *   Implemented `update_network_state`.
    *   Before routing a packet, the Global Model predicts the **Anomaly Probability** of the flow features.
    *   This probability is assigned to the network links as an `anomaly_score`.

3.  **Dynamic Routing Algorithm**:
    *   **Cost Formula**: `Link_Cost = Base_Latency + (Alpha * Anomaly_Score)`.
    *   **Logic**: High anomaly scores drastically increase the link cost.
    *   **Rerouting**: Used `networkx.dijkstra_path` with the dynamic `cost` weight.
    *   **Result**: The simulation logs show traffic being **rerouted via longer paths** to avoid links associated with high anomaly probabilities (Attacks).

### Final Status:
*   **Data**: Prepared and Processed.
*   **Brain**: Federated Model Training & Aggregation working.
*   **Body**: Network Simulation running.
*   **Action**: Dynamic Routing successfully avoiding anomalies.

## Phase 5: Evaluation & Reporting

**Goal**: Prove the effectiveness of the Anomaly-Aware Routing through comparative analysis.

### Methodology
*   **Scenarios**:
    *   **Baseline**: Standard routing (ignoring anomaly scores). Attacks cause link degradation (Latency spikes, Packet drops).
    *   **Proposed**: Anomaly-Aware routing (high cost for anomalies).
*   **Metrics**:
    *   **Packet Delivery Ratio (PDR)**.
    *   **Average Latency**.

### Results
*   **Baseline**: High latency spikes observed during attack waves. PDR dropped due to congestion on attacked links.
*   **Proposed**: Maintained low, stable latency by proactively rerouting traffic around attacks. PDR remained high.
*   **Visualization**: Generated `evaluation_results.png` showing the clear performance gap between the two approaches.

## project Complete
The federated learning-driven anomaly detection system has been successfully implemented and verified. The simulation demonstrates that dynamic, risk-aware routing significantly improves network QoS in the face of cyber attacks.




