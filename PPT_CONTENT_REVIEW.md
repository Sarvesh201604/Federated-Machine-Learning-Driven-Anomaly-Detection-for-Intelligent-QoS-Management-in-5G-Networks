# PowerPoint Presentation Content - Project Review
## Federated Learning-Driven Anomaly Detection for 5G QoS Routing

---

## SLIDE 1: TITLE SLIDE
**Title:** Federated Learning-Driven Anomaly Detection for Intelligent QoS Management in 5G Networks

**Subtitle:** Project Review Presentation

**Your Details:**
- Name: [Your Name]
- Roll No: [Your Roll No]
- Guide: [Guide Name]
- Date: February 2026

---

## SLIDE 2: AGENDA
**Today's Discussion:**

1. Problem Statement & Motivation
2. Proposed Solution Overview
3. System Architecture
4. Technology Stack
5. Key Modules & Implementation
6. Novelty & Innovation
7. Experimental Setup
8. Results & Performance
9. Dashboard Demo
10. Challenges Faced
11. Conclusion & Future Work

---

## SLIDE 3: PROBLEM STATEMENT

**Challenge in Modern 5G Networks:**

❌ **Traditional QoS Routing Limitations:**
- Only considers network metrics (latency, bandwidth, load)
- Cannot distinguish normal vs abnormal traffic behavior
- Treats all traffic equally regardless of patterns
- Vulnerable to anomalous traffic affecting service quality

🎯 **Impact:**
- Critical services (emergency calls, telemedicine) affected by abnormal traffic
- IoT floods, DDoS attacks degrade network performance
- Reactive rather than proactive traffic management

**Need:** Intelligent, behavior-aware routing system

---

## SLIDE 4: PROPOSED SOLUTION

**Intelligent QoS Routing Framework with Federated Learning**

🧠 **Key Concept:**
Integrate ML-based anomaly detection directly into routing decisions

**Traditional Routing:**
```
Cost = Latency + Load
```

**Our Intelligent Routing:**
```
Cost = Latency + Load + (Anomaly_Probability × Penalty)
```

✅ **Benefits:**
- Behavior-aware routing decisions
- Distributed learning without centralized data collection
- Protects QoS for legitimate traffic
- Adapts dynamically to network conditions

---

## SLIDE 5: SYSTEM ARCHITECTURE

**5-Phase Workflow:**

```
┌─────────────────────────────────────────────────┐
│  Phase 1: DATA COLLECTION                       │
│  └─ Traffic metrics at each Base Station        │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  Phase 2: LOCAL TRAINING                        │
│  └─ Each BS trains local ML model               │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  Phase 3: FEDERATED AGGREGATION (FedAvg)        │
│  └─ Server aggregates model weights             │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  Phase 4: GLOBAL MODEL DISTRIBUTION             │
│  └─ Updated model sent to all nodes             │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  Phase 5: INTELLIGENT ROUTING                   │
│  └─ Anomaly-aware path selection                │
└─────────────────────────────────────────────────┘
```

**Architecture Highlights:**
- Distributed design matching real 5G infrastructure
- Privacy-preserving (no raw data sharing)
- Scalable for large networks

---

## SLIDE 6: TECHNOLOGY STACK

**Programming & Libraries:**
- **Language:** Python 3.x
- **ML Framework:** scikit-learn (MLPClassifier)
- **Network Simulation:** NetworkX
- **Data Processing:** NumPy, Pandas
- **Visualization:** Matplotlib, Streamlit
- **Algorithm:** FedAvg (Federated Averaging)

**Dataset:**
- **UNSW-NB15:** Network intrusion detection dataset
- **Features Used:** 11 QoS-relevant features
  - Traffic Load: sload, dload, rate
  - Latency/Jitter: sjit, djit, tcprtt
  - Protocol & Service type

**Infrastructure:**
- 5 Base Stations (simulated)
- 200 training samples per BS
- 10 Federated Learning rounds

---

## SLIDE 7: KEY MODULES - DATA & LEARNING

**1. data_loader.py - Data Preprocessing**
- Feature selection (11 QoS features)
- Categorical encoding (proto, service)
- MinMaxScaler normalization (0-1 range)
- Dataset split into 5 client chunks

**2. local_model.py - Local Training**
- MLPClassifier architecture: Input(8) → Hidden(10,5) → Output(2)
- Activation: ReLU
- Optimizer: Adam
- Each BS trains independently

**3. federated_server.py - FedAvg Aggregation**
```
Global_Weight = (1/N) × Σ(Local_Weights)
```
- Collects weights from all nodes
- Averages layer-by-layer
- Distributes global model back

---

## SLIDE 8: KEY MODULES - ROUTING (NOVELTY)

**4. anomaly_router.py - Intelligent Routing**

**Process Flow:**
1. **Traffic Feature Extraction:** [Latency, Throughput, Loss, Jitter, ...]
2. **ML Prediction:** Anomaly probability (0-1)
3. **Cost Calculation:**
   - Normal traffic: `Cost = Latency + Load`
   - Anomalous traffic: `Cost = Latency + Load + (P_anomaly × 100)`
4. **Path Selection:** Dijkstra's algorithm on modified costs
5. **Result:** High anomaly → expensive link → alternate route chosen

**Classification Thresholds:**
- **Normal:** Probability < 0.5
- **Suspicious:** 0.5 ≤ Probability < 0.7
- **Malicious:** Probability ≥ 0.7

---

## SLIDE 9: NOVELTY & INNOVATION

**Three Key Innovations:**

🎯 **1. Behavior-Aware QoS Routing**
- First system to integrate ML anomaly detection into routing cost
- Traffic behavior influences path selection
- Goes beyond traditional metric-based routing

🤝 **2. Federated Learning for 5G QoS**
- Distributed learning matching 5G architecture
- Privacy-preserving (no raw data leaves BS)
- Each BS learns from local patterns

🛡️ **3. QoS Protection Focus**
- Not a security project - focus on QoS protection
- Anomaly detection for performance maintenance
- Protects critical services from abnormal traffic impact

**Distinction:** Using anomaly detection for **QoS protection**, NOT security

---

## SLIDE 10: EXPERIMENTAL SETUP

**Network Configuration:**
- **Topology:** 5 Base Stations with interconnections
- **Training Data:** 200 samples per BS (140 normal, 60 anomaly)
- **FL Rounds:** 10 rounds of training
- **Test Flows:** 100 traffic flows

**Comparison Scenarios:**
1. **Baseline:** Traditional routing (Latency + Load)
2. **Proposed:** Intelligent routing (Latency + Load + Anomaly Cost)

**Evaluation Metrics:**
- Average Latency (ms)
- Packet Delivery Ratio (PDR)
- Anomaly Detection Rate
- Model Accuracy

---

## SLIDE 11: RESULTS - PERFORMANCE METRICS

**🎉 Key Performance Results:**

| Metric | Baseline | Our System | Improvement |
|--------|----------|------------|-------------|
| **Average Latency** | 51.93 ms | 26.76 ms | **48.47% ↓** |
| **FL Accuracy** | - | 88.3% | ✅ |
| **Anomaly Detection Rate** | 0% | 32% | ✅ |
| **Packet Delivery Ratio** | Lower | Higher | ✅ |

**Learning Curve:**
- Initial Accuracy: 34%
- Final Accuracy: 88.3%
- Converges after 8-10 rounds

**Interpretation:**
✅ Significant latency reduction during attacks
✅ High detection accuracy ensures reliable routing
✅ System successfully protects QoS for legitimate traffic

---

## SLIDE 12: RESULTS - VISUAL INSIGHTS

**What the Graphs Show:**

📈 **Latency Comparison Graph:**
- **Red Line (Baseline):** Spikes to 200ms during attacks
- **Green Line (Our System):** Stays low ~27ms even during attacks
- **Why?** Intelligent routing avoids high-anomaly links

📊 **FL Training Accuracy:**
- All 5 base stations improve over rounds
- Convergence shows effective learning
- Final global model: 88.3% accuracy

🌐 **Network Topology Visualization:**
- Color-coded flows: Green (normal), Orange (suspicious), Red (malicious)
- Shows dynamic path selection based on anomaly scores

---

## SLIDE 13: DASHBOARD DEMONSTRATION

**Interactive Streamlit Dashboard (`visual_dashboard.py`)**

**5 Tabs:**

✅ **Network Topology** - Visual 5G network map
✅ **Performance Metrics** - Baseline vs Intelligent routing comparison
✅ **Traffic Flows** - Animated flow visualization with anomaly detection
✅ **FL Training** - Learning curves for each base station
✅ **Detailed Results** - Complete data tables & CSV export

**Demo Commands:**
```bash
pip install -r requirements_dashboard.txt
streamlit run visual_dashboard.py
```

**Dashboard Benefits:**
- Real-time visualization
- Interactive scenario testing
- Clear performance comparison
- Professional presentation

---

## SLIDE 14: CHALLENGES & SOLUTIONS

**Challenges Faced:**

⚠️ **Challenge 1: Routing Loop Detection**
- **Problem:** Legitimate high-priority traffic falsely flagged
- **Solution:** Implemented threshold-based classification (Normal/Suspicious/Malicious)

⚠️ **Challenge 2: Data Imbalance**
- **Problem:** UNSW-NB15 has more normal samples than attacks
- **Solution:** Balanced dataset (140 normal, 60 anomaly per BS)

⚠️ **Challenge 3: Real-time Performance**
- **Problem:** ML prediction adds computational overhead
- **Solution:** Optimized feature extraction, lightweight MLP model

⚠️ **Challenge 4: FL Convergence**
- **Problem:** Different BS have different traffic patterns
- **Solution:** FedAvg naturally handles data heterogeneity, 10 rounds sufficient

---

## SLIDE 15: CONCLUSION & FUTURE WORK

**Project Summary:**
✅ Successfully implemented FL-based anomaly detection for 5G QoS
✅ Achieved **48.47% latency improvement** over baseline
✅ Demonstrated behavior-aware intelligent routing
✅ Privacy-preserving distributed learning architecture

**Key Takeaways:**
- Anomaly detection can enhance QoS routing, not just security
- Federated Learning matches real 5G distributed architecture
- Behavior-aware routing protects critical services

**Future Enhancements:**

🔮 **1. Deep Learning Models**
- LSTM for temporal traffic pattern analysis
- CNN for spatial network topology features

🔮 **2. Multi-Objective Optimization**
- Balance latency, energy, cost simultaneously
- Pareto-optimal routing

🔮 **3. Real-world Deployment**
- Integration with OpenAirInterface 5G testbed
- Field testing with actual base stations

🔮 **4. Adaptive Penalty**
- Dynamic penalty values based on network state
- Reinforcement learning for cost optimization

---

## BACKUP SLIDES

### SLIDE 16: VIVA Q&A QUICK REFERENCE

**Q: What is Federated Learning?**
A: Distributed ML where each node trains locally, only shares model weights (not data). Aggregation via FedAvg.

**Q: What is FedAvg?**
A: Federated Averaging. Global_Weight = (1/N) × Σ(Local_Weights). Simple but effective.

**Q: Why not centralized ML?**
A: 5G is distributed. Centralized would require all BS to send data to one location - privacy issues, scalability problems, doesn't match real architecture.

**Q: How does routing work?**
A: Traffic → Feature extraction → ML prediction → Anomaly probability → Cost calculation → Dijkstra shortest path → Route selection

**Q: What is your dataset?**
A: UNSW-NB15 network intrusion dataset. We selected 11 QoS-relevant features. Total: 1000 training samples (200/BS), separate test set.

**Q: Time complexity?**
A: ML prediction: O(1) per flow. Dijkstra: O(E log V). Total negligible overhead.

---

### SLIDE 17: FORMULA SUMMARY

**Cost Calculation Formula:**
```
Cost(u,v) = Base_Cost + Anomaly_Cost

Where:
  Base_Cost = Latency(u,v) + α × Load(u,v)
  Anomaly_Cost = β × P_anomaly(u,v)
  
  α = load weight (typically 1.0)
  β = anomaly penalty (typically 100)
  P_anomaly = ML predicted probability [0,1]
```

**FedAvg Formula:**
```
W_global^(t+1) = (1/N) × Σ(W_local_i^(t))

Where:
  W = model weights
  N = number of base stations
  t = training round
```

---

## END OF PRESENTATION

**Thank you for your attention!**

**Questions?**
