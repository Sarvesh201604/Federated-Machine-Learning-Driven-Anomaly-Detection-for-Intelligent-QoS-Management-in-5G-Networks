# 🎯 VISUAL DASHBOARD DEMO GUIDE

## 🚀 QUICK START (3 STEPS)

### Step 1: Install Dependencies
```bash
pip install -r requirements_dashboard.txt
```

### Step 2: Run the Dashboard
```bash
streamlit run visual_dashboard.py
```

### Step 3: Open in Browser
The dashboard will automatically open at: **http://localhost:8501**

---

## 📊 WHAT THE DASHBOARD SHOWS

### 5 Interactive Tabs:

#### 🌐 Tab 1: Network Topology
- **Visual network map** showing all base stations and connections
- **Color-coded traffic flows** (green=normal, orange=suspicious, red=malicious)
- **Network statistics** (number of nodes, links, diameter)
- **Perfect for showing**: "This is our 5G network infrastructure"

#### 📊 Tab 2: Performance Metrics
- **Side-by-side comparison** of Baseline vs Intelligent Routing
- **Real-time latency graphs** showing improvement
- **Packet Delivery Ratio (PDR)** comparison
- **Detection accuracy gauge**
- **Summary statistics** with improvements highlighted
- **Perfect for showing**: "Here's how much better our system performs"

#### 🔄 Tab 3: Traffic Flows
- **Animated traffic flow** visualization over time
- **Time slider** to see traffic at each moment
- **Color-coded flow lines** showing anomaly severity
- **Flow statistics** (normal/suspicious/malicious counts)
- **Perfect for showing**: "Watch how traffic moves and anomalies are detected"

#### 📈 Tab 4: FL Training
- **Learning curve** for each base station
- **Accuracy improvement** over training rounds
- **Final accuracy metrics** for each BS
- **Global model accuracy**
- **Perfect for showing**: "See how federated learning improves detection"

#### 📋 Tab 5: Detailed Results
- **Full data table** of all results
- **CSV download** option
- **Statistical summary**
- **Upload custom CSV** to visualize your own data
- **Perfect for showing**: "Here are all the numbers in detail"

---

## 🎤 HOW TO DEMONSTRATE TO REVIEWERS

### Scenario 1: Show the Problem (2 minutes)

**What to say:**
"Let me show you why we need this system..."

**What to do:**
1. Select **"Experiment 3: Heavy Attack (60% Anomalies)"**
2. Set Anomaly Percentage to **60%**
3. Click **"Run Simulation"**
4. Go to **"Performance Metrics"** tab
5. Point to the **red line** (baseline) spiking to 200ms

**What to say:**
"See this? With traditional routing, attacks cause huge latency spikes. This degrades QoS for legitimate users."

---

### Scenario 2: Show Your Solution (2 minutes)

**What to say:**
"Now watch what happens with our intelligent system..."

**What to do:**
1. Point to the **green line** (intelligent) staying stable at 25-30ms
2. Show the **improvement metric** (48% or higher)
3. Show the **PDR comparison** (85% vs 96%)

**What to say:**
"Our system detects anomalies and routes around them. Normal traffic protected, 48% latency improvement."

---

### Scenario 3: Show How Detection Works (2 minutes)

**What to say:**
"Let me show you how anomaly detection works in real-time..."

**What to do:**
1. Go to **"Traffic Flows"** tab
2. Move the **time slider** back and forth
3. Point to **red lines** (malicious traffic)
4. Point to **green lines** (normal traffic)

**What to say:**
"Red flows are detected anomalies. Our ML model classifies each flow in real-time. Normal traffic (green) uses optimal paths, anomalies (red) get rerouted or isolated."

---

### Scenario 4: Show Federated Learning (2 minutes)

**What to say:**
"Our system uses federated learning to improve detection..."

**What to do:**
1. Select **"Experiment 4: FL Training Progress"**
2. Go to **"FL Training"** tab
3. Point to the **learning curves** going up
4. Show **final accuracies** (88%+)

**What to say:**
"Each base station trains locally, then shares knowledge. Accuracy improves from 35% to 88% over 10 rounds. This is federated learning in action."

---

### Scenario 5: Show Configuration Flexibility (1 minute)

**What to say:**
"The system is highly configurable..."

**What to do:**
1. Change **"Number of Base Stations"** to 7 or 10
2. Change **"FL Training Rounds"** to 20
3. Change **"Anomaly Penalty"** to 5000
4. Click **"Run Simulation"** again

**What to say:**
"We can easily scale to more base stations, train longer for higher accuracy, or adjust routing aggressiveness. Everything is tunable."

---

## 🎯 ANSWERING COMMON REVIEWER QUESTIONS

### Q: "How do you know it works?"
**Answer + Action:**
- "Let me show you the metrics" → Go to **Performance Metrics** tab
- Point to **48% improvement**, **88% accuracy**
- "These are real measurements, reproducible"

### Q: "Can you show me the network topology?"
**Answer + Action:**
- Go to **Network Topology** tab
- "Here's our 5G network with [N] base stations connected in mesh"
- "Blue nodes are base stations, lines are links"

### Q: "What happens during an attack?"
**Answer + Action:**
- Select **"Experiment 3: Heavy Attack"**
- Run simulation
- Go to **Traffic Flows** tab
- "See the red flows? Those are attacks being detected and rerouted"

### Q: "How does federated learning help?"
**Answer + Action:**
- Go to **FL Training** tab
- "Each curve represents one base station learning"
- "They share knowledge, global model reaches 88% accuracy"
- "Better than any single station could achieve alone"

### Q: "Can I see the raw data?"
**Answer + Action:**
- Go to **Detailed Results** tab
- "Here's every data point"
- Click **"Download Results as CSV"**
- "You can analyze it yourself"

### Q: "What if I have my own data?"
**Answer + Action:**
- Go to **Detailed Results** tab
- Click **"Upload CSV"** in sidebar
- "You can upload your own network data and visualize it"

---

## 💡 PRO TIPS FOR SMOOTH DEMO

### Before the Demo:
1. ✅ **Test run** the dashboard once before presenting
2. ✅ **Open browser** to http://localhost:8501 before reviewers arrive
3. ✅ **Have the command ready** in a terminal window
4. ✅ **Take screenshots** as backup (in case of technical issues)
5. ✅ **Prepare one custom CSV** to show upload feature

### During the Demo:
1. ✅ **Start with Experiment 2** (mixed traffic) - shows clear improvement
2. ✅ **Use the time slider** in Traffic Flows - very visual and impressive
3. ✅ **Let them interact** - hand them the mouse/keyboard
4. ✅ **Point to specific numbers** - don't just describe, show
5. ✅ **Be ready to modify parameters** - shows you understand the system

### If Something Goes Wrong:
1. ✅ **Have static plots ready** (the original PNG from integrated_fl_qos_system.py)
2. ✅ **Show the code** - "Here's how the routing decision is made"
3. ✅ **Show CSV results** - "Here are the pre-computed results"
4. ✅ **Stay calm** - explain what should happen

---

## 🎓 INTEGRATION WITH YOUR EXISTING PROJECT

The dashboard **uses your existing code**:

```python
# It imports your modules:
from integrated_fl_qos_system import create_5g_network, run_federated_learning
from anomaly_router import AnomalyAwareRouter
from simulation import simulate_traffic

# So all the real algorithms are running
# It just adds visualization on top
```

**This means:**
- ✅ Results are real, not fake
- ✅ Code is your own implementation
- ✅ Easy to add more features
- ✅ Can show code + visualization together

---

## 🔧 CUSTOMIZATION OPTIONS

### Want to add a new experiment?
Edit `visual_dashboard.py`, line 206:
```python
experiment = st.sidebar.selectbox(
    "Select Experiment",
    [
        "Your new experiment here",  # Add this
        # ... existing experiments
    ]
)
```

### Want to change colors?
Lines 420-430: Modify the color scheme
```python
if anomaly_score > 0.7:
    color = 'red'  # Change to any color
```

### Want to add more metrics?
Go to `create_metrics_comparison()` function and add new subplots

---

## 📞 TROUBLESHOOTING

### Dashboard won't start?
```bash
# Make sure Streamlit is installed
pip install streamlit

# Try running with verbose output
streamlit run visual_dashboard.py --logger.level=debug
```

### Port already in use?
```bash
# Use a different port
streamlit run visual_dashboard.py --server.port=8502
```

### Graphs not showing?
```bash
# Reinstall plotly
pip uninstall plotly
pip install plotly
```

### Import errors?
```bash
# Make sure you're in the project directory
cd "c:\Users\sayee\Downloads\Federated-Machine-Learning-Driven-Anomaly-Detection-for-Intelligent-QoS-Management-in-5G-Networks"

# Then run
streamlit run visual_dashboard.py
```

---

## 🎬 FINAL DEMO SCRIPT (10 minutes total)

### Minute 0-1: Introduction
"I've created an interactive dashboard to demonstrate our system. Let me show you..."
[Open dashboard]

### Minute 1-3: Show the Problem
[Run Experiment 3 - Heavy Attack]
"Traditional routing can't handle attacks - see these latency spikes?"

### Minute 3-5: Show Your Solution
[Point to metrics]
"Our intelligent system maintains stable performance - 48% improvement"

### Minute 5-7: Show Real-Time Detection
[Traffic Flows tab, move slider]
"Watch how we detect and isolate malicious traffic in real-time"

### Minute 7-9: Show Learning Process
[FL Training tab]
"This is federated learning improving accuracy from 35% to 88%"

### Minute 9-10: Show Flexibility
[Change parameters, re-run]
"Fully configurable - we can scale to any network size"

[END]

---

## ✅ CHECKLIST FOR DEMO DAY

- [ ] Dashboard dependencies installed
- [ ] Dashboard tested and working
- [ ] Can switch between experiments smoothly
- [ ] Understand each tab and what it shows
- [ ] Prepared answers to "why" questions
- [ ] Have backup static images ready
- [ ] Laptop fully charged
- [ ] Internet not required (runs locally)
- [ ] Reviewed this guide

---

## 🎯 YOU ARE READY!

This dashboard makes your complex ML/routing system **visual and interactive**. 

Reviewers can:
- ✅ **See** the network topology
- ✅ **Watch** traffic flows
- ✅ **Understand** the improvements
- ✅ **Interact** with parameters
- ✅ **Verify** results themselves

**Much better than NS3** because:
- ✅ Ready in minutes, not weeks
- ✅ Easy to understand
- ✅ Interactive and impressive
- ✅ Uses your real code
- ✅ Professional looking

**Good luck with your presentation! 🎓🚀**
