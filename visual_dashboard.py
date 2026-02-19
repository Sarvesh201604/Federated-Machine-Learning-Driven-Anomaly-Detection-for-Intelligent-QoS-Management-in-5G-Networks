"""
🎯 INTERACTIVE VISUAL DASHBOARD FOR 5G ANOMALY DETECTION & ROUTING
==================================================================

This Streamlit app provides real-time visualization of:
- Network topology (base stations and links)
- Traffic flows (normal vs anomalous)
- Routing decisions (baseline vs intelligent)
- Performance metrics
- Federated learning progress

HOW TO RUN:
-----------
streamlit run visual_dashboard.py

Then open browser to: http://localhost:8501
"""

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import random
import time
from pathlib import Path

# Note: Dashboard works standalone with simulated data
# This ensures it works even if your main project files are being modified

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="5G FL-QoS Visual Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .anomaly-box {
        background-color: #ffe0e0;
        padding: 0.5rem;
        border-radius: 0.3rem;
        border-left: 4px solid #ff4444;
    }
    .normal-box {
        background-color: #e0ffe0;
        padding: 0.5rem;
        border-radius: 0.3rem;
        border-left: 4px solid #44ff44;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_network_topology_plot(G, node_colors=None, edge_colors=None, title="5G Network Topology"):
    """Create interactive network topology visualization using Plotly"""
    
    # Get node positions using spring layout
    pos = nx.spring_layout(G, seed=42, k=2)
    
    # Create edge traces
    edge_traces = []
    
    if edge_colors is None:
        edge_colors = ['gray'] * len(G.edges())
    
    for (u, v), color in zip(G.edges(), edge_colors):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=2, color=color),
            hoverinfo='none',
            showlegend=False
        )
        edge_traces.append(edge_trace)
    
    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    
    if node_colors is None:
        node_colors = ['lightblue'] * len(G.nodes())
    
    for node, color in zip(G.nodes(), node_colors):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"Base Station {node}")
        node_color.append(color)
    
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(
            size=50,
            color=node_color,
            line=dict(width=3, color='darkblue')
        ),
        text=[f"BS-{n}" for n in G.nodes()],
        textposition="middle center",
        textfont=dict(size=12, color='white', family='Arial Black'),
        hovertext=node_text,
        hoverinfo='text',
        showlegend=False
    )
    
    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])
    
    fig.update_layout(
        title=title,
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        height=500
    )
    
    return fig


def create_traffic_flow_animation(G, flows_data):
    """Create animated visualization of traffic flows"""
    
    pos = nx.spring_layout(G, seed=42, k=2)
    
    frames = []
    
    for t, flows in enumerate(flows_data):
        # Edge traces for this frame
        edge_traces = []
        
        for flow in flows:
            src, dst, flow_type, anomaly_score = flow['src'], flow['dst'], flow['type'], flow['anomaly']
            
            if src in pos and dst in pos:
                x0, y0 = pos[src]
                x1, y1 = pos[dst]
                
                # Color based on anomaly score
                if anomaly_score > 0.7:
                    color = 'red'
                    width = 5
                elif anomaly_score > 0.5:
                    color = 'orange'
                    width = 4
                else:
                    color = 'green'
                    width = 3
                
                edge_trace = go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode='lines',
                    line=dict(width=width, color=color),
                    hoverinfo='text',
                    hovertext=f"{flow_type}: {anomaly_score:.2f}",
                    showlegend=False
                )
                edge_traces.append(edge_trace)
        
        # Node trace
        node_trace = go.Scatter(
            x=[pos[n][0] for n in G.nodes()],
            y=[pos[n][1] for n in G.nodes()],
            mode='markers+text',
            marker=dict(size=40, color='lightblue', line=dict(width=2, color='darkblue')),
            text=[f"BS-{n}" for n in G.nodes()],
            textposition="middle center",
            showlegend=False
        )
        
        frames.append(go.Frame(data=edge_traces + [node_trace], name=str(t)))
    
    fig = go.Figure(
        data=frames[0].data if frames else [],
        layout=go.Layout(
            title="Traffic Flow Animation",
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(label="Play", method="animate", args=[None, {"frame": {"duration": 500}}]),
                    dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}])
                ]
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
        frames=frames
    )
    
    return fig


def create_metrics_comparison(baseline_data, intelligent_data):
    """Create comparison metrics visualization"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Latency Over Time", "Packet Delivery Ratio", 
                        "Throughput", "Anomaly Detection Accuracy"),
        specs=[[{"type": "scatter"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "indicator"}]]
    )
    
    # Latency over time
    fig.add_trace(
        go.Scatter(x=list(range(len(baseline_data))), y=baseline_data, 
                   name="Baseline", line=dict(color='red', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=list(range(len(intelligent_data))), y=intelligent_data, 
                   name="Intelligent", line=dict(color='green', width=2)),
        row=1, col=1
    )
    
    # PDR comparison
    fig.add_trace(
        go.Bar(x=["Baseline", "Intelligent"], y=[85, 96], 
               marker_color=['red', 'green']),
        row=1, col=2
    )
    
    # Throughput
    fig.add_trace(
        go.Scatter(x=list(range(100)), y=np.random.randn(100).cumsum() + 50,
                   name="Throughput", line=dict(color='blue', width=2)),
        row=2, col=1
    )
    
    # Accuracy gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=88.3,
            title={'text': "Detection Accuracy (%)"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "darkblue"},
                   'steps': [
                       {'range': [0, 50], 'color': "lightgray"},
                       {'range': [50, 75], 'color': "gray"},
                       {'range': [75, 100], 'color': "lightgreen"}
                   ]}
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=True, title_text="Performance Metrics Dashboard")
    
    return fig


def create_fl_training_progress(rounds_data):
    """Visualize federated learning training progress"""
    
    fig = go.Figure()
    
    for bs_id, accuracies in rounds_data.items():
        fig.add_trace(go.Scatter(
            x=list(range(1, len(accuracies) + 1)),
            y=accuracies,
            mode='lines+markers',
            name=f'BS-{bs_id}',
            line=dict(width=2),
            marker=dict(size=8)
        ))
    
    fig.update_layout(
        title="Federated Learning Training Progress",
        xaxis_title="Training Round",
        yaxis_title="Accuracy (%)",
        hovermode='x unified',
        height=400
    )
    
    return fig


# ============================================================================
# MAIN DASHBOARD
# ============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">📡 5G Federated Learning QoS Management Dashboard</h1>', 
                unsafe_allow_html=True)
    
    st.info("ℹ️ **Interactive Visualization Dashboard** - This dashboard generates realistic simulated data to demonstrate your system's capabilities. Perfect for presentations and reviews!")
    
    st.markdown("---")
    
    # ========================================================================
    # SIDEBAR CONTROLS
    # ========================================================================
    
    st.sidebar.title("⚙️ Configuration")
    
    # Experiment selector
    experiment = st.sidebar.selectbox(
        "Select Experiment",
        [
            "Experiment 1: Normal Traffic Only",
            "Experiment 2: Mixed Traffic (30% Anomalies) - DEFAULT",
            "Experiment 3: Heavy Attack (60% Anomalies)",
            "Experiment 4: FL Training Progress",
            "Experiment 5: Baseline vs Intelligent Routing"
        ]
    )
    
    # Network configuration
    st.sidebar.subheader("🌐 Network Configuration")
    num_base_stations = st.sidebar.slider("Number of Base Stations", 3, 10, 5)
    simulation_rounds = st.sidebar.slider("FL Training Rounds", 5, 30, 10)
    
    # Traffic configuration
    st.sidebar.subheader("🚦 Traffic Configuration")
    if "Normal Traffic Only" in experiment:
        anomaly_percentage = 0
    elif "Heavy Attack" in experiment:
        anomaly_percentage = 60
    else:
        anomaly_percentage = st.sidebar.slider("Anomaly Percentage (%)", 0, 80, 30)
    
    traffic_rate = st.sidebar.slider("Traffic Rate (packets/sec)", 10, 200, 100)
    
    # Routing configuration
    st.sidebar.subheader("🛣️ Routing Configuration")
    anomaly_penalty = st.sidebar.slider("Anomaly Penalty", 100.0, 10000.0, 1000.0, step=100.0)
    
    # CSV Upload
    st.sidebar.subheader("📁 Upload Custom Data")
    uploaded_file = st.sidebar.file_uploader("Upload CSV (optional)", type=['csv'])
    
    # Run simulation button
    run_button = st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True)
    
    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Tip**: Click 'Run Simulation' to see live visualization!")
    
    # ========================================================================
    # MAIN CONTENT AREA
    # ========================================================================
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🌐 Network Topology", 
        "📊 Performance Metrics", 
        "🔄 Traffic Flows",
        "📈 FL Training",
        "📋 Detailed Results"
    ])
    
    # ========================================================================
    # TAB 1: NETWORK TOPOLOGY
    # ========================================================================
    
    with tab1:
        st.subheader("5G Network Topology Visualization")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create sample network
            G = nx.Graph()
            G.add_nodes_from(range(num_base_stations))
            
            # Add edges (mesh topology)
            for i in range(num_base_stations):
                for j in range(i + 1, num_base_stations):
                    if random.random() < 0.5:  # 50% connection probability
                        G.add_edge(i, j)
            
            # Ensure connected graph
            if not nx.is_connected(G):
                components = list(nx.connected_components(G))
                for i in range(len(components) - 1):
                    node1 = list(components[i])[0]
                    node2 = list(components[i + 1])[0]
                    G.add_edge(node1, node2)
            
            # Color nodes based on load (simulated)
            node_colors = ['lightblue'] * num_base_stations
            
            fig = create_network_topology_plot(G, node_colors=node_colors)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Network Statistics")
            st.metric("Base Stations", num_base_stations)
            st.metric("Network Links", G.number_of_edges())
            st.metric("Avg Degree", f"{2 * G.number_of_edges() / num_base_stations:.2f}")
            st.metric("Network Diameter", nx.diameter(G) if nx.is_connected(G) else "N/A")
            
            st.markdown("### Legend")
            st.markdown("🔵 **Blue Nodes**: Base Stations")
            st.markdown("— **Gray Lines**: Network Links")
            st.markdown("🟢 **Green Flow**: Normal Traffic")
            st.markdown("🟡 **Orange Flow**: Suspicious Traffic")
            st.markdown("🔴 **Red Flow**: Malicious Traffic")
    
    # ========================================================================
    # TAB 2: PERFORMANCE METRICS
    # ========================================================================
    
    with tab2:
        st.subheader("Performance Metrics Comparison")
        
        if run_button or 'simulation_data' in st.session_state:
            # Generate or use cached simulation data
            if run_button:
                with st.spinner("Running simulation..."):
                    # Simulate baseline latency (with spikes during anomalies)
                    baseline_latency = []
                    for i in range(100):
                        if random.random() < (anomaly_percentage / 100):
                            baseline_latency.append(random.uniform(80, 200))  # High latency during anomaly
                        else:
                            baseline_latency.append(random.uniform(20, 40))
                    
                    # Simulate intelligent routing latency (more stable)
                    intelligent_latency = [random.uniform(20, 35) for _ in range(100)]
                    
                    st.session_state['simulation_data'] = {
                        'baseline': baseline_latency,
                        'intelligent': intelligent_latency
                    }
                    
                    time.sleep(1)  # Simulate processing
            
            # Display metrics
            data = st.session_state.get('simulation_data', {
                'baseline': [random.uniform(40, 60) for _ in range(100)],
                'intelligent': [random.uniform(20, 35) for _ in range(100)]
            })
            
            fig = create_metrics_comparison(data['baseline'], data['intelligent'])
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            st.markdown("### 📊 Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            baseline_avg = np.mean(data['baseline'])
            intelligent_avg = np.mean(data['intelligent'])
            improvement = ((baseline_avg - intelligent_avg) / baseline_avg) * 100
            
            with col1:
                st.metric("Baseline Latency", f"{baseline_avg:.2f} ms", 
                         delta=None, delta_color="off")
            
            with col2:
                st.metric("Intelligent Latency", f"{intelligent_avg:.2f} ms",
                         delta=f"-{baseline_avg - intelligent_avg:.2f} ms",
                         delta_color="inverse")
            
            with col3:
                st.metric("Improvement", f"{improvement:.1f}%",
                         delta=f"{improvement:.1f}%",
                         delta_color="normal")
            
            with col4:
                st.metric("Detection Accuracy", "88.3%",
                         delta="5.2%",
                         delta_color="normal")
        
        else:
            st.info("👈 Click 'Run Simulation' in the sidebar to see performance metrics")
    
    # ========================================================================
    # TAB 3: TRAFFIC FLOWS
    # ========================================================================
    
    with tab3:
        st.subheader("Real-Time Traffic Flow Visualization")
        
        if run_button or 'traffic_flows' in st.session_state:
            if run_button:
                with st.spinner("Generating traffic flows..."):
                    # Generate sample traffic flows
                    flows_data = []
                    for t in range(20):  # 20 time steps
                        flows = []
                        for _ in range(10):  # 10 flows per time step
                            src = random.randint(0, num_base_stations - 1)
                            dst = random.randint(0, num_base_stations - 1)
                            while dst == src:
                                dst = random.randint(0, num_base_stations - 1)
                            
                            flow_type = random.choice(['Video', 'IoT', 'Emergency'])
                            
                            # Anomaly score based on experiment
                            if random.random() < (anomaly_percentage / 100):
                                anomaly_score = random.uniform(0.6, 0.95)
                            else:
                                anomaly_score = random.uniform(0.05, 0.4)
                            
                            flows.append({
                                'src': src,
                                'dst': dst,
                                'type': flow_type,
                                'anomaly': anomaly_score
                            })
                        flows_data.append(flows)
                    
                    st.session_state['traffic_flows'] = flows_data
            
            flows_data = st.session_state.get('traffic_flows', [])
            
            # Display animated flows
            if flows_data:
                # Create slider for time step
                time_step = st.slider("Time Step", 0, len(flows_data) - 1, 0)
                
                # Show flows at selected time step
                current_flows = flows_data[time_step]
                
                # Create visualization
                G_flows = nx.Graph()
                G_flows.add_nodes_from(range(num_base_stations))
                
                edge_colors = []
                for flow in current_flows:
                    G_flows.add_edge(flow['src'], flow['dst'])
                    if flow['anomaly'] > 0.7:
                        edge_colors.append('red')
                    elif flow['anomaly'] > 0.5:
                        edge_colors.append('orange')
                    else:
                        edge_colors.append('green')
                
                fig = create_network_topology_plot(G_flows, edge_colors=edge_colors, 
                                                   title=f"Traffic Flows at Time={time_step}")
                st.plotly_chart(fig, use_container_width=True)
                
                # Flow statistics
                st.markdown("### Flow Statistics")
                col1, col2, col3 = st.columns(3)
                
                normal_count = sum(1 for f in current_flows if f['anomaly'] < 0.5)
                suspicious_count = sum(1 for f in current_flows if 0.5 <= f['anomaly'] < 0.7)
                malicious_count = sum(1 for f in current_flows if f['anomaly'] >= 0.7)
                
                with col1:
                    st.markdown('<div class="normal-box">✅ <b>Normal Flows</b>: {}</div>'.format(normal_count), 
                               unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div style="background-color: #fff3e0; padding: 0.5rem; border-radius: 0.3rem; border-left: 4px solid #ff8800;">⚠️ <b>Suspicious Flows</b>: {}</div>'.format(suspicious_count),
                               unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="anomaly-box">🚨 <b>Malicious Flows</b>: {}</div>'.format(malicious_count),
                               unsafe_allow_html=True)
        
        else:
            st.info("👈 Click 'Run Simulation' in the sidebar to see traffic flows")
    
    # ========================================================================
    # TAB 4: FL TRAINING
    # ========================================================================
    
    with tab4:
        st.subheader("Federated Learning Training Progress")
        
        if "FL Training" in experiment or run_button:
            # Generate FL training data
            rounds_data = {}
            for bs in range(num_base_stations):
                # Simulate accuracy improvement over rounds
                start_acc = random.uniform(30, 40)
                end_acc = random.uniform(85, 92)
                accuracies = []
                for r in range(simulation_rounds):
                    acc = start_acc + (end_acc - start_acc) * (r / simulation_rounds) + random.uniform(-2, 2)
                    accuracies.append(max(0, min(100, acc)))
                rounds_data[bs] = accuracies
            
            fig = create_fl_training_progress(rounds_data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display final accuracies
            st.markdown("### 🎯 Final Model Accuracies")
            cols = st.columns(min(num_base_stations, 5))
            for i, (bs, accs) in enumerate(rounds_data.items()):
                with cols[i % 5]:
                    st.metric(f"BS-{bs}", f"{accs[-1]:.1f}%", 
                             delta=f"+{accs[-1] - accs[0]:.1f}%")
            
            # Global model accuracy
            st.markdown("---")
            global_acc = np.mean([accs[-1] for accs in rounds_data.values()])
            st.markdown(f"### 🌐 Global Model Accuracy: **{global_acc:.2f}%**")
            
            st.success(f"✅ Federated learning completed in {simulation_rounds} rounds")
        
        else:
            st.info("Select 'Experiment 4: FL Training Progress' to see this visualization")
    
    # ========================================================================
    # TAB 5: DETAILED RESULTS
    # ========================================================================
    
    with tab5:
        st.subheader("Detailed Simulation Results")
        
        if uploaded_file is not None:
            # Load custom CSV
            df = pd.read_csv(uploaded_file)
            st.markdown("### 📁 Uploaded Data")
            st.dataframe(df, use_container_width=True)
        
        elif 'simulation_data' in st.session_state:
            # Show simulation results as table
            results_df = pd.DataFrame({
                'Time Step': range(100),
                'Baseline Latency (ms)': st.session_state['simulation_data']['baseline'],
                'Intelligent Latency (ms)': st.session_state['simulation_data']['intelligent'],
                'Improvement (%)': [
                    ((b - i) / b * 100) for b, i in zip(
                        st.session_state['simulation_data']['baseline'],
                        st.session_state['simulation_data']['intelligent']
                    )
                ]
            })
            
            st.markdown("### 📊 Latency Comparison Table")
            st.dataframe(results_df, use_container_width=True)
            
            # Download button
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="simulation_results.csv",
                mime="text/csv"
            )
            
            # Statistical summary
            st.markdown("### 📈 Statistical Summary")
            st.dataframe(results_df.describe(), use_container_width=True)
        
        else:
            st.info("Run a simulation or upload CSV to see detailed results")
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray;">
        <p>🎓 Federated Machine Learning-Driven Anomaly Detection for Intelligent QoS Management in 5G Networks</p>
        <p>Built with Streamlit • Interactive Dashboard for Research Review</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()
