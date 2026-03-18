"""
🎯 ANIMATED COLORFUL DASHBOARD - 5G ANOMALY DETECTION & ROUTING
================================================================

FEATURES:
- 🎨 Colorful animated UI
- 🔄 Auto-playing traffic simulations
- 📊 Real-time updating metrics
- 🌈 Gradient effects and animations
- 🎬 Video-like flow visualization
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

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="🚀 5G FL-QoS Live Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# COLORFUL CSS STYLING
# ============================================================================

st.markdown("""
<style>
    /* Main background gradient */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Animated header */
    .animated-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        padding: 2rem;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1, #FFA07A, #98D8C8);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient 3s ease infinite;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Colorful metric boxes */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        margin: 0.5rem 0;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.05); }
    }
    
    /* Success metric */
    .metric-success {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    
    /* Warning metric */
    .metric-warning {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    
    /* Info metric */
    .metric-info {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    
    /* Animated status badges */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: bold;
        color: white;
        margin: 0.5rem;
        animation: bounce 1s infinite;
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    .badge-normal {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .badge-suspicious {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    
    .badge-malicious {
        background: linear-gradient(135deg, #ff0844 0%, #ffb199 100%);
    }
    
    /* Progress bar animation */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2, #667eea);
        background-size: 200% 100%;
        animation: progressAnim 2s linear infinite;
    }
    
    @keyframes progressAnim {
        0% { background-position: 100% 0; }
        100% { background-position: -100% 0; }
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        transform: scale(1.1);
        box-shadow: 0 8px 16px rgba(0,0,0,0.3);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: linear-gradient(90deg, #667eea, #764ba2);
        padding: 1rem;
        border-radius: 15px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 10px;
        padding: 1rem 2rem;
        font-weight: bold;
    }
    
    /* Animation for network nodes */
    @keyframes nodeGlow {
        0%, 100% { filter: drop-shadow(0 0 5px rgba(102, 126, 234, 0.8)); }
        50% { filter: drop-shadow(0 0 20px rgba(118, 75, 162, 1)); }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_animated_network(G, traffic_data=None, step=0):
    """Create animated network with flowing traffic"""
    pos = nx.spring_layout(G, seed=42, k=2, iterations=50)
    
    # Create edge traces with animation
    edge_traces = []
    
    if traffic_data and step < len(traffic_data):
        flows = traffic_data[step]
        for flow in flows:
            src, dst, anomaly = flow['src'], flow['dst'], flow['anomaly']
            
            if src in pos and dst in pos:
                x0, y0 = pos[src]
                x1, y1 = pos[dst]
                
                # Color based on anomaly score
                if anomaly > 0.7:
                    color = 'rgba(255, 0, 68, 0.8)'
                    width = 6
                elif anomaly > 0.5:
                    color = 'rgba(255, 165, 0, 0.8)'
                    width = 5
                else:
                    color = 'rgba(56, 239, 125, 0.8)'
                    width = 4
                
                # Create flowing effect
                edge_trace = go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=width, color=color),
                    hoverinfo='skip',
                    showlegend=False
                )
                edge_traces.append(edge_trace)
                
                # Add animated arrows (particles)
                progress = (step % 20) / 20
                arrow_x = x0 + (x1 - x0) * progress
                arrow_y = y0 + (y1 - y0) * progress
                
                arrow_trace = go.Scatter(
                    x=[arrow_x],
                    y=[arrow_y],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color=color,
                        symbol='arrow',
                        line=dict(width=2, color='white')
                    ),
                    hoverinfo='skip',
                    showlegend=False
                )
                edge_traces.append(arrow_trace)
    
    # Create node trace
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(
            size=60,
            color=['#667eea', '#764ba2', '#4facfe', '#38ef7d', '#f093fb'][:len(G.nodes())],
            line=dict(width=4, color='white'),
            symbol='circle'
        ),
        text=[f"<b>BS-{n}</b>" for n in G.nodes()],
        textposition="middle center",
        textfont=dict(size=14, color='white', family='Arial Black'),
        hovertext=[f"Base Station {n}<br>Load: {random.randint(20,80)}%<br>Status: Active" for n in G.nodes()],
        hoverinfo='text',
        showlegend=False
    )
    
    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])
    
    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=500
    )
    
    return fig


def create_animated_metrics(baseline_data, intelligent_data, step):
    """Create animated real-time metrics"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("📡 Real-Time Latency", "📦 Packet Delivery Ratio", 
                        "🚀 Throughput", "🎯 Detection Accuracy"),
        specs=[[{"type": "scatter"}, {"type": "indicator"}],
               [{"type": "scatter"}, {"type": "indicator"}]]
    )
    
    # Animated latency line
    current_baseline = baseline_data[:step+1]
    current_intelligent = intelligent_data[:step+1]
    
    fig.add_trace(
        go.Scatter(
            x=list(range(len(current_baseline))), 
            y=current_baseline,
            name="Baseline",
            line=dict(color='#ff0844', width=3),
            fill='tozeroy',
            fillcolor='rgba(255, 8, 68, 0.2)'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=list(range(len(current_intelligent))), 
            y=current_intelligent,
            name="Intelligent",
            line=dict(color='#38ef7d', width=3),
            fill='tozeroy',
            fillcolor='rgba(56, 239, 125, 0.2)'
        ),
        row=1, col=1
    )
    
    # PDR Gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=96,
            delta={'reference': 85, 'increasing': {'color': "#38ef7d"}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#667eea"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 85], 'color': "#f093fb"},
                    {'range': [85, 100], 'color': "#38ef7d"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ),
        row=1, col=2
    )
    
    # Throughput
    throughput = np.sin(np.linspace(0, step/10, step+1)) * 20 + 50
    fig.add_trace(
        go.Scatter(
            x=list(range(len(throughput))),
            y=throughput,
            name="Throughput",
            line=dict(color='#4facfe', width=3, shape='spline'),
            fill='tozeroy',
            fillcolor='rgba(79, 172, 254, 0.3)'
        ),
        row=2, col=1
    )
    
    # Accuracy Gauge
    accuracy = min(88.3, 35 + (step / 100) * 53.3)
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=accuracy,
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#764ba2"},
                'steps': [
                    {'range': [0, 50], 'color': "#ffb199"},
                    {'range': [50, 75], 'color': "#f5576c"},
                    {'range': [75, 100], 'color': "#38ef7d"}
                ]
            },
            number={'suffix': "%", 'font': {'size': 40}}
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=12)
    )
    
    return fig


def create_3d_network_viz(G, step):
    """Create 3D network visualization"""
    pos = nx.spring_layout(G, dim=3, seed=42)
    
    # Create 3D edges
    edge_x, edge_y, edge_z = [], [], []
    for edge in G.edges():
        x0, y0, z0 = pos[edge[0]]
        x1, y1, z1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])
    
    edge_trace = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='rgba(125, 125, 125, 0.5)', width=2),
        hoverinfo='none'
    )
    
    # Create 3D nodes
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_z = [pos[n][2] for n in G.nodes()]
    
    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers+text',
        marker=dict(
            size=20,
            color=['#667eea', '#764ba2', '#4facfe', '#38ef7d', '#f093fb'][:len(G.nodes())],
            line=dict(color='white', width=2)
        ),
        text=[f"BS-{n}" for n in G.nodes()],
        textposition="top center",
        hovertext=[f"Base Station {n}" for n in G.nodes()],
        hoverinfo='text'
    )
    
    fig = go.Figure(data=[edge_trace, node_trace])
    
    fig.update_layout(
        showlegend=False,
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, title=''),
            yaxis=dict(showgrid=False, showticklabels=False, title=''),
            zaxis=dict(showgrid=False, showticklabels=False, title=''),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        height=500,
        margin=dict(l=0, r=0, b=0, t=0)
    )
    
    # Add camera rotation animation
    camera = dict(
        eye=dict(
            x=2*np.cos(step/10),
            y=2*np.sin(step/10),
            z=1.5
        )
    )
    fig.update_layout(scene_camera=camera)
    
    return fig


# ============================================================================
# MAIN DASHBOARD
# ============================================================================

def main():
    # Animated header
    st.markdown('<h1 class="animated-header">🚀 5G FEDERATED LEARNING QoS DASHBOARD 🎯</h1>', 
                unsafe_allow_html=True)
    
    # ========================================================================
    # SIDEBAR CONTROLS
    # ========================================================================
    
    with st.sidebar:
        st.markdown("## ⚙️ Control Panel")
        
        # Experiment selector
        experiment = st.selectbox(
            "🔬 Select Experiment",
            [
                "🟢 Experiment 1: Normal Traffic",
                "🟡 Experiment 2: Mixed (30% Anomalies)",
                "🔴 Experiment 3: Heavy Attack (60%)",
                "📈 Experiment 4: FL Training",
                "⚡ Experiment 5: Comparison"
            ]
        )
        
        st.markdown("---")
        
        # Network config
        st.markdown("### 🌐 Network Configuration")
        num_bs = st.slider("📡 Base Stations", 3, 10, 5)
        
        # Traffic config
        st.markdown("### 🚦 Traffic Settings")
        if "Normal" in experiment:
            anomaly_pct = 0
        elif "Heavy" in experiment:
            anomaly_pct = 60
        else:
            anomaly_pct = st.slider("⚠️ Anomaly %", 0, 80, 30)
        
        # Time step slider for animation
        st.markdown("### ⏱️ Animation Control")
        time_step = st.slider("Time Step", 0, 99, 0, key="time_step")
        
        st.markdown("---")
        
        # Control buttons
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            run_sim = st.button("🚀 RUN", type="primary")
        with col_btn2:
            if st.button("🔄 NEXT"):
                st.session_state.step = (time_step + 1) % 100
                st.rerun()
        
        st.markdown("---")
        
        # Live status
        st.markdown("### 📊 Live Status")
        status_placeholder = st.empty()
    
    # ========================================================================
    # MAIN CONTENT
    # ========================================================================
    
    # Initialize session state
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'step' not in st.session_state:
        st.session_state.step = 0
    if 'traffic_data' not in st.session_state:
        st.session_state.traffic_data = []
    
    # Start simulation
    if run_sim:
        st.session_state.running = True
        st.session_state.step = 0
    
    # Generate network
    G = nx.Graph()
    G.add_nodes_from(range(num_bs))
    for i in range(num_bs):
        for j in range(i+1, min(i+3, num_bs)):
            G.add_edge(i, j)
    
    # Generate traffic data
    if not st.session_state.traffic_data:
        traffic_data = []
        for t in range(100):
            flows = []
            for _ in range(15):
                src = random.randint(0, num_bs-1)
                dst = random.randint(0, num_bs-1)
                while dst == src:
                    dst = random.randint(0, num_bs-1)
                
                anomaly =random.uniform(0.05, 0.4) if random.random() > (anomaly_pct/100) else random.uniform(0.6, 0.95)
                
                flows.append({'src': src, 'dst': dst, 'anomaly': anomaly})
            traffic_data.append(flows)
        st.session_state.traffic_data = traffic_data
    
    # Generate latency data
    if 'latency_data' not in st.session_state:
        baseline = [random.uniform(40, 60) if random.random() > (anomaly_pct/100) 
                   else random.uniform(80, 200) for _ in range(100)]
        intelligent = [random.uniform(20, 35) for _ in range(100)]
        st.session_state.latency_data = (baseline, intelligent)
    
    # ========================================================================
    # TABS
    # ========================================================================
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌐 LIVE NETWORK", 
        "📊 METRICS", 
        "🎯 3D VIEW",
        "📈 ANALYTICS"
    ])
    
    # ========================================================================
    # TAB 1: LIVE NETWORK
    # ========================================================================
    
    with tab1:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            network_placeholder = st.empty()
        
        with col2:
            st.markdown("### 📊 Network Stats")
            
            # Count flows by type
            if time_step < len(st.session_state.traffic_data):
                current_flows = st.session_state.traffic_data[time_step]
                normal = sum(1 for f in current_flows if f['anomaly'] < 0.5)
                suspicious = sum(1 for f in current_flows if 0.5 <= f['anomaly'] < 0.7)
                malicious = sum(1 for f in current_flows if f['anomaly'] >= 0.7)
                
                st.markdown(f"""
                <div class="metric-card metric-success">
                    🟢 NORMAL<br>{normal} flows
                </div>
                <div class="metric-card metric-warning">
                    🟡 SUSPICIOUS<br>{suspicious} flows
                </div>
                <div class="metric-card" style="background: linear-gradient(135deg, #ff0844 0%, #ffb199 100%);">
                    🔴 MALICIOUS<br>{malicious} flows
                </div>
                """, unsafe_allow_html=True)
            
            # Progress bar
            progress = time_step / 100
            st.progress(progress)
            st.markdown(f"**Step {time_step}/100**")
    
    # ========================================================================
    # TAB 2: METRICS
    # ========================================================================
    
    with tab2:
        metrics_placeholder = st.empty()
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        baseline_avg = np.mean(st.session_state.latency_data[0])
        intelligent_avg = np.mean(st.session_state.latency_data[1])
        improvement = ((baseline_avg - intelligent_avg) / baseline_avg) * 100
        
        with col1:
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #ff0844 0%, #ffb199 100%);">
                📊 BASELINE<br>{baseline_avg:.1f} ms
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card metric-success">
                ⚡ INTELLIGENT<br>{intelligent_avg:.1f} ms
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card metric-info">
                🎯 IMPROVEMENT<br>{improvement:.1f}%
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                🔍 ACCURACY<br>88.3%
            </div>
            """, unsafe_allow_html=True)
    
    # ========================================================================
    # TAB 3: 3D VIEW
    # ========================================================================
    
    with tab3:
        viz_3d_placeholder = st.empty()
        
        st.markdown("""
        <div style="text-align: center; color: white; font-size: 1.2rem; margin-top: 2rem;">
            🌟 <b>Rotating 3D Network Visualization</b> 🌟<br>
            Watch the network topology in full 3D perspective!
        </div>
        """, unsafe_allow_html=True)
    
    # ========================================================================
    # TAB 4: ANALYTICS
    # ========================================================================
    
    with tab4:
        st.markdown("### 📊 Detailed Analytics Dashboard")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # FL Training Progress
            st.markdown("#### 📈 Federated Learning Progress")
            rounds = list(range(1, 11))
            accuracies = [35 + (88-35) * (r/10) + random.uniform(-2, 2) for r in rounds]
            
            fig_fl = go.Figure()
            fig_fl.add_trace(go.Scatter(
                x=rounds,
                y=accuracies,
                mode='lines+markers',
                line=dict(color='#38ef7d', width=4),
                marker=dict(size=12, color='#667eea', line=dict(width=2, color='white')),
                fill='tozeroy',
                fillcolor='rgba(56, 239, 125, 0.3)'
            ))
            fig_fl.update_layout(
                xaxis_title="Training Round",
                yaxis_title="Accuracy (%)",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=300
            )
            st.plotly_chart(fig_fl, width='stretch')
        
        with col2:
            # Traffic Distribution
            st.markdown("#### 🎯 Traffic Distribution")
            current_flows = st.session_state.traffic_data[time_step]
            normal = sum(1 for f in current_flows if f['anomaly'] < 0.5)
            suspicious = sum(1 for f in current_flows if 0.5 <= f['anomaly'] < 0.7)
            malicious = sum(1 for f in current_flows if f['anomaly'] >= 0.7)
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Normal', 'Suspicious', 'Malicious'],
                values=[normal, suspicious, malicious],
                marker=dict(colors=['#38ef7d', '#f093fb', '#ff0844']),
                hole=0.4
            )])
            fig_pie.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=300
            )
            st.plotly_chart(fig_pie, width='stretch')
        
        # Comparison table
        st.markdown("#### 📋 Performance Comparison")
        comparison_df = pd.DataFrame({
            'Metric': ['Avg Latency', 'Packet Delivery', 'Detection Accuracy', 'Throughput'],
            'Baseline': ['51.93 ms', '85%', '75%', '45 Mbps'],
            'Intelligent': ['26.76 ms', '96%', '88.3%', '68 Mbps'],
            'Improvement': ['48.47% ⬇️', '11% ⬆️', '13.3% ⬆️', '51% ⬆️']
        })
        
        st.dataframe(
            comparison_df.style.apply(
                lambda x: ['background: linear-gradient(90deg, #667eea, #764ba2); color: white']*len(x), 
                axis=1
            ),
            width='stretch'
        )
    
    # ========================================================================
    # ANIMATION LOOP
    # ========================================================================
    
    # Use the time_step from slider
    current_step = time_step
    
    if st.session_state.running:
        # Update network visualization
        with tab1:
            with col1:
                fig_network = create_animated_network(
                    G, 
                    st.session_state.traffic_data, 
                    current_step
                )
                network_placeholder.plotly_chart(fig_network, width='stretch')
        
        # Update metrics
        with tab2:
            fig_metrics = create_animated_metrics(
                st.session_state.latency_data[0],
                st.session_state.latency_data[1],
                current_step
            )
            metrics_placeholder.plotly_chart(fig_metrics, width='stretch')
        
        # Update 3D view
        with tab3:
            fig_3d = create_3d_network_viz(G, current_step)
            viz_3d_placeholder.plotly_chart(fig_3d, width='stretch')
        
        # Update status
        with status_placeholder.container():
            status = "🟢 SYSTEM ACTIVE" if current_step % 2 == 0 else "🟢 MONITORING..."
            st.markdown(f"""
            <div class="status-badge badge-normal">
                {status}
            </div>
            """, unsafe_allow_html=True)
            st.metric("Current Step", current_step)


# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()
