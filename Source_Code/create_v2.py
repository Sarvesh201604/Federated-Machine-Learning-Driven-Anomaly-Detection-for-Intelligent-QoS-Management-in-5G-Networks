import os

source_file = r'c:\Users\sayee\Downloads\Federated-Machine-Learning-Driven-Anomaly-Detection-for-Intelligent-QoS-Management-in-5G-Networks\viva_presentation_dashboard.py'
dest_file = r'c:\Users\sayee\Downloads\Federated-Machine-Learning-Driven-Anomaly-Detection-for-Intelligent-QoS-Management-in-5G-Networks\viva_presentation_dashboard_v2.py'

with open(source_file, 'r', encoding='utf-8') as f:
    code = f.read()

extra_code = """
# =============================================================================
# STEP 4: ADVANCED INTELLIGENT NETWORK MANAGER FEATURES
# =============================================================================

st.markdown("---")
st.header("Step 4: Advanced Intelligent Network Manager Features (Added)")
st.markdown("This section demonstrates the three advanced QoS and security features integrating the **Intelligent Network Manager**.")

def draw_advanced_network(G, path=None, title="", blocked=False, congested_nodes=None):
    import plotly.graph_objects as go
    import networkx as nx
    # Fixed layout for our 6-node network
    pos = {
        0: (0, 1), 1: (1, 2), 2: (1, 0), 3: (2, 2), 4: (2, 0), 5: (3, 1)
    }
    
    edge_traces = []
    
    # Active edges
    active_edges = set()
    if path and not blocked:
        for i in range(len(path)-1):
            active_edges.add((path[i], path[i+1]))
            active_edges.add((path[i+1], path[i]))
            
    for u, v in G.edges():
        x0, y0 = pos.get(u, (0,0))
        x1, y1 = pos.get(v, (0,0))
        
        in_path = (u,v) in active_edges or (v,u) in active_edges
        color = '#a3a3a3'
        width = 2
        
        if in_path:
            width = 4
            color = '#21c354' # green
            
        edge_trace = go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode='lines',
            line=dict(width=width, color=color),
            hoverinfo='none'
        )
        edge_traces.append(edge_trace)

    congested_nodes = congested_nodes or []
    node_x = [pos.get(n, (0,0))[0] for n in G.nodes()]
    node_y = [pos.get(n, (0,0))[1] for n in G.nodes()]
    node_colors = ['#ff4b4b' if n in congested_nodes else '#1f77b4' for n in G.nodes()]
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(size=50, color=node_colors, line=dict(width=2, color='white')),
        text=[f"BS-{n}" for n in G.nodes()],
        textposition="middle center",
        textfont=dict(color='white', weight='bold'),
        hoverinfo='none'
    )
    
    data = edge_traces + [node_trace]
    
    if blocked and path:
        x0, y0 = pos.get(path[0], (0,0))
        blocked_trace = go.Scatter(
            x=[x0], y=[y0],
            mode='markers+text',
            marker=dict(size=45, color='#ff4b4b', symbol='x', line=dict(width=2, color='white')),
            text=["BLOCKED"],
            textposition="bottom center",
            textfont=dict(color='red', weight='bold', size=16),
            hoverinfo='none'
        )
        data.append(blocked_trace)
        
    fig = go.Figure(data=data)
    fig.update_layout(
        title=title,
        showlegend=False,
        hovermode='closest',
        plot_bgcolor='white',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=350,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

st.markdown("We have prepared a separate mock network topology here mapped to the 3 demos of Intelligent Network Manager. Click on the buttons below to run the simulation dynamically.")

col_adv1, col_adv2, col_adv3 = st.columns(3)

def get_adv_manager():
    import networkx as nx
    G_adv = nx.Graph()
    G_adv.add_nodes_from(range(6))
    edges = [
        (0, 1, {'base_latency': 10, 'capacity': 100.0, 'current_load': 0.0}),
        (0, 2, {'base_latency': 15, 'capacity': 100.0, 'current_load': 0.0}),
        (1, 3, {'base_latency': 12, 'capacity': 100.0, 'current_load': 0.0}),
        (1, 2, {'base_latency': 20, 'capacity': 100.0, 'current_load': 0.0}),
        (2, 4, {'base_latency': 18, 'capacity': 100.0, 'current_load': 0.0}),
        (3, 5, {'base_latency': 10, 'capacity': 100.0, 'current_load': 0.0}),
        (4, 5, {'base_latency': 15, 'capacity': 100.0, 'current_load': 0.0}),
        (3, 4, {'base_latency': 25, 'capacity': 100.0, 'current_load': 0.0})
    ]
    for u, v, attrs in edges:
        G_adv.add_edge(u, v, **attrs)
        
    class MockFLModel:
        def predict_anomaly_score(self, _): return 0.0
        
    try:
        from intelligent_network_manager import IntelligentNetworkManager
        return IntelligentNetworkManager(G_adv, MockFLModel(), num_nodes=6)
    except ImportError:
        return None

manager = get_adv_manager()

if manager:
    with col_adv1:
        st.subheader("1. Load Balancing")
        st.info("High throughput detected on BS-0 -> BS-1 edge. Normal traffic is rerouted.")
        if st.button("Run Load Balancer Demo", key="lb_demo"):
            manager.simulate_link_traffic(0, 1, throughput=85, capacity=100)
            traffic = {'latency': 15, 'throughput': 85, 'packet_loss': 0.01, 'jitter': 2, 'queue_length': 30, 'load': 0.85, 'traffic_type': 0}
            res = manager.smart_route(0, 5, traffic, anomaly_score=0.05)
            
            st.success(f"Action: {res['action']}")
            st.write(f"Path taken: {res['path']}")
            st.caption(f"Reason: {res['reason']}")
            st.plotly_chart(draw_advanced_network(manager.graph, path=res['path'], title="Rerouted to avoid high link load"), use_container_width=True)

    with col_adv2:
        st.subheader("2. Self-Healing")
        st.warning("BS-2 becomes highly congested! The network alerts peers & reroutes.")
        if st.button("Run Self-Healing Demo", key="sh_demo"):
            manager.simulate_node_load_event(2, load=0.91)
            traffic = {'latency': 15, 'throughput': 50, 'packet_loss': 0.01, 'jitter': 2, 'traffic_type': 0}
            res = manager.smart_route(0, 4, traffic, anomaly_score=0.1, observer_node=0)
            
            st.success(f"Action: {res['action']}")
            st.write(f"Path taken: {res['path']}")
            st.caption(f"Reason: {res['reason']}")
            st.plotly_chart(draw_advanced_network(manager.graph, path=res['path'], congested_nodes=[2], title="Rerouted around BS-2 (Red)"), use_container_width=True)

    with col_adv3:
        st.subheader("3. Distributed Blacklist")
        st.error("BS-0 detects attack, broadcasts fingerprint. BS-3 blocks it instantly.")
        if st.button("Run Blacklisting Demo", key="bl_demo"):
            traffic_att = {'latency': 250, 'throughput': 5, 'packet_loss': 0.45, 'jitter': 80, 'traffic_type': 1}
            # Node 0 reports anomaly
            manager.smart_route(0, 5, traffic_att, anomaly_score=0.85, observer_node=0)
            
            # Node 3 receives identical flow
            res2 = manager.smart_route(3, 5, traffic_att, anomaly_score=0.85, observer_node=3)
            
            st.success(f"Action: {res2['action']}")
            st.write(f"Path taken: {res2['path']}")
            st.caption(f"Reason: {res2['reason']}")
            st.plotly_chart(draw_advanced_network(manager.graph, path=[3], blocked=True, title="Blocked at BS-3 (Broadcasted Blacklist)"), use_container_width=True)
else:
    st.error("Could not import intelligent_network_manager.py! Please ensure it's in the same directory.")
"""

with open(dest_file, 'w', encoding='utf-8') as f:
    f.write(code + '\n' + extra_code)
print("File successfully created!")
