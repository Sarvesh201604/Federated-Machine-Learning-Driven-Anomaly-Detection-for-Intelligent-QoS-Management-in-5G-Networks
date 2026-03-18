"""
INTELLIGENT NETWORK MANAGER
============================
Novel Add-On Features for the Federated ML QoS System in 5G Networks

Three Key Innovations Implemented Here:

  FEATURE 1 - THROUGHPUT LOAD BALANCER:
    When a link/node has very HIGH throughput (near capacity) but is NOT
    an anomaly or attack, the system intelligently REROUTES traffic to
    under-utilized links to maintain QoS. This is pure load-balancing,
    not security-based routing.

  FEATURE 2 - CONGESTION-AWARE SELF-HEALING:
    All nodes share real-time congestion status with each other (distributed
    intelligence). When any node becomes congested, ALL nodes automatically
    update their routing tables and find alternate paths WITHOUT human
    intervention. Simulates network-wide distributed awareness.

  FEATURE 3 - DISTRIBUTED ANOMALY BLACKLISTING:
    When ONE node detects a suspicious/anomalous flow (via FL model), it
    immediately BROADCASTS a "threat alert" to ALL other nodes. Every node
    then adds that flow's fingerprint to its local blacklist and refuses to
    forward matching packets. Simulates cooperative intrusion prevention.

NOVELTY STATEMENT:
    Extended Cost = Latency + Load + (Anomaly × Penalty) + ThroughputPressure
    + CongestionDetour + BlacklistCheck
"""

import networkx as nx
import numpy as np
import threading
import time
import hashlib
from collections import defaultdict, deque
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set


# =============================================================================
# FEATURE 1: HIGH-THROUGHPUT LOAD BALANCER
# =============================================================================

class ThroughputLoadBalancer:
    """
    Detects high-throughput (non-anomaly) conditions on links and nodes
    and proactively reroutes traffic to maintain QoS.

    Logic:
      - If link utilization > HIGH_THROUGHPUT_THRESHOLD, the link is
        under HIGH LOAD (but not attacked).
      - Traffic is redirected to links with more headroom.
      - The anomaly model is consulted first to confirm it is NOT an attack.
    """

    THROUGHPUT_THRESHOLD = 0.75   # 75% utilization = HIGH (needs reroute)
    CRITICAL_THRESHOLD   = 0.90   # 90% utilization = CRITICAL
    ANOMALY_MAX_SCORE    = 0.40   # If anomaly score > this, treat as attack, not load

    def __init__(self, graph: nx.Graph, global_model):
        self.graph = graph
        self.global_model = global_model

        # Per-link throughput tracking (rolling window)
        self.link_throughput_history: Dict[tuple, deque] = defaultdict(
            lambda: deque(maxlen=20)
        )
        # Per-node utilization
        self.node_utilization: Dict[int, float] = {}

        # Stats
        self.stats = {
            'total_checks': 0,
            'rerouted_high_throughput': 0,
            'not_rerouted_anomaly': 0,
            'flows_load_balanced': 0
        }

    def record_flow(self, u: int, v: int, throughput: float, capacity: float):
        """Record a traffic flow on a link for utilization tracking."""
        utilization = min(throughput / max(capacity, 1.0), 1.0)
        self.link_throughput_history[(u, v)].append(utilization)
        self.link_throughput_history[(v, u)].append(utilization)

        # Update graph edge attribute
        if self.graph.has_edge(u, v):
            self.graph[u][v]['current_load'] = utilization

    def get_link_utilization(self, u: int, v: int) -> float:
        """Return average utilization over recent history for a link."""
        history = self.link_throughput_history.get((u, v), deque())
        if not history:
            # Fall back to graph attribute
            if self.graph.has_edge(u, v):
                return self.graph[u][v].get('current_load', 0.0)
            return 0.0
        return float(np.mean(history))

    def is_high_throughput_not_attack(self, u: int, v: int,
                                      traffic_features: dict,
                                      anomaly_score: float) -> bool:
        """
        Returns True only if:
          1. Link utilization is above HIGH_THROUGHPUT_THRESHOLD
          2. The ML anomaly score is BELOW ANOMALY_MAX_SCORE (not an attack)
        """
        utilization = self.get_link_utilization(u, v)
        high_load = utilization >= self.THROUGHPUT_THRESHOLD
        is_safe_traffic = anomaly_score < self.ANOMALY_MAX_SCORE
        return high_load and is_safe_traffic

    def find_least_loaded_path(self, source: int, target: int,
                               exclude_congested: bool = True) -> Optional[List[int]]:
        """
        Dijkstra's path that MINIMIZES link utilization (load-balancing).
        This is different from the anomaly router which minimizes anomaly cost.
        """
        # Create a temporary weight on edges = utilization-based cost
        temp_graph = self.graph.copy()

        for u, v, data in temp_graph.edges(data=True):
            util = self.get_link_utilization(u, v)
            base_lat = data.get('base_latency', 10.0)

            if exclude_congested and util >= self.CRITICAL_THRESHOLD:
                # Temporarily inflate cost to make it unattractive
                temp_graph[u][v]['lb_cost'] = base_lat + 10000.0
            else:
                # Cost = latency penalized by utilization
                temp_graph[u][v]['lb_cost'] = base_lat * (1.0 + 3.0 * util)

        try:
            path = nx.dijkstra_path(temp_graph, source, target, weight='lb_cost')
            return path
        except nx.NetworkXNoPath:
            # Fallback: any available path
            try:
                return nx.shortest_path(self.graph, source, target,
                                        weight='base_latency')
            except Exception:
                return None

    def should_reroute(self, source: int, target: int,
                       current_path: List[int],
                       traffic_features: dict,
                       anomaly_score: float) -> Tuple[bool, Optional[List[int]], str]:
        """
        Main decision function.

        Returns:
            (should_reroute, new_path, reason)
        """
        self.stats['total_checks'] += 1

        # Check if current path has any high-throughput links (non-attack)
        congested_links = []
        for i in range(len(current_path) - 1):
            u, v = current_path[i], current_path[i + 1]
            if self.is_high_throughput_not_attack(u, v, traffic_features, anomaly_score):
                congested_links.append((u, v))

        if not congested_links:
            return False, None, "path_ok"

        # High throughput detected → find least-loaded alternative
        alt_path = self.find_least_loaded_path(source, target)

        if alt_path and alt_path != current_path:
            self.stats['rerouted_high_throughput'] += 1
            self.stats['flows_load_balanced'] += 1
            reason = (f"HIGH_THROUGHPUT_REROUTE: {len(congested_links)} link(s) "
                      f"overloaded, anomaly_score={anomaly_score:.2f} (safe traffic)")
            return True, alt_path, reason
        else:
            return False, None, "no_better_path"

    def get_stats(self) -> dict:
        return self.stats.copy()


# =============================================================================
# FEATURE 2: CONGESTION-AWARE SELF-HEALING ROUTING
# =============================================================================

class CongestionAwareSelfHealer:
    """
    Distributed congestion awareness across all nodes.

    All nodes share their congestion status with their neighbors using a
    gossip-like spreading protocol. When a node becomes congested:
      1. It immediately notifies ALL its neighbors.
      2. Neighbors propagate the info further (for k hops).
      3. Every node updates its routing table to avoid the congested node/link.
      4. New paths are computed automatically (SELF-HEALING).

    This simulates a distributed control plane in 5G (like SDN with local agents).
    """

    CONGESTION_LOAD_THRESHOLD = 0.80   # 80% node load = congested
    RECOVERY_THRESHOLD        = 0.50   # below 50% = recovered
    GOSSIP_HOPS               = 3      # how far the alert spreads

    def __init__(self, graph: nx.Graph):
        self.graph = graph

        # Per-node congestion state: {node_id: {'is_congested': bool, 'load': float, 'ts': time}}
        self.node_states: Dict[int, dict] = {}
        for node in graph.nodes():
            self.node_states[node] = {
                'is_congested': False,
                'load': 0.0,
                'timestamp': datetime.now().isoformat(),
                'alerted_by': None
            }

        # Congestion alerts received from neighbors (distributed awareness)
        # {node_id: set of congested_node_ids it knows about}
        self.known_congested: Dict[int, Set[int]] = defaultdict(set)

        # Self-healing event log
        self.healing_log: List[dict] = []

        # Stats
        self.stats = {
            'congestion_events': 0,
            'self_healing_reroutes': 0,
            'alerts_propagated': 0,
            'recoveries': 0
        }

    # ------------------------------------------------------------------
    # Node state management
    # ------------------------------------------------------------------

    def update_node_load(self, node_id: int, load: float):
        """Update a node's current load and trigger congestion alert if needed."""
        was_congested = self.node_states[node_id]['is_congested']
        is_congested  = load >= self.CONGESTION_LOAD_THRESHOLD

        self.node_states[node_id]['load'] = load
        self.node_states[node_id]['is_congested'] = is_congested
        self.node_states[node_id]['timestamp'] = datetime.now().isoformat()

        if is_congested and not was_congested:
            # NEW congestion event → broadcast to neighbors
            self.stats['congestion_events'] += 1
            self._broadcast_congestion_alert(node_id, hops=self.GOSSIP_HOPS)

        elif not is_congested and was_congested:
            # Recovery event
            self.stats['recoveries'] += 1
            self._broadcast_recovery(node_id)

    def _broadcast_congestion_alert(self, congested_node: int, hops: int):
        """
        Gossip protocol: spread congestion alert to neighbors up to `hops` hops away.
        Each neighbor adds the congested node to its 'known_congested' set.
        """
        visited = set()
        queue = [(congested_node, hops)]

        while queue:
            current, remaining_hops = queue.pop(0)
            if current in visited or remaining_hops <= 0:
                continue
            visited.add(current)

            for neighbor in self.graph.neighbors(current):
                if neighbor != congested_node:
                    # Neighbor learns about the congestion
                    self.known_congested[neighbor].add(congested_node)
                    self.stats['alerts_propagated'] += 1
                    if remaining_hops - 1 > 0:
                        queue.append((neighbor, remaining_hops - 1))

        print(f"  🔔 [SELF-HEAL] Congestion alert from BS-{congested_node} "
              f"propagated to {len(visited)-1} nodes over {self.GOSSIP_HOPS} hops")

    def _broadcast_recovery(self, recovered_node: int):
        """Notify neighbors that a previously congested node has recovered."""
        for neighbor in self.graph.neighbors(recovered_node):
            self.known_congested[neighbor].discard(recovered_node)
        print(f"  ✅ [SELF-HEAL] BS-{recovered_node} congestion CLEARED — "
              f"routes updated network-wide")

    # ------------------------------------------------------------------
    # Self-healing path finding
    # ------------------------------------------------------------------

    def find_healed_path(self, source: int, target: int,
                         observer_node: int) -> Tuple[Optional[List[int]], str]:
        """
        Find a path that avoids ALL nodes known to be congested by observer_node.
        This is the self-healing: automated path recomputation.

        Args:
            source: Flow source
            target: Flow destination
            observer_node: The node making the routing decision (has local view)

        Returns:
            (path, status_message)
        """
        known_bad = self.known_congested.get(observer_node, set())

        if not known_bad:
            # No known congestion → use normal path
            try:
                path = nx.shortest_path(self.graph, source, target,
                                        weight='base_latency')
                return path, "NORMAL_ROUTE"
            except Exception:
                return None, "NO_PATH"

        # Build subgraph excluding known congested nodes
        # (but always include source and target themselves)
        nodes_to_keep = [n for n in self.graph.nodes()
                         if n not in known_bad or n == source or n == target]
        subgraph = self.graph.subgraph(nodes_to_keep).copy()

        try:
            path = nx.shortest_path(subgraph, source, target, weight='base_latency')
            self.stats['self_healing_reroutes'] += 1
            self.healing_log.append({
                'timestamp': datetime.now().isoformat(),
                'source': source,
                'target': target,
                'observer': observer_node,
                'avoided_nodes': list(known_bad),
                'path': path
            })
            msg = (f"SELF_HEALED: avoided congested nodes {list(known_bad)}, "
                   f"new path: {path}")
            return path, msg
        except nx.NetworkXNoPath:
            # Even after removing congested nodes, no path found → try best effort
            try:
                fallback = nx.shortest_path(self.graph, source, target,
                                            weight='base_latency')
                return fallback, f"FALLBACK_ROUTE (congested nodes: {list(known_bad)})"
            except Exception:
                return None, "NO_PATH_EVEN_FALLBACK"

    def get_congestion_map(self) -> Dict[int, dict]:
        """Return current congestion state of all nodes."""
        return {node: state.copy() for node, state in self.node_states.items()}

    def get_stats(self) -> dict:
        return self.stats.copy()

    def get_healing_log(self) -> List[dict]:
        return self.healing_log.copy()


# =============================================================================
# FEATURE 3: DISTRIBUTED ANOMALY BLACKLISTING
# =============================================================================

class FlowBlacklist:
    """
    Cooperative, network-wide anomaly blacklisting.

    When any node detects a suspicious/anomalous packet flow, it:
      1. Creates a "flow fingerprint" (hash of key traffic properties).
      2. Broadcasts the fingerprint + reason to ALL other nodes.
      3. Every node adds the fingerprint to its LOCAL blacklist.
      4. Future packets matching the fingerprint are DROPPED / REJECTED
         at every node before processing — cooperative intrusion prevention.

    This simulates the "tell neighbor nodes" behavior the user described.
    """

    BLACKLIST_TTL_SECONDS = 300   # Entries expire after 5 minutes
    ANOMALY_THRESHOLD     = 0.65  # Score above this triggers blacklisting

    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes

        # Per-node blacklist: {node_id: {fingerprint: entry_dict}}
        self.node_blacklists: Dict[int, Dict[str, dict]] = {
            i: {} for i in range(num_nodes)
        }

        # Global alert log (what broadcasts were sent)
        self.alert_log: List[dict] = []

        # Stats
        self.stats = {
            'alerts_sent': 0,
            'flows_blocked_early': 0,
            'blacklist_entries_total': 0,
            'expired_entries_cleaned': 0
        }

    # ------------------------------------------------------------------
    # Fingerprinting
    # ------------------------------------------------------------------

    @staticmethod
    def fingerprint_flow(traffic_features: dict) -> str:
        """
        Create a fingerprint for a traffic flow from its key features.
        Uses a hash of binned feature values (tolerates minor fluctuations).
        """
        # Bin key fields to create a stable fingerprint
        key_parts = [
            f"lat={int(traffic_features.get('latency', 0) // 20)}",       # bin by 20ms
            f"tput={int(traffic_features.get('throughput', 0) // 10)}",   # bin by 10 Mbps
            f"loss={round(traffic_features.get('packet_loss', 0), 1)}",   # round to 1dp
            f"jit={int(traffic_features.get('jitter', 0) // 10)}",        # bin by 10ms
            f"type={int(traffic_features.get('traffic_type', 0))}",
        ]
        raw_key = "|".join(key_parts)
        return hashlib.md5(raw_key.encode()).hexdigest()[:12]

    # ------------------------------------------------------------------
    # Alert & blocking
    # ------------------------------------------------------------------

    def report_anomaly(self, detecting_node: int,
                       traffic_features: dict,
                       anomaly_score: float,
                       reason: str = "FL_DETECTION") -> str:
        """
        Called by a node when it detects an anomalous flow.
        Broadcasts a blacklist alert to ALL other nodes.

        Returns: fingerprint string
        """
        if anomaly_score < self.ANOMALY_THRESHOLD:
            return ""  # not serious enough

        fingerprint = self.fingerprint_flow(traffic_features)
        entry = {
            'fingerprint': fingerprint,
            'anomaly_score': anomaly_score,
            'reason': reason,
            'detected_by': detecting_node,
            'timestamp': datetime.now().isoformat(),
            'expires_at': time.time() + self.BLACKLIST_TTL_SECONDS,
            'features_snapshot': {
                'latency': traffic_features.get('latency'),
                'throughput': traffic_features.get('throughput'),
                'packet_loss': traffic_features.get('packet_loss'),
                'jitter': traffic_features.get('jitter'),
                'traffic_type': traffic_features.get('traffic_type')
            }
        }

        # Add to ALL nodes' blacklists (simulate broadcast)
        for node_id in range(self.num_nodes):
            self.node_blacklists[node_id][fingerprint] = entry

        # Log the alert
        self.alert_log.append({
            'from_node': detecting_node,
            'fingerprint': fingerprint,
            'score': anomaly_score,
            'broadcast_to': list(range(self.num_nodes)),
            'timestamp': entry['timestamp']
        })

        self.stats['alerts_sent'] += 1
        self.stats['blacklist_entries_total'] += self.num_nodes

        print(f"  🚨 [BLACKLIST] BS-{detecting_node} flagged flow [{fingerprint}] "
              f"(score={anomaly_score:.2f}) → broadcast to ALL {self.num_nodes} nodes")

        return fingerprint

    def is_blacklisted(self, node_id: int, traffic_features: dict) -> Tuple[bool, str]:
        """
        Check if a flow is blacklisted at a given node.

        Returns:
            (is_blocked, reason_string)
        """
        fingerprint = self.fingerprint_flow(traffic_features)
        blacklist = self.node_blacklists.get(node_id, {})

        if fingerprint not in blacklist:
            return False, ""

        entry = blacklist[fingerprint]

        # Check TTL
        if time.time() > entry.get('expires_at', 0):
            del blacklist[fingerprint]
            self.stats['expired_entries_cleaned'] += 1
            return False, "EXPIRED"

        self.stats['flows_blocked_early'] += 1
        reason = (f"BLACKLISTED at BS-{node_id}: fingerprint={fingerprint}, "
                  f"originally detected by BS-{entry['detected_by']}, "
                  f"score={entry['anomaly_score']:.2f}")
        return True, reason

    def clean_expired(self):
        """Remove expired blacklist entries from all nodes."""
        now = time.time()
        for node_id, blacklist in self.node_blacklists.items():
            expired = [fp for fp, e in blacklist.items()
                       if now > e.get('expires_at', 0)]
            for fp in expired:
                del blacklist[fp]
                self.stats['expired_entries_cleaned'] += 1

    def get_blacklist_size(self, node_id: int) -> int:
        return len(self.node_blacklists.get(node_id, {}))

    def get_stats(self) -> dict:
        return self.stats.copy()

    def get_alert_log(self) -> List[dict]:
        return self.alert_log.copy()


# =============================================================================
# UNIFIED MANAGER: Brings all three features together
# =============================================================================

class IntelligentNetworkManager:
    """
    Top-level manager that integrates all three advanced features:

    1. ThroughputLoadBalancer  — High-throughput rerouting (non-attack)
    2. CongestionAwareSelfHealer — Distributed congestion healing
    3. FlowBlacklist              — Cooperative anomaly blacklisting

    Usage:
        manager = IntelligentNetworkManager(graph, global_model, num_nodes=5)
        result  = manager.smart_route(source, dest, traffic_features, anomaly_score)
    """

    def __init__(self, graph: nx.Graph, global_model, num_nodes: int):
        self.graph        = graph
        self.global_model = global_model
        self.num_nodes    = num_nodes

        self.load_balancer = ThroughputLoadBalancer(graph, global_model)
        self.self_healer   = CongestionAwareSelfHealer(graph)
        self.blacklist     = FlowBlacklist(num_nodes)

        self.routing_log: List[dict] = []
        self.stats = {
            'total_flows': 0,
            'blocked_blacklisted': 0,
            'rerouted_load_balance': 0,
            'rerouted_self_heal': 0,
            'normal_routes': 0
        }

    # ------------------------------------------------------------------
    # Main routing decision (all three features applied in order)
    # ------------------------------------------------------------------

    def smart_route(self,
                    source: int,
                    target: int,
                    traffic_features: dict,
                    anomaly_score: float,
                    observer_node: Optional[int] = None) -> dict:
        """
        Intelligent routing integrating all three features.

        Decision order:
          STEP 1 → Check blacklist: if match → DROP (cooperative block)
          STEP 2 → Get self-healed path (avoids known congested nodes)
          STEP 3 → Check if current path is high-throughput → load-balance
          STEP 4 → If anomaly score high → report + add to blacklist
          STEP 5 → Return final decision

        Args:
            source: Source node ID
            target: Destination node ID
            traffic_features: Dict of traffic metrics
            anomaly_score: Float 0-1 from FL model
            observer_node: Node making the decision (defaults to source)

        Returns:
            Dict with keys: path, action, reason, anomaly_score, rerouted
        """
        self.stats['total_flows'] += 1
        if observer_node is None:
            observer_node = source

        result = {
            'source': source,
            'target': target,
            'anomaly_score': anomaly_score,
            'path': None,
            'action': None,
            'reason': '',
            'rerouted': False,
            'features_used': ['throughput_lb', 'congestion_heal', 'blacklist']
        }

        # ── STEP 1: Blacklist check ──────────────────────────────────────
        is_blocked, block_reason = self.blacklist.is_blacklisted(
            observer_node, traffic_features
        )
        if is_blocked:
            self.stats['blocked_blacklisted'] += 1
            result['action'] = 'BLOCKED'
            result['reason'] = f"[BLACKLIST] {block_reason}"
            result['path']   = []
            self._log(result)
            return result

        # ── STEP 2: Self-healing path (avoids known congested nodes) ────
        healed_path, heal_msg = self.self_healer.find_healed_path(
            source, target, observer_node
        )
        current_path = healed_path or []

        if 'SELF_HEALED' in heal_msg:
            self.stats['rerouted_self_heal'] += 1
            result['rerouted'] = True
            result['reason'] += f"[SELF-HEAL] {heal_msg} | "

        # ── STEP 3: Throughput load-balancing ───────────────────────────
        if current_path:
            should_lb, lb_path, lb_reason = self.load_balancer.should_reroute(
                source, target, current_path, traffic_features, anomaly_score
            )
            if should_lb and lb_path:
                self.stats['rerouted_load_balance'] += 1
                current_path   = lb_path
                result['rerouted'] = True
                result['reason'] += f"[LOAD-BALANCE] {lb_reason} | "

        # ── STEP 4: If anomaly is high → report and blacklist ───────────
        if anomaly_score >= self.blacklist.ANOMALY_THRESHOLD:
            fp = self.blacklist.report_anomaly(
                detecting_node=observer_node,
                traffic_features=traffic_features,
                anomaly_score=anomaly_score,
                reason="FL_ANOMALY_DETECTION"
            )
            result['reason'] += f"[ANOMALY_REPORTED] fingerprint={fp} | "
            result['action'] = 'ANOMALY_REROUTE_AND_BLACKLIST'

        # ── STEP 5: Final result ─────────────────────────────────────────
        if result['action'] not in ('BLOCKED', 'ANOMALY_REROUTE_AND_BLACKLIST'):
            result['action'] = 'REROUTED' if result['rerouted'] else 'NORMAL'
            if not result['rerouted']:
                self.stats['normal_routes'] += 1

        result['path'] = current_path
        if not result['reason']:
            result['reason'] = 'NORMAL_ROUTE'

        self._log(result)
        return result

    # ------------------------------------------------------------------
    # Network state simulation helpers
    # ------------------------------------------------------------------

    def simulate_node_load_event(self, node_id: int, load: float):
        """
        Simulate a node load change event.
        Updates the self-healer's congestion awareness.
        """
        self.self_healer.update_node_load(node_id, load)
        # Also update graph attribute
        if node_id in self.graph.nodes():
            self.graph.nodes[node_id]['current_load'] = load

    def simulate_link_traffic(self, u: int, v: int,
                               throughput: float, capacity: float):
        """
        Simulate traffic on a link.
        Updates the load balancer's throughput history.
        """
        self.load_balancer.record_flow(u, v, throughput, capacity)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_full_report(self) -> dict:
        """Return unified statistics from all three subsystems."""
        return {
            'manager_stats': self.stats.copy(),
            'load_balancer': self.load_balancer.get_stats(),
            'self_healer': self.self_healer.get_stats(),
            'blacklist': self.blacklist.get_stats(),
            'congestion_map': self.self_healer.get_congestion_map(),
            'healing_log_entries': len(self.self_healer.get_healing_log()),
            'blacklist_alert_log_entries': len(self.blacklist.get_alert_log())
        }

    def print_report(self):
        """Pretty-print the full system report."""
        report = self.get_full_report()
        print("\n" + "=" * 70)
        print(" INTELLIGENT NETWORK MANAGER — FULL REPORT")
        print("=" * 70)

        print("\n📊 OVERALL STATS:")
        for k, v in report['manager_stats'].items():
            print(f"   {k:35s}: {v}")

        print("\n🔀 LOAD BALANCER STATS:")
        for k, v in report['load_balancer'].items():
            print(f"   {k:35s}: {v}")

        print("\n🔧 SELF-HEALER STATS:")
        for k, v in report['self_healer'].items():
            print(f"   {k:35s}: {v}")

        print("\n🚫 BLACKLIST STATS:")
        for k, v in report['blacklist'].items():
            print(f"   {k:35s}: {v}")

        print("\n📡 CONGESTION MAP (per node):")
        for node_id, state in report['congestion_map'].items():
            status = "🔴 CONGESTED" if state['is_congested'] else "🟢 OK"
            print(f"   BS-{node_id}: {status}  load={state['load']:.2f}")

        print("=" * 70)

    def _log(self, result: dict):
        self.routing_log.append({
            'timestamp': datetime.now().isoformat(),
            **result
        })


# =============================================================================
# STANDALONE DEMO / TEST
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  INTELLIGENT NETWORK MANAGER — STANDALONE DEMO")
    print("=" * 70)

    # Build a simple 5-node network
    G = nx.Graph()
    for i in range(5):
        G.add_node(i, type='gNB', name=f'BS-{i}')

    edges = [
        (0, 1, {'base_latency': 10.0, 'capacity': 100.0, 'current_load': 0.0}),
        (1, 2, {'base_latency': 15.0, 'capacity': 100.0, 'current_load': 0.0}),
        (2, 3, {'base_latency': 12.0, 'capacity': 100.0, 'current_load': 0.0}),
        (3, 4, {'base_latency': 10.0, 'capacity': 100.0, 'current_load': 0.0}),
        (4, 0, {'base_latency': 18.0, 'capacity': 100.0, 'current_load': 0.0}),
        (0, 2, {'base_latency': 25.0, 'capacity': 100.0, 'current_load': 0.0}),
        (1, 3, {'base_latency': 20.0, 'capacity': 100.0, 'current_load': 0.0}),
        (2, 4, {'base_latency': 22.0, 'capacity': 100.0, 'current_load': 0.0}),
    ]
    for u, v, attrs in edges:
        G.add_edge(u, v, **attrs)

    # Mock model
    class MockModel:
        def predict_anomaly_score(self, features):
            return 0.1

    manager = IntelligentNetworkManager(G, MockModel(), num_nodes=5)

    # ── DEMO 1: Normal but high-throughput flow (load balancing) ────────
    print("\n" + "─" * 60)
    print("DEMO 1: HIGH THROUGHPUT — Not an attack, needs load balancing")
    print("─" * 60)
    manager.simulate_link_traffic(0, 1, throughput=82, capacity=100)  # 82% load
    normal_high_tput = {
        'latency': 18, 'throughput': 82, 'packet_loss': 0.01,
        'jitter': 2, 'queue_length': 30, 'load': 0.82, 'traffic_type': 0
    }
    result = manager.smart_route(0, 3, normal_high_tput, anomaly_score=0.05)
    print(f"  Action  : {result['action']}")
    print(f"  Path    : {result['path']}")
    print(f"  Reason  : {result['reason']}")
    print(f"  Rerouted: {result['rerouted']}")

    # ── DEMO 2: Congestion event + self-healing ──────────────────────────
    print("\n" + "─" * 60)
    print("DEMO 2: NODE CONGESTION → SELF-HEALING auto-reroute")
    print("─" * 60)
    manager.simulate_node_load_event(node_id=2, load=0.91)  # BS-2 congested!
    result = manager.smart_route(0, 4, normal_high_tput, anomaly_score=0.05,
                                 observer_node=1)
    print(f"  Action  : {result['action']}")
    print(f"  Path    : {result['path']}")
    print(f"  Reason  : {result['reason']}")
    print(f"  Rerouted: {result['rerouted']}")

    # ── DEMO 3: Anomaly detected → blacklist broadcast ───────────────────
    print("\n" + "─" * 60)
    print("DEMO 3: ANOMALY DETECTED → blacklist broadcast to all nodes")
    print("─" * 60)
    attack_flow = {
        'latency': 240, 'throughput': 5, 'packet_loss': 0.45,
        'jitter': 85, 'queue_length': 910, 'load': 0.95, 'traffic_type': 1
    }
    result = manager.smart_route(0, 4, attack_flow, anomaly_score=0.89,
                                 observer_node=0)
    print(f"  Action  : {result['action']}")
    print(f"  Path    : {result['path']}")
    print(f"  Reason  : {result['reason']}")

    # ── DEMO 4: Same attack flow arrives at ANOTHER node ─────────────────
    print("\n" + "─" * 60)
    print("DEMO 4: SAME ATTACK reaches BS-3 → blocked by cooperative blacklist!")
    print("─" * 60)
    result = manager.smart_route(3, 4, attack_flow, anomaly_score=0.89,
                                 observer_node=3)
    print(f"  Action  : {result['action']}")
    print(f"  Reason  : {result['reason']}")

    # ── FULL REPORT ───────────────────────────────────────────────────────
    manager.print_report()
