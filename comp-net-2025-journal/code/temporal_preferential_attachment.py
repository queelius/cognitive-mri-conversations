#!/usr/bin/env python3
"""
Preferential attachment testing, bridge formation dynamics,
and densification law analysis for the temporal network.
"""

import csv
import json
import numpy as np
import networkx as nx
import community.community_louvain as community_louvain
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from scipy import stats as sp_stats
from typing import Optional

from temporal_data_loader import TemporalDataset


# The 5 conference paper bridge conversations
BRIDGE_CONVERSATIONS = [
    'geometric-mean-calculation',
    'mcts-code-analysis-suggestions',
    'loss-in-llm-training',
    'algotree-generate-unit-tests-flattree',
    'compile-cuda-program-linux',
]


@dataclass
class MonthlyAttachment:
    """Preferential attachment statistics for a single month."""
    month: str
    new_nodes: int
    new_edges_from_new: int  # edges connecting new nodes to existing nodes
    correlation: float       # Spearman correlation: degree vs attachment count
    p_value: float
    null_mean: float         # mean correlation under null model
    null_std: float
    z_score: float           # (observed - null_mean) / null_std


@dataclass
class BridgeSnapshot:
    """Betweenness centrality of a bridge conversation at a snapshot."""
    node_id: str
    month: str
    betweenness: float
    degree: int
    num_neighbor_communities: int  # communities among neighbors
    is_bridge_threshold: bool      # top 5% betweenness


# ─── Preferential Attachment ─────────────────────────────────────────

def test_preferential_attachment(dataset: TemporalDataset,
                                 n_permutations: int = 1000,
                                 random_state: int = 42) -> tuple:
    """
    Test preferential attachment for each month.

    For each month:
    1. Network state at t-1 (before arrivals)
    2. New nodes at month t
    3. For each new node, which existing nodes it connects to
    4. Correlation between existing node degree and receiving-new-edge probability
    5. Compare to null model (permutations of uniform random attachment)

    Returns (monthly_results, pooled_attachment_kernel).
    """
    rng = np.random.RandomState(random_state)
    months = dataset.months_sorted

    # Pre-index edges by endpoint
    edges_by_node = defaultdict(list)
    for src, dst, weight in dataset.edges:
        edges_by_node[src].append((src, dst, weight))
        edges_by_node[dst].append((src, dst, weight))

    active_nodes = set()
    active_edges = set()
    G = nx.Graph()

    monthly_results = []
    # For pooled kernel: collect (degree_of_target, count) across all months
    pooled_degree_attachment = []  # (existing_node_degree, was_attached_to)

    for mi, month in enumerate(months):
        convos = dataset.monthly_buckets.get(month, [])
        new_ids = set(c.node_id for c in convos)

        if mi == 0 or not active_nodes:
            # First month — no existing network to attach to
            for nid in new_ids:
                active_nodes.add(nid)
                G.add_node(nid)
            # Add any edges among first-month nodes
            for nid in new_ids:
                for src, dst, w in edges_by_node.get(nid, []):
                    ek = (min(src, dst), max(src, dst))
                    if ek not in active_edges and src in active_nodes and dst in active_nodes:
                        active_edges.add(ek)
                        G.add_edge(src, dst, weight=w)
            continue

        # Existing nodes before this month's arrivals
        existing_nodes = set(active_nodes)
        existing_degrees = {n: G.degree(n) for n in existing_nodes}

        # Add new nodes
        for nid in new_ids:
            active_nodes.add(nid)
            G.add_node(nid)

        # Find edges from new nodes to existing nodes
        attachment_targets = []  # existing nodes that new nodes connect to
        for nid in new_ids:
            for src, dst, w in edges_by_node.get(nid, []):
                ek = (min(src, dst), max(src, dst))
                if ek not in active_edges and src in active_nodes and dst in active_nodes:
                    active_edges.add(ek)
                    G.add_edge(src, dst, weight=w)
                    # Identify the existing endpoint
                    other = dst if src == nid else src
                    if other in existing_nodes:
                        attachment_targets.append(other)

        # Also add edges between new nodes and among existing (delayed edges)
        # (already handled above since we check all edges for new nodes)

        n_new_edges = len(attachment_targets)
        if n_new_edges < 3 or len(existing_nodes) < 5:
            monthly_results.append(MonthlyAttachment(
                month=month, new_nodes=len(new_ids),
                new_edges_from_new=n_new_edges,
                correlation=0, p_value=1.0,
                null_mean=0, null_std=0, z_score=0,
            ))
            continue

        # Count attachments per existing node
        attach_counts = defaultdict(int)
        for target in attachment_targets:
            attach_counts[target] += 1

        # Build arrays for correlation
        existing_list = list(existing_nodes)
        deg_array = np.array([existing_degrees[n] for n in existing_list])
        attach_array = np.array([attach_counts.get(n, 0) for n in existing_list])

        # Pooled data for kernel estimation
        for n in existing_list:
            if existing_degrees[n] > 0:
                pooled_degree_attachment.append(
                    (existing_degrees[n], attach_counts.get(n, 0))
                )

        # Observed correlation (Spearman — handles non-linearity)
        if np.std(deg_array) > 0 and np.std(attach_array) > 0:
            obs_corr, obs_p = sp_stats.spearmanr(deg_array, attach_array)
        else:
            obs_corr, obs_p = 0.0, 1.0

        # Null model: permute attachment targets uniformly
        null_corrs = []
        for _ in range(n_permutations):
            perm_targets = rng.choice(existing_list, size=n_new_edges, replace=True)
            perm_counts = defaultdict(int)
            for t in perm_targets:
                perm_counts[t] += 1
            perm_array = np.array([perm_counts.get(n, 0) for n in existing_list])
            if np.std(perm_array) > 0:
                r, _ = sp_stats.spearmanr(deg_array, perm_array)
                null_corrs.append(r)
            else:
                null_corrs.append(0.0)

        null_mean = np.mean(null_corrs)
        null_std = np.std(null_corrs) if len(null_corrs) > 1 else 1e-10
        z_score = (obs_corr - null_mean) / max(null_std, 1e-10)

        monthly_results.append(MonthlyAttachment(
            month=month, new_nodes=len(new_ids),
            new_edges_from_new=n_new_edges,
            correlation=float(obs_corr), p_value=float(obs_p),
            null_mean=float(null_mean), null_std=float(null_std),
            z_score=float(z_score),
        ))

    # Compute pooled attachment kernel
    kernel = compute_attachment_kernel(pooled_degree_attachment)

    return monthly_results, kernel


def compute_attachment_kernel(pooled_data: list) -> dict:
    """
    Compute attachment kernel Pi(k) from pooled degree-attachment data.

    Bins existing nodes by degree and measures empirical attachment
    probability per bin. Fits Pi(k) ~ k^alpha.

    Returns dict with bins, probabilities, alpha, and fit quality.
    """
    if not pooled_data:
        return {'alpha': 0, 'r_squared': 0, 'bins': [], 'probs': []}

    degrees = np.array([d for d, _ in pooled_data])
    attachments = np.array([a for _, a in pooled_data])

    # Log-spaced bins for degree
    max_deg = max(degrees)
    if max_deg < 2:
        return {'alpha': 0, 'r_squared': 0, 'bins': [], 'probs': []}

    bin_edges = np.unique(np.logspace(0, np.log10(max_deg), 15).astype(int))
    if len(bin_edges) < 3:
        bin_edges = np.arange(1, max_deg + 1)

    bin_centers = []
    bin_probs = []

    for i in range(len(bin_edges) - 1):
        mask = (degrees >= bin_edges[i]) & (degrees < bin_edges[i + 1])
        if mask.sum() > 0:
            prob = attachments[mask].mean()
            center = degrees[mask].mean()
            bin_centers.append(float(center))
            bin_probs.append(float(prob))

    # Fit power law in log-log space
    alpha = 0.0
    r_squared = 0.0
    if len(bin_centers) >= 3:
        valid = [(c, p) for c, p in zip(bin_centers, bin_probs) if c > 0 and p > 0]
        if len(valid) >= 3:
            log_k = np.log10([c for c, _ in valid])
            log_p = np.log10([p for _, p in valid])
            slope, intercept, r, _, _ = sp_stats.linregress(log_k, log_p)
            alpha = float(slope)
            r_squared = float(r ** 2)

    return {
        'alpha': alpha,
        'r_squared': r_squared,
        'bins': bin_centers,
        'probs': bin_probs,
    }


# ─── Bridge Formation Dynamics ───────────────────────────────────────

def track_bridge_dynamics(dataset: TemporalDataset,
                          random_state: int = 42) -> list:
    """
    Track the 5 conference paper bridges over time.

    For each: betweenness centrality at every snapshot from its creation
    month onward. Classify bridge type using neighbor community membership.
    """
    months = dataset.months_sorted

    # Pre-index edges
    edges_by_node = defaultdict(list)
    for src, dst, weight in dataset.edges:
        edges_by_node[src].append((src, dst, weight))
        edges_by_node[dst].append((src, dst, weight))

    # Find creation months for bridge conversations
    bridge_created = {}
    for conv in dataset.conversations:
        if conv.node_id in BRIDGE_CONVERSATIONS:
            bridge_created[conv.node_id] = conv.created.strftime('%Y-%m')

    active_nodes = set()
    active_edges = set()
    G = nx.Graph()
    results = []

    for month in months:
        convos = dataset.monthly_buckets.get(month, [])
        new_ids = [c.node_id for c in convos]

        for nid in new_ids:
            active_nodes.add(nid)
            G.add_node(nid)

        for nid in new_ids:
            for src, dst, w in edges_by_node.get(nid, []):
                ek = (min(src, dst), max(src, dst))
                if ek not in active_edges and src in active_nodes and dst in active_nodes:
                    active_edges.add(ek)
                    G.add_edge(src, dst, weight=w)

        # Get connected subgraph
        connected = {n for n in G.nodes() if G.degree(n) > 0}
        if len(connected) < 3:
            continue

        G_connected = G.subgraph(connected).copy()
        components = list(nx.connected_components(G_connected))
        if not components:
            continue

        giant = max(components, key=len)
        G_giant = G_connected.subgraph(giant).copy()

        # Compute betweenness centrality
        betw = nx.betweenness_centrality(G_giant, k=min(200, len(G_giant)))

        # Compute community partition for bridge classification
        if G_giant.number_of_edges() > 0:
            partition = community_louvain.best_partition(G_giant, random_state=random_state)
        else:
            partition = {}

        # Top 5% threshold
        betw_values = sorted(betw.values(), reverse=True)
        threshold_idx = max(1, int(len(betw_values) * 0.05))
        bridge_threshold = betw_values[threshold_idx - 1] if betw_values else 0

        for bridge_id in BRIDGE_CONVERSATIONS:
            if bridge_id not in G_giant.nodes():
                continue

            b_betw = betw.get(bridge_id, 0)
            b_deg = G_giant.degree(bridge_id)

            # Count distinct communities among neighbors
            neighbor_comms = set()
            for neighbor in G_giant.neighbors(bridge_id):
                if neighbor in partition:
                    neighbor_comms.add(partition[neighbor])

            results.append(BridgeSnapshot(
                node_id=bridge_id,
                month=month,
                betweenness=float(b_betw),
                degree=b_deg,
                num_neighbor_communities=len(neighbor_comms),
                is_bridge_threshold=b_betw >= bridge_threshold,
            ))

    return results


# ─── Densification Law ───────────────────────────────────────────────

def compute_densification(snapshots_csv: str) -> dict:
    """
    Compute densification exponent: e(t) ~ n(t)^alpha in log-log space.

    Returns dict with alpha, r_squared, nodes array, edges array.
    """
    import pandas as pd
    df = pd.read_csv(snapshots_csv)

    # Use connected_nodes and edges (only meaningful portion)
    nodes = df['connected_nodes'].values.astype(float)
    edges = df['edges'].values.astype(float)

    # Filter to points where both > 0
    mask = (nodes > 0) & (edges > 0)
    nodes = nodes[mask]
    edges = edges[mask]

    if len(nodes) < 3:
        return {'alpha': 0, 'r_squared': 0, 'nodes': [], 'edges': []}

    log_n = np.log10(nodes)
    log_e = np.log10(edges)

    slope, intercept, r, p, se = sp_stats.linregress(log_n, log_e)

    return {
        'alpha': float(slope),
        'r_squared': float(r ** 2),
        'intercept': float(intercept),
        'p_value': float(p),
        'nodes': nodes.tolist(),
        'edges': edges.tolist(),
    }


# ─── Model Era Sub-Networks ─────────────────────────────────────────

def compute_model_era_metrics(dataset: TemporalDataset,
                              random_state: int = 42) -> list:
    """
    Build separate sub-networks per model era and compare metrics.
    """
    from collections import Counter

    era_nodes = defaultdict(set)
    for conv in dataset.conversations:
        era_nodes[conv.model_era].add(conv.node_id)

    results = []
    for era in ['gpt35', 'gpt4', 'gpt4o', 'reasoning', 'gpt45']:
        nodes = era_nodes.get(era, set())
        if not nodes:
            continue

        # Build subgraph with only edges between nodes of this era
        G = nx.Graph()
        for n in nodes:
            G.add_node(n)
        for src, dst, w in dataset.edges:
            if src in nodes and dst in nodes:
                G.add_edge(src, dst, weight=w)

        connected = {n for n in G.nodes() if G.degree(n) > 0}
        n_connected = len(connected)
        n_edges = G.number_of_edges()

        row = {
            'era': era,
            'total_nodes': len(nodes),
            'connected_nodes': n_connected,
            'edges': n_edges,
            'density': nx.density(G.subgraph(connected)) if n_connected > 1 else 0,
            'avg_degree': 0, 'avg_clustering': 0, 'modularity': 0,
            'num_communities': 0, 'avg_betweenness': 0,
        }

        if n_connected < 3 or n_edges < 2:
            results.append(row)
            continue

        G_conn = G.subgraph(connected).copy()
        degrees = [d for _, d in G_conn.degree()]
        row['avg_degree'] = float(np.mean(degrees))
        row['avg_clustering'] = nx.average_clustering(G_conn)

        # Community detection on giant component
        components = list(nx.connected_components(G_conn))
        if components:
            giant = max(components, key=len)
            G_giant = G_conn.subgraph(giant).copy()
            if G_giant.number_of_edges() > 0:
                partition = community_louvain.best_partition(G_giant, random_state=random_state)
                row['modularity'] = community_louvain.modularity(partition, G_giant)
                row['num_communities'] = len(set(partition.values()))

                betw = nx.betweenness_centrality(G_giant, k=min(100, len(G_giant)))
                row['avg_betweenness'] = float(np.mean(list(betw.values())))

        results.append(row)

    return results


# ─── I/O ─────────────────────────────────────────────────────────────

def save_attachment_results(monthly: list, kernel: dict, output_path: str):
    """Save preferential attachment results to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ['month', 'new_nodes', 'new_edges_from_new',
                  'correlation', 'p_value', 'null_mean', 'null_std', 'z_score']

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in monthly:
            writer.writerow({fn: getattr(m, fn) for fn in fieldnames})

    # Save kernel as separate JSON
    kernel_path = output_path.parent / 'attachment_kernel.json'
    with open(kernel_path, 'w') as f:
        json.dump(kernel, f, indent=2)

    print(f"  Saved {len(monthly)} monthly attachment results to {output_path}")
    print(f"  Saved attachment kernel (alpha={kernel['alpha']:.3f}) to {kernel_path}")


def save_bridge_dynamics(bridges: list, output_path: str):
    """Save bridge dynamics to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ['node_id', 'month', 'betweenness', 'degree',
                  'num_neighbor_communities', 'is_bridge_threshold']

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for b in bridges:
            writer.writerow({
                'node_id': b.node_id,
                'month': b.month,
                'betweenness': b.betweenness,
                'degree': b.degree,
                'num_neighbor_communities': b.num_neighbor_communities,
                'is_bridge_threshold': b.is_bridge_threshold,
            })

    print(f"  Saved {len(bridges)} bridge snapshots to {output_path}")


def save_model_era_metrics(era_metrics: list, output_path: str):
    """Save model era metrics to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ['era', 'total_nodes', 'connected_nodes', 'edges',
                  'density', 'avg_degree', 'avg_clustering',
                  'modularity', 'num_communities', 'avg_betweenness']

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in era_metrics:
            writer.writerow(row)

    print(f"  Saved {len(era_metrics)} model era metrics to {output_path}")


if __name__ == '__main__':
    import argparse
    from temporal_data_loader import build_temporal_dataset, print_dataset_summary

    parser = argparse.ArgumentParser(description='Preferential attachment & bridge analysis')
    parser.add_argument('--embeddings-dir',
                        default='../../data/embeddings')
    parser.add_argument('--edges-file',
                        default='../../data/network/edges_user2.0-ai1.0_t0.9.json')
    parser.add_argument('--snapshots-csv', default='../data/temporal/temporal_metrics.csv')
    parser.add_argument('--output-dir', default='../data/temporal')
    args = parser.parse_args()

    dataset = build_temporal_dataset(args.embeddings_dir, args.edges_file)

    print("Testing preferential attachment...")
    monthly, kernel = test_preferential_attachment(dataset)
    print(f"  Kernel alpha = {kernel['alpha']:.3f} (R² = {kernel['r_squared']:.3f})")

    print("\nTracking bridge dynamics...")
    bridges = track_bridge_dynamics(dataset)
    bridge_ids = set(b.node_id for b in bridges)
    print(f"  Tracked {len(bridge_ids)} bridges across {len(bridges)} snapshots")

    print("\nComputing densification law...")
    densification = compute_densification(args.snapshots_csv)
    print(f"  Densification alpha = {densification['alpha']:.3f} (R² = {densification['r_squared']:.3f})")

    print("\nComputing model era metrics...")
    era_metrics = compute_model_era_metrics(dataset)
    for em in era_metrics:
        print(f"  {em['era']:12s}: {em['connected_nodes']:4d} connected, "
              f"density={em['density']:.4f}, mod={em['modularity']:.3f}")

    output_dir = Path(args.output_dir)
    save_attachment_results(monthly, kernel, output_dir / 'preferential_attachment.csv')
    save_bridge_dynamics(bridges, output_dir / 'bridge_dynamics.csv')
    save_model_era_metrics(era_metrics, output_dir / 'model_era_metrics.csv')

    # Save densification result
    with open(output_dir / 'densification.json', 'w') as f:
        json.dump(densification, f, indent=2)
    print(f"  Saved densification (alpha={densification['alpha']:.3f}) to {output_dir / 'densification.json'}")
