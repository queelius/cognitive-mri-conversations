#!/usr/bin/env python3
"""
Temporal snapshot builder for cumulative monthly network evolution.

At month t, the network G(t) contains all conversations with created <= end_of(t).
An edge is included iff both endpoints exist in G(t). This models real network growth.
"""

import csv
import numpy as np
import networkx as nx
import community.community_louvain as community_louvain
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from temporal_data_loader import TemporalDataset


@dataclass
class SnapshotMetrics:
    """Metrics for a single cumulative monthly snapshot."""
    month: str  # 'YYYY-MM'
    # Basic
    total_nodes: int = 0       # all conversations up to this month
    connected_nodes: int = 0   # nodes appearing in at least one edge
    edges: int = 0
    density: float = 0.0
    # Components
    num_components: int = 0
    giant_component_size: int = 0
    giant_component_fraction: float = 0.0
    # Degree
    avg_degree: float = 0.0
    std_degree: float = 0.0
    max_degree: int = 0
    # Clustering
    avg_clustering: float = 0.0
    transitivity: float = 0.0
    # Path
    avg_shortest_path: Optional[float] = None
    # Community
    modularity: float = 0.0
    num_communities: int = 0
    # Centrality
    avg_betweenness: float = 0.0
    avg_degree_centrality: float = 0.0
    assortativity: Optional[float] = None
    # Derived
    new_nodes: int = 0
    new_edges: int = 0
    edges_per_node: float = 0.0
    # Model era counts for this snapshot
    isolated_nodes: int = 0


def build_cumulative_snapshots(dataset: TemporalDataset,
                               random_state: int = 42) -> list:
    """
    Build cumulative monthly snapshots and compute metrics per snapshot.

    Uses incremental optimization: maintains running node/edge sets and only
    checks new edges involving newly arrived nodes.
    """
    np.random.seed(random_state)
    months = dataset.months_sorted

    # Pre-index edges by endpoint for O(1) lookup
    edges_by_node = {}
    for src, dst, weight in dataset.edges:
        edges_by_node.setdefault(src, []).append((src, dst, weight))
        edges_by_node.setdefault(dst, []).append((src, dst, weight))

    active_nodes = set()
    active_edges = set()  # (min(src,dst), max(src,dst)) for dedup
    G = nx.Graph()

    prev_node_count = 0
    prev_edge_count = 0
    snapshots = []

    for month in months:
        convos = dataset.monthly_buckets.get(month, [])
        new_node_ids = [c.node_id for c in convos]

        # Add new nodes to the graph
        for nid in new_node_ids:
            active_nodes.add(nid)
            G.add_node(nid)

        # Check for new edges involving newly arrived nodes
        for nid in new_node_ids:
            for src, dst, weight in edges_by_node.get(nid, []):
                edge_key = (min(src, dst), max(src, dst))
                if edge_key not in active_edges and src in active_nodes and dst in active_nodes:
                    active_edges.add(edge_key)
                    G.add_edge(src, dst, weight=weight)

        # Compute metrics
        metrics = _compute_snapshot_metrics(
            G, month, active_nodes,
            new_nodes=len(new_node_ids),
            new_edges=len(active_edges) - prev_edge_count,
            random_state=random_state,
        )
        snapshots.append(metrics)

        prev_node_count = len(active_nodes)
        prev_edge_count = len(active_edges)

    return snapshots


def _compute_snapshot_metrics(G: nx.Graph, month: str, all_nodes: set,
                              new_nodes: int, new_edges: int,
                              random_state: int = 42) -> SnapshotMetrics:
    """Compute comprehensive network metrics for a snapshot."""
    m = SnapshotMetrics(month=month)
    m.total_nodes = len(all_nodes)
    m.new_nodes = new_nodes
    m.new_edges = new_edges

    # Only consider the subgraph of connected nodes
    connected = set(G.nodes()) & {n for n in G.nodes() if G.degree(n) > 0}
    m.connected_nodes = len(connected)
    m.edges = G.number_of_edges()
    m.isolated_nodes = m.total_nodes - m.connected_nodes

    if m.connected_nodes < 2 or m.edges == 0:
        return m

    # Work with connected subgraph only
    G_connected = G.subgraph(connected).copy()

    m.density = nx.density(G_connected)
    m.edges_per_node = m.edges / m.connected_nodes if m.connected_nodes > 0 else 0

    # Components
    components = list(nx.connected_components(G_connected))
    m.num_components = len(components)

    if not components:
        return m

    giant = max(components, key=len)
    m.giant_component_size = len(giant)
    m.giant_component_fraction = len(giant) / m.connected_nodes

    G_giant = G_connected.subgraph(giant).copy()

    # Degree stats
    degrees = [d for _, d in G_giant.degree()]
    if degrees:
        m.avg_degree = float(np.mean(degrees))
        m.std_degree = float(np.std(degrees))
        m.max_degree = max(degrees)

    # Clustering
    m.avg_clustering = nx.average_clustering(G_giant)
    m.transitivity = nx.transitivity(G_giant)

    # Path length (sample if giant component is large)
    if len(G_giant) > 1 and nx.is_connected(G_giant):
        if len(G_giant) <= 500:
            m.avg_shortest_path = nx.average_shortest_path_length(G_giant)
        else:
            # Sample 200 nodes for estimation
            sample = list(np.random.choice(list(G_giant.nodes()),
                                           min(200, len(G_giant)), replace=False))
            lengths = []
            for s in sample[:50]:
                sp = nx.single_source_shortest_path_length(G_giant, s)
                lengths.extend(sp.values())
            if lengths:
                m.avg_shortest_path = float(np.mean([l for l in lengths if l > 0]))

    # Community detection
    if G_giant.number_of_edges() > 0:
        partition = community_louvain.best_partition(G_giant, random_state=random_state)
        m.modularity = community_louvain.modularity(partition, G_giant)
        m.num_communities = len(set(partition.values()))

    # Centrality
    deg_cent = nx.degree_centrality(G_giant)
    m.avg_degree_centrality = float(np.mean(list(deg_cent.values())))

    betw_cent = nx.betweenness_centrality(G_giant, k=min(100, len(G_giant)))
    m.avg_betweenness = float(np.mean(list(betw_cent.values())))

    # Assortativity
    try:
        m.assortativity = nx.degree_assortativity_coefficient(G_giant)
    except (nx.NetworkXError, ValueError):
        m.assortativity = None

    return m


def save_snapshot_metrics(snapshots: list, output_path: str):
    """Save snapshot metrics to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        'month', 'total_nodes', 'connected_nodes', 'edges', 'density',
        'num_components', 'giant_component_size', 'giant_component_fraction',
        'avg_degree', 'std_degree', 'max_degree',
        'avg_clustering', 'transitivity', 'avg_shortest_path',
        'modularity', 'num_communities',
        'avg_betweenness', 'avg_degree_centrality', 'assortativity',
        'new_nodes', 'new_edges', 'edges_per_node', 'isolated_nodes',
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in snapshots:
            row = {}
            for fn in fieldnames:
                val = getattr(s, fn, None)
                row[fn] = '' if val is None else val
            writer.writerow(row)

    print(f"  Saved {len(snapshots)} snapshot rows to {output_path}")


if __name__ == '__main__':
    import argparse
    from temporal_data_loader import build_temporal_dataset, print_dataset_summary

    parser = argparse.ArgumentParser(description='Build temporal snapshots')
    parser.add_argument('--embeddings-dir',
                        default='../../data/embeddings')
    parser.add_argument('--edges-file',
                        default='../../data/network/edges_user2.0-ai1.0_t0.9.json')
    parser.add_argument('--output', default='../data/temporal/temporal_metrics.csv')
    args = parser.parse_args()

    dataset = build_temporal_dataset(args.embeddings_dir, args.edges_file)
    print_dataset_summary(dataset)

    print("Building cumulative snapshots...")
    snapshots = build_cumulative_snapshots(dataset)

    print(f"\nSnapshot summary (first 5, last 3):")
    for s in snapshots[:5] + snapshots[-3:]:
        print(f"  {s.month}: {s.total_nodes} nodes, {s.edges} edges, "
              f"mod={s.modularity:.3f}, comms={s.num_communities}")

    save_snapshot_metrics(snapshots, args.output)
