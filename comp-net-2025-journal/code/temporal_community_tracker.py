#!/usr/bin/env python3
"""
Community tracker across temporal snapshots.

Tracks community identity using Jaccard overlap alignment,
classifies lifecycle events (birth, death, continuation, merge, split),
and assigns topic labels from conversation filenames.
"""

import json
import networkx as nx
import numpy as np
import community.community_louvain as community_louvain
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from temporal_data_loader import TemporalDataset


# Topic classification patterns (from create_cleaner_community_evolution.py)
TOPIC_PATTERNS = {
    'ML/AI': ['llm', 'ai', 'ml', 'neural', 'training', 'model', 'gpt', 'embedding',
              'transformer', 'attention', 'fine-tun', 'deep-learn', 'machine-learn'],
    'Programming': ['python', 'code', 'function', 'algorithm', 'debug', 'compile',
                    'git', 'rust', 'java', 'bash', 'script', 'api', 'software'],
    'Stats/Math': ['stat', 'probability', 'mle', 'distribution', 'math', 'calculus',
                   'variance', 'bayesian', 'regression', 'hypothesis', 'entropy'],
    'Philosophy': ['philosophy', 'consciousness', 'ethics', 'moral', 'exist',
                   'epistemol', 'ontolog', 'metaphys'],
    'Health': ['cancer', 'health', 'medical', 'therapy', 'pain', 'chemotherapy',
               'treatment', 'diagnosis'],
    'Networks': ['network', 'graph', 'gephi', 'complex', 'node', 'edge',
                 'community', 'centrality', 'modularity'],
}


def classify_topic(node_name: str) -> str:
    """Classify a conversation by topic based on filename keywords."""
    name_lower = node_name.lower()
    topics = []
    for topic, terms in TOPIC_PATTERNS.items():
        if any(term in name_lower for term in terms):
            topics.append(topic)
    return topics[0] if len(topics) == 1 else (topics[0] if topics else 'General')


def classify_community_topic(node_ids: list) -> dict:
    """
    Get dominant topic for a community based on member conversation names.

    Returns dict with 'dominant' topic and 'distribution' of topic proportions.
    """
    all_topics = []
    for nid in node_ids:
        name_lower = nid.lower()
        found = []
        for topic, terms in TOPIC_PATTERNS.items():
            if any(term in name_lower for term in terms):
                found.append(topic)
        all_topics.extend(found if found else ['General'])

    counts = Counter(all_topics)
    total = sum(counts.values())
    distribution = {t: c / total for t, c in counts.most_common()}
    dominant = counts.most_common(1)[0][0] if counts else 'General'

    return {'dominant': dominant, 'distribution': distribution}


@dataclass
class TrackedCommunity:
    """A community tracked across time with a persistent ID."""
    tracked_id: int
    month: str
    node_ids: list
    size: int
    topic_info: dict  # {dominant, distribution}
    louvain_id: int   # original Louvain partition ID for this snapshot


@dataclass
class CommunityEvent:
    """A lifecycle event for a community."""
    month: str
    event_type: str  # birth, death, continuation, merge, split
    tracked_ids: list  # IDs involved
    details: str = ''


def run_louvain_per_snapshot(dataset: TemporalDataset,
                             random_state: int = 42) -> list:
    """
    Run Louvain at each cumulative snapshot, returning community node sets.

    Returns list of (month, {community_id: set_of_node_ids}, graph).
    """
    months = dataset.months_sorted
    edges_by_node = {}
    for src, dst, weight in dataset.edges:
        edges_by_node.setdefault(src, []).append((src, dst, weight))
        edges_by_node.setdefault(dst, []).append((src, dst, weight))

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
            for src, dst, weight in edges_by_node.get(nid, []):
                ek = (min(src, dst), max(src, dst))
                if ek not in active_edges and src in active_nodes and dst in active_nodes:
                    active_edges.add(ek)
                    G.add_edge(src, dst, weight=weight)

        # Get giant component for community detection
        connected = {n for n in G.nodes() if G.degree(n) > 0}
        if len(connected) < 3:
            results.append((month, {}, G.copy()))
            continue

        G_connected = G.subgraph(connected).copy()
        components = list(nx.connected_components(G_connected))
        if not components:
            results.append((month, {}, G.copy()))
            continue

        giant = max(components, key=len)
        G_giant = G_connected.subgraph(giant).copy()

        if G_giant.number_of_edges() == 0:
            results.append((month, {}, G.copy()))
            continue

        partition = community_louvain.best_partition(G_giant, random_state=random_state)

        communities = defaultdict(set)
        for node, cid in partition.items():
            communities[cid].add(node)

        results.append((month, dict(communities), G.copy()))

    return results


def track_communities(snapshot_communities: list,
                      jaccard_threshold: float = 0.3) -> tuple:
    """
    Track community identity across snapshots using Jaccard overlap.

    Returns (tracked_communities_by_month, events).
    """
    next_tracked_id = 0
    # Map from previous community set (as frozenset) -> tracked_id
    prev_mapping = {}  # louvain_id -> (tracked_id, node_set)

    all_tracked = []  # list of TrackedCommunity
    all_events = []   # list of CommunityEvent

    for month, communities, _ in snapshot_communities:
        if not communities:
            continue

        curr_communities = {}  # louvain_id -> node_set
        for cid, nodes in communities.items():
            if len(nodes) >= 2:  # ignore singleton communities
                curr_communities[cid] = nodes

        if not prev_mapping:
            # First snapshot with communities — all births
            for cid, nodes in curr_communities.items():
                topic_info = classify_community_topic(list(nodes))
                tc = TrackedCommunity(
                    tracked_id=next_tracked_id,
                    month=month,
                    node_ids=list(nodes),
                    size=len(nodes),
                    topic_info=topic_info,
                    louvain_id=cid,
                )
                all_tracked.append(tc)
                prev_mapping[cid] = (next_tracked_id, nodes)
                all_events.append(CommunityEvent(
                    month=month, event_type='birth',
                    tracked_ids=[next_tracked_id],
                    details=f"New community: {topic_info['dominant']} ({len(nodes)} nodes)"
                ))
                next_tracked_id += 1
            continue

        # Compute Jaccard overlap matrix between prev and curr
        prev_items = list(prev_mapping.items())  # [(louvain_id, (tracked_id, node_set)), ...]
        curr_items = list(curr_communities.items())

        # Jaccard matrix: prev_idx x curr_idx
        jaccard_matrix = np.zeros((len(prev_items), len(curr_items)))
        for pi, (_, (_, prev_nodes)) in enumerate(prev_items):
            for ci, (_, curr_nodes) in enumerate(curr_items):
                intersection = len(prev_nodes & curr_nodes)
                union = len(prev_nodes | curr_nodes)
                jaccard_matrix[pi, ci] = intersection / union if union > 0 else 0

        # Find best matches
        # For each curr community, find best prev match
        curr_best_prev = {}  # curr_idx -> (prev_idx, jaccard)
        for ci in range(len(curr_items)):
            if len(prev_items) > 0:
                best_pi = int(np.argmax(jaccard_matrix[:, ci]))
                best_j = jaccard_matrix[best_pi, ci]
                if best_j >= jaccard_threshold:
                    curr_best_prev[ci] = (best_pi, best_j)

        # For each prev community, find best curr match
        prev_best_curr = {}  # prev_idx -> (curr_idx, jaccard)
        for pi in range(len(prev_items)):
            if len(curr_items) > 0:
                best_ci = int(np.argmax(jaccard_matrix[pi, :]))
                best_j = jaccard_matrix[pi, best_ci]
                if best_j >= jaccard_threshold:
                    prev_best_curr[pi] = (best_ci, best_j)

        # Classify events and assign tracked IDs
        new_mapping = {}
        assigned_curr = set()

        # 1. Continuations: mutual best matches (1:1)
        for ci, (pi, j) in curr_best_prev.items():
            if pi in prev_best_curr and prev_best_curr[pi][0] == ci:
                # Mutual best match = continuation
                prev_lid, (tracked_id, _) = prev_items[pi]
                curr_lid, curr_nodes = curr_items[ci]
                topic_info = classify_community_topic(list(curr_nodes))
                tc = TrackedCommunity(
                    tracked_id=tracked_id,
                    month=month,
                    node_ids=list(curr_nodes),
                    size=len(curr_nodes),
                    topic_info=topic_info,
                    louvain_id=curr_lid,
                )
                all_tracked.append(tc)
                new_mapping[curr_lid] = (tracked_id, curr_nodes)
                assigned_curr.add(ci)
                all_events.append(CommunityEvent(
                    month=month, event_type='continuation',
                    tracked_ids=[tracked_id],
                    details=f"{topic_info['dominant']} ({len(curr_nodes)} nodes, J={j:.2f})"
                ))

        # 2. Merges: multiple prev communities' best match is same curr community (N:1)
        curr_from_prev = defaultdict(list)
        for pi, (ci, j) in prev_best_curr.items():
            if ci not in assigned_curr:
                curr_from_prev[ci].append((pi, j))

        for ci, prev_matches in curr_from_prev.items():
            if len(prev_matches) > 1 and ci not in assigned_curr:
                # Merge: take the tracked_id of the largest previous community
                prev_tracked = []
                for pi, j in prev_matches:
                    _, (tid, pnodes) = prev_items[pi]
                    prev_tracked.append((tid, len(pnodes)))
                surviving_tid = max(prev_tracked, key=lambda x: x[1])[0]

                curr_lid, curr_nodes = curr_items[ci]
                topic_info = classify_community_topic(list(curr_nodes))
                tc = TrackedCommunity(
                    tracked_id=surviving_tid,
                    month=month,
                    node_ids=list(curr_nodes),
                    size=len(curr_nodes),
                    topic_info=topic_info,
                    louvain_id=curr_lid,
                )
                all_tracked.append(tc)
                new_mapping[curr_lid] = (surviving_tid, curr_nodes)
                assigned_curr.add(ci)

                merged_ids = [t[0] for t in prev_tracked]
                all_events.append(CommunityEvent(
                    month=month, event_type='merge',
                    tracked_ids=merged_ids,
                    details=f"Communities {merged_ids} merged into {surviving_tid} "
                            f"({topic_info['dominant']}, {len(curr_nodes)} nodes)"
                ))

        # 3. Splits: one prev community maps to multiple unassigned curr communities (1:N)
        prev_to_curr = defaultdict(list)
        for ci in range(len(curr_items)):
            if ci not in assigned_curr and ci in curr_best_prev:
                pi, j = curr_best_prev[ci]
                prev_to_curr[pi].append((ci, j))

        for pi, curr_matches in prev_to_curr.items():
            if len(curr_matches) > 1:
                _, (orig_tid, _) = prev_items[pi]
                new_tids = []
                for ci, j in curr_matches:
                    if ci in assigned_curr:
                        continue
                    curr_lid, curr_nodes = curr_items[ci]
                    topic_info = classify_community_topic(list(curr_nodes))
                    # First split inherits the original ID, rest get new IDs
                    if not new_tids:
                        tid = orig_tid
                    else:
                        tid = next_tracked_id
                        next_tracked_id += 1
                    tc = TrackedCommunity(
                        tracked_id=tid,
                        month=month,
                        node_ids=list(curr_nodes),
                        size=len(curr_nodes),
                        topic_info=topic_info,
                        louvain_id=curr_lid,
                    )
                    all_tracked.append(tc)
                    new_mapping[curr_lid] = (tid, curr_nodes)
                    assigned_curr.add(ci)
                    new_tids.append(tid)

                if new_tids:
                    all_events.append(CommunityEvent(
                        month=month, event_type='split',
                        tracked_ids=[orig_tid] + new_tids,
                        details=f"Community {orig_tid} split into {new_tids}"
                    ))

        # 4. Births: curr communities with no match
        for ci in range(len(curr_items)):
            if ci not in assigned_curr:
                curr_lid, curr_nodes = curr_items[ci]
                topic_info = classify_community_topic(list(curr_nodes))
                tc = TrackedCommunity(
                    tracked_id=next_tracked_id,
                    month=month,
                    node_ids=list(curr_nodes),
                    size=len(curr_nodes),
                    topic_info=topic_info,
                    louvain_id=curr_lid,
                )
                all_tracked.append(tc)
                new_mapping[curr_lid] = (next_tracked_id, curr_nodes)
                assigned_curr.add(ci)
                all_events.append(CommunityEvent(
                    month=month, event_type='birth',
                    tracked_ids=[next_tracked_id],
                    details=f"New: {topic_info['dominant']} ({len(curr_nodes)} nodes)"
                ))
                next_tracked_id += 1

        # 5. Deaths: prev communities with no match in curr
        matched_prev = set()
        for ci, (pi, _) in curr_best_prev.items():
            if ci in assigned_curr:
                matched_prev.add(pi)
        for pi, (pi_lid, (tid, pnodes)) in enumerate(prev_items):
            if pi not in matched_prev and pi not in {pm[0] for pms in prev_to_curr.values() for pm in pms}:
                # Check if truly dead — no curr community overlaps at all
                max_j = max(jaccard_matrix[pi, :]) if len(curr_items) > 0 else 0
                if max_j < jaccard_threshold:
                    all_events.append(CommunityEvent(
                        month=month, event_type='death',
                        tracked_ids=[tid],
                        details=f"Community {tid} dissolved (was {len(pnodes)} nodes)"
                    ))

        prev_mapping = new_mapping

    return all_tracked, all_events


def save_tracked_communities(tracked: list, output_path: str):
    """Save tracked communities to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = []
    for tc in tracked:
        data.append({
            'tracked_id': tc.tracked_id,
            'month': tc.month,
            'size': tc.size,
            'topic_dominant': tc.topic_info['dominant'],
            'topic_distribution': tc.topic_info['distribution'],
            'node_ids': tc.node_ids,
        })

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {len(data)} tracked community records to {output_path}")


def save_community_events(events: list, output_path: str):
    """Save community events to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = []
    for e in events:
        data.append({
            'month': e.month,
            'event_type': e.event_type,
            'tracked_ids': e.tracked_ids,
            'details': e.details,
        })

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {len(data)} community events to {output_path}")


def get_quarterly_community_flows(tracked: list) -> dict:
    """
    Aggregate communities by quarter for alluvial diagram readability.

    Returns {quarter_label: {tracked_id: size}} for communities >= 10 nodes.
    """
    quarterly = defaultdict(lambda: defaultdict(int))

    for tc in tracked:
        year, month = tc.month.split('-')
        q = (int(month) - 1) // 3 + 1
        quarter = f"{year}-Q{q}"
        # Keep the max size seen in this quarter for each tracked_id
        if tc.size > quarterly[quarter][tc.tracked_id]:
            quarterly[quarter][tc.tracked_id] = tc.size

    return {q: dict(comms) for q, comms in sorted(quarterly.items())}


if __name__ == '__main__':
    import argparse
    from temporal_data_loader import build_temporal_dataset, print_dataset_summary

    parser = argparse.ArgumentParser(description='Track communities over time')
    parser.add_argument('--embeddings-dir',
                        default='../../dev/ablation_study/data/embeddings/chatgpt-json-llm-user2.0-ai1.0')
    parser.add_argument('--edges-file',
                        default='../../dev/ablation_study/data/edges_filtered/edges_chatgpt-json-llm-user2.0-ai1.0_t0.9.json')
    parser.add_argument('--output-dir', default='../data/temporal')
    args = parser.parse_args()

    dataset = build_temporal_dataset(args.embeddings_dir, args.edges_file)

    print("Running Louvain per snapshot...")
    snapshot_communities = run_louvain_per_snapshot(dataset)

    print("Tracking communities...")
    tracked, events = track_communities(snapshot_communities)

    # Summary
    from collections import Counter
    event_counts = Counter(e.event_type for e in events)
    print(f"\n  Event summary:")
    for etype, count in event_counts.most_common():
        print(f"    {etype:15s}: {count}")

    unique_tracked = len(set(tc.tracked_id for tc in tracked))
    print(f"  Unique tracked communities: {unique_tracked}")

    output_dir = Path(args.output_dir)
    save_tracked_communities(tracked, output_dir / 'tracked_communities.json')
    save_community_events(events, output_dir / 'community_events.json')
