#!/usr/bin/env python3
"""
Temporal data loader for the journal extension.

Loads pre-computed embeddings metadata and edges, builds a time-indexed
conversation catalog with model era classification and monthly bucketing.
"""

import json
import glob
import os
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from typing import Optional


@dataclass
class ConversationRecord:
    """Metadata for a single conversation (no embedding vectors)."""
    node_id: str
    title: str
    created: datetime
    model: Optional[str]
    model_era: str


@dataclass
class TemporalDataset:
    """Complete temporal dataset with conversations, edges, and indices."""
    conversations: list  # sorted by created time
    edges: list  # [(src, dst, weight), ...]
    node_lookup: dict  # node_id -> ConversationRecord
    monthly_buckets: dict  # 'YYYY-MM' -> [ConversationRecord, ...]
    date_range: tuple  # (earliest, latest)

    @property
    def months_sorted(self):
        return sorted(self.monthly_buckets.keys())

    @property
    def node_ids_in_edges(self):
        """Set of all node IDs that appear in at least one edge."""
        ids = set()
        for src, dst, _ in self.edges:
            ids.add(src)
            ids.add(dst)
        return ids


def classify_model_era(model: Optional[str]) -> str:
    """
    Map raw model string to era label.

    Era definitions based on OpenAI release timeline:
    - gpt35: GPT-3.5 era (None/legacy models, Dec 2022 - Mar 2023)
    - gpt4: GPT-4 era (gpt-4, Mar 2023 - May 2024)
    - gpt4o: GPT-4o era (gpt-4o variants, May 2024 - Sep 2024)
    - reasoning: Reasoning models (o1, o3, Oct 2024 - Mar 2025)
    - gpt45: GPT-4.5 era (gpt-4-5, Mar 2025+)
    """
    if model is None:
        return 'gpt35'

    m = model.lower()

    if m in ('gpt-4-5',):
        return 'gpt45'

    if any(m.startswith(p) for p in ('o1', 'o3')):
        return 'reasoning'

    if 'gpt-4o' in m or m == 'auto':
        return 'gpt4o'

    if m.startswith('gpt-4'):
        return 'gpt4'

    # Legacy models (text-davinci, research, etc.)
    if m in ('text-davinci-002-render-sha', 'research'):
        return 'gpt35'

    return 'gpt35'


def load_conversations(embeddings_dir: str) -> list:
    """
    Read conversation JSONs and extract metadata only (not vectors).

    Returns list of ConversationRecord sorted by creation time.
    """
    embeddings_dir = Path(embeddings_dir)
    records = []

    for json_file in sorted(embeddings_dir.glob('*.json')):
        with open(json_file, 'r', encoding='utf-8') as f:
            doc = json.load(f)

        node_id = json_file.stem
        title = doc.get('title', node_id)
        created_str = doc.get('created')
        model = doc.get('model')

        # Parse creation date
        created = None
        if created_str:
            try:
                created = datetime.strptime(created_str, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                try:
                    created = datetime.fromisoformat(created_str)
                except (ValueError, TypeError):
                    pass

        if created is None:
            continue  # Skip conversations without valid timestamps

        model_era = classify_model_era(model)
        records.append(ConversationRecord(
            node_id=node_id,
            title=title,
            created=created,
            model=model,
            model_era=model_era,
        ))

    records.sort(key=lambda r: r.created)
    return records


def load_edges(edges_file: str) -> list:
    """Read JSON array of [src, dst, weight] edges."""
    with open(edges_file, 'r', encoding='utf-8') as f:
        edges_data = json.load(f)

    edges = []
    for edge in edges_data:
        if len(edge) >= 3:
            src, dst, weight = edge[0], edge[1], edge[2]
            if weight is not None:
                edges.append((src, dst, float(weight)))
    return edges


def build_temporal_dataset(embeddings_dir: str, edges_file: str) -> TemporalDataset:
    """
    Orchestrate loading: conversations + edges, sort by time, bucket by month.
    """
    conversations = load_conversations(embeddings_dir)
    edges = load_edges(edges_file)

    # Build lookup
    node_lookup = {r.node_id: r for r in conversations}

    # Monthly buckets
    monthly_buckets = defaultdict(list)
    for conv in conversations:
        month_key = conv.created.strftime('%Y-%m')
        monthly_buckets[month_key].append(conv)

    date_range = (conversations[0].created, conversations[-1].created)

    dataset = TemporalDataset(
        conversations=conversations,
        edges=edges,
        node_lookup=node_lookup,
        monthly_buckets=dict(monthly_buckets),
        date_range=date_range,
    )

    return dataset


def print_dataset_summary(dataset: TemporalDataset):
    """Print summary statistics of the loaded dataset."""
    from collections import Counter

    print(f"\n{'='*60}")
    print("TEMPORAL DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"  Conversations: {len(dataset.conversations)}")
    print(f"  Edges:         {len(dataset.edges)}")
    print(f"  Date range:    {dataset.date_range[0].strftime('%Y-%m-%d')} to "
          f"{dataset.date_range[1].strftime('%Y-%m-%d')}")
    print(f"  Months:        {len(dataset.monthly_buckets)}")

    # Connected nodes
    connected = dataset.node_ids_in_edges
    print(f"  Nodes in edges: {len(connected)} / {len(dataset.conversations)}")

    # Model era distribution
    era_counts = Counter(c.model_era for c in dataset.conversations)
    print(f"\n  Model era distribution:")
    for era in ['gpt35', 'gpt4', 'gpt4o', 'reasoning', 'gpt45']:
        print(f"    {era:12s}: {era_counts.get(era, 0):5d}")

    # Monthly conversation counts
    print(f"\n  Monthly conversation counts:")
    for month in dataset.months_sorted:
        count = len(dataset.monthly_buckets[month])
        bar = '#' * (count // 5)
        print(f"    {month}: {count:4d} {bar}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Load temporal dataset')
    parser.add_argument('--embeddings-dir',
                        default='../../dev/ablation_study/data/embeddings/chatgpt-json-llm-user2.0-ai1.0')
    parser.add_argument('--edges-file',
                        default='../../dev/ablation_study/data/edges_filtered/edges_chatgpt-json-llm-user2.0-ai1.0_t0.9.json')
    args = parser.parse_args()

    dataset = build_temporal_dataset(args.embeddings_dir, args.edges_file)
    print_dataset_summary(dataset)
