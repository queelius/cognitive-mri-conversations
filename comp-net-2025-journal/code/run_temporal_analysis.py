#!/usr/bin/env python3
"""
CLI runner orchestrating all temporal network evolution analyses.

Usage:
    # Run everything
    python run_temporal_analysis.py

    # Selective analyses
    python run_temporal_analysis.py --analyses growth community
    python run_temporal_analysis.py --analyses attachment bridges eras densification

    # Custom paths
    python run_temporal_analysis.py \
        --embeddings-dir ../../dev/ablation_study/data/embeddings/chatgpt-json-llm-user2.0-ai1.0 \
        --edges-file ../../dev/ablation_study/data/edges_filtered/edges_chatgpt-json-llm-user2.0-ai1.0_t0.9.json \
        --output-dir ../paper/figures/temporal \
        --data-output-dir ../data/temporal
"""

import argparse
import time
import traceback
from pathlib import Path


ALL_ANALYSES = ['growth', 'community', 'attachment', 'bridges', 'eras', 'densification', 'figures']


def run_analysis(name, func, *args, **kwargs):
    """Run a single analysis step with error handling and timing."""
    print(f"\n{'─'*60}")
    print(f"  Running: {name}")
    print(f"{'─'*60}")
    t0 = time.time()
    try:
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        print(f"  ✓ {name} completed in {elapsed:.1f}s")
        return result
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  ✗ {name} FAILED after {elapsed:.1f}s: {e}")
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Temporal Network Evolution Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--embeddings-dir',
                        default='../../dev/ablation_study/data/embeddings/chatgpt-json-llm-user2.0-ai1.0',
                        help='Directory with conversation embedding JSONs')
    parser.add_argument('--edges-file',
                        default='../../dev/ablation_study/data/edges_filtered/edges_chatgpt-json-llm-user2.0-ai1.0_t0.9.json',
                        help='Filtered edges JSON file')
    parser.add_argument('--output-dir', default='../paper/figures/temporal',
                        help='Directory for figure output')
    parser.add_argument('--data-output-dir', default='../data/temporal',
                        help='Directory for data output (CSV/JSON)')
    parser.add_argument('--analyses', nargs='+', choices=ALL_ANALYSES,
                        default=ALL_ANALYSES,
                        help='Which analyses to run (default: all)')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    # Create output directories
    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    analyses = set(args.analyses)
    total_t0 = time.time()

    print("="*60)
    print("  TEMPORAL NETWORK EVOLUTION ANALYSIS")
    print(f"  Analyses: {', '.join(sorted(analyses))}")
    print("="*60)

    # ─── Step 1: Load data (always needed) ───────────────────────────

    from temporal_data_loader import build_temporal_dataset, print_dataset_summary

    dataset = run_analysis(
        "Load temporal dataset",
        build_temporal_dataset,
        args.embeddings_dir, args.edges_file,
    )
    if dataset is None:
        print("FATAL: Could not load dataset. Aborting.")
        return
    print_dataset_summary(dataset)

    # ─── Step 2: Build cumulative snapshots ──────────────────────────

    snapshots = None
    if analyses & {'growth', 'densification', 'figures'}:
        from temporal_snapshots import build_cumulative_snapshots, save_snapshot_metrics

        snapshots = run_analysis(
            "Build cumulative monthly snapshots",
            build_cumulative_snapshots,
            dataset, random_state=args.random_state,
        )
        if snapshots:
            save_snapshot_metrics(snapshots, data_dir / 'temporal_metrics.csv')

    # ─── Step 3: Track communities ───────────────────────────────────

    tracked = None
    events = None
    if analyses & {'community', 'figures'}:
        from temporal_community_tracker import (
            run_louvain_per_snapshot, track_communities,
            save_tracked_communities, save_community_events,
        )

        snapshot_communities = run_analysis(
            "Run Louvain per snapshot",
            run_louvain_per_snapshot,
            dataset, random_state=args.random_state,
        )
        if snapshot_communities:
            result = run_analysis(
                "Track communities across time",
                track_communities,
                snapshot_communities,
            )
            if result:
                tracked, events = result
                save_tracked_communities(tracked, data_dir / 'tracked_communities.json')
                save_community_events(events, data_dir / 'community_events.json')

                from collections import Counter
                event_counts = Counter(e.event_type for e in events)
                print(f"    Event summary: {dict(event_counts.most_common())}")
                print(f"    Unique tracked communities: {len(set(tc.tracked_id for tc in tracked))}")

    # ─── Step 4: Preferential attachment ─────────────────────────────

    if 'attachment' in analyses:
        from temporal_preferential_attachment import (
            test_preferential_attachment, save_attachment_results,
        )

        result = run_analysis(
            "Test preferential attachment",
            test_preferential_attachment,
            dataset, n_permutations=1000, random_state=args.random_state,
        )
        if result:
            monthly, kernel = result
            save_attachment_results(monthly, kernel, data_dir / 'preferential_attachment.csv')
            print(f"    Attachment kernel alpha = {kernel['alpha']:.3f} (R² = {kernel['r_squared']:.3f})")

    # ─── Step 5: Bridge dynamics ─────────────────────────────────────

    if 'bridges' in analyses:
        from temporal_preferential_attachment import (
            track_bridge_dynamics, save_bridge_dynamics,
        )

        bridges = run_analysis(
            "Track bridge formation dynamics",
            track_bridge_dynamics,
            dataset, random_state=args.random_state,
        )
        if bridges:
            save_bridge_dynamics(bridges, data_dir / 'bridge_dynamics.csv')
            bridge_ids = set(b.node_id for b in bridges)
            print(f"    Tracked {len(bridge_ids)} bridges across {len(bridges)} snapshots")

    # ─── Step 6: Model era comparison ────────────────────────────────

    if 'eras' in analyses:
        from temporal_preferential_attachment import (
            compute_model_era_metrics, save_model_era_metrics,
        )

        era_metrics = run_analysis(
            "Compare model era sub-networks",
            compute_model_era_metrics,
            dataset, random_state=args.random_state,
        )
        if era_metrics:
            save_model_era_metrics(era_metrics, data_dir / 'model_era_metrics.csv')
            for em in era_metrics:
                print(f"    {em['era']:12s}: {em['connected_nodes']:4d} connected, "
                      f"mod={em['modularity']:.3f}")

    # ─── Step 7: Densification law ───────────────────────────────────

    if 'densification' in analyses:
        import json
        from temporal_preferential_attachment import compute_densification

        metrics_csv = data_dir / 'temporal_metrics.csv'
        if metrics_csv.exists():
            dens = run_analysis(
                "Compute densification law",
                compute_densification,
                str(metrics_csv),
            )
            if dens:
                with open(data_dir / 'densification.json', 'w') as f:
                    json.dump(dens, f, indent=2)
                print(f"    Densification alpha = {dens['alpha']:.3f} (R² = {dens['r_squared']:.3f})")
        else:
            print("  Skipping densification — temporal_metrics.csv not found")

    # ─── Step 8: Generate figures ────────────────────────────────────

    if 'figures' in analyses:
        from temporal_figures import generate_all_figures

        run_analysis(
            "Generate all figures",
            generate_all_figures,
            str(data_dir), str(output_dir),
        )

    # ─── Summary ─────────────────────────────────────────────────────

    total_elapsed = time.time() - total_t0
    print(f"\n{'='*60}")
    print(f"  COMPLETE — Total time: {total_elapsed:.1f}s")
    print(f"{'='*60}")

    # List outputs
    print(f"\n  Data outputs ({data_dir}):")
    for f in sorted(data_dir.glob('*')):
        size = f.stat().st_size
        print(f"    {f.name:40s} {size:>8,d} bytes")

    print(f"\n  Figure outputs ({output_dir}):")
    for f in sorted(output_dir.glob('*')):
        size = f.stat().st_size
        print(f"    {f.name:40s} {size:>8,d} bytes")

    # Key findings
    print(f"\n  KEY FINDINGS:")
    metrics_csv = data_dir / 'temporal_metrics.csv'
    if metrics_csv.exists():
        import pandas as pd
        df = pd.read_csv(metrics_csv)
        last = df.iloc[-1]
        print(f"    Final network: {int(last['total_nodes'])} total nodes, "
              f"{int(last['connected_nodes'])} connected, {int(last['edges'])} edges")
        print(f"    Final modularity: {last['modularity']:.3f}")
        print(f"    Final communities: {int(last['num_communities'])}")

    dens_json = data_dir / 'densification.json'
    if dens_json.exists():
        import json
        with open(dens_json) as f:
            dens = json.load(f)
        print(f"    Densification: alpha={dens['alpha']:.3f} (super-linear)" if dens['alpha'] > 1
              else f"    Densification: alpha={dens['alpha']:.3f}")

    kernel_json = data_dir / 'attachment_kernel.json'
    if kernel_json.exists():
        import json
        with open(kernel_json) as f:
            kernel = json.load(f)
        print(f"    Preferential attachment: alpha={kernel['alpha']:.3f} (sub-linear)")


if __name__ == '__main__':
    main()
