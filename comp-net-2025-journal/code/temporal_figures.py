#!/usr/bin/env python3
"""
Publication-quality figures for the temporal network evolution analysis.

Generates 7 figures with consistent styling and phase shading.
"""

import json
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime
from pathlib import Path
from collections import defaultdict, Counter

# ─── Style Configuration ─────────────────────────────────────────────

def setup_style():
    """Match existing paper figure style."""
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'savefig.dpi': 300,
        'figure.dpi': 100,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
    })
    sns.set_style("whitegrid")


# Phase definitions
PHASES = [
    ('Phase 1:\nEarly', '2022-12', '2023-02', '#FFE0E0'),      # light red
    ('Phase 2:\nExploration', '2023-03', '2023-07', '#E0F0E0'),  # light green
    ('Phase 3:\nEstablished', '2023-08', '2024-01', '#E0E0FF'),  # light blue
    ('Phase 4:\nGPT-4o', '2024-02', '2024-09', '#FFF0E0'),      # light orange
    ('Phase 5:\nReasoning', '2024-10', '2025-04', '#F0E0FF'),    # light purple
]

# Topic colors (consistent with existing paper figures)
TOPIC_COLORS = {
    'ML/AI': '#1f77b4',
    'Programming': '#ff7f0e',
    'Stats/Math': '#2ca02c',
    'Philosophy': '#d62728',
    'Health': '#9467bd',
    'Networks': '#8c564b',
    'General': '#7f7f7f',
}


def month_to_date(month_str):
    """Convert 'YYYY-MM' to datetime for plotting."""
    return datetime.strptime(month_str + '-15', '%Y-%m-%d')


def add_phase_shading(ax, months=None):
    """Add phase shading as light pastel vertical bands."""
    for label, start, end, color in PHASES:
        start_dt = month_to_date(start)
        end_dt = month_to_date(end)
        ax.axvspan(start_dt, end_dt, alpha=0.3, color=color, zorder=0)


def add_phase_labels(ax, y_pos=0.97):
    """Add phase labels at top of axes."""
    for label, start, end, color in PHASES:
        start_dt = month_to_date(start)
        end_dt = month_to_date(end)
        mid = start_dt + (end_dt - start_dt) / 2
        ax.text(mid, y_pos, label, transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=7, color='gray', style='italic')


# ─── Figure 1: Growth Curves ─────────────────────────────────────────

def plot_growth_curves(metrics_csv: str, output_path: str):
    """1x3: Cumulative nodes+bar, cumulative edges+bar, edges/node."""
    setup_style()
    df = pd.read_csv(metrics_csv)
    dates = [month_to_date(m) for m in df['month']]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (0) Cumulative nodes with monthly bar
    ax = axes[0]
    add_phase_shading(ax)
    ax.plot(dates, df['total_nodes'], 'b-', linewidth=2, label='Total')
    ax.plot(dates, df['connected_nodes'], 'r-', linewidth=2, label='Connected')
    ax2 = ax.twinx()
    ax2.bar(dates, df['new_nodes'], width=20, alpha=0.3, color='steelblue', label='New/month')
    ax.set_ylabel('Cumulative Nodes')
    ax2.set_ylabel('New Nodes/Month', color='steelblue')
    ax.legend(loc='upper left')
    ax.set_title('(a) Node Accumulation')

    # (1) Cumulative edges with monthly bar
    ax = axes[1]
    add_phase_shading(ax)
    ax.plot(dates, df['edges'], 'r-', linewidth=2)
    ax2 = ax.twinx()
    ax2.bar(dates, df['new_edges'], width=20, alpha=0.3, color='coral', label='New/month')
    ax.set_ylabel('Cumulative Edges')
    ax2.set_ylabel('New Edges/Month', color='coral')
    ax.set_title('(b) Edge Accumulation')

    # (2) Edges per node (connectivity ratio)
    ax = axes[2]
    add_phase_shading(ax)
    ax.plot(dates, df['edges_per_node'], 'm-', linewidth=2)
    ax.set_ylabel('Edges per Connected Node')
    ax.set_title('(c) Connectivity Ratio')

    for ax in axes.flat:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    fig.suptitle('Network Growth Over 29 Months', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    _save_figure(fig, output_path, 'growth_curves')


# ─── Figure 2: Structural Evolution ──────────────────────────────────

def plot_structural_evolution(metrics_csv: str, output_path: str):
    """2x2: Modularity, community count, clustering+transitivity, giant component."""
    setup_style()
    df = pd.read_csv(metrics_csv)
    dates = [month_to_date(m) for m in df['month']]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (0,0) Modularity
    ax = axes[0, 0]
    add_phase_shading(ax)
    # Filter to months with meaningful data (>= 50 total nodes)
    mask = df['total_nodes'] >= 50
    d_filt = [d for d, m in zip(dates, mask) if m]
    ax.plot(d_filt, df.loc[mask, 'modularity'], 'b-o', linewidth=2, markersize=4)
    ax.set_ylabel('Modularity')
    ax.set_title('(a) Modularity')
    ax.set_ylim(0, 1)

    # (0,1) Community count
    ax = axes[0, 1]
    add_phase_shading(ax)
    ax.plot(d_filt, df.loc[mask, 'num_communities'], 'r-o', linewidth=2, markersize=4)
    ax.set_ylabel('Number of Communities')
    ax.set_title('(b) Community Count')

    # (1,0) Clustering + Transitivity
    ax = axes[1, 0]
    add_phase_shading(ax)
    ax.plot(d_filt, df.loc[mask, 'avg_clustering'], 'g-o', linewidth=2, markersize=4, label='Avg Clustering')
    ax.plot(d_filt, df.loc[mask, 'transitivity'], 'm-s', linewidth=2, markersize=4, label='Transitivity')
    ax.set_ylabel('Coefficient')
    ax.set_title('(c) Clustering & Transitivity')
    ax.legend()

    # (1,1) Giant component fraction
    ax = axes[1, 1]
    add_phase_shading(ax)
    ax.plot(d_filt, df.loc[mask, 'giant_component_fraction'], 'k-o', linewidth=2, markersize=4)
    ax.set_ylabel('Giant Component Fraction')
    ax.set_title('(d) Giant Component')
    ax.set_ylim(0, 1.05)

    for ax in axes.flat:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    fig.suptitle('Structural Evolution of the Knowledge Network', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    _save_figure(fig, output_path, 'structural_evolution')


# ─── Figure 3: Community Timeline ────────────────────────────────────

def plot_community_timeline(tracked_json: str, events_json: str, output_path: str):
    """
    Horizontal swimlane chart showing community lifespans and size evolution.
    Each significant community (max size >= 10) gets a horizontal ribbon whose
    height at each month is proportional to community size. Colored by topic.
    """
    setup_style()

    with open(tracked_json) as f:
        tracked = json.load(f)

    # Build per-community monthly data
    community_data = defaultdict(dict)  # {tid: {month: {size, topic}}}
    for tc in tracked:
        community_data[tc['tracked_id']][tc['month']] = {
            'size': tc['size'],
            'topic': tc['topic_dominant'],
        }

    # Filter to significant communities (max size >= 10)
    max_sizes = {}
    for tid, months in community_data.items():
        max_sizes[tid] = max(info['size'] for info in months.values())
    significant = {tid for tid, ms in max_sizes.items() if ms >= 10}

    if not significant:
        print("  Warning: no significant communities for timeline")
        return

    # Get dominant topic per community (most common across months)
    tid_topics = {}
    for tid in significant:
        topics = [info['topic'] for info in community_data[tid].values()]
        tid_topics[tid] = Counter(topics).most_common(1)[0][0]

    # Sort by birth date (earliest first at top), then by max size desc
    def sort_key(tid):
        birth = min(community_data[tid].keys())
        return (birth, -max_sizes[tid])
    sorted_ids = sorted(significant, key=sort_key)

    # All months in the dataset
    all_months = sorted({m for tc in tracked for m in [tc['month']]})
    month_dates = [month_to_date(m) for m in all_months]

    # Layout constants
    row_height = 1.0
    max_ribbon = 0.42  # max half-height of ribbon (leaves gap between rows)
    global_max_size = max(max_sizes[tid] for tid in sorted_ids)

    fig, ax = plt.subplots(figsize=(14, 0.7 * len(sorted_ids) + 2))

    # Phase shading first (behind everything)
    add_phase_shading(ax)

    for row, tid in enumerate(sorted_ids):
        y_center = len(sorted_ids) - 1 - row  # earliest at top
        months_data = community_data[tid]
        active_months = sorted(months_data.keys())

        # Build arrays for the ribbon
        dates = [month_to_date(m) for m in active_months]
        sizes = [months_data[m]['size'] for m in active_months]

        # Scale ribbon height: size -> half-height
        half_heights = [max_ribbon * (s / global_max_size) for s in sizes]

        topic = tid_topics[tid]
        color = TOPIC_COLORS.get(topic, '#7f7f7f')

        # Draw ribbon as filled area between y_center ± half_height
        upper = [y_center + h for h in half_heights]
        lower = [y_center - h for h in half_heights]
        ax.fill_between(dates, lower, upper, color=color, alpha=0.8,
                        edgecolor='white', linewidth=0.5)

        # Birth marker
        ax.plot(dates[0], y_center, '*', color=color, markersize=8,
                markeredgecolor='black', markeredgewidth=0.5, zorder=5)

        # Right-side label: topic (max size)
        label = f"{topic} ({max_sizes[tid]})"
        ax.text(month_to_date('2025-05'), y_center, label,
                va='center', ha='left', fontsize=8, color=color, fontweight='bold')

    # Configure axes
    ax.set_yticks(range(len(sorted_ids)))
    ax.set_yticklabels([f'C{tid}' for tid in reversed(sorted_ids)], fontsize=8)
    ax.set_ylim(-0.7, len(sorted_ids) - 0.3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Extend x-axis slightly for labels
    ax.set_xlim(month_to_date('2022-12'), month_to_date('2025-08'))

    # Phase labels at top
    add_phase_labels(ax)

    # Topic legend (deduplicated)
    legend_handles = {}
    for tid in sorted_ids:
        topic = tid_topics[tid]
        if topic not in legend_handles:
            legend_handles[topic] = mpatches.Patch(
                color=TOPIC_COLORS.get(topic, '#7f7f7f'), label=topic)
    # Add birth marker to legend
    from matplotlib.lines import Line2D
    birth_handle = Line2D([0], [0], marker='*', color='gray', markersize=8,
                          markeredgecolor='black', markeredgewidth=0.5,
                          linestyle='None', label='Birth month')
    handles = list(legend_handles.values()) + [birth_handle]
    ax.legend(handles=handles, loc='lower right', fontsize=9, ncol=2)

    ax.set_title('Community Lifecycles', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save_figure(fig, output_path, 'community_timeline')


# ─── Figure 4: Preferential Attachment ───────────────────────────────

def plot_preferential_attachment(attachment_csv: str, kernel_json: str,
                                 output_path: str):
    """1x3: Attachment kernel (log-log), correlation time series vs null, histogram."""
    setup_style()

    df = pd.read_csv(attachment_csv)
    with open(kernel_json) as f:
        kernel = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # (0) Attachment kernel (log-log)
    ax = axes[0]
    if kernel['bins'] and kernel['probs']:
        bins = np.array(kernel['bins'])
        probs = np.array(kernel['probs'])
        valid = (bins > 0) & (probs > 0)
        if valid.any():
            ax.scatter(bins[valid], probs[valid], s=50, c='steelblue', zorder=5)
            # Fit line
            log_k = np.log10(bins[valid])
            log_p = kernel['alpha'] * log_k + np.log10(probs[valid]).mean() - kernel['alpha'] * log_k.mean()
            ax.plot(bins[valid], 10**log_p, 'r--', linewidth=2,
                    label=f"$\\Pi(k) \\sim k^{{{kernel['alpha']:.2f}}}$\n$R^2 = {kernel['r_squared']:.3f}$")
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Degree $k$')
    ax.set_ylabel('Attachment Probability $\\Pi(k)$')
    ax.set_title('(a) Attachment Kernel')
    ax.legend(fontsize=10)

    # (1) Correlation time series vs null
    ax = axes[1]
    dates = [month_to_date(m) for m in df['month']]
    # Filter out months with 0 correlation (no data)
    mask = df['new_edges_from_new'] >= 3
    d_filt = [d for d, m in zip(dates, mask) if m]
    ax.plot(d_filt, df.loc[mask, 'correlation'], 'b-o', linewidth=2, markersize=4,
            label='Observed')
    ax.fill_between(d_filt,
                    df.loc[mask, 'null_mean'] - 2 * df.loc[mask, 'null_std'],
                    df.loc[mask, 'null_mean'] + 2 * df.loc[mask, 'null_std'],
                    alpha=0.3, color='gray', label='Null ±2σ')
    ax.plot(d_filt, df.loc[mask, 'null_mean'], 'k--', linewidth=1, alpha=0.5)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.set_ylabel('Spearman Correlation')
    ax.set_title('(b) Degree-Attachment Correlation')
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Add phase shading to correlation panel
    add_phase_shading(ax)

    # (2) Z-score distribution
    ax = axes[2]
    z_scores = df.loc[mask, 'z_score'].values
    ax.hist(z_scores, bins=15, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(x=1.96, color='red', linestyle='--', label='p=0.05')
    ax.axvline(x=-1.96, color='red', linestyle='--')
    ax.axvline(x=np.median(z_scores), color='orange', linestyle='-', linewidth=2,
               label=f'Median z={np.median(z_scores):.1f}')
    ax.set_xlabel('Z-score (vs null model)')
    ax.set_ylabel('Count (months)')
    ax.set_title('(c) Attachment Signal Strength')
    ax.legend(fontsize=9)

    fig.suptitle('Preferential Attachment Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save_figure(fig, output_path, 'preferential_attachment')


# ─── Figure 5: Bridge Dynamics ───────────────────────────────────────

def plot_bridge_dynamics(bridge_csv: str, output_path: str):
    """Betweenness over time for 5 bridges, creation date markers."""
    setup_style()

    df = pd.read_csv(bridge_csv)

    fig, ax = plt.subplots(figsize=(14, 6))
    add_phase_shading(ax)

    bridge_colors = {
        'geometric-mean-calculation': '#1f77b4',
        'mcts-code-analysis-suggestions': '#ff7f0e',
        'loss-in-llm-training': '#2ca02c',
        'algotree-generate-unit-tests-flattree': '#d62728',
        'compile-cuda-program-linux': '#9467bd',
    }

    bridge_labels = {
        'geometric-mean-calculation': 'Geometric Mean Calc.',
        'mcts-code-analysis-suggestions': 'MCTS Code Analysis',
        'loss-in-llm-training': 'Loss in LLM Training',
        'algotree-generate-unit-tests-flattree': 'AlgoTree Unit Tests',
        'compile-cuda-program-linux': 'CUDA Compilation',
    }

    for bridge_id in df['node_id'].unique():
        bdf = df[df['node_id'] == bridge_id].copy()
        dates = [month_to_date(m) for m in bdf['month']]
        color = bridge_colors.get(bridge_id, 'gray')
        label = bridge_labels.get(bridge_id, bridge_id)

        ax.plot(dates, bdf['betweenness'], '-o', color=color, linewidth=2,
                markersize=4, label=label)

        # Mark creation month (first appearance)
        ax.axvline(x=dates[0], color=color, linestyle=':', alpha=0.5)
        ax.annotate('★', xy=(dates[0], bdf['betweenness'].iloc[0]),
                    fontsize=14, color=color, ha='center', va='bottom')

    ax.set_ylabel('Betweenness Centrality')
    ax.set_xlabel('Month')
    ax.set_title('Bridge Conversation Dynamics Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, ncol=2)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    add_phase_labels(ax)

    plt.tight_layout()
    _save_figure(fig, output_path, 'bridge_dynamics')


# ─── Figure 6: Model Era Comparison ──────────────────────────────────

def plot_model_era_comparison(era_csv: str, output_path: str):
    """2x3 bar charts comparing metrics across model eras."""
    setup_style()

    df = pd.read_csv(era_csv)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    era_labels = {
        'gpt35': 'GPT-3.5',
        'gpt4': 'GPT-4',
        'gpt4o': 'GPT-4o',
        'reasoning': 'Reasoning\n(o1/o3)',
        'gpt45': 'GPT-4.5',
    }
    era_colors = ['#FFB3B3', '#B3D9FF', '#B3FFB3', '#FFE0B3', '#E0B3FF']

    metrics = [
        ('density', 'Density', '(a)'),
        ('avg_degree', 'Avg Degree', '(b)'),
        ('avg_clustering', 'Avg Clustering', '(c)'),
        ('modularity', 'Modularity', '(d)'),
        ('num_communities', 'Communities', '(e)'),
        ('avg_betweenness', 'Avg Betweenness', '(f)'),
    ]

    for idx, (metric, ylabel, panel) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        x = np.arange(len(df))
        labels = [era_labels.get(e, e) for e in df['era']]
        values = df[metric].values.astype(float)

        bars = ax.bar(x, values, color=era_colors[:len(x)], edgecolor='gray',
                      linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{panel} {ylabel}')

        # Add value labels on bars
        for bar, val in zip(bars, values):
            if val > 0:
                fmt = f'{val:.4f}' if val < 0.01 else (f'{val:.3f}' if val < 1 else f'{val:.1f}')
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        fmt, ha='center', va='bottom', fontsize=8)

    fig.suptitle('Network Metrics by Model Era', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    _save_figure(fig, output_path, 'model_era_comparison')


# ─── Figure 7: Densification Law ─────────────────────────────────────

def plot_densification_law(densification_json: str, output_path: str):
    """Log-log edges vs nodes, power law fit, gamma annotation."""
    setup_style()

    with open(densification_json) as f:
        dens = json.load(f)

    fig, ax = plt.subplots(figsize=(8, 6))

    nodes = np.array(dens['nodes'])
    edges = np.array(dens['edges'])

    ax.scatter(nodes, edges, s=60, c='steelblue', zorder=5, edgecolors='white',
               linewidths=0.5, label='Monthly snapshots')

    # Fit line
    if nodes.size > 2:
        log_n = np.log10(nodes)
        n_fit = np.logspace(np.log10(nodes.min()), np.log10(nodes.max()), 100)
        log_e_fit = dens['alpha'] * np.log10(n_fit) + dens['intercept']
        ax.plot(n_fit, 10**log_e_fit, 'r--', linewidth=2,
                label=f"$e(t) \\sim n(t)^{{{dens['alpha']:.2f}}}$\n$R^2 = {dens['r_squared']:.3f}$")

    # Reference lines
    n_ref = np.logspace(np.log10(nodes.min()), np.log10(nodes.max()), 50)
    # Linear reference (gamma=1)
    scale_1 = edges.mean() / nodes.mean()
    ax.plot(n_ref, scale_1 * n_ref, ':', color='gray', alpha=0.5,
            label='$\\gamma=1$ (constant avg degree)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Connected Nodes $n(t)$')
    ax.set_ylabel('Edges $e(t)$')
    ax.set_title('Densification Law: Super-linear Edge Growth', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')

    # Annotate gamma
    ax.text(0.95, 0.05,
            f"$\\gamma = {dens['alpha']:.3f}$\nSuper-linear\ndensification",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=12,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    _save_figure(fig, output_path, 'densification_law')


# ─── Utility ─────────────────────────────────────────────────────────

def _save_figure(fig, output_dir, name):
    """Save figure as both PDF and PNG."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_dir / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(output_dir / f'{name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {name}.pdf and {name}.png to {output_dir}")


def generate_all_figures(data_dir: str, output_dir: str):
    """Generate all 7 figures."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    print("\nGenerating figures...")

    print("  [1/7] Growth curves...")
    plot_growth_curves(data_dir / 'temporal_metrics.csv', output_dir)

    print("  [2/7] Structural evolution...")
    plot_structural_evolution(data_dir / 'temporal_metrics.csv', output_dir)

    print("  [3/7] Community timeline...")
    plot_community_timeline(
        data_dir / 'tracked_communities.json',
        data_dir / 'community_events.json',
        output_dir,
    )

    print("  [4/7] Preferential attachment...")
    plot_preferential_attachment(
        data_dir / 'preferential_attachment.csv',
        data_dir / 'attachment_kernel.json',
        output_dir,
    )

    print("  [5/7] Bridge dynamics...")
    plot_bridge_dynamics(data_dir / 'bridge_dynamics.csv', output_dir)

    print("  [6/7] Model era comparison...")
    plot_model_era_comparison(data_dir / 'model_era_metrics.csv', output_dir)

    print("  [7/7] Densification law...")
    plot_densification_law(data_dir / 'densification.json', output_dir)

    print("\nAll figures generated successfully!")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate temporal analysis figures')
    parser.add_argument('--data-dir', default='../data/temporal')
    parser.add_argument('--output-dir', default='../paper/figures/temporal')
    args = parser.parse_args()

    generate_all_figures(args.data_dir, args.output_dir)
