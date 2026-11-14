# Data Directory

Aggregate data from the parameter study behind the Complex Networks 2025 paper. The per-conversation embeddings and the edge list that the paper's network was built from are not published, since they identify individual conversations (removed for privacy on 14 November 2026); they can be regenerated from a conversation export with the pipeline in `code/` or [chatgpt-complex-net](https://github.com/queelius/chatgpt-complex-net) (DOI: [10.5281/zenodo.15314235](https://doi.org/10.5281/zenodo.15314235)).

## Ablation Study Data (`ablation/`)

Results from a 63-configuration ablation study (9 user:AI weight ratios x 7 similarity thresholds) validating the pipeline parameters reported in the conference paper.

### `ablation_2d_results.csv` / `ablation_2d_results.json`

Full results matrix: network statistics for all 63 configurations. CSV and JSON formats contain the same data.

Key columns: `num_nodes`, `num_edges`, `density`, `avg_degree`, `avg_clustering`, `modularity`, `num_communities`, `user_weight`, `ai_weight`, `ratio`, `threshold`.

### `key_configurations_summary.csv` / `key_configurations_summary.tex`

Summary table of notable configurations (optimal, extremes, baselines) in CSV and LaTeX formats.

### `threshold_analysis_metadata.json` / `edge_generation_metadata.json`

Pipeline metadata: thresholds tested, embedding sources, generation timestamps, edge counts before/after filtering.

### `focused/`

Slices through the parameter space holding one dimension fixed:

- `fixed_ratio_2-1.csv` -- Metrics across all thresholds at the optimal 2:1 user:AI ratio
- `fixed_threshold_0.9.csv` -- Metrics across all ratios at the optimal theta=0.9 threshold

### `figures/`

Publication-quality figures (PDF only) from the ablation analysis:

- `metrics_heatmaps_2d.pdf` -- Heatmaps of key metrics across the full parameter space
- `threshold_evolution.pdf` -- Network metrics as a function of threshold
- `weight_ratio_analysis_t0.9.pdf` -- Effect of user:AI ratio at theta=0.9
- `threshold_analysis_r2.0-1.0.pdf` -- Effect of threshold at 2:1 ratio

## Provenance

| Parameter | Value |
|-----------|-------|
| Embedding model | `nomic-embed-text` (768-dim, 8192 token context) |
| User:AI weight ratio | 2:1 (α = 2.0) |
| Similarity threshold | θ = 0.9 |
| Community detection | Louvain method |
| Corpus | 1,908 ChatGPT conversations, Dec 2022 -- Apr 2025 |
| Full network | 601 nodes, 1,718 edges (all connected components) |
| Giant component | 449 nodes, 1,615 edges, 15 communities (modularity 0.750) |
