# Data Directory

Derived data supporting the analyses in both papers of this research compendium. All files here are sufficient to reproduce every figure and table in the published results.

**Papers supported:**

| Paper | Venue | Data |
|-------|-------|------|
| Temporal Evolution of Cognitive Knowledge Networks | PLOS Complex Systems | `temporal/` |
| Cognitive MRI of AI Conversations | Complex Networks 2025 (Springer) | `ablation/` |
| Both papers | — | `embeddings/`, `network/` |

## Temporal Analysis Data (`temporal/`)

Monthly network snapshots tracking the evolution of a semantic similarity network built from 1,908 ChatGPT conversations (December 2022 -- April 2025). These files back all figures and tables in the journal paper.

### `temporal_metrics.csv`

Monthly aggregate network statistics. One row per month.

| Column | Description |
|--------|-------------|
| `month` | Year-month (YYYY-MM) |
| `total_nodes` | Cumulative conversations added |
| `connected_nodes` | Nodes with at least one edge above threshold |
| `edges` | Number of edges in the network |
| `density` | Edge density |
| `num_components` | Connected components |
| `giant_component_size` | Nodes in the largest component |
| `giant_component_fraction` | Giant component / connected nodes |
| `avg_degree`, `std_degree`, `max_degree` | Degree distribution statistics |
| `avg_clustering` | Average clustering coefficient |
| `transitivity` | Global transitivity |
| `avg_shortest_path` | Mean shortest path in giant component |
| `modularity` | Louvain modularity |
| `num_communities` | Community count |
| `avg_betweenness` | Mean betweenness centrality |
| `avg_degree_centrality` | Mean degree centrality |
| `assortativity` | Degree assortativity |
| `new_nodes`, `new_edges` | Nodes/edges added this month |
| `edges_per_node` | Running edges per node ratio |
| `isolated_nodes` | Nodes with no edges |

*Used in:* Figures 1 (growth curves), 2 (structural evolution), 3 (densification law)

### `bridge_dynamics.csv`

Per-node, per-month bridge metrics tracking how individual conversations serve as cross-community connectors.

| Column | Description |
|--------|-------------|
| `node_id` | Conversation identifier |
| `month` | Year-month |
| `betweenness` | Betweenness centrality |
| `degree` | Node degree |
| `num_neighbor_communities` | Communities connected through this node |
| `is_bridge_threshold` | Boolean: qualifies as bridge node |

*Used in:* Figure 5 (bridge dynamics)

### `preferential_attachment.csv`

Monthly test of preferential attachment: do new conversations preferentially connect to high-degree existing nodes?

| Column | Description |
|--------|-------------|
| `month` | Year-month |
| `new_nodes` | New conversations added |
| `new_edges_from_new` | Edges from new to existing nodes |
| `correlation` | Pearson r between existing degree and new edges received |
| `p_value` | Statistical significance |
| `null_mean`, `null_std` | Null model (random attachment) parameters |
| `z_score` | Standardized effect size vs. null |

*Used in:* Figure 6 (preferential attachment)

### `model_era_metrics.csv`

Network statistics segmented by ChatGPT model era (GPT-3.5, GPT-4, GPT-4o, Reasoning).

| Column | Description |
|--------|-------------|
| `era` | Model era (`gpt35`, `gpt4`, `gpt4o`, `reasoning`) |
| `total_nodes` ... `avg_betweenness` | Same metrics as `temporal_metrics.csv` |

*Used in:* Figure 7 (model era comparison)

### `densification.json`

Power-law fit for the densification law: E(t) ~ N(t)^alpha.

| Key | Description |
|-----|-------------|
| `alpha` | Densification exponent |
| `r_squared` | Goodness of fit |
| `intercept` | Log-space intercept |
| `p_value` | Significance of fit |
| `nodes`, `edges` | Raw data arrays used for fitting |

*Used in:* Figure 3 (densification law)

### `attachment_kernel.json`

Attachment kernel analysis: probability of new edges as a function of existing degree.

| Key | Description |
|-----|-------------|
| `alpha` | Power-law exponent of attachment kernel |
| `r_squared` | Goodness of fit |
| `bins`, `probs` | Degree bins and corresponding attachment probabilities |

*Used in:* Figure 6 (preferential attachment)

### `community_events.json`

List of 252 community-level events (births, deaths, merges, splits, grows, shrinks) detected across monthly snapshots.

*Used in:* Figure 4 (community timeline)

### `tracked_communities.json`

229 tracked community entities with membership continuity across time, used to construct the community lifecycle visualization.

*Used in:* Figure 4 (community timeline)

## Ablation Study Data (`ablation/`)

Results from a 63-configuration ablation study (9 user:AI weight ratios x 7 similarity thresholds) validating the pipeline parameters reported in the conference paper.

### `ablation_2d_results.csv` / `ablation_2d_results.json`

Full results matrix: network statistics for all 63 configurations. CSV and JSON formats contain the same data.

Key columns: `num_nodes`, `num_edges`, `density`, `avg_degree`, `avg_clustering`, `modularity`, `num_communities`, `user_weight`, `ai_weight`, `ratio`, `threshold`.

### `key_configurations_summary.csv` / `key_configurations_summary.tex`

Summary table of notable configurations (optimal, extremes, baselines) in CSV and LaTeX formats.

### `threshold_analysis_metadata.json` / `edge_generation_metadata.json`

Pipeline metadata: thresholds tested, embedding sources, generation timestamps, edge counts before/after filtering.

### `community_evolution_summary.md`

Narrative summary of how community structure changes across the parameter space.

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

## Embeddings (`embeddings/`)

Per-conversation semantic embeddings for all 1,908 conversations, generated at the optimal 2:1 user:AI weight ratio. Messages have been stripped for privacy; only embeddings and metadata are included.

Each JSON file contains:

| Key | Description |
|-----|-------------|
| `title` | Conversation title |
| `conversation_id` | OpenAI conversation UUID |
| `model` | ChatGPT model used (when available) |
| `created`, `updated` | Timestamps |
| `embeddings.role_aggregate.vector` | 768-dim weighted embedding (user=2.0, ai=1.0) |
| `embeddings.role_aggregate.per_role.user.vector` | 768-dim user-message embedding |
| `embeddings.role_aggregate.per_role.assistant.vector` | 768-dim AI-response embedding |
| `embeddings.role_aggregate.metadata` | Algorithm, weights, aggregation method |

The per-role vectors are the raw Ollama outputs (model: `nomic-embed-text`) and are identical across all weight ratio configurations. Any weight ratio can be reconstructed by re-combining the per-role vectors.

*Used in:* Both papers — these are the foundation for all network construction.

## Primary Network (`network/`)

### `edges_user2.0-ai1.0_t0.9.json`

The edge list for the primary network used in both papers: 2:1 user:AI weight ratio, θ=0.9 similarity threshold. Each entry is `[node_a, node_b, cosine_similarity]`. Contains 601 connected nodes and 1,718 edges across all components. The giant component (449 nodes, 1,615 edges) analyzed in the conference paper is the largest connected component of this network.

*Used in:* All network analyses in both papers.

## Conversation Corpus (`conversations/`)

See `conversations/README.md` for details. The raw conversations are not yet included due to ongoing privacy sanitization. The derived data in `temporal/` and `ablation/` is sufficient to reproduce all published results.

## Regenerating Large Artifacts

The following large intermediate artifacts are excluded from this repository but can be regenerated using the analysis pipeline:

| Artifact | Size | Command |
|----------|------|---------|
| All 9 weight-ratio embeddings | ~1.8 GB | `python cli.py node-embeddings ...` (requires Ollama) |
| Unfiltered edge lists | ~1.6 GB | `python cli.py edges-gpu ...` |
| All 63 filtered edge lists | ~98 MB | `python cli.py cut-off ...` |

Note: The primary embedding configuration (2:1 ratio) is included in `embeddings/`, and the primary edge list is in `network/`. Only the remaining 8 configurations and 62 edge lists need regeneration.

The pipeline code is available at [chatgpt-complex-net](https://github.com/queelius/chatgpt-complex-net) (DOI: [10.5281/zenodo.15314235](https://doi.org/10.5281/zenodo.15314235)).

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
