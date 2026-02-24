# Papermill State — Cognitive MRI of AI Conversations (Journal Extension)

> Last refreshed: 2026-02-24

## Project Identity

| Field | Value |
|-------|-------|
| Title | Temporal Evolution of Cognitive Knowledge Networks in AI-Assisted Conversations |
| Stage | **drafting** — analysis code complete, paper not yet written |
| Format | LaTeX (PLOS template, not yet set up) |
| Venue | PLOS Complex Systems — Special Issue on Complex Networks 2025 |
| Submission ID | 41 |
| Deadline | **2026-02-27** |
| Base paper | Complex Networks 2025 (Springer proceedings, published) |

## Authors

| Name | Email | ORCID | Affiliation |
|------|-------|-------|-------------|
| Alexander Towell | atowell@siue.edu | 0000-0001-6443-9897 | Southern Illinois University Edwardsville |
| John Matta | jmatta@siue.edu | — | Southern Illinois University Edwardsville |

## Thesis

**Claim**: The semantic similarity network of AI conversation archives exhibits temporal evolution patterns — super-linear densification, sub-linear preferential attachment, stable community structure, and bridge emergence dynamics — that reveal how personal knowledge exploration accelerates and self-organizes over time.

**Novelty beyond conference paper**:
1. Temporal network evolution analysis (29 monthly cumulative snapshots)
2. Densification law discovery (α=1.405, super-linear)
3. Preferential attachment kernel quantification (α=0.763, sub-linear)
4. Community lifecycle tracking via Jaccard alignment (35 tracked communities)
5. Bridge formation dynamics (5 bridges tracked from creation)
6. Model era sub-network comparison (GPT-3.5 through GPT-4.5)

**Conference paper contribution** (baseline): Static network analysis of 449 conversations at θ=0.9, 15 communities, heterogeneous topology, bridge conversation taxonomy, 63-configuration ablation study.

## Venue Requirements

- 30%+ new original material beyond conference proceedings
- Conference paper must be cited
- Must clearly state novelty vs conference paper
- iThenticate similarity must stay below 30%
- PLOS Complex Systems format (open access)

## Repository Structure

```
comp-net-2025-camera-ready/          # PUBLISHED conference paper
  paper/paper.tex                    # Camera-ready (svproc.cls)
  slides/slide-pretty.tex            # Conference presentation

comp-net-2025-journal/               # JOURNAL EXTENSION (this work)
  code/
    temporal_data_loader.py          # Data loading, model era classification
    temporal_snapshots.py            # 29 cumulative monthly snapshots
    temporal_community_tracker.py    # Jaccard-based community tracking
    temporal_preferential_attachment.py  # Attachment, bridges, densification
    temporal_figures.py              # 7 publication-quality figures
    run_temporal_analysis.py         # CLI orchestrator (38s full pipeline)
    # Legacy multi-layer code (not used for temporal extension):
    core_types.py, concept_extraction.py, instantiation_scoring.py,
    association_network.py, hidden_connections.py, multilayer_analysis.py,
    visualization.py
  data/temporal/                     # Generated analysis data
    temporal_metrics.csv             # 29 rows, full network metrics per month
    tracked_communities.json         # 231 community records across time
    community_events.json            # 251 lifecycle events
    preferential_attachment.csv      # 28 monthly attachment test results
    attachment_kernel.json           # Pooled kernel (α=0.763)
    bridge_dynamics.csv              # 57 bridge snapshots
    model_era_metrics.csv            # 5 era sub-network metrics
    densification.json               # α=1.405, R²=0.993
  paper/
    figures/temporal/                 # 7 figures × {pdf,png}
      growth_curves.pdf
      structural_evolution.pdf
      community_timeline.pdf
      preferential_attachment.pdf
      bridge_dynamics.pdf
      model_era_comparison.pdf
      densification_law.pdf
    # paper.tex — NOT YET CREATED

code/                                # Original conference pipeline
dev/                                 # Raw data (gitignored)
  ablation_study/data/embeddings/    # 1908 conversation JSONs
  ablation_study/data/edges_filtered/ # Pre-computed edges
```

## Key Research Parameters

| Parameter | Value | Source |
|-----------|-------|-------|
| Dataset | 1908 ChatGPT conversations | Dec 2022 – Apr 2025 |
| Connected nodes | 601 (at θ=0.9) | 1307 isolated |
| Edges | 1718 | Cosine similarity ≥ 0.9 |
| User:AI weight | 2:1 (α=2.0) | Ablation study |
| Threshold | θ=0.9 | Phase transition analysis |
| Embedding model | nomic-embed-text (768-dim) | Ollama API |
| Monthly snapshots | 29 | Dec 2022 – Apr 2025 |

## Key Findings

| Finding | Value | Interpretation |
|---------|-------|----------------|
| Densification exponent | α=1.405 (R²=0.993) | Super-linear — knowledge exploration accelerates |
| Preferential attachment | α=0.763 (R²=0.914) | Sub-linear — between random and Barabási-Albert |
| Modularity stabilization | ~0.75 by mid-2023 | Community structure emerges early, persists |
| Community lifecycles | 35 unique, 196 continuations | Stable knowledge domains with gradual births |
| Bridge emergence | 5 tracked from creation | "Geometric Mean Calc." dominates throughout |
| Final modularity | 0.750 | Matches conference paper |
| Final communities | 14–15 | Consistent with conference paper |

## Model Era Distribution

| Era | Conversations | Connected | Key Metric |
|-----|--------------|-----------|------------|
| GPT-3.5 (None) | 1214 | 360 | Modularity 0.675 |
| GPT-4 | 44 | 15 | Density 0.095 |
| GPT-4o | 453 | 112 | Modularity 0.668 |
| Reasoning (o1/o3) | 181 | 32 | Modularity 0.353 |
| GPT-4.5 | 16 | 2 | Too small for metrics |

## Five Phases (for figure shading)

1. **Early** (Dec 2022 – Feb 2023): 59 conversations, network bootstrapping
2. **Exploration** (Mar – Jul 2023): 635 conversations, rapid growth burst
3. **Established** (Aug 2023 – Jan 2024): 402 conversations, structure consolidation
4. **GPT-4o** (Feb – Sep 2024): 396 conversations, new model capabilities
5. **Reasoning** (Oct 2024 – Apr 2025): 416 conversations, reasoning model era

## What's Done

- [x] Temporal analysis pipeline (6 Python modules)
- [x] All 7 publication-quality figures generated
- [x] All intermediate data cached as CSV/JSON
- [x] Full pipeline verified (38s end-to-end)
- [x] Sanity checks passed (final metrics match conference paper)

## What's Remaining

- [ ] Write journal paper LaTeX (PLOS template)
- [ ] Set up PLOS Complex Systems LaTeX template
- [ ] Literature survey for temporal network evolution references
- [ ] Write introduction (frame temporal extension)
- [ ] Write methods (temporal snapshot construction, community tracking, attachment testing)
- [ ] Write results (present findings with figure references)
- [ ] Write discussion (implications for knowledge exploration, distributed cognition)
- [ ] Ensure 30%+ new material vs conference paper
- [ ] Run iThenticate similarity check
- [ ] Get co-author review

## Prior Art (to survey)

- Leskovec et al. — densification laws in real-world networks
- Barabási & Albert — preferential attachment
- Palla et al. — overlapping community dynamics
- Holme & Saramäki — temporal networks review
- Mucha et al. — community structure in time-dependent networks
- Conference paper's own reference list (to cite and extend)

## Experiments

| Name | Status | Output |
|------|--------|--------|
| Cumulative snapshots (29 months) | complete | temporal_metrics.csv |
| Community tracking (Jaccard) | complete | tracked_communities.json, community_events.json |
| Preferential attachment (1000 permutations) | complete | preferential_attachment.csv, attachment_kernel.json |
| Bridge dynamics (5 bridges) | complete | bridge_dynamics.csv |
| Densification law | complete | densification.json |
| Model era comparison | complete | model_era_metrics.csv |
| Figures (7 total) | complete | paper/figures/temporal/*.{pdf,png} |

## Review History

None yet — paper not written.
