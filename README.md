# Cognitive MRI of AI Conversations: Research Compendium

<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->

Applying complex network analysis to ChatGPT conversation archives to reveal knowledge organization, community structure, and temporal evolution patterns.

**Authors:**
[Alexander Towell](https://orcid.org/0000-0001-6443-9897) and
[John Matta](https://orcid.org/0000-0002-7666-1409)
— Southern Illinois University Edwardsville

## Overview

This repository transforms sequential AI conversation logs into semantic similarity networks, revealing latent cognitive structure in AI-assisted knowledge exploration. We analyze 1,908 ChatGPT conversations spanning December 2022 to April 2025, constructing networks that expose knowledge communities, bridge conversations, and temporal evolution patterns invisible in linear logs.

## Papers

| Paper | Venue | Status |
|-------|-------|--------|
| **Temporal Evolution of Cognitive Knowledge Networks in AI-Assisted Conversations** | PLOS Complex Systems | Submitted |
| **Cognitive MRI of AI Conversations: Analyzing AI Interactions through Semantic Embedding Networks** | Complex Networks 2025 (Springer) | Published |

## Repository Structure

```
├── comp-net-2025-journal/           # Journal extension (PLOS Complex Systems)
│   └── paper/PLOS/                  # Submission-ready paper, figures, refs
├── comp-net-2025-camera-ready/      # Published conference paper (Springer)
│   ├── paper/                       # Camera-ready paper
│   └── slides/                      # Conference presentation (Beamer)
├── data/                            # Reproducibility data for both papers
│   ├── embeddings/                  # 1,908 conversation embeddings (2:1 ratio)
│   ├── network/                     # Primary edge list (449 nodes, 1,615 edges)
│   ├── temporal/                    # Journal paper: monthly network snapshots
│   ├── ablation/                    # Conference paper: 63-config parameter study
│   └── conversations/               # Placeholder (sanitization in progress)
├── code/                            # Analysis pipeline
│   ├── cli.py                       # Main CLI (embeddings, edges, export)
│   ├── networks.py                  # Network statistics & metrics
│   ├── embedding/                   # LLM & TF-IDF embedding models
│   ├── graph/                       # Edge generation, GPU acceleration
│   └── run_ablation_study.py        # 63-config ablation study
└── dev/                             # Research data (conversations, ablation results)
```

## Quick Start

```bash
cd code
source venv/bin/activate

# Generate embeddings (requires Ollama with nomic-embed-text)
python cli.py node-embeddings --input-dir ../dev/chatgpt-4-11-2025_json_no_embeddings \
    --method role-aggregate --embedding-method llm --output-dir ../dev/chatgpt-json-llm

# Build similarity network
python cli.py edges-gpu --input-dir ./embeddings_json --output-file edges.json
python cli.py cut-off --input-file edges.json --output-file filtered.json --cutoff 0.9

# Export for Gephi
python cli.py export --nodes-dir ./embeddings_json --edges-file filtered.json --format gexf -o graph.gexf
```

## Citation

If you use this work, please cite the journal paper:

```bibtex
@article{towell2026temporal,
  author  = {Towell, Alexander and Matta, John},
  title   = {Temporal Evolution of Cognitive Knowledge Networks in AI-Assisted Conversations},
  journal = {PLOS Complex Systems},
  year    = {2026},
  note    = {Submitted}
}
```

See [CITATION.cff](CITATION.cff) for machine-readable citation metadata. The analysis pipeline is available separately at [chatgpt-complex-net](https://github.com/queelius/chatgpt-complex-net) (DOI: [10.5281/zenodo.15314235](https://doi.org/10.5281/zenodo.15314235)).

## License

[MIT](LICENSE)
