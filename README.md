# Cognitive MRI of AI Conversations: Research Compendium

> **Frozen.** This repository holds the materials of the Complex Networks 2025 paper, as released in v2.0.0
> (February 2026) and archived on Zenodo. On 14 November 2026 the conversation-level data files (titles,
> identifiers and URLs of the conversations) were removed from its history for privacy, and the draft of a journal
> extension, which was not published, was removed from the current files; that draft remains in the history and in
> the Zenodo v2.0.0 archive.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18778377.svg)](https://doi.org/10.5281/zenodo.18778377)

Applying complex network analysis to ChatGPT conversation archives to reveal knowledge organization and community
structure.

**Authors:**
[Alexander Towell](https://orcid.org/0000-0001-6443-9897) and
[John Matta](https://orcid.org/0000-0002-7666-1409)
— Southern Illinois University Edwardsville

## Overview

This repository transforms sequential AI conversation logs into semantic similarity networks. We analyze 1,908
ChatGPT conversations spanning December 2022 to April 2025, constructing networks that expose knowledge communities
and bridge conversations that are invisible in linear logs.

## Paper

**Cognitive MRI of AI Conversations: Analyzing AI Interactions Through Semantic Embedding Networks.**
Alexander Towell and John Matta. In *Complex Networks & Their Applications XIV* (Complex Networks 2025), Studies in
Computational Intelligence, Springer, 2026, pp. 315–327. https://doi.org/10.1007/978-3-032-16723-1_27

## Repository Structure

```
├── comp-net-2025-camera-ready/      # The published conference paper (Springer)
│   ├── paper/                       # Camera-ready paper
│   └── slides/                      # Conference presentation (Beamer)
├── data/
│   └── ablation/                    # 63-configuration parameter study behind the paper's parameter choices
└── code/                            # Embedding + network construction pipeline
    ├── cli.py                       # Main CLI (embeddings, edges, export)
    ├── networks.py                  # Network statistics & metrics
    ├── embedding/                   # LLM & TF-IDF embedding models
    ├── graph/                       # Edge generation, GPU acceleration
    └── run_ablation_study.py        # 63-config ablation study
```

## Data

`data/ablation/` holds the aggregate results of the parameter study (see [`data/README.md`](data/README.md)). The
per-conversation embeddings and the edge list are not published, since they identify individual conversations; they
can be regenerated from a conversation export with the pipeline below.

## Building the Pipeline from Scratch

To regenerate embeddings and networks from raw conversations (requires [Ollama](https://ollama.ai) with
`nomic-embed-text`):

```bash
pip install -r code/requirements.txt

cd code
python cli.py node-embeddings --input-dir <conversations-dir> \
    --method role-aggregate --embedding-method llm --output-dir <output-dir>
python cli.py edges-gpu --input-dir <output-dir> --output-file edges.json
python cli.py cut-off --input-file edges.json --output-file filtered.json --cutoff 0.9
python cli.py export --nodes-dir <output-dir> --edges-file filtered.json --format gexf -o graph.gexf
```

The pipeline code is also available as a standalone package at
[chatgpt-complex-net](https://github.com/queelius/chatgpt-complex-net) (DOI:
[10.5281/zenodo.15314235](https://doi.org/10.5281/zenodo.15314235)).

## Citation

If you use this work, please cite the paper:

```bibtex
@incollection{towell2026cognitive,
  author    = {Towell, Alexander and Matta, John},
  title     = {Cognitive {MRI} of {AI} Conversations: Analyzing {AI} Interactions Through Semantic Embedding Networks},
  booktitle = {Complex Networks \& Their Applications {XIV}},
  series    = {Studies in Computational Intelligence},
  publisher = {Springer Nature Switzerland},
  year      = {2026},
  pages     = {315--327},
  doi       = {10.1007/978-3-032-16723-1_27}
}
```

See [CITATION.cff](CITATION.cff) for machine-readable citation metadata.

## License

[MIT](LICENSE)
