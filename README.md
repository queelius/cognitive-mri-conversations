# Cognitive MRI of AI Conversations

Analyzing AI interactions through semantic embedding networks using complex network analysis.

**Authors:** Alex Towell and John Matta
**Institution:** Southern Illinois University Edwardsville
**Conference:** Complex Networks 2025

## Abstract

We present a "cognitive MRI" methodology that transforms sequential AI conversation logs into semantic networks, revealing latent thought structure. Using LLM-generated embeddings of 449 ChatGPT conversations, we construct a similarity network that exposes knowledge organization patterns invisible in linear logs.

Our analysis reveals heterogeneous network topology: theoretical domains (ML/AI) exhibit hub-and-spoke patterns while practical domains (programming) show hierarchical tree structures. We identify three bridge types connecting knowledge communities: evolutionary bridges (topic drift), integrative bridges (deliberate synthesis), and pure bridges (critical minimal-connection links).

**Keywords:** AI conversation, complex networks, semantic embedding, conversation analysis, knowledge exploration

## Repository Structure

```
.
├── code/                              # Python implementation
│   ├── cli.py                         # Main CLI interface
│   ├── networks.py                    # Network generation & analysis
│   ├── embedding/                     # LLM & TF-IDF embedding modules
│   └── graph/                         # Graph construction & export
├── comp-net-2025-camera-ready/        # Conference submission
│   ├── paper/                         # Camera-ready paper & LaTeX
│   ├── supplemental-docs/             # Supplementary materials
│   └── abstract-extended/             # Extended abstract
└── dev/                               # Research data & notes
```

## Key Findings

- **15 distinct knowledge communities** with 0.75 modularity score
- **Non-standard degree distribution** challenging scale-free assumptions
- **Three bridge conversation types** connecting communities:
  - Evolutionary bridges (organic topic drift)
  - Integrative bridges (deliberate concept synthesis)
  - Pure bridges (minimal but critical connections)

## Usage

```bash
cd code
pip install -r requirements.txt

# Generate embeddings
python cli.py node-embeddings --input-dir <conversations> --method role-aggregate

# Build similarity network
python cli.py edges-gpu --input-dir <embeddings> --output-file edges.json

# Export for visualization
python cli.py export --nodes-dir <embeddings> --edges-file edges.json --format gexf
```

## License

MIT License - see [LICENSE](LICENSE)

## Citation

See [CITATION.cff](CITATION.cff) for citation information.
