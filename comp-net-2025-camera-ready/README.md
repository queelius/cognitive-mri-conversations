# Complex Networks 2025 Camera-Ready Submission

**Paper:** Cognitive MRI of AI Conversations: Analyzing AI Interactions through Semantic Embedding Networks

**Authors:** Alex Towell and John Matta, Southern Illinois University Edwardsville

**Conference:** Complex Networks 2025

## Abstract

Through a single-user case study of 449 ChatGPT conversations, we introduce a cognitive MRI applying network analysis to reveal thought topology hidden in linear conversation logs. We construct semantic similarity networks with user-weighted embeddings to identify knowledge communities and bridge conversations that enable cross-domain flow. Our analysis reveals heterogeneous topology: theoretical domains exhibit hub-and-spoke structures while practical domains show tree-like hierarchies. We identify three distinct bridge types that facilitate knowledge integration across communities.

## Directory Structure

```
comp-net-2025-camera-ready/
├── paper/                  # Camera-ready paper
│   ├── paper.tex           # LaTeX source
│   ├── paper.pdf           # Compiled paper
│   ├── images/             # Paper figures
│   └── svproc.cls          # Springer proceedings class
├── slides/                 # Conference presentation
│   ├── 41_Towell_Alex.pdf  # Official conference slides
│   ├── slide-pretty.tex    # Beamer source
│   └── images -> paper/images
├── abstract-extended/      # Extended abstract (LNCS format)
│   ├── abstract.tex        # LaTeX source
│   └── abstract.pdf        # Compiled abstract
├── supplemental-docs/      # Administrative documents
│   └── consent_to_publish_*.pdf
└── artifacts/              # Generated visualizations
    ├── *_clean.pdf         # Publication-ready figures
    ├── ablation_study*.pdf # Ablation study results
    └── *.png, *.xcf        # Working image files
```

## Building

### Paper
```bash
cd paper
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

### Slides
```bash
cd slides
pdflatex slide-pretty.tex
pdflatex slide-pretty.tex
```

### Extended Abstract
```bash
cd abstract-extended
pdflatex abstract.tex
bibtex abstract
pdflatex abstract.tex
pdflatex abstract.tex
```
