# Hierarchical Cognitive Memory Networks

**Extended paper for PLOS Complex Systems Special Issue**

This extends the Complex Networks 2025 conference paper "Cognitive MRI of AI Conversations" with a novel hierarchical, multi-layer network approach inspired by cognitive memory systems.

## Core Contribution

Transform flat conversation networks into **multi-layer cognitive memory networks**:

- **Episodic Layer (E)**: Conversations — specific instances (existing)
- **Concept Layer (C)**: Abstract principles extracted via LLM — semantic memory (new)
- **Cross-layer Links**: Instantiation relationships connecting concepts to episodes

### The Key Insight

Two conversations with LOW embedding similarity can be connected through shared abstract concepts — revealing **hidden cognitive structure** invisible to geometric similarity alone.

## Node Types

| Type | Cognitive Basis | Description |
|------|-----------------|-------------|
| **Episodic (E)** | Episodic memory | Individual conversations |
| **Concept (C)** | Semantic memory | General principles from episode clusters |

## Link Types

| Type | Notation | Connects | Basis |
|------|----------|----------|-------|
| **Similarity** | E—E | Episode ↔ Episode | Embedding cosine |
| **Instantiation** | C→E | Concept → Episode | LLM judgment |
| **Association** | C—C | Concept ↔ Concept | Structural (Jaccard) |

## Directory Structure

```
comp-net-2025-journal/
├── paper/
│   ├── paper.tex          # Main paper
│   ├── figures/           # Generated figures
│   └── refs.bib           # References
├── code/
│   ├── core_types.py              # Node and link type definitions
│   ├── concept_extraction.py      # LLM concept extraction
│   ├── instantiation_scoring.py   # C→E link creation
│   ├── association_network.py     # C—C Jaccard computation
│   ├── hidden_connections.py      # Find concept-connected pairs
│   ├── multilayer_analysis.py     # Community detection, metrics
│   └── visualization.py           # Network visualization & export
├── data/
│   ├── concepts.json              # Extracted concepts
│   ├── instantiation_links.json   # C→E links
│   └── association_edges.json     # C—C links
└── README.md
```

## Key Analyses

1. **Hidden Connections**: Episode pairs connected via shared concepts but not directly linked
2. **Concept Hubs**: Concepts that connect the most episodes
3. **Bridge Concepts**: Concepts spanning multiple episodic communities
4. **Concept Communities**: Do concepts cluster differently than episodes?

## Timeline

- **Deadline**: February 27, 2026
- **Requirement**: 30%+ new material, <30% similarity to conference paper

## Related

- Conference paper: `../comp-net-2025-camera-ready/paper/`
- Original code: `../code/`
- Plan: `~/.claude/plans/prancy-stirring-ritchie.md`
