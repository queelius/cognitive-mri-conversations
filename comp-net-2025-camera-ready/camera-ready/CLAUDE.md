# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Camera-ready submission directory for "Cognitive MRI of AI Conversations" paper accepted to Complex Networks 2025 conference. Contains the final full paper, 4-page abstract, conference presentation slides, and supplementary materials for conference publication.

## Directory Structure

```
camera-ready/
├── paper.tex                    # Main camera-ready paper (full version)
├── paper.pdf                    # Built PDF output
├── slide-draft.tex              # Conference presentation (12 minutes, 11+1 slides)
├── slide-draft.pdf              # Compiled presentation with speaker notes
├── slide-draft-practice.md      # Practice script with timing and gestures
├── archive/                     # Historical slide versions (slide*.tex, slides*.tex)
├── images/                      # Shared image directory (symlinked from parent)
├── svproc.cls                   # Springer proceedings class file
├── spmpsci.bst                  # Springer bibliography style
└── email_to_matta*.txt          # Communication files

../camera-ready-abstract/
├── abstract.tex                 # 4-page conference abstract
├── abstract.pdf                 # Built abstract PDF
├── images/                      # Subset of images for abstract
├── llncs.cls                    # LLNCS class file for abstract
├── refs.bib                     # Bibliography (51 references)
└── splncs*.bst                  # LLNCS bibliography styles

../images/                       # Main image repository
├── *.pdf, *.pdf_tex             # Inkscape-exported figures with LaTeX overlays
├── *.png                        # Raster figures
├── export-svg.sh                # SVG to PDF conversion script
├── ablation_*.{pdf,png}         # Ablation study visualizations
├── threshold_*.{pdf,png}        # Network threshold analysis plots
└── community_*.{pdf,png}        # Community detection visualizations
```

## Common Commands

### Building the Full Paper

```bash
# From camera-ready/ directory
cd camera-ready
pdflatex paper.tex
bibtex paper              # Only needed if references change
pdflatex paper.tex        # Second pass for cross-references
pdflatex paper.tex        # Final pass for TOC/refs

# Quick rebuild (no bibliography changes)
pdflatex paper.tex
```

### Building the 4-Page Abstract

```bash
# From camera-ready-abstract/ directory
cd camera-ready-abstract
pdflatex abstract.tex
bibtex abstract           # Only if references change
pdflatex abstract.tex
pdflatex abstract.tex
```

### Building the Conference Presentation

```bash
# Build with speaker notes (for practice)
pdflatex slide-draft.tex

# Build without speaker notes (for delivery)
# Comment out in slide-draft.tex:
#   \usepackage{pgfpages}
#   \setbeameroption{show notes on second screen=right}
# Then: pdflatex slide-draft.tex
```

**Presentation Structure:**
- **Duration**: 12 minutes (11 main slides + 2 backup slides)
- **Format**: Beamer 16:9, Boadilla theme with whale color scheme
- **Speaker notes**: Enabled via pgfpages, displayed on right side
- **Practice script**: `slide-draft-practice.md` contains timing breakdowns and gesture cues
- **Archive**: Old iterations stored in `archive/` directory

### Converting SVG Figures to PDF

The paper uses Inkscape-exported PDFs with LaTeX text overlays for vector graphics:

```bash
cd images
./export-svg.sh

# Manual conversion for single file
inkscape figure.svg --export-type=pdf --export-latex --export-dpi=300 -o figure.pdf
```

This generates:
- `figure.pdf` - Graphics without text
- `figure.pdf_tex` - LaTeX overlay with text positioning

Include in LaTeX with:
```latex
\begin{figure}
  \centering
  \input{images/figure.pdf_tex}
  \caption{...}
\end{figure}
```

## Document Classes and Formats

### Full Paper (`paper.tex`)
- **Class**: `svproc.cls` (Springer Proceedings)
- **Style**: `spmpsci.bst` (Springer bibliography)
- **Images**: `\graphicspath{{./images/}}` points to parent `images/` directory
- **Key packages**: tikz, algorithm2e, subcaption, booktabs
- **Spacing**: Custom tight spacing for conference format
  ```latex
  \setlength{\textfloatsep}{5pt plus 1pt minus 2pt}
  \setlength{\intextsep}{5pt plus 1pt minus 2pt}
  ```

### 4-Page Abstract (`abstract.tex`)
- **Class**: `llncs.cls` (LLNCS - Lecture Notes in Computer Science)
- **Style**: Multiple options (`splncs.bst`, `splncs03.bst`, `splncs_srt.bst`)
- **Bibliography**: `refs.bib` (51 references including key citations)
- **Page limit**: 4 pages strict for conference abstract submission

### Conference Presentation (`slide-draft.tex`)
- **Class**: Beamer with 16:9 aspect ratio
- **Theme**: Boadilla with whale color scheme (modern, block-compatible)
- **Key packages**: tikz (diagrams), tcolorbox (example boxes), fontawesome5 (icons), booktabs (tables)
- **Speaker notes**: Implemented with pgfpages showing notes on second screen
- **Slide count**: 11 main slides + 2 hidden backup slides (using `[noframenumbering]`)
- **Footline**: Custom footline that respects `[noframenumbering]` for backup slides
- **Visual elements**:
  - TikZ overlay callouts on Slide 7 (network visualization)
  - Color-coded bridge types on Slide 9 (blue/purple/red)
  - Metrics comparison table on Slide 8
  - Concrete query example box on Slide 10
- **Practice notes**: Detailed timing and gesture instructions in `slide-draft-practice.md`

## Key Content Components

### Paper Structure
1. **Introduction**: Cognitive MRI metaphor, distributed cognition framework
2. **Related Work**: Complex networks, semantic embeddings, conversation analysis
3. **Methods**:
   - Data: 449 conversations from 1,908 total (filtered at θ=0.9)
   - Embeddings: `nomic-embed-text` with 2:1 user-weighting
   - Network construction: Cosine similarity edges
4. **Results**:
   - 15 communities (modularity 0.750)
   - Heterogeneous topology (hub-and-spoke vs tree-like)
   - Three bridge types (evolutionary, integrative, pure)
5. **Ablation Study**: 63-configuration parameter sweep validating design choices
6. **Discussion**: Implications for distributed cognition, limitations

### Presentation Flow (12 minutes)
1. **Setup (3 min)**: Title → Scale/Stakes → Iceberg metaphor → Log-to-MRI transformation
2. **Methods (3 min)**: User-weighted embedding → 2D ablation study (63 configurations)
3. **Results (4.5 min)**: Network reveal → Heterogeneity → Bridge taxonomy
4. **Vision & Conclusion (1.5 min)**: Personal knowledge cartography → Proof of concept

### Image Categories
- **Network visualizations**: `0.9-giant-*.pdf` (main network at θ=0.9)
- **Ablation studies**: `ablation_*.{pdf,png}` (parameter sensitivity analysis)
- **Community analysis**: `cluster-vis-*.png`, `community_*.{pdf,png}`
- **Degree distributions**: `degree_distribution*.{pdf,png}`, `powerlaw_fit.png`
- **Threshold analysis**: `threshold_*.{pdf,png}` (network evolution across thresholds)
- **Bridge analysis**: `bridge*.{pdf,png}`, `betweenness*.{pdf,png}`

## Bibliography Management

The abstract uses `refs.bib` (51 entries) containing:
- **Foundational**: Hutchins (distributed cognition), Wegner (transactive memory)
- **Network theory**: Granovetter (weak ties), Burt (structural holes), Blondel (Louvain)
- **Embeddings**: Mikolov (Word2Vec), Nussbaum (nomic-embed)
- **Complex networks**: Barabási, Newman, Watts-Strogatz
- **NLP/Conversation**: Various recent papers on dialogue analysis

When adding references:
1. Add to `camera-ready-abstract/refs.bib`
2. Use consistent entry types (@article, @book, @inproceedings)
3. Include DOIs/URLs where available
4. Rebuild with `bibtex abstract` then two `pdflatex` passes

## Special LaTeX Considerations

### Figure Formats
- **Vector graphics**: Use `\input{images/figure.pdf_tex}` for Inkscape exports
- **Raster graphics**: Use `\includegraphics{images/figure.png}` for screenshots/plots
- **Dual format**: Some figures exist as both `.pdf` and `.png` for compatibility

### Space Optimization
Both documents use aggressive space optimization for page limits:
```latex
\setlength{\textfloatsep}{5pt plus 1pt minus 2pt}
\setlength{\intextsep}{5pt plus 1pt minus 2pt}
\setlength{\abovecaptionskip}{3pt}
\setlength{\belowcaptionskip}{3pt}
```

### Algorithm Formatting
Uses `algorithm2e` package for pseudocode:
```latex
\usepackage{algorithm}
\usepackage{algorithmicx}
\usepackage{algpseudocode}
```

## Submission Materials

The repository contains:
- `camera-ready.zip` - Full paper submission package
- `camera-ready-abstract.zip` - 4-page abstract submission
- `supplemental-docs/` - Copyright forms and waivers

## Common Issues and Solutions

### Bibliography Not Updating
```bash
# Full rebuild sequence
rm paper.aux paper.bbl paper.blg
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

### Missing Images
- Verify `\graphicspath{{./images/}}` matches actual directory
- For camera-ready paper, images are in `../images/`
- For abstract, images are in local `images/` subdirectory

### PDF/PDF_tex Inclusion Issues
Ensure both files exist:
```bash
ls images/figure.pdf images/figure.pdf_tex
```
If missing, regenerate with `export-svg.sh`

### Class File Errors
- Full paper requires `svproc.cls` in same directory
- Abstract requires `llncs.cls` in same directory
- Both are provided in respective directories

## Camera-Ready Workflow

1. **Final edits** to `paper.tex` or `abstract.tex`
2. **Rebuild** with full LaTeX/BibTeX cycle
3. **Verify** page count (abstract must be ≤4 pages)
4. **Check** all figures render correctly
5. **Update** any modified images in `images/` directory
6. **Rebuild** one final time to ensure consistency
7. **Create submission package** (already done as `.zip` files)

## Presentation Development Workflow

### Iterative Refinement Process
Presentations go through multiple iterations stored in `archive/`:
1. Initial draft with content and structure
2. Add speaker notes using `\note{}` commands in each frame
3. Refine timing to meet 12-minute constraint
4. Add visual enhancements (callouts, color coding, examples)
5. Update practice notes with gesture cues and timing

### Working with Speaker Notes
Speaker notes are displayed on a second screen during practice:
```latex
\begin{frame}{Slide Title}
    % Slide content here

    \note{
        \textbf{Slide X: Title (Xmin Xs)}
        \begin{itemize}
            \item \textbf{Point 1 (Xs):} [GESTURE CUE] "Exact words to say"
            \item \textbf{Point 2 (Xs):} [GESTURE CUE] "More exact words"
        \end{itemize}
    }
\end{frame}
```

### Backup Slides
Hide slides from numbering while keeping them accessible during Q&A:
```latex
\begin{frame}[noframenumbering]{Backup: Technical Details}
    % Content only shown if manually advanced past conclusion
\end{frame}
```

### Visual Design Patterns
- **TikZ overlays**: Use `remember picture, overlay` for annotations on top of figures
- **Color coding**: Establish consistent color meanings (blue=theoretical, green=practical, etc.)
- **Metrics tables**: Use `booktabs` for professional table formatting
- **Example boxes**: Use `tcolorbox` with light background colors for concrete examples

### Slide Spacing Issues
When content is being cut off at the bottom of slides:
1. Reduce `\vspace{}` commands within blocks (0.1cm → 0.02cm)
2. Use smaller font sizes for less critical content (`\scriptsize`, `\tiny`)
3. Scale TikZ graphics down (e.g., `scale=0.85`)
4. Remove spacing before footer elements (`\vspace{0cm}`)

### Custom Footline for Backup Slides
The presentation uses a custom footline that respects `[noframenumbering]`:
```latex
\setbeamertemplate{footline}{
  \leavevmode%
  \hbox{%
  \begin{beamercolorbox}[wd=\paperwidth,ht=2.25ex,dp=1ex,right]{date in head/foot}%
    \usebeamerfont{date in head/foot}\insertframenumber{} / \inserttotalframenumber\hspace*{2ex}
  \end{beamercolorbox}}%
  \vskip0pt%
}
```
This ensures backup slides don't increment the slide count (shows "11 / 11" instead of "13 / 13").
