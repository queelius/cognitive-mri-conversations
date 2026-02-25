# CHANGELOG

## v0.3 -- Round 2 revisions (2026-02-25)

Addresses substantive feedback from co-author John Matta after full paper read-through.

### Figure changes

- **Dropped density panel from Figure 1** (was panel c). The density panel showed
  a monotonically decreasing curve, which is a mathematical inevitability for any
  growing network with densification exponent gamma < 2: density = 2e/n(n-1)
  decreases whenever edges grow slower than n^2, which is true for all empirically
  observed densification exponents. The panel was not wrong, but it was misleading --
  readers naturally interpreted the falling curve as the network becoming "less dense,"
  when the opposite is true (the network densifies super-linearly, as shown in the
  dedicated densification law figure). Removing it eliminates confusion without losing
  any information. Figure 1 is now a 3-panel layout: (a) Node Accumulation,
  (b) Edge Accumulation, (c) Connectivity Ratio.
- **Dropped model era comparison figure** (was Fig 7). The table alone conveys the
  same information more compactly. Section compressed to 1 paragraph + 1 table
  and reframed as exploratory.
- **Added network visualization figure** (Fig 2). Static snapshot of the knowledge
  network showing Louvain communities labeled by topic, reproduced from the
  conference paper. Makes the paper self-contained for readers without access to
  the Springer proceedings.

### Text additions

- **Modularity caveat (Section 4.2)**: Added explanation that the modularity step
  change is partly mechanical -- when isolated clusters merge into the giant
  component, modularity captures inter-group separation, inflating the apparent
  jump magnitude.
- **Bridge taxonomy (Section 4.6)**: Defined evolutionary, integrative, and pure
  bridge types inline so the paper stands alone without requiring the conference
  paper.
- **Cognitive theories (Related Work)**: New subsection "Distributed cognition and
  the extended mind" covering Hutchins (1995) and Clark & Chalmers (1998), explaining
  how these frameworks motivate treating the conversation archive as an external
  cognitive artifact.
- **Missing citations (Discussion 5.1)**: Added Bartlett (1932) for schema theory,
  March (1991) for exploration/exploitation, Hills et al. (2015) for cognitive
  search. Sentences now properly cite the literature they reference.
- **Multi-user motivation (Conclusion)**: Expanded with 3 sentences explaining why
  multi-user studies matter -- different densification exponents for specialists
  vs. generalists, attachment exponent variation with cognitive style, and which
  parameters might be universal vs. individual.
- **Model era limitations**: Softened to lead with what IS meaningful (GPT-3.5/GPT-4o
  consistency) rather than what's weak.

### Bibliography

- Added `bartlett1932remembering` (Remembering)
- Added `march1991exploration` (Exploration and Exploitation in Organizational Learning)
- Added `hills2015exploration` (Exploration versus Exploitation in Space, Mind, and Society)

---

## v0.2 -- Round 1 revisions (2025-06-10)

Formatting and style corrections from co-author review.

- Replaced bold bullet-point lists with flowing prose paragraphs (PLOS style)
- Replaced em-dashes (--) with commas or semicolons throughout
- Removed stray references to "conference paper" where "previous work" is appropriate
- Standardized citation formatting
- Minor prose tightening throughout
