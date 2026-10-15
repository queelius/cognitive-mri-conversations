# Focused primary-literature review: networks and memory

Search/check date: **2026-10-15**. This review targets novelty threats to temporal
nulls, projected hierarchies, memory architecture/forgetting, and semantic
percolation. It is not exhaustive. Generic public-conversation/HCI positioning is
covered by another audit. Claims about priority are limited accordingly. Primary
abstracts, publication records and the indicated method passages were checked;
no reported score is an independent reproduction.

## Temporal and semantic-network inference

| Primary work and verified status | What is already established | Consequence for CMRI |
|---|---|---|
| Gauvin et al., *Randomized reference models for temporal networks*, **SIAM Review 64(4), 763–830 (2022)**. [Accepted manuscript/record](https://arxiv.org/abs/1806.04032), [DOI](https://doi.org/10.1137/19M1242252). | Taxonomy of constrained temporal randomizations, what they preserve/destroy, and how composition changes a null. | A timestamp shuffle or “separate topology from sequence” premise is not a new general method. CMRI could contribute calibration for growing thresholded item graphs and a concrete empirical result. “Inside the band means geometry alone” exceeds what the conditional comparison establishes. |
| Newman, Strogatz & Watts, *Random graphs with arbitrary degree distributions and their applications*, **Physical Review E 64, 026118 (2001)**. [Primary manuscript/record](https://arxiv.org/abs/cond-mat/0007235), [DOI](https://doi.org/10.1103/PhysRevE.64.026118). | Random graph theory includes bipartite degree distributions and their projections, component transitions and distances. | A random unipartite graph is not the only or automatically appropriate comparison for an episode–concept projection. Construction and degree constraints can generate apparent topology without the proposed cognitive mechanism. |
| Fujiki & Mizutaka, *Properties of the connected components in projections of random bipartite networks: effects of clique size fluctuations*, **Applied Network Science 9, 60 (2024)**. [Primary article](https://link.springer.com/article/10.1007/s41109-024-00664-w). | Studies component structure and correlations created by projections of random bipartite networks; finite clustering persists in such projections. | This is a particularly direct modern threat to interpreting a projected graph's ER-relative clustering as evidence of memory-like organization. The local degree-preserving diagnostic asks precisely whether the simple ER claim is specific. |
| Gerlach & Altmann, *Scaling laws and fluctuations in the statistics of word frequencies*, **New Journal of Physics 16, 113010 (2014)**. [Primary manuscript/record](https://arxiv.org/abs/1406.4441), [DOI](https://doi.org/10.1088/1367-2630/16/11/113010). | Explains vocabulary-growth scaling and fluctuations using heavy-tailed word frequencies and document/topic heterogeneity. | Heaps-like growth need not identify consolidation. Topic heterogeneity and finite sampling must be distinguished from psychological mechanisms. The exact finite-incidence expectation in this audit is an elementary conditional calculation, not a claimed new Heaps theory. |
| Gerlach, Peixoto & Altmann, *A network approach to topic models*, **Science Advances 4, eaaq1360 (2018)**. [Primary manuscript/record](https://arxiv.org/abs/1708.01677), [DOI](https://doi.org/10.1126/sciadv.aaq1360). | Represents corpora as document–word bipartite networks, hierarchically clusters both sides, and infers topic counts with a nonparametric SBM. | A bipartite document/concept hierarchy is not novel in itself. CMRI's specific LLM-extracted noun phrases differ from words, but fixed 500/50/8 cuts are a weaker hierarchy-selection claim than an automatic structural baseline. Compare fit/prediction and stability, not only interpretability. |
| Steyvers & Tenenbaum, *The Large-Scale Structure of Semantic Networks: Statistical Analyses and a Model of Semantic Growth*, **Cognitive Science 29 (2005)**. [Author-hosted paper](https://web.mit.edu/cocosci/archive/Papers/03nSteyvers.pdf), [publisher](https://onlinelibrary.wiley.com/doi/abs/10.1207/s15516709cog2901_3). | Analyzes semantic networks from different human linguistic/association resources and models semantic growth. | The qualitative analogy has prior motivation. Matching a derived small-world ratio does not establish a common mechanism, equivalent measurement, or an individual's memory state. The datasets and graph-construction rules differ. |

The local negative hierarchy finding is stronger than an abstract warning about
nulls: after preserving every episode's concept count and each concept's frequency,
the observed projection has **lower** mean clustering and **longer** paths than all
16 diagnostic null draws. This result is conditional on that null and the saved
extraction, and its finite-chain/mixing limits are explicit in the
[audit](../audits/networks-memory.md). It does not establish that all semantic
structure disappears.

## Null implementation and percolation

* Strona et al., *A fast and unbiased procedure to randomize ecological binary
  matrices with fixed row and column totals*, **Nature Communications 5, 4114
  (2014)**, [primary DOI](https://doi.org/10.1038/ncomms5114). The publisher page
  was inaccessible in this session. This is the original Curveball reference;
  the implementation comparison is grounded additionally in the primary work below.
* Carstens & Kleer, *Comparing the Switch and Curveball Markov Chains for Sampling
  Binary Matrices with Fixed Marginals*, [2017 primary manuscript](https://arxiv.org/abs/1709.07290).
  The checked arXiv record supplies no journal reference, so it is treated here as
  a manuscript. It formalizes binomial trades and convergence comparisons. Uniform
  stationary sampling is not a guarantee that an arbitrary short chain is mixed.
  The original CMRI script uses a forced subset-swap move, not the exact uniform
  allocation used in the new diagnostic; preserving margins alone does not validate
  mixing. No claim is made that the original chain lacks a uniform stationary law.
* Rastelli et al., *Balancing novelty and appropriateness leads to creative
  associations in children*, **PNAS Nexus 1(5), pgac273 (2022)**,
  [primary article](https://academic.oup.com/pnasnexus/article/1/5/pgac273/6865385).
  Its “Networks Percolation” method thresholds a weighted semantic network,
  tracks the largest component, and uses an area-under-curve summary. This already
  occupies a generic semantic-threshold resilience analysis. CMRI would need more
  than a new corpus and a sharper fragmentation plot. This paper does not establish
  a universality class for the CMRI construction.
* Newman & Ziff, *Efficient Monte Carlo algorithm and high-precision results for
  percolation*, **Physical Review Letters 85, 4104–4107 (2000)**,
  [primary manuscript/record](https://arxiv.org/abs/cond-mat/0005264).
  Its controlled site/bond occupation experiments contrast with a single observed
  cosine-threshold sweep. Occupation ensemble, system size and the order parameter
  must be defined before borrowing critical exponents.
* Aldous, *Brownian excursions, critical random graphs and the multiplicative
  coalescent*, **Annals of Probability 25(2), 812–854 (1997)**,
  [author publication list](https://www.stat.berkeley.edu/~aldous/Papers/bibliog.html),
  [institutional record](https://escholarship.org/uc/item/5s77j58b).
  The primary indexed abstract describes the critical random-graph window
  p=n⁻¹+t n⁻⁴ᐟ³ and component size n²ᐟ³. The institutional full page required a
  browser challenge. The narrow implication is that random-graph finite-size
  scaling requires its own specification; simply assigning a lattice-style ν=.5
  to the CMRI threshold family is not a justified test.

## Memory architecture and selective forgetting

| Primary work and current status | Closest overlap | Remaining distinction, if any |
|---|---|---|
| Li et al., *TiMem: Temporal-Hierarchical Memory Consolidation for Long-Horizon Conversational Agents*, **Findings of ACL 2026**, pp. 21700–21720. [Published paper](https://aclanthology.org/2026.findings-acl.1091/), [arXiv v2](https://arxiv.org/abs/2601.02845). | Temporal hierarchy, semantic consolidation from conversations to abstract representations, query-dependent recall and benchmark evaluation. | A generic temporal/hierarchical conversational-memory system is already covered. CMRI is descriptive archive analysis, not an evaluated memory agent; changing the framing to architecture requires actual storage/retrieval and utility tests. No numerical comparison is warranted between these different tasks. |
| Yue et al., *HyperMem: Hypergraph Memory for Long-Term Conversations*, **ACL 2026 long papers**, pp. 35237–35254. [Published paper](https://aclanthology.org/2026.acl-long.1627/), [method text](https://arxiv.org/html/2604.08256v1). | Topic/episode/fact hierarchy, hyperedges for associations across dispersed episodes, provenance-linked facts, and coarse-to-fine retrieval. | “Many-to-many associations invisible to partitions” is not sufficient architectural novelty. It remains possible to study a distinct measurement failure in extracted concept graphs, but not to claim that hierarchy/hyperedges alone solve cross-episode memory. Methods §§3.1–3.3 were inspected; published headline scores were not reproduced. |
| Rusu, Khanzadeh & Alalfi, *Selective Forgetting: A Graph-Based Memory Framework for Long-Term LLM Agents*, **29 Aug 2026 preprint**. [Primary paper/record](https://arxiv.org/abs/2608.28978). | Typed graph memory, recency/frequency/degree/age pruning; a negative graph-versus-vector result and separate modest storage-pruning experiment. | Generic graph-forgetting novelty is preempted. Its graph-storage experiment differs from the CMRI pilot's severe extractive token caps; the local pilot is not a numerical replication. A bridge-specific mechanism would have to survive nuisance controls and genuine cross-topic support annotation. |
| Yu, Lin & Wu, *WhenLoss: Diagnosing Write and Retrieval Bottlenecks in Long-Context Memory Systems*, **23 May 2026 preprint**. [Primary paper](https://arxiv.org/abs/2605.24579). | Fixed-reader conditions distinguish write loss and retrieval loss; proposed predictive compression acts at write time. | A general “needed evidence was discarded” finding or write-versus-read distinction is not new. The local no-reader preservation screen is a reasonable early gate but currently fails to establish a bridge-specific benefit. |
| *MemORAI: Memory Organization and Retrieval via Adaptive Graph Intelligence for LLM Conversational Agents*, **Findings of ACL 2026**. [Primary publication](https://aclanthology.org/2026.findings-acl.1408/). | Provenance-preserving graph organization and query-adaptive PageRank; the saved prior review checked degree-bias discussion. | Static PageRank in the local pilot is a simple screen, not an implementation of this query-conditioned method. Generic graph provenance/importance weighting cannot be the central novelty claim. |
| *StreamMeCo: Long-Term Agent Memory Compression for Efficient Streaming Video Understanding*, **Findings of ACL 2026**. [Primary publication](https://aclanthology.org/2026.findings-acl.647/). | Connectivity-guided compression in streaming video memory. | Graph-based eviction is not new across modalities. The distinct conversational evidence unit and equal token caps require their own claim and evaluation, not cross-paper score comparison. |

The last four comparisons extend/recheck the completed
[selective-forgetting literature record](../../../../papers/selective-forgetting/LITERATURE.md).
Its full experiment was not repeated. On current evidence, the appropriate next
step is measurement validation or stopping, not adding a more expensive reader.

## Search record and limits

All queries below were run on 2026-10-15. Exact-title follow-ups and direct opens
were used to verify publication records; secondary indexes were discovery aids
only, not evidence for technical claims.

| Query group | Queries |
|---|---|
| Temporal nulls | `Gauvin randomized reference models temporal networks SIAM Review 2022`; `"Randomized reference models for temporal networks" Gauvin` |
| Projection and vocabulary | `semantic network bipartite projection small world clustering artifact random intersection graph Newman Strogatz Watts 2001`; `Heaps law vocabulary growth finite size random sampling order Gerlach Altmann 2013`; `"Random graphs with arbitrary degree distributions" Newman Strogatz Watts bipartite`; `"Scaling laws and fluctuations in the statistics of word frequencies"` |
| Null algorithms | `"Curveball" algorithm fixed row column sums Strona 2014 Nature Communications`; `"Clustering as a measure of the bipartite nature of networks"` |
| Hierarchy and memory | `site.arxiv.org semantic networks latent hierarchy topic model network Gerlach Peixoto Altmann 2018`; `site.arxiv.org "hierarchical memory" "conversations" 2025 2026`; `"TiMem" site:aclanthology.org`; `"HyperMem" site:aclanthology.org`; `"WhenLoss" "memory" "2026"`; `"Selective Forgetting" "Rusu"` |
| Percolation and construction | `site.arxiv.org "percolation" "semantic" "network" threshold`; `"Network construction from time series" "spurious" Bialonski 2010`; `"Network construction and fundamental results" percolation correlation networks threshold`; `"percolation" "semantic networks" "2022" Kenett`; `"Brownian excursions" "critical random graphs" Aldous 1997 n 2/3`; `"Efficient Monte Carlo algorithm" "percolation" Newman Ziff 2000 finite size` |
| Status and original comparison | `"Cognitive MRI of AI Conversations" Springer`; `"The large-scale structure of semantic networks" Steyvers Tenenbaum 2005` |

Direct primary opens additionally checked the three ACL memory publications,
the hierarchy author's accepted-status page, arXiv publication metadata, and
the 2024 projection article. The 2026 memory literature is moving quickly;
absence of a still-closer paper in this focused review is not a priority claim.
The role-motif rejection rests on a mathematical representation degeneracy,
so no broad motif-literature survey was needed to reject that exact proposal.
