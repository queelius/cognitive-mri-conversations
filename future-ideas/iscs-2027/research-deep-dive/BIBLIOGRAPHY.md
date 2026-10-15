# Working bibliography and evidence map

Search cutoff: **2026-10-15**. This is a linked working bibliography, not an exhaustive systematic review. Short method names are used for some entries; primary records supply full author lists and formal citations. Publication status is stated only to the extent verified. Detailed novelty comparisons and reading depth are in [semantic/time-series](literature/semantic-timeseries.md), [cross-sectional/HCI](literature/cross-sectional.md), and [networks/memory](literature/networks-memory.md). [Search log and limitations](literature/SEARCH-LOG.md).

## Semantic representation, dynamics and time series

| Key | Primary reference | Status / relevance |
|---|---|---|
| D01 | Nour, McNamee, Fradkin & Dolan. [Charting trajectories of human thought using large language models](https://arxiv.org/abs/2509.14455). 2025. | Preprint, VECTOR; external behavioral validation comparator. |
| D02 | Toro-Hernández, Vieira Filho & Cabral-Carvalho. [Characterizing Human Semantic Navigation in Concept Production as Trajectories in Embedding Space](https://arxiv.org/abs/2602.05971). 2026. | ICLR 2026. Methods/code inspected; v2 methods also checked. [Pinned implementation](https://github.com/jesuinovieira/semtraj-iclr2026/blob/7506e25c03036909ab1ad381c678c7b45d19e50e/src/metrics.py). |
| D03 | Chang et al. [The Dynamics of Human and AI-Generated Language: How Semantics Fluctuates across Different Timescales](https://arxiv.org/abs/2606.11371). 2026. | Accepted/published author record, Computer Speech & Language, [DOI](https://doi.org/10.1016/j.csl.2026.102013); future issue date differs from availability. |
| D04 | Angus, Smith & Wiles. [Human Communication as Coupled Time Series: Quantifying Multi-Participant Recurrence](https://doi.org/10.1109/TASL.2012.2189566). 2012. | IEEE Transactions on Audio, Speech, and Language Processing; foundational interaction recurrence. |
| D05 | Angus & Wiles. [Social semantic networks: Measuring topic and participant relationships using conceptual recurrence plots](https://doi.org/10.1063/1.5024809). 2018. | Chaos; semantic networks and discourse validation. |
| D06 | Chowdhury. [Weakly Supervised Temporal Modeling of Latent Dynamics in Dyadic Conversations](https://aclanthology.org/2026.sigdial-1.31/). 2026. | SIGDIAL; abstract-level negative comparator for temporal prediction. |
| D07 | Arora, Liang & Ma. [A Simple but Tough-to-Beat Baseline for Sentence Embeddings](https://openreview.net/forum?id=SyK00v5xx). 2017. | ICLR; weighted pooling. Primary full-text access blocked on the checked route. |
| D08 | Ma et al. [Mega: Moving Average Equipped Gated Attention](https://arxiv.org/abs/2209.10655). | ICLR 2023; learned sequence/EMA baseline family. |
| D09 | Bamler & Mandt. [Dynamic Word Embeddings](https://proceedings.mlr.press/v70/bamler17a.html). 2017. | ICML; historical semantic change, distinct from within-dialogue state. |
| D10 | Fermanian. [Embedding and learning with signatures](https://arxiv.org/abs/1911.13211). | Computational Statistics & Data Analysis, 2021, [DOI](https://doi.org/10.1016/j.csda.2020.107148); possible path baseline. |
| D11 | [SuperDialseg](https://aclanthology.org/2023.emnlp-main.249/). 2023. | EMNLP; dialogue segmentation benchmark. [Official code](https://github.com/Coldog2333/SuperDialseg), data access incompletely checked. |
| D12 | Gao et al. [DialSTART / topic-aware dialogue segmentation](https://arxiv.org/abs/2305.02747). 2023. | Primary preprint inspected; competing segmentation method. |
| D13 | [Def-DTS](https://aclanthology.org/2025.findings-acl.1066/). 2025. | Findings of ACL; dialogue topic/intent segmentation. |
| D14 | Jia & Diaz-Rodriguez. [Embed-KCPD](https://arxiv.org/html/2601.18788v1). 2026. | Preprint; embedding-based kernel change points. |
| D15 | Pan et al. [SeCom](https://proceedings.iclr.cc/paper_files/paper/2025/hash/e56f394bbd4f0ec81393d767caa5a31b-Abstract-Conference.html). 2025. | ICLR; segment/compress conversational memory. [Official code](https://github.com/microsoft/SeCom). |

## Public conversations, purpose and human–AI interaction

| Key | Primary reference | Status / relevance |
|---|---|---|
| H01 | Hicke & Tomlinson. [Adopt ≠ Adapt](https://arxiv.org/html/2605.29018v1). 2026. | May preprint; repository identifies EMNLP 2026. Direct usage/activity/longitudinal comparator. |
| H02 | Potts & Sudhof. [A paradox of AI fluency](https://arxiv.org/html/2604.25905v1). 2026. | April primary manuscript; release of task/fluency/outcome annotations. |
| H03 | Karnam et al. [Bowling with ChatGPT](https://arxiv.org/html/2602.01114v1). 2026. | Web Conference; donated histories, purposes and steering. Code release does not provide unrestricted histories. |
| H04 | Shaikh et al. [Navigating Rifts in Human–LLM Grounding](https://aclanthology.org/2025.acl-long.1016/). 2025. | ACL; observable grounding and repair constructs. |
| H05 | Jin et al. [ThoughtTrace](https://arxiv.org/html/2605.20087v1). 2026. | May primary manuscript; participant-reported reasons/reactions and distinct model-assigned labels. |
| H06 | Zhang, Liu & August. [Re-Centering Humans in LLM Personalization](https://arxiv.org/html/2606.06614v1). 2026. | June preprint; human assessments of inferred attributes/personalization. |
| H07 | Ma et al. [How People Use ChatGPT in Australia](https://arxiv.org/html/2609.28990v1). 2026. | September manuscript with OzCHI 2026 proceedings metadata; geographical usage cross-section. |
| H08 | [PRISM Alignment dataset paper](https://arxiv.org/abs/2404.16019). 2024. | Primary paper/data; participant preferences and conversation ratings. |
| H09 | [Four-country donated-history study](https://arxiv.org/html/2609.38279v1). 2026. | September primary manuscript; restricted academic-access policy, not unrestricted raw data. |

## Networks, nulls, hierarchy and growth

| Key | Primary reference | Status / relevance |
|---|---|---|
| N01 | Gauvin et al. [Randomized reference models for temporal networks](https://doi.org/10.1137/19M1242252). 2022. | SIAM Review 64(4), 763–830; [accepted manuscript](https://arxiv.org/abs/1806.04032). |
| N02 | Newman, Strogatz & Watts. [Random graphs with arbitrary degree distributions and their applications](https://doi.org/10.1103/PhysRevE.64.026118). 2001. | Physical Review E 64, 026118; bipartite/projected random graphs. |
| N03 | Fujiki & Mizutaka. [Properties of the connected components in projections of random bipartite networks: effects of clique size fluctuations](https://doi.org/10.1007/s41109-024-00664-w). 2024. | Applied Network Science 9, 60. |
| N04 | Gerlach & Altmann. [Scaling laws and fluctuations in the statistics of word frequencies](https://doi.org/10.1088/1367-2630/16/11/113010). 2014. | New Journal of Physics 16, 113010; vocabulary-growth alternatives. |
| N05 | Gerlach, Peixoto & Altmann. [A network approach to topic models](https://doi.org/10.1126/sciadv.aaq1360). 2018. | Science Advances 4, eaaq1360; bipartite hierarchy comparator. |
| N06 | Steyvers & Tenenbaum. [The Large-Scale Structure of Semantic Networks: Statistical Analyses and a Model of Semantic Growth](https://web.mit.edu/cocosci/archive/Papers/03nSteyvers.pdf). 2005. | Cognitive Science 29; human semantic-network precedent. |
| N07 | Strona et al. [A fast and unbiased procedure to randomize ecological binary matrices with fixed row and column totals](https://doi.org/10.1038/ncomms5114). 2014. | Nature Communications 5, 4114; Curveball. Publisher access blocked in this run. |
| N08 | Carstens & Kleer. [Comparing the Switch and Curveball Markov Chains for Sampling Binary Matrices with Fixed Marginals](https://arxiv.org/abs/1709.07290). 2017. | Primary manuscript; convergence qualifications for diagnostic nulls. |
| N09 | Rastelli et al. [Balancing novelty and appropriateness leads to creative associations in children](https://academic.oup.com/pnasnexus/article/1/5/pgac273/6865385). 2022. | PNAS Nexus 1(5), pgac273; semantic-network threshold analysis. |
| N10 | Newman & Ziff. [Efficient Monte Carlo algorithm and high-precision results for percolation](https://arxiv.org/abs/cond-mat/0005264). 2000. | Physical Review Letters 85, 4104–4107. |
| N11 | Aldous. [Brownian excursions, critical random graphs and the multiplicative coalescent](https://escholarship.org/uc/item/5s77j58b). 1997. | Annals of Probability 25(2), 812–854; finite-size ensemble matters. |

## Agent memory and evidence loss

| Key | Primary reference | Status / relevance |
|---|---|---|
| M01 | Li et al. [TiMem: Temporal-Hierarchical Memory Consolidation for Long-Horizon Conversational Agents](https://aclanthology.org/2026.findings-acl.1091/). 2026. | Findings of ACL; direct hierarchy/consolidation comparator. |
| M02 | Yue et al. [HyperMem: Hypergraph Memory for Long-Term Conversations](https://aclanthology.org/2026.acl-long.1627/). 2026. | ACL long papers; episode/fact associations and retrieval. |
| M03 | Rusu, Khanzadeh & Alalfi. [Selective Forgetting: A Graph-Based Memory Framework for Long-Term LLM Agents](https://arxiv.org/abs/2608.28978). 2026. | August preprint; graph eviction, distinct storage regime from local token-capped pilot. |
| M04 | Yu, Lin & Wu. [WhenLoss: Diagnosing Write and Retrieval Bottlenecks in Long-Context Memory Systems](https://arxiv.org/abs/2605.24579). 2026. | May preprint; established write-versus-retrieval diagnostic. |
| M05 | [MemORAI: Memory Organization and Retrieval via Adaptive Graph Intelligence for LLM Conversational Agents](https://aclanthology.org/2026.findings-acl.1408/). 2026. | Findings of ACL; graph provenance/query-adaptive retrieval. |
| M06 | [StreamMeCo: Long-Term Agent Memory Compression for Efficient Streaming Video Understanding](https://aclanthology.org/2026.findings-acl.647/). 2026. | Findings of ACL; topology-guided compression in another modality. |
| M07 | [MemRiskBench](https://arxiv.org/html/2609.14976v1). 2026. | September preprint, inspected in the completed pilot; evaluation selection differs from memory compression. |
| M08 | [LongMemEval official repository](https://github.com/xiaowu0162/LongMemEval), [cleaned release](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned). | Public benchmark resources; exact local run pins in selective-forgetting. V2 is a distinct untested release. |

## Data, software and informal proposals

Public-data versions, licenses, access and usable units are centrally recorded in [public-data.md](audits/public-data.md) and its [machine-readable index](checks/public-cross-section/dataset-index.json). These include WildChat releases, Adopt ≠ Adapt annotations, LMSYS, ThoughtTrace, PRISM, Re-Centering Humans and EconomicIndex. They are not interchangeable participant populations.

- [TraceScope](https://github.com/Pixedar/TraceScope): inspected public software/README; flow visualization, not an independently validated mechanism.
- [Baladi, DSFT](https://zenodo.org/records/20303214): May 2026 archival proposal; repository resource type does not establish peer review.
- [Audria topic-drift post](https://audria.tech/blogs/topic-drift-detection/): engineering idea discovery, not technical validation.
- [Nour's author announcement](https://www.linkedin.com/posts/matthew-nour_llms-llm-psychiatry-activity-7374714916974370816-uufQ): discovery route to D01; no clinical conclusion transferred to CMRI.

Local papers and their evidentiary/publication status are in the [audited inventory](INVENTORY.md). The official [ISCS call](https://iutdijon.u-bourgogne.fr/ccs-france/) and [submission rules](https://iutdijon.u-bourgogne.fr/ccs-france/submissions/) are documented separately in [VENUE.md](VENUE.md). Earlier transit references remain in the [October 14 memo](../README.md#3-transit-restoration-under-uncertain-public-data); that deferred direction did not receive the same depth as the five selected candidates.
