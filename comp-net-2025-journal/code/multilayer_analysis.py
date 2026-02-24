"""
Multi-Layer Network Analysis

Analyzes the hierarchical cognitive memory network:
- Episodic layer (E—E similarity links)
- Concept layer (C—C association links)
- Cross-layer (C→E instantiation links)

Computes community structure, centrality, and layer coupling metrics.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
import math

from core_types import (
    EpisodicNode, ConceptNode,
    SimilarityLink, InstantiationLink, AssociationLink
)


@dataclass
class MultiLayerNetwork:
    """Complete multi-layer cognitive memory network."""
    episodes: list[EpisodicNode]
    concepts: list[ConceptNode]
    ee_links: list[SimilarityLink]
    ce_links: list[InstantiationLink]
    cc_links: list[AssociationLink]

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    @property
    def num_concepts(self) -> int:
        return len(self.concepts)

    @property
    def abstraction_ratio(self) -> float:
        """Ratio of concepts to episodes."""
        if self.num_episodes == 0:
            return 0.0
        return self.num_concepts / self.num_episodes


@dataclass
class LayerMetrics:
    """Metrics for a single network layer."""
    num_nodes: int
    num_edges: int
    density: float
    avg_degree: float
    max_degree: int
    num_components: int
    largest_component_size: int


@dataclass
class CrossLayerMetrics:
    """Metrics for cross-layer coupling."""
    num_ce_links: int
    avg_ce_per_concept: float
    avg_ce_per_episode: float
    concept_coverage: float  # Fraction of episodes with at least one concept
    episode_coverage: float  # Fraction of concepts with at least one episode


def compute_layer_metrics(
    nodes: list,
    edges: list,
    get_node_ids: callable,
    get_edge_endpoints: callable
) -> LayerMetrics:
    """
    Compute metrics for a single network layer.

    Args:
        nodes: List of node objects
        edges: List of edge objects
        get_node_ids: Function to extract ID from node
        get_edge_endpoints: Function to extract (id1, id2) from edge

    Returns:
        LayerMetrics
    """
    node_ids = {get_node_ids(n) for n in nodes}
    n = len(node_ids)

    if n == 0:
        return LayerMetrics(0, 0, 0.0, 0.0, 0, 0, 0)

    # Build adjacency for degree calculation and component finding
    adjacency = defaultdict(set)
    for edge in edges:
        id1, id2 = get_edge_endpoints(edge)
        if id1 in node_ids and id2 in node_ids:
            adjacency[id1].add(id2)
            adjacency[id2].add(id1)

    degrees = [len(adjacency.get(nid, set())) for nid in node_ids]
    num_edges = sum(degrees) // 2

    # Find connected components via BFS
    visited = set()
    components = []
    for start in node_ids:
        if start in visited:
            continue
        component = set()
        queue = [start]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            for neighbor in adjacency.get(node, set()):
                if neighbor not in visited:
                    queue.append(neighbor)
        components.append(component)

    max_possible_edges = n * (n - 1) // 2

    return LayerMetrics(
        num_nodes=n,
        num_edges=num_edges,
        density=num_edges / max_possible_edges if max_possible_edges > 0 else 0.0,
        avg_degree=sum(degrees) / n if n > 0 else 0.0,
        max_degree=max(degrees) if degrees else 0,
        num_components=len(components),
        largest_component_size=max(len(c) for c in components) if components else 0
    )


def compute_episodic_layer_metrics(
    episodes: list[EpisodicNode],
    ee_links: list[SimilarityLink]
) -> LayerMetrics:
    """Compute metrics for the episodic (E—E) layer."""
    return compute_layer_metrics(
        episodes,
        ee_links,
        lambda e: e.id,
        lambda l: (l.episode1_id, l.episode2_id)
    )


def compute_concept_layer_metrics(
    concepts: list[ConceptNode],
    cc_links: list[AssociationLink]
) -> LayerMetrics:
    """Compute metrics for the concept (C—C) layer."""
    return compute_layer_metrics(
        concepts,
        cc_links,
        lambda c: c.id,
        lambda l: (l.concept1_id, l.concept2_id)
    )


def compute_cross_layer_metrics(
    episodes: list[EpisodicNode],
    concepts: list[ConceptNode],
    ce_links: list[InstantiationLink]
) -> CrossLayerMetrics:
    """Compute metrics for the cross-layer (C→E) coupling."""
    concept_ids = {c.id for c in concepts}
    episode_ids = {e.id for e in episodes}

    # Count links per concept and episode
    concepts_per_episode = defaultdict(set)
    episodes_per_concept = defaultdict(set)

    for link in ce_links:
        if link.concept_id in concept_ids and link.episode_id in episode_ids:
            concepts_per_episode[link.episode_id].add(link.concept_id)
            episodes_per_concept[link.concept_id].add(link.episode_id)

    num_concepts = len(concept_ids)
    num_episodes = len(episode_ids)

    return CrossLayerMetrics(
        num_ce_links=len(ce_links),
        avg_ce_per_concept=(
            sum(len(eps) for eps in episodes_per_concept.values()) / num_concepts
            if num_concepts > 0 else 0.0
        ),
        avg_ce_per_episode=(
            sum(len(cs) for cs in concepts_per_episode.values()) / num_episodes
            if num_episodes > 0 else 0.0
        ),
        concept_coverage=(
            len(concepts_per_episode) / num_episodes
            if num_episodes > 0 else 0.0
        ),
        episode_coverage=(
            len(episodes_per_concept) / num_concepts
            if num_concepts > 0 else 0.0
        )
    )


def detect_concept_communities(
    concepts: list[ConceptNode],
    cc_links: list[AssociationLink],
    resolution: float = 1.0
) -> dict[str, int]:
    """
    Detect communities in the concept layer using a simple modularity-based approach.

    This is a simplified Louvain-like algorithm. For production, consider using
    networkx.community or python-louvain.

    Args:
        concepts: List of concept nodes
        cc_links: List of C—C association links
        resolution: Resolution parameter (higher = more communities)

    Returns:
        Dict mapping concept_id -> community_id
    """
    # Build adjacency with weights
    adjacency = defaultdict(dict)
    for link in cc_links:
        adjacency[link.concept1_id][link.concept2_id] = link.weight
        adjacency[link.concept2_id][link.concept1_id] = link.weight

    concept_ids = [c.id for c in concepts]

    # Initialize: each node in its own community
    community = {cid: i for i, cid in enumerate(concept_ids)}

    # Total edge weight
    m = sum(link.weight for link in cc_links)
    if m == 0:
        return community

    # Node strengths (weighted degree)
    strength = {cid: sum(adjacency[cid].values()) for cid in concept_ids}

    # Simple greedy optimization
    improved = True
    while improved:
        improved = False
        for node in concept_ids:
            current_comm = community[node]

            # Find neighboring communities
            neighbor_comms = set()
            for neighbor in adjacency[node]:
                neighbor_comms.add(community[neighbor])

            if not neighbor_comms:
                continue

            # Calculate modularity gain for moving to each neighbor community
            best_gain = 0.0
            best_comm = current_comm

            for target_comm in neighbor_comms:
                if target_comm == current_comm:
                    continue

                # Calculate gain (simplified)
                # Sum of weights to nodes in target community
                sum_in = sum(
                    adjacency[node].get(other, 0)
                    for other in concept_ids
                    if community[other] == target_comm
                )

                # Sum of strengths in target community
                sum_tot = sum(
                    strength[other]
                    for other in concept_ids
                    if community[other] == target_comm
                )

                k_i = strength[node]
                gain = sum_in - resolution * k_i * sum_tot / (2 * m)

                if gain > best_gain:
                    best_gain = gain
                    best_comm = target_comm

            if best_comm != current_comm:
                community[node] = best_comm
                improved = True

    # Renumber communities to be contiguous
    unique_comms = sorted(set(community.values()))
    comm_map = {old: new for new, old in enumerate(unique_comms)}
    return {cid: comm_map[comm] for cid, comm in community.items()}


def compare_community_structures(
    episode_communities: dict[str, int],
    concept_communities: dict[str, int],
    ce_links: list[InstantiationLink]
) -> dict:
    """
    Compare episodic and concept community structures.

    Analyzes how concepts group compared to episodes, using C→E links.

    Returns dict with:
        - nmi: Normalized mutual information (if applicable)
        - concept_community_episode_distribution: How episodes distribute in concept communities
        - bridge_concepts: Concepts linking multiple episode communities
    """
    # Build mappings
    concept_to_episodes = defaultdict(set)
    for link in ce_links:
        concept_to_episodes[link.concept_id].add(link.episode_id)

    # For each concept community, what episode communities are represented?
    concept_comm_to_ep_comms = defaultdict(lambda: defaultdict(int))
    for concept_id, concept_comm in concept_communities.items():
        for ep_id in concept_to_episodes.get(concept_id, set()):
            if ep_id in episode_communities:
                ep_comm = episode_communities[ep_id]
                concept_comm_to_ep_comms[concept_comm][ep_comm] += 1

    # Find bridge concepts (span multiple episode communities)
    bridge_concepts = []
    for concept_id, concept_comm in concept_communities.items():
        ep_ids = concept_to_episodes.get(concept_id, set())
        ep_comms = {episode_communities[ep_id] for ep_id in ep_ids if ep_id in episode_communities}
        if len(ep_comms) > 1:
            bridge_concepts.append({
                "concept_id": concept_id,
                "concept_community": concept_comm,
                "episode_communities": sorted(ep_comms),
                "num_episode_communities": len(ep_comms)
            })

    bridge_concepts.sort(key=lambda x: x["num_episode_communities"], reverse=True)

    return {
        "concept_comm_episode_distribution": dict(concept_comm_to_ep_comms),
        "bridge_concepts": bridge_concepts,
        "num_bridge_concepts": len(bridge_concepts),
        "num_concept_communities": len(set(concept_communities.values())),
        "num_episode_communities": len(set(episode_communities.values()))
    }


def compute_concept_hub_scores(
    concepts: list[ConceptNode],
    ce_links: list[InstantiationLink],
    cc_links: list[AssociationLink]
) -> dict[str, dict]:
    """
    Compute hub scores for concepts.

    A concept hub connects many episodes and/or bridges between other concepts.

    Returns dict: concept_id -> {
        instantiation_degree: number of episodes
        association_degree: number of connected concepts
        hub_score: combined metric
    }
    """
    # Count instantiation links per concept
    inst_degree = defaultdict(int)
    for link in ce_links:
        inst_degree[link.concept_id] += 1

    # Count association links per concept
    assoc_degree = defaultdict(int)
    for link in cc_links:
        assoc_degree[link.concept1_id] += 1
        assoc_degree[link.concept2_id] += 1

    # Compute hub scores
    scores = {}
    for concept in concepts:
        cid = concept.id
        inst = inst_degree.get(cid, 0)
        assoc = assoc_degree.get(cid, 0)

        # Simple combined hub score (can be weighted)
        hub_score = inst + assoc

        scores[cid] = {
            "instantiation_degree": inst,
            "association_degree": assoc,
            "hub_score": hub_score
        }

    return scores


def project_concepts_to_episodes(
    episodes: list[EpisodicNode],
    ce_links: list[InstantiationLink]
) -> list[tuple[str, str, float]]:
    """
    Project concept layer onto episode layer.

    Creates "virtual" E—E edges for episodes that share concepts.
    These are the basis of "hidden connections" when they don't have direct E—E links.

    Returns list of (episode1_id, episode2_id, strength) tuples.
    """
    # Build concept -> episodes mapping
    concept_episodes = defaultdict(set)
    for link in ce_links:
        concept_episodes[link.concept_id].add(link.episode_id)

    # Find episode pairs that share concepts
    episode_pairs = defaultdict(list)
    for concept_id, eps in concept_episodes.items():
        eps_list = list(eps)
        for i, e1 in enumerate(eps_list):
            for e2 in eps_list[i + 1:]:
                pair = (min(e1, e2), max(e1, e2))
                episode_pairs[pair].append(concept_id)

    # Convert to list with strength (number of shared concepts)
    projection = []
    for (e1, e2), shared_concepts in episode_pairs.items():
        projection.append((e1, e2, float(len(shared_concepts))))

    return projection


def analyze_multilayer_network(network: MultiLayerNetwork) -> dict:
    """
    Comprehensive analysis of the multi-layer network.

    Returns dict with all computed metrics and analyses.
    """
    episodic_metrics = compute_episodic_layer_metrics(
        network.episodes, network.ee_links
    )
    concept_metrics = compute_concept_layer_metrics(
        network.concepts, network.cc_links
    )
    cross_layer_metrics = compute_cross_layer_metrics(
        network.episodes, network.concepts, network.ce_links
    )

    hub_scores = compute_concept_hub_scores(
        network.concepts, network.ce_links, network.cc_links
    )

    # Top hubs
    top_hubs = sorted(
        hub_scores.items(),
        key=lambda x: x[1]["hub_score"],
        reverse=True
    )[:10]

    return {
        "abstraction_ratio": network.abstraction_ratio,
        "episodic_layer": {
            "num_nodes": episodic_metrics.num_nodes,
            "num_edges": episodic_metrics.num_edges,
            "density": episodic_metrics.density,
            "avg_degree": episodic_metrics.avg_degree,
            "num_components": episodic_metrics.num_components
        },
        "concept_layer": {
            "num_nodes": concept_metrics.num_nodes,
            "num_edges": concept_metrics.num_edges,
            "density": concept_metrics.density,
            "avg_degree": concept_metrics.avg_degree,
            "num_components": concept_metrics.num_components
        },
        "cross_layer": {
            "num_links": cross_layer_metrics.num_ce_links,
            "avg_per_concept": cross_layer_metrics.avg_ce_per_concept,
            "avg_per_episode": cross_layer_metrics.avg_ce_per_episode,
            "concept_coverage": cross_layer_metrics.concept_coverage,
            "episode_coverage": cross_layer_metrics.episode_coverage
        },
        "top_concept_hubs": [
            {"concept_id": cid, **scores}
            for cid, scores in top_hubs
        ]
    }


if __name__ == "__main__":
    print("Multi-Layer Network Analysis Module")
    print("=" * 40)
    print()
    print("Key functions:")
    print("  compute_episodic_layer_metrics() - E—E layer metrics")
    print("  compute_concept_layer_metrics() - C—C layer metrics")
    print("  compute_cross_layer_metrics() - C→E coupling metrics")
    print("  detect_concept_communities() - Community detection in C layer")
    print("  compare_community_structures() - Compare E vs C communities")
    print("  analyze_multilayer_network() - Comprehensive analysis")
