"""
Concept Association Network (C—C Links)

Computes associations between concepts based on overlapping episode sets.
Primary method: Jaccard similarity of instantiated episodes.
"""

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

from core_types import (
    ConceptNode, InstantiationLink, AssociationLink,
    DEFAULT_ASSOCIATION_THRESHOLD
)


@dataclass
class AssociationNetworkStats:
    """Statistics about the C—C association network."""
    num_concepts: int
    num_links: int
    density: float
    avg_weight: float
    max_weight: float
    isolated_concepts: int  # Concepts with no C—C links


def build_concept_episode_sets(
    ce_links: list[InstantiationLink]
) -> dict[str, set[str]]:
    """Build mapping from concept_id -> set of episode_ids."""
    concept_episodes = defaultdict(set)
    for link in ce_links:
        concept_episodes[link.concept_id].add(link.episode_id)
    return dict(concept_episodes)


def jaccard_similarity(set1: set, set2: set) -> float:
    """Compute Jaccard similarity: |intersection| / |union|."""
    if not set1 and not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def compute_association_network(
    concepts: list[ConceptNode],
    ce_links: list[InstantiationLink],
    threshold: float = DEFAULT_ASSOCIATION_THRESHOLD,
    include_shared_episodes: bool = True
) -> tuple[list[AssociationLink], AssociationNetworkStats]:
    """
    Compute C—C association links based on Jaccard similarity of episode sets.

    Args:
        concepts: List of concept nodes
        ce_links: List of C→E instantiation links
        threshold: Minimum Jaccard similarity for creating link (γ)
        include_shared_episodes: Whether to store shared episode IDs in links

    Returns:
        Tuple of (links, stats)
    """
    concept_episodes = build_concept_episode_sets(ce_links)
    concept_ids = [c.id for c in concepts]

    links = []
    weights = []

    for c1_id, c2_id in combinations(concept_ids, 2):
        eps1 = concept_episodes.get(c1_id, set())
        eps2 = concept_episodes.get(c2_id, set())

        jaccard = jaccard_similarity(eps1, eps2)

        if jaccard >= threshold:
            shared = list(eps1 & eps2) if include_shared_episodes else []
            links.append(AssociationLink(
                concept1_id=c1_id,
                concept2_id=c2_id,
                weight=jaccard,
                shared_episodes=shared
            ))
            weights.append(jaccard)

    # Compute stats
    num_concepts = len(concept_ids)
    max_possible_links = num_concepts * (num_concepts - 1) // 2
    density = len(links) / max_possible_links if max_possible_links > 0 else 0.0

    # Find isolated concepts
    linked_concepts = set()
    for link in links:
        linked_concepts.add(link.concept1_id)
        linked_concepts.add(link.concept2_id)
    isolated = num_concepts - len(linked_concepts)

    stats = AssociationNetworkStats(
        num_concepts=num_concepts,
        num_links=len(links),
        density=density,
        avg_weight=sum(weights) / len(weights) if weights else 0.0,
        max_weight=max(weights) if weights else 0.0,
        isolated_concepts=isolated
    )

    return links, stats


def get_concept_neighbors(
    concept_id: str,
    cc_links: list[AssociationLink]
) -> list[tuple[str, float]]:
    """Get neighbors of a concept in the C—C network with weights."""
    neighbors = []
    for link in cc_links:
        if link.concept1_id == concept_id:
            neighbors.append((link.concept2_id, link.weight))
        elif link.concept2_id == concept_id:
            neighbors.append((link.concept1_id, link.weight))
    return sorted(neighbors, key=lambda x: x[1], reverse=True)


def compute_concept_centrality(
    concepts: list[ConceptNode],
    cc_links: list[AssociationLink]
) -> dict[str, dict[str, float]]:
    """
    Compute centrality metrics for concepts in C—C network.

    Returns dict: concept_id -> {degree, weighted_degree, ...}
    """
    # Build adjacency
    adjacency = defaultdict(list)
    for link in cc_links:
        adjacency[link.concept1_id].append((link.concept2_id, link.weight))
        adjacency[link.concept2_id].append((link.concept1_id, link.weight))

    centrality = {}
    for concept in concepts:
        neighbors = adjacency.get(concept.id, [])
        centrality[concept.id] = {
            "degree": len(neighbors),
            "weighted_degree": sum(w for _, w in neighbors),
            "avg_neighbor_weight": (
                sum(w for _, w in neighbors) / len(neighbors)
                if neighbors else 0.0
            )
        }

    return centrality


def find_bridge_concepts(
    concepts: list[ConceptNode],
    ce_links: list[InstantiationLink],
    episode_communities: dict[str, int]
) -> list[tuple[str, int, list[int]]]:
    """
    Find concepts that span multiple episodic communities.

    These are "bridge concepts" — abstract principles connecting different domains.

    Args:
        concepts: List of concept nodes
        ce_links: List of C→E instantiation links
        episode_communities: Mapping from episode_id -> community_id

    Returns:
        List of (concept_id, num_communities, community_list) sorted by num_communities desc
    """
    concept_episodes = build_concept_episode_sets(ce_links)

    bridge_concepts = []
    for concept in concepts:
        eps = concept_episodes.get(concept.id, set())
        communities = set()
        for ep_id in eps:
            if ep_id in episode_communities:
                communities.add(episode_communities[ep_id])

        if len(communities) > 1:
            bridge_concepts.append((
                concept.id,
                len(communities),
                sorted(communities)
            ))

    return sorted(bridge_concepts, key=lambda x: x[1], reverse=True)


if __name__ == "__main__":
    print("Concept Association Network Module")
    print("=" * 40)
    print()
    print("Key functions:")
    print("  compute_association_network() - Build C—C links from C→E links")
    print("  find_bridge_concepts() - Find concepts spanning multiple communities")
    print("  compute_concept_centrality() - Centrality metrics for concepts")
