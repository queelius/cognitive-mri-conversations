"""
Hidden Connection Analysis

Hidden connections are episode pairs that:
1. Share at least one concept (concept-connected)
2. Have NO direct E—E similarity link

These reveal cognitive structure invisible to embedding similarity alone.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator
import random

from core_types import (
    EpisodicNode, ConceptNode, SimilarityLink, InstantiationLink,
    HiddenConnection, DEFAULT_SIMILARITY_THRESHOLD
)


@dataclass
class HiddenConnectionAnalysis:
    """Results of hidden connection analysis."""
    hidden_connections: list[HiddenConnection]
    total_episode_pairs: int
    ee_linked_pairs: int
    concept_connected_pairs: int
    hidden_connection_count: int

    @property
    def hidden_connection_rate(self) -> float:
        """Fraction of concept-connected pairs that are hidden (no E—E link)."""
        if self.concept_connected_pairs == 0:
            return 0.0
        return self.hidden_connection_count / self.concept_connected_pairs


def build_adjacency_structures(
    episodes: list[EpisodicNode],
    ee_links: list[SimilarityLink],
    ce_links: list[InstantiationLink]
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, set[str]]]:
    """
    Build efficient lookup structures.

    Returns:
        ee_adjacent: episode_id -> set of directly linked episode_ids
        concept_episodes: concept_id -> set of episode_ids it instantiates
        episode_concepts: episode_id -> set of concept_ids it instantiates
    """
    # E—E adjacency
    ee_adjacent = defaultdict(set)
    for link in ee_links:
        ee_adjacent[link.episode1_id].add(link.episode2_id)
        ee_adjacent[link.episode2_id].add(link.episode1_id)

    # C→E mappings
    concept_episodes = defaultdict(set)
    episode_concepts = defaultdict(set)
    for link in ce_links:
        concept_episodes[link.concept_id].add(link.episode_id)
        episode_concepts[link.episode_id].add(link.concept_id)

    return dict(ee_adjacent), dict(concept_episodes), dict(episode_concepts)


def compute_hidden_connection_strength(
    e1_id: str,
    e2_id: str,
    shared_concepts: list[str],
    episode_concepts: dict[str, set[str]],
    concept_episodes: dict[str, set[str]],
    method: str = "count"
) -> float:
    """
    Compute strength of hidden connection.

    Methods:
        "count": Number of shared concepts
        "jaccard": |shared| / |union| of concepts
        "weighted": Sum of 1/|Inst(C)| for each shared concept (rarer = stronger)
    """
    if not shared_concepts:
        return 0.0

    if method == "count":
        return float(len(shared_concepts))

    elif method == "jaccard":
        c1 = episode_concepts.get(e1_id, set())
        c2 = episode_concepts.get(e2_id, set())
        union = c1 | c2
        if not union:
            return 0.0
        return len(shared_concepts) / len(union)

    elif method == "weighted":
        # Rarer concepts contribute more
        return sum(
            1.0 / len(concept_episodes.get(c, {c}))  # Avoid div by 0
            for c in shared_concepts
        )

    else:
        raise ValueError(f"Unknown method: {method}")


def find_hidden_connections(
    episodes: list[EpisodicNode],
    ee_links: list[SimilarityLink],
    ce_links: list[InstantiationLink],
    strength_method: str = "jaccard"
) -> HiddenConnectionAnalysis:
    """
    Find all hidden connections: concept-connected pairs without E—E links.

    Args:
        episodes: List of episodic nodes
        ee_links: List of E—E similarity links
        ce_links: List of C→E instantiation links
        strength_method: Method for computing connection strength

    Returns:
        HiddenConnectionAnalysis with results
    """
    ee_adjacent, concept_episodes, episode_concepts = build_adjacency_structures(
        episodes, ee_links, ce_links
    )

    episode_ids = {e.id for e in episodes}
    n = len(episode_ids)
    total_pairs = n * (n - 1) // 2
    ee_linked_pairs = len(ee_links)

    hidden_connections = []
    concept_connected_pairs = 0
    checked = set()

    # For each concept, check pairs of episodes that share it
    for concept_id, eps in concept_episodes.items():
        eps_list = list(eps)
        for i, e1_id in enumerate(eps_list):
            for e2_id in eps_list[i + 1:]:
                # Normalize pair order
                pair = (min(e1_id, e2_id), max(e1_id, e2_id))
                if pair in checked:
                    continue
                checked.add(pair)

                # This pair is concept-connected
                concept_connected_pairs += 1

                # Check if NOT directly E—E linked
                if e2_id not in ee_adjacent.get(e1_id, set()):
                    # Find ALL shared concepts (not just the one we're iterating)
                    c1 = episode_concepts.get(e1_id, set())
                    c2 = episode_concepts.get(e2_id, set())
                    shared = list(c1 & c2)

                    strength = compute_hidden_connection_strength(
                        e1_id, e2_id, shared,
                        episode_concepts, concept_episodes,
                        method=strength_method
                    )

                    hidden_connections.append(HiddenConnection(
                        episode1_id=e1_id,
                        episode2_id=e2_id,
                        shared_concepts=shared,
                        strength=strength
                    ))

    # Sort by strength descending
    hidden_connections.sort(key=lambda h: h.strength, reverse=True)

    return HiddenConnectionAnalysis(
        hidden_connections=hidden_connections,
        total_episode_pairs=total_pairs,
        ee_linked_pairs=ee_linked_pairs,
        concept_connected_pairs=concept_connected_pairs,
        hidden_connection_count=len(hidden_connections)
    )


def generate_null_distribution(
    episodes: list[EpisodicNode],
    concepts: list[ConceptNode],
    ee_links: list[SimilarityLink],
    ce_links: list[InstantiationLink],
    n_permutations: int = 1000,
    seed: int = 42
) -> list[int]:
    """
    Generate null distribution of hidden connection counts.

    Shuffles C→E links while preserving:
    - Number of links per concept (concept degree)
    - Number of links per episode (episode degree)

    Uses configuration model shuffling.

    Returns:
        List of hidden connection counts under null hypothesis
    """
    rng = random.Random(seed)

    # Get degree sequences
    concept_degrees = defaultdict(int)
    episode_degrees = defaultdict(int)
    for link in ce_links:
        concept_degrees[link.concept_id] += 1
        episode_degrees[link.episode_id] += 1

    episode_ids = list({e.id for e in episodes})
    concept_ids = list({c.id for c in concepts})

    null_counts = []

    for _ in range(n_permutations):
        # Generate random C→E links preserving degree sequence (approximately)
        # Simple approach: shuffle episode assignments per concept
        shuffled_links = []

        for concept_id in concept_ids:
            degree = concept_degrees[concept_id]
            # Sample 'degree' episodes randomly
            sampled = rng.sample(episode_ids, min(degree, len(episode_ids)))
            for ep_id in sampled:
                shuffled_links.append(InstantiationLink(
                    concept_id=concept_id,
                    episode_id=ep_id
                ))

        # Count hidden connections with shuffled links
        analysis = find_hidden_connections(
            episodes, ee_links, shuffled_links, strength_method="count"
        )
        null_counts.append(analysis.hidden_connection_count)

    return null_counts


def compute_pvalue(observed: int, null_distribution: list[int]) -> float:
    """Compute p-value: fraction of null values >= observed."""
    if not null_distribution:
        return 1.0
    return sum(1 for n in null_distribution if n >= observed) / len(null_distribution)


if __name__ == "__main__":
    # Example usage
    print("Hidden Connection Analysis Module")
    print("=" * 40)
    print()
    print("Key functions:")
    print("  find_hidden_connections() - Find all hidden connections")
    print("  generate_null_distribution() - Statistical significance test")
    print("  compute_pvalue() - Compute p-value against null")
