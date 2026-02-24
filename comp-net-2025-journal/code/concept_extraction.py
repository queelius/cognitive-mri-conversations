"""
Concept Extraction via LLM

Extracts abstract concepts (semantic memory) from episodic communities.
Uses LLM to identify general principles shared across conversation clusters.
"""

import json
import re
from dataclasses import dataclass
from typing import Optional, Callable
import hashlib

from core_types import EpisodicNode, ConceptNode, AbstractionLevel


@dataclass
class ExtractionConfig:
    """Configuration for concept extraction."""
    min_concepts_per_community: int = 2
    max_concepts_per_community: int = 5
    include_excerpts: bool = True
    excerpt_max_chars: int = 500
    abstraction_level: AbstractionLevel = AbstractionLevel.MEDIUM


@dataclass
class ExtractionResult:
    """Result of concept extraction for a community."""
    community_id: int
    concepts: list[ConceptNode]
    raw_response: str
    episode_count: int


def generate_concept_id(name: str, community_id: int) -> str:
    """Generate a unique concept ID from name and community."""
    hash_input = f"{community_id}:{name.lower().strip()}"
    return f"C_{hashlib.md5(hash_input.encode()).hexdigest()[:8]}"


def build_extraction_prompt(
    episodes: list[EpisodicNode],
    community_id: int,
    config: ExtractionConfig
) -> str:
    """
    Build the LLM prompt for concept extraction.

    Args:
        episodes: Episodes in the community
        community_id: ID of the community
        config: Extraction configuration

    Returns:
        Prompt string for LLM
    """
    # Build episode descriptions
    episode_descriptions = []
    for ep in episodes:
        desc = f"- {ep.title}"
        if config.include_excerpts and ep.content:
            excerpt = ep.content[:config.excerpt_max_chars]
            if len(ep.content) > config.excerpt_max_chars:
                excerpt += "..."
            desc += f"\n  Excerpt: {excerpt}"
        episode_descriptions.append(desc)

    episodes_text = "\n".join(episode_descriptions)

    prompt = f"""Here are {len(episodes)} conversations from a single user's chat history that cluster together semantically (Community {community_id}):

{episodes_text}

Identify {config.min_concepts_per_community}-{config.max_concepts_per_community} general principles, concepts, or abstract themes that these conversations share.

Requirements:
- Be specific but abstract — identify underlying ideas, not just surface topics
- Each concept should apply to multiple conversations in this cluster
- Focus on cognitive patterns, problem-solving approaches, or knowledge structures
- Avoid overly generic concepts like "learning" or "problem-solving"

Return your response as a JSON array with this structure:
[
  {{
    "name": "Short concept name (2-5 words)",
    "description": "One sentence explaining the concept and how it manifests in these conversations",
    "example_episodes": ["title1", "title2"]  // 2-3 example conversation titles
  }}
]

Return ONLY the JSON array, no other text."""

    return prompt


def parse_extraction_response(
    response: str,
    community_id: int,
    episode_titles: set[str]
) -> list[ConceptNode]:
    """
    Parse LLM response into ConceptNode objects.

    Args:
        response: Raw LLM response
        community_id: Source community ID
        episode_titles: Valid episode titles for validation

    Returns:
        List of ConceptNode objects
    """
    # Try to extract JSON from response
    # Handle cases where LLM adds extra text
    json_match = re.search(r'\[.*\]', response, re.DOTALL)
    if not json_match:
        raise ValueError(f"Could not find JSON array in response: {response[:200]}...")

    try:
        concepts_data = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in response: {e}")

    concepts = []
    for item in concepts_data:
        name = item.get("name", "").strip()
        description = item.get("description", "").strip()
        examples = item.get("example_episodes", [])

        if not name or not description:
            continue

        # Validate example episodes exist
        valid_examples = [ex for ex in examples if ex in episode_titles]

        concept = ConceptNode(
            id=generate_concept_id(name, community_id),
            name=name,
            description=description,
            source_community=community_id,
            abstraction_level=AbstractionLevel.MEDIUM,
            example_episodes=valid_examples
        )
        concepts.append(concept)

    return concepts


def extract_concepts_from_community(
    episodes: list[EpisodicNode],
    community_id: int,
    llm_call: Callable[[str], str],
    config: Optional[ExtractionConfig] = None
) -> ExtractionResult:
    """
    Extract concepts from a single episodic community.

    Args:
        episodes: Episodes in the community
        community_id: ID of the community
        llm_call: Function that takes prompt string, returns LLM response
        config: Extraction configuration

    Returns:
        ExtractionResult with concepts and metadata
    """
    if config is None:
        config = ExtractionConfig()

    prompt = build_extraction_prompt(episodes, community_id, config)
    response = llm_call(prompt)

    episode_titles = {ep.title for ep in episodes}
    concepts = parse_extraction_response(response, community_id, episode_titles)

    return ExtractionResult(
        community_id=community_id,
        concepts=concepts,
        raw_response=response,
        episode_count=len(episodes)
    )


def extract_all_concepts(
    episodes: list[EpisodicNode],
    llm_call: Callable[[str], str],
    config: Optional[ExtractionConfig] = None
) -> tuple[list[ConceptNode], list[ExtractionResult]]:
    """
    Extract concepts from all episodic communities.

    Args:
        episodes: All episodes (must have community_id set)
        llm_call: Function that takes prompt string, returns LLM response
        config: Extraction configuration

    Returns:
        Tuple of (all_concepts, extraction_results_by_community)
    """
    if config is None:
        config = ExtractionConfig()

    # Group episodes by community
    communities: dict[int, list[EpisodicNode]] = {}
    for ep in episodes:
        if ep.community_id is not None:
            if ep.community_id not in communities:
                communities[ep.community_id] = []
            communities[ep.community_id].append(ep)

    all_concepts = []
    all_results = []

    for community_id in sorted(communities.keys()):
        community_episodes = communities[community_id]
        result = extract_concepts_from_community(
            community_episodes, community_id, llm_call, config
        )
        all_concepts.extend(result.concepts)
        all_results.append(result)

    return all_concepts, all_results


def validate_extraction_consistency(
    episodes: list[EpisodicNode],
    community_id: int,
    llm_call: Callable[[str], str],
    n_runs: int = 3,
    config: Optional[ExtractionConfig] = None
) -> dict:
    """
    Validate extraction consistency by running multiple times.

    Returns dict with:
        - concept_names: list of sets of concept names per run
        - overlap_rate: fraction of concepts appearing in multiple runs
        - stable_concepts: concepts appearing in all runs
    """
    if config is None:
        config = ExtractionConfig()

    all_names = []
    for _ in range(n_runs):
        result = extract_concepts_from_community(
            episodes, community_id, llm_call, config
        )
        names = {c.name.lower() for c in result.concepts}
        all_names.append(names)

    # Find stable concepts (appear in all runs)
    stable = set.intersection(*all_names) if all_names else set()

    # Calculate overlap rate
    all_unique = set.union(*all_names) if all_names else set()
    if not all_unique:
        overlap_rate = 0.0
    else:
        # Count how many times each concept appears
        appearance_counts = {}
        for names in all_names:
            for name in names:
                appearance_counts[name] = appearance_counts.get(name, 0) + 1

        # Fraction appearing more than once
        multi_appear = sum(1 for c in appearance_counts.values() if c > 1)
        overlap_rate = multi_appear / len(all_unique)

    return {
        "concept_names": all_names,
        "overlap_rate": overlap_rate,
        "stable_concepts": stable,
        "n_runs": n_runs
    }


def merge_similar_concepts(
    concepts: list[ConceptNode],
    similarity_threshold: float = 0.8,
    embedding_fn: Optional[Callable[[str], list[float]]] = None
) -> list[ConceptNode]:
    """
    Merge concepts with similar names/descriptions across communities.

    This handles cases where the same abstract concept is extracted
    from different communities with slightly different names.

    Args:
        concepts: List of concepts to potentially merge
        similarity_threshold: Minimum similarity for merging
        embedding_fn: Optional function to compute embeddings for semantic comparison

    Returns:
        Deduplicated list of concepts
    """
    if not embedding_fn:
        # Simple name-based deduplication
        seen = {}
        result = []
        for c in concepts:
            key = c.name.lower().strip()
            if key not in seen:
                seen[key] = c
                result.append(c)
            else:
                # Merge example_episodes
                existing = seen[key]
                existing.example_episodes = list(
                    set(existing.example_episodes) | set(c.example_episodes)
                )
        return result

    # TODO: Implement embedding-based similarity merging
    # For now, fall back to name-based
    return merge_similar_concepts(concepts, similarity_threshold, None)


if __name__ == "__main__":
    print("Concept Extraction Module")
    print("=" * 40)
    print()
    print("Key functions:")
    print("  extract_concepts_from_community() - Extract from single community")
    print("  extract_all_concepts() - Extract from all communities")
    print("  validate_extraction_consistency() - Check extraction reliability")
    print("  merge_similar_concepts() - Deduplicate across communities")
