"""
Instantiation Scoring (C→E Links)

Determines which episodes instantiate which concepts via LLM judgment.
Creates weighted directed links from concepts to their exemplar episodes.
"""

import json
import re
from dataclasses import dataclass
from typing import Optional, Callable
from itertools import product

from core_types import (
    EpisodicNode, ConceptNode, InstantiationLink,
    DEFAULT_INSTANTIATION_THRESHOLD
)


@dataclass
class ScoringConfig:
    """Configuration for instantiation scoring."""
    threshold: float = DEFAULT_INSTANTIATION_THRESHOLD  # Min score for link (0-3)
    batch_size: int = 5  # Episodes per LLM call for batch mode
    include_excerpts: bool = True
    excerpt_max_chars: int = 300


@dataclass
class ScoringResult:
    """Result of scoring a concept-episode pair."""
    concept_id: str
    episode_id: str
    score: float  # 0-3 scale
    raw_response: str
    reasoning: Optional[str] = None


def build_single_scoring_prompt(
    concept: ConceptNode,
    episode: EpisodicNode,
    config: ScoringConfig
) -> str:
    """Build prompt for scoring a single concept-episode pair."""
    episode_content = episode.title
    if config.include_excerpts and episode.content:
        excerpt = episode.content[:config.excerpt_max_chars]
        if len(episode.content) > config.excerpt_max_chars:
            excerpt += "..."
        episode_content += f"\n\nExcerpt: {excerpt}"

    prompt = f"""Concept: {concept.name}
Description: {concept.description}

Conversation: {episode_content}

Does this conversation exemplify or discuss this concept?

Rate on a scale of 0-3:
0 = Not at all related
1 = Tangentially related, concept not central
2 = Related, concept is present but not the main focus
3 = Strongly exemplifies, concept is central to the conversation

Note: The conversation may exemplify the concept even if it doesn't use the same words — look for the underlying idea.

Return your response as JSON:
{{"score": <0-3>, "reasoning": "brief explanation"}}

Return ONLY the JSON, no other text."""

    return prompt


def build_batch_scoring_prompt(
    concept: ConceptNode,
    episodes: list[EpisodicNode],
    config: ScoringConfig
) -> str:
    """Build prompt for scoring multiple episodes against one concept."""
    episode_texts = []
    for i, ep in enumerate(episodes):
        text = f"{i+1}. {ep.title}"
        if config.include_excerpts and ep.content:
            excerpt = ep.content[:config.excerpt_max_chars]
            if len(ep.content) > config.excerpt_max_chars:
                excerpt += "..."
            text += f"\n   Excerpt: {excerpt}"
        episode_texts.append(text)

    episodes_text = "\n\n".join(episode_texts)

    prompt = f"""Concept: {concept.name}
Description: {concept.description}

Rate how well each of these conversations exemplifies this concept.

Conversations:
{episodes_text}

Rating scale (0-3):
0 = Not at all related
1 = Tangentially related
2 = Related, concept present
3 = Strongly exemplifies

Return as JSON array:
[
  {{"id": 1, "score": <0-3>}},
  {{"id": 2, "score": <0-3>}},
  ...
]

Return ONLY the JSON array."""

    return prompt


def parse_single_response(response: str) -> tuple[float, Optional[str]]:
    """Parse response for single concept-episode scoring."""
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if not json_match:
        # Try to extract just a number
        num_match = re.search(r'\b([0-3])\b', response)
        if num_match:
            return float(num_match.group(1)), None
        raise ValueError(f"Could not parse response: {response[:200]}")

    try:
        data = json.loads(json_match.group())
        score = float(data.get("score", 0))
        reasoning = data.get("reasoning")
        return min(3.0, max(0.0, score)), reasoning
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Invalid JSON in response: {e}")


def parse_batch_response(
    response: str,
    episode_ids: list[str]
) -> dict[str, float]:
    """Parse response for batch scoring."""
    json_match = re.search(r'\[.*\]', response, re.DOTALL)
    if not json_match:
        raise ValueError(f"Could not find JSON array in response: {response[:200]}")

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in response: {e}")

    scores = {}
    for item in data:
        idx = item.get("id", 0) - 1  # Convert to 0-indexed
        if 0 <= idx < len(episode_ids):
            score = float(item.get("score", 0))
            scores[episode_ids[idx]] = min(3.0, max(0.0, score))

    return scores


def score_single_pair(
    concept: ConceptNode,
    episode: EpisodicNode,
    llm_call: Callable[[str], str],
    config: Optional[ScoringConfig] = None
) -> ScoringResult:
    """
    Score a single concept-episode pair.

    Args:
        concept: The concept node
        episode: The episode node
        llm_call: Function that takes prompt, returns response
        config: Scoring configuration

    Returns:
        ScoringResult with score and metadata
    """
    if config is None:
        config = ScoringConfig()

    prompt = build_single_scoring_prompt(concept, episode, config)
    response = llm_call(prompt)
    score, reasoning = parse_single_response(response)

    return ScoringResult(
        concept_id=concept.id,
        episode_id=episode.id,
        score=score,
        raw_response=response,
        reasoning=reasoning
    )


def score_concept_episodes_batch(
    concept: ConceptNode,
    episodes: list[EpisodicNode],
    llm_call: Callable[[str], str],
    config: Optional[ScoringConfig] = None
) -> list[ScoringResult]:
    """
    Score multiple episodes against a single concept in batches.

    Args:
        concept: The concept node
        episodes: List of episode nodes
        llm_call: Function that takes prompt, returns response
        config: Scoring configuration

    Returns:
        List of ScoringResults
    """
    if config is None:
        config = ScoringConfig()

    results = []

    # Process in batches
    for i in range(0, len(episodes), config.batch_size):
        batch = episodes[i:i + config.batch_size]
        episode_ids = [ep.id for ep in batch]

        prompt = build_batch_scoring_prompt(concept, batch, config)
        response = llm_call(prompt)

        try:
            scores = parse_batch_response(response, episode_ids)
            for ep_id, score in scores.items():
                results.append(ScoringResult(
                    concept_id=concept.id,
                    episode_id=ep_id,
                    score=score,
                    raw_response=response
                ))
        except ValueError:
            # Fall back to individual scoring
            for ep in batch:
                try:
                    result = score_single_pair(concept, ep, llm_call, config)
                    results.append(result)
                except ValueError:
                    # Skip on failure
                    pass

    return results


def create_instantiation_links(
    concepts: list[ConceptNode],
    episodes: list[EpisodicNode],
    llm_call: Callable[[str], str],
    config: Optional[ScoringConfig] = None,
    scope: str = "same_community"
) -> tuple[list[InstantiationLink], list[ScoringResult]]:
    """
    Create C→E instantiation links by scoring all relevant pairs.

    Args:
        concepts: List of concept nodes
        episodes: List of episode nodes
        llm_call: Function that takes prompt, returns response
        config: Scoring configuration
        scope: Which pairs to score:
            - "same_community": Only score episodes in concept's source community
            - "all": Score all concept-episode pairs (expensive!)
            - "extended": Same community + random sample from others

    Returns:
        Tuple of (instantiation_links, all_scoring_results)
    """
    if config is None:
        config = ScoringConfig()

    # Group episodes by community
    community_episodes: dict[int, list[EpisodicNode]] = {}
    for ep in episodes:
        cid = ep.community_id if ep.community_id is not None else -1
        if cid not in community_episodes:
            community_episodes[cid] = []
        community_episodes[cid].append(ep)

    all_results = []
    links = []

    for concept in concepts:
        # Determine which episodes to score
        if scope == "same_community" and concept.source_community is not None:
            target_episodes = community_episodes.get(concept.source_community, [])
        elif scope == "all":
            target_episodes = episodes
        else:
            # extended: same community + sample from others
            target_episodes = community_episodes.get(concept.source_community, [])
            # TODO: Add sampling from other communities

        if not target_episodes:
            continue

        # Score in batches
        results = score_concept_episodes_batch(
            concept, target_episodes, llm_call, config
        )
        all_results.extend(results)

        # Create links for scores above threshold
        for result in results:
            if result.score >= config.threshold:
                links.append(InstantiationLink(
                    concept_id=result.concept_id,
                    episode_id=result.episode_id,
                    score=result.score
                ))

    return links, all_results


def compute_concept_coverage(
    concepts: list[ConceptNode],
    episodes: list[EpisodicNode],
    ce_links: list[InstantiationLink]
) -> dict:
    """
    Compute coverage statistics for concepts.

    Returns dict with:
        - concepts_by_episode_count: Distribution of episode counts per concept
        - episodes_with_concepts: Fraction of episodes linked to at least one concept
        - avg_concepts_per_episode: Average concepts per episode
        - orphan_episodes: Episodes with no concept links
    """
    concept_episode_counts = {c.id: 0 for c in concepts}
    episode_concept_counts = {e.id: 0 for e in episodes}

    for link in ce_links:
        if link.concept_id in concept_episode_counts:
            concept_episode_counts[link.concept_id] += 1
        if link.episode_id in episode_concept_counts:
            episode_concept_counts[link.episode_id] += 1

    orphan_episodes = [
        ep_id for ep_id, count in episode_concept_counts.items()
        if count == 0
    ]

    episode_counts = list(concept_episode_counts.values())
    concept_counts = list(episode_concept_counts.values())

    return {
        "avg_episodes_per_concept": (
            sum(episode_counts) / len(episode_counts) if episode_counts else 0
        ),
        "max_episodes_per_concept": max(episode_counts) if episode_counts else 0,
        "min_episodes_per_concept": min(episode_counts) if episode_counts else 0,
        "episodes_with_concepts": (
            sum(1 for c in concept_counts if c > 0) / len(concept_counts)
            if concept_counts else 0
        ),
        "avg_concepts_per_episode": (
            sum(concept_counts) / len(concept_counts) if concept_counts else 0
        ),
        "orphan_episode_count": len(orphan_episodes),
        "orphan_episodes": orphan_episodes
    }


if __name__ == "__main__":
    print("Instantiation Scoring Module")
    print("=" * 40)
    print()
    print("Key functions:")
    print("  score_single_pair() - Score one concept-episode pair")
    print("  score_concept_episodes_batch() - Batch scoring")
    print("  create_instantiation_links() - Create all C→E links")
    print("  compute_concept_coverage() - Coverage statistics")
