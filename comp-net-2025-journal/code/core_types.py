"""
Hierarchical Cognitive Memory Networks: Core Type Definitions

Node Types:
    - Episodic (E): Individual conversations (specific instances)
    - Concept (C): Abstract principles extracted from episode clusters

Link Types:
    - E—E (Similarity): Embedding cosine similarity
    - C→E (Instantiation): Concept exemplified by episode
    - C—C (Association): Concepts with overlapping episode sets
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class AbstractionLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class EpisodicNode:
    """A conversation — specific instance (episodic memory)."""
    id: str
    title: str
    content: str
    timestamp: Optional[str] = None
    community_id: Optional[int] = None  # From conference paper clustering
    embedding: Optional[list[float]] = None

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, EpisodicNode) and self.id == other.id


@dataclass
class ConceptNode:
    """An abstract principle — semantic memory extracted via LLM."""
    id: str
    name: str
    description: str
    source_community: Optional[int] = None  # Community it was extracted from
    abstraction_level: AbstractionLevel = AbstractionLevel.MEDIUM
    example_episodes: list[str] = field(default_factory=list)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, ConceptNode) and self.id == other.id


@dataclass
class SimilarityLink:
    """E—E link: Embedding cosine similarity between episodes."""
    episode1_id: str
    episode2_id: str
    weight: float  # Cosine similarity in [θ, 1.0]

    def __post_init__(self):
        # Normalize order for consistency
        if self.episode1_id > self.episode2_id:
            self.episode1_id, self.episode2_id = self.episode2_id, self.episode1_id


@dataclass
class InstantiationLink:
    """C→E link: Concept instantiated by episode (directed)."""
    concept_id: str
    episode_id: str
    score: float = 1.0  # LLM judgment score (0-1 normalized, or 0-3 raw)


@dataclass
class AssociationLink:
    """C—C link: Concepts with overlapping episode sets."""
    concept1_id: str
    concept2_id: str
    weight: float  # Jaccard coefficient in [γ, 1.0]
    shared_episodes: list[str] = field(default_factory=list)

    def __post_init__(self):
        # Normalize order for consistency
        if self.concept1_id > self.concept2_id:
            self.concept1_id, self.concept2_id = self.concept2_id, self.concept1_id


@dataclass
class HiddenConnection:
    """
    A pair of episodes that are concept-connected but NOT directly E—E linked.

    These reveal cognitive structure invisible to embedding similarity.
    """
    episode1_id: str
    episode2_id: str
    shared_concepts: list[str]
    strength: float = 0.0  # HCS metric

    def __post_init__(self):
        if self.episode1_id > self.episode2_id:
            self.episode1_id, self.episode2_id = self.episode2_id, self.episode1_id


# Default thresholds (from plan)
DEFAULT_SIMILARITY_THRESHOLD = 0.9    # θ: E—E links
DEFAULT_INSTANTIATION_THRESHOLD = 2   # τ: C→E links (on 0-3 scale)
DEFAULT_ASSOCIATION_THRESHOLD = 0.1   # γ: C—C links (Jaccard)
