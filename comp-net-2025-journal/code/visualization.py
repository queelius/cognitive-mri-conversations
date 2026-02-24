"""
Multi-Layer Network Visualization

Visualization utilities for hierarchical cognitive memory networks.
Supports multi-layer layouts, community coloring, and export to various formats.
"""

import json
from dataclasses import dataclass
from typing import Optional
import math

from core_types import (
    EpisodicNode, ConceptNode,
    SimilarityLink, InstantiationLink, AssociationLink,
    HiddenConnection
)


@dataclass
class VisualizationConfig:
    """Configuration for network visualization."""
    episodic_color: str = "#4A90D9"  # Blue for episodes
    concept_color: str = "#D94A4A"   # Red for concepts
    ee_link_color: str = "#AAAAAA"   # Gray for E—E
    ce_link_color: str = "#7B68EE"   # Purple for C→E
    cc_link_color: str = "#FF6B6B"   # Light red for C—C
    hidden_link_color: str = "#2ECC71"  # Green for hidden connections

    episodic_size: float = 10.0
    concept_size: float = 20.0

    layer_separation: float = 100.0  # Vertical separation between layers


def export_to_gexf(
    episodes: list[EpisodicNode],
    concepts: list[ConceptNode],
    ee_links: list[SimilarityLink],
    ce_links: list[InstantiationLink],
    cc_links: list[AssociationLink],
    output_path: str,
    config: Optional[VisualizationConfig] = None,
    episode_communities: Optional[dict[str, int]] = None,
    concept_communities: Optional[dict[str, int]] = None
) -> None:
    """
    Export multi-layer network to GEXF format for Gephi.

    Args:
        episodes: List of episodic nodes
        concepts: List of concept nodes
        ee_links: E—E similarity links
        ce_links: C→E instantiation links
        cc_links: C—C association links
        output_path: Path to write GEXF file
        config: Visualization configuration
        episode_communities: Optional community assignments for episodes
        concept_communities: Optional community assignments for concepts
    """
    if config is None:
        config = VisualizationConfig()

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gexf xmlns="http://www.gexf.net/1.2draft" version="1.2">',
        '  <meta>',
        '    <creator>Hierarchical Cognitive Memory Networks</creator>',
        '    <description>Multi-layer episodic-concept network</description>',
        '  </meta>',
        '  <graph mode="static" defaultedgetype="undirected">',
        '    <attributes class="node">',
        '      <attribute id="0" title="layer" type="string"/>',
        '      <attribute id="1" title="community" type="integer"/>',
        '      <attribute id="2" title="label" type="string"/>',
        '    </attributes>',
        '    <attributes class="edge">',
        '      <attribute id="0" title="link_type" type="string"/>',
        '    </attributes>',
        '    <nodes>'
    ]

    # Add episodic nodes
    for ep in episodes:
        comm = episode_communities.get(ep.id, 0) if episode_communities else 0
        label = ep.title.replace('"', '&quot;').replace('&', '&amp;')[:50]
        lines.append(f'      <node id="{ep.id}" label="{label}">')
        lines.append(f'        <attvalues>')
        lines.append(f'          <attvalue for="0" value="episodic"/>')
        lines.append(f'          <attvalue for="1" value="{comm}"/>')
        lines.append(f'          <attvalue for="2" value="{label}"/>')
        lines.append(f'        </attvalues>')
        lines.append(f'        <viz:color r="74" g="144" b="217"/>')
        lines.append(f'        <viz:size value="{config.episodic_size}"/>')
        lines.append(f'        <viz:position y="0"/>')
        lines.append(f'      </node>')

    # Add concept nodes
    for c in concepts:
        comm = concept_communities.get(c.id, 0) if concept_communities else 0
        label = c.name.replace('"', '&quot;').replace('&', '&amp;')[:50]
        lines.append(f'      <node id="{c.id}" label="{label}">')
        lines.append(f'        <attvalues>')
        lines.append(f'          <attvalue for="0" value="concept"/>')
        lines.append(f'          <attvalue for="1" value="{comm}"/>')
        lines.append(f'          <attvalue for="2" value="{label}"/>')
        lines.append(f'        </attvalues>')
        lines.append(f'        <viz:color r="217" g="74" b="74"/>')
        lines.append(f'        <viz:size value="{config.concept_size}"/>')
        lines.append(f'        <viz:position y="{config.layer_separation}"/>')
        lines.append(f'      </node>')

    lines.append('    </nodes>')
    lines.append('    <edges>')

    edge_id = 0

    # Add E—E edges
    for link in ee_links:
        lines.append(
            f'      <edge id="{edge_id}" source="{link.episode1_id}" '
            f'target="{link.episode2_id}" weight="{link.weight}">'
        )
        lines.append('        <attvalues>')
        lines.append('          <attvalue for="0" value="similarity"/>')
        lines.append('        </attvalues>')
        lines.append('      </edge>')
        edge_id += 1

    # Add C→E edges (directed in concept, but GEXF may not distinguish)
    for link in ce_links:
        lines.append(
            f'      <edge id="{edge_id}" source="{link.concept_id}" '
            f'target="{link.episode_id}" weight="{link.score}">'
        )
        lines.append('        <attvalues>')
        lines.append('          <attvalue for="0" value="instantiation"/>')
        lines.append('        </attvalues>')
        lines.append('      </edge>')
        edge_id += 1

    # Add C—C edges
    for link in cc_links:
        lines.append(
            f'      <edge id="{edge_id}" source="{link.concept1_id}" '
            f'target="{link.concept2_id}" weight="{link.weight}">'
        )
        lines.append('        <attvalues>')
        lines.append('          <attvalue for="0" value="association"/>')
        lines.append('        </attvalues>')
        lines.append('      </edge>')
        edge_id += 1

    lines.append('    </edges>')
    lines.append('  </graph>')
    lines.append('</gexf>')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def export_to_json(
    episodes: list[EpisodicNode],
    concepts: list[ConceptNode],
    ee_links: list[SimilarityLink],
    ce_links: list[InstantiationLink],
    cc_links: list[AssociationLink],
    output_path: str,
    episode_communities: Optional[dict[str, int]] = None,
    concept_communities: Optional[dict[str, int]] = None
) -> None:
    """
    Export multi-layer network to JSON format for D3.js or custom visualization.

    Creates a JSON structure with separate arrays for each layer and link type.
    """
    data = {
        "nodes": {
            "episodes": [
                {
                    "id": ep.id,
                    "title": ep.title,
                    "layer": "episodic",
                    "community": episode_communities.get(ep.id) if episode_communities else None
                }
                for ep in episodes
            ],
            "concepts": [
                {
                    "id": c.id,
                    "name": c.name,
                    "description": c.description,
                    "layer": "concept",
                    "community": concept_communities.get(c.id) if concept_communities else None
                }
                for c in concepts
            ]
        },
        "links": {
            "similarity": [
                {
                    "source": link.episode1_id,
                    "target": link.episode2_id,
                    "weight": link.weight,
                    "type": "E-E"
                }
                for link in ee_links
            ],
            "instantiation": [
                {
                    "source": link.concept_id,
                    "target": link.episode_id,
                    "score": link.score,
                    "type": "C-E"
                }
                for link in ce_links
            ],
            "association": [
                {
                    "source": link.concept1_id,
                    "target": link.concept2_id,
                    "weight": link.weight,
                    "shared_episodes": link.shared_episodes,
                    "type": "C-C"
                }
                for link in cc_links
            ]
        },
        "metadata": {
            "num_episodes": len(episodes),
            "num_concepts": len(concepts),
            "num_ee_links": len(ee_links),
            "num_ce_links": len(ce_links),
            "num_cc_links": len(cc_links)
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def export_hidden_connections_json(
    hidden_connections: list[HiddenConnection],
    episodes: list[EpisodicNode],
    concepts: list[ConceptNode],
    output_path: str
) -> None:
    """
    Export hidden connections to JSON for visualization.

    Includes episode and concept details for richer visualization.
    """
    episode_map = {ep.id: ep for ep in episodes}
    concept_map = {c.id: c for c in concepts}

    data = {
        "hidden_connections": [
            {
                "episode1": {
                    "id": hc.episode1_id,
                    "title": episode_map.get(hc.episode1_id, EpisodicNode(
                        id=hc.episode1_id, title="Unknown", content=""
                    )).title
                },
                "episode2": {
                    "id": hc.episode2_id,
                    "title": episode_map.get(hc.episode2_id, EpisodicNode(
                        id=hc.episode2_id, title="Unknown", content=""
                    )).title
                },
                "shared_concepts": [
                    {
                        "id": cid,
                        "name": concept_map.get(cid, ConceptNode(
                            id=cid, name="Unknown", description=""
                        )).name
                    }
                    for cid in hc.shared_concepts
                ],
                "strength": hc.strength
            }
            for hc in hidden_connections
        ],
        "summary": {
            "total_hidden_connections": len(hidden_connections),
            "avg_strength": (
                sum(hc.strength for hc in hidden_connections) / len(hidden_connections)
                if hidden_connections else 0.0
            ),
            "max_strength": (
                max(hc.strength for hc in hidden_connections)
                if hidden_connections else 0.0
            )
        }
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def generate_tikz_multilayer(
    episodes: list[EpisodicNode],
    concepts: list[ConceptNode],
    ee_links: list[SimilarityLink],
    ce_links: list[InstantiationLink],
    cc_links: list[AssociationLink],
    max_nodes: int = 20,
    config: Optional[VisualizationConfig] = None
) -> str:
    """
    Generate TikZ code for a multi-layer network diagram.

    Suitable for LaTeX papers. Returns TikZ code as a string.
    """
    if config is None:
        config = VisualizationConfig()

    # Limit nodes for readability
    display_episodes = episodes[:max_nodes]
    display_concepts = concepts[:max_nodes]

    episode_ids = {ep.id for ep in display_episodes}
    concept_ids = {c.id for c in display_concepts}

    # Filter links to displayed nodes
    display_ee = [
        l for l in ee_links
        if l.episode1_id in episode_ids and l.episode2_id in episode_ids
    ][:30]

    display_ce = [
        l for l in ce_links
        if l.concept_id in concept_ids and l.episode_id in episode_ids
    ][:50]

    display_cc = [
        l for l in cc_links
        if l.concept1_id in concept_ids and l.concept2_id in concept_ids
    ][:20]

    lines = [
        r'\begin{tikzpicture}[',
        r'  episode/.style={circle, fill=blue!30, minimum size=8pt},',
        r'  concept/.style={circle, fill=red!30, minimum size=12pt},',
        r'  ee/.style={gray, opacity=0.5},',
        r'  ce/.style={purple, opacity=0.6, dashed},',
        r'  cc/.style={red!60, opacity=0.7}',
        r']',
        r'',
        r'% Episodic layer (bottom)',
    ]

    # Position episodes in a circle at y=0
    n_ep = len(display_episodes)
    for i, ep in enumerate(display_episodes):
        angle = 360 * i / n_ep
        x = 4 * math.cos(math.radians(angle))
        y = 4 * math.sin(math.radians(angle))
        safe_id = ep.id.replace('_', '')
        lines.append(f'\\node[episode] ({safe_id}) at ({x:.2f},{y:.2f}) {{}};')

    lines.append('')
    lines.append(r'% Concept layer (top)')

    # Position concepts in a smaller circle at y=layer_separation
    n_c = len(display_concepts)
    y_offset = config.layer_separation / 10  # Scale for TikZ
    for i, c in enumerate(display_concepts):
        angle = 360 * i / n_c
        x = 2 * math.cos(math.radians(angle))
        y = 2 * math.sin(math.radians(angle)) + y_offset
        safe_id = c.id.replace('_', '')
        lines.append(f'\\node[concept] ({safe_id}) at ({x:.2f},{y:.2f}) {{}};')

    lines.append('')
    lines.append(r'% E—E links')
    for link in display_ee:
        id1 = link.episode1_id.replace('_', '')
        id2 = link.episode2_id.replace('_', '')
        lines.append(f'\\draw[ee] ({id1}) -- ({id2});')

    lines.append('')
    lines.append(r'% C→E links')
    for link in display_ce:
        cid = link.concept_id.replace('_', '')
        eid = link.episode_id.replace('_', '')
        lines.append(f'\\draw[ce] ({cid}) -- ({eid});')

    lines.append('')
    lines.append(r'% C—C links')
    for link in display_cc:
        id1 = link.concept1_id.replace('_', '')
        id2 = link.concept2_id.replace('_', '')
        lines.append(f'\\draw[cc] ({id1}) -- ({id2});')

    lines.append('')
    lines.append(r'% Layer labels')
    lines.append(r'\node[anchor=east] at (-5,0) {\textbf{Episodic}};')
    lines.append(f'\\node[anchor=east] at (-5,{y_offset:.1f}) {{\\textbf{{Concept}}}};')

    lines.append(r'\end{tikzpicture}')

    return '\n'.join(lines)


def create_layer_summary_table(
    episodes: list[EpisodicNode],
    concepts: list[ConceptNode],
    ee_links: list[SimilarityLink],
    ce_links: list[InstantiationLink],
    cc_links: list[AssociationLink]
) -> str:
    """
    Create a LaTeX table summarizing the multi-layer network.
    """
    n_ep = len(episodes)
    n_c = len(concepts)
    n_ee = len(ee_links)
    n_ce = len(ce_links)
    n_cc = len(cc_links)

    max_ee = n_ep * (n_ep - 1) // 2
    max_cc = n_c * (n_c - 1) // 2
    max_ce = n_c * n_ep

    ee_density = n_ee / max_ee if max_ee > 0 else 0
    cc_density = n_cc / max_cc if max_cc > 0 else 0
    ce_density = n_ce / max_ce if max_ce > 0 else 0

    table = r"""
\begin{table}[h]
\centering
\caption{Multi-layer network summary}
\label{tab:multilayer-summary}
\begin{tabular}{lrrr}
\toprule
\textbf{Layer/Link Type} & \textbf{Count} & \textbf{Max Possible} & \textbf{Density} \\
\midrule
Episodic nodes (E) & """ + str(n_ep) + r""" & -- & -- \\
Concept nodes (C) & """ + str(n_c) + r""" & -- & -- \\
\midrule
E—E similarity & """ + str(n_ee) + r""" & """ + str(max_ee) + r""" & """ + f"{ee_density:.4f}" + r""" \\
C→E instantiation & """ + str(n_ce) + r""" & """ + str(max_ce) + r""" & """ + f"{ce_density:.4f}" + r""" \\
C—C association & """ + str(n_cc) + r""" & """ + str(max_cc) + r""" & """ + f"{cc_density:.4f}" + r""" \\
\bottomrule
\end{tabular}
\end{table}
"""
    return table


if __name__ == "__main__":
    print("Multi-Layer Network Visualization Module")
    print("=" * 40)
    print()
    print("Key functions:")
    print("  export_to_gexf() - Export for Gephi")
    print("  export_to_json() - Export for D3.js / web visualization")
    print("  export_hidden_connections_json() - Export hidden connections")
    print("  generate_tikz_multilayer() - Generate TikZ code for LaTeX")
    print("  create_layer_summary_table() - Generate LaTeX summary table")
