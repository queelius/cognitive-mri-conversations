"""Tests of mathematical invariants, independent from empirical outcomes."""
from itertools import permutations
import numpy as np
import pytest
from audit_checks import (expected_vocabulary, graph_metrics, margins, projection,
                          randomize, role_entropy, trade, vocabulary)


@pytest.mark.parametrize("rows", [
    [{0, 1}, {1, 2}, {2, 3}], [{0}, {1, 2, 3}, set()],
    [{0, 1}, {0, 1}], [set(), set()], [{0, 1, 2}, {0}],
])
def test_trade_preserves_every_margin(rows):
    original = margins(rows)
    rng = np.random.default_rng(19)
    randomize(rows, 100, rng)
    assert margins(rows) == original


def test_uniform_trade_can_stay_and_can_change():
    outcomes = set()
    for seed in range(100):
        rows = [{0}, {1}]
        trade(rows, 0, 1, np.random.default_rng(seed))
        outcomes.add(tuple(frozenset(row) for row in rows))
    assert outcomes == {(frozenset({0}), frozenset({1})),
                        (frozenset({1}), frozenset({0}))}


def test_exact_vocabulary_matches_exhaustive_orderings():
    rows = [{0, 1}, {0}, {1, 2}, set()]
    empirical = np.mean([vocabulary(list(order)) for order in permutations(rows)], axis=0)
    np.testing.assert_allclose(expected_vocabulary(rows), empirical, rtol=0, atol=1e-14)


def test_projection_is_union_of_episode_cliques():
    rows = [{0, 1, 2}, {2, 3}, {4}]
    graph = projection(rows)
    assert set(map(frozenset, graph.edges())) == {
        frozenset({0, 1}), frozenset({0, 2}), frozenset({1, 2}), frozenset({2, 3})}
    assert graph.degree(4) == 0
    stats = graph_metrics([{0, 1, 2}, {2, 3, 4}])
    assert stats.total_nodes == 5 and stats.total_edges == 6
    assert stats.clustering == pytest.approx(13 / 15)


def test_alternation_has_zero_conditional_entropy():
    assert role_entropy([0, 1] * 10) == 0
    assert role_entropy([0, 0, 1, 0, 1, 1]) > 0
    assert role_entropy([]) == 0
