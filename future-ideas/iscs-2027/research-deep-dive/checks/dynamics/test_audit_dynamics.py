"""Scientific regression checks; source projects remain untouched."""
import json

import numpy as np
import pytest

import audit_dynamics as audit


def test_exact_exchangeable_counterexample():
    points = np.array([[1., 0.], [0., 1.], [-1., 0.]])
    result = audit.exchangeable_triple(points)
    assert result["mean_turning_cosine"] == pytest.approx(-np.sqrt(2) / 3)
    assert result["ratio_of_expectations"] == pytest.approx(-0.5)
    assert abs(result["mean_turning_cosine"] + 0.5) > 0.02


def test_equilateral_special_case_does_equal_minus_half():
    theta = np.arange(3) * 2 * np.pi / 3
    points = np.column_stack([np.cos(theta), np.sin(theta)])
    assert audit.exchangeable_triple(points)["mean_turning_cosine"] == pytest.approx(-0.5)


def test_zero_padding_cannot_rescue_claim():
    points = np.array([[1., 0.], [0., 1.], [-1., 0.]])
    padded = np.pad(points, ((0, 0), (0, 254)))
    assert audit.exchangeable_triple(padded) == audit.exchangeable_triple(points)


def test_loocv_does_not_choose_using_held_out_case():
    scores = np.array([[0.9, 0.1], [0.9, 0.1], [0.0, 1.0]])
    result = audit.leave_one_out(scores, ["a", "b"])
    assert result["chosen_labels"] == ["b", "b", "a"]
    np.testing.assert_allclose(result["held_out_scores"], [0.1, 0.1, 0.0])


def test_prefix_scores_are_conversation_weighted_and_skip_first_prefix():
    saved = {"lens_means_real": {"uniform": 0., "uniform+mass": 0.}, "per_case": {
        "one": {"archetype": "focused", "K": 2, "sims": {"uniform": [1., 0.2], "uniform+mass": [1., 0.3]}},
        "two": {"archetype": "focused", "K": 4, "sims": {"uniform": [1., 0.8, 0.8, 0.8], "uniform+mass": [1., 0.9, 0.9, 0.9]}},
        "synthetic_one": {"archetype": "synthetic_focused", "K": 2, "sims": {"uniform": [1., 0.], "uniform+mass": [1., 0.]}}
    }}
    scores, labels, identifiers = audit.prefix_score_matrix(saved)
    assert labels == ["uniform", "uniform+mass"]
    assert identifiers == ["one", "two"]
    np.testing.assert_allclose(scores.mean(axis=0), [0.5, 0.6])


def test_prediction_scores_use_prior_messages_only():
    points = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
    scores = audit.prediction_scores(points, np.array([0., 1.]), epsilon=0.)
    np.testing.assert_allclose(scores[:, 0], [0., 0., 0.], atol=1e-15)
    assert scores[1, 1] == pytest.approx(-1 / np.sqrt(2))
    modified = points.copy()
    modified[-1] = [1., 0.]
    np.testing.assert_array_equal(scores[:-1], audit.prediction_scores(modified, np.array([0., 1.]), epsilon=0.)[:-1])


def test_selected_alpha_does_not_use_test_targets():
    rng = np.random.default_rng(5)
    points = audit.unit_rows(rng.normal(size=(40, 8)))
    modified = points.copy()
    modified[20:] = audit.unit_rows(rng.normal(size=(20, 8)))
    grid = np.array([0.1, 0.5, 0.9])
    first = audit.chronological_evaluation(points, grid, fixed_alpha=0.5)
    second = audit.chronological_evaluation(modified, grid, fixed_alpha=0.5)
    assert first["selected_alpha"] == second["selected_alpha"]
    assert first["n_train_pairs"] == 19
    assert first["n_test_pairs"] == 20


def test_mean_difference_bootstrap_is_paired():
    result = audit.mean_interval(np.ones(12) * 0.125, np.random.default_rng(3), draws=100)
    assert result["mean"] == pytest.approx(0.125)
    assert result["ci95"] == pytest.approx([0.125, 0.125])


def test_calculus_identities_and_counterexamples():
    result = audit.calculus_checks(20261015)
    assert result["leaky_recurrence_identity_error"] < 1e-12
    assert result["leaky_force_equals_increment_error"] > 0.1
    assert result["pure_sum_force_equals_increment_error"] < 1e-12
    assert result["normalized_force_equals_increment_error"] > 0.1
    assert result["mean_velocity_endpoint_identity_error"] < 1e-12
    assert result["mean_second_difference_boundary_identity_error"] < 1e-12
    assert result["unit_adjacent_cosine_identity_error"] < 1e-12
    assert result["velocity_gram_rigid_transform_error"] < 1e-12
    assert result["normalized_trajectory_translation_error"] > 0.1
    assert result["straight_path_second_difference_norm"] == pytest.approx(1.)
    assert result["straight_path_curvature_radius_is_infinite"] is True
    assert result["uniform_reordering_error"] < 1e-12
    assert result["mass_weighted_reordering_error"] < 1e-12
    assert result["exponential_reordering_error"] > 0.1


def test_archived_prefix_integration():
    source = audit.ROOT / "papers/embedding-dynamics/experiments/results/exp04_prefix.json"
    saved = json.loads(source.read_text())
    result = audit.prefix_check(saved, seed=20261015, draws=100)
    assert result["n_real_conversations"] == 12
    assert result["n_eligible_prefixes"] == 305
    assert result["n_lenses_selected_over"] == 26
    assert result["max_discrepancy_from_archived_lens_means"] < 1e-12
    assert result["best_mean"] == pytest.approx(saved["gate"]["best_lens_mean"])
    assert sum(result["loocv"]["selected_lens_counts"].values()) == 12
    assert result["exact_retrieval_prediction_met"] is False
    assert result["order_sensitivity_tested"] is False


def test_completed_pipeline_artifact_has_no_silent_failure():
    path = audit.HERE / "results.json"
    if not path.exists():
        pytest.skip("Run audit_dynamics.py before the artifact integration check")
    result = json.loads(path.read_text())
    for stage in ["turning", "calculus", "prefix", "prediction"]:
        assert not result[stage].get("failed", False)
        assert result[stage]["runtime_seconds"] >= 0
    assert result["prediction"]["n_selected"] == result["prediction"]["n_loaded"]
    assert result["prediction"]["n_loaded"] <= 240
    assert sum(group["n"] for group in result["prediction"]["by_source"].values()) == result["prediction"]["n_loaded"]
    assert all(result["source_integrity_after"].values())


def test_results_export_aggregates_not_identifiers_or_vectors():
    path = audit.HERE / "results.json"
    if not path.exists():
        pytest.skip("Run audit_dynamics.py before the artifact privacy check")
    forbidden_keys = {"cid", "conversation_id", "message_id", "title", "content", "embedding", "messages"}

    def inspect(value):
        if isinstance(value, dict):
            assert not forbidden_keys.intersection(value)
            for item in value.values():
                inspect(item)
        elif isinstance(value, list):
            for item in value:
                inspect(item)

    inspect(json.loads(path.read_text()))
