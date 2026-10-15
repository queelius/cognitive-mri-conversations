"""Controls for bootstrap-refit exclusion and resample weighting."""
import numpy as np
import pytest

from audit_dynamics import leave_one_out
from selection_uncertainty import refit_from_counts, sample_valid_counts


def test_excludes_every_copy_of_held_out_identity():
    scores = np.array([[1., 0., .1], [0., .5, .1]])
    result = refit_from_counts(scores, np.array([[2, 1]]), baseline_index=2)
    np.testing.assert_array_equal(result["selected_indices"], [[1, 0]])
    np.testing.assert_allclose(result["mean_gains"], [-.1])
    # Removing only one of the first case's two copies would select lens 0
    # for that held-out identity, leaking its score and yielding +.5667.


def test_training_and_evaluation_preserve_multiplicity():
    scores = np.array([[1., 0., 0.], [0., .6, 0.], [.4, .2, 0.]])
    result = refit_from_counts(scores, np.array([[1, 2, 1]]), baseline_index=2)
    np.testing.assert_array_equal(result["selected_indices"], [[1, 0, 1]])
    np.testing.assert_allclose(result["mean_gains"], [.05])


def test_unobserved_case_does_not_enter_training_or_evaluation():
    scores = np.array([[1., 0., .1], [0., .5, .1], [0., 100., 0.]])
    with_unobserved = refit_from_counts(scores, np.array([[2, 1, 0]]), baseline_index=2)
    without_unobserved = refit_from_counts(scores[:2], np.array([[2, 1]]), baseline_index=2)
    np.testing.assert_allclose(with_unobserved["mean_gains"], without_unobserved["mean_gains"])


def test_all_one_counts_match_original_loocv():
    scores = np.array([[.9, .1, .2], [.9, .1, .2], [0., 1., .2]])
    original = leave_one_out(scores, ["a", "b", "baseline"])
    result = refit_from_counts(scores, np.ones((1, 3), dtype=int), baseline_index=2)
    expected = np.mean(np.array(original["held_out_scores"]) - scores[:, 2])
    assert result["mean_gains"][0] == pytest.approx(expected)


def test_single_unique_draw_is_rejected_before_fitting():
    scores = np.array([[1., 0.], [0., 1.]])
    with pytest.raises(ValueError, match="two distinct"):
        refit_from_counts(scores, np.array([[2, 0]]), baseline_index=0)


def test_sampler_redraws_and_records_degenerate_draws():
    class CountsFixture:
        calls = 0

        def multinomial(self, n, probabilities, size):
            self.calls += 1
            assert n == 2
            np.testing.assert_allclose(probabilities, [.5, .5])
            if self.calls == 1:
                assert size == 2
                return np.array([[2, 0], [1, 1]])
            assert size == 1
            return np.array([[1, 1]])

    counts, rejected = sample_valid_counts(2, 2, CountsFixture())
    np.testing.assert_array_equal(counts, [[1, 1], [1, 1]])
    assert rejected == 1


def test_vectorized_result_matches_direct_repeated_row_reference():
    scores = np.array([[.8, .2, .1], [.1, .7, .2], [.6, .5, .3], [.2, .4, .2]])
    counts = np.array([[2, 1, 0, 1], [0, 2, 1, 1], [1, 1, 1, 1]])
    result = refit_from_counts(scores, counts, baseline_index=2)
    expected = []
    for weights in counts:
        sample = np.repeat(np.arange(len(scores)), weights)
        gains = []
        for identity in sample:
            training = sample[sample != identity]
            choice = scores[training].mean(axis=0).argmax()
            gains.append(scores[identity, choice] - scores[identity, 2])
        expected.append(np.mean(gains))
    np.testing.assert_allclose(result["mean_gains"], expected)


def test_original_lens_order_breaks_exact_ties():
    scores = np.array([[.8, .1], [.5, .5]])
    result = refit_from_counts(scores, np.array([[1, 1]]), baseline_index=1)
    assert result["selected_indices"][0, 0] == 0


@pytest.mark.parametrize("counts", [np.array([[1., .5]]), np.array([[-1, 3]]), np.array([[0, 0]])])
def test_invalid_counts_are_rejected(counts):
    with pytest.raises(ValueError):
        refit_from_counts(np.array([[1., 0.], [0., 1.]]), counts, baseline_index=0)
