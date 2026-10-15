"""Tests record the defect without modifying the read-only library."""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from shape_counterexample import check


def test_positive_scalar_cosine_is_constant():
    values = np.array([.001, .01, .1, 1., 2.])[:, None]
    np.testing.assert_allclose(cosine_similarity(values), 1., atol=1e-15)


def test_library_shape_cannot_distinguish_distinct_positive_profiles():
    result = check()
    assert result["all_profile_values_strictly_positive"]
    assert result["profile_euclidean_difference"] > .6
    assert result["maximum_absolute_scalar_cosine_cost"] < 1e-15
    assert abs(result["library_shape_distance"]) < 1e-15
