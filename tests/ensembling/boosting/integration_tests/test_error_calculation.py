from src.ensembling.boosting.danboost import DanBoost
from src.base_learners.tensorflow import MLP
import numpy as np
import pytest

def test_calc_error_raises_value_error():
    """
    Test that calc_error raises ValueError for unsupported error types.
    """
    ensemble = DanBoost(base_estimator=MLP, n_estimators=1)
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    ensemble.fit_ensemble(
        X=X,
        y=y,
        error_type="unsupported_error_type"
    )

    with pytest.raises(ValueError):
        ensemble.calc_error(X, y, error_type="unsupported_error_type")
