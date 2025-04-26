import pytest
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from ml.model import (
    train_model,
    compute_model_metrics,
    save_model,
    load_model
)
from train_model import X_train, y_train

def testTrainAlgorithm():
    """
    This test is designed to ensure the model is trained using the correct algorithm, RandomForestClassifier.
    """
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier), f"The incorrect training algorithm was used: {type(model)}"

def testMetricValues():
    """
    # This test is designed to determine if the metrics return a value between 0 and 1.
    """
    y_true = np.array([0,1,1,0])
    y_pred = np.array([0,1,1,0])
    p, r, fb = compute_model_metrics(y_true, y_pred)

    assert 0 <= p <= 1, "Precision is outside of expected value bounds"
    assert 0 <= r <= 1, "Recall score is outside of expected value bounds"
    assert 0 <= fb <= 1, "fBeta is outside of expected value bounds"

def testSaveLoad(tmp_path):
    """
    # This test is designed to validate save_model and load_model function correctly.
    """

    X = np.array([[1,3], [5,7], [9, 11]])
    y = np.array([0,1,0])

    model = train_model(X, y)

    temp = tmp_path / "rf.pkl"

    save_model(model, str(temp))

    loaded = load_model(str(temp))

    assert isinstance(loaded, RandomForestClassifier), f"The loaded model is using the incorrect training model: {type(loaded)}"

    np.testing.assert_array_equal(model.predict(X), loaded.predict(X))