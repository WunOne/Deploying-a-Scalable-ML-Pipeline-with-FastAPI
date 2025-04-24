import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import (train_model, compute_model_metrics)
from train_model import X_train, y_train


# TODO: add necessary import

def testTrainAlgorithm():
    """
    This test is designed to ensure the model is trained using the correct algorithm, RandomForestClassifier.
    """
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier), f"The incorrect training algorithm was used: {type(model)}"


# TODO: implement the second test. Change the function name and input as needed
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

# TODO: implement the third test. Change the function name and input as needed
def testMetricType():
    """
    # This test is designed to determine if the returned metrics are of the correct type, float.    """

    y_true = np.array([0,1,1,0])
    y_pred = np.array([0,1,1,0])
    p, r, fb = compute_model_metrics(y_true, y_pred)

    assert isinstance(p, float), "Precision did not return a float"
    assert isinstance(r, float), "Recall score did not return a float"
    assert isinstance(fb, float), "fBeta did not return a float"