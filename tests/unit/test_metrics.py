import numpy as np
from ml.evaluation.metrics import evaluate_predictions

def test_perfect_predictions_score_one():
    y=np.array([[0,0,3,2,3,1,3],[1,3,1,3,0,3,2],[2,2,3,3,3,2,0]])
    m=evaluate_predictions(y,y)
    assert m["f1_sentiment"] == 1.0
    assert m["f1_aspect_present"] == 1.0
    assert m["f1_combined"] == 1.0

def test_hallucinated_aspect_is_penalized():
    y=np.array([[2,3,3,3,3,3,3],[0,0,3,3,3,3,3],[1,3,1,3,3,3,3],[2,2,3,3,3,3,3]])
    pred=y.copy();pred[:,6]=2  # hallucinate service on every review
    assert evaluate_predictions(y,pred)["f1_combined"] < evaluate_predictions(y,y)["f1_combined"]
