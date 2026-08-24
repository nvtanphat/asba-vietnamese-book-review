from ml.models.registry import MODEL_REGISTRY

def test_expected_models_registered():
    assert set(MODEL_REGISTRY)=={'logistic','linear_svm','textcnn','bilstm','phobert','xlmr','mdeberta','vit5'}
