import numpy as np
from ml.models.sklearn_multitask import SklearnMultiTaskABSA

def _synthetic():
    texts=[];ys=[]
    words=['bad','neutral','good']
    for i in range(96):
        s=i%3
        aspects=[(i+j)%4 for j in range(6)]
        texts.append(f"{words[s]} review token{i%12} " + ' '.join(f"a{j}_{v}" for j,v in enumerate(aspects)))
        ys.append([s,*aspects])
    return texts,np.asarray(ys)

def test_logistic_probability_contract():
    x, y = _synthetic()
    for loss in ("ce", "focal"):
        m = SklearnMultiTaskABSA(
            "logistic",
            "logistic",
            {
                "loss_type": loss,
                "focal_gamma": 2.0,
                "focal_rounds": 2,
                "C": 1.0,
                "max_iter": 200,
                "word_tfidf": {"min_df": 1, "max_features": 1000},
                "char_tfidf": {"min_df": 1, "max_features": 1000},
            },
        )
        m.fit(x, y)
        p = m.predict_proba(x[:5])
        assert len(p) == 7
        assert p[0].shape == (5, 3)
        assert all(z.shape == (5, 4) for z in p[1:])
        assert all(np.allclose(z.sum(1), 1, atol=1e-5) for z in p)

