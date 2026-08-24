"""Migrated from 03_1_baseline_5fold_cv.ipynb.

This file preserves the original experiment code for audit/reproducibility.
The production benchmark uses the modular ml/ package instead.
"""


# %% [code cell 1]
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer, StandardScaler, Normalizer
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

RANDOM_STATE = 42
ASPECT_COLS = [
    'as_content',
    'as_physical',
    'as_price',
    'as_packaging',
    'as_delivery',
    'as_service',
]
SENTIMENT_COL = 'sentiment'
TEXT_COL = 'content'
ID_COL = 'review_id'
ABSENT_LABEL = 3
USE_VN_SEGMENTATION = True

BEST_CFG = {
    'name': 'best_lr_saga_c8_min4__none',
    'model': 'logreg_saga',
    'C': 8.0,
    'min_df': 4,
    'max_df': 0.95,
    'char_max_df': 0.95,
    'neutral_class_boost': 1.35,
    'neutral_weight_cap_ratio': 2.5,
    'presence_threshold_neutral_weight': 0.35,
    'dim_reduce': 'none',
}


def find_project_root() -> Path:
    cwd = Path.cwd().resolve()
    for c in [cwd, *cwd.parents]:
        has_notebook_dir = (c / 'notebooks').exists()
        has_interim = (c / 'data' / 'interim' / 'raw_train' / 'train.json').exists()
        has_processed = (c / 'data' / 'processed' / 'train_clean.json').exists()
        if has_notebook_dir and (has_interim or has_processed):
            return c
    return cwd



ROOT = find_project_root()
DATA_SOURCES = {
    'interim': {
        'train': ROOT / 'data' / 'interim' / 'raw_train' / 'train.json',
        'val': ROOT / 'data' / 'interim' / 'raw_val' / 'val.json',
        'test': ROOT / 'data' / 'interim' / 'raw_test' / 'test.json',
    },
    'processed': {
        'train': ROOT / 'data' / 'processed' / 'train_clean.json',
        'val': ROOT / 'data' / 'processed' / 'val_clean.json',
        'test': ROOT / 'data' / 'processed' / 'test_clean.json',
    },
}

print('ROOT:', ROOT)
print('Available sources:', list(DATA_SOURCES))


# %% [code cell 2]
def _build_vn_segmenter():
    if not USE_VN_SEGMENTATION:
        return None, 'disabled'

    try:
        from underthesea import word_tokenize as uts_word_tokenize

        def segment_fn(text: str) -> str:
            return uts_word_tokenize(text, format='text')

        return segment_fn, 'underthesea'
    except Exception:
        return None, 'fallback_raw_text'


def _validate_schema(df: pd.DataFrame, name: str) -> None:
    required = {ID_COL, TEXT_COL, SENTIMENT_COL, *ASPECT_COLS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f'{name} missing columns: {missing}')


def _prepare_frame(df: pd.DataFrame, name: str, segmenter=None) -> pd.DataFrame:
    _validate_schema(df, name)
    out = df.copy()
    out[TEXT_COL] = out[TEXT_COL].fillna('').astype(str)

    if segmenter is not None:
        out[TEXT_COL] = out[TEXT_COL].map(lambda x: segmenter(x) if x else x)

    out[SENTIMENT_COL] = pd.to_numeric(out[SENTIMENT_COL], errors='coerce')
    out = out.dropna(subset=[SENTIMENT_COL])
    out[SENTIMENT_COL] = out[SENTIMENT_COL].astype(int)
    out = out[out[SENTIMENT_COL].isin([0, 1, 2])].reset_index(drop=True)

    for col in ASPECT_COLS:
        out[col] = pd.to_numeric(out[col], errors='coerce')
        out[col] = out[col].where(out[col].isin([0, 1, 2]), np.nan)

    return out


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    segmenter, segmenter_name = _build_vn_segmenter()

    train = _prepare_frame(pd.read_json(TRAIN_PATH), 'train', segmenter=segmenter)
    val = _prepare_frame(pd.read_json(VAL_PATH), 'val', segmenter=segmenter)
    test = _prepare_frame(pd.read_json(TEST_PATH), 'test', segmenter=segmenter)

    print(f'Vietnamese segmenter: {segmenter_name}')
    return train, val, test


def to_aspect_matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame[ASPECT_COLS].fillna(ABSENT_LABEL).astype(int).to_numpy()


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> float:
    _, _, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average='macro',
        zero_division=0,
    )
    return float(f1)


def compute_metrics_absa(
    true_sent: np.ndarray,
    pred_sent: np.ndarray,
    true_aspects: np.ndarray,
    pred_aspects: np.ndarray,
    aspect_cols: list[str],
) -> dict:
    f1_sent = macro_f1(true_sent, pred_sent, labels=[0, 1, 2])

    present_mask = true_aspects.flatten() != ABSENT_LABEL
    f1_asp_present = 0.0
    f1_asp_neutral_present = 0.0
    acc_asp_present = 0.0
    if present_mask.any():
        true_present = true_aspects.flatten()[present_mask]
        pred_present = pred_aspects.flatten()[present_mask]
        f1_asp_present = macro_f1(true_present, pred_present, labels=[0, 1, 2])
        _, _, f1_asp_neutral_present, _ = precision_recall_fscore_support(
            true_present,
            pred_present,
            labels=[1],
            average='macro',
            zero_division=0,
        )
        acc_asp_present = accuracy_score(true_present, pred_present)

    f1_asp_all = macro_f1(true_aspects.flatten(), pred_aspects.flatten(), labels=[0, 1, 2, 3])

    asp_f1s = {}
    asp_neutral_f1s = {}
    for i, col in enumerate(aspect_cols):
        mask_i = true_aspects[:, i] != ABSENT_LABEL
        if mask_i.any():
            true_i = true_aspects[:, i][mask_i]
            pred_i = pred_aspects[:, i][mask_i]
            f1_i = macro_f1(true_i, pred_i, labels=[0, 1, 2])
            _, _, f1_i_neutral, _ = precision_recall_fscore_support(
                true_i,
                pred_i,
                labels=[1],
                average='macro',
                zero_division=0,
            )
        else:
            f1_i = 0.0
            f1_i_neutral = 0.0
        asp_f1s[f'f1_{col}'] = round(f1_i, 4)
        asp_neutral_f1s[f'f1_neutral_{col}'] = round(float(f1_i_neutral), 4)

    f1_combined = 0.5 * f1_sent + 0.5 * f1_asp_present

    metrics = {
        'f1_sentiment': round(f1_sent, 4),
        'f1_aspect_all': round(f1_asp_all, 4),
        'f1_aspect_present': round(f1_asp_present, 4),
        'f1_aspect_neutral_present': round(float(f1_asp_neutral_present), 4),
        'f1_combined': round(f1_combined, 4),
        'acc_sentiment': round(float(accuracy_score(true_sent, pred_sent)), 4),
        'acc_aspect_present': round(float(acc_asp_present), 4),
        'acc_aspect_all': round(float(accuracy_score(true_aspects.flatten(), pred_aspects.flatten())), 4),
        **asp_f1s,
        **asp_neutral_f1s,
    }
    return metrics


# %% [code cell 3]
def _build_meta_features(text_series) -> np.ndarray:
    s = pd.Series(text_series).fillna('').astype(str)
    char_count = s.str.len().to_numpy(dtype=np.float32)
    word_count = s.str.split().str.len().to_numpy(dtype=np.float32)
    exclaim_count = s.str.count('!').to_numpy(dtype=np.float32)
    question_count = s.str.count(r'\?').to_numpy(dtype=np.float32)
    uppercase_ratio = s.map(lambda t: sum(ch.isupper() for ch in t) / max(len(t), 1)).to_numpy(dtype=np.float32)
    return np.vstack([char_count, word_count, exclaim_count, question_count, uppercase_ratio]).T


def _build_text_features(cfg: dict) -> FeatureUnion:
    min_df = cfg['min_df']
    max_df = cfg['max_df']
    char_max_df = cfg.get('char_max_df', max_df)

    return FeatureUnion([
        ('word_tfidf', TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=min_df,
            max_df=max_df,
            lowercase=False,
            sublinear_tf=True,
            strip_accents=None,
        )),
        ('char_tfidf', TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 5),
            min_df=min_df,
            max_df=char_max_df,
            lowercase=False,
            sublinear_tf=True,
            strip_accents=None,
        )),
        ('meta_features', Pipeline([
            ('build', FunctionTransformer(_build_meta_features, validate=False)),
            ('scale', StandardScaler(with_mean=False)),
        ])),
    ])


class SafeSelectKBest(BaseEstimator, TransformerMixin):
    def __init__(self, score_func=chi2, k=20000):
        self.score_func = score_func
        self.k = k

    def fit(self, X, y=None):
        if self.k == 'all':
            k_effective = 'all'
        else:
            k_effective = min(int(self.k), int(X.shape[1]))
        self.selector_ = SelectKBest(score_func=self.score_func, k=k_effective)
        self.selector_.fit(X, y)
        return self

    def transform(self, X):
        return self.selector_.transform(X)


def _build_dim_reducer(cfg: dict):
    method = cfg.get('dim_reduce', 'none')
    if method == 'none':
        return 'passthrough'
    if method == 'chi2':
        return SafeSelectKBest(score_func=chi2, k=cfg.get('chi2_k', 20000))
    if method == 'svd':
        return Pipeline([
            ('svd', TruncatedSVD(n_components=cfg.get('svd_components', 512), random_state=RANDOM_STATE)),
            ('norm', Normalizer(copy=False)),
        ])
    if method == 'chi2_svd':
        return Pipeline([
            ('chi2', SafeSelectKBest(score_func=chi2, k=cfg.get('chi2_k', 20000))),
            ('svd', TruncatedSVD(n_components=cfg.get('svd_components', 512), random_state=RANDOM_STATE)),
            ('norm', Normalizer(copy=False)),
        ])
    if method == 'l1_select':
        l1_estimator = LogisticRegression(
            penalty='l1', solver='saga', C=cfg.get('l1_select_C', 0.5), max_iter=3000, random_state=RANDOM_STATE
        )
        return SelectFromModel(l1_estimator, threshold=cfg.get('l1_threshold', 'median'))
    raise ValueError(f'Unsupported dim_reduce method: {method}')


def _make_classifier(cfg: dict, class_weight='balanced'):
    model_name = cfg.get('model', 'logreg_lbfgs')

    if model_name == 'dummy_majority':
        return DummyClassifier(strategy='most_frequent')
    if model_name == 'linear_svc':
        return LinearSVC(C=cfg['C'], class_weight=class_weight, random_state=RANDOM_STATE)
    if model_name == 'sgd_elasticnet':
        return SGDClassifier(
            loss='log_loss',
            penalty='elasticnet',
            alpha=cfg.get('alpha', 1e-5),
            l1_ratio=cfg.get('l1_ratio', 0.15),
            class_weight=class_weight,
            max_iter=4000,
            tol=1e-3,
            random_state=RANDOM_STATE,
        )

    solver = 'saga' if model_name == 'logreg_saga' else 'lbfgs'
    return LogisticRegression(
        C=cfg['C'], class_weight=class_weight, max_iter=5000, solver=solver, random_state=RANDOM_STATE
    )


def make_pipeline(*, cfg: dict, class_weight='balanced') -> Pipeline:
    return Pipeline([
        ('features', _build_text_features(cfg)),
        ('dim_reduce', _build_dim_reducer(cfg)),
        ('clf', _make_classifier(cfg, class_weight=class_weight)),
    ])


def _fit_dummy_model(dummy_model: DummyClassifier, y_train: np.ndarray) -> DummyClassifier:
    x_stub = np.zeros((len(y_train), 1), dtype=np.float32)
    dummy_model.fit(x_stub, y_train)
    return dummy_model


def _predict_positive_score(model, x_split: pd.Series) -> np.ndarray:
    if isinstance(model, DummyClassifier):
        x_stub = np.zeros((len(x_split), 1), dtype=np.float32)
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(x_stub)
            classes = model.classes_
            if 1 in classes:
                idx = int(np.where(classes == 1)[0][0])
                return probs[:, idx]
            return np.zeros(len(x_split), dtype=float)
        return model.predict(x_stub).astype(float)

    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(x_split)
        classes = model.classes_
        if 1 in classes:
            idx = int(np.where(classes == 1)[0][0])
            return probs[:, idx]
        return np.zeros(len(x_split), dtype=float)

    if hasattr(model, 'decision_function'):
        scores = np.asarray(model.decision_function(x_split), dtype=float)
        if scores.ndim == 2:
            if hasattr(model, 'classes_') and 1 in model.classes_:
                idx = int(np.where(model.classes_ == 1)[0][0])
                scores = scores[:, idx]
            else:
                scores = scores[:, -1]
        scores = np.clip(scores, -30, 30)
        return 1.0 / (1.0 + np.exp(-scores))

    return model.predict(x_split).astype(float)


def _predict_sentiment_labels(x_split: pd.Series, sent_model, sent_fallback: int) -> np.ndarray:
    if sent_model is None or isinstance(sent_model, DummyClassifier):
        return np.full(shape=len(x_split), fill_value=sent_fallback, dtype=int)
    return sent_model.predict(x_split).astype(int)


def fit_presence_model(x_train: pd.Series, y_presence: np.ndarray, cfg: dict):
    unique_presence = np.unique(y_presence)
    majority_label = int(pd.Series(y_presence).mode().iloc[0])

    if unique_presence.shape[0] < 2:
        dummy = DummyClassifier(strategy='constant', constant=majority_label)
        return _fit_dummy_model(dummy, y_presence)

    model = make_pipeline(cfg=cfg, class_weight='balanced')
    model.fit(x_train, y_presence)
    return model


def fit_aspect_sent_model(x_train: pd.Series, y_aspect: np.ndarray, cfg: dict):
    present_mask = y_aspect != ABSENT_LABEL
    if present_mask.sum() == 0:
        return None, 0

    y_present = y_aspect[present_mask]
    unique_present = np.unique(y_present)
    majority_label = int(pd.Series(y_present).mode().iloc[0])

    if unique_present.shape[0] < 2:
        dummy = DummyClassifier(strategy='constant', constant=majority_label)
        dummy = _fit_dummy_model(dummy, y_present)
        return dummy, majority_label

    classes = np.unique(y_present)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_present)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

    neutral_boost = float(cfg.get('neutral_class_boost', 1.0))
    neutral_cap_ratio = float(cfg.get('neutral_weight_cap_ratio', 2.5))
    if 1 in class_weight and neutral_boost != 1.0:
        non_neutral = [w for cls, w in class_weight.items() if cls != 1]
        non_neutral_max = max(non_neutral) if non_neutral else class_weight[1]
        boosted_weight = class_weight[1] * neutral_boost
        capped_weight = non_neutral_max * neutral_cap_ratio
        class_weight[1] = float(min(boosted_weight, capped_weight))

    model = make_pipeline(cfg=cfg, class_weight=class_weight)
    model.fit(x_train[present_mask], y_present)
    return model, majority_label


def tune_presence_threshold(
    x_val: pd.Series,
    y_val_aspect: np.ndarray,
    presence_model,
    sent_model,
    sent_fallback: int,
    neutral_weight: float = 0.0,
) -> float:
    true_present_mask = y_val_aspect != ABSENT_LABEL
    if not true_present_mask.any():
        return 0.5

    sent_pred = _predict_sentiment_labels(x_val, sent_model, sent_fallback)
    presence_score = _predict_positive_score(presence_model, x_val)

    thr_grid = np.linspace(0.05, 0.90, 35)
    best_thr = 0.5
    best_obj = -1.0

    for thr in thr_grid:
        pred_presence = (presence_score >= thr).astype(int)
        pred_aspect = np.where(pred_presence == 0, ABSENT_LABEL, sent_pred)

        f1_all = macro_f1(y_val_aspect, pred_aspect, labels=[0, 1, 2, ABSENT_LABEL])
        f1_present_macro = macro_f1(
            y_val_aspect[true_present_mask], pred_aspect[true_present_mask], labels=[0, 1, 2]
        )
        _, _, f1_neutral_present, _ = precision_recall_fscore_support(
            y_val_aspect[true_present_mask],
            pred_aspect[true_present_mask],
            labels=[1],
            average='macro',
            zero_division=0,
        )

        neutral_obj = (1.0 - neutral_weight) * f1_present_macro + neutral_weight * float(f1_neutral_present)
        objective = 0.8 * neutral_obj + 0.2 * f1_all

        if objective > best_obj:
            best_obj = float(objective)
            best_thr = float(thr)

    return best_thr


def predict_aspect(
    x_split: pd.Series,
    presence_model,
    presence_threshold: float,
    sent_model,
    sent_fallback: int,
) -> np.ndarray:
    presence_score = _predict_positive_score(presence_model, x_split)
    pred_presence = (presence_score >= presence_threshold).astype(int)
    pred_sent = _predict_sentiment_labels(x_split, sent_model, sent_fallback)
    pred_aspect = np.where(pred_presence == 0, ABSENT_LABEL, pred_sent)
    return pred_aspect.astype(int)


# %% [code cell 4]
# CV 5-fold on ALL splits: compare raw vs processed with paired IDs
from sklearn.model_selection import StratifiedKFold

CV_N_SPLITS = 5
CV_RANDOM_STATE = 42
CV_PAIR_SOURCES = ('interim', 'processed')
CV_TUNE_THRESHOLD = True  # False to run faster
CV_REQUIRE_LABEL_MATCH = True  # keep only paired rows with identical labels in both sources


def _load_concat_source(source: str) -> pd.DataFrame:
    if source not in DATA_SOURCES:
        raise ValueError(f'Unknown source: {source}. Valid: {list(DATA_SOURCES)}')

    segmenter, segmenter_name = _build_vn_segmenter()
    frames = []
    for split_name in ('train', 'val', 'test'):
        split_path = DATA_SOURCES[source][split_name]
        if not split_path.exists():
            raise FileNotFoundError(f'Missing {source}:{split_name} at {split_path}')
        frame = _prepare_frame(pd.read_json(split_path), f'{source}_{split_name}', segmenter=segmenter)
        frame['__split__'] = split_name
        frames.append(frame)

    out = pd.concat(frames, ignore_index=True)
    before = len(out)
    out[ID_COL] = out[ID_COL].astype(str)
    out = out.drop_duplicates(subset=[ID_COL], keep='first').reset_index(drop=True)
    dropped_dup = before - len(out)

    print(f"{source}: rows(all splits)={before:,}, unique_by_id={len(out):,}, dropped_dup={dropped_dup:,}, segmenter={segmenter_name}")
    return out


def _build_paired_dataset(source_a: str, source_b: str) -> pd.DataFrame:
    df_a = _load_concat_source(source_a)
    df_b = _load_concat_source(source_b)

    keep_cols = [ID_COL, TEXT_COL, SENTIMENT_COL, *ASPECT_COLS]
    a = df_a[keep_cols].copy()
    b = df_b[keep_cols].copy()

    a_rename = {TEXT_COL: f'text_{source_a}', SENTIMENT_COL: f'{SENTIMENT_COL}_{source_a}'}
    b_rename = {TEXT_COL: f'text_{source_b}', SENTIMENT_COL: f'{SENTIMENT_COL}_{source_b}'}
    for col in ASPECT_COLS:
        a_rename[col] = f'{col}_{source_a}'
        b_rename[col] = f'{col}_{source_b}'

    a = a.rename(columns=a_rename)
    b = b.rename(columns=b_rename)

    merged = a.merge(b, on=ID_COL, how='inner')
    print(f'Paired by {ID_COL}: {len(merged):,} rows')

    if CV_REQUIRE_LABEL_MATCH:
        sent_eq = merged[f'{SENTIMENT_COL}_{source_a}'] == merged[f'{SENTIMENT_COL}_{source_b}']
        aspect_eq = np.ones(len(merged), dtype=bool)
        for col in ASPECT_COLS:
            left = pd.to_numeric(merged[f'{col}_{source_a}'], errors='coerce').fillna(ABSENT_LABEL).astype(int)
            right = pd.to_numeric(merged[f'{col}_{source_b}'], errors='coerce').fillna(ABSENT_LABEL).astype(int)
            aspect_eq &= (left.to_numpy() == right.to_numpy())

        keep_mask = sent_eq.to_numpy() & aspect_eq
        dropped = int((~keep_mask).sum())
        if dropped > 0:
            print(f'Dropped label-mismatched paired rows: {dropped:,}')
        merged = merged.loc[keep_mask].reset_index(drop=True)

    paired = pd.DataFrame({
        ID_COL: merged[ID_COL].astype(str),
        f'text_{source_a}': merged[f'text_{source_a}'].fillna('').astype(str),
        f'text_{source_b}': merged[f'text_{source_b}'].fillna('').astype(str),
        SENTIMENT_COL: pd.to_numeric(merged[f'{SENTIMENT_COL}_{source_a}'], errors='coerce').astype(int),
    })

    for col in ASPECT_COLS:
        paired[col] = pd.to_numeric(merged[f'{col}_{source_a}'], errors='coerce')

    print(f'Final paired dataset for CV: {len(paired):,} rows')
    return paired


def _run_fold_for_source(
    X_train: pd.Series,
    X_val: pd.Series,
    y_train_sent: np.ndarray,
    y_val_sent: np.ndarray,
    y_train_asp: np.ndarray,
    y_val_asp: np.ndarray,
) -> tuple[dict, float]:
    start = time.time()

    sentiment_model_fold = make_pipeline(cfg=BEST_CFG, class_weight='balanced')
    sentiment_model_fold.fit(X_train, y_train_sent)
    pred_val_sent = sentiment_model_fold.predict(X_val).astype(int)

    pred_val_asp = np.full_like(y_val_asp, fill_value=ABSENT_LABEL)

    for asp_i, _col in enumerate(ASPECT_COLS):
        y_train_col = y_train_asp[:, asp_i]
        y_val_col = y_val_asp[:, asp_i]

        y_presence_train = (y_train_col != ABSENT_LABEL).astype(int)
        presence_model = fit_presence_model(X_train, y_presence_train, BEST_CFG)
        sent_model, sent_fallback = fit_aspect_sent_model(X_train, y_train_col, BEST_CFG)

        if CV_TUNE_THRESHOLD:
            presence_threshold = tune_presence_threshold(
                x_val=X_val,
                y_val_aspect=y_val_col,
                presence_model=presence_model,
                sent_model=sent_model,
                sent_fallback=sent_fallback,
                neutral_weight=BEST_CFG.get('presence_threshold_neutral_weight', 0.0),
            )
        else:
            presence_threshold = 0.5

        pred_val_col = predict_aspect(
            x_split=X_val,
            presence_model=presence_model,
            presence_threshold=presence_threshold,
            sent_model=sent_model,
            sent_fallback=sent_fallback,
        )
        pred_val_asp[:, asp_i] = pred_val_col

    metrics = compute_metrics_absa(
        true_sent=y_val_sent,
        pred_sent=pred_val_sent,
        true_aspects=y_val_asp,
        pred_aspects=pred_val_asp,
        aspect_cols=ASPECT_COLS,
    )

    fit_time_sec = time.time() - start
    return metrics, fit_time_sec


source_a, source_b = CV_PAIR_SOURCES
paired_df = _build_paired_dataset(source_a, source_b)

X_a = paired_df[f'text_{source_a}'].reset_index(drop=True)
X_b = paired_df[f'text_{source_b}'].reset_index(drop=True)
y_sent = paired_df[SENTIMENT_COL].to_numpy()
y_asp = to_aspect_matrix(paired_df)

skf = StratifiedKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE)
rows = []

print('\n' + '=' * 72)
print(f'Running paired CV with {CV_N_SPLITS} folds on {len(paired_df):,} samples')

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_a, y_sent), start=1):
    y_train_sent = y_sent[train_idx]
    y_val_sent = y_sent[val_idx]
    y_train_asp = y_asp[train_idx]
    y_val_asp = y_asp[val_idx]

    X_train_a = X_a.iloc[train_idx]
    X_val_a = X_a.iloc[val_idx]
    metrics_a, time_a = _run_fold_for_source(
        X_train=X_train_a,
        X_val=X_val_a,
        y_train_sent=y_train_sent,
        y_val_sent=y_val_sent,
        y_train_asp=y_train_asp,
        y_val_asp=y_val_asp,
    )
    rows.append({
        'fold': fold_idx,
        'source': source_a,
        'n_train': int(len(train_idx)),
        'n_val': int(len(val_idx)),
        'fit_time_sec': round(float(time_a), 2),
        **metrics_a,
    })

    X_train_b = X_b.iloc[train_idx]
    X_val_b = X_b.iloc[val_idx]
    metrics_b, time_b = _run_fold_for_source(
        X_train=X_train_b,
        X_val=X_val_b,
        y_train_sent=y_train_sent,
        y_val_sent=y_val_sent,
        y_train_asp=y_train_asp,
        y_val_asp=y_val_asp,
    )
    rows.append({
        'fold': fold_idx,
        'source': source_b,
        'n_train': int(len(train_idx)),
        'n_val': int(len(val_idx)),
        'fit_time_sec': round(float(time_b), 2),
        **metrics_b,
    })

    print(
        f"fold {fold_idx}/{CV_N_SPLITS} | "
        f"{source_a} f1_combined={metrics_a['f1_combined']:.4f} ({time_a:.1f}s) | "
        f"{source_b} f1_combined={metrics_b['f1_combined']:.4f} ({time_b:.1f}s)"
    )

cv_df = pd.DataFrame(rows)

metric_cols = [
    'f1_combined',
    'f1_sentiment',
    'f1_aspect_present',
    'f1_aspect_neutral_present',
    'fit_time_sec',
]

cv_summary = (
    cv_df.groupby('source')[metric_cols]
    .agg(['mean', 'std'])
    .reset_index()
)
cv_summary.columns = ['source'] + [f'{m}_{stat}' for m in metric_cols for stat in ('mean', 'std')]

pivoted = cv_df.pivot(index='fold', columns='source')
delta_df = pd.DataFrame({'fold': sorted(cv_df['fold'].unique())})
for m in ['f1_combined', 'f1_sentiment', 'f1_aspect_present', 'f1_aspect_neutral_present', 'fit_time_sec']:
    a_vals = pivoted[(m, source_a)].reindex(delta_df['fold']).to_numpy()
    b_vals = pivoted[(m, source_b)].reindex(delta_df['fold']).to_numpy()
    delta_df[f'delta_{m}_{source_b}_minus_{source_a}'] = b_vals - a_vals

out_root = ROOT / 'experiments'
out_root.mkdir(parents=True, exist_ok=True)
tag = f'baseline_logreg_cv{CV_N_SPLITS}_all_paired_{source_a}_vs_{source_b}'
cv_detail_path = out_root / f'{tag}_detail.csv'
cv_summary_path = out_root / f'{tag}_summary.csv'
cv_delta_path = out_root / f'{tag}_delta.csv'

cv_df.to_csv(cv_detail_path, index=False)
cv_summary.to_csv(cv_summary_path, index=False)
delta_df.to_csv(cv_delta_path, index=False)

print('\nSaved CV detail :', cv_detail_path)
print('Saved CV summary:', cv_summary_path)
print('Saved CV delta  :', cv_delta_path)

cv_summary
