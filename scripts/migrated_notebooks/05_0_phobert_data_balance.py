"""Migrated from 05_0_phobert_data_balance.ipynb.

This file preserves the original experiment code for audit/reproducibility.
The production benchmark uses the modular ml/ package instead.
"""


# %% [code cell 1]
# NOTEBOOK_ONLY: !pip install umap-learn


# %% [code cell 2]
import os, sys, shutil, warnings
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
)

try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    HAS_ITERSTRAT = True
except ImportError:
    HAS_ITERSTRAT = False
    print('[WARN] iterstrat not available, falling back to single-label stratify.')

from transformers import (
    AutoTokenizer, RobertaModel, RobertaPreTrainedModel,
    RobertaConfig, Trainer, TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import default_data_collator

import umap
from sklearn.manifold import TSNE
from tqdm.auto import tqdm
from transformers import TrainerCallback

import os
import glob
from IPython.display import Image, display

warnings.filterwarnings('ignore')


# %% [code cell 3]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Thiết bị:', device)


# %% [code cell 4]
# Cấu hình hyperparameter
MODEL_NAME    = 'vinai/phobert-base-v2'
MAX_LENGTH    = 160
BATCH_SIZE    = 16
EPOCHS        = 7
LEARNING_RATE = 2e-5

# Cấu hình loss
BASE_FOCAL_GAMMA = 2.0
CLASS_BALANCED_BETA   = 0.999
ASPECT_ABSENT_WEIGHT_SCALE = 0.2
ABSENT_ASPECT_CLASS   = 3

# Trọng số loss cho tác vụ
DEFAULT_SENTIMENT_LOSS_WEIGHT = 0.5
DEFAULT_ASPECT_LOSS_WEIGHT    = 0.5

# Trọng số loss 2 tầng cho aspect
STAGE1_LOSS_WEIGHT = 0.25   # presence detection
STAGE2_LOSS_WEIGHT = 0.75   # sentiment classification

# Cấu hình Focal cải tiến
IMPROVED_FOCAL_SENTIMENT_SMOOTHING  = 0.05
IMPROVED_FOCAL_ASPECT_SMOOTHING     = 0.10
IMPROVED_FOCAL_ASPECT_GAMMA_PRESENT = 2.5
IMPROVED_FOCAL_ASPECT_GAMMA_ABSENT  = 1.0
IMPROVED_FOCAL_IGNORE_EASY_ABSENT   = True
IMPROVED_FOCAL_IGNORE_THRESHOLD     = 0.5

# Cấu hình balanced sampling
SAMPLER_TEMPERATURE = 0.5
SAMPLER_WEIGHT_CAP = 4.0

NEUTRAL_SAMPLER_TEMPERATURE = 0.45

# Neutral-focus config
NEUTRAL_ASPECT_GAMMA = 0.8
NEUTRAL_ASPECT_SMOOTHING = 0.05
THRESHOLD_NEUTRAL_WEIGHT = 0.35


# %% [code cell 5]
SENTIMENT_LABELS = {0: 'Tiêu cực', 1: 'Trung lập', 2: 'Tích cực'}
ASPECT_LABELS    = {0: 'Tiêu cực', 1: 'Trung lập', 2: 'Tích cực', 3: 'Không nhắc đến'}
ASPECT_COLS      = ['as_content', 'as_physical', 'as_price', 'as_packaging', 'as_delivery', 'as_service']
TARGET_COLS      = ['sentiment'] + ASPECT_COLS


# %% [code cell 6]
DATA_ROOT = Path('/kaggle/input/datasets/nguynvntnpht/tiki-cleaned-book-reviews-absa')
TRAIN_PATH = DATA_ROOT / 'train_clean.json'
TEST_PATH  = DATA_ROOT / 'test_clean.json'
VAL_PATH   = DATA_ROOT / 'val_clean.json'



# %% [code cell 7]
def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    text_col = 'text' if 'text' in df.columns else 'content'
    df = df[[text_col] + TARGET_COLS].dropna(subset=[text_col, 'sentiment']).copy()
    df = df.rename(columns={text_col: 'text'})
    df['text'] = df['text'].astype(str).str.strip()
    df = df[df['text'].ne('')]
    df['sentiment'] = pd.to_numeric(df['sentiment'], errors='coerce').astype('Int64')
    df = df.dropna(subset=['sentiment'])
    df['sentiment'] = df['sentiment'].astype(int)
    df = df[df['sentiment'].isin([0, 1, 2])]
    for col in ASPECT_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(3).astype(int)
        df[col] = df[col].apply(lambda x: x if x in [0, 1, 2, 3] else 3)
    return df.reset_index(drop=True)

train_df = prepare_df(pd.read_json(TRAIN_PATH))
val_df   = prepare_df(pd.read_json(VAL_PATH))
test_df  = prepare_df(pd.read_json(TEST_PATH))
full_train_df = train_df.copy()
print(f'Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}')
print('\nPhan phoi sentiment (train):')
print(train_df['sentiment'].value_counts().sort_index().rename(index=SENTIMENT_LABELS))


# %% [code cell 8]
import re
from itertools import combinations


def _normalize_text_series(df: pd.DataFrame) -> pd.Series:
    return (
        df['text']
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r'\s+', ' ', regex=True)
    )


def _row_signature_series(df: pd.DataFrame) -> pd.Series:
    label_part = df[['sentiment'] + ASPECT_COLS].astype(str).agg('|'.join, axis=1)
    return _normalize_text_series(df) + '||' + label_part


def report_split_overlap(splits: dict[str, pd.DataFrame], sample_n: int = 8) -> None:
    normalized = {name: _normalize_text_series(df) for name, df in splits.items()}
    signatures = {name: _row_signature_series(df) for name, df in splits.items()}

    print('=== Within-split duplicates ===')
    for name, df in splits.items():
        dup_text = int(normalized[name].duplicated().sum())
        dup_row = int(signatures[name].duplicated().sum())
        print(f'{name:>5}: rows={len(df):,} | duplicate text={dup_text:,} | duplicate text+labels={dup_row:,}')

    print('\n=== Cross-split overlap ===')
    for left, right in combinations(splits.keys(), 2):
        shared_text = sorted(set(normalized[left]) & set(normalized[right]))
        shared_row = sorted(set(signatures[left]) & set(signatures[right]))
        print(f'{left} ? {right}: text={len(shared_text):,} | text+labels={len(shared_row):,}')

        if shared_text:
            sample_texts = pd.DataFrame({'text': shared_text[:sample_n]})
            display(sample_texts)

            # Show how labels differ for a small sample of overlapping texts.
            for text in shared_text[:min(3, len(shared_text))]:
                print(f'\nText sample: {text}')
                for split_name, df in splits.items():
                    rows = df[_normalize_text_series(df) == text][['sentiment'] + ASPECT_COLS].drop_duplicates()
                    if not rows.empty:
                        print(f'  {split_name}:')
                        display(rows.head(sample_n))


split_dfs = {
    'train': train_df,
    'val': val_df,
    'test': test_df,
}
report_split_overlap(split_dfs)


# %% [code cell 9]
# Split da duoc load truc tiep tu Kaggle paths o cell truoc.
# Giu cell nay de cac cell phia sau khong can doi.
print(f'Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}')
print('\nPhan phoi sentiment (train):')
print(train_df['sentiment'].value_counts().sort_index().rename(index=SENTIMENT_LABELS))


# %% [code cell 10]
# Notebook nay da bo data-level oversampling.
# Neu can can bang hon, se dung sampler on-the-fly theo phan phoi nhan.


# %% [code cell 11]
train_variants = {
    'clean': train_df.copy().reset_index(drop=True),
}

for name, df in train_variants.items():
    print(f'{name}: {len(df):,} rows | sentiment: {dict(df["sentiment"].value_counts().sort_index())}')

clean_train_df = train_variants['clean']


# %% [code cell 12]
def compute_class_balanced_weights(labels, classes, beta=0.999, normalize=True, cap=None):
    values = np.asarray(labels)
    counts = {c: int((values == c).sum()) for c in classes}
    weights = []
    for c in classes:
        n = counts.get(c, 0)
        w = (1 - beta) / (1 - beta**n) if n > 0 else 0.0
        if cap is not None:
            w = min(w, cap)
        weights.append(w)
    weights = np.array(weights, dtype=float)
    # Normalize theo max để cap=1.0 giữ nguyên hiệu lực
    if normalize and weights.max() > 0:
        weights /= weights.max()
    return weights

def downweight_absent(weights, absent_class=3, scale=ASPECT_ABSENT_WEIGHT_SCALE):
    w = weights.copy()
    if len(w) > absent_class and w[absent_class] > 0:
        w[absent_class] *= scale
    if w.max() > 0:
        w /= w[w > 0].mean()
    return w


# %% [code cell 13]
# Sentiment weights
cb_sent_weights = compute_class_balanced_weights(
    clean_train_df['sentiment'].values, classes=[0, 1, 2], beta=CLASS_BALANCED_BETA
)

# neutral_capped: cap trung lập ≤ 1.0 trước khi normalize
cb_sent_weights_neutral_capped = compute_class_balanced_weights(
    clean_train_df['sentiment'].values, classes=[0, 1, 2], beta=CLASS_BALANCED_BETA,
    cap=1.0,
)

# Aspect weights (class_balanced + down-weight absent)
cb_aspect_weights = {}
for col in ASPECT_COLS:
    w = compute_class_balanced_weights(clean_train_df[col].values, classes=[0,1,2,3])
    cb_aspect_weights[col] = downweight_absent(w)

# Tensors — chỉ giữ sentiment (aspect dùng presence/stage2 tensors riêng)
sent_weight_tensors = {
    'class_balanced': torch.tensor(cb_sent_weights, dtype=torch.float),
    'neutral_capped': torch.tensor(cb_sent_weights_neutral_capped, dtype=torch.float),
}

print('Sentiment weights (class_balanced):',
      {SENTIMENT_LABELS[i]: round(float(w), 3) for i, w in enumerate(cb_sent_weights)})
print('Sentiment weights (neutral_capped) :',
      {SENTIMENT_LABELS[i]: round(float(w), 3) for i, w in enumerate(cb_sent_weights_neutral_capped)})


def build_joint_label_sampler(
    frame: pd.DataFrame,
    temperature: float = SAMPLER_TEMPERATURE,
    include_neutral_bucket: bool = False,
):
    aspect_values = frame[ASPECT_COLS].apply(pd.to_numeric, errors='coerce').fillna(3).astype(int)
    present_count = aspect_values.ne(3).sum(axis=1).clip(upper=len(ASPECT_COLS)).astype(int)

    group_parts = [
        'sent=' + frame['sentiment'].astype(str),
        '|present=' + present_count.astype(str),
    ]

    if include_neutral_bucket:
        neutral_count = aspect_values.eq(1).sum(axis=1).astype(int)
        neutral_bucket = neutral_count.clip(upper=2).astype(int)
        group_parts.append('|neutral=' + neutral_bucket.astype(str))

    joint_group = group_parts[0]
    for part in group_parts[1:]:
        joint_group = joint_group + part

    counts = joint_group.value_counts()
    sample_weights = joint_group.map(lambda g: counts[g] ** (-temperature)).astype(float)
    sample_weights = sample_weights / sample_weights.mean()
    sample_weights = sample_weights.clip(upper=SAMPLER_WEIGHT_CAP)
    sample_weights = sample_weights / sample_weights.mean()

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights.values, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    summary = (
        pd.DataFrame({'group': joint_group, 'weight': sample_weights})
        .groupby('group', as_index=False)
        .agg(count=('group', 'size'), avg_weight=('weight', 'mean'))
        .sort_values(['count', 'group'], ascending=[False, True])
        .reset_index(drop=True)
    )
    return sampler, summary


# %% [code cell 14]
# Dynamic aspect weights computed from train_df
def mean_normalized_class_balanced_weights(labels, classes, beta=0.9995, cap=6.0):
    weights = compute_class_balanced_weights(labels, classes, beta=beta, normalize=False, cap=cap)
    valid = weights > 0
    if valid.any():
        weights = weights / weights[valid].mean()
    return weights

aspect_frame = train_df
PRESENCE_WEIGHTS = {}
STAGE2_WEIGHTS = {}

for col in ASPECT_COLS:
    aspect_values = pd.to_numeric(aspect_frame[col], errors="coerce").fillna(ABSENT_ASPECT_CLASS).astype(int)
    presence_labels = (aspect_values != ABSENT_ASPECT_CLASS).astype(int)
    presence_w = mean_normalized_class_balanced_weights(presence_labels, [0, 1])

    present_values = aspect_values[aspect_values != ABSENT_ASPECT_CLASS]
    if len(present_values):
        stage2_w = mean_normalized_class_balanced_weights(present_values, [0, 1, 2])
    else:
        stage2_w = np.ones(3, dtype=float)

    PRESENCE_WEIGHTS[col] = presence_w.tolist()
    STAGE2_WEIGHTS[col] = stage2_w.tolist()

presence_weight_tensors = {col: torch.tensor(w, dtype=torch.float) for col, w in PRESENCE_WEIGHTS.items()}
stage2_weight_tensors   = {col: torch.tensor(w, dtype=torch.float) for col, w in STAGE2_WEIGHTS.items()}

print("Presence weights:")
for col in ASPECT_COLS:
    print(f"{col:14} {np.round(PRESENCE_WEIGHTS[col], 4).tolist()}")

print("Stage-2 weights:")
for col in ASPECT_COLS:
    print(f"{col:14} {np.round(STAGE2_WEIGHTS[col], 4).tolist()}")


# %% [code cell 15]
class ABSAModel(RobertaPreTrainedModel):
    """
    PhoBERT với 3 loại head:
      - sentiment_head       : (B, 3) — cảm xúc tổng thể
      - presence_heads [×6]  : (B, 2) — có/không nhắc đến từng aspect
      - aspect_sentiment_heads [×6] : (B, 3) — cảm xúc từng aspect (chỉ khi present)

    Logits layout: [sent(3) | pres_0..5(2) | asp_sent_0..5(3)]
    """
    config_class = RobertaConfig

    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        drop_p = getattr(config, 'classifier_dropout', None) or getattr(config, 'hidden_dropout_prob', 0.1)
        self.dropout = nn.Dropout(drop_p)
        n = len(ASPECT_COLS)
        self.sentiment_head        = nn.Linear(config.hidden_size, 3)
        self.presence_heads        = nn.ModuleList([nn.Linear(config.hidden_size, 2) for _ in range(n)])
        self.aspect_sentiment_heads = nn.ModuleList([nn.Linear(config.hidden_size, 3) for _ in range(n)])
        self.post_init()

    @classmethod
    def _can_set_experts_implementation(cls):
        return False

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        cls = self.dropout(outputs.last_hidden_state[:, 0, :])
        self._last_cls = cls

        sent_logits     = self.sentiment_head(cls)
        pres_logits     = torch.stack([h(cls) for h in self.presence_heads], dim=1)
        asp_sent_logits = torch.stack([h(cls) for h in self.aspect_sentiment_heads], dim=1)

        logits = torch.cat([
            sent_logits,
            pres_logits.view(cls.size(0), -1),
            asp_sent_logits.view(cls.size(0), -1),
        ], dim=-1)
        return SequenceClassifierOutput(logits=logits)


N_ASPECTS    = len(ASPECT_COLS)
SENT_DIM     = 3
PRES_DIM     = 2
ASP_SENT_DIM = 3

def parse_logits(logits):
    s = SENT_DIM
    p = N_ASPECTS * PRES_DIM
    return logits[:, :s], logits[:, s:s+p].view(-1, N_ASPECTS, PRES_DIM), logits[:, s+p:].view(-1, N_ASPECTS, ASP_SENT_DIM)


# %% [code cell 16]
def build_smoothed_targets(logits, targets, smoothing):
    n_cls = logits.size(-1)
    with torch.no_grad():
        t = torch.full_like(logits, smoothing / n_cls)
        t.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing + smoothing / n_cls)
    return t

def resolve_gamma_per_sample(targets, logits, gamma, gamma_by_class):
    if gamma_by_class is None:
        return torch.full((targets.size(0),), float(gamma), dtype=logits.dtype, device=logits.device)
    return torch.as_tensor(gamma_by_class, dtype=logits.dtype, device=logits.device)[targets]

def filter_easy_absent(loss, targets, true_probs, ignore_easy_absent, ignore_threshold, absent_class):
    if not (ignore_easy_absent and absent_class is not None and ignore_threshold is not None):
        return loss.mean()
    keep = ~((targets == absent_class) & (true_probs > ignore_threshold))
    return loss[keep].mean() if keep.any() else loss.mean()

def improved_focal_loss(logits, targets, *, gamma=2.0, alpha=None, smoothing=0.0,
                         gamma_by_class=None, ignore_easy_absent=False,
                         ignore_threshold=None, absent_class=None):
    smoothed = build_smoothed_targets(logits, targets, smoothing)
    log_probs  = F.log_softmax(logits, dim=-1)
    true_probs = log_probs.exp().gather(1, targets.unsqueeze(1)).squeeze(1).clamp(1e-6, 1.0)
    focal_gamma = resolve_gamma_per_sample(targets, logits, gamma, gamma_by_class)
    loss = ((1 - true_probs).pow(focal_gamma)) * (-(smoothed * log_probs).sum(dim=-1))
    if alpha is not None:
        loss = loss * alpha.to(logits.device)[targets]
    return filter_easy_absent(loss, targets, true_probs, ignore_easy_absent, ignore_threshold, absent_class)

def asymmetric_focal_loss(logits, targets, *, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, class_weights=None):
    probs = torch.softmax(logits, dim=-1)
    log_p = torch.log(probs.clamp(min=1e-8))
    C     = logits.size(-1)
    oh    = F.one_hot(targets, C).float().to(logits.device)
    p_pos = (probs * oh).sum(-1)
    loss_p = -(1 - p_pos).pow(gamma_pos) * (log_p * oh).sum(-1)
    pn_s  = (probs * (1 - oh) + clip).clamp(max=1.0)
    l1mp  = torch.log((1.0 - probs * (1 - oh)).clamp(min=1e-8))
    loss_n = -(pn_s.pow(gamma_neg) * (1 - oh) * l1mp).sum(-1)
    loss  = loss_p + loss_n
    if class_weights is not None:
        loss = loss * class_weights.to(logits.device)[targets]
    return loss.mean()


# %% [code cell 17]
class SupervisedContrastiveLoss(nn.Module):
    """Kéo CLS embeddings cùng class lại gần nhau → tăng separability Trung lập."""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        B = features.size(0)
        if B < 2:
            return features.new_tensor(0.0)
        feat = F.normalize(features, dim=-1)
        sim  = feat @ feat.T / self.temperature
        pos  = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        pos.fill_diagonal_(0.0)
        denom = torch.log(
            (torch.exp(sim) * (1 - torch.eye(B, device=sim.device))).sum(-1).clamp(min=1e-8)
        )
        loss = -((pos * sim).sum(-1) / pos.sum(-1).clamp(min=1) - denom)
        return loss.mean()

supervised_contrastive = SupervisedContrastiveLoss(temperature=0.07)


# %% [code cell 18]
def compute_label_loss(logits, targets, *, loss_name='ce', class_weights=None,
                        gamma=2.0, alpha=None, focal_params=None):
    fp = dict(focal_params or {})
    if loss_name == 'ce':
        return F.cross_entropy(logits, targets, weight=class_weights)
    if loss_name == 'improved_focal':
        return improved_focal_loss(
            logits, targets,
            gamma=fp.get('gamma', gamma),
            alpha=alpha,
            smoothing=fp.get('smoothing', 0.0),
            gamma_by_class=fp.get('gamma_by_class'),
            ignore_easy_absent=fp.get('ignore_easy_absent', False),
            ignore_threshold=fp.get('ignore_threshold'),
            absent_class=fp.get('absent_class'),
        )
    if loss_name == 'asl':
        return asymmetric_focal_loss(
            logits, targets,
            gamma_neg=fp.get('asl_gamma_neg', 4.0),
            gamma_pos=fp.get('asl_gamma_pos', 1.0),
            clip=fp.get('asl_clip', 0.05),
            class_weights=class_weights,
        )
    raise ValueError(f'Unsupported loss_name: {loss_name!r}')


# %% [code cell 19]
def normalize_task_weights(sw, aw):
    total = float(sw) + float(aw)
    return float(sw) / total, float(aw) / total

def compute_sentiment_loss(trainer, logits, labels):
    sent_logits, _, _ = parse_logits(logits)
    w = trainer.sent_weights.to(logits.device) if trainer.use_class_weights and trainer.sent_weights is not None else None
    # improved_focal_loss dùng alpha chứ không dùng class_weights
    # truyền sent_weights trực tiếp làm alpha để neutral_capped có hiệu lực
    alpha = w if trainer.loss_name == 'improved_focal' else None
    fp = dict(trainer.focal_config.get('sentiment') or {})
    return compute_label_loss(sent_logits, labels[:, 0],
                              loss_name=trainer.loss_name,
                              class_weights=w, gamma=trainer.focal_gamma,
                              alpha=alpha, focal_params=fp)

def compute_aspect_loss(trainer, logits, labels):
    _, pres_logits, asp_sent_logits = parse_logits(logits)
    aspect_labels = labels[:, 1:]
    s1_losses, s2_losses = [], []
    for i, col in enumerate(ASPECT_COLS):
        asp_label  = aspect_labels[:, i]
        pres_label = (asp_label != 3).long()
        pres_w     = presence_weight_tensors[col].to(logits.device)
        s1_losses.append(F.cross_entropy(pres_logits[:, i, :], pres_label, weight=pres_w))
        present_mask = asp_label != 3
        if present_mask.any():
            s2_logits = asp_sent_logits[:, i, :][present_mask]
            s2_labels = asp_label[present_mask]
            s2_w      = stage2_weight_tensors[col].to(logits.device)
            fp = dict(trainer.focal_config.get('aspect', {}))
            if 'gamma_by_class' in fp:
                fp['gamma_by_class'] = fp['gamma_by_class'][:3]
            for key in ('ignore_easy_absent', 'ignore_threshold', 'absent_class'):
                fp.pop(key, None)
            s2_losses.append(compute_label_loss(
                s2_logits, s2_labels,
                loss_name=trainer.loss_name,
                class_weights=s2_w, gamma=trainer.focal_gamma,
                focal_params=fp,
            ))
    loss_s1 = sum(s1_losses) / len(s1_losses)
    loss_s2 = sum(s2_losses) / len(s2_losses) if s2_losses else logits.new_zeros(())
    return STAGE1_LOSS_WEIGHT * loss_s1 + STAGE2_LOSS_WEIGHT * loss_s2


class ABSAMultiTaskTrainer(Trainer):
    def __init__(self, *args, sent_weights=None, loss_name='ce',
                 use_class_weights=False, focal_gamma=2.0,
                 focal_config=None, sentiment_loss_weight=0.5,
                 aspect_loss_weight=0.5, scl_weight=0.0,
                 train_sampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sent_weights   = sent_weights
        self.loss_name      = loss_name
        self.use_class_weights = use_class_weights
        self.focal_gamma    = focal_gamma
        self.focal_config   = focal_config or {}
        self.sentiment_loss_weight, self.aspect_loss_weight = normalize_task_weights(
            sentiment_loss_weight, aspect_loss_weight
        )
        self.scl_weight = scl_weight
        self.train_sampler = train_sampler

    def get_train_dataloader(self):
        if self.train_sampler is None:
            return super().get_train_dataloader()
        if self.train_dataset is None:
            raise ValueError('Trainer requires a train_dataset when train_sampler is set.')
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=self.train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop('labels')
        outputs = model(**inputs)
        logits  = outputs.logits

        loss = (self.sentiment_loss_weight * compute_sentiment_loss(self, logits, labels)
                + self.aspect_loss_weight  * compute_aspect_loss(self, logits, labels))

        if self.scl_weight > 0 and model.training:
            cls_vec = getattr(model, '_last_cls', None)
            if cls_vec is not None:
                loss = loss + self.scl_weight * supervised_contrastive(cls_vec, labels[:, 0])

        return (loss, outputs) if return_outputs else loss


# %% [code cell 20]
def compute_metrics(eval_pred):
    """Metric chuẩn cho checkpoint selection và báo cáo raw (argmax)."""
    logits, labels = eval_pred
    s = SENT_DIM
    p = N_ASPECTS * PRES_DIM

    pred_sent = np.argmax(logits[:, :s], axis=-1)
    pred_presence = np.argmax(
        logits[:, s:s+p].reshape(-1, N_ASPECTS, PRES_DIM),
        axis=-1,
    )
    pred_asp_sent = np.argmax(
        logits[:, s+p:].reshape(-1, N_ASPECTS, ASP_SENT_DIM),
        axis=-1,
    )

    true_sent = labels[:, 0]
    true_aspects = labels[:, 1:]
    pred_aspects = np.where(pred_presence == 0, 3, pred_asp_sent)

    f1_sentiment = precision_recall_fscore_support(
        true_sent, pred_sent, labels=[0, 1, 2],
        average='macro', zero_division=0
    )[2]

    present_mask = true_aspects.flatten() != 3
    if present_mask.any():
        true_present = true_aspects.flatten()[present_mask]
        pred_present = pred_aspects.flatten()[present_mask]
        f1_aspect_present = precision_recall_fscore_support(
            true_present, pred_present, labels=[0, 1, 2],
            average='macro', zero_division=0
        )[2]
        f1_aspect_neutral_present = precision_recall_fscore_support(
            true_present, pred_present, labels=[1],
            average='macro', zero_division=0
        )[2]
    else:
        f1_aspect_present = 0.0
        f1_aspect_neutral_present = 0.0

    f1_combined = 0.5 * f1_sentiment + 0.5 * f1_aspect_present
    accuracy = accuracy_score(true_sent, pred_sent)

    f1_aspect_all = precision_recall_fscore_support(
        true_aspects.flatten(), pred_aspects.flatten(),
        labels=[0, 1, 2, 3], average='macro', zero_division=0
    )[2]

    aspect_diagnostics = {}
    for i, col in enumerate(ASPECT_COLS):
        mask = true_aspects[:, i] != 3
        if mask.any():
            f1_i = precision_recall_fscore_support(
                true_aspects[:, i][mask],
                pred_aspects[:, i][mask],
                labels=[0, 1, 2],
                average='macro',
                zero_division=0,
            )[2]
        else:
            f1_i = 0.0
        aspect_diagnostics[f'diagnostic_f1_{col}'] = round(float(f1_i), 4)

    return {
        'f1_sentiment': round(float(f1_sentiment), 4),
        'f1_aspect_present': round(float(f1_aspect_present), 4),
        'f1_combined': round(float(f1_combined), 4),
        'accuracy': round(float(accuracy), 4),
        'diagnostic_f1_aspect_all': round(float(f1_aspect_all), 4),
        'diagnostic_f1_aspect_neutral_present': round(
            float(f1_aspect_neutral_present), 4
        ),
        **aspect_diagnostics,
    }


# %% [code cell 21]
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

ABSA_PROMPT_PREFIX = 'ABSA review aspect content physical price packaging delivery service: '

def build_absa_input(text: str) -> str:
    return f"{ABSA_PROMPT_PREFIX}{str(text).strip()}"

def tokenize_frame(frame: pd.DataFrame) -> Dataset:
    def _tokenize(examples):
        texts = [build_absa_input(t) for t in examples['text']]
        enc = tokenizer(texts, padding='max_length',
                        truncation=True, max_length=MAX_LENGTH)
        enc['labels'] = [[examples[col][i] for col in TARGET_COLS]
                         for i in range(len(texts))]
        return enc
    ds = Dataset.from_pandas(frame.reset_index(drop=True)).map(
        _tokenize, batched=True, remove_columns=['text'] + TARGET_COLS)
    ds.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    return ds

val_encoded  = tokenize_frame(val_df)
test_encoded = tokenize_frame(test_df)
print('Val encoded:', len(val_encoded), '| Test encoded:', len(test_encoded))


# %% [code cell 22]
# [added] tokenization preview
from IPython.display import display

def show_tokenized_sample(df, idx=0, max_rows=60):
    row = df.iloc[idx]
    text = row['text']

    enc = tokenizer(
        build_absa_input(text),
        truncation=True,
        max_length=MAX_LENGTH,
        padding='max_length',
    )
    tokens = tokenizer.convert_ids_to_tokens(enc['input_ids'])
    valid_len = int(sum(enc['attention_mask']))
    decoded = tokenizer.decode(enc['input_ids'], skip_special_tokens=True)

    print('Original text:')
    print(text)
    print('\nDecoded text:')
    print(decoded)
    print(f'\nToken count (including special tokens): {valid_len}')

    preview = pd.DataFrame({
        'token': tokens[:valid_len],
        'input_id': enc['input_ids'][:valid_len],
        'attention_mask': enc['attention_mask'][:valid_len],
    })
    display(preview.head(max_rows))

show_tokenized_sample(train_df, idx=0)


# %% [code cell 23]
class EmbeddingVisualizerCallback(TrainerCallback):
    def __init__(self, val_dataset, tokenizer, output_dir, label_names, per_device_batch_size=32):
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir) / "convergence_plots"
        self.label_names = label_names
        self.batch_size = per_device_batch_size
        
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Visualizer] Ảnh quá trình hội tụ sẽ được lưu tại: {self.output_dir}")

    def get_val_embeddings(self, model):
        model.eval()
        # Sử dụng default_data_collator để tương thích 100% với HuggingFace Dataset
        dataloader = DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            collate_fn=default_data_collator
        )
        all_embeddings, all_labels = [], []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="[Visualizer] Trích xuất Embeddings Val"):
                input_ids = batch['input_ids'].to(model.device)
                attention_mask = batch['attention_mask'].to(model.device)
                
                # Lấy nhãn sentiment tổng thể (cột đầu tiên trong labels)
                labels = batch['labels'][:, 0].to(model.device) 
                
                # Trích xuất embeddings từ RoBERTa base
                outputs = model.roberta(input_ids=input_ids, attention_mask=attention_mask)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                all_embeddings.append(cls_embeddings)
                all_labels.append(labels.cpu().numpy())

        return np.concatenate(all_embeddings), np.concatenate(all_labels)

    def plot_embeddings(self, embeddings, labels, epoch):
        print(f"[Visualizer] Đang giảm chiều và vẽ ảnh Epoch {epoch}...")
        
        # Ánh xạ ID nhãn sang tên nhãn (Negative, Neutral, Positive)
        named_labels = [self.label_names[l] for l in labels]

        # Giảm chiều UMAP
        reducer_umap = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='cosine', random_state=42)
        umap_data = reducer_umap.fit_transform(embeddings)

        # Giảm chiều t-SNE
        reducer_tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, n_jobs=-1)
        tsne_data = reducer_tsne.fit_transform(embeddings)

        df = pd.DataFrame({
            'UMAP_X': umap_data[:, 0], 'UMAP_Y': umap_data[:, 1],
            'tSNE_X': tsne_data[:, 0], 'tSNE_Y': tsne_data[:, 1],
            'Sentiment': named_labels
        })

        fig, axes = plt.subplots(2, 1, figsize=(10, 16), gridspec_kw={'hspace': 0.3})
        
        # Vẽ biểu đồ
        sns.scatterplot(data=df, x='UMAP_X', y='UMAP_Y', hue='Sentiment', palette='viridis', alpha=0.6, ax=axes[0], s=30)
        axes[0].set_title(f'UMAP - [CLS] Embeddings - Epoch {epoch}', fontsize=14, fontweight='bold')
        
        sns.scatterplot(data=df, x='tSNE_X', y='tSNE_Y', hue='Sentiment', palette='viridis', alpha=0.6, ax=axes[1], s=30)
        axes[1].set_title(f't-SNE - [CLS] Embeddings - Epoch {epoch}', fontsize=14, fontweight='bold')

        file_path = self.output_dir / f"epoch_{epoch:03d}.png"
        plt.tight_layout()
        plt.savefig(file_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        # Chạy khi hàm trainer.evaluate() được gọi (cuối mỗi epoch theo cấu hình của bạn)
        if state.is_world_process_zero:
            embeddings, labels = self.get_val_embeddings(model)
            self.plot_embeddings(embeddings, labels, int(state.epoch))


# %% [code cell 24]
EXP_TRAIN_EPOCHS = EPOCHS
USE_FP16 = torch.cuda.is_available()

def build_training_args(output_dir: str) -> TrainingArguments:
    kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=EXP_TRAIN_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        gradient_accumulation_steps=2,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type='cosine',
        warmup_ratio=0.1,
        weight_decay=0.05,
        max_grad_norm=1.0,
        fp16=USE_FP16,
        logging_dir=str(Path(output_dir) / 'logs'),
        logging_steps=100,
        save_strategy='epoch',
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='f1_combined',
        greater_is_better=True,
        report_to='none',
    )
    if 'eval_strategy' in TrainingArguments.__init__.__code__.co_varnames:
        kwargs['eval_strategy'] = 'epoch'
    else:
        kwargs['evaluation_strategy'] = 'epoch'
    return TrainingArguments(**kwargs)


# %% [code cell 25]
# Focal config chia s? cho c?c experiment
FOCAL_CONFIG_STANDARD = {
    'sentiment': {
        'gamma':    BASE_FOCAL_GAMMA,
        'smoothing': IMPROVED_FOCAL_SENTIMENT_SMOOTHING,
    },
    'aspect': {
        'gamma':    BASE_FOCAL_GAMMA,
        'smoothing': IMPROVED_FOCAL_ASPECT_SMOOTHING,
        'gamma_by_class': [
            IMPROVED_FOCAL_ASPECT_GAMMA_PRESENT,
            IMPROVED_FOCAL_ASPECT_GAMMA_PRESENT,
            IMPROVED_FOCAL_ASPECT_GAMMA_PRESENT,
            IMPROVED_FOCAL_ASPECT_GAMMA_ABSENT,
        ],
        'ignore_easy_absent': IMPROVED_FOCAL_IGNORE_EASY_ABSENT,
        'ignore_threshold':   IMPROVED_FOCAL_IGNORE_THRESHOLD,
        'absent_class':       ABSENT_ASPECT_CLASS,
    },
}

# Focal config c?i ti?n Trung l?p:
#   - Sentiment neutral d?ng gamma th?p h?n ?? gi?m over-suppression
#   - Aspect neutral d?ng gamma/smoothing ri?ng ?? t?ng recall l?p trung l?p
FOCAL_CONFIG_NEUTRAL = {
    'sentiment': {
        'gamma':          BASE_FOCAL_GAMMA,
        'smoothing':      IMPROVED_FOCAL_SENTIMENT_SMOOTHING,
        'gamma_by_class': [2.5, 1.0, 2.5],
    },
    'aspect': {
        'gamma':    BASE_FOCAL_GAMMA,
        'smoothing': NEUTRAL_ASPECT_SMOOTHING,
        'gamma_by_class': [
            IMPROVED_FOCAL_ASPECT_GAMMA_PRESENT,
            NEUTRAL_ASPECT_GAMMA,
            IMPROVED_FOCAL_ASPECT_GAMMA_PRESENT,
            IMPROVED_FOCAL_ASPECT_GAMMA_ABSENT,
        ],
        'ignore_easy_absent': IMPROVED_FOCAL_IGNORE_EASY_ABSENT,
        'ignore_threshold':   IMPROVED_FOCAL_IGNORE_THRESHOLD,
        'absent_class':       ABSENT_ASPECT_CLASS,
    },
}


# %% [code cell 26]
EXPERIMENTS = [
    # Baseline: CE thuan
    {
        'name': 'baseline',
        'train_key': 'clean',
        'loss_name': 'ce',
        'use_class_weights': False,
        'sent_weight_key': None,
        'train_sampler': None,
        'sampler_temperature': None,
        'sentiment_loss_weight': 0.5,
        'aspect_loss_weight':    0.5,
        'scl_weight': 0.0,
        'focal_config': FOCAL_CONFIG_STANDARD,
    },
    # Clean + CE + class-balanced weights
    {
        'name': 'clean_class_balanced_ce',
        'train_key': 'clean',
        'loss_name': 'ce',
        'use_class_weights': True,
        'sent_weight_key': 'class_balanced',
        'train_sampler': None,
        'sampler_temperature': None,
        'sentiment_loss_weight': 0.3,
        'aspect_loss_weight':    0.7,
        'scl_weight': 0.0,
        'focal_config': FOCAL_CONFIG_STANDARD,
    },
    # Clean + CE + joint sampler
    {
        'name': 'clean_joint_balanced_ce',
        'train_key': 'clean',
        'loss_name': 'ce',
        'use_class_weights': False,
        'sent_weight_key': None,
        'train_sampler': 'joint_balanced',
        'sampler_temperature': 0.5,
        'sampler_include_neutral_bucket': False,
        'sentiment_loss_weight': 0.3,
        'aspect_loss_weight':    0.7,
        'scl_weight': 0.0,
        'focal_config': FOCAL_CONFIG_STANDARD,
    },
    # Clean + Improved Focal + joint sampler + class-balanced sentiment
    {
        'name': 'clean_joint_balanced_focal',
        'train_key': 'clean',
        'loss_name': 'improved_focal',
        'use_class_weights': True,
        'sent_weight_key': 'class_balanced',
        'train_sampler': 'joint_balanced',
        'sampler_temperature': 0.5,
        'sampler_include_neutral_bucket': False,
        'sentiment_loss_weight': 0.3,
        'aspect_loss_weight':    0.7,
        'scl_weight': 0.0,
        'focal_config': FOCAL_CONFIG_STANDARD,
    },
    # Clean + Improved Focal + neutral-focus aspect + joint sampler (neutral bucket)
    {
        'name': 'clean_joint_balanced_neutral_aspect',
        'train_key': 'clean',
        'loss_name': 'improved_focal',
        'use_class_weights': True,
        'sent_weight_key': 'neutral_capped',
        'train_sampler': 'joint_balanced',
        'sampler_temperature': NEUTRAL_SAMPLER_TEMPERATURE,
        'sampler_include_neutral_bucket': True,
        'sentiment_loss_weight': 0.3,
        'aspect_loss_weight':    0.7,
        'scl_weight': 0.0,
        'focal_config': FOCAL_CONFIG_NEUTRAL,
    },
    # Clean + Improved Focal + neutral focus + SCL + joint sampler (neutral bucket)
    {
        'name': 'clean_joint_balanced_neutral_scl',
        'train_key': 'clean',
        'loss_name': 'improved_focal',
        'use_class_weights': True,
        'sent_weight_key': 'neutral_capped',
        'train_sampler': 'joint_balanced',
        'sampler_temperature': NEUTRAL_SAMPLER_TEMPERATURE,
        'sampler_include_neutral_bucket': True,
        'sentiment_loss_weight': 0.3,
        'aspect_loss_weight':    0.7,
        'scl_weight': 0.15,
        'focal_config': FOCAL_CONFIG_NEUTRAL,
    },
]


# %% [code cell 27]
def run_experiment(cfg: dict) -> dict:
    train_frame   = train_variants[cfg["train_key"]]
    train_encoded = tokenize_frame(train_frame)

    model = ABSAModel.from_pretrained(MODEL_NAME).to(device)
    output_dir    = f'./absa_results/{cfg["name"]}'
    training_args = build_training_args(output_dir)

    sent_weights = None
    if cfg['use_class_weights'] and cfg.get('sent_weight_key'):
        sent_weights = sent_weight_tensors[cfg['sent_weight_key']]

    train_sampler = None
    if cfg.get('train_sampler') == 'joint_balanced':
        train_sampler, sampler_summary = build_joint_label_sampler(
            train_frame,
            temperature=cfg.get('sampler_temperature', SAMPLER_TEMPERATURE),
            include_neutral_bucket=cfg.get('sampler_include_neutral_bucket', False),
        )
        print('Joint sampler preview:')
        print(sampler_summary.head(12).to_string(index=False))

    # --- ĐOẠN MỚI THÊM VÀO: Khởi tạo Visualizer Callback ---
    visualizer_callback = EmbeddingVisualizerCallback(
        val_dataset=val_encoded,
        tokenizer=tokenizer,
        output_dir=output_dir, 
        label_names=SENTIMENT_LABELS,
        per_device_batch_size=BATCH_SIZE * 2
    )
    # -------------------------------------------------------

    trainer = ABSAMultiTaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_encoded,
        eval_dataset=val_encoded,
        compute_metrics=compute_metrics,
        sent_weights=sent_weights,
        loss_name=cfg['loss_name'],
        use_class_weights=cfg['use_class_weights'],
        focal_gamma=BASE_FOCAL_GAMMA,
        focal_config=cfg.get('focal_config', FOCAL_CONFIG_STANDARD),
        sentiment_loss_weight=cfg.get('sentiment_loss_weight', DEFAULT_SENTIMENT_LOSS_WEIGHT),
        aspect_loss_weight=cfg.get('aspect_loss_weight', DEFAULT_ASPECT_LOSS_WEIGHT),
        scl_weight=cfg.get('scl_weight', 0.0),
        train_sampler=train_sampler,
        callbacks=[visualizer_callback] # <-- THÊM CALLBACK VÀO ĐÂY
    )

    print()
    print(f"=== {cfg['name']} | train={len(train_frame):,} | loss={cfg['loss_name']} | scl={cfg['scl_weight']} ===")
    trainer.train()

    val_metrics = trainer.evaluate(val_encoded)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Dọn checkpoint trung gian
    del trainer
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    for ckpt in Path(output_dir).glob('checkpoint-*'):
        shutil.rmtree(ckpt, ignore_errors=True)

    return {
        'name':                 cfg['name'],
        'train_key':            cfg['train_key'],
        'train_rows':           len(train_frame),
        'loss_name':            cfg['loss_name'],
        'use_class_weights':    cfg['use_class_weights'],
        'sent_weight_key':      cfg.get('sent_weight_key'),
        'train_sampler':        cfg.get('train_sampler'),
        'sampler_temperature':  cfg.get('sampler_temperature'),
        'sampler_include_neutral_bucket': cfg.get('sampler_include_neutral_bucket', False),
        'scl_weight':           cfg.get('scl_weight', 0.0),
        'output_dir':           output_dir,
        'focal_config':         cfg.get('focal_config', FOCAL_CONFIG_STANDARD),
        'val_metrics':          val_metrics,
    }


# %% [code cell 28]
experiment_results = []
for cfg in EXPERIMENTS:
    experiment_results.append(run_experiment(cfg))


# %% [code cell 29]
# 1. Đường dẫn đến thư mục chứa ảnh
plot_dir = './absa_results/clean_joint_balanced_focal/convergence_plots'

# 2. Tìm tất cả các file .png trong thư mục và sắp xếp theo thứ tự bảng chữ cái (epoch_001 -> epoch_00n)
image_paths = sorted(glob.glob(f"{plot_dir}/*.png"))

# 3. Hiển thị ảnh
if not image_paths:
    print(f" Không tìm thấy ảnh nào trong thư mục: {plot_dir}")
    print("Vui lòng kiểm tra lại quá trình huấn luyện xem thư mục đã được tạo ra chưa.")
else:
    print(f"ĐANG HIỂN THỊ BIỂU ĐỒ HỘI TỤ ({len(image_paths)} Epochs)\n" + "="*50)
    for img_path in image_paths:
        file_name = os.path.basename(img_path)
        print(f"\n  Giai đoạn: {file_name}")
        # Hiển thị ảnh
        display(Image(filename=img_path))


# %% [code cell 30]
def mv(metrics, key):
    return metrics.get(f'eval_{key}', metrics.get(key))

rows = []
for result in experiment_results:
    rows.append({
        'experiment': result['name'],
        'train_key': result['train_key'],
        'train_rows': result['train_rows'],
        'loss_name': result['loss_name'],
        'class_balanced_weights': result['use_class_weights'],
        'train_sampler': result.get('train_sampler'),
        'sampler_temperature': result.get('sampler_temperature'),
        'sampler_neutral_bucket': result.get(
            'sampler_include_neutral_bucket', False
        ),
        'scl_weight': result['scl_weight'],
        'sent_weight_key': result.get('sent_weight_key'),
        'val_f1_sentiment': mv(result['val_metrics'], 'f1_sentiment'),
        'val_f1_aspect_present': mv(
            result['val_metrics'], 'f1_aspect_present'
        ),
        'val_f1_combined': mv(result['val_metrics'], 'f1_combined'),
        'val_accuracy': mv(result['val_metrics'], 'accuracy'),
    })

comparison_df = (
    pd.DataFrame(rows)
    .sort_values('val_f1_combined', ascending=False)
    .reset_index(drop=True)
)
display(comparison_df.round(4))

Path('experiments/reports').mkdir(parents=True, exist_ok=True)
comparison_df.to_csv(
    'experiments/reports/experiment_summary.csv',
    index=False,
)

best_result = next(
    result
    for result in experiment_results
    if result['name'] == comparison_df.iloc[0]['experiment']
)
print('Best experiment selected by validation F1 Combined:', best_result['name'])


# %% [code cell 31]
best_model_dir   = Path(best_result['output_dir'])
best_model       = ABSAModel.from_pretrained(best_model_dir).to(device)
best_tokenizer   = AutoTokenizer.from_pretrained(best_model_dir)
best_sent_weights = None
if best_result['use_class_weights'] and best_result.get('sent_weight_key'):
    best_sent_weights = sent_weight_tensors[best_result['sent_weight_key']]

eval_args = build_training_args(str(best_model_dir / 'eval_only'))
best_trainer = ABSAMultiTaskTrainer(
    model=best_model,
    args=eval_args,
    eval_dataset=val_encoded,
    compute_metrics=compute_metrics,
    sent_weights=best_sent_weights,
    loss_name=best_result['loss_name'],
    use_class_weights=best_result['use_class_weights'],
    focal_gamma=BASE_FOCAL_GAMMA,
    focal_config=best_result.get('focal_config', FOCAL_CONFIG_STANDARD),
    train_sampler=None,
)

val_result  = best_trainer.evaluate(val_encoded)
test_result = best_trainer.evaluate(test_encoded)
val_outputs  = best_trainer.predict(val_encoded)
test_outputs = best_trainer.predict(test_encoded)


def softmax_np(x):
    x = np.asarray(x)
    x = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(x)
    return exp / np.clip(exp.sum(axis=-1, keepdims=True), 1e-12, None)


def calibrate_presence_thresholds(logits, labels, neutral_weight: float = 0.0):
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    s = SENT_DIM
    p = N_ASPECTS * PRES_DIM
    pres_probs = softmax_np(logits[:, s:s+p].reshape(-1, N_ASPECTS, PRES_DIM))[:, :, 1]
    pred_asp_sent = np.argmax(logits[:, s+p:].reshape(-1, N_ASPECTS, ASP_SENT_DIM), axis=-1)
    thresholds = {}
    rows = []
    candidates = np.linspace(0.05, 0.95, 19)
    for i, col in enumerate(ASPECT_COLS):
        y_true = labels[:, 1 + i]
        present_mask = y_true != 3
        scores = pres_probs[:, i]
        best_t, best_obj, best_f1, best_neu_f1, best_p, best_r = 0.5, -1.0, 0.0, 0.0, 0.0, 0.0
        for t in candidates:
            y_pred = np.where(scores >= t, pred_asp_sent[:, i], 3)
            if present_mask.any():
                precision, recall, f1_macro, _ = precision_recall_fscore_support(
                    y_true[present_mask], y_pred[present_mask], labels=[0, 1, 2], average='macro', zero_division=0
                )
                neutral_f1 = precision_recall_fscore_support(
                    y_true[present_mask], y_pred[present_mask], labels=[1], average='macro', zero_division=0
                )[2]
            else:
                precision, recall, f1_macro, neutral_f1 = 0.0, 0.0, 0.0, 0.0

            objective = (1.0 - neutral_weight) * f1_macro + neutral_weight * neutral_f1
            if objective > best_obj:
                best_t = float(t)
                best_obj = float(objective)
                best_f1 = float(f1_macro)
                best_neu_f1 = float(neutral_f1)
                best_p = float(precision)
                best_r = float(recall)

        thresholds[col] = best_t
        rows.append({
            'aspect': col,
            'threshold': best_t,
            'macro_f1': best_f1,
            'neutral_f1': best_neu_f1,
            'objective': best_obj,
            'precision': best_p,
            'recall': best_r,
        })

    return thresholds, pd.DataFrame(rows).sort_values('objective', ascending=False)


def decode_with_thresholds(logits, thresholds):
    logits = np.asarray(logits)
    s = SENT_DIM
    p = N_ASPECTS * PRES_DIM
    pred_sent = np.argmax(logits[:, :s], axis=-1)
    pres_probs = softmax_np(logits[:, s:s+p].reshape(-1, N_ASPECTS, PRES_DIM))[:, :, 1]
    pred_presence = np.zeros((logits.shape[0], N_ASPECTS), dtype=int)
    for i, col in enumerate(ASPECT_COLS):
        pred_presence[:, i] = (pres_probs[:, i] >= thresholds.get(col, 0.5)).astype(int)
    pred_asp_sent = np.argmax(logits[:, s+p:].reshape(-1, N_ASPECTS, ASP_SENT_DIM), axis=-1)
    pred_asps = np.where(pred_presence == 0, 3, pred_asp_sent)
    return pred_sent, pred_presence, pred_asp_sent, pred_asps


best_presence_thresholds, presence_threshold_df = calibrate_presence_thresholds(
    val_outputs.predictions,
    val_outputs.label_ids,
    neutral_weight=THRESHOLD_NEUTRAL_WEIGHT,
)
val_sent_cal, val_presence_cal, val_asp_sent_cal, val_asps_cal = decode_with_thresholds(
    val_outputs.predictions, best_presence_thresholds
)
test_sent_cal, test_presence_cal, test_asp_sent_cal, test_asps_cal = decode_with_thresholds(
    test_outputs.predictions, best_presence_thresholds
)

true_sent_val  = val_df['sentiment'].values
true_asps_val  = val_df[ASPECT_COLS].values
true_sent_test = test_df['sentiment'].values
true_asps_test = test_df[ASPECT_COLS].values


def metrics_from_decoded(true_sent, pred_sent, true_asps, pred_asps):
    """Tính metric chuẩn sau khi decode bằng threshold đã cho."""
    f1_sentiment = precision_recall_fscore_support(
        true_sent, pred_sent, labels=[0, 1, 2],
        average='macro', zero_division=0
    )[2]

    true_flat = true_asps.flatten()
    pred_flat = pred_asps.flatten()
    present_mask = true_flat != 3

    if present_mask.any():
        f1_aspect_present = precision_recall_fscore_support(
            true_flat[present_mask],
            pred_flat[present_mask],
            labels=[0, 1, 2],
            average='macro',
            zero_division=0,
        )[2]
        f1_aspect_neutral_present = precision_recall_fscore_support(
            true_flat[present_mask],
            pred_flat[present_mask],
            labels=[1],
            average='macro',
            zero_division=0,
        )[2]
    else:
        f1_aspect_present = 0.0
        f1_aspect_neutral_present = 0.0

    f1_aspect_all = precision_recall_fscore_support(
        true_flat, pred_flat, labels=[0, 1, 2, 3],
        average='macro', zero_division=0
    )[2]

    return {
        'f1_sentiment': round(float(f1_sentiment), 4),
        'f1_aspect_present': round(float(f1_aspect_present), 4),
        'f1_combined': round(
            float(0.5 * f1_sentiment + 0.5 * f1_aspect_present), 4
        ),
        'accuracy': round(float(accuracy_score(true_sent, pred_sent)), 4),
        'diagnostic_f1_aspect_all': round(float(f1_aspect_all), 4),
        'diagnostic_f1_aspect_neutral_present': round(
            float(f1_aspect_neutral_present), 4
        ),
    }


val_cal_metrics = metrics_from_decoded(
    true_sent_val, val_sent_cal, true_asps_val, val_asps_cal
)
test_cal_metrics = metrics_from_decoded(
    true_sent_test, test_sent_cal, true_asps_test, test_asps_cal
)

selected_configuration = pd.DataFrame([{
    'joint_balanced sampler': (
        best_result.get('train_sampler') == 'joint_balanced'
    ),
    'class-balanced weights': bool(best_result.get('use_class_weights')),
    'improved focal loss': (
        best_result.get('loss_name') == 'improved_focal'
    ),
    'threshold calibration': True,
    'threshold fitted on': 'validation',
    'threshold applied to': 'validation + test',
}])

def standard_row(split, mode, metrics):
    return {
        'Split': split,
        'Mode': mode,
        'F1 Sentiment': metrics['f1_sentiment'],
        'F1 Aspect-Present': metrics['f1_aspect_present'],
        'F1 Combined': metrics['f1_combined'],
        'Accuracy': metrics['accuracy'],
    }

val_raw_metrics = {
    'f1_sentiment': val_result['eval_f1_sentiment'],
    'f1_aspect_present': val_result['eval_f1_aspect_present'],
    'f1_combined': val_result['eval_f1_combined'],
    'accuracy': val_result['eval_accuracy'],
}
test_raw_metrics = {
    'f1_sentiment': test_result['eval_f1_sentiment'],
    'f1_aspect_present': test_result['eval_f1_aspect_present'],
    'f1_combined': test_result['eval_f1_combined'],
    'accuracy': test_result['eval_accuracy'],
}

raw_calibrated_table = pd.DataFrame([
    standard_row('Validation', 'Raw argmax', val_raw_metrics),
    standard_row(
        'Validation', 'Calibrated threshold', val_cal_metrics
    ),
    standard_row('Test', 'Raw argmax', test_raw_metrics),
    standard_row('Test', 'Calibrated threshold', test_cal_metrics),
])

print('Selected experiment:', best_result['name'])
print('\nConfiguration actually used by selected experiment:')
display(selected_configuration)

print('\nPresence thresholds fitted only on VALIDATION:')
display(presence_threshold_df.round(4))

print('\nRAW vs CALIBRATED - STANDARD METRICS')
display(raw_calibrated_table.round(4))

print('\nDiagnostics (not used for ranking):')
display(pd.DataFrame([
    {
        'Split': 'Validation',
        'Mode': 'Calibrated threshold',
        'F1 Aspect-All': val_cal_metrics[
            'diagnostic_f1_aspect_all'
        ],
        'F1 Neutral Aspect-Present': val_cal_metrics[
            'diagnostic_f1_aspect_neutral_present'
        ],
    },
    {
        'Split': 'Test',
        'Mode': 'Calibrated threshold',
        'F1 Aspect-All': test_cal_metrics[
            'diagnostic_f1_aspect_all'
        ],
        'F1 Neutral Aspect-Present': test_cal_metrics[
            'diagnostic_f1_aspect_neutral_present'
        ],
    },
]).round(4))


# %% [code cell 32]
raw_pred = test_outputs.predictions
s = SENT_DIM
p = N_ASPECTS * PRES_DIM

pred_sent = test_sent_cal
pred_asps = test_asps_cal

true_sent  = test_df['sentiment'].values
true_asps  = test_df[ASPECT_COLS].values

print('=== OVERALL SENTIMENT (TEST SET) ===')
print(classification_report(
    true_sent, pred_sent, labels=[0,1,2],
    target_names=['Negative', 'Neutral', 'Positive'], zero_division=0))

print('=== 6 ASPECTS: present only (TEST SET, calibrated threshold) ===')
present_mask = true_asps.flatten() != 3
print(classification_report(
    true_asps.flatten()[present_mask],
    pred_asps.flatten()[present_mask],
    labels=[0,1,2],
    target_names=['Negative', 'Neutral', 'Positive'], zero_division=0))


# %% [code cell 33]
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
axes = axes.flatten()
NAMES = ['Negative', 'Neutral', 'Positive']
ASP_DISPLAY = ['Content', 'Physical', 'Price', 'Packaging', 'Delivery', 'Service']

cm_sent = confusion_matrix(true_sent, pred_sent, labels=[0,1,2])
sns.heatmap(cm_sent, annot=True, fmt='d', cmap='Blues',
            xticklabels=NAMES, yticklabels=NAMES, ax=axes[0], cbar=False)
axes[0].set_title('Overall Sentiment', fontweight='bold')
axes[0].set_ylabel('True'); axes[0].set_xlabel('Predicted')

for j, (col, disp) in enumerate(zip(ASPECT_COLS, ASP_DISPLAY)):
    ax  = axes[j + 1]
    idx = ASPECT_COLS.index(col)
    tc  = true_asps[:, idx]; pc = pred_asps[:, idx]
    mask = tc != 3
    if not mask.any():
        ax.set_visible(False); continue
    cm_asp = confusion_matrix(tc[mask], pc[mask], labels=[0,1,2])
    sns.heatmap(cm_asp, annot=True, fmt='d', cmap='Oranges',
                xticklabels=NAMES, yticklabels=NAMES, ax=ax, cbar=False)
    ax.set_title(f'{disp} (n={mask.sum()}, thr={best_presence_thresholds.get(col, 0.5):.2f})', fontweight='bold')
    ax.set_ylabel('True'); ax.set_xlabel('Predicted')

if len(axes) > 7:
    axes[7].set_visible(False)

plt.suptitle(f'Confusion Matrices - {best_result["name"]} (TEST SET, calibrated thresholds)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('confusion_matrix_test.png', dpi=150, bbox_inches='tight')
plt.show()


# %% [code cell 34]
from pathlib import Path


def decode_raw_outputs(logits):
    logits = np.asarray(logits)
    s = SENT_DIM
    p = N_ASPECTS * PRES_DIM
    pred_sent_raw = np.argmax(logits[:, :s], axis=-1)
    pred_presence_raw = np.argmax(logits[:, s:s+p].reshape(-1, N_ASPECTS, PRES_DIM), axis=-1)
    pred_asp_sent_raw = np.argmax(logits[:, s+p:].reshape(-1, N_ASPECTS, ASP_SENT_DIM), axis=-1)
    pred_asps_raw = np.where(pred_presence_raw == 0, 3, pred_asp_sent_raw)
    return pred_sent_raw, pred_asps_raw


def build_error_analysis_df(texts, true_sent, pred_sent, true_asps, pred_asps, mode_name: str, present_only: bool = False):
    rows = []
    for idx, text in enumerate(texts):
        sent_ok = int(true_sent[idx]) == int(pred_sent[idx])
        aspect_mismatches = []
        aspects_checked = 0

        for j, col in enumerate(ASPECT_COLS):
            true_label = int(true_asps[idx, j])
            pred_label = int(pred_asps[idx, j])
            if present_only and true_label == 3:
                continue
            aspects_checked += 1
            if true_label != pred_label:
                aspect_mismatches.append(col)

        if sent_ok and not aspect_mismatches:
            continue

        rows.append({
            'mode': mode_name,
            'idx': idx,
            'text': text,
            'true_sentiment': SENTIMENT_LABELS[int(true_sent[idx])],
            'pred_sentiment': SENTIMENT_LABELS[int(pred_sent[idx])],
            'sent_ok': sent_ok,
            'present_only': present_only,
            'n_aspects_checked': aspects_checked,
            'n_aspect_errors': len(aspect_mismatches),
            'aspect_errors': ', '.join(aspect_mismatches) if aspect_mismatches else '',
            'true_aspects': ' | '.join(
                f"{col}:{ASPECT_LABELS[int(true_asps[idx, j])]}" for j, col in enumerate(ASPECT_COLS)
            ),
            'pred_aspects': ' | '.join(
                f"{col}:{ASPECT_LABELS[int(pred_asps[idx, j])]}" for j, col in enumerate(ASPECT_COLS)
            ),
        })

    return pd.DataFrame(rows)


def summarize_aspect_error_rates(true_asps, pred_asps, present_only: bool = False):
    rows = []
    for j, col in enumerate(ASPECT_COLS):
        if present_only:
            mask = true_asps[:, j] != 3
        else:
            mask = np.ones(true_asps.shape[0], dtype=bool)

        checked = int(mask.sum())
        if checked == 0:
            err_cnt = 0
            err_rate = 0.0
        else:
            err_cnt = int((true_asps[mask, j] != pred_asps[mask, j]).sum())
            err_rate = 100.0 * err_cnt / checked

        rows.append({
            'aspect': col,
            'checked_rows': checked,
            'error_count': err_cnt,
            'error_rate_pct': round(err_rate, 2),
        })

    return pd.DataFrame(rows).sort_values(['error_rate_pct', 'error_count'], ascending=False).reset_index(drop=True)


def print_error_overview(name, true_sent, pred_sent, true_asps, pred_asps, present_only: bool):
    sent_err_mask = true_sent != pred_sent

    if present_only:
        aspect_check_mask = true_asps != 3
    else:
        aspect_check_mask = np.ones_like(true_asps, dtype=bool)

    aspect_mismatch_mask = (true_asps != pred_asps) & aspect_check_mask
    aspect_err_row_mask = aspect_mismatch_mask.any(axis=1)
    any_err_row_mask = sent_err_mask | aspect_err_row_mask

    sent_err = int(sent_err_mask.sum())
    aspect_err = int(aspect_err_row_mask.sum())
    both_err = int((sent_err_mask & aspect_err_row_mask).sum())
    any_err = int(any_err_row_mask.sum())
    total = int(len(true_sent))

    print(f'[{name}] Rows with any error: {any_err:,} / {total:,}')
    print(f'[{name}] Sentiment errors: {sent_err:,} | Aspect-row errors: {aspect_err:,} | Both: {both_err:,}')


def _truncate_text(s, max_len: int):
    s = str(s).replace('\n', ' ')
    if len(s) <= max_len:
        return s
    return s[:max_len] + ' ...'


def display_error_samples(err_df, top_n=20, show_full_text=True, text_len=260, aspect_len=220):
    if err_df.empty:
        print('No rows to display.')
        return pd.DataFrame()

    show_cols = [
        'idx', 'text', 'true_sentiment', 'pred_sentiment', 'sent_ok',
        'present_only', 'n_aspects_checked', 'n_aspect_errors', 'aspect_errors',
        'true_aspects', 'pred_aspects',
    ]

    view = err_df.sort_values(['n_aspect_errors', 'sent_ok'], ascending=[False, True]).head(top_n).copy()

    if not show_full_text:
        view['text'] = view['text'].map(lambda x: _truncate_text(x, text_len))
        view['aspect_errors'] = view['aspect_errors'].map(lambda x: _truncate_text(x, 140))
        view['true_aspects'] = view['true_aspects'].map(lambda x: _truncate_text(x, aspect_len))
        view['pred_aspects'] = view['pred_aspects'].map(lambda x: _truncate_text(x, aspect_len))

    with pd.option_context('display.max_colwidth', None, 'display.max_rows', top_n):
        display(
            view[show_cols].style.set_properties(
                subset=['text', 'aspect_errors', 'true_aspects', 'pred_aspects'],
                **{'white-space': 'pre-wrap', 'text-align': 'left'}
            )
        )

    return view


true_sent = test_df['sentiment'].values
true_asps = test_df[ASPECT_COLS].values
pred_sent_cal = test_sent_cal
pred_asps_cal = test_asps_cal
pred_sent_raw, pred_asps_raw = decode_raw_outputs(test_outputs.predictions)

views = [
    ('RAW_ALL_LABELS', pred_sent_raw, pred_asps_raw, False),
    ('CALIBRATED_ALL_LABELS', pred_sent_cal, pred_asps_cal, False),
    ('CALIBRATED_PRESENT_ONLY', pred_sent_cal, pred_asps_cal, True),
]

report_dir = Path('experiments/reports')
report_dir.mkdir(parents=True, exist_ok=True)
TOP_N = 20
SHOW_FULL_TEXT = True  # set False if output is too wide

for name, pred_sent_view, pred_asps_view, present_only in views:
    print()
    print('=' * 24 + f' {name} ' + '=' * 24)

    print_error_overview(name, true_sent, pred_sent_view, true_asps, pred_asps_view, present_only)

    err_df = build_error_analysis_df(
        test_df['text'].values,
        true_sent,
        pred_sent_view,
        true_asps,
        pred_asps_view,
        mode_name=name,
        present_only=present_only,
    )

    if err_df.empty:
        print(f'[{name}] No error rows.')
        top_df = err_df
    else:
        top_df = display_error_samples(err_df, top_n=TOP_N, show_full_text=SHOW_FULL_TEXT)

    aspect_summary_df = summarize_aspect_error_rates(
        true_asps=true_asps,
        pred_asps=pred_asps_view,
        present_only=present_only,
    )
    print(f'[{name}] Aspect error summary:')
    display(aspect_summary_df)

    err_df.to_csv(report_dir / f'error_rows_{name.lower()}.csv', index=False)
    top_df.to_csv(report_dir / f'error_rows_{name.lower()}_top{TOP_N}.csv', index=False)
    aspect_summary_df.to_csv(report_dir / f'aspect_error_summary_{name.lower()}.csv', index=False)


# %% [code cell 35]
final_dir = Path('./absa_results') / f'final_model_{best_result["name"]}'
final_dir.mkdir(parents=True, exist_ok=True)
best_model.save_pretrained(final_dir)
best_tokenizer.save_pretrained(final_dir)
print('Saved final model to:', final_dir)


# %% [code cell 36]
def predict_review(text: str, model, tokenizer, thresholds: dict) -> dict:
    import torch
    import numpy as np
    
    # 1. Chuẩn bị định dạng prompt văn bản y hệt lúc training
    input_text = f"{ABSA_PROMPT_PREFIX}{str(text).strip()}"
    
    # 2. Mã hóa (Tokenize)
    enc = tokenizer(
        input_text, 
        padding='max_length',
        truncation=True, 
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )
    
    input_ids = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)
    
    # 3. Dự đoán (Tắt tính toán Gradient để tiết kiệm bộ nhớ)
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
    logits = outputs.logits.cpu().numpy()
    
    # 4. Sử dụng bộ giải mã có áp dụng Threshold tối ưu
    pred_sent, _, _, pred_asps = decode_with_thresholds(logits, thresholds)
    
    sentiment_map = {0: 'Tiêu cực', 1: 'Trung lập', 2: 'Tích cực', 3: 'Không nhắc đến'}
    
    result = {
        'overall_sentiment': sentiment_map.get(pred_sent[0], 'Unknown'),
        'aspects': {}
    }
    
    # 5. Format lại danh sách aspect
    for i, col in enumerate(ASPECT_COLS):
        asp_val = pred_asps[0, i]
        if asp_val != 3: 
            aspect_name = col.replace('as_', '')
            result['aspects'][aspect_name] = sentiment_map.get(asp_val, 'Unknown')
            
    return result

import json

print("================= THỬ NGHIỆM INFERENCE VS GROUND TRUTH =================")
# Lấy ngẫu nhiên vài mẫu từ Test set để kiểm tra khả năng dự đoán thực tế
sample_df = test_df.sample(n=5)
sentiment_map = {0: 'Tiêu cực', 1: 'Trung lập', 2: 'Tích cực', 3: 'Không nhắc đến'}

for idx, row in sample_df.iterrows():
    text = row['text']
    print(f"\n[Mẫu ID: {idx}]")
    print(f"Bình luận: '{text}'")
    
    print("Thực tế:")
    print(f" - Overall: {sentiment_map.get(row['sentiment'], 'Unknown')}")
    
    print("Dự đoán (PhoBERT):")
    # (RESET threshold về 0.5 vì trước đó bị chọn quá thấp do cách tính cũ)
    fallback_th = {c: 0.5 for c in ASPECT_COLS}
    output = predict_review(text, best_model, best_tokenizer, fallback_th)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print("-" * 60)
