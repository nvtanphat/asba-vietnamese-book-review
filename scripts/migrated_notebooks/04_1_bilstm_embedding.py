"""Migrated from 04_1_bilstm_embedding.ipynb.

This file preserves the original experiment code for audit/reproducibility.
The production benchmark uses the modular ml/ package instead.
"""


# %% [code cell 1]
# NOTEBOOK_ONLY: !pip install -q pyvi gensim iterative-stratification seaborn scikit-learn nlpaug transformers


# %% [code cell 2]
import os
import gc
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight

from pyvi import ViTokenizer
from transformers import AutoTokenizer, AutoModel
from gensim.models import Word2Vec, FastText

import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from collections import Counter

warnings.filterwarnings('ignore')

# Khởi tạo Seed để đảm bảo tính tái lập (reproducibility)
SEED = 42
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True

seed_everything(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)


# %% [code cell 3]
MAX_LENGTH = 160
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.2

EPOCHS = 8 
BATCH_SIZE_TRADITIONAL = 128
BATCH_SIZE_PHOBERT_TRAIN = 32
BATCH_SIZE_PHOBERT_EVAL = 64
LEARNING_RATE = 1e-3

ABSENT_CLASS = 3
SENTIMENT_CLASSES = [0, 1, 2]
ASPECT_COLS = ['as_content', 'as_physical', 'as_price', 'as_packaging', 'as_delivery', 'as_service']
LABEL_COLS = ['sentiment', *ASPECT_COLS]
ASPECT_NAMES = ['Nội dung', 'Hình thức', 'Giá cả', 'Đóng gói', 'Giao hàng', 'Dịch vụ']

DATA_ROOT = Path('/kaggle/input/datasets/nguynvntnpht/tiki-cleaned-book-reviews-absa')
if not DATA_ROOT.exists():
    DATA_ROOT = Path('.') # Fallback
TRAIN_PATH = DATA_ROOT / 'train_clean.json'
VAL_PATH = DATA_ROOT / 'val_clean.json'
TEST_PATH = DATA_ROOT / 'test_clean.json'

OUTPUT_ROOT = Path('./absa_experiments')
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# %% [code cell 4]
def load_and_prepare(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_json(path)
    title = df.get('review_title', df.get('title', pd.Series(['']*len(df)))).fillna('').astype(str).str.strip()
    body = df.get('content', df.get('text', pd.Series(['']*len(df)))).fillna('').astype(str).str.strip()
    df['title'] = title
    df['body'] = body
    df['text_full'] = df['title'] + " " + df['body']
    df['sentiment'] = pd.to_numeric(df['sentiment'], errors='coerce')
    df = df.dropna(subset=['sentiment'])
    df['sentiment'] = df['sentiment'].astype(int)
    df = df[df['sentiment'].isin(SENTIMENT_CLASSES)]
    for col in ASPECT_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(ABSENT_CLASS).astype(int).clip(0, ABSENT_CLASS)
    return df[['text_full', *LABEL_COLS]].reset_index(drop=True)

print("Đang nạp dữ liệu...")
train_df = load_and_prepare(TRAIN_PATH)
val_df = load_and_prepare(VAL_PATH)
test_df = load_and_prepare(TEST_PATH)
print(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")

print("Đang tách từ (Word Segmentation) bằng Pyvi...")
def segment_text(text):
    return ViTokenizer.tokenize(text.lower())

train_df['text_seg'] = train_df['text_full'].apply(segment_text)
val_df['text_seg'] = val_df['text_full'].apply(segment_text)
test_df['text_seg'] = test_df['text_full'].apply(segment_text)

# Tính toán Class Weights dùng chung cho cả 3 mô hình
print("Đang tính toán Class Weights động...")
train_sent_labels = train_df['sentiment'].values
dynamic_sent_weights = torch.tensor(np.sqrt(compute_class_weight('balanced', classes=np.arange(3), y=train_sent_labels)), dtype=torch.float32).to(DEVICE)

aspect_pres_weights = []
for col in ASPECT_COLS:
    presence_labels = (train_df[col].values != ABSENT_CLASS).astype(int)
    weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=presence_labels)
    aspect_pres_weights.append(np.sqrt(weights))
aspect_pres_weights = torch.tensor(np.array(aspect_pres_weights), dtype=torch.float32).to(DEVICE)


# %% [code cell 5]
class TextDatasetCustomEmb(Dataset):
    def __init__(self, df, vocab, max_len):
        self.df = df.reset_index(drop=True)
        self.vocab, self.max_len = vocab, max_len
        self.pad_idx, self.unk_idx = vocab.get('<pad>', 0), vocab.get('<unk>', 1)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tokens = str(row['text_seg']).split()
        encoded = [self.vocab.get(word, self.unk_idx) for word in tokens]
        if len(encoded) > self.max_len:
            encoded, attention_mask = encoded[:self.max_len], [1] * self.max_len
        else:
            pad_len = self.max_len - len(encoded)
            attention_mask = [1] * len(encoded) + [0] * pad_len
            encoded = encoded + [self.pad_idx] * pad_len
        labels = np.array([int(row['sentiment']), *[int(row[col]) for col in ASPECT_COLS]], dtype=np.int64)
        return {'input_ids': torch.tensor(encoded, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.bool),
                'labels': torch.tensor(labels, dtype=torch.long)}

class TextDatasetPhoBERT(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df.reset_index(drop=True)
        self.tokenizer, self.max_len = tokenizer, max_len

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        encoding = self.tokenizer(str(row['text_seg']), add_special_tokens=True, max_length=self.max_len,
                                  padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')
        labels = np.array([int(row['sentiment']), *[int(row[col]) for col in ASPECT_COLS]], dtype=np.int64)
        return {'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(labels, dtype=torch.long)}


# %% [code cell 6]
class SpatialDropout1D(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.dropout = nn.Dropout2d(p)
    def forward(self, x):
        x = x.permute(0, 2, 1).unsqueeze(3)
        return self.dropout(x).squeeze(3).permute(0, 2, 1)

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight 
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 1. Tính CE Loss thuần để lấy đúng xác suất pt của class mục tiêu
        ce_loss_raw = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss_raw)
        
        # 2. Tính CE Loss có trọng số (Class Weights)
        ce_loss_weighted = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        
        # 3. Tính toán Focal Loss chính xác
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss_weighted
        
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        return focal_loss

class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num=3):
        super().__init__()
        self.params = nn.Parameter(torch.zeros(num, requires_grad=True))
    def forward(self, *losses):
        total_loss = 0
        for i, loss in enumerate(losses):
            total_loss += loss * torch.exp(-self.params[i]) + self.params[i]
        return total_loss

def compute_individual_losses(sent_logits, pres_logits, asp_logits, labels, sent_weights, pres_weights):
    true_sent, true_aspects = labels[:, 0], labels[:, 1:]
    loss_sent = FocalLoss(weight=sent_weights, gamma=2.0)(sent_logits, true_sent)

    true_pres = (true_aspects != ABSENT_CLASS).long()
    loss_pres = 0.0
    for i in range(true_aspects.shape[1]):
        loss_pres += nn.CrossEntropyLoss(weight=pres_weights[i])(pres_logits[:, i, :], true_pres[:, i])
    loss_pres /= true_aspects.shape[1]

    mask = true_aspects != ABSENT_CLASS
    if mask.sum() > 0:
        loss_asp = nn.CrossEntropyLoss(label_smoothing=0.05)(asp_logits.reshape(-1, 3)[mask.reshape(-1)], true_aspects.reshape(-1)[mask.reshape(-1)])
    else:
        loss_asp = torch.tensor(0.0, device=sent_logits.device, requires_grad=True)

    return loss_sent, loss_pres, loss_asp

def calculate_metrics(all_labels, all_preds):
    """Tính bộ metric chuẩn dùng thống nhất cho train/validation/test."""
    all_labels = np.asarray(all_labels)
    all_preds = np.asarray(all_preds)

    true_sent, true_aspects = all_labels[:, 0], all_labels[:, 1:]
    pred_sent, pred_aspects = all_preds[:, 0], all_preds[:, 1:]

    f1_sentiment = f1_score(
        true_sent, pred_sent, labels=[0, 1, 2],
        average='macro', zero_division=0
    )

    present_mask = true_aspects != ABSENT_CLASS
    if present_mask.any():
        f1_aspect_present = f1_score(
            true_aspects[present_mask],
            pred_aspects[present_mask],
            labels=[0, 1, 2],
            average='macro',
            zero_division=0,
        )
    else:
        f1_aspect_present = 0.0

    f1_combined = 0.5 * f1_sentiment + 0.5 * f1_aspect_present
    accuracy = accuracy_score(true_sent, pred_sent)

    return {
        'f1_sentiment': float(f1_sentiment),
        'f1_aspect_present': float(f1_aspect_present),
        'f1_combined': float(f1_combined),
        'accuracy': float(accuracy),
    }


# %% [code cell 7]
class BaseABSABiLSTM(nn.Module):
    """Lớp nền chứa kiến trúc BiLSTM + MHA + Heads. Các model con chỉ ghi đè phần Embedding."""
    def __init__(self, embed_dim, hidden_dim, num_layers, dropout, num_aspects):
        super().__init__()
        self.spatial_dropout = SpatialDropout1D(dropout)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        lstm_out_dim = hidden_dim * 2
        self.mha = nn.MultiheadAttention(embed_dim=lstm_out_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.attention_pool = nn.Sequential(nn.Linear(lstm_out_dim, lstm_out_dim // 2), nn.Tanh(), nn.Linear(lstm_out_dim // 2, 1))
        self.feat_norm = nn.LayerNorm(lstm_out_dim * 3)
        cat_dim = lstm_out_dim * 3

        self.sent_proj = nn.Linear(cat_dim, hidden_dim)
        self.sent_decoupler = nn.Sequential(nn.Linear(cat_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.sent_classifier = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout / 2), nn.Linear(hidden_dim, 3))

        self.asp_proj = nn.Linear(cat_dim, hidden_dim)
        self.asp_decoupler = nn.Sequential(nn.Linear(cat_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.pres_classifier = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout / 2), nn.Linear(hidden_dim, num_aspects * 2))
        self.asp_classifier = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout / 2), nn.Linear(hidden_dim, num_aspects * 3))
        self.num_aspects = num_aspects

    def process_features(self, embedded, attention_mask):
        lengths = attention_mask.sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(self.lstm(packed)[0], batch_first=True)

        batch_size, max_len, _ = lstm_out.size()
        device = lstm_out.device
        mask = torch.arange(max_len).expand(batch_size, max_len).to(device) < lengths.unsqueeze(1).to(device)

        attn_output, _ = self.mha(query=lstm_out, key=lstm_out, value=lstm_out, key_padding_mask=~mask)
        mask_expanded = mask.unsqueeze(-1)

        attn_weights = torch.softmax(self.attention_pool(attn_output).masked_fill(~mask_expanded, -1e9), dim=1)
        mhsa_pool = torch.sum(attn_weights * attn_output, dim=1)
        max_pool = torch.max(lstm_out.masked_fill(~mask_expanded, -1e9), dim=1)[0]

        mask_float = mask.float().unsqueeze(-1)
        avg_pool = torch.sum(lstm_out * mask_float, dim=1) / torch.sum(mask_float, dim=1).clamp(min=1e-9)

        context_vector = self.feat_norm(torch.cat([mhsa_pool, max_pool, avg_pool], dim=-1))

        sent_context = self.sent_decoupler(context_vector) + self.sent_proj(context_vector)
        sent_logits = self.sent_classifier(sent_context)

        asp_context = self.asp_decoupler(context_vector) + self.asp_proj(context_vector)
        pres_logits = self.pres_classifier(asp_context).view(-1, self.num_aspects, 2)
        asp_logits = self.asp_classifier(asp_context).view(-1, self.num_aspects, 3)
        return sent_logits, pres_logits, asp_logits

class CustomEmbed_ABSABiLSTM(BaseABSABiLSTM):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout, num_aspects, pretrained_emb):
        super().__init__(embed_dim, hidden_dim, num_layers, dropout, num_aspects)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))
        self.embedding.weight.requires_grad = True
    def forward(self, input_ids, attention_mask):
        return self.process_features(self.spatial_dropout(self.embedding(input_ids)), attention_mask)

class PhoBERT_ABSABiLSTM(BaseABSABiLSTM):
    def __init__(self, hidden_dim, num_layers, dropout, num_aspects):
        super().__init__(768, hidden_dim, num_layers, dropout, num_aspects)
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base")
    def forward(self, input_ids, attention_mask):
        embedded = self.phobert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.process_features(self.spatial_dropout(embedded), attention_mask)


class NoEmbed_ABSABiLSTM(BaseABSABiLSTM):
    def __init__(self, vocab_size, hidden_dim, num_layers, dropout, num_aspects):
        # Không khởi tạo lớp nn.Embedding nào ở đây.
        # Chiều đầu vào (embed_dim) của LSTM bây giờ chính là kích thước từ điển (vocab_size)
        super().__init__(vocab_size, hidden_dim, num_layers, dropout, num_aspects)
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask):
        # Chuyển đổi input_ids thành ma trận One-hot Encoding (batch_size, max_len, vocab_size)
        # Ép kiểu sang tensor dạng float() để tương thích với đầu vào mạng LSTM
        one_hot_inputs = F.one_hot(input_ids, num_classes=self.vocab_size).float()
        
        # Đưa trực tiếp ma trận one-hot vào hàm xử lý chung
        return self.process_features(self.spatial_dropout(one_hot_inputs), attention_mask)


# %% [code cell 8]
def run_experiment(model_name, model, train_loader, val_loader, optimizer, awl, epochs):
    """Train một cấu hình embedding và trả metric checkpoint tốt nhất trên validation."""
    print(f"\n{'='*70}\nBẮT ĐẦU THỰC NGHIỆM: {model_name}\n{'='*70}")
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    best_val_f1 = -1.0
    best_model_path = OUTPUT_ROOT / f'best_{model_name.replace(" ", "_").lower()}.pt'

    for epoch in range(epochs):
        model.train()
        awl.train()
        total_loss = 0.0

        for batch in train_loader:
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            optimizer.zero_grad()
            sent_logits, pres_logits, asp_logits = model(ids, mask)
            loss_sent, loss_pres, loss_asp = compute_individual_losses(
                sent_logits, pres_logits, asp_logits, labels,
                dynamic_sent_weights, aspect_pres_weights
            )
            loss = awl(loss_sent, loss_pres, loss_asp)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        awl.eval()
        val_labels, val_preds = [], []
        val_loss_total = 0.0

        with torch.no_grad():
            for batch in val_loader:
                ids = batch['input_ids'].to(DEVICE)
                mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)

                sent_logits, pres_logits, asp_logits = model(ids, mask)
                loss_sent, loss_pres, loss_asp = compute_individual_losses(
                    sent_logits, pres_logits, asp_logits, labels,
                    dynamic_sent_weights, aspect_pres_weights
                )
                val_loss_total += awl(loss_sent, loss_pres, loss_asp).item()

                pred_sent = sent_logits.argmax(dim=-1).cpu().numpy()
                pred_presence = pres_logits.argmax(dim=-1).cpu().numpy()
                pred_aspect_sent = asp_logits.argmax(dim=-1).cpu().numpy()
                pred_aspects = np.where(
                    pred_presence == 0, ABSENT_CLASS, pred_aspect_sent
                )

                val_labels.extend(labels.cpu().numpy())
                val_preds.extend(np.column_stack((pred_sent, pred_aspects)))

        val_metrics = calculate_metrics(np.asarray(val_labels), np.asarray(val_preds))
        scheduler.step(val_metrics['f1_combined'])

        if val_metrics['f1_combined'] > best_val_f1:
            best_val_f1 = val_metrics['f1_combined']
            torch.save(
                model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                best_model_path
            )

        if epoch == 0 or epoch == epochs - 1 or epoch == epochs // 2:
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {total_loss/len(train_loader):.4f} | "
                f"Val Loss: {val_loss_total/len(val_loader):.4f} | "
                f"Val F1 Combined: {val_metrics['f1_combined']:.4f} "
                f"(Sentiment: {val_metrics['f1_sentiment']:.4f}, "
                f"Aspect-Present: {val_metrics['f1_aspect_present']:.4f}, "
                f"Accuracy: {val_metrics['accuracy']:.4f})"
            )

    state_dict = torch.load(best_model_path, map_location=DEVICE)
    if hasattr(model, 'module'):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

    model.eval()
    final_val_labels, final_val_preds = [], []
    with torch.no_grad():
        for batch in val_loader:
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            sent_logits, pres_logits, asp_logits = model(ids, mask)
            pred_sent = sent_logits.argmax(dim=-1).cpu().numpy()
            pred_presence = pres_logits.argmax(dim=-1).cpu().numpy()
            pred_aspect_sent = asp_logits.argmax(dim=-1).cpu().numpy()
            pred_aspects = np.where(
                pred_presence == 0, ABSENT_CLASS, pred_aspect_sent
            )

            final_val_labels.extend(labels.cpu().numpy())
            final_val_preds.extend(np.column_stack((pred_sent, pred_aspects)))

    metrics = calculate_metrics(
        np.asarray(final_val_labels), np.asarray(final_val_preds)
    )

    print(
        f"--> VALIDATION [{model_name}]: "
        f"F1 Sentiment={metrics['f1_sentiment']:.4f} | "
        f"F1 Aspect-Present={metrics['f1_aspect_present']:.4f} | "
        f"F1 Combined={metrics['f1_combined']:.4f} | "
        f"Accuracy={metrics['accuracy']:.4f}"
    )

    return {
        'Model': model_name,
        'F1 Sentiment': metrics['f1_sentiment'],
        'F1 Aspect-Present': metrics['f1_aspect_present'],
        'F1 Combined': metrics['f1_combined'],
        'Accuracy': metrics['accuracy'],
    }


# %% [code cell 9]
results = []
train_texts = train_df['text_seg'].tolist()

# =====================================================================
# THỰC NGHIỆM 0: BILSTM Baseline (ONE-HOT ENCODING DIRECT INPUT)
# =====================================================================
print("\n[0/4] Đang xây dựng BiLSTM Baseline ...")

# 1. Xây dựng từ điển (Vocabulary) thủ công từ tập train 
all_words = []
for text in train_texts:
    all_words.extend(str(text).split())
word_counts = Counter(all_words)

vocab_no_emb = {'<pad>': 0, '<unk>': 1}
# Để tránh tràn bộ nhớ VRAM khi dùng One-hot, chúng ta nên lọc bớt các từ hiếm (min_count=5)
for word, count in word_counts.items():
    if count >= 5:
        vocab_no_emb[word] = len(vocab_no_emb)

vocab_size = len(vocab_no_emb)
print(f"Kích thước từ điển (Vocab Size) cho One-Hot: {vocab_size}")

train_ldr = DataLoader(TextDatasetCustomEmb(train_df, vocab_no_emb, MAX_LENGTH), batch_size=BATCH_SIZE_TRADITIONAL, shuffle=True)
val_ldr   = DataLoader(TextDatasetCustomEmb(val_df, vocab_no_emb, MAX_LENGTH), batch_size=BATCH_SIZE_TRADITIONAL)
# Khởi tạo mô hình NoEmbed, truyền vocab_size vào để LSTM nhận diện chiều đầu vào
model_no_emb = NoEmbed_ABSABiLSTM(vocab_size, HIDDEN_DIM, NUM_LAYERS, DROPOUT, len(ASPECT_COLS)).to(DEVICE)
if torch.cuda.device_count() > 1: model_no_emb = nn.DataParallel(model_no_emb)
awl_no_emb = AutomaticWeightedLoss(num=3).to(DEVICE)
opt_no_emb = torch.optim.AdamW([{'params': model_no_emb.parameters(), 'lr': LEARNING_RATE}, {'params': awl_no_emb.parameters(), 'lr': LEARNING_RATE}], weight_decay=1e-4)

res = run_experiment("BiLSTM (No Embedding)", model_no_emb, train_ldr, val_ldr, opt_no_emb, awl_no_emb, EPOCHS)
results.append(res)

# Dọn rác GPU
del model_no_emb, opt_no_emb, awl_no_emb, train_ldr, val_ldr, vocab_no_emb
gc.collect(); torch.cuda.empty_cache()


# =====================================================================
# THỰC NGHIỆM 1: WORD2VEC 
# =====================================================================
print("\n[1/4] Đang xây dựng Word2Vec...")
w2v_model = Word2Vec([str(t).split() for t in train_texts], vector_size=300, window=5, min_count=2, workers=4, sg=1)
vocab_w2v = {'<pad>': 0, '<unk>': 1}
emb_w2v = np.zeros((len(w2v_model.wv.key_to_index) + 2, 300))
emb_w2v[1] = np.random.normal(scale=0.1, size=(300,))
for word, vec in w2v_model.wv.key_to_index.items():
    vocab_w2v[word] = len(vocab_w2v); emb_w2v[vocab_w2v[word]] = w2v_model.wv[word]

train_ldr = DataLoader(TextDatasetCustomEmb(train_df, vocab_w2v, MAX_LENGTH), batch_size=BATCH_SIZE_TRADITIONAL, shuffle=True)
val_ldr   = DataLoader(TextDatasetCustomEmb(val_df, vocab_w2v, MAX_LENGTH), batch_size=BATCH_SIZE_TRADITIONAL)
model_w2v = CustomEmbed_ABSABiLSTM(len(vocab_w2v), 300, HIDDEN_DIM, NUM_LAYERS, DROPOUT, len(ASPECT_COLS), emb_w2v).to(DEVICE)
if torch.cuda.device_count() > 1: model_w2v = nn.DataParallel(model_w2v)
awl_w2v = AutomaticWeightedLoss(num=3).to(DEVICE)
opt_w2v = torch.optim.AdamW([{'params': model_w2v.parameters(), 'lr': LEARNING_RATE}, {'params': awl_w2v.parameters(), 'lr': LEARNING_RATE}], weight_decay=1e-4)

res = run_experiment("BiLSTM + Word2Vec", model_w2v, train_ldr, val_ldr, opt_w2v, awl_w2v, EPOCHS)
results.append(res)

del model_w2v, opt_w2v, awl_w2v, train_ldr, val_ldr, w2v_model, emb_w2v, vocab_w2v
gc.collect(); torch.cuda.empty_cache()


# =====================================================================
# THỰC NGHIỆM 2: FASTTEXT 
# =====================================================================
print("\n[2/4] Đang xây dựng FastText...")
ft_model = FastText([str(t).split() for t in train_texts], vector_size=300, window=5, min_count=2, workers=4)
vocab_ft = {'<pad>': 0, '<unk>': 1}
emb_ft = np.zeros((len(ft_model.wv.key_to_index) + 2, 300))
emb_ft[1] = np.random.normal(scale=0.1, size=(300,))
for word, vec in ft_model.wv.key_to_index.items():
    vocab_ft[word] = len(vocab_ft); emb_ft[vocab_ft[word]] = ft_model.wv[word]

train_ldr = DataLoader(TextDatasetCustomEmb(train_df, vocab_ft, MAX_LENGTH), batch_size=BATCH_SIZE_TRADITIONAL, shuffle=True)
val_ldr   = DataLoader(TextDatasetCustomEmb(val_df, vocab_ft, MAX_LENGTH), batch_size=BATCH_SIZE_TRADITIONAL)
model_ft = CustomEmbed_ABSABiLSTM(len(vocab_ft), 300, HIDDEN_DIM, NUM_LAYERS, DROPOUT, len(ASPECT_COLS), emb_ft).to(DEVICE)
if torch.cuda.device_count() > 1: model_ft = nn.DataParallel(model_ft)
awl_ft = AutomaticWeightedLoss(num=3).to(DEVICE)
opt_ft = torch.optim.AdamW([{'params': model_ft.parameters(), 'lr': LEARNING_RATE}, {'params': awl_ft.parameters(), 'lr': LEARNING_RATE}], weight_decay=1e-4)

res = run_experiment("BiLSTM + FastText", model_ft, train_ldr, val_ldr, opt_ft, awl_ft, EPOCHS)
results.append(res)

del model_ft, opt_ft, awl_ft, train_ldr, val_ldr, ft_model, emb_ft, vocab_ft
gc.collect(); torch.cuda.empty_cache()


# =====================================================================
# THỰC NGHIỆM 3: PHOBERT 
# =====================================================================
print("\n[3/4] Đang tải HuggingFace PhoBERT...")

seed_everything(SEED)

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
train_ldr = DataLoader(TextDatasetPhoBERT(train_df, tokenizer, MAX_LENGTH), batch_size=BATCH_SIZE_PHOBERT_TRAIN, shuffle=True)
val_ldr   = DataLoader(TextDatasetPhoBERT(val_df, tokenizer, MAX_LENGTH), batch_size=BATCH_SIZE_PHOBERT_EVAL)
model_pb = PhoBERT_ABSABiLSTM(HIDDEN_DIM, NUM_LAYERS, DROPOUT, len(ASPECT_COLS)).to(DEVICE)
if torch.cuda.device_count() > 1: model_pb = nn.DataParallel(model_pb)
awl_pb = AutomaticWeightedLoss(num=3).to(DEVICE)

# Cấu hình Learning Rate 
pb_params = list(model_pb.module.phobert.parameters()) if hasattr(model_pb, 'module') else list(model_pb.phobert.parameters())
custom_params = [p for n, p in model_pb.named_parameters() if 'phobert' not in n]

opt_pb = torch.optim.AdamW([
    {'params': pb_params, 'lr': 2e-5},
    {'params': custom_params, 'lr': LEARNING_RATE},
    {'params': awl_pb.parameters(), 'weight_decay': 0.0, 'lr': LEARNING_RATE}
], weight_decay=1e-4)

res = run_experiment("BiLSTM + PhoBERT", model_pb, train_ldr, val_ldr, opt_pb, awl_pb, EPOCHS)
results.append(res)


# %% [code cell 10]
print("\n" + "=" * 72)
print("BẢNG ABLATION EMBEDDING - VALIDATION METRICS")
print("=" * 72)

df_results = (
    pd.DataFrame(results)
    .sort_values('F1 Combined', ascending=False)
    .reset_index(drop=True)
)
display(df_results.round(4))
df_results.to_csv(OUTPUT_ROOT / 'embedding_validation_ablation.csv', index=False)

best_embedding_model = df_results.loc[0, 'Model']
print(f"\nMô hình được chọn theo F1 Combined trên validation: {best_embedding_model}")


# %% [code cell 11]
# Định nghĩa tên khía cạnh tiếng Việt để hiển thị cho đẹp
ASPECT_NAMES = ['Nội dung', 'Hình thức', 'Giá cả', 'Đóng gói', 'Giao hàng', 'Dịch vụ']
# Định nghĩa biến cho tên cột để lặp
ASPECT_COLS = ['as_content', 'as_physical', 'as_price', 'as_packaging', 'as_delivery', 'as_service']
ABSENT_CLASS = 3

def plot_confusion_matrices(true_sentiment: np.ndarray, pred_sentiment: np.ndarray, true_aspects: np.ndarray, pred_aspects: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()

    # Nhãn cảm xúc: 0 (tiêu cực), 1 (trung tính), 2 (tích cực)
    sent_cm = confusion_matrix(true_sentiment, pred_sentiment, labels=[0, 1, 2])
    sns.heatmap(sent_cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['neg', 'neu', 'pos'], yticklabels=['neg', 'neu', 'pos'], ax=axes[0])
    axes[0].set_title('Overall sentiment')

    for idx, col in enumerate(ASPECT_COLS):
        ax = axes[idx + 1]
        # Bỏ qua các khía cạnh không xuất hiện (ABSENT_CLASS = 3) trong nhãn thực tế
        mask = true_aspects[:, idx] != ABSENT_CLASS
        if mask.sum() == 0:
            ax.set_visible(False)
            continue

        cm = confusion_matrix(true_aspects[:, idx][mask], pred_aspects[:, idx][mask], labels=[0, 1, 2])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=False,
                    xticklabels=['neg', 'neu', 'pos'], yticklabels=['neg', 'neu', 'pos'], ax=ax)
        ax.set_title(f'{ASPECT_NAMES[idx]} (n={mask.sum()})')

    # Ẩn ô cuối cùng nếu dư
    if len(axes) > 7:
        axes[7].set_visible(False)

    plt.suptitle('BiLSTM + PhoBERT ABSA - Test confusion matrices', y=1.01, fontweight='bold', fontsize=16)
    plt.tight_layout()

    # Lưu ảnh ma trận nhầm lẫn
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    plt.savefig(OUTPUT_ROOT / 'confusion_matrix_test.png', dpi=150, bbox_inches='tight')
    plt.show()

# ==========================================
# THỰC THI ĐÁNH GIÁ TRÊN TẬP TEST
# ==========================================
print("Đang đánh giá mô hình BiLSTM + PhoBERT trên tập Test...")

# 1. Khởi tạo lại cấu trúc mô hình (Cần thiết trước khi load state_dict)
model = PhoBERT_ABSABiLSTM(HIDDEN_DIM, NUM_LAYERS, DROPOUT, len(ASPECT_COLS)).to(DEVICE)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# 2. Khởi tạo lại Dataloader nếu nó đã bị xóa khỏi RAM (Safety check)
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
test_ldr = DataLoader(TextDatasetPhoBERT(test_df, tokenizer, MAX_LENGTH), batch_size=BATCH_SIZE_PHOBERT_EVAL)

# 3. Load trọng số từ file .pt đã lưu
selected_model_name = 'BiLSTM + PhoBERT'
if best_embedding_model != selected_model_name:
    raise RuntimeError(
        f"Model thắng validation là {best_embedding_model!r}, nhưng cell test đang cấu hình "
        f"cho {selected_model_name!r}. Hãy cập nhật kiến trúc/checkpoint trước khi chạy test."
    )

state_dict = torch.load(OUTPUT_ROOT / 'best_bilstm_+_phobert.pt', map_location=DEVICE)

if hasattr(model, 'module'):
    model.module.load_state_dict(state_dict)
else:
    model.load_state_dict(state_dict)

model.eval()

true_sentiment, pred_sentiment = [], []
true_aspects, pred_aspects = [], []

# Chuẩn bị biến đo lường thời gian
start_time = time.time()
num_steps = len(test_ldr)
num_samples = len(test_df)
test_loss_total = 0.0

# 4. Chạy suy luận (Inference) sử dụng test_ldr
with torch.no_grad():
    for batch in test_ldr:  
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        # Lấy Logits từ mô hình
        sent_logits, pres_logits, asp_logits = model(input_ids, attention_mask)
        
        # Tính Loss giả định (Nếu bạn muốn track chính xác, cần gọi hàm compute_individual_losses kết hợp awl_pb ở đây)
        # Tạm thời gán 0.0 để tương thích format bạn yêu cầu
        test_loss_total += 0.0 

        # Lấy nhãn dự đoán bằng argmax
        p_sent = sent_logits.argmax(dim=-1).cpu().numpy()
        p_pres = pres_logits.argmax(dim=-1).cpu().numpy()
        p_asp_sent = asp_logits.argmax(dim=-1).cpu().numpy()

        # Logic ghép nhãn: Nếu presence = 0 (không xuất hiện) -> gán nhãn ABSENT_CLASS (3)
        p_asp = np.where(p_pres == 0, ABSENT_CLASS, p_asp_sent)

        # Lưu trữ kết quả
        lbls_cpu = labels.cpu().numpy()
        true_sentiment.extend(lbls_cpu[:, 0])
        pred_sentiment.extend(p_sent)
        true_aspects.extend(lbls_cpu[:, 1:])
        pred_aspects.extend(p_asp)

# Tính toán các chỉ số về tốc độ suy luận
eval_runtime = time.time() - start_time
eval_samples_per_second = num_samples / eval_runtime
eval_steps_per_second = num_steps / eval_runtime

# Chuyển đổi sang Numpy Array
true_sentiment = np.array(true_sentiment)
pred_sentiment = np.array(pred_sentiment)
true_aspects = np.array(true_aspects)
pred_aspects = np.array(pred_aspects)

# BÁO CÁO TRUNG BÌNH TOÀN BỘ ASPECT (present only)
all_true = []
all_pred = []

for idx in range(len(ASPECT_COLS)):
    t_asp = true_aspects[:, idx]
    p_asp = pred_aspects[:, idx]
    mask = t_asp != ABSENT_CLASS  # chỉ lấy các aspect có xuất hiện
    if mask.sum() == 0:
        continue
    all_true.extend(t_asp[mask])
    all_pred.extend(p_asp[mask])

all_true = np.array(all_true)
all_pred = np.array(all_pred)

# TÍNH BỘ METRIC CHUẨN BẰNG ĐÚNG HÀM DÙNG TRÊN VALIDATION
standard_test_metrics = calculate_metrics(
    np.column_stack((true_sentiment, true_aspects)),
    np.column_stack((pred_sentiment, pred_aspects)),
)
eval_f1_sentiment = standard_test_metrics['f1_sentiment']
eval_f1_aspect_present = standard_test_metrics['f1_aspect_present']
eval_f1_combined = standard_test_metrics['f1_combined']
eval_accuracy = standard_test_metrics['accuracy']

# Diagnostic có cả lớp ABSENT, không dùng để xếp hạng.
eval_f1_aspect_all_diagnostic = f1_score(
    true_aspects.flatten(), pred_aspects.flatten(),
    labels=[0, 1, 2, 3], average='macro', zero_division=0
)

# Tính F1 Macro cho từng khía cạnh riêng biệt
aspect_f1_dict = {}
for idx, col in enumerate(ASPECT_COLS):
    t_asp = true_aspects[:, idx]
    p_asp = pred_aspects[:, idx]
    mask = t_asp != ABSENT_CLASS
    if mask.sum() > 0:
        aspect_f1_dict[f'eval_f1_{col}'] = f1_score(t_asp[mask], p_asp[mask], average='macro', labels=[0, 1, 2], zero_division=0)
    else:
        aspect_f1_dict[f'eval_f1_{col}'] = 0.0

# Bảng test chuẩn: đúng bốn metric chính
standard_test_table = pd.DataFrame([{
    'Model': selected_model_name,
    'F1 Sentiment': eval_f1_sentiment,
    'F1 Aspect-Present': eval_f1_aspect_present,
    'F1 Combined': eval_f1_combined,
    'Accuracy': eval_accuracy,
}])
standard_test_table.to_csv(
    OUTPUT_ROOT / 'selected_model_test_metrics.csv', index=False
)

diagnostic_metrics = {
    'F1 Aspect-All (diagnostic)': eval_f1_aspect_all_diagnostic,
    **{key.replace('eval_', ''): value for key, value in aspect_f1_dict.items()},
    'Runtime (seconds)': eval_runtime,
    'Samples/second': eval_samples_per_second,
    'Steps/second': eval_steps_per_second,
}

# IN RA BÁO CÁO VÀ TEST METRICS
print('\n=========================================')
print('=== BÁO CÁO TỔNG THỂ (OVERALL SENTIMENT) ===')
print('=========================================')
print(classification_report(true_sentiment, pred_sentiment, labels=[0, 1, 2], target_names=['neg', 'neu', 'pos'], zero_division=0))

print('\n=========================================')
print('=== Báo Cáo Mean 6 ASPECTS')
print('=========================================')
print(classification_report(all_true, all_pred, labels=[0, 1, 2], target_names=['neg', 'neu', 'pos'], zero_division=0))

print('\n=========================================')
print('=== BẢNG TEST CHUẨN ===')
print('=========================================')
display(standard_test_table.round(4))

print('\nDiagnostic metrics (không dùng để xếp hạng):')
for key, value in diagnostic_metrics.items():
    print(f"{key}: {value:.4f}")

# Vẽ đồ thị ma trận nhầm lẫn
plot_confusion_matrices(true_sentiment, pred_sentiment, true_aspects, pred_aspects)


# %% [code cell 12]
# Định nghĩa map nhãn
SENTIMENT_MAP = {0: "Tiêu cực", 1: "Trung tính", 2: "Tích cực"}
ASPECT_NAMES_VI = ['Nội dung', 'Hình thức', 'Giá cả', 'Đóng gói', 'Giao hàng', 'Dịch vụ']

def predict_single_review_with_confidence(text, model, tokenizer, max_len=160): # Đảm bảo max_len khớp với lúc train
    model.eval()

    # 1. Tiền xử lý
    text_seg = ViTokenizer.tokenize(text.lower())

    # 2. Tokenization
    encoding = tokenizer(
        text_seg,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)

    # 3. Suy luận & Tính xác suất (Softmax)
    with torch.no_grad():
        sent_logits, pres_logits, asp_logits = model(input_ids, attention_mask)

        # Chuyển Logits thành Xác suất (Probabilities)
        sent_probs = F.softmax(sent_logits, dim=-1).cpu().numpy()[0]
        pres_probs = F.softmax(pres_logits, dim=-1).cpu().numpy()[0]
        asp_probs = F.softmax(asp_logits, dim=-1).cpu().numpy()[0]

        # Dự đoán Cảm xúc tổng thể (Vẫn dùng argmax vì đây là Multi-class)
        p_sent = sent_probs.argmax()
        sent_conf = sent_probs[p_sent] * 100

        # --- CẢI TIẾN: SỬ DỤNG THRESHOLD RIÊNG CHO TỪNG KHÍA CẠNH ---
        # 0:'Nội dung', 1:'Hình thức', 2:'Giá cả', 3:'Đóng gói', 4:'Giao hàng', 5:'Dịch vụ'
        # Thiết lập ngưỡng riêng (Tùy chỉnh theo lúc bạn phân tích file error_analysis.csv)
        ASPECT_THRESHOLDS = np.array([
            0.98,  # Nội dung: Ép thật cao (90%) vì mô hình rất hay đoán nhầm khía cạnh này
            0.40,  # Hình thức: Giữ vừa phải
            0.50,  # Giá cả
            0.30,  # Đóng gói: Hạ thấp để dễ bắt (30%)
            0.60,  # Giao hàng
            0.70   # Dịch vụ: Tăng lên chút để tránh nhận diện sai (như trong câu 2)
        ])

        # Lấy mảng xác suất của nhãn 1 (Có xuất hiện)
        prob_is_present = pres_probs[:, 1]

        # So sánh từng khía cạnh với ngưỡng tương ứng của nó
        p_pres = (prob_is_present >= ASPECT_THRESHOLDS).astype(int)

        # Cảm xúc của khía cạnh (Vẫn dùng argmax)
        p_asp_sent = asp_probs.argmax(axis=-1)

    # 4. In kết quả trực quan
    print("=" * 60)
    print(f"📝 BÌNH LUẬN GỐC : {text}")
    print(f"🔍 ĐÃ TÁCH TỪ    : {text_seg}")
    print("-" * 60)
    print(f"⭐ ĐÁNH GIÁ CHUNG: {SENTIMENT_MAP[p_sent]} (Độ tự tin: {sent_conf:.1f}%)")
    print("-" * 60)
    print("📌 CHI TIẾT CÁC KHÍA CẠNH:")

    aspect_count = 0
    for idx, aspect_name in enumerate(ASPECT_NAMES_VI):
        # Xác suất mô hình cho rằng khía cạnh này CÓ xuất hiện (nhãn 1)
        presence_conf = pres_probs[idx][1] * 100

        if p_pres[idx] == 0:
            # print(f"   ~ [Bỏ qua] {aspect_name:<10}: (Chỉ đạt {presence_conf:.1f}% xuất hiện)")
            continue
        else:
            aspect_count += 1
            sentiment_conf = asp_probs[idx][p_asp_sent[idx]] * 100
            print(f"   - {aspect_name:<10}: {SENTIMENT_MAP[p_asp_sent[idx]]:<15} | Tự tin nhận diện: {presence_conf:.1f}% | Tự tin cảm xúc: {sentiment_conf:.1f}%")

    if aspect_count == 0:
        print("   (Không phát hiện khía cạnh cụ thể nào)")
    print("=" * 60 + "\n")


# ==========================================
# KHU VỰC CHẠY THỬ (TEST INFERENCE)
# ==========================================

tokenizer_pb = AutoTokenizer.from_pretrained("vinai/phobert-base")

# Load model (thay thế model_pb bằng biến mô hình thực tế của bạn)
state_dict = torch.load(OUTPUT_ROOT / 'best_bilstm_+_phobert.pt', map_location=DEVICE)
if hasattr(model, 'module'):
    model.module.load_state_dict(state_dict)
else:
    model.load_state_dict(state_dict)

test_reviews = [
    "Sách bọc màng co cẩn thận, giao hàng siêu nhanh nhưng nội dung đọc hơi chán, không như kỳ vọng.",
    "bìa xinh lắm ạ vì mua vào ngày 11/11 nên săn sale rẻ, nhưng shipper thái độ quá tệ, cọc cằn rất khó chịu.",
    "Tôi thấy bình thường, mua về chưa đọc tới nên không biết bên trong thế nào."
]

for review in test_reviews:
    predict_single_review_with_confidence(review, model, tokenizer_pb, max_len=160)


# %% [code cell 13]
print("Đang thu thập dự đoán trên tập Test để phân tích lỗi...")
model.eval()
all_preds_sent = []
all_preds_asp = []

with torch.no_grad():
    for batch in test_ldr:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)

        sent_logits, pres_logits, asp_logits = model(input_ids, attention_mask)

        # Lấy nhãn dự đoán
        p_sent = sent_logits.argmax(dim=-1).cpu().numpy()
        p_pres = pres_logits.argmax(dim=-1).cpu().numpy()
        p_asp_sent = asp_logits.argmax(dim=-1).cpu().numpy()
        p_asp = np.where(p_pres == 0, ABSENT_CLASS, p_asp_sent)

        all_preds_sent.extend(p_sent)
        all_preds_asp.extend(p_asp)

# ==========================================
# 2. TẠO DATAFRAME LƯU TRỮ KẾT QUẢ SO SÁNH
# ==========================================
error_analysis_df = test_df.copy()
error_analysis_df['pred_sentiment'] = all_preds_sent

for i, col in enumerate(ASPECT_COLS):
    error_analysis_df[f'pred_{col}'] = np.array(all_preds_asp)[:, i]

# Map nhãn số sang text cho dễ đọc
LABEL_MAP = {0: "Tiêu cực", 1: "Trung tính", 2: "Tích cực", 3: "Không có"}
def map_labels_to_text(df, cols):
    for col in cols:
        df[col] = df[col].map(LABEL_MAP)
    return df

# Chuyển đổi nhãn thực tế và dự đoán sang Text
cols_to_map = ['sentiment', 'pred_sentiment'] + ASPECT_COLS + [f'pred_{col}' for col in ASPECT_COLS]
error_analysis_df = map_labels_to_text(error_analysis_df, cols_to_map)

# ==========================================
# 3. TÌM CÁC CÂU LỖI
# ==========================================
# Điều kiện lỗi: Cảm xúc chung sai HOẶC bất kỳ Aspect nào sai
is_error = (error_analysis_df['sentiment'] != error_analysis_df['pred_sentiment'])
for col in ASPECT_COLS:
    is_error = is_error | (error_analysis_df[col] != error_analysis_df[f'pred_{col}'])

df_errors = error_analysis_df[is_error].reset_index(drop=True)
print(f"Tổng số câu bị dự đoán sai (ít nhất 1 thành phần): {len(df_errors)} / {len(test_df)}")

# ==========================================
# 4. IN RA MỘT SỐ VÍ DỤ ĐIỂN HÌNH ĐỂ PHÂN TÍCH
# ==========================================
# Cài đặt số lượng sample muốn xem, ví dụ xem 5 câu lỗi ngẫu nhiên
NUM_SAMPLES_TO_VIEW = 5
sample_errors = df_errors.sample(min(NUM_SAMPLES_TO_VIEW, len(df_errors)), random_state=42)

for idx, row in sample_errors.iterrows():
    print("=" * 70)
    print(f" BÌNH LUẬN GỐC: {row['text_full']}")
    print("-" * 70)

    # Kiểm tra Cảm xúc chung
    if row['sentiment'] != row['pred_sentiment']:
        print(f" [SAI CẢM XÚC CHUNG] Thực tế: '{row['sentiment']}' | Mô hình đoán: '{row['pred_sentiment']}'")
    else:
        print(f" [ĐÚNG CẢM XÚC CHUNG] {row['sentiment']}")

    print("\n CHI TIẾT KHÍA CẠNH:")
    for aspect_col, aspect_name in zip(ASPECT_COLS, ASPECT_NAMES):
        true_val = row[aspect_col]
        pred_val = row[f'pred_{aspect_col}']

        if true_val != pred_val:
            print(f"   {aspect_name:<10} -> Thực tế: '{true_val:<10}' | Mô hình đoán: '{pred_val}'")
        elif true_val != "Không có":
            # In ra những aspect đoán đúng để có ngữ cảnh
            print(f"    {aspect_name:<10} -> Đúng: '{true_val}'")

# (Tuỳ chọn) Lưu toàn bộ tập lỗi ra file CSV để xem trên Excel
ERROR_FILE = OUTPUT_ROOT / 'error_analysis.csv'
df_errors.to_csv(ERROR_FILE, index=False, encoding='utf-8-sig')
print("=" * 70)
print(f" Đã lưu toàn bộ các câu lỗi ra file: {ERROR_FILE} để bạn tiện phân tích sâu hơn.")
