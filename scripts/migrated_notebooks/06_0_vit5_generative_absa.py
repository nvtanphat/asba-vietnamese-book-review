"""Migrated from 06_0_vit5_generative_absa.ipynb.

This file preserves the original experiment code for audit/reproducibility.
The production benchmark uses the modular ml/ package instead.
"""


# %% [code cell 1]
# Gỡ bản PyTorch mặc định của Kaggle (tránh xung đột)
# NOTEBOOK_ONLY: !pip uninstall -y torch torchvision torchaudio

# Ép version đồng bộ cho cả transformers, accelerate và peft
# NOTEBOOK_ONLY: !pip install -q transformers==4.39.3 accelerate==0.28.0 peft==0.10.0 sentencepiece


# %% [code cell 2]
import json, os, random, warnings
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments, 
    DataCollatorForSeq2Seq,
    set_seed
)
from torch.utils.data import Dataset

from huggingface_hub import hf_hub_download
from transformers import T5Tokenizer, DataCollatorForSeq2Seq
from transformers import EarlyStoppingCallback
import pandas as pd
import re
from sklearn.utils import resample
from transformers import EarlyStoppingCallback, Seq2SeqTrainingArguments, Seq2SeqTrainer

import torch.nn as nn
from transformers import T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, TaskType

from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from torch.utils.data import Dataset

from sklearn.utils.class_weight import compute_class_weight

# Tắt cảnh báo TF/CUDA (giúp output sạch hơn, giải quyết các dòng E0000)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


# Thiết lập Random Seed
SEED = 42
set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Cấu hình tham số
MODEL_NAME = 'VietAI/vit5-base'
MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 64
EPOCHS = 8
BATCH_SIZE = 8
LEARNING_RATE = 2e-4 

TRAIN_PATH = '/kaggle/input/datasets/jyang10/tiki-cleaned-book-reviews/train_clean.json'
TEST_PATH = '/kaggle/input/datasets/jyang10/tiki-cleaned-book-reviews/test_clean.json'
VAL_PATH=  '/kaggle/input/datasets/jyang10/tiki-cleaned-book-reviews/val_clean.json'

OUTPUT_DIR = './vit5_absa_results'

ASPECT_COLS = ['as_content', 'as_physical', 'as_price', 'as_packaging', 'as_delivery', 'as_service']
ASPECT_NAMES = ['Content', 'Physical', 'Price', 'Packaging', 'Delivery', 'Service']

# Từ điển ánh xạ nhãn
LABEL_2_TEXT = {0: 'tiêu cực', 1: 'trung tính', 2: 'tích cực'}
TEXT_2_LABEL = {'tiêu cực': 0, 'trung tính': 1, 'tích cực': 2}
ABSENT_CLASS = 3

print("Device:", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


# %% [code cell 3]
def format_target_text(row):
    """Chuyển đổi nhãn dạng số thành chuỗi Text-to-Text mục tiêu."""
    parts = []
    # Cảm xúc tổng thể
    if pd.notna(row['sentiment']) and int(row['sentiment']) in LABEL_2_TEXT:
        parts.append(f"Tổng thể: {LABEL_2_TEXT[int(row['sentiment'])]}")
    else:
        parts.append(f"Tổng thể: {LABEL_2_TEXT[1]}") # default trung tính
        
    # Các khía cạnh
    for col, name in zip(ASPECT_COLS, ASPECT_NAMES):
        val = row[col]
        if pd.notna(val) and int(val) in LABEL_2_TEXT:
            parts.append(f"{name}: {LABEL_2_TEXT[int(val)]}")
            
    return ", ".join(parts)


def load_and_prepare_data(path, is_train=False):
    df = pd.read_json(path).copy()
    
    title = df.get('review_title', df.get('title', pd.Series(['']*len(df)))).fillna('').astype(str).str.strip()
    body = df.get('content', df.get('text', pd.Series(['']*len(df)))).fillna('').astype(str).str.strip()
    
    # CẢI TIẾN 1: Mở rộng định nghĩa Neutral trong Prompt
    instruction = (
        "Nhiệm vụ: Phân tích cảm xúc (tích cực, trung tính, tiêu cực) cho Tổng thể và các khía cạnh. "
        "Quy tắc quan trọng: Đánh giá 'trung tính' nếu câu mang tính chất trần thuật khách quan (không bộc lộ cảm xúc rõ ràng) "
        "HOẶC chứa cả ý khen lẫn chê ở mức độ tương đương nhau. "
        "Văn bản: "
    )
    
    df['input_text'] = instruction + title + ". " + body
    
    df['sentiment'] = pd.to_numeric(df['sentiment'], errors='coerce')
    df = df.dropna(subset=['sentiment']).copy()
    
    for col in ASPECT_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(ABSENT_CLASS).astype(int).clip(0, ABSENT_CLASS)
        
    df['target_text'] = df.apply(format_target_text, axis=1)
    
    # CẢI TIẾN 2: Tính toán Sample Weights dựa trên Overall Sentiment (Chỉ áp dụng cho tập Train)
    if is_train:
        y_train = df['sentiment'].values
        # Tính trọng số nghịch đảo: Class nào ít sẽ có trọng số cao
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        weight_dict = {cls: weight for cls, weight in zip(np.unique(y_train), class_weights)}
        
        # Tạo cột weight cho từng dòng dữ liệu
        df['sample_weight'] = df['sentiment'].map(weight_dict)
    else:
        df['sample_weight'] = 1.0 # Val và Test giữ nguyên trọng số là 1
    
    return df[['input_text', 'target_text', 'sentiment', 'sample_weight'] + ASPECT_COLS].reset_index(drop=True)

# Khởi tạo dữ liệu
train_df = load_and_prepare_data(TRAIN_PATH, is_train=True)
val_df   = load_and_prepare_data(VAL_PATH, is_train=False)
test_df  = load_and_prepare_data(TEST_PATH, is_train=False)


# %% [code cell 4]
print("Khởi tạo Tokenizer chuẩn...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

class Seq2SeqABSADataset(Dataset):
    def __init__(self, df, tokenizer, max_input_len=MAX_INPUT_LENGTH, max_target_len=MAX_TARGET_LENGTH):
        self.df = df
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        model_inputs = self.tokenizer(
            row['input_text'], 
            max_length=self.max_input_len, 
            truncation=True,
            padding=False
        )
        
        labels = self.tokenizer(
            text_target=row['target_text'], 
            max_length=self.max_target_len, 
            truncation=True,
            padding=False
        )
        
        model_inputs["labels"] = labels["input_ids"]
        # Thêm sample_weight vào dictionary (HuggingFace DataCollator sẽ tự động gom batch thành Tensor)
        model_inputs["sample_weights"] = row['sample_weight']
        
        return model_inputs

# Khởi tạo lại dataset
train_dataset = Seq2SeqABSADataset(train_df, tokenizer)
val_dataset = Seq2SeqABSADataset(val_df, tokenizer)
test_dataset = Seq2SeqABSADataset(test_df, tokenizer)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model if 'model' in locals() else None, padding="longest")

print("Khởi tạo Tokenizer và Dataset thành công!")


# %% [code cell 5]
def parse_generated_text(text):
    """Dịch ngược văn bản sinh ra thành mảng nhãn dạng số để tính Metrics."""
    res = {'sentiment': 1} # Khởi tạo mặc định
    for c in ASPECT_COLS: res[c] = ABSENT_CLASS
        
    # CHÚ Ý SỬA Ở ĐÂY: Split bằng dấu phẩy
    parts = [p.strip() for p in text.split(',')]
    
    for p in parts:
        if ':' not in p: continue
        key, val = p.split(':', 1)
        key, val = key.strip(), val.strip().lower()
        
        label = TEXT_2_LABEL.get(val, -1)
        if label != -1:
            if key == 'Tổng thể': res['sentiment'] = label
            elif key == 'Content': res['as_content'] = label
            elif key == 'Physical': res['as_physical'] = label
            elif key == 'Price': res['as_price'] = label
            elif key == 'Packaging': res['as_packaging'] = label
            elif key == 'Delivery': res['as_delivery'] = label
            elif key == 'Service': res['as_service'] = label
    return res
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
        
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    true_sent, pred_sent = [], []
    true_aspects, pred_aspects = [], []
    
    for d_pred, d_label in zip(decoded_preds, decoded_labels):
        p_dict = parse_generated_text(d_pred)
        l_dict = parse_generated_text(d_label)
        
        true_sent.append(l_dict['sentiment'])
        pred_sent.append(p_dict['sentiment'])
        
        true_aspects.append([l_dict[c] for c in ASPECT_COLS])
        pred_aspects.append([p_dict[c] for c in ASPECT_COLS])
        
    true_sent = np.array(true_sent)
    pred_sent = np.array(pred_sent)
    true_aspects = np.array(true_aspects)
    pred_aspects = np.array(pred_aspects)
    
    # Tính toán F1
    f1_sentiment = f1_score(true_sent, pred_sent, labels=[0, 1, 2], average='macro', zero_division=0)
    
    present_mask = true_aspects.flatten() != ABSENT_CLASS
    if present_mask.any():
        f1_aspect_present = f1_score(
            true_aspects.flatten()[present_mask],
            pred_aspects.flatten()[present_mask],
            labels=[0, 1, 2],
            average='macro',
            zero_division=0
        )
    else:
        f1_aspect_present = 0.0
        
    f1_final = 0.5 * f1_sentiment + 0.5 * f1_aspect_present
    
    return {
        "f1_sentiment": round(f1_sentiment, 4), 
        "f1_aspect_present": round(f1_aspect_present, 4),
        "f1_final": round(f1_final, 4)
    }


# %% [code cell 6]
print("Đang khởi tạo mô hình gốc ViT5...")
base_model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# CẢI TIẾN 3: Mở rộng LoRA target modules
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "k", "v", "o", "wi_0", "wi_1", "wo"], 
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# ==========================================
# CẢI TIẾN 4: CUSTOM TRAINER XỬ LÝ WEIGHTED LOSS & EVALUATION KHÔNG BỊ LỖI
# ==========================================
class WeightedSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Lấy nhãn và trọng số ra khỏi batch inputs
        labels = inputs.pop("labels")
        weights = inputs.pop("sample_weights", None) 
        
        # Forward pass lấy Logits
        outputs = model(**inputs)
        logits = outputs.logits 
        
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        loss = loss.view(logits.size(0), logits.size(1))
        mask = (labels != -100).float()
        
        loss_per_sequence = (loss * mask).sum(dim=1) / mask.sum(dim=1)
        
        if weights is not None:
            weights = weights.to(loss_per_sequence.device)
            loss_per_sequence = loss_per_sequence * weights
            
        loss = loss_per_sequence.mean()
        
        return (loss, outputs) if return_outputs else loss

    # FIX LỖI Ở ĐÂY: Can thiệp vào quá trình Prediction/Evaluation
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **gen_kwargs):
        # Lọc bỏ key 'sample_weights' (nếu có) trước khi đưa vào model.generate()
        filtered_inputs = {k: v for k, v in inputs.items() if k != "sample_weights"}
        
        # Trả lại dữ liệu sạch cho hàm gốc xử lý tiếp
        return super().prediction_step(
            model, 
            filtered_inputs, 
            prediction_loss_only, 
            ignore_keys=ignore_keys,
            **gen_kwargs
        )

# Data Collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest")

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",  
    save_strategy="epoch",
    learning_rate=3e-4, 
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,
    weight_decay=0.01, 
    save_total_limit=2,
    num_train_epochs=EPOCHS,
    predict_with_generate=True, 
    generation_max_length=MAX_TARGET_LENGTH,
    generation_num_beams=3,     
    load_best_model_at_end=True,     
    metric_for_best_model="f1_sentiment", 
    greater_is_better=True, 
    fp16=torch.cuda.is_available(),
    report_to="none"
)

# Khởi tạo Custom Trainer
trainer = WeightedSeq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer, 
    data_collator=data_collator, 
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)] 
)

print("Sẵn sàng huấn luyện với Weighted Custom Loss đã fix triệt để!")


# %% [code cell 7]
print("Bắt đầu quá trình Fine-tuning ViT5 bằng LoRA...")
trainer.train()

# Lưu mô hình cuối cùng
final_model_dir = './vit5_absa_lora_final'
trainer.save_model(final_model_dir)
tokenizer.save_pretrained(final_model_dir)                                                       
print(f"Đã lưu mô hình (LoRA adapter) tại: {final_model_dir}")


# %% [code cell 8]
def plot_confusion_matrices(true_sentiment: np.ndarray, pred_sentiment: np.ndarray, true_aspects: np.ndarray, pred_aspects: np.ndarray) -> None:
    # Tạo thư mục lưu ảnh nếu chưa có
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ==========================================
    # 1. Vẽ Confusion Matrix cho Overall Sentiment
    # ==========================================
    plt.figure(figsize=(6, 5))
    sent_cm = confusion_matrix(true_sentiment, pred_sentiment, labels=[0, 1, 2])
    sns.heatmap(sent_cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['neg', 'neu', 'pos'], yticklabels=['neg', 'neu', 'pos'],
                annot_kws={"size": 14}) # Tăng kích thước chữ bên trong ma trận
    
    plt.title('Overall Sentiment', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/cm_overall_sentiment.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ==========================================
    # 2. Vẽ Confusion Matrix cho từng Khía cạnh (Aspects)
    # ==========================================
    for idx, (col, aspect_name) in enumerate(zip(ASPECT_COLS, ASPECT_NAMES)):
        # Bỏ qua các khía cạnh không xuất hiện (ABSENT_CLASS = 3) trong nhãn thực tế
        mask = true_aspects[:, idx] != ABSENT_CLASS
        if mask.sum() == 0:
            continue
            
        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(true_aspects[:, idx][mask], pred_aspects[:, idx][mask], labels=[0, 1, 2])
        
        # Dùng colormap Oranges cho các khía cạnh để phân biệt với Overall (Blues)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=False, 
                    xticklabels=['neg', 'neu', 'pos'], yticklabels=['neg', 'neu', 'pos'],
                    annot_kws={"size": 14})
        
        plt.title(f'Aspect: {aspect_name} (n={mask.sum()})', fontsize=16, fontweight='bold', pad=15)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/cm_aspect_{aspect_name.lower()}.png', dpi=150, bbox_inches='tight')
        plt.show()


# ==========================================
# THỰC THI ĐÁNH GIÁ TRÊN TẬP TEST
# ==========================================
print("Đánh giá mô hình trên tập Test...")
raw_pred = trainer.predict(test_dataset)
print("Metrics từ Trainer:", raw_pred.metrics)

# 1. Giải mã (Decode) Token IDs sinh ra thành Văn bản tiếng Việt
preds = np.where(raw_pred.predictions != -100, raw_pred.predictions, tokenizer.pad_token_id)
decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

true_sentiment, pred_sentiment = [], []
true_aspects, pred_aspects = [], []

# 2. Phân tích (Parse) văn bản thành nhãn số
for idx, row in test_df.iterrows():
    p_dict = parse_generated_text(decoded_preds[idx])
    
    true_sentiment.append(int(row['sentiment']))
    pred_sentiment.append(p_dict['sentiment'])
    
    true_aspects.append([int(row[c]) for c in ASPECT_COLS])
    pred_aspects.append([p_dict[c] for c in ASPECT_COLS])

true_sentiment = np.array(true_sentiment)
pred_sentiment = np.array(pred_sentiment)
true_aspects = np.array(true_aspects)
pred_aspects = np.array(pred_aspects)

# 3. In Báo cáo phân loại (Classification Report)
print('\n=========================================')
print('=== BÁO CÁO TỔNG THỂ (OVERALL SENTIMENT) ===')
print('=========================================')
print(classification_report(true_sentiment, pred_sentiment, labels=[0, 1, 2], target_names=['neg', 'neu', 'pos'], zero_division=0))

# ==========================================
# 5. BÁO CÁO TRUNG BÌNH TOÀN BỘ ASPECT (present only)
# ==========================================

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

print('\n=========================================')
print('=== 6 ASPECTS')
print('=========================================')

print(classification_report(
    all_true,
    all_pred,
    labels=[0, 1, 2],
    target_names=['Negative', 'Neutral', 'Positive'],
    zero_division=0
))

print('\n=========================================')
print('=== BÁO CÁO CHI TIẾT TỪNG KHÍA CẠNH ===')
print('=========================================')
for idx, (col_name, aspect_name) in enumerate(zip(ASPECT_COLS, ASPECT_NAMES)):
    t_asp = true_aspects[:, idx]
    p_asp = pred_aspects[:, idx]
    
    mask = t_asp != ABSENT_CLASS
    
    print(f'\n--- Khía cạnh: {aspect_name.upper()} (n = {mask.sum()}) ---')
    
    if mask.sum() > 0:
        print(classification_report(
            t_asp[mask], 
            p_asp[mask], 
            labels=[0, 1, 2], 
            target_names=['neg', 'neu', 'pos'], 
            zero_division=0
        ))
    else:
        print(f"Không có dữ liệu thực tế cho khía cạnh này trong tập Test.")
