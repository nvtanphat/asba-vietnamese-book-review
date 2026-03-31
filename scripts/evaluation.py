import json
import pandas as pd
import torch
from sklearn.metrics import classification_report
from src.models.predictor import ABSAPredictor, ASPECT_COLS, ASPECT_NAMES

def run_evaluation(data_path="data/processed/test_clean.json"):
    print(f"--- Bắt đầu đánh giá mô hình PhoBERT trên tập Test ---")
    
    # 1. Load data
    df_test = pd.read_json(data_path)
    print(f"Số lượng mẫu test: {len(df_test)}")
    
    # 2. Khởi tạo Predictor (sử dụng phiên bản PhoBERT Edition)
    predictor = ABSAPredictor()
    
    # 3. Chạy dự đoán
    texts = df_test["content"].astype(str).tolist()
    predictions = predictor.predict(texts)
    
    # 4. Chuẩn bị dữ liệu để tính metrics
    # Sentiment tổng thể
    y_true_sent = df_test["sentiment_llm"].astype(int).tolist()
    y_pred_sent = [p["overall"] for p in predictions]
    
    print("\n" + "="*50)
    print("=== BÁO CÁO CẢM XÚC TỔNG THỂ (OVERALL SENTIMENT) ===")
    print("="*50)
    print(classification_report(y_true_sent, y_pred_sent, target_names=["neg", "neu", "pos"], zero_division=0))
    
    # Từng Aspect
    print("\n" + "="*50)
    print("=== BÁO CÁO CHI TIẾT TỪNG KHÍA CẠNH (ASPECTS) ===")
    print("="*50)
    
    for idx, col in enumerate(ASPECT_COLS):
        # Lọc các dòng thực sự có gán nhãn cho aspect này (khác null/-1)
        # null là không nhắc
        mask = df_test[col].notna()
        if mask.any():
            y_true_asp = df_test.loc[mask, col].astype(int).tolist()
            y_pred_asp = [predictions[i]["aspects"][col] for i in df_test[mask].index]
            
            print(f"\n--- Khía cạnh: {ASPECT_NAMES[idx].upper()} (n={mask.sum()}) ---")
            print(classification_report(y_true_asp, y_pred_asp, labels=[0, 1, 2], target_names=["neg", "neu", "pos"], zero_division=0))
        else:
            print(f"\n--- Khía cạnh: {ASPECT_NAMES[idx].upper()} (Không có dữ liệu test) ---")

if __name__ == "__main__":
    run_evaluation()
