import pandas as pd
import os

# Đường dẫn tới các file dữ liệu đã làm sạch
data_files = {
    'Train': 'data/processed/train_clean.json',
    'Val': 'data/processed/val_clean.json',
    'Test': 'data/processed/test_clean.json'
}

# 6 cột khía cạnh cần kiểm tra
aspect_cols = ['as_content', 'as_physical', 'as_price', 'as_packaging', 'as_delivery', 'as_service']

def check_label_distribution(files, columns):
    results = []
    
    for name, path in files.items():
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            print(f"Cảnh báo: Không tìm thấy file hoặc file trống: {path}")
            continue
            
        try:
            # Thử đọc kiểu thông thường trước
            df = pd.read_json(path)
        except ValueError:
            # Nếu lỗi, thử đọc kiểu lines=True
            df = pd.read_json(path, lines=True)
            
        total_rows = len(df)
        
        for col in columns:
            if col not in df.columns:
                continue
            # Đếm số lượng từng nhãn (kể cả NaN)
            counts = df[col].value_counts(dropna=False)
            
            for label, count in counts.items():
                label_str = "Không nhắc" if pd.isna(label) else str(label)
                results.append({
                    'Dataset': name,
                    'Aspect': col,
                    'Label': label_str,
                    'Count': count,
                    'Percentage (%)': round((count / total_rows) * 100, 2)
                })
                
    return pd.DataFrame(results)

# Chạy kiểm tra
dist_df = check_label_distribution(data_files, aspect_cols)

if not dist_df.empty:
    # Hiển thị kết quả dưới dạng bảng xoay (Pivot)
    pivot_dist = dist_df.pivot_table(
        index=['Aspect', 'Label'], 
        columns='Dataset', 
        values='Percentage (%)'
    ).fillna(0)

    print("\n" + "="*50)
    print("--- TỈ LỆ PHẦN TRĂM CÁC NHÃN TRÊN TỪNG TẬP DỮ LIỆU ---")
    print("="*50)
    print(pivot_dist)
    print("="*50)
else:
    print("Không có dữ liệu để phân tích.")
