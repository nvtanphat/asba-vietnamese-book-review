# Vietnamese Book Review ABSA

Phan tich cam xuc da khia canh tren danh gia sach Tiki.

Du an su dung quy trinh tien xu ly theo huong giu song song hai tang du lieu:
- `raw`: de doi chieu va thuc nghiem ve sau
- `clean`: de huan luyen multi-task mac dinh

## 1. Tong quan pipeline

Pipeline hien tai gom 2 pha:

1. Dataset curation
   - Doc du lieu raw.
   - Loai bo cac dong co `sentiment_llm` khong hop le.
   - Chia du lieu thanh `train/val/test = 70/15/15`.
   - Viec chia split duoc thuc hien theo `group_key` suy ra tu text sau cac buoc clean rule-based de tranh trung noi dung xuyen split.

2. Split-specific cleaning
   - Moi split raw duoc clean rieng.
   - Output clean mac dinh chi giu:
     - `review_id`
     - `content`
     - `sentiment_llm`
     - `as_content`
     - `as_physical`
     - `as_price`
     - `as_packaging`
     - `as_delivery`
     - `as_service`
   - Cac buoc clean hien tai la rule-based/stateless, khong co `fit` tren `val/test`.

## 2. Cau truc thu muc

```text
DoAn2/
|-- data/
|   |-- raw/                         # Du lieu goc tu Tiki
|   |-- interim/
|   |   |-- raw_train/train.json     # Raw train split
|   |   |-- raw_val/val.json         # Raw validation split
|   |   `-- raw_test/test.json       # Raw test split
|   `-- processed/
|       |-- train_clean.json         # Clean train cho multi-task
|       |-- val_clean.json           # Clean val cho multi-task
|       `-- test_clean.json          # Clean test cho multi-task
|-- src/
|   |-- analysis/
|   `-- preprocessing/
|       |-- pipeline.py
|       |-- split_dataset.py
|       `-- cli.py
|-- notebooks/
|-- dashboard.py
|-- requirements.txt
`-- README.md
```

## 3. Cach chay

Dung o thu muc goc du an.

### 3.1 Tao raw split va clean split

```bash
python -m src.preprocessing.split_dataset
```

Lenh nay se tao 6 file chinh:

- `data/interim/raw_train/train.json`
- `data/interim/raw_val/val.json`
- `data/interim/raw_test/test.json`
- `data/processed/train_clean.json`
- `data/processed/val_clean.json`
- `data/processed/test_clean.json`

### 3.2 Clean rieng tung split raw

```bash
python -m src.preprocessing.cli --split train
python -m src.preprocessing.cli --split val
python -m src.preprocessing.cli --split test
```

### 3.3 Quet loi du lieu

```bash
python -m src.analysis.scan_cli
```

Mac dinh scanner uu tien `data/interim/raw_train/train.json`.

## 4. Contract du lieu

### Raw split

Raw split duoc giu lai de:
- thu nghiem cac chien luoc preprocess khac nhau
- doi chieu truoc/sau clean
- truy vet ve du lieu goc

### Clean split

Clean split la artifact mac dinh de train multi-task.

Moi file clean chi co 9 cot:
- `review_id`
- `content`
- `sentiment_llm`
- `as_content`
- `as_physical`
- `as_price`
- `as_packaging`
- `as_delivery`
- `as_service`

## 5. Ghi chu ve data leakage

- `val/test` duoc xem la du lieu chua tung thay trong pha hoc tham so.
- Cac buoc clean rule-based hien tai khong hoc tham so tu du lieu.
- Moi preprocessing co `fit` trong tuong lai phai duoc hoc tren `train` va chi `transform` cho `val/test`.

## 6. Moi truong

- Python >= 3.8
- Cac thu vien trong `requirements.txt`

Neu gap loi OpenBLAS khi chay trong moi truong gioi han tai nguyen, co the dat:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
python -m src.preprocessing.split_dataset
```
