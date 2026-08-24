import requests
import time
import random
import os
import pandas as pd
import concurrent.futures
import threading
import hashlib
import re
from datetime import datetime
from fake_useragent import UserAgent
from tqdm import tqdm
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
OUTPUT_DIR = "crawl_data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "tiki_books_reviews_v2.csv")
TARGETS = {
    1: 1500,  # Negative
    2: 1500,  # Negative
    3: 3000,  # Neutral
    4: 1500,  # Positive
    5: 4000   # Positive
}

MAX_PAGES = 50       # Fetch more pages to find rare low ratings
BATCH_SIZE = 50      # Save every 50 reviews
MAX_WORKERS = 15     # Concurrent threads
MIN_LENGTH = 10      # Skip too short reviews
MAX_PER_PRODUCT_PHASE2 = 4 # Limit 4-5* reviews per book to ensure diversity
# Root Category IDs for Books on Tiki
ROOT_CATEGORIES = [316, 320] # Sách tiếng Việt, Sách tiếng Anh

# Targeted search to find more potential negative reviews
NEGATIVE_KEYWORDS = [
    "sách lậu", "giấy mỏng", "dịch dở", "sai chính tả", "sách rách", 
    "chữ mờ", "đóng gói kém", "giao sai sách", "không màng co",
    "bản dịch tồi", "sách photo", "in lỗi", "bung gáy", "giấy đen",
    "sách giả", "lừa đảo", "thất vọng", "không giống mô tả",
    "giao chậm", "lừa gạt", "hư hỏng", "bẩn", "nhàu"
]
STOP_PHRASES = [
    # Shipping & Packaging
    "giao hàng nhanh", "đóng gói cẩn thận", "giao hàng nhanh chóng", "đóng gói kỹ",
    "giao hàng cực nhanh", "đóng gói rất cẩn thận", "giao hàng siêu nhanh",
    "bọc kỹ", "hàng giao nhanh", "đóng gói đẹp", "shipper thân thiện",
    "giao hàng sớm hơn dự kiến", "đóng gói chắc chắn", "vận chuyển nhanh",
    # Shop & Service
    "shop phục vụ tốt", "shop tư vấn nhiệt tình", "ủng hộ shop", "sẽ mua lại",
    "cảm ơn shop", "shop uy tín", "đáng mua", "nên mua", "tặng shop 5 sao",
    "phục vụ rất tốt", "chăm sóc khách hàng tốt", "shop đóng gói rất kỹ",
    # Unread / Unused
    "chưa đọc", "chưa đọc nên chưa biết", "chưa đọc thử", "mới nhận hàng chưa xem",
    "mua về để đó", "chưa dùng", "chưa mở ra xem",
    # Short & Generic
    "ok", "tốt", "rất tốt", "hài lòng", "tuyệt vời", "good", "nice", "very good",
    "ổn", "bình thường", "tạm được", "cũng được", "hơi tệ", "kém",
    # As Described
    "hàng giống hình", "giống mô tả", "sách đúng như hình", "đúng mẫu mã",
    "đúng như mô tả", "sản phẩm tuyệt vời",
    # Extra
    "hàng đẹp", "ngon bổ rẻ", "giá rẻ", "đáng tiền", "rất hài lòng",
    "không có gì để chê", "chất lượng tuyệt vời", "sách mới", "sách đẹp"
]
class BookCrawler:
    def __init__(self):
        self.lock = threading.Lock()
        self.file_lock = threading.Lock()
        self.seen_hashes = set()
        self.seen_products = set()
        self.counts = {k: 0 for k in TARGETS.keys()}
        self.ua = UserAgent()
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        self._load_progress()
    def _hash(self, text):
        if not text: return ""
        normalized = re.sub(r'[^\w\s]', '', text.lower())
        normalized = "".join(normalized.split())
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    def _is_stop_phrase(self, text):
        if not text: return True
        text_lower = text.lower().strip()
        for phrase in STOP_PHRASES:
            if text_lower == phrase: return True
        text_clean = re.sub(r'[^\w\s]', '', text_lower)
        for phrase in STOP_PHRASES:
            phrase_clean = re.sub(r'[^\w\s]', '', phrase)
            if text_clean == phrase_clean: return True
        return False
    def _load_progress(self):
        if os.path.exists(OUTPUT_FILE):
            try:
                df = pd.read_csv(OUTPUT_FILE)
                if 'content' in df.columns:
                    for c in df['content'].dropna():
                        self.seen_hashes.add(self._hash(str(c)))
                if 'rating' in df.columns:
                    for r in TARGETS.keys():
                        self.counts[r] = len(df[df['rating'] == r])
                if 'product_id' in df.columns:
                    self.seen_products.update(df['product_id'].unique())
                print(f"Loaded existing progress: {self.counts}")
            except Exception as e:
                print(f"Error loading progress: {e}")
    def _print_groups(self):
        g1_count = self.counts[1] + self.counts[2]
        g1_target = TARGETS[1] + TARGETS[2]
        
        g2_count = self.counts[3]
        g2_target = TARGETS[3]
        
        g3_count = self.counts[4] + self.counts[5]
        g3_target = TARGETS[4] + TARGETS[5]
        
        total_count = sum(self.counts.values())
        total_target = sum(TARGETS.values())
        
        print(f"Groups Status: Neg(1-2*):{g1_count}/{g1_target} | Neu(3*):{g2_count}/{g2_target} | Pos(4-5*):{g3_count}/{g3_target} | Total:{total_count}/{total_target}")
    def _headers(self):
        return {
            'User-Agent': self.ua.random,
            'Accept': 'application/json',
            'Referer': 'https://tiki.vn/'
        }
    
    def _fetch(self, url, params=None):
        session = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        try:
            resp = session.get(url, params=params, headers=self._headers(), timeout=15)
            if resp.status_code == 200:
                return resp.json()
        except:
            pass
        return None
    
    def _get_subcategories(self, parent_id):
        print(f"Discovering subcategories for Parent ID: {parent_id}")
        data = self._fetch("https://tiki.vn/api/v2/categories", params={'parent_id': parent_id})
        if data and data.get('data'):
            return data['data']
        return []

    def _search_by_category(self, cat_id, cat_name):
        products = []
        for page in range(1, 15): 
            data = self._fetch("https://tiki.vn/api/v2/products", 
                              params={'category': cat_id, 'page': page, 'limit': 50})
            if not data or not data.get('data'): break
            for p in data['data']:
                if p['id'] not in self.seen_products and p.get('review_count', 0) > 0:
                    p['cat_name'] = cat_name
                    products.append(p)
                    self.seen_products.add(p['id'])
            if len(products) > 300: break
        return products

    def _search_by_keywords(self, keywords):
        products = []
        for q in tqdm(keywords, desc="Targeted Search"):
            data = self._fetch("https://tiki.vn/api/v2/products", params={'q': q, 'limit': 50})
            if data and data.get('data'):
                for p in data['data']:
                    if p['id'] not in self.seen_products and p.get('review_count', 0) > 0:
                        p['cat_name'] = f"Keyword: {q}"
                        products.append(p)
                        self.seen_products.add(p['id'])
        return products

    def _done(self):
        return all(self.counts[k] >= TARGETS[k] for k in TARGETS)
    
    def _rating_full(self, r):
        return self.counts[r] >= TARGETS[r]
    
    def _save(self, reviews):
        if not reviews: return
        df = pd.DataFrame(reviews)
        with self.file_lock:
            mode = 'a' if os.path.exists(OUTPUT_FILE) else 'w'
            header = not os.path.exists(OUTPUT_FILE)
            df.to_csv(OUTPUT_FILE, mode=mode, header=header, index=False, encoding='utf-8-sig')
    
    def _process(self, item, product):
        review_id = str(item.get('id'))
        rating = item.get('rating')
        title = (item.get('title') or '').strip()
        content = (item.get('content') or '').strip()
        text = content if content else title
        
        if len(text) < MIN_LENGTH: return None
        if self._is_stop_phrase(text): return None
        
        h = self._hash(text)
        with self.lock:
            if h in self.seen_hashes: return None
            if rating not in TARGETS or self.counts[rating] >= TARGETS[rating]: return None
            self.counts[rating] += 1
            self.seen_hashes.add(h)
        
        return {
            'review_id': review_id,
            'rating': rating,
            'review_title': title,
            'content': content,
            'product_id': product['id'],
            'product_name': product.get('name'),
            'category': product.get('cat_name'),
            'created_at': item.get('created_at')
        }
    
    def _crawl_rating(self, product, rating, max_per_product=None):
        reviews = []
        count = 0
        for page in range(1, MAX_PAGES + 1):
            if self._done() or self._rating_full(rating): break
            if max_per_product and count >= max_per_product: break
            
            data = self._fetch("https://tiki.vn/api/v2/reviews",
                params={'product_id': product['id'], 'limit': 20, 'page': page, 'stars': rating})
            if not data or not data.get('data'): break
            
            for item in data['data']:
                if max_per_product and count >= max_per_product: break
                r = self._process(item, product)
                if r:
                    reviews.append(r)
                    count += 1
            time.sleep(random.uniform(0.1, 0.4))
        return reviews

    def _crawl_batch(self, products, ratings, desc, max_per_product=None):
        if not products: return
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            for rating in ratings:
                if self._rating_full(rating): continue
                futures = [ex.submit(self._crawl_rating, p, rating, max_per_product) for p in products]
                batch = []
                for f in tqdm(concurrent.futures.as_completed(futures), total=len(products), desc=f"{desc} {rating}*"):
                    res = f.result()
                    if res:
                        batch.extend(res)
                        if len(batch) >= BATCH_SIZE:
                            self._save(batch)
                            batch = []
                    if self._rating_full(rating): break
                self._save(batch)
                self._print_groups()
    
    def run(self):
        print("Book Review Crawler - NEGATIVE-FIRST POOL STRATEGY")
        # 1. Thu thập tất cả sản phẩm từ mọi ngách trước
        all_products_pool = []
        all_subcats = []
        for root_id in ROOT_CATEGORIES:
            all_subcats.extend(self._get_subcategories(root_id))
        
        print(f"Discovering products across {len(all_subcats)} categories...")
        for cat in tqdm(all_subcats, desc="Product Discovery"):
            products = self._search_by_category(cat['id'], cat['name'])
            all_products_pool.extend(products)
        # Thêm sản phẩm từ Negative Keywords
        print("\nAdding targeted products for negative reviews...")
        neg_targeted_products = self._search_by_keywords(NEGATIVE_KEYWORDS)
        all_products_pool.extend(neg_targeted_products)
        # Deduplication of pool
        unique_pool = []
        seen_ids = set()
        for p in all_products_pool:
            if p['id'] not in seen_ids:
                unique_pool.append(p)
                seen_ids.add(p['id'])
        random.shuffle(unique_pool)
        print(f"\nTotal unique products in pool: {len(unique_pool)}")
        # 2. CHIẾN THUẬT QUÉT NGANG (PRIORITY RATINGS)
        # BƯỚC 1: Săn lùng cực phẩm 1* và 2* trên TOÀN BỘ list sản phẩm
        if not (self._rating_full(1) and self._rating_full(2)):
            print("\n>>> PHASE 1: Hunting 1* & 2* across ALL products...")
            self._crawl_batch(unique_pool, [1, 2], "Global Neg", max_per_product=None)
        # BƯỚC 2: Quét 3* (Neutral)
        if not self._rating_full(3):
            print("\n>>> PHASE 2: Collecting 3* (Neutral)...")
            self._crawl_batch(unique_pool, [3], "Global Neu", max_per_product=10)
        # BƯỚC 3: Cuối cùng mới lấy 4* và 5*
        if not (self._rating_full(4) and self._rating_full(5)):
            print("\n>>> PHASE 3: Collecting 4* & 5* (Diversified)...")
            self._crawl_batch(unique_pool, [4, 5], "Global Pos", max_per_product=MAX_PER_PRODUCT_PHASE2)
        print("CRAWLING DONE")
        self._print_groups()
        print(f"Main output: {OUTPUT_FILE}")
if __name__ == "__main__":
    BookCrawler().run()
