import torch
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from src.preprocessing.pipeline import clean_text_series
from .architectures import ABSAModel, ASPECT_COLS, ASPECT_NAMES

class ABSAPredictor:
    """
    Predictor chuyên biệt cho PhoBERT (Multi-head classification).
    Được tối ưu hóa cho bài toán ABSA với 6 khía cạnh chính.
    """
    def __init__(self, model_id="hoangloc112/ABSA-TIKI-BOOK", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        
        # Load Tokenizer và Model từ subfolder phobert
        self.tokenizer = AutoTokenizer.from_pretrained(f"{model_id}", subfolder="phobert")
        self.model = ABSAModel.from_pretrained(f"{model_id}", subfolder="phobert").to(self.device)
        
        self.model.eval()

    def _preprocess(self, texts):
        cleaned = clean_text_series(pd.Series(texts), lowercase=True).tolist()
        return [str(t) for t in cleaned]

    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]
            
        cleaned = self._preprocess(texts)
        inputs = self.tokenizer(cleaned, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
        # Parse logits thành các phần (sentiment, presence, asp_sent)
        s = 3
        p = len(ASPECT_COLS) * 2
        sent_logits = logits[:, :s]
        pres_logits = logits[:, s : s + p].view(-1, len(ASPECT_COLS), 2)
        asp_sent_logits = logits[:, s + p :].view(-1, len(ASPECT_COLS), 3)

        results = []
        for i in range(len(texts)):
            overall_probs = torch.softmax(sent_logits[i], dim=-1).cpu().tolist()
            overall_sent = torch.argmax(sent_logits[i]).item()
            
            aspects = {}
            aspect_probs = {}
            for j, col in enumerate(ASPECT_COLS):
                pres_probs = torch.softmax(pres_logits[i, j], dim=-1)
                presence_conf = pres_probs[1].item()
                is_present = (torch.argmax(pres_probs).item() == 1) and (presence_conf > 0.65)
                
                if is_present:
                    aspect_sent = torch.argmax(asp_sent_logits[i, j]).item()
                    asp_probs = torch.softmax(asp_sent_logits[i, j], dim=-1).cpu().tolist()
                else:
                    aspect_sent = -1
                    asp_probs = [0, 0, 0]
                aspects[col] = aspect_sent
                aspect_probs[col] = {"presence": round(presence_conf, 3), "sentiment": asp_probs}
            
            results.append({
                "overall": overall_sent,
                "overall_probs": overall_probs,
                "aspects": aspects,
                "aspect_probs": aspect_probs,
            })
        return results

# Ví dụ sử dụng:
# predictor = ABSAPredictor(model_type="phobert")
# res = predictor.predict("Sách rất hay, giao hàng tận nơi nhanh chóng.")
