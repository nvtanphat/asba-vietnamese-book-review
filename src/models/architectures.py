import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel, RobertaPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

# Danh sách Aspect chuẩn cho đồ án Tiki
ASPECT_COLS = ["as_content", "as_physical", "as_price", "as_packaging", "as_delivery", "as_service"]
ASPECT_NAMES = ["Nội dung", "Hình thức", "Giá cả", "Đóng gói", "Giao hàng", "Dịch vụ"]

SENT_DIM = 3      # pos, neg, neu
PRES_DIM = 2      # present, absent
ASP_SENT_DIM = 3  # pos, neg, neu

class ABSAModel(RobertaPreTrainedModel):
    """
    Kiến trúc PhoBERT Multi-head cho bài toán ABSA:
    - 1 đầu ra cho Sentiment tổng thể.
    - 6 đầu ra cho Presence của từng Aspect.
    - 6 đầu ra cho Sentiment của từng Aspect.
    """
    config_class = RobertaConfig

    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        drop_p = getattr(config, "classifier_dropout", None) or getattr(config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(drop_p)
        
        n = len(ASPECT_COLS)
        self.sentiment_head = nn.Linear(config.hidden_size, SENT_DIM)
        self.presence_heads = nn.ModuleList([nn.Linear(config.hidden_size, PRES_DIM) for _ in range(n)])
        self.aspect_sentiment_heads = nn.ModuleList([nn.Linear(config.hidden_size, ASP_SENT_DIM) for _ in range(n)])
        
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        # Lấy vector CLS (token đầu tiên)
        cls_output = self.dropout(outputs.last_hidden_state[:, 0, :])

        sent_logits = self.sentiment_head(cls_output)
        pres_logits = torch.stack([h(cls_output) for h in self.presence_heads], dim=1)
        asp_sent_logits = torch.stack([h(cls_output) for h in self.aspect_sentiment_heads], dim=1)

        # Gộp tất cả logits thành 1 tensor duy nhất để tương thích với Trainer nếu cần
        # Layout: [sent(3) | pres_0..5(2*6) | asp_sent_0..5(3*6)]
        logits = torch.cat([
            sent_logits,
            pres_logits.view(cls_output.size(0), -1),
            asp_sent_logits.view(cls_output.size(0), -1),
        ], dim=-1)
        
        return SequenceClassifierOutput(logits=logits)

def parse_logits(logits):
    """Helper function để tách logits tổng hợp ra thành từng phần riêng biệt."""
    s = SENT_DIM
    p = len(ASPECT_COLS) * PRES_DIM
    
    sent = logits[:, :s]
    pres = logits[:, s : s + p].view(-1, len(ASPECT_COLS), PRES_DIM)
    asp_sent = logits[:, s + p :].view(-1, len(ASPECT_COLS), ASP_SENT_DIM)
    
    return sent, pres, asp_sent
