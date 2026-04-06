import torch
import torch.nn as nn
from transformers import AutoModel, RobertaConfig, RobertaModel, RobertaPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

# Standard aspect list for the Tiki ABSA task
ASPECT_COLS = ["as_content", "as_physical", "as_price", "as_packaging", "as_delivery", "as_service"]
ASPECT_NAMES = ["Noi dung", "Hinh thuc", "Gia ca", "Dong goi", "Giao hang", "Dich vu"]

SENT_DIM = 3      # neg, neu, pos
PRES_DIM = 2      # absent, present
ASP_SENT_DIM = 3  # neg, neu, pos


class ABSAModel(RobertaPreTrainedModel):
    """
    PhoBERT multi-head architecture:
    - 1 head for overall sentiment
    - 6 heads for aspect presence
    - 6 heads for aspect sentiment
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
        cls_output = self.dropout(outputs.last_hidden_state[:, 0, :])

        sent_logits = self.sentiment_head(cls_output)
        pres_logits = torch.stack([h(cls_output) for h in self.presence_heads], dim=1)
        asp_sent_logits = torch.stack([h(cls_output) for h in self.aspect_sentiment_heads], dim=1)

        logits = torch.cat(
            [
                sent_logits,
                pres_logits.view(cls_output.size(0), -1),
                asp_sent_logits.view(cls_output.size(0), -1),
            ],
            dim=-1,
        )
        return SequenceClassifierOutput(logits=logits)


class SpatialDropout1D(nn.Module):
    """Drop whole embedding channels, used in the BiLSTM notebook model."""

    def __init__(self, p: float):
        super().__init__()
        self.dropout = nn.Dropout2d(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1).unsqueeze(3)
        return self.dropout(x).squeeze(3).permute(0, 2, 1)


class BaseABSABiLSTM(nn.Module):
    """
    BiLSTM + MHA + multi-head decoder from notebooks/04_absa_bilstm.ipynb.
    Subclasses only need to provide the embedding source.
    """

    def __init__(self, embed_dim: int, hidden_dim: int, num_layers: int, dropout: float, num_aspects: int):
        super().__init__()
        self.spatial_dropout = SpatialDropout1D(dropout)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_out_dim = hidden_dim * 2
        self.mha = nn.MultiheadAttention(embed_dim=lstm_out_dim, num_heads=4, dropout=dropout, batch_first=True)
        self.attention_pool = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.Tanh(),
            nn.Linear(lstm_out_dim // 2, 1),
        )
        self.feat_norm = nn.LayerNorm(lstm_out_dim * 3)
        cat_dim = lstm_out_dim * 3

        self.sent_proj = nn.Linear(cat_dim, hidden_dim)
        self.sent_decoupler = nn.Sequential(
            nn.Linear(cat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.sent_classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim, SENT_DIM),
        )

        self.asp_proj = nn.Linear(cat_dim, hidden_dim)
        self.asp_decoupler = nn.Sequential(
            nn.Linear(cat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.pres_classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim, num_aspects * PRES_DIM),
        )
        self.asp_classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim, num_aspects * ASP_SENT_DIM),
        )
        self.num_aspects = num_aspects

    def process_features(self, embedded: torch.Tensor, attention_mask: torch.Tensor):
        lengths = attention_mask.sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(self.lstm(packed)[0], batch_first=True)

        batch_size, max_len, _ = lstm_out.size()
        device = lstm_out.device
        mask = torch.arange(max_len, device=device).expand(batch_size, max_len) < lengths.unsqueeze(1).to(device)

        attn_output, _ = self.mha(query=lstm_out, key=lstm_out, value=lstm_out, key_padding_mask=~mask)
        mask_expanded = mask.unsqueeze(-1)

        attn_scores = self.attention_pool(attn_output).masked_fill(~mask_expanded, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=1)
        mhsa_pool = torch.sum(attn_weights * attn_output, dim=1)
        max_pool = torch.max(lstm_out.masked_fill(~mask_expanded, -1e9), dim=1)[0]

        mask_float = mask.float().unsqueeze(-1)
        avg_pool = torch.sum(lstm_out * mask_float, dim=1) / torch.sum(mask_float, dim=1).clamp(min=1e-9)

        context_vector = self.feat_norm(torch.cat([mhsa_pool, max_pool, avg_pool], dim=-1))

        sent_context = self.sent_decoupler(context_vector) + self.sent_proj(context_vector)
        sent_logits = self.sent_classifier(sent_context)

        asp_context = self.asp_decoupler(context_vector) + self.asp_proj(context_vector)
        pres_logits = self.pres_classifier(asp_context).view(-1, self.num_aspects, PRES_DIM)
        asp_logits = self.asp_classifier(asp_context).view(-1, self.num_aspects, ASP_SENT_DIM)
        return sent_logits, pres_logits, asp_logits


class PhoBERTABSABiLSTM(BaseABSABiLSTM):
    """PhoBERT embeddings + BiLSTM decoder."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.4,
        num_aspects: int = 6,
        phobert_model_name_or_path: str = "vinai/phobert-base",
        phobert_subfolder: str | None = None,
        local_files_only: bool = False,
    ):
        super().__init__(
            embed_dim=768,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            num_aspects=num_aspects,
        )
        load_kwargs = {"local_files_only": local_files_only}
        if phobert_subfolder:
            load_kwargs["subfolder"] = phobert_subfolder
        self.phobert = AutoModel.from_pretrained(phobert_model_name_or_path, **load_kwargs)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        embedded = self.phobert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        embedded = self.spatial_dropout(embedded)
        return self.process_features(embedded, attention_mask)


def parse_logits(logits: torch.Tensor):
    """Split concatenated logits from ABSAModel into dedicated heads."""
    s = SENT_DIM
    p = len(ASPECT_COLS) * PRES_DIM

    sent = logits[:, :s]
    pres = logits[:, s : s + p].view(-1, len(ASPECT_COLS), PRES_DIM)
    asp_sent = logits[:, s + p :].view(-1, len(ASPECT_COLS), ASP_SENT_DIM)
    return sent, pres, asp_sent
