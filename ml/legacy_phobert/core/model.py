"""ABSAModel architecture and logit parser."""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel, RobertaPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from .config import ASP_SENT_DIM, N_ASPECTS, PRES_DIM, SENT_DIM


class ABSAModel(RobertaPreTrainedModel):
    """
    PhoBERT with three head families:
      - sentiment_head          : (B, 3)  overall sentiment
      - presence_heads [×6]     : (B, 2)  aspect present / absent
      - aspect_sentiment_heads [×6] : (B, 3)  per-aspect sentiment

    Logits layout: [sent(3) | pres_0..5(2) | asp_sent_0..5(3)]
    _last_cls holds the CLS embedding for Supervised Contrastive Loss.
    """
    config_class = RobertaConfig

    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        drop_p = getattr(config, "classifier_dropout", None) or getattr(config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(drop_p)
        n = N_ASPECTS
        self.sentiment_head         = nn.Linear(config.hidden_size, SENT_DIM)
        self.presence_heads         = nn.ModuleList([nn.Linear(config.hidden_size, PRES_DIM) for _ in range(n)])
        self.aspect_sentiment_heads = nn.ModuleList([nn.Linear(config.hidden_size, ASP_SENT_DIM) for _ in range(n)])
        self.post_init()

    @classmethod
    def _can_set_experts_implementation(cls):
        return False

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        cls     = self.dropout(outputs.last_hidden_state[:, 0, :])
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


def parse_logits(logits):
    """Split flat logit tensor into (sent, presence, asp_sent) views."""
    s = SENT_DIM
    p = N_ASPECTS * PRES_DIM
    return (
        logits[:, :s],
        logits[:, s : s + p].view(-1, N_ASPECTS, PRES_DIM),
        logits[:, s + p :].view(-1, N_ASPECTS, ASP_SENT_DIM),
    )
