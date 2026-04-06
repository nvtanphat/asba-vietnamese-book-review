import os
from pathlib import Path
import warnings

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from src.preprocessing.pipeline import clean_text_series
from .architectures import ABSAModel, ASPECT_COLS, ASPECT_NAMES, PhoBERTABSABiLSTM, parse_logits

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from pyvi import ViTokenizer
except Exception:  # pragma: no cover - optional dependency at runtime
    ViTokenizer = None


DEFAULT_MODEL_ID = "hoangloc112/ABSA-TIKI-BOOK"
DEFAULT_LOCAL_MODEL_DIR = Path("data") / "models" / "ABSA-TIKI-BOOK"
PHOBERT_SUBFOLDER = "phobert"
BILSTM_WEIGHTS_FILE = "BiLSTM-Phobert/bilstm_phobert.pt"


MODEL_VARIANTS = {
    "phobert": {
        "label": "PhoBERT Multi-head",
        "max_length": 256,
    },
    "bilstm-phobert": {
        "label": "BiLSTM + PhoBERT",
        "max_length": 160,
        "weights_file": BILSTM_WEIGHTS_FILE,
        "aspect_thresholds": [0.98, 0.40, 0.50, 0.30, 0.60, 0.70],
    },
}


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class ABSAPredictor:
    """
    Unified predictor for:
    - 'phobert': PhoBERT multi-head (Transformers style checkpoint)
    - 'bilstm-phobert': PhoBERT embeddings + BiLSTM decoder (notebook model)
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        model_variant: str = "phobert",
        device=None,
        offline: bool | None = None,
    ):
        if model_variant not in MODEL_VARIANTS:
            raise ValueError(f"Unsupported model_variant='{model_variant}'. Supported: {list(MODEL_VARIANTS)}")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_id = model_id
        self.model_variant = model_variant
        self.max_length = MODEL_VARIANTS[model_variant]["max_length"]
        self.aspect_thresholds = MODEL_VARIANTS["bilstm-phobert"]["aspect_thresholds"]
        self.offline = _env_flag("ABSA_OFFLINE", False) if offline is None else offline

        self.local_model_root = self._resolve_local_model_root()
        self.model_source = str(self.local_model_root) if self.local_model_root else self.model_id
        self.local_files_only = self.offline or (self.local_model_root is not None)

        tokenizer_source, tokenizer_subfolder = self._resolve_tokenizer_source()
        tokenizer_kwargs = {"local_files_only": self.local_files_only}
        if tokenizer_subfolder:
            tokenizer_kwargs["subfolder"] = tokenizer_subfolder
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)

        self.model = self._load_model().to(self.device)
        self.model.eval()

    def _resolve_local_model_root(self) -> Path | None:
        candidates: list[Path] = []

        model_dir_env = os.getenv("ABSA_MODEL_DIR")
        if model_dir_env:
            candidates.append(Path(model_dir_env))

        model_id_path = Path(self.model_id)
        if model_id_path.exists():
            candidates.append(model_id_path)

        candidates.append(DEFAULT_LOCAL_MODEL_DIR)

        for candidate in candidates:
            if (candidate / PHOBERT_SUBFOLDER / "config.json").exists():
                return candidate
        return None

    def _resolve_tokenizer_source(self) -> tuple[str, str | None]:
        if self.local_model_root:
            return str(self.local_model_root), PHOBERT_SUBFOLDER
        return self.model_id, PHOBERT_SUBFOLDER

    def _load_model(self):
        if self.model_variant == "phobert":
            return ABSAModel.from_pretrained(
                self.model_source,
                subfolder=PHOBERT_SUBFOLDER,
                local_files_only=self.local_files_only,
            )
        return self._load_bilstm_phobert_model()

    def _resolve_bilstm_weight_path(self) -> str:
        local_candidates: list[Path] = []

        if self.local_model_root:
            local_candidates.append(self.local_model_root / BILSTM_WEIGHTS_FILE)
        local_candidates.extend(
            [
                Path("BiLSTM-Phobert") / "bilstm_phobert.pt",
                Path("absa_experiments") / "best_bilstm_+_phobert.pt",
                Path("data") / "models" / "bilstm_phobert.pt",
            ]
        )

        for candidate in local_candidates:
            if candidate.exists():
                return str(candidate)

        if self.local_files_only:
            raise RuntimeError(
                "Cannot find local BiLSTM checkpoint. "
                "Please run: python scripts/prefetch_absa_models.py"
            )

        try:
            return hf_hub_download(
                repo_id=self.model_id,
                filename=MODEL_VARIANTS["bilstm-phobert"]["weights_file"],
                local_files_only=self.local_files_only,
            )
        except Exception as exc:
            raise RuntimeError(
                "Cannot load BiLSTM-Phobert weights. "
                "Expected local file or HF file 'BiLSTM-Phobert/bilstm_phobert.pt'."
            ) from exc

    def _load_bilstm_phobert_model(self):
        model = PhoBERTABSABiLSTM(
            hidden_dim=256,
            num_layers=2,
            dropout=0.4,
            num_aspects=len(ASPECT_COLS),
            phobert_model_name_or_path=self.model_source,
            phobert_subfolder=PHOBERT_SUBFOLDER,
            local_files_only=self.local_files_only,
        )
        weight_path = self._resolve_bilstm_weight_path()
        try:
            state = torch.load(weight_path, map_location=self.device, weights_only=False)
        except TypeError:
            state = torch.load(weight_path, map_location=self.device)

        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise RuntimeError("Invalid BiLSTM-Phobert checkpoint format. Expected a state_dict.")

        # Handle DataParallel checkpoints from notebook training.
        if state and all(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

        model.load_state_dict(state, strict=True)
        return model

    def _preprocess(self, texts):
        cleaned = clean_text_series(pd.Series(texts), lowercase=True).tolist()
        cleaned = [str(t) for t in cleaned]
        if self.model_variant == "bilstm-phobert" and ViTokenizer is not None:
            cleaned = [ViTokenizer.tokenize(t) for t in cleaned]
        return cleaned

    def _run_heads(self, inputs):
        if self.model_variant == "phobert":
            logits = self.model(**inputs).logits
            return parse_logits(logits)
        return self.model(inputs["input_ids"], inputs["attention_mask"])

    def _is_present(self, pres_probs: torch.Tensor, aspect_idx: int) -> bool:
        presence_conf = pres_probs[1].item()
        if self.model_variant == "bilstm-phobert":
            return presence_conf >= self.aspect_thresholds[aspect_idx]
        return (torch.argmax(pres_probs).item() == 1) and (presence_conf > 0.65)

    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        cleaned = self._preprocess(texts)
        inputs = self.tokenizer(
            cleaned,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            sent_logits, pres_logits, asp_sent_logits = self._run_heads(inputs)

        results = []
        for i in range(len(texts)):
            overall_probs = torch.softmax(sent_logits[i], dim=-1).cpu().tolist()
            overall_sent = torch.argmax(sent_logits[i]).item()

            aspects = {}
            aspect_probs = {}
            for j, col in enumerate(ASPECT_COLS):
                pres_probs = torch.softmax(pres_logits[i, j], dim=-1)
                presence_conf = pres_probs[1].item()
                is_present = self._is_present(pres_probs, j)

                if is_present:
                    aspect_sent = torch.argmax(asp_sent_logits[i, j]).item()
                    asp_probs = torch.softmax(asp_sent_logits[i, j], dim=-1).cpu().tolist()
                else:
                    aspect_sent = -1
                    asp_probs = [0.0, 0.0, 0.0]

                aspects[col] = aspect_sent
                aspect_probs[col] = {"presence": round(presence_conf, 3), "sentiment": asp_probs}

            results.append(
                {
                    "overall": overall_sent,
                    "overall_probs": overall_probs,
                    "aspects": aspects,
                    "aspect_probs": aspect_probs,
                }
            )
        return results
