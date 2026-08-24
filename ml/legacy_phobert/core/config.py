"""Hyperparameters, aspect labels, and experiment configuration."""
from pathlib import Path

# ── Model ─────────────────────────────────────────────────────────────────
MODEL_NAME    = "vinai/phobert-base-v2"
MAX_LENGTH    = 160
BATCH_SIZE    = 16
EPOCHS        = 7
LEARNING_RATE = 2e-5

# ── Labels ────────────────────────────────────────────────────────────────
ASPECT_COLS   = ["as_content", "as_physical", "as_price", "as_packaging", "as_delivery", "as_service"]
TARGET_COLS   = ["sentiment"] + ASPECT_COLS
SENTIMENT_LABELS = {0: "Tiêu cực", 1: "Trung lập", 2: "Tích cực"}
ASPECT_LABELS    = {0: "Tiêu cực", 1: "Trung lập", 2: "Tích cực", 3: "Không nhắc đến"}

# ── Logit layout ──────────────────────────────────────────────────────────
N_ASPECTS    = len(ASPECT_COLS)
SENT_DIM     = 3
PRES_DIM     = 2
ASP_SENT_DIM = 3
ABSENT_ASPECT_CLASS = 3

# ── Loss ──────────────────────────────────────────────────────────────────
ASPECT_ABSENT_WEIGHT_SCALE    = 0.2
CLASS_BALANCED_BETA           = 0.999
BASE_FOCAL_GAMMA              = 2.0
DEFAULT_SENTIMENT_LOSS_WEIGHT = 0.5
DEFAULT_ASPECT_LOSS_WEIGHT    = 0.5
STAGE1_LOSS_WEIGHT = 0.25   # presence detection
STAGE2_LOSS_WEIGHT = 0.75   # aspect sentiment

IMPROVED_FOCAL_SENTIMENT_SMOOTHING  = 0.05
IMPROVED_FOCAL_ASPECT_SMOOTHING     = 0.10
IMPROVED_FOCAL_ASPECT_GAMMA_PRESENT = 2.5
IMPROVED_FOCAL_ASPECT_GAMMA_ABSENT  = 1.0
IMPROVED_FOCAL_IGNORE_EASY_ABSENT   = True
IMPROVED_FOCAL_IGNORE_THRESHOLD     = 0.5

# ── Sampling ──────────────────────────────────────────────────────────────
SAMPLER_TEMPERATURE         = 0.5
SAMPLER_WEIGHT_CAP          = 4.0
NEUTRAL_SAMPLER_TEMPERATURE = 0.45
NEUTRAL_ASPECT_GAMMA        = 0.8
NEUTRAL_ASPECT_SMOOTHING    = 0.05
THRESHOLD_NEUTRAL_WEIGHT    = 0.35

# ── Paths ─────────────────────────────────────────────────────────────────
ABSA_PROMPT_PREFIX = "ABSA review aspect content physical price packaging delivery service: "
DEFAULT_DATA_ROOT  = Path("/kaggle/input/datasets/nguynvntnpht/tiki-cleaned-book-reviews-absa")

# ── Focal configs ─────────────────────────────────────────────────────────
FOCAL_CONFIG_STANDARD = {
    "sentiment": {"gamma": BASE_FOCAL_GAMMA, "smoothing": IMPROVED_FOCAL_SENTIMENT_SMOOTHING},
    "aspect": {
        "gamma": BASE_FOCAL_GAMMA, "smoothing": IMPROVED_FOCAL_ASPECT_SMOOTHING,
        "gamma_by_class": [IMPROVED_FOCAL_ASPECT_GAMMA_PRESENT] * 3 + [IMPROVED_FOCAL_ASPECT_GAMMA_ABSENT],
        "ignore_easy_absent": IMPROVED_FOCAL_IGNORE_EASY_ABSENT,
        "ignore_threshold":   IMPROVED_FOCAL_IGNORE_THRESHOLD,
        "absent_class":       ABSENT_ASPECT_CLASS,
    },
}

FOCAL_CONFIG_NEUTRAL = {
    "sentiment": {
        "gamma": BASE_FOCAL_GAMMA, "smoothing": IMPROVED_FOCAL_SENTIMENT_SMOOTHING,
        "gamma_by_class": [2.5, 1.0, 2.5],
    },
    "aspect": {
        "gamma": BASE_FOCAL_GAMMA, "smoothing": NEUTRAL_ASPECT_SMOOTHING,
        "gamma_by_class": [IMPROVED_FOCAL_ASPECT_GAMMA_PRESENT, NEUTRAL_ASPECT_GAMMA,
                           IMPROVED_FOCAL_ASPECT_GAMMA_PRESENT, IMPROVED_FOCAL_ASPECT_GAMMA_ABSENT],
        "ignore_easy_absent": IMPROVED_FOCAL_IGNORE_EASY_ABSENT,
        "ignore_threshold":   IMPROVED_FOCAL_IGNORE_THRESHOLD,
        "absent_class":       ABSENT_ASPECT_CLASS,
    },
}

# Plain improved-focal, no per-class gamma / no easy-absent filtering — this is the
# exact loss recipe used by 05-phobert-balance-experiment-under1mb.ipynb.
FOCAL_CONFIG_CLEAN = {
    "sentiment": {"gamma": BASE_FOCAL_GAMMA, "smoothing": IMPROVED_FOCAL_SENTIMENT_SMOOTHING},
    "aspect":    {"gamma": BASE_FOCAL_GAMMA, "smoothing": IMPROVED_FOCAL_ASPECT_SMOOTHING},
}

# Rare-aspect loss multiplier clip range (see trainer.compute_aspect_loss_multipliers),
# matching the notebook's `np.clip(raw_mult, 0.75, 1.60)`.
ASPECT_MULT_CLIP_MIN = 0.75
ASPECT_MULT_CLIP_MAX = 1.60

# ── Best experiment — mirrors 05-phobert-balance-experiment-under1mb.ipynb's
# "clean_joint_balanced_focal" run exactly (calibrated val f1_combined=0.7976,
# test f1_combined=0.7924). No SCL, no neutral-bucket sampler, no per-class gamma —
# those belonged to an earlier, superseded "neutral_scl" experiment.
BEST_EXPERIMENT = {
    "name":                           "clean_joint_balanced_focal",
    "train_key":                      "clean",
    "loss_name":                      "improved_focal",
    "use_class_weights":              True,
    "sent_weight_key":                "mean_capped",
    "train_sampler":                  "joint_balanced",
    "sampler_temperature":            SAMPLER_TEMPERATURE,
    "sampler_include_neutral_bucket": False,
    "sentiment_loss_weight":          0.3,
    "aspect_loss_weight":             0.7,
    "scl_weight":                     0.0,
    "focal_config":                   FOCAL_CONFIG_CLEAN,
}
