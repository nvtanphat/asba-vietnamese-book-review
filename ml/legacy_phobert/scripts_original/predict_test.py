import json
import sys
from pathlib import Path

# Ensure we can import absa_core
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from absa_core.models.predictor import ABSAPredictor, ASPECT_COLS

def main():
    data_path = Path("data/processed/test_clean.json")
    if not data_path.exists():
        print(f"Error: Test set not found at {data_path}")
        return

    print(f"Loading test dataset from {data_path}...")
    with open(data_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    print(f"Loaded {len(samples)} samples.")

    print("Initializing ABSAPredictor...")
    predictor = ABSAPredictor(model_id="data/models/ABSA-TIKI-BOOK", model_variant="phobert")
    print(f"Predictor initialized on device: {predictor.device}")

    texts = [s["content"] for s in samples]
    
    print("Running predictions...")
    batch_size = 64
    predictions = []
    total = len(texts)
    
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_preds = predictor.predict(texts[start:end])
        predictions.extend(batch_preds)
        print(f"Predicted {end}/{total} reviews")

    # Map output samples
    output_samples = []
    for sample, pred in zip(samples, predictions):
        output_entry = {
            "review_id": sample.get("review_id"),
            "content": sample.get("content"),
            "true_overall": sample.get("sentiment"),
            "pred_overall": pred["overall"],
            "pred_overall_probs": [round(p, 4) for p in pred["overall_probs"]],
            "true_aspects": {col: sample.get(col) for col in ASPECT_COLS},
            "pred_aspects": pred["aspects"],
            "pred_aspect_details": pred["aspect_probs"]
        }
        output_samples.append(output_entry)

    output_path = Path("data/processed/test_predictions.json")
    print(f"Saving predictions to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_samples, f, ensure_ascii=False, indent=2)

    print("Predictions saved successfully!")

if __name__ == "__main__":
    main()
