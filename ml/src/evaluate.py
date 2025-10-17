"""
Loads pretrained IsolationForest, scores ml/data/eval.csv, emits:
 - ml_out/ml_metrics.json: precision/recall/f1 (if labels), anomaly_rate, mean_anomaly_score
 - ml_out/ml_predictions.csv: row-wise prediction (1=anomaly, 0=normal) + anomaly_score
"""
from pathlib import Path
import numpy as np
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score
from features import load_csv
from utils import ensure_dir, write_json, log

MODEL_PATH = Path("ml/models/isolation_forest_v1.pkl")
EVAL_PATH  = Path("ml/data/eval.csv")
OUT_DIR    = ensure_dir("ml_out")
OUT_METRICS = OUT_DIR / "ml_metrics.json"
OUT_PRED    = OUT_DIR / "ml_predictions.csv"

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run ml/src/train.py first or provide the .pkl.")

    log("Loading model")
    model = joblib.load(MODEL_PATH)

    log(f"Loading eval data from {EVAL_PATH}")
    X, y, df = load_csv(str(EVAL_PATH))

    # IsolationForest: -1 = anomaly, 1 = normal
    pred_raw = model.predict(X)
    pred = (pred_raw == -1).astype(int)  # 1=anomaly
    scores = -model.score_samples(X)     # higher => more anomalous

    metrics = {}
    if y is not None:
        metrics["precision"] = float(precision_score(y, pred, zero_division=0))
        metrics["recall"]    = float(recall_score(y, pred, zero_division=0))
        metrics["f1"]        = float(f1_score(y, pred, zero_division=0))
    metrics["anomaly_rate"] = float(np.mean(pred))
    metrics["mean_anomaly_score"] = float(np.mean(scores))

    log("Writing outputs")
    write_json(OUT_METRICS, metrics)
    df.assign(prediction=pred, anomaly_score=scores).to_csv(OUT_PRED, index=False)

    log(f"Metrics: {metrics}")

if __name__ == "__main__":
    main()
