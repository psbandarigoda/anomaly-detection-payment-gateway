"""
Offline training script (run locally or in a scheduled workflow).
Reads ml/data/train.csv, trains IsolationForest, writes ml/models/isolation_forest_model_v1.pkl
"""
from pathlib import Path
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from features import load_csv
from utils import ensure_dir, log

TRAIN_CSV = Path("ml/data/train.csv")
MODEL_PATH = Path("ml/models/isolation_forest_model_v1.pkl")

def main():
    log("Loading training data")
    X, _, _ = load_csv(str(TRAIN_CSV))

    log("Building pipeline (StandardScaler + IsolationForest)")
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("iso", IsolationForest(
            n_estimators=200,
            max_samples="auto",
            contamination="auto",  # let IF estimate; you can set e.g. 0.02 if known
            random_state=42,
            bootstrap=False,
            n_jobs=-1
        ))
    ])

    log("Fitting model")
    pipe.fit(X)

    ensure_dir(MODEL_PATH.parent)
    joblib.dump(pipe, MODEL_PATH)
    log(f"Saved model -> {MODEL_PATH}")

if __name__ == "__main__":
    main()
