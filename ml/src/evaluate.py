# ml/src/evaluate.py
from pathlib import Path
import json, joblib, numpy as np, pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score,
    matthews_corrcoef, roc_auc_score, average_precision_score
)
from utils import ensure_dir, write_json, log, read_json

MODEL_PATH = Path("ml/models/isolation_forest_model_v1.pkl")
META_PATH  = Path("ml/models/isolation_forest_v1.meta.json")
EVAL_PATH  = Path("ml/data/eval.csv")
OUT_DIR    = ensure_dir("ml_out")
OUT_METRICS = OUT_DIR / "ml_metrics.json"
OUT_PRED    = OUT_DIR / "ml_predictions.csv"

def resolve_feature_order(expected_from_meta, model):
    """
    Return a feature name list that matches the model's training spec.
    Priority:
      1) model.feature_names_in_ (if available)
      2) expected_from_meta trimmed/expanded to model.n_features_in_
    """
    # Prefer exact training names if present
    if hasattr(model, "feature_names_in_") and len(getattr(model, "feature_names_in_")) > 0:
        names = list(model.feature_names_in_)
        log(f"Using feature_names_in_ from model (n={len(names)})")
        return names

    n_model = getattr(model, "n_features_in_", None)
    if n_model is None:
        # Fall back to meta as-is
        log("Model lacks n_features_in_; falling back to meta expected_features")
        return list(expected_from_meta)

    exp = list(expected_from_meta)
    if len(exp) == n_model:
        return exp

    if len(exp) > n_model:
        # Trim extras but preserve order
        trimmed = exp[:n_model]
        removed = exp[n_model:]
        log(f"WARNING: meta expected_features has {len(exp)} cols, model expects {n_model}. "
            f"Trimming extras: {removed}")
        return trimmed

    # len(exp) < n_model: pad with dummy columns
    missing_count = n_model - len(exp)
    pads = [f"_pad_{i}" for i in range(missing_count)]
    log(f"WARNING: meta expected_features has {len(exp)} cols, model expects {n_model}. "
        f"Padding with zeros for columns: {pads}")
    return exp + pads

def align(df: pd.DataFrame, ordered_names, fill_value=0.0) -> pd.DataFrame:
    """Ensure all ordered_names exist, fill missing with zeros, return df in that order only."""
    for col in ordered_names:
        if col not in df.columns:
            df[col] = fill_value
    # filter strictly to the ordered names (drop any extras)
    return df[ordered_names]

def _confusion(y_true, y_pred):
    """Return TP, FP, TN, FN for positive class=1."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, fp, tn, fn

def _safe_metric(fn, *args, **kwargs):
    try:
        return float(fn(*args, **kwargs))
    except Exception:
        return None

def _has_both_classes(y):
    if y is None:
        return False
    y = np.asarray(y).astype(int)
    return np.any(y == 0) and np.any(y == 1)

def _threshold_metrics(y, scores, threshold):
    """
    Given y (0/1) and anomaly scores (higher => more anomalous), binarize at threshold
    and return confusion + precision/recall/f1/accuracy/specificity/balanced_accuracy/mcc.
    """
    y = np.asarray(y).astype(int)
    scores = np.asarray(scores).astype(float)
    pred = (scores >= threshold).astype(int)

    tp, fp, tn, fn = _confusion(y, pred)
    precision = _safe_metric(precision_score, y, pred, zero_division=0)
    recall    = _safe_metric(recall_score, y, pred, zero_division=0)
    f1        = _safe_metric(f1_score, y, pred, zero_division=0)
    acc       = _safe_metric(accuracy_score, y, pred)
    bal_acc   = _safe_metric(balanced_accuracy_score, y, pred)
    # Specificity = TN / (TN + FP)
    specificity = float(tn) / float(tn + fp) if (tn + fp) > 0 else None
    mcc       = _safe_metric(matthews_corrcoef, y, pred)

    return {
        "threshold": float(threshold),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": bal_acc,
        "f1": f1,
        "mcc": mcc,
        "n_predicted_anomalies": int(pred.sum()),
        "n_predicted_normals":   int((1 - pred).sum())
    }

def main():
    log("Loading model")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)
    model = joblib.load(MODEL_PATH)

    log("Loading meta")
    meta = read_json(META_PATH)
    if not meta or "expected_features" not in meta:
        raise FileNotFoundError(f"Missing/invalid meta: {META_PATH}")
    expected_from_meta = meta["expected_features"]

    log(f"Loading eval: {EVAL_PATH}")
    df = pd.read_csv(EVAL_PATH)
    y = df["label"].astype(int).values if "label" in df.columns else None

    # Resolve the final ordered feature list to match the model
    ordered_features = resolve_feature_order(expected_from_meta, model)

    # Build the design matrix; unseen pads are zeros; coerce numerics to safe floats
    Xdf = align(df.copy(), ordered_features).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X = Xdf.to_numpy(dtype=float)

    log("Predicting")
    # IsolationForest: predict -> {-1 anomaly, 1 normal}
    pred_raw = model.predict(X)
    pred = (pred_raw == -1).astype(int)
    # Higher score => more anomalous
    scores = -model.score_samples(X)

    # Base metrics (backwards compatible fields)
    metrics = {}
    metrics["anomaly_rate"] = float(np.mean(pred))
    metrics["mean_anomaly_score"] = float(np.mean(scores))
    metrics["n_features_eval"] = int(X.shape[1])
    metrics["n_features_model_expected"] = int(getattr(model, "n_features_in_", X.shape[1]))
    metrics["n_samples"] = int(X.shape[0])

    if y is not None:
        # Dataset composition
        metrics["n_positive"] = int(np.sum(y == 1))
        metrics["n_negative"] = int(np.sum(y == 0))

        # Confusion & core metrics using model's own decision rule
        tp, fp, tn, fn = _confusion(y, pred)
        metrics["TP"] = tp
        metrics["FP"] = fp
        metrics["TN"] = tn
        metrics["FN"] = fn

        metrics["precision"] = _safe_metric(precision_score, y, pred, zero_division=0)
        metrics["recall"]    = _safe_metric(recall_score,    y, pred, zero_division=0)
        metrics["f1"]        = _safe_metric(f1_score,        y, pred, zero_division=0)
        metrics["accuracy"]  = _safe_metric(accuracy_score,  y, pred)
        metrics["balanced_accuracy"] = _safe_metric(balanced_accuracy_score, y, pred)
        # Specificity = TN / (TN + FP)
        metrics["specificity"] = (float(tn) / float(tn + fp)) if (tn + fp) > 0 else None
        metrics["mcc"]        = _safe_metric(matthews_corrcoef, y, pred)

        # Curve metrics (only if both classes are present)
        if np.any(y == 0) and np.any(y == 1):
            metrics["roc_auc"] = _safe_metric(roc_auc_score, y, scores)
            metrics["pr_auc"]  = _safe_metric(average_precision_score, y, scores)
        else:
            metrics["roc_auc"] = None
            metrics["pr_auc"]  = None

        # Threshold variants on scores
        if X.shape[0] > 0:
            # percentiles: p95/p99 (stricter anomaly definitions)
            p95 = float(np.percentile(scores, 95))
            p99 = float(np.percentile(scores, 99))
            # represent the model's current decision boundary via a hint threshold
            if np.any(pred == 1):
                th_hint = float(scores[pred == 1].min())
            else:
                th_hint = float("inf")

            metrics["by_threshold"] = {
                "model_predict": {
                    "threshold_hint": th_hint,
                    **_threshold_metrics(y, scores, threshold=th_hint)
                },
                "p95": _threshold_metrics(y, scores, threshold=p95),
                "p99": _threshold_metrics(y, scores, threshold=p99),
            }

            # Best-F1 scan (unique score cutoffs, limit to 400 samples for speed)
            uniq = np.unique(scores)
            if uniq.size > 400:
                idx = np.linspace(0, uniq.size - 1, 400).astype(int)
                cand = uniq[idx]
            else:
                cand = uniq

            best = {"f1": -1.0, "threshold": None, "metrics": None}
            for t in cand:
                m = _threshold_metrics(y, scores, threshold=float(t))
                if m["f1"] is not None and m["f1"] > best["f1"]:
                    best = {"f1": m["f1"], "threshold": m["threshold"], "metrics": m}

            metrics["best_f1_threshold"] = {
                "threshold": float(best["threshold"]) if best["threshold"] is not None else None,
                "f1": float(best["f1"]) if best["f1"] is not None and best["f1"] >= 0 else None,
                "metrics": best["metrics"]
            }

    # Persist outputs
    write_json(OUT_METRICS, metrics)

    df_out = df.copy()
    df_out["prediction"] = pred
    df_out["anomaly_score"] = scores
    df_out.to_csv(OUT_PRED, index=False)

    log(json.dumps(metrics))

if __name__ == "__main__":
    main()
