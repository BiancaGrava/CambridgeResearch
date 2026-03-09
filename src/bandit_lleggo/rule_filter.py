from __future__ import annotations
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd

from .rule_pool import Rule
from .metrics_calib import ece_score, conformal_width
from .metrics_fair import tpr_gap

def _to_numpy_1d(arr):
    if hasattr(arr, "to_numpy"):
        return arr.to_numpy()
    return np.asarray(arr)

def _to_binary_numeric(y):
    """
    Convert labels to {0,1} if they are strings/categorical.
    If already numeric, cast to float.
    """
    y_arr = _to_numpy_1d(y)
    if y_arr.dtype.kind in {"U", "S", "O"} or pd.api.types.is_categorical_dtype(y_arr):
        classes = np.unique(y_arr)
        if len(classes) == 2:
            return (y_arr == classes[1]).astype(float)
        maj = classes[np.argmax([(y_arr == c).sum() for c in classes])]
        return (y_arr == maj).astype(float)
    return y_arr.astype(float)

def filter_rules(rules: List[Rule], X, y, task_type: str,
                 thresholds: Dict[str, float], groups=None) -> Tuple[List[Rule], Dict[str, Any]]:
    accepted, logs = [], {"rejected": []}

    y_num = _to_binary_numeric(y) if task_type == "classification" else _to_numpy_1d(y)
    groups_arr = _to_numpy_1d(groups) if groups is not None else None

    for r in rules:
        mask = _mask_from_expr(r.expr, X)
        if mask.sum() < max(5, int(0.01 * len(X))):
            logs["rejected"].append((r.expr, "low_support"))
            print("[RULE-REJECT]", r.expr, "low_support")
            continue

        util = _utility(mask, y_num, task_type)
        if task_type == "classification":
            risk = ece_score(y_num[mask], _leaf_probs(y_num[mask]))
        else:
            risk = conformal_width(y_num[mask], _leaf_preds(y_num[mask]))
        fair = 0.0
        if groups_arr is not None and task_type == "classification":
            fair = tpr_gap(y_true=(y_num > 0.5), mask=mask, groups=groups_arr)

        if util >= thresholds["u"] and risk <= thresholds["ece"] and abs(fair) <= thresholds["fair"]:
            r.meta.update({"util": float(util), "risk": float(risk), "fair": float(fair)})
            accepted.append(r)
        else:
            reason = f"u={util:.3f}, risk={risk:.3f}, fair={fair:.3f}"
            logs["rejected"].append((r.expr, reason))
            print("[RULE-REJECT]", r.expr, reason)

    return accepted, logs

def _mask_from_expr(expr: str, X):
    """
    Build a boolean mask from a rule expression. We allow pandas.eval for convenience.
    """
    if hasattr(X, "eval"):
        try:
            return X.eval(expr).astype(bool).to_numpy()
        except Exception:
            return np.zeros(len(X), dtype=bool)
    return np.ones(len(X), dtype=bool)

def _utility(mask, y_num, task_type):
    if task_type == "classification":
        p = y_num[mask].mean()
        return -(p*np.log(p + 1e-9) + (1-p)*np.log(1-p + 1e-9))
    return y_num[mask].var()

def _leaf_probs(y_subset):
    p = float(np.mean(y_subset)) if len(y_subset) > 0 else 0.5
    return np.full(len(y_subset), p, dtype=float)

def _leaf_preds(y_subset):
    m = float(np.mean(y_subset)) if len(y_subset) > 0 else 0.0
    return np.full(len(y_subset), m, dtype=float)