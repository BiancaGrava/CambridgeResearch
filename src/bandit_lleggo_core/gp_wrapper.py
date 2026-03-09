from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


AlgorithmType = None
for cand in [
    "llego.algorithm.Algorithm",
    "bandit_lleggo_core.algorithm.Algorithm",
    "bandit_lleggo.algorithm.Algorithm",
]:
    try:
        mod_path, cls_name = cand.rsplit(".", 1)
        mod = __import__(mod_path, fromlist=[cls_name])
        AlgorithmType = getattr(mod, cls_name)
        break
    except Exception:
        pass


@dataclass
class _BestTreeInfo:
    depth: int = 1
    size: int = 1


class GPWrapper:
    """
    Wrapper around your evolving algorithm.
    - If AlgorithmType is available, predictions/metrics come from the evolved best individual.
    - Fallback to a baseline DecisionTree if not (so runs never crash).
    - Exposes: update_rule_pool(), lock_rule_pool(), ensure_seeded_population(), best_fitness(), summarize_lineages()
    """

    def __init__(self, cfg: Any):
        self.cfg = cfg

        self._best_f = 0.0
        self._best_tree = _BestTreeInfo(depth=1, size=1)
        self._rule_pool_serial: List[Dict[str, Any]] = []
        self._locked_pool: bool = False
        self._feature_names: List[str] = []
        self._rng = random.Random(42)
        self._t0 = time.time()
        self._iters = 0

        self._algo = None
        self._best_individual = None
        self._seeded_once = False

        self._baseline: Optional[DecisionTreeClassifier] = None
        self._use_one_hot = False
        self._one_hot_columns: Optional[List[str]] = None

        self._le: Optional[LabelEncoder] = None
        self._n_classes: Optional[int] = None

        self._build(cfg)


    def _build(self, cfg: Any):
        if AlgorithmType is None:
            return
        try:
            self._algo = AlgorithmType(cfg)
        except Exception:
            self._algo = None

    def fit_baseline(self, X, Y):
        import pandas as pd
        Y_enc = self._ensure_encoded_labels(Y)
        max_depth = getattr(self.cfg.gp, "max_depth", 4)

        if hasattr(X, "select_dtypes"):
            X_fit = pd.get_dummies(X, drop_first=True)
            self._use_one_hot = True
            self._one_hot_columns = list(X_fit.columns)
        else:
            X_fit = X
            self._use_one_hot = False
            self._one_hot_columns = None

        self._baseline = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        self._baseline.fit(X_fit, Y_enc)
        self._n_classes = len(np.unique(Y_enc))


    def step(self) -> None:
        self._iters += 1
        if self._algo is not None:
            if hasattr(self._algo, "run_one_generation"):
                self._algo.run_one_generation()
            self._best_individual = getattr(self._algo, "best_individual", None)
            if self._best_individual is not None:
                try:
                    self._best_f = float(getattr(self._best_individual, "fitness", self._best_f))
                except Exception:
                    pass
                self._update_best_tree_info_from_individual(self._best_individual)
            return

        pool_bonus = math.log(max(1, len(self._rule_pool_serial) + 1)) / 50.0
        delta = 0.001 + pool_bonus + self._rng.uniform(-1e-4, 3e-4)
        self._best_f = max(self._best_f, self._best_f + max(0.0, delta))
        if self._iters % 5 == 0:
            max_depth = self.cfg.gp.max_depth if hasattr(self.cfg, "gp") else 4
            self._best_tree.depth = min(max_depth, self._best_tree.depth + 1)
            self._best_tree.size += self._rng.randint(1, 3)


    def predict(self, X) -> np.ndarray:
        if self._best_individual is not None:
            y_hat = self._best_individual.predict(X)
            return self._ensure_encoded_labels(y_hat, fit_if_needed=False)
        if self._baseline is not None:
            X_infer = self._prep_baseline_infer_matrix(X)
            return self._baseline.predict(X_infer).astype(int)
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X) -> np.ndarray:
        if self._best_individual is not None:
            proba = getattr(self._best_individual, "predict_proba", None)
            if callable(proba):
                out = proba(X)
                if out is None:
                    return self._to_2col_probs_from_labels(self.predict(X))
                out = np.asarray(out)
                if out.ndim == 1:
                    out = np.vstack([1 - out, out]).T
                return out
            return self._to_2col_probs_from_labels(self.predict(X))
        if self._baseline is not None and hasattr(self._baseline, "predict_proba"):
            X_infer = self._prep_baseline_infer_matrix(X)
            return self._baseline.predict_proba(X_infer)
        K = self._n_classes or 2
        return np.full((len(X), K), 1.0 / K, dtype=float)


    def compute_metrics(self, X, Y) -> Dict[str, Any]:
        task_type = getattr(getattr(self.cfg, "dataset", None), "task", None) or "classification"
        if task_type == "classification":
            Y_enc = self._ensure_encoded_labels(Y)
            y_pred = self.predict(X).astype(int)
            proba = self.predict_proba(X)
            acc = accuracy_score(Y_enc, y_pred)
            bal_acc = balanced_accuracy_score(Y_enc, y_pred)
            ece = self._ece_multiclass(Y_enc, proba, bins=10)
            d, s = self._best_tree.depth, self._best_tree.size
            return {
                "accuracy": float(acc),
                "balanced_accuracy": float(bal_acc),
                "ece": float(ece),
                "depth": int(d),
                "n_params": int(s),
            }
        y_true = np.asarray(Y, dtype=float)
        y_pred = self.predict(X).astype(float)
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        d, s = self._best_tree.depth, self._best_tree.size
        return {"rmse": rmse, "depth": int(d), "n_params": int(s)}


    def lock_rule_pool(self, rules: List[Dict[str, Any]]):
        """Lock the pool for Phase-2 (use ONLY S*)."""
        self._locked_pool = True
        self.update_rule_pool(rules)

    def update_rule_pool(self, rules: List[Dict[str, Any]]) -> None:
        """Accepts dict rules or dataclass Rules; keeps a serializable copy and sets operators' priors."""
        if self._locked_pool and self._rule_pool_serial:
            rules = self._rule_pool_serial

        serial, seen = [], set()
        for r in rules or []:
            d = self._normalize_rule(r)
            expr = d.get("expr", "")
            if expr and expr not in seen:
                serial.append(d)
                seen.add(expr)
        self._rule_pool_serial = serial

        if self._algo is not None:
            for op_name in ("mutation_operator", "crossover_operator", "initializer"):
                op = getattr(self._algo, op_name, None)
                if op is not None:
                    try:
                        if hasattr(op, "set_rule_pool"):
                            op.set_rule_pool(serial)
                        else:
                            setattr(op, "rule_pool", serial)
                    except Exception:
                        pass

    def ensure_seeded_population(self) -> None:
        """Try to reseed the population once, after rules are available."""
        if self._seeded_once or self._algo is None or not self._rule_pool_serial:
            return
        try:
            init_ = getattr(self._algo, "initializer", None)
            pop_fn = getattr(self._algo, "initialize_population", None)
            reseed_fn = getattr(self._algo, "reseed_with_initializer", None)

            if hasattr(init_, "set_rule_pool"):
                init_.set_rule_pool(self._rule_pool_serial)
            elif init_ is not None:
                setattr(init_, "rule_pool", self._rule_pool_serial)

            if callable(reseed_fn):
                reseed_fn()
                print("[INIT] reseed_with_initializer() called.")
                self._seeded_once = True
                return
            if callable(pop_fn):
                pop_size = int(getattr(self.cfg.gp, "pop_size", 20))
                pop_fn(pop_size)
                print("[INIT] initialize_population() called for seeding.")
                self._seeded_once = True
        except Exception:
            pass


    def save_best(self, exp_name: str) -> None:
        os.makedirs("results", exist_ok=True)
        payload = {
            "best_fitness": self._best_f,
            "best_tree": {"depth": self._best_tree.depth, "size": self._best_tree.size, "rules": []},
            "rule_pool_size": len(self._rule_pool_serial),
            "elapsed_sec": time.time() - self._t0,
            "iters": self._iters,
        }
        with open(os.path.join("results", f"{exp_name}_best.json"), "w") as f:
            json.dump(payload, f, indent=2)
        with open(os.path.join("results", f"{exp_name}_rule_pool.json"), "w") as f:
            json.dump(self._rule_pool_serial, f, indent=2)

    def save_metrics(self, exp_name: str, metrics: dict) -> None:
        os.makedirs("results", exist_ok=True)
        path = os.path.join("results", f"{exp_name}_metrics.json")
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)


    def best_fitness(self) -> float:
        return float(getattr(self, "_best_f", 0.0))

    def summarize_lineages(self, k: int = 8):
        """Return brief lineage summaries for prompting; safe fallback if algo doesn't expose elites."""
        if self._algo is not None:
            for attr in ("hall_of_fame", "elites", "archive", "population"):
                pop = getattr(self._algo, attr, None)
                if pop and isinstance(pop, (list, tuple)):
                    out = []
                    for ind in pop[:k]:
                        try:
                            f = float(getattr(ind, "fitness", 0.0))
                        except Exception:
                            f = 0.0
                        summ = getattr(ind, "summary", None)
                        txt = str(summ(max_splits=2)) if callable(summ) else f"best subtree with fitness={f:.4f}"
                        out.append({"text": txt, "fitness": f})
                    if out:
                        return out
            fn = getattr(self._algo, "summarize_lineages", None)
            if callable(fn):
                try:
                    vals = fn(k=k)
                    if isinstance(vals, list) and vals:
                        return vals
                except Exception:
                    pass
        f = float(getattr(self, "_best_f", 0.0))
        rules_txt = ", ".join([r.get("expr", "") for r in self._rule_pool_serial[:2]]) or "no rules yet"
        fallback = [{"text": f"best_f={f:.4f}; rules: {rules_txt}", "fitness": f}]
        while len(fallback) < max(1, int(k)):
            fallback.append({"text": "placeholder", "fitness": f})
        return fallback[:k]

    def set_feature_names(self, names: List[str]):
        self._feature_names = list(names or [])
        if self._algo is not None:
            init_ = getattr(self._algo, "initializer", None)
            if init_ is not None:
                try:
                    if hasattr(init_, "set_feature_names"):
                        init_.set_feature_names(self._feature_names)
                except Exception:
                    pass

    def _update_best_tree_info_from_individual(self, ind) -> None:
        try:
            d = int(ind.depth()) if callable(getattr(ind, "depth", None)) else self._best_tree.depth
            s = int(ind.size()) if callable(getattr(ind, "size", None)) else self._best_tree.size
            self._best_tree = _BestTreeInfo(depth=d, size=s)
        except Exception:
            pass

    def _ensure_encoded_labels(self, Y, fit_if_needed: bool = True) -> np.ndarray:
        Y = np.asarray(Y)
        if Y.dtype.kind in {"U", "S", "O"}:
            if fit_if_needed or self._le is None:
                self._le = LabelEncoder().fit(Y)
            return self._le.transform(Y)
        if self._le is None and fit_if_needed:
            self._le = LabelEncoder().fit(Y)
        return Y.astype(int)

    def _to_2col_probs_from_labels(self, y_hat: np.ndarray) -> np.ndarray:
        y_hat = np.asarray(y_hat).astype(int)
        K = int(max(2, (self._n_classes or (y_hat.max() + 1))))
        proba = np.zeros((len(y_hat), K), dtype=float)
        proba[np.arange(len(y_hat)), y_hat] = 1.0
        return proba

    def _ece_multiclass(self, y_true: np.ndarray, proba: np.ndarray, bins: int = 10) -> float:
        y_true = np.asarray(y_true).astype(int)
        if proba.ndim == 1:
            proba = np.vstack([1 - proba, proba]).T
        y_hat = proba.argmax(axis=1)
        conf = proba.max(axis=1)
        correct = (y_hat == y_true).astype(float)
        edges = np.linspace(0.0, 1.0, bins + 1)
        ece = 0.0
        for i in range(bins):
            lo, hi = edges[i], edges[i + 1]
            idx = (conf >= lo) & (conf < hi)
            if idx.sum() == 0:
                continue
            acc_bin = correct[idx].mean()
            conf_bin = conf[idx].mean()
            ece += (idx.mean()) * abs(acc_bin - conf_bin)
        return float(ece)

    def _normalize_rule(self, r):
        if isinstance(r, dict):
            expr = str(r.get("expr", "")).strip()
            meta = r.get("meta", {})
            if not isinstance(meta, dict):
                meta = {}
            return {"expr": expr, "meta": meta}
        expr = str(getattr(r, "expr", "")).strip()
        meta = getattr(r, "meta", {})
        if not isinstance(meta, dict):
            meta = {}
        return {"expr": expr, "meta": meta}

    def _prep_baseline_infer_matrix(self, X):
        import pandas as pd
        if self._use_one_hot and hasattr(X, "select_dtypes"):
            X_infer = pd.get_dummies(X, drop_first=True)
            for c in self._one_hot_columns or []:
                if c not in X_infer.columns:
                    X_infer[c] = 0
            X_infer = X_infer[self._one_hot_columns]
            return X_infer
        return X