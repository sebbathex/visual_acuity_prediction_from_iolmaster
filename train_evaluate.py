"""
Step 3 – Classification and evaluation (manuscript: "Machine Learning and Optimization Pipeline").

Identical protocol for every task and feature set:
* fixed patient-level train/test split from step 2
* z-standardization (fitted on the training folds only; done by PyCaret)
* 10-fold StratifiedGroupKFold in the training set, patient as group
* six classifiers with default hyperparameters and balanced class weights (no SMOTE, no tuning)
* selection of ONE classifier per feature set: highest mean cross-validated macro-AUC
* holdout evaluation with 95% CIs from a patient-level cluster bootstrap (primary results)

Output in results/{task}/{feature_set}/: cv_summary.csv, holdout.csv, pred_{model}.csv, run_info.json

Usage: python train_evaluate.py --task sameday|postop|postop_full --features yolo|clip|dinov2|biometry|biometry_dinov2
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import config as C

# Balanced class weights per classifier (keys -> PyCaret model IDs). XGBoost: see below.
MODEL_KWARGS = {"lr": dict(class_weight="balanced", max_iter=5000),
                "rf": dict(class_weight="balanced"), "et": dict(class_weight="balanced"),
                "lightgbm": dict(class_weight="balanced", verbosity=-1),
                "catboost": dict(auto_class_weights="Balanced")}


class BalancedXGBClassifier(XGBClassifier):
    """ XGBoost has no class_weight for multiclass problems. Balanced sample weights are computed in
    every fit() (each CV fold and the final model), this is equivalent to class_weight="balanced"."""
    def fit(self, X, y, **kw):
        kw.setdefault("sample_weight", compute_sample_weight("balanced", y))
        return super().fit(X, y, **kw)


class PatientGroupKFold(StratifiedGroupKFold):
    """Fails loudly if PyCaret does not pass the patient groups (guarantees patient-level CV)."""
    def split(self, X, y=None, groups=None):
        if groups is None:
            raise RuntimeError("CV called without patient groups")
        return super().split(X, y, groups)


def auc_macro_ovr(y_true, y_score, **_):
    """Unweighted (macro) one-vs-rest AUC. PyCaret's default AUC is support-weighted, and for
    logistic regression it may receive decision-function logits; these are converted by softmax,
    which equals predict_proba() for multinomial logistic regression."""
    s = np.asarray(y_score, dtype=float)
    if not np.allclose(s.sum(axis=1), 1.0, atol=1e-6):
        s = softmax(s, axis=1)
    return roc_auc_score(y_true, s, multi_class="ovr", average="macro")


def point_metrics(y, P) -> dict:
    """Holdout metrics for class probabilities P (n_eyes × 3)."""
    pred = P.argmax(1)
    out = {"AUC_macro": roc_auc_score(y, P, multi_class="ovr", average="macro"),
           "Accuracy": accuracy_score(y, pred), "Balanced_accuracy": balanced_accuracy_score(y, pred),
           "F1_macro": f1_score(y, pred, average="macro"), "Kappa": cohen_kappa_score(y, pred)}
    out.update({f"AUC_class{c}": roc_auc_score(y == c, P[:, c]) for c in range(P.shape[1])})
    return out


def patient_rows(groups):
    """Row indices for  the eyes of each patient (unit of resampling in the cluster bootstrap)."""
    _, inv = np.unique(groups, return_inverse=True)
    return [np.flatnonzero(inv == i) for i in range(inv.max() + 1)]


def cluster_bootstrap(y, P, groups, n_boot=C.N_BOOT, seed=C.SEED) -> dict:
    """95% percentile CIs. Patients are resampled with replacement, keeping both eyes together,
    because fellow eyes are correlated. Resamples lacking a class are skipped."""
    rng, rows = np.random.default_rng(seed), patient_rows(groups)
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([rows[i] for i in rng.integers(0, len(rows), len(rows))])
        if len(np.unique(y[idx])) == P.shape[1]:
            draws.append(point_metrics(y[idx], P[idx]))
    return {k: np.percentile([d[k] for d in draws], [2.5, 97.5]) for k in draws[0]}


def run(task: str, feature_set: str, models=C.MODELS, folds=C.N_FOLDS, n_boot=C.N_BOOT) -> None:
    from pycaret.classification import ClassificationExperiment

    df = pd.read_parquet(C.WORK_DIR / f"{task}_{feature_set}.parquet")
    df = df.merge(pd.read_csv(C.WORK_DIR / f"{task}_split.csv")[["eye_id", "set"]], on="eye_id", validate="1:1")
    feats = [c for c in df.columns if c not in ("eye_id", "patient_id", "target", "set")]
    tr = df[df.set == "train"].reset_index(drop=True)
    te = df[df.set == "test"].reset_index(drop=True)

    exp = ClassificationExperiment()
    exp.setup(data=tr[feats + ["patient_id", "target"]], test_data=te[feats + ["patient_id", "target"]],
              target="target", index=False, session_id=C.SEED,
              ignore_features=["patient_id"],                      
              normalize=True, normalize_method="zscore",
              fold_strategy=PatientGroupKFold(n_splits=folds, shuffle=True, random_state=C.SEED),
              fold=folds, fold_groups="patient_id", fix_imbalance=False,
              verbose=False, html=False, log_experiment=False, system_log=False)
    exp.remove_metric("auc")
    exp.add_metric("auc_macro", "AUC", auc_macro_ovr, target="pred_proba")
    assert np.array_equal(exp.get_config("y_train").to_numpy(), tr["target"].to_numpy())   # row order kept

    out = C.RESULTS_DIR / task / feature_set
    out.mkdir(parents=True, exist_ok=True)
    y_te, cv_rows, ho_rows = te["target"].to_numpy(), [], []
    for m in models:
        est = (BalancedXGBClassifier(random_state=C.SEED, verbosity=0, tree_method="hist")
               if m == "xgboost" else m)
        model = exp.create_model(est, verbose=False, **MODEL_KWARGS.get(m, {}))
        cv = exp.pull()                                           # per-fold CV results (secondary)
        cv_rows.append({"model": m, "AUC_mean": cv.loc["Mean", "AUC"], "AUC_sd": cv.loc["Std", "AUC"],
                        "Accuracy_mean": cv.loc["Mean", "Accuracy"], "Kappa_mean": cv.loc["Mean", "Kappa"]})

        pred = exp.predict_model(model, raw_score=True, round=10, verbose=False)
        P = pred[sorted(c for c in pred.columns if c.startswith("prediction_score_"))].to_numpy(float)
        P /= P.sum(1, keepdims=True)                              # remove rounding residue
        assert np.array_equal(pred["target"].to_numpy().astype(int), y_te)
        ci = cluster_bootstrap(y_te, P, te["patient_id"].to_numpy(), n_boot)
        ho_rows.append({"model": m, **point_metrics(y_te, P),
                        **{f"{k}_lo": v[0] for k, v in ci.items()}, **{f"{k}_hi": v[1] for k, v in ci.items()}})
        pd.DataFrame({"eye_id": te.eye_id, "patient_id": te.patient_id, "target": y_te,
                      **{f"p{c}": P[:, c] for c in range(P.shape[1])}}).to_csv(out / f"pred_{m}.csv", index=False)
        print(f"{task}/{feature_set} {m:9s} CV-AUC {cv_rows[-1]['AUC_mean']:.3f} | "
              f"holdout AUC {ho_rows[-1]['AUC_macro']:.3f} ({ci['AUC_macro'][0]:.3f}-{ci['AUC_macro'][1]:.3f})")

    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(out / "cv_summary.csv", index=False)
    pd.DataFrame(ho_rows).to_csv(out / "holdout.csv", index=False)
    best = cv_df.loc[cv_df["AUC_mean"].idxmax(), "model"]          # pre-specified selection rule
    (out / "run_info.json").write_text(json.dumps({
        "task": task, "features": feature_set, "n_features": len(feats), "train_eyes": len(tr),
        "test_eyes": len(te), "best_model": best,
        "majority_class_accuracy_test": float(np.bincount(y_te).max() / len(y_te))}, indent=2))
    print(f"-> selected by CV macro-AUC: {best}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["sameday", "postop", "postop_full"])
    ap.add_argument("--features", required=True)
    ap.add_argument("--models", nargs="+", default=C.MODELS)
    ap.add_argument("--n-boot", type=int, default=C.N_BOOT)
    a = ap.parse_args()
    run(a.task, a.features, a.models, n_boot=a.n_boot)
