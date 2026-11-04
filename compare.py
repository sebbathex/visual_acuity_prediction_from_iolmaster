"""
Step 4 – Paired comparisons (manuscript: "Comparison of Embedding Architectures",
"Comparison With Numerical Biometry").

For two feature sets, the classifiers selected by cross-validation are compared on IDENTICAL holdout
eyes. The difference in macro-AUC is resampled with a patient-level cluster bootstrap. This replaces
DeLong's test, which does not extend directly to a macro-averaged one-vs-rest AUC.

Output: results/{task}/comparisons.csv
Usage:  python compare.py --task sameday|postop
"""
import argparse
import itertools
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config as C
from train_evaluate import patient_rows


def paired_auc_difference(y, Pa, Pb, groups, n_boot=C.N_BOOT, seed=C.SEED) -> dict:
    """Difference in macro-AUC (a − b), 95% percentile CI and two-sided bootstrap P value."""
    auc = lambda yy, P: roc_auc_score(yy, P, multi_class="ovr", average="macro")   # noqa: E731
    rng, rows = np.random.default_rng(seed), patient_rows(groups)
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([rows[i] for i in rng.integers(0, len(rows), len(rows))])
        if len(np.unique(y[idx])) == Pa.shape[1]:
            diffs.append(auc(y[idx], Pa[idx]) - auc(y[idx], Pb[idx]))
    diffs = np.array(diffs)
    p = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))   # P = 0 means P < 1/n_boot
    return {"diff": auc(y, Pa) - auc(y, Pb), "ci_low": np.percentile(diffs, 2.5),
            "ci_high": np.percentile(diffs, 97.5), "p": p}


def selected_predictions(task, feature_set):
    d = C.RESULTS_DIR / task / feature_set
    best = json.loads((d / "run_info.json").read_text())["best_model"]
    p = pd.read_csv(d / f"pred_{best}.csv").sort_values("eye_id").reset_index(drop=True)
    return best, p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["sameday", "postop"])
    task = ap.parse_args().task

    pairs = list(itertools.combinations(C.EMBEDDERS, 2))                     # embedders
    pairs += [("biometry", e) for e in C.EMBEDDERS] + [("biometry_dinov2", "dinov2")]
    rows = []
    for a, b in pairs:
        if not all((C.RESULTS_DIR / task / f / "run_info.json").exists() for f in (a, b)):
            continue
        (ma, pa), (mb, pb) = selected_predictions(task, a), selected_predictions(task, b)
        assert (pa.eye_id.values == pb.eye_id.values).all()                    # identical test eyes
        r = paired_auc_difference(pa.target.to_numpy(), pa.filter(regex=r"^p\d$").to_numpy(),
                                  pb.filter(regex=r"^p\d$").to_numpy(), pa.patient_id.to_numpy())
        rows.append({"A": f"{a} ({ma})", "B": f"{b} ({mb})", **{k: round(v, 3) for k, v in r.items()}})
    out = pd.DataFrame(rows)
    out.to_csv(C.RESULTS_DIR / task / "comparisons.csv", index=False)
    print(out.to_string(index=False))
