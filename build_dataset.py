"""
Step 2 – Analysis datasets (manuscript: "Dataset Construction and Inclusion Criteria").

Input:  work/embeddings_{task}_{embedder}.parquet (step 1) and
        input/clinical_sameday.csv: patient_id, eye, va                (decimal BCVA, day of biometry)
        input/clinical_postop.csv:  patient_id, eye, va_biometry, va_postop, registry_complete (0/1)
Output: work/{task}_{embedder}.parquet  features + target
        work/{task}_split.csv           eye_id, patient_id, target, set (train/test)
        work/{task}_flow.json           number of eyes after each exclusion step

Key design points
* Only eyes available for ALL three embedders are kept, so every embedder is trained and tested on
  identical eyes.
* ONE patient-level split (GroupShuffleSplit) is shared by all embedders. Both eyes of a patient
  are always in the same partition.

Usage: python build_dataset.py --task sameday|postop [--include-incomplete]
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

import config as C


def targets_sameday(df: pd.DataFrame, flow: dict) -> pd.DataFrame:
    flow["eyes"] = len(df)
    df = df[df["va"].notna()];         flow["after_missing_VA"] = len(df)
    df = df[df["va"] > 0];             flow["after_VA_eq_0"] = len(df)
    df = df[df["va"] <= C.SAMEDAY_BINS[-1]]; flow["after_VA_gt_1.25"] = len(df)
    # pd.cut is right-closed: (0, 0.30], (0.30, 0.63], (0.63, 1.25]
    return df.assign(target=pd.cut(df["va"], C.SAMEDAY_BINS, labels=False, right=True).astype(int))


def targets_postop(df: pd.DataFrame, flow: dict, include_incomplete: bool) -> pd.DataFrame:
    flow["eyes"] = len(df)
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = -np.log10(df["va_biometry"]) + np.log10(df["va_postop"])   # logMAR(pre) - logMAR(post)
    valid = np.isfinite(gain) & (df["va_biometry"] < 5)                    # codes >= 5 mean "missing"
    df = df.assign(gain=gain)[valid];  flow["after_missing_or_invalid_VA"] = len(df)
    if C.POSTOP_EXCLUDE_WORSENING:
        df = df[df["gain"] >= 0];      flow["after_exclude_worsening"] = len(df)
    if not include_incomplete:         # primary analysis; sensitivity analysis keeps these eyes
        df = df[df["registry_complete"] == 1]; flow["after_incomplete_registry"] = len(df)
    bins = [-np.inf] + C.POSTOP_BINS + [np.inf]
    return df.assign(target=pd.cut(df["gain"], bins, labels=False, right=True).astype(int))


def main(task: str, include_incomplete: bool) -> None:
    flow = {}
    clin = pd.read_csv(C.INPUT_DIR / f"clinical_{task}.csv")
    clin = (targets_sameday(clin, flow) if task == "sameday"
            else targets_postop(clin, flow, include_incomplete))
    clin["eye_id"] = clin["patient_id"].astype(str) + "_" + clin["eye"]
    clin = clin.drop_duplicates("eye_id").set_index("eye_id")

    # Features of the images used for this task, per embedder
    feats = {}
    for emb in C.EMBEDDERS:
        e = pd.read_parquet(C.WORK_DIR / f"embeddings_{task}_{emb}.parquet")
        e.index = e["patient_id"].astype(str) + "_" + e["eye"]
        cols = [c for c in e.columns if c[:2] in C.IMAGES[task]]
        feats[emb] = e.loc[~e.index.duplicated(), cols]

    # Eyes with a valid target and embeddings from all three networks (sorted -> reproducible split)
    common = sorted(set(clin.index).intersection(*(set(f.index) for f in feats.values())))
    flow["final_all_embedders"] = len(common)

    keys = clin.loc[common, ["patient_id", "target"]].reset_index()
    gss = GroupShuffleSplit(n_splits=1, test_size=C.TEST_SIZE, random_state=C.SEED)
    tr, te = next(gss.split(keys, keys["target"], groups=keys["patient_id"]))
    keys["set"] = "train"
    keys.loc[te, "set"] = "test"
    assert not set(keys.loc[tr, "patient_id"]) & set(keys.loc[te, "patient_id"])   # no patient in both

    name = task + ("_full" if include_incomplete else "")
    keys.to_csv(C.WORK_DIR / f"{name}_split.csv", index=False)
    for emb, f in feats.items():
        keys[["eye_id", "patient_id", "target"]].join(f, on="eye_id").to_parquet(
            C.WORK_DIR / f"{name}_{emb}.parquet", index=False)

    pat = keys.groupby("patient_id").size()
    flow.update(patients=int(pat.size), bilateral_patients=int((pat > 1).sum()),
                train_eyes=len(tr), test_eyes=len(te),
                test_patients=int(keys.loc[te, "patient_id"].nunique()),
                class_counts=keys["target"].value_counts().sort_index().to_dict())
    (C.WORK_DIR / f"{name}_flow.json").write_text(json.dumps(flow, indent=2, default=int))
    print(json.dumps(flow, indent=2, default=int))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["sameday", "postop"])
    ap.add_argument("--include-incomplete", action="store_true",
                    help="postop sensitivity analysis: keep eyes with incomplete registry entries")
    a = ap.parse_args()
    main(a.task, a.include_incomplete)
