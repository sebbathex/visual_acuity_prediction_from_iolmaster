"""
Step 2b – Feature sets of the basic biometry reference model (In the manuscript: "Biometry Reference Model",
Multimedia Appendix 1, Section S3).

Input:  biometry_{task}.csv with the IOLMaster 700 measurements already assigned to each eye
        (same-day: examination on the day of the VA measurement. Postoperative: last examination within
        365 days before surgery): patient_id, eye, r1, r2, ast, axl, acd. Several rows per eye
        (repeated measurements of that examination) are averaged.
Output: {task}_biometry.parquet          5 biometry variables + 1 missingness indicator
        {task}_biometry_dinov2.parquet   the same combined with the DINOv2 embedding

Age was deliberately not included, so that the reference model has no access to information that is
unavailable to the image models. Lens thickness was not part of the available database export.
Missing values are imputed with the training-fold mean inside the PyCaret pipeline (step 3), so all
eyes are retained and the test sets are identical to those of the image models.

Usage: python biometry_features.py --task sameday|postop
"""

import argparse
import numpy as np
import pandas as pd
import config as C


def biometry_features(b: pd.DataFrame) -> pd.DataFrame:
    b = b.copy()
    for v, (lo, hi) in C.BIOMETRY_LIMITS.items():                    # implausible -> missing
        b.loc[b[v].notna() & ~b[v].between(lo, hi), v] = np.nan
    b = b.groupby(["patient_id", "eye"], as_index=False)[list(C.BIOMETRY_LIMITS)].mean()
    return pd.DataFrame({
        "eye_id": b["patient_id"].astype(str) + "_" + b["eye"],
        "k_mean_mm": (b["r1"] + b["r2"]) / 2,                        # mean corneal radius
        "k_diff_mm": (b["r1"] - b["r2"]).abs(),                      # difference of the two radii
        "ast_abs": b["ast"].abs(),                                   # magnitude of astigmatism (axis not used)
        "axl": b["axl"], "acd": b["acd"],
        
        # single indicator: no valid biometry value at all for this eye
        "bio_missing": b[list(C.BIOMETRY_LIMITS)].isna().all(axis=1).astype(int)})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["sameday", "postop"])
    task = ap.parse_args().task

    keys = pd.read_csv(C.WORK_DIR / f"{task}_split.csv")[["eye_id", "patient_id", "target"]]
    bio = biometry_features(pd.read_csv(C.INPUT_DIR / f"biometry_{task}.csv"))
    base = keys.merge(bio, on="eye_id", how="left", validate="1:1")
    base["bio_missing"] = base["bio_missing"].fillna(1).astype(int)   # eyes without a matched examination
    base.to_parquet(C.WORK_DIR / f"{task}_biometry.parquet", index=False)

    dino = pd.read_parquet(C.WORK_DIR / f"{task}_dinov2.parquet").drop(columns=["patient_id", "target"])
    base.merge(dino, on="eye_id", validate="1:1").to_parquet(C.WORK_DIR / f"{task}_biometry_dinov2.parquet",
                                                            index=False)
    print(f"{task}: {len(base)} eyes, {int(base.bio_missing.sum())} without any valid biometry value")
