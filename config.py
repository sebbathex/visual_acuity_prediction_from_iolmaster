"""
Central configuration. Every parameter reported in the Methods section is defined here.

Manuscript: "Inference of Visual Acuity and Postoperative Outcomes from Routine Biometry Images
Using Pretrained Embedding Architectures and Machine Learning Techniques" (JMIR AI, ms#93240).
"""
import os
from pathlib import Path

# Data location: by default next to the code; set COMPENDIUM_DATA_DIR to keep data outside the repository.
ROOT = Path(os.environ.get("COMPENDIUM_DATA_DIR", Path(__file__).resolve().parent))
INPUT_DIR = ROOT / "input"        # input data (due to safety and privacy reasons there is no data included in this repository)
WORK_DIR = ROOT / "work"          # embeddings and analysis datasets
RESULTS_DIR = ROOT / "results"    # tables and predictions

# ── Reproducibility 
SEED = 123            # used for split, CV folds, classifiers and bootstrap
TEST_SIZE = 0.30      # share of patients in the holdout test set
N_FOLDS = 10          # StratifiedGroupKFold within the training set (patient = group)
N_BOOT = 2000         # patient-level cluster bootstrap resamples

# Images 

# Explanation: FR = en-face photograph, AS = horizontal anterior segment scan, MC = macular (fixation) OCT.
# Crop boxes (left, upper, right, lower) in pixels, applied to the image regions rendered from the
# IOLMaster 700 PDF report. They remove labels and borders around each region.
CROP_BOXES = {"FR": (0, 0, 380, 280), "AS": (0, 30, 500, 185), "MC": (23, 0, 115, 205)}

# The postoperative task uses AS + MC only: a design decision made before the analysis to reduce
# dimensionality in the small cohort and to focus on lens and macula. This was identical for all embedders.
IMAGES = {"sameday": ["FR", "AS", "MC"], "postop": ["AS", "MC"]}

EMBEDDERS = ["yolo", "clip", "dinov2"]
EMB_DIM = {"yolo": 256, "clip": 512, "dinov2": 384}   # dimension per image

# Targets 
# Same-day: decimal BCVA on the day of biometry, intervals left-open/right-closed.
#   0 = low (0, 0.30]   1 = middle (0.30, 0.63]   2 = high (0.63, 1.25]
SAMEDAY_BINS = [0.0, 0.30, 0.63, 1.25]       # VA == 0 and VA > 1.25 are excluded as implausible

# Postoperative: gain in logMAR = logMAR(day of biometry) - logMAR(postoperative BCVA).
#   0 = minimal [0, 0.104]   1 = moderate (0.104, 0.301]   2 = large > 0.301
# 0.104 logMAR ~ one chart line; 0.301 logMAR = doubling of decimal VA.
POSTOP_BINS = [0.10374944, 0.30103]
POSTOP_EXCLUDE_WORSENING = True              # gain < 0 excluded (target = magnitude of gain)

# Classifiers
# Pre-specified set, default hyperparameters, balanced class weights. No tuning.
MODELS = ["lr", "rf", "et", "lightgbm", "xgboost", "catboost"]
# Pre-specified selection criterion: highest mean macro-AUC (one-vs-rest) across the CV folds.
SELECTION_METRIC = "AUC"

# Biometry reference model 
# Plausibility limits. Values outside are set to missing and imputed with the training-fold mean.
BIOMETRY_LIMITS = {"r1": (6.0, 10.0), "r2": (6.0, 10.0), "axl": (18.0, 35.0),
                   "acd": (1.5, 6.0), "ast": (-10.0, 10.0)}
