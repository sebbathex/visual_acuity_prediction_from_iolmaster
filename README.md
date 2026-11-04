# Visual acuity from routine IOLMaster 700 biometry images – analysis code compendium

Code accompanying the article *“Inference of Visual Acuity and Postoperative Outcomes from Routine
Biometry Images Using Pretrained Embedding Architectures and Machine Learning Techniques: A
Feasibility Study”* (JMIR AI, manuscript 93240).

This repository is a **curated compendium** of the analysis code. It was prepared and curated for publication so that no sensitive data or institution-specific information (data access routines, server addresses, file paths, patient identifiers) leave the clinics IT infrastructure. This repsoitory was therefore optimized for readability and better understanding. 

It implements the analysis protocol and all parameters used for the published results in a compacted form. 

Patient data was strictly excluded. 

Applied to the study data, the compendium reproduces the cohort flow and the patient-level
train/test split of the published analysis exactly. For the postoperative task, it also reproduces
the published holdout results identically (classifier selection, macro-AUCs with 95% CIs, paired comparisons).

## Pipeline

| Step | Script                                                                                                                    | Manuscript section                                                        |
| ---- | ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| 1    | `extract_embeddings.py` – crop the three image regions, embed them with YOLO11n-cls, CLIP ViT-B/32 and DINOv2 ViT-S/14 | Feature Extraction                                                        |
| 2    | `build_dataset.py` – target classes, exclusions, identical eyes for all embedders, one patient-level split             | Dataset Construction and Inclusion Criteria                               |
| 2b   | `biometry_features.py` – numerical biometry variables and missingness indicator                                        | Biometry Reference Model                                                  |
| 3    | `train_evaluate.py` – PyCaret protocol, model selection by CV macro-AUC, holdout with cluster bootstrap                | Machine Learning and Optimization Pipeline                                |
| 4    | `compare.py` – paired cluster bootstrap of differences in holdout macro-AUC                                            | Comparison of Embedding Architectures, Comparison With Numerical Biometry |

All parameters (seed, split, folds, class boundaries, crop boxes, classifiers, plausibility limits) are
defined in `config.py`.

## Protocol in brief

* **Images:** en-face photograph (FR), horizontal anterior segment scan (AS) and macular OCT (MC),
  rendered from the IOLMaster 700 (Carl Zeiss Meditec, Germany) PDF report. Same-day prediction task: all three images. Postoperative task:
  AS + MC (the reader can find the explanation for this study design choice in teh manuscript).
* **Embeddings:** pretrained weights only, no fine-tuning. Per-image vectors are concatenated.
* **Split:** 70/30 patient-level `GroupShuffleSplit` (seed 123), shared by all embedders.
* **Cross-validation:** 10-fold `StratifiedGroupKFold` in the training set, patient as group.
* **Classifiers:** logistic regression, random forest, extra trees, LightGBM, XGBoost, CatBoost.
  Default hyperparameters, balanced class weights, z-standardization fitted on the training folds.
* **Selection:** one classifier per embedder, highest mean cross-validated macro-AUC (one-vs-rest).
* **Primary results:** holdout metrics with 95% CIs from a patient-level cluster bootstrap
  (2,000 resamples). Cross-validation means ± SD are secondary results.

## Input data (not included)

Place the following files in `input/`:

| File                     | Columns                                                                                                             |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------- |
| `images_{task}.csv`    | `patient_id, eye, path_FR, path_AS, path_MC` (images of the examination used for the task)                        |
| `clinical_sameday.csv` | `patient_id, eye, va` (decimal BCVA on the day of biometry)                                                       |
| `clinical_postop.csv`  | `patient_id, eye, va_biometry, va_postop, registry_complete`                                                      |
| `biometry_{task}.csv`  | `patient_id, eye, r1, r2, ast, axl, acd` (examination assigned as described in Multimedia Appendix 1, Section S3) |

`eye` is `OD` or `OS`. Decimal VA codes of 5 or higher denote missing values.

To keep data outside the repository folder, set `COMPENDIUM_DATA_DIR` to a directory containing
`input/`; `work/` and `results/` are then created there as well.

## Usage

```bash
uv sync --extra embeddings          # or: pip install -e ".[embeddings]"
for e in yolo clip dinov2; do python extract_embeddings.py --task sameday --embedder $e; done   # same for postop
python build_dataset.py --task sameday
python build_dataset.py --task postop
python build_dataset.py --task postop --include-incomplete      # sensitivity analysis (postop_full)
python biometry_features.py --task sameday                      # same for postop
for e in yolo clip dinov2 biometry biometry_dinov2; do
    python train_evaluate.py --task sameday --features $e       # same for postop; postop_full: embedders only
done
python compare.py --task sameday                                # same for postop
```

## Notes

* In the original pipeline, the cropped image regions were stored as JPEG files before embedding, and
  the embeddings were stored as text with five significant digits. Embeddings recomputed from images
  with this compendium can therefore differ slightly, which may change results in the last decimal places.
* The YOLO11n-cls embedding is the output of Ultralytics' `embed()` function, i.e. the globally
  average-pooled output of the last backbone layer (layer 9) before the classification head.
* PyCaret's built-in AUC is replaced by an unweighted (macro) one-vs-rest AUC; see `auc_macro_ovr`.
* XGBoost receives balanced sample weights through `BalancedXGBClassifier`, this approach is equivalent to
  `class_weight="balanced"` for the other classifiers.

## Citation

Please cite the article according to the common standards. 

## License

MIT License, © 2026 Sebastian Arens, Klinik für Augenheilkunde, Universitätsklinikum Freiburg.
See [LICENSE](LICENSE).
