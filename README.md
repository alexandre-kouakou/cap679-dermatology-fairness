# Cross-Population Fairness in Skin-Lesion Classification

CAP679 — Next Gen Computing Project II · Lovely Professional University

Evaluating and mitigating skin-tone bias in deep-learning skin-lesion
classifiers, with a focus on transfer to sub-Saharan African populations.

## What this project does
Trains CNN and Vision-Transformer classifiers on Western dermatology datasets
(Fitzpatrick17k, ISIC) and evaluates them out-of-distribution on diverse sets
(DDI, PASSION) to measure how performance and fairness degrade on darker skin.
It then compares fairness-mitigation methods for whether their gains transfer
across populations.

## ⚠️ Data policy — read before you touch the datasets
- **Never commit images or raw datasets.** They are covered by non-commercial
  research licenses (PASSION, DDI, Fitzpatrick17k) and are far too large for git.
  The `.gitignore` blocks them; do not override it.
- Keep all downloaded images **local only**. Delete Fitzpatrick17k images when
  the research is complete (per its data-request terms).
- Only small label CSVs and metadata belong in the repo.

## Repository layout
```
configs/       experiment configs (yaml)
data_labels/   label CSVs + splits (NO images)
src/data/      dataset classes, preprocessing, label harmonisation
src/models/    backbones and training
src/eval/      fairness metrics, evaluation, figures
scripts/       data prep + download (01_, 02_, 03_)
notebooks/     exploration only (keep outputs cleared)
results/       figures and tables (no heavy dumps)
paper/         the conference paper draft
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Pipeline (run order)
```bash
# 1. labels (CSV only) — already prepared
python scripts/01_prepare_labels.py --csv fitzpatrick17k.csv --outdir ./data_labels

# 2. images (run locally; fetches from atlas sites)
python scripts/02_download_images.py --labels ./data_labels/labels_clean.csv --imgdir ./images

# 3. reconcile splits to downloaded images
python scripts/03_reconcile.py --labeldir ./data_labels --imgdir ./images
```

## Team & workflow
- Member A — Data · Member B — Modelling · Member C — Methods · Member D — Evaluation
- Branch per feature: `feature/<name>`; open a PR into `main`; one teammate reviews.
- Never push directly to `main`. Never commit data or checkpoints.
- Log experiments to Weights & Biases, not to git.

## License
Code: MIT (see LICENSE). Datasets retain their own licenses and are NOT included.
