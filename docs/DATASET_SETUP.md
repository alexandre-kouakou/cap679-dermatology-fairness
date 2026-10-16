# Dataset Setup Guide

**Read this before touching any data.** Each teammate downloads every dataset
**under their own access agreement**. We do **not** re-share images between us —
the licenses forbid it (PASSION and DDI especially). What we share is this repo:
the scripts and label files that turn a fresh download into an identical local
setup. Follow the steps below once, on your own machine.

---

## Golden rules (do not break these)
1. **Never share image files** with each other — not by Drive, WhatsApp, USB, or repo.
   Everyone requests access themselves. This keeps us inside every license.
2. **Never commit images or datasets** to git. The `.gitignore` blocks them; don't override.
3. **Keep all images local.** Delete Fitzpatrick17k when the project ends (its terms require it).
4. **In the paper**, cite each dataset and link only its official page — never the data itself,
   and never a direct download link (PASSION forbids this explicitly).

---

## The four datasets

### 1. DDI — Diverse Dermatology Images  *(each person requests)*
- Role: biopsy-confirmed diverse **test** set (our sandbox / gold standard).
- Get it: https://ddi-dataset.github.io/ → follow the Stanford AIMI / Redivis link,
  register, and accept the **Research Use Agreement** yourself.
- Notes: small (656 images); ships with evaluation code. Fast to obtain — do this first.

### 2. Fitzpatrick17k  *(shared request — all names listed)*
- Role: primary **training** set with skin-tone labels.
- Get it: complete the request form and email `groh@mit.edu`. **All four of us are
  named as collaborators on the request**, so each of us is authorised.
- You already have the label CSV in this repo (`data_labels/`). To get the **images**,
  run the download script (below). Expect some dead URLs — that's normal.

### 3. PASSION — sub-Saharan African dermatology  *(each person requests)*
- Role: African-population **evaluation** set (needed from ~Week 8).
- Get it: https://passionderm.github.io/ → submit the data-use request **yourself**.
- ⚠️ License: non-commercial; **no re-identification**; **no re-sharing** — do not pass
  the files or the download link to anyone. Each person requests separately.

### 4. ISIC / HAM10000  *(open — download directly)*
- Role: large **training** source; no native skin-tone labels (we derive an ITA proxy
  or use it label-free — see the team decision note).
- Get it: https://www.isic-archive.com/ (or the Kaggle / Harvard Dataverse mirror).
- License: CC-BY-NC. Attribute in the paper.

---

## Once you have access — reproduce the local setup

```bash
# 0. clone the repo and install deps
git clone <our-repo-url> && cd cap679-dermatology-fairness
python -m venv .venv && source .venv/bin/activate    # Win: .venv\Scripts\activate
pip install -r requirements.txt

# 1. labels already exist in data_labels/ (no images). For Fitzpatrick17k images:
python scripts/02_download_images.py --labels ./data_labels/labels_clean.csv --imgdir ./images --limit 50
#   ^ test run first (50 images). If it works, drop --limit for the full download.

# 2. filter splits to what actually downloaded
python scripts/03_reconcile.py --labeldir ./data_labels --imgdir ./images

# 3. put DDI / ISIC / PASSION under these local paths (do NOT commit them):
#    ./data/ddi/       ./data/isic/       ./data/passion/
```

After this, everyone's `./images` and `./data` look the same, produced from the
same scripts and the same fixed seed (42) — but each was downloaded under that
person's own agreement. That's the point: reproducible setup, zero re-sharing.

---

## What IS ok to share (and already is, via this repo)
- The **label CSVs / splits** in `data_labels/` — small derived metadata, not images.
- All **code**, configs, and docs.
- Results: figures and tables (no raw image dumps).

If onboarding a teammate is slow, the fix is making sure their name is on each
dataset request — not copying files. Ask the data lead (Member A) if unsure.
