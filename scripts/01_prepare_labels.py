"""
01_prepare_labels.py
--------------------------------------------------------------------
Fitzpatrick17k: clean the metadata CSV, add a 3-point skin-tone group,
and produce reproducible stratified train / val / test splits.

This step uses ONLY the CSV. It does not need the images.
Output: a clean labels file + three split files, all keyed by md5hash.

Usage:
    python 01_prepare_labels.py \
        --csv fitzpatrick17k.csv \
        --outdir ./data_labels \
        --seed 42
--------------------------------------------------------------------
"""
import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split


# Map Fitzpatrick 1-6 -> your 3-point scale.
# light = I-II, medium = III-IV, dark = V-VI.
FST_TO_GROUP = {1: "light", 2: "light",
                3: "medium", 4: "medium",
                5: "dark", 6: "dark"}

# The classification target. three_partition_label = benign / malignant / non-neoplastic.
TARGET_COL = "three_partition_label"
SKIN_COL = "fitzpatrick_scale"     # the human label (NOT fitzpatrick_centaur)
ID_COL = "md5hash"


def prepare(csv_path, outdir, seed=42, val_frac=0.15, test_frac=0.15):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(csv_path)
    n_start = len(df)

    # --- 1. drop rows with unknown skin tone (coded -1) ---
    df = df[df[SKIN_COL].isin([1, 2, 3, 4, 5, 6])].copy()
    n_after_skin = len(df)

    # --- 2. drop rows with missing target or id ---
    df = df.dropna(subset=[TARGET_COL, ID_COL]).copy()
    n_after_target = len(df)

    # --- 3. add the 3-point skin group ---
    df["skin_group"] = df[SKIN_COL].map(FST_TO_GROUP)

    # --- 4. a combined stratification key: disease-class x skin-group ---
    # This keeps BOTH the target balance and the skin-tone balance across splits,
    # so your small 'dark' group is represented in val and test.
    df["strat_key"] = df[TARGET_COL].astype(str) + "__" + df["skin_group"]

    # Some strat combos may be tiny; fold any singleton keys into a catch-all
    # so train_test_split does not fail on a class with 1 member.
    vc = df["strat_key"].value_counts()
    rare = vc[vc < 3].index
    df.loc[df["strat_key"].isin(rare), "strat_key"] = "rare_combo"

    # --- 5. stratified split: first carve out test, then val from the rest ---
    train_val, test = train_test_split(
        df, test_size=test_frac, random_state=seed, stratify=df["strat_key"])
    val_rel = val_frac / (1.0 - test_frac)
    train, val = train_test_split(
        train_val, test_size=val_rel, random_state=seed,
        stratify=train_val["strat_key"])

    # --- 6. write outputs ---
    keep_cols = [ID_COL, SKIN_COL, "skin_group", "label",
                 "nine_partition_label", TARGET_COL, "qc", "url"]
    clean = df[keep_cols]
    clean.to_csv(os.path.join(outdir, "labels_clean.csv"), index=False)
    train[keep_cols].to_csv(os.path.join(outdir, "train.csv"), index=False)
    val[keep_cols].to_csv(os.path.join(outdir, "val.csv"), index=False)
    test[keep_cols].to_csv(os.path.join(outdir, "test.csv"), index=False)

    # --- 7. report ---
    print(f"Rows in original CSV        : {n_start}")
    print(f"After dropping unknown skin : {n_after_skin}  (-{n_start - n_after_skin})")
    print(f"After dropping missing tgt  : {n_after_target}")
    print(f"Final usable rows           : {len(clean)}")
    print()
    print("Skin-group counts (whole clean set):")
    print(clean["skin_group"].value_counts().reindex(["light", "medium", "dark"]))
    print()
    for name, part in [("train", train), ("val", val), ("test", test)]:
        g = part["skin_group"].value_counts().reindex(["light", "medium", "dark"])
        print(f"{name:5s} n={len(part):5d} | light={g['light']:5d} "
              f"medium={g['medium']:5d} dark={g['dark']:5d}")
    print()
    print("Target distribution (whole clean set):")
    print(clean[TARGET_COL].value_counts())
    print(f"\nWrote labels_clean.csv, train.csv, val.csv, test.csv to {outdir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="fitzpatrick17k.csv")
    ap.add_argument("--outdir", default="./data_labels")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15)
    args = ap.parse_args()
    prepare(args.csv, args.outdir, args.seed, args.val_frac, args.test_frac)
