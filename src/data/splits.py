import pandas as pd
from sklearn.model_selection import train_test_split


def _strat_key(df):
    """Combined stratification key, e.g. 'dark_1'."""
    return df["skin_group"] + "_" + df["target"].astype(str)


def make_splits(df, seed, val_frac, test_frac):
    """Stratified train/val/test split on skin_group x target.

    Returns a copy of df with a 'split' column ('train', 'val', 'test').
    Assumes one image per group_id (true for Fitzpatrick17k); the leakage
    guard below stops it if that assumption is broken (e.g. ISIC).
    """
    rest, test = train_test_split(
        df, test_size=test_frac, random_state=seed, stratify=_strat_key(df))
    train, val = train_test_split(
        rest, test_size=val_frac / (1 - test_frac), random_state=seed,
        stratify=_strat_key(rest))

    out = pd.concat([train.assign(split="train"),
                     val.assign(split="val"),
                     test.assign(split="test")])
    out = out.sort_index().reset_index(drop=True)

    # guard: no group may appear in two splits
    ids = {s: set(out.loc[out["split"] == s, "group_id"]) for s in ("train", "val", "test")}
    overlap = (ids["train"] & ids["val"]) | (ids["train"] & ids["test"]) | (ids["val"] & ids["test"])
    assert not overlap, f"{len(overlap)} group_ids appear in more than one split"

    return out


def save_splits(df, path):
    """Save only group_id + split (IDs, not paths), in a stable order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    (df[["group_id", "split"]]
       .sort_values(["split", "group_id"])
       .to_csv(path, index=False))