from pathlib import Path
import pandas as pd

COLUMNS = COLUMNS = ["image_path", "target", "skin_group", "group_id", "source"]
FITZ_SKIN_MAP = {
    1: "light", 
    2: "light", 
    3: "medium", 
    4: "medium", 
    5: "dark", 
    6: "dark"
}

DDI_SKIN_MAP = {
    12: "light", 
    34: "medium", 
    56: "dark"
}

def prepare_fitzpatrick(csv_path, img_dir, verbose = True):
    """
    Fitzpatrick17k -> shared record for the benign vs malignant task.
    """
    raw = pd.read_csv(csv_path)

    if verbose:
        print(f"raw rows:                                   {len(raw)}")


    # 1. drop unknown skin type
    df = raw[raw["fitzpatrick"] != -1].copy()
    if verbose:
        print(f"after dropping unknown skin:                {len(df)}")

    # 2. keep only the lesion classes
    df = df[df["three_partition_label"].isin(["benign", "malignant"])].copy()
    if verbose:
        print(f"after keeping benign/malignant skin:        {len(df)}")

    # 3. build the five shared columns
    df["skin_group"] = df["fitzpatrick"].map(FITZ_SKIN_MAP)
    df["target"] = (df["three_partition_label"] == "malignant").astype(int)
    df["image_path"] = df["md5hash"].apply(lambda h: str(Path(img_dir) / f"{h}.jpg"))
    df["group_id"] = df["md5hash"]
    df["source"] = "fitzpatrick17k"

    # 4. fail loudly if anything is wrong
    assert df["skin_group"].notna().all(), "a skin type was not mapped"
    assert df["group_id"].is_unique, "duplicate image IDs"
    missing = ~df["image_path"].map(lambda p: Path(p).exists())
    assert not missing.any(), f"{missing.sum()} images not found in {img_dir}"

    return df[COLUMNS].reset_index(drop = True)

def prepare_ddi(csv_path, img_dir, verbose = True):
    """
    DDI -> shared record for the benign vs malignant task
    """
    df = pd.read_csv(csv_path, index_col = 0)

    if verbose:
        print(f"ddi rows:                                   {len(df)}")

    df["skin_group"] = df["skin_tone"].map(DDI_SKIN_MAP)
    df["target"] = df["malignant"].astype(int)
    df["image_path"] = df["DDI_file"].apply(lambda f: str(Path(img_dir) / f))
    df["group_id"] = df["DDI_file"]
    df["source"] = "ddi"

    assert df["skin_group"].notna().all(), "a skin tone was not mapped"
    assert df["group_id"].is_unique, "duplicate image IDs"
    missing = ~df["image_path"].map(lambda p: Path(p).exists())
    assert not missing.any(), f"{missing.sum()} images not found in {img_dir}"

    return df[COLUMNS].reset_index(drop = True)

def prepare_isic(csv_path, img_dir, verbose = True):
    """
    ISIC -> shared record for the benign vs malignant task.
    """
    raw = pd.read_csv(csv_path)

    if verbose:
        print(f"raw rows:                                   {len(raw)}")

    df = raw[raw["diagnosis_1"].isin(["Benign", "Malignant"])].copy()
    if verbose:
        print(f"after keeping benign/malignant skin:        {len(df)}")

    df["skin_group"] = None
    df["target"] = (df["diagnosis_1"] == "Malignant").astype(int)
    df["image_path"] = df["isic_id"].apply(lambda h: str(Path(img_dir) / f"{h}.jpg"))
    df["group_id"] = df["lesion_id"].fillna(df["isic_id"])
    df["source"] = "isic"

    assert df["skin_group"].isna().all(), "ISIC unexpectedly has skin-tone values"
    assert df["isic_id"].is_unique, "duplicate image IDs"
    assert df["group_id"].notna().all(), "some images have no group_id"
    missing = ~df["image_path"].map(lambda p: Path(p).exists())
    assert not missing.any(), f"{missing.sum()} images not found in {img_dir}"

    return df[COLUMNS].reset_index(drop = True)


