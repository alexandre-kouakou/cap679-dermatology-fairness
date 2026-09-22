"""
03_reconcile.py
--------------------------------------------------------------------
After downloading, some images will be missing (dead URLs).
This filters train/val/test to ONLY rows whose image actually exists,
so your model never tries to open a file that isn't there.

Run AFTER 02_download_images.py.

Usage:
    python 03_reconcile.py --labeldir ./data_labels --imgdir ./images
--------------------------------------------------------------------
"""
import argparse
import os
import pandas as pd

ID_COL = "md5hash"


def reconcile(labeldir, imgdir):
    have = {os.path.splitext(f)[0]
            for f in os.listdir(imgdir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
            and os.path.getsize(os.path.join(imgdir, f)) > 0}
    print(f"Images present on disk: {len(have)}")

    for split in ["train", "val", "test", "labels_clean"]:
        path = os.path.join(labeldir, f"{split}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        before = len(df)
        df = df[df[ID_COL].isin(have)].copy()
        out = os.path.join(labeldir, f"{split}_final.csv")
        df.to_csv(out, index=False)
        if "skin_group" in df.columns:
            g = df["skin_group"].value_counts().reindex(
                ["light", "medium", "dark"]).fillna(0).astype(int)
            extra = f"| light={g['light']} medium={g['medium']} dark={g['dark']}"
        else:
            extra = ""
        print(f"{split:14s} {before:5d} -> {len(df):5d}  "
              f"({100*len(df)/max(before,1):.1f}% kept) {extra}")
        print(f"   wrote {out}")

    print("\nUse the *_final.csv files for all training and evaluation.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeldir", default="./data_labels")
    ap.add_argument("--imgdir", default="./images")
    args = ap.parse_args()
    reconcile(args.labeldir, args.imgdir)
