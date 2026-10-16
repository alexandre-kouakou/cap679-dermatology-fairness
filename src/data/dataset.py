"""
Fitzpatrick17k PyTorch Dataset.
Reads a *_final.csv (md5hash, skin_group, three_partition_label) and the
matching <md5hash>.jpg images from an image directory.

To be implemented in Week 3 — see roadmap.
"""
import os
from torch.utils.data import Dataset
from PIL import Image


class SkinLesionDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None,
                 target_col="three_partition_label"):
        import pandas as pd
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.target_col = target_col
        self.classes = sorted(self.df[target_col].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(os.path.join(self.img_dir, f"{row['md5hash']}.jpg")).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.class_to_idx[row[self.target_col]]
        # skin_group returned too, for fairness-stratified evaluation
        return img, label, row["skin_group"]
