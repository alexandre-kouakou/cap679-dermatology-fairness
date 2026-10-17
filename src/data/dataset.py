import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class SkinLesionDataset(Dataset):
    """Serves (image, target, skin_group) tuples from a splits-merged table.

    Expected columns: image_path, target, skin_group.
    `target` is already 0/1, so no class mapping is needed.
    """

    def __init__(self, df: pd.DataFrame, transform = None):
        self.df = df.reset_index(drop = True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]

        with Image.open(row["image_path"]) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        target = int(row["target"])

        # PyTorch can't batch None; ISIC rows have no skin group.
        skin_group = row["skin_group"]
        skin_group = "unknown" if pd.isna(skin_group) else str(skin_group)

        return image, target, skin_group