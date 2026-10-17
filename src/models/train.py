import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.config import (
    BATCH_SIZE, CHECKPOINT_DIR, FITZ_CSV, FITZ_IMG_DIR, LR, MAX_EPOCHS, MODEL_NAME, NUM_WORKERS,
    PATIENCE, RANDOM_SEED, RESULTS_DIR, WEIGHT_DECAY, SPLITS_DIR,
)
from src.data.dataset import SkinLesionDataset
from src.data.harmonise import prepare_fitzpatrick
from src.data.transforms import get_transforms
from src.models.model import build_model, get_device
from src.utils import set_seed


def train_one_epoch(model, loader, loss_fn, optimizer, device) -> float:
    model.train()
    running, n = 0.0, 0
    for imgs, targets, _ in loader:
        imgs = imgs.to(device)
        targets = targets.to(device).float()

        optimizer.zero_grad()
        logits = model(imgs).squeeze(1)          # [B, 1] -> [B]
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        running += loss.item() * len(targets)
        n += len(targets)
    return running / n


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    """One pass, no learning. Returns loss and CPU-collected predictions.

    Returns (mean_loss, probs, targets, skin_groups) with probs/targets as
    numpy arrays and skin_groups as a list — Chapter 3 reuses these directly.
    """
    model.eval()
    running, n = 0.0, 0
    all_probs, all_targets, all_groups = [], [], []

    for imgs, targets, groups in loader:
        imgs = imgs.to(device)
        targets_dev = targets.to(device).float()

        logits = model(imgs).squeeze(1)
        loss = loss_fn(logits, targets_dev)

        running += loss.item() * len(targets)
        n += len(targets)

        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_targets.append(targets.numpy())
        all_groups.extend(groups)

    probs = np.concatenate(all_probs)
    targets = np.concatenate(all_targets)
    return running / n, probs, targets, all_groups


def main() -> None:
    set_seed(RANDOM_SEED)
    device = get_device()
    print(f"device={device} model={MODEL_NAME} seed={RANDOM_SEED}")

    # --- data -------------------------------------------------------------
    fitz_clean = prepare_fitzpatrick(
    csv_path = FITZ_CSV,
    img_dir = FITZ_IMG_DIR,
    verbose = False
    )
    splits = pd.read_csv(SPLITS_DIR / "fitzpatrick17k_splits.csv")
    fitz = fitz_clean.merge(splits, on="group_id", validate="one_to_one")
    assert len(fitz) == 4320, f"expected 4320 rows after merge, got {len(fitz)}"

    train_df = fitz[fitz["split"] == "train"].reset_index(drop=True)
    val_df   = fitz[fitz["split"] == "val"].reset_index(drop=True)
    print(f"train={len(train_df)} val={len(val_df)}")
    
    model = build_model(MODEL_NAME).to(device)

    # NOTE: get_transforms() uses ImageNet stats from config. If MODEL_NAME changes
    # to a backbone with different pretrained_cfg (e.g. ViT → 0.5/0.5), update
    # config.IMAGENET_MEAN / IMAGENET_STD to match model.pretrained_cfg.
    train_ds = SkinLesionDataset(train_df, get_transforms(True))
    val_ds   = SkinLesionDataset(val_df,   get_transforms(False))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0)

    # --- optimisation -----------------------------------------------------
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # --- training loop ----------------------------------------------------
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"{MODEL_NAME}_seed{RANDOM_SEED}.pt"

    best_auc, best_epoch = -1.0, 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_probs, val_targets, _ = evaluate(model, val_loader, loss_fn, device)
        val_auc = roc_auc_score(val_targets, val_probs)
        dt = time.time() - t0

        print(f"epoch {epoch:2d} | train {train_loss:.4f} | val {val_loss:.4f} "
              f"| val AUC {val_auc:.4f} | {dt:.1f}s")

        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "val_auc": val_auc, "seconds": dt})

        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"early stopping at epoch {epoch} "
                      f"(best val AUC {best_auc:.4f})")
                break

    pd.DataFrame(history).to_csv(
        RESULTS_DIR / f"{MODEL_NAME}_seed{RANDOM_SEED}_history.csv", index=False)
    print(f"best val AUC {best_auc:.4f} at epoch {best_epoch} | checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    main()