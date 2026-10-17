import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.config import (
    BATCH_SIZE, CHECKPOINT_DIR, FITZ_CSV, FITZ_IMG_DIR, DDI_CSV, DDI_IMG_DIR, MODEL_NAME, NUM_WORKERS,
    RANDOM_SEED, RESULTS_DIR, SPLITS_DIR,
)
from src.data.dataset import SkinLesionDataset
from src.data.harmonise import prepare_ddi, prepare_fitzpatrick
from src.data.transforms import get_transforms
from src.models.model import build_model, get_device
from src.models.train import evaluate


def _predict(model, df, device):
    """Run the model over df and return (probs, targets) as numpy arrays."""
    ds = SkinLesionDataset(df, get_transforms(False))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS,
                        persistent_workers=NUM_WORKERS > 0)
    _, probs, targets, _ = evaluate(model, loader, torch.nn.BCEWithLogitsLoss(), device)
    return probs, targets


def _save_predictions(df, probs, targets, set_name, run_name):
    out = df.copy().reset_index(drop=True)
    assert len(out) == len(probs) == len(targets), "row/prediction length mismatch"
    out["prob"] = probs
    assert (out["target"].values == targets).all(), "predictions misaligned with rows"

    cols = ["group_id", "source", "skin_group", "target", "prob"]
    out_dir = RESULTS_DIR / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_name}_{set_name}.csv"
    out[cols].to_csv(path, index=False)
    return path


def main() -> None:
    device = get_device()
    run_name = f"{MODEL_NAME}_seed{RANDOM_SEED}"
    ckpt_path = CHECKPOINT_DIR / f"{run_name}.pt"
    print(f"device={device} run={run_name}")
    print(f"loading checkpoint: {ckpt_path}")

    model = build_model(MODEL_NAME, pretrained=False)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()

    # --- build the three evaluation tables --------------------------------
    fitz_clean = prepare_fitzpatrick(
        csv_path = FITZ_CSV,
        img_dir = FITZ_IMG_DIR,
        verbose = False,
    )
    splits = pd.read_csv(SPLITS_DIR / "fitzpatrick17k_splits.csv")
    fitz = fitz_clean.merge(splits, on="group_id", validate="one_to_one")
    assert len(fitz) == 4320, f"expected 4320 rows after merge, got {len(fitz)}"

    fitz_val  = fitz[fitz["split"] == "val"].reset_index(drop=True)
    fitz_test = fitz[fitz["split"] == "test"].reset_index(drop=True)

    ddi = prepare_ddi(
        csv_path = DDI_CSV, 
        img_dir = DDI_IMG_DIR, 
        verbose = False
    )
    ddi = ddi.reset_index(drop=True)

    sets = {
        "fitz_val":  fitz_val,
        "fitz_test": fitz_test,
        "ddi":       ddi,
    }

    # --- run inference, save, report --------------------------------------
    print()
    results = {}
    for name, df in sets.items():
        probs, targets = _predict(model, df, device)
        auc = roc_auc_score(targets, probs)
        path = _save_predictions(df, probs, targets, name, run_name)
        results[name] = auc
        print(f"{name:10s} n={len(df):4d}  AUC={auc:.4f}  -> {path.name}")

    # --- sanity check: val must reproduce the training-loop best ---------
    print()
    hist = pd.read_csv(RESULTS_DIR / f"{run_name}_history.csv")
    expected_val = hist["val_auc"].max()
    if abs(results["fitz_val"] - expected_val) > 1e-4:
        print(f"WARNING: val AUC {results['fitz_val']:.4f} != expected {expected_val:.4f}")
        print("  checkpoint or evaluation pipeline is not what you think it is.")
    else:
        print(f"val AUC reproduces training-loop best ({expected_val:.4f}).")


if __name__ == "__main__":
    main()