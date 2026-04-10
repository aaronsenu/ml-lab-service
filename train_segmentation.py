"""
train_segmentation.py
---------------------
Trains a UNet (ResNet34 backbone, pretrained on ImageNet) on the aerial
house-segmentation dataset produced by generate_masks.py.

Run after generating data:
    python generate_masks.py --demo --out_dir data --n_samples 120
    python train_segmentation.py --data_dir data --epochs 15 --out_dir outputs
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from tqdm import tqdm


# ── Dataset ───────────────────────────────────────────────────────────────────

class HouseSegDataset(Dataset):
    def __init__(self, split_dir: str, img_size: int = 256):
        self.img_dir  = os.path.join(split_dir, "images")
        self.mask_dir = os.path.join(split_dir, "masks")
        self.files    = sorted(os.listdir(self.img_dir))
        self.img_tf   = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225]),
        ])
        self.mask_tf  = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img   = Image.open(os.path.join(self.img_dir,  fname)).convert("RGB")
        mask  = Image.open(os.path.join(self.mask_dir, fname)).convert("L")
        img   = self.img_tf(img)
        mask  = (self.mask_tf(mask) > 0.5).float()
        return img, mask


# ── Metrics ───────────────────────────────────────────────────────────────────

def iou_score(pred: torch.Tensor, target: torch.Tensor,
              threshold: float = 0.5, eps: float = 1e-6) -> float:
    pred   = (pred > threshold).float()
    inter  = (pred * target).sum()
    union  = pred.sum() + target.sum() - inter
    return ((inter + eps) / (union + eps)).item()


def dice_score(pred: torch.Tensor, target: torch.Tensor,
               threshold: float = 0.5, eps: float = 1e-6) -> float:
    pred   = (pred > threshold).float()
    inter  = (pred * target).sum()
    return ((2 * inter + eps) / (pred.sum() + target.sum() + eps)).item()


# ── Train / eval loops ────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train() if training else model.eval()
    total_loss, total_iou, total_dice, n = 0, 0, 0, 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, masks in tqdm(loader, leave=False,
                                desc="train" if training else "val  "):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss  = criterion(preds, masks)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            probs = torch.sigmoid(preds)
            total_loss += loss.item() * imgs.size(0)
            total_iou  += iou_score (probs, masks) * imgs.size(0)
            total_dice += dice_score(probs, masks) * imgs.size(0)
            n          += imgs.size(0)

    return total_loss / n, total_iou / n, total_dice / n


# ── Visualisation ─────────────────────────────────────────────────────────────

def save_sample_predictions(model, dataset, device, out_path: str, n: int = 4):
    model.eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    fig, axes = plt.subplots(n, 3, figsize=(10, n * 3))
    axes[0, 0].set_title("Aerial Image",   fontsize=12)
    axes[0, 1].set_title("Ground Truth",   fontsize=12)
    axes[0, 2].set_title("Predicted Mask", fontsize=12)

    indices = np.random.choice(len(dataset), n, replace=False)
    with torch.no_grad():
        for row, idx in enumerate(indices):
            img, mask = dataset[idx]
            pred = torch.sigmoid(model(img.unsqueeze(0).to(device))).cpu()

            # Denormalise image for display
            img_disp = (img * std + mean).permute(1, 2, 0).numpy().clip(0, 1)
            axes[row, 0].imshow(img_disp)
            axes[row, 1].imshow(mask.squeeze(), cmap="gray")
            axes[row, 2].imshow((pred.squeeze() > 0.5).float(), cmap="gray")
            for ax in axes[row]:
                ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved predictions → {out_path}")


def plot_curves(history: dict, out_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = ["loss", "iou", "dice"]
    titles  = ["Loss", "IoU", "Dice Score"]
    for ax, m, t in zip(axes, metrics, titles):
        ax.plot(history[f"train_{m}"], label="Train")
        ax.plot(history[f"val_{m}"],   label="Val")
        ax.set_title(t); ax.set_xlabel("Epoch")
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved training curves → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Datasets & loaders
    train_ds = HouseSegDataset(os.path.join(args.data_dir, "train"))
    val_ds   = HouseSegDataset(os.path.join(args.data_dir, "val"))
    test_ds  = HouseSegDataset(os.path.join(args.data_dir, "test"))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    print(f"Dataset sizes — train: {len(train_ds)}, "
          f"val: {len(val_ds)}, test: {len(test_ds)}")

    # Model – UNet with ImageNet-pretrained ResNet34 encoder
    model = smp.Unet(
        encoder_name    ="resnet34",
        encoder_weights ="imagenet",
        in_channels     =3,
        classes         =1,
    ).to(device)

    criterion = smp.losses.DiceLoss(mode="binary")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5)

    # Training loop
    history = {k: [] for k in
               ["train_loss","train_iou","train_dice",
                "val_loss",  "val_iou",  "val_dice"]}
    best_val_iou = 0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_iou, tr_dice = run_epoch(
            model, train_loader, criterion, optimizer, device, training=True)
        va_loss, va_iou, va_dice = run_epoch(
            model, val_loader,   criterion, optimizer, device, training=False)
        scheduler.step(va_loss)

        history["train_loss"].append(tr_loss)
        history["train_iou" ].append(tr_iou)
        history["train_dice"].append(tr_dice)
        history["val_loss"  ].append(va_loss)
        history["val_iou"   ].append(va_iou)
        history["val_dice"  ].append(va_dice)

        print(f"Epoch {epoch:02d}/{args.epochs}  "
              f"train loss={tr_loss:.4f} iou={tr_iou:.4f} dice={tr_dice:.4f}  |  "
              f"val   loss={va_loss:.4f} iou={va_iou:.4f} dice={va_dice:.4f}")

        if va_iou > best_val_iou:
            best_val_iou = va_iou
            torch.save(model.state_dict(),
                       os.path.join(args.out_dir, "best_model.pth"))

    # Test evaluation
    model.load_state_dict(torch.load(
        os.path.join(args.out_dir, "best_model.pth"), map_location=device))
    te_loss, te_iou, te_dice = run_epoch(
        model, test_loader, criterion, None, device, training=False)

    print(f"\n=== Test Results ===")
    print(f"  Loss : {te_loss:.4f}")
    print(f"  IoU  : {te_iou:.4f}")
    print(f"  Dice : {te_dice:.4f}")

    # Save metrics
    metrics = {
        "test_loss": te_loss,
        "test_iou" : te_iou,
        "test_dice": te_dice,
        "best_val_iou": best_val_iou,
        "epochs": args.epochs,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save plots
    plot_curves(history, os.path.join(args.out_dir, "training_curves.png"))
    save_sample_predictions(model, test_ds, device,
                            os.path.join(args.out_dir, "predictions.png"))
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="data")
    parser.add_argument("--out_dir",    default="outputs")
    parser.add_argument("--epochs",     type=int,   default=15)
    parser.add_argument("--batch_size", type=int,   default=8)
    parser.add_argument("--lr",         type=float, default=1e-3)
    main(parser.parse_args())