"""
generate_masks.py
-----------------
Week 7 pixel mask generation code.
Converts raw aerial images + annotation colour maps into binary house masks.

For this lab we use the Inria Aerial Image Labeling Dataset convention:
  - Ground truth masks are already single-channel PNGs (white = building, black = background)
  - If using colour-coded annotations, the building colour is extracted via threshold.

When no real dataset is available the script can also synthesise a small demo
dataset (--demo flag) so the full pipeline can be exercised end-to-end.
"""

import os
import argparse
import random
import numpy as np
from PIL import Image, ImageDraw
import cv2

# ── Config ────────────────────────────────────────────────────────────────────
PATCH_SIZE   = 256          # pixels – crops fed to the model
BUILDING_RGB = (255, 255, 255)  # Inria: buildings are white in GT masks


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_binary_mask(gt_path: str) -> np.ndarray:
    """
    Load a ground-truth mask PNG and return a binary np.ndarray
    where 1 = building pixel, 0 = background.
    Handles both:
      • Single-channel grayscale masks  (Inria convention)
      • RGB colour-coded masks          (threshold on white)
    """
    img = np.array(Image.open(gt_path).convert("RGB"))
    # Threshold: pixels close to pure white are buildings
    mask = (
        (img[:, :, 0] > 200) &
        (img[:, :, 1] > 200) &
        (img[:, :, 2] > 200)
    ).astype(np.uint8)
    return mask


def save_patch_pair(image: np.ndarray, mask: np.ndarray,
                    out_img_dir: str, out_mask_dir: str, idx: int):
    """Save a (image, mask) patch pair as PNGs."""
    os.makedirs(out_img_dir,  exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)
    Image.fromarray(image).save(os.path.join(out_img_dir,  f"{idx:05d}.png"))
    # Save mask as 0/255 single-channel for easy viewing
    Image.fromarray((mask * 255).astype(np.uint8)).save(
        os.path.join(out_mask_dir, f"{idx:05d}.png"))


def tile_image(image: np.ndarray, mask: np.ndarray,
               patch_size: int = PATCH_SIZE):
    """Yield non-overlapping (image_patch, mask_patch) tiles."""
    h, w = image.shape[:2]
    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            yield (image[y:y+patch_size, x:x+patch_size],
                   mask [y:y+patch_size, x:x+patch_size])


# ── Real dataset processing ───────────────────────────────────────────────────

def process_real_dataset(images_dir: str, masks_dir: str,
                          out_dir: str, patch_size: int = PATCH_SIZE):
    """
    Process a real aerial dataset (Inria layout expected):
      images_dir/  *.tif or *.png  – aerial images
      masks_dir/   *.png           – corresponding GT masks (same stem)
    """
    img_files = sorted([f for f in os.listdir(images_dir)
                        if f.lower().endswith((".png", ".tif", ".jpg"))])

    idx = 0
    for fname in img_files:
        stem = os.path.splitext(fname)[0]
        img_path  = os.path.join(images_dir, fname)
        mask_path = os.path.join(masks_dir,  stem + ".png")
        if not os.path.exists(mask_path):
            print(f"  [skip] no mask for {fname}")
            continue

        image = np.array(Image.open(img_path).convert("RGB"))
        mask  = extract_binary_mask(mask_path)

        for split in ("train", "val", "test"):
            pass  # split assignment happens after all patches are collected

        for img_patch, msk_patch in tile_image(image, mask, patch_size):
            save_patch_pair(img_patch, msk_patch,
                            os.path.join(out_dir, "images"),
                            os.path.join(out_dir, "masks"), idx)
            idx += 1

    print(f"Saved {idx} patches from real dataset → {out_dir}")
    return idx


# ── Demo / synthetic dataset ──────────────────────────────────────────────────

def _random_building(draw: ImageDraw.ImageDraw, w: int, h: int):
    """Draw a random rectangle (building footprint) on a PIL draw context."""
    bx = random.randint(10, w - 60)
    by = random.randint(10, h - 60)
    bw = random.randint(20, min(80, w - bx - 10))
    bh = random.randint(20, min(80, h - by - 10))
    draw.rectangle([bx, by, bx + bw, by + bh], fill=255)


def generate_demo_dataset(out_dir: str, n_samples: int = 120,
                           patch_size: int = PATCH_SIZE, seed: int = 42):
    """
    Synthesise a small aerial-style dataset:
      • 'aerial' images  – green/brown background + grey rectangles (buildings)
      • binary masks     – white where buildings appear
    Splits automatically into train / val / test (70 / 15 / 15 %).
    """
    random.seed(seed)
    np.random.seed(seed)

    splits = {"train": int(0.70 * n_samples),
              "val":   int(0.15 * n_samples)}
    splits["test"] = n_samples - splits["train"] - splits["val"]

    idx = 0
    for split, count in splits.items():
        img_dir  = os.path.join(out_dir, split, "images")
        mask_dir = os.path.join(out_dir, split, "masks")
        os.makedirs(img_dir,  exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        for _ in range(count):
            # ── Aerial image ──────────────────────────────────────────────────
            # Terrain: greenish / brownish noise
            terrain = np.random.randint(60, 140,
                                        (patch_size, patch_size, 3),
                                        dtype=np.uint8)
            terrain[:, :, 0] = np.clip(terrain[:, :, 0] - 30, 0, 255)  # less red
            terrain[:, :, 2] = np.clip(terrain[:, :, 2] - 30, 0, 255)  # less blue

            img_pil  = Image.fromarray(terrain)
            mask_pil = Image.new("L", (patch_size, patch_size), 0)

            img_draw  = ImageDraw.Draw(img_pil)
            mask_draw = ImageDraw.Draw(mask_pil)

            n_buildings = random.randint(1, 6)
            for _ in range(n_buildings):
                bx = random.randint(10, patch_size - 60)
                by = random.randint(10, patch_size - 60)
                bw = random.randint(20, min(70, patch_size - bx - 10))
                bh = random.randint(20, min(70, patch_size - by - 10))
                # Draw grey building on image
                grey = random.randint(150, 220)
                img_draw.rectangle( [bx, by, bx+bw, by+bh], fill=(grey, grey, grey))
                mask_draw.rectangle([bx, by, bx+bw, by+bh], fill=255)

            img_pil.save( os.path.join(img_dir,  f"{idx:05d}.png"))
            mask_pil.save(os.path.join(mask_dir, f"{idx:05d}.png"))
            idx += 1

    print(f"Demo dataset: {idx} samples saved → {out_dir}")
    print(f"  train={splits['train']}  val={splits['val']}  test={splits['test']}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pixel mask generation")
    parser.add_argument("--demo",       action="store_true",
                        help="Generate synthetic demo dataset")
    parser.add_argument("--images_dir", default="raw/images")
    parser.add_argument("--masks_dir",  default="raw/masks")
    parser.add_argument("--out_dir",    default="data")
    parser.add_argument("--n_samples",  type=int, default=120,
                        help="Number of synthetic samples (--demo only)")
    parser.add_argument("--patch_size", type=int, default=PATCH_SIZE)
    args = parser.parse_args()

    if args.demo:
        generate_demo_dataset(args.out_dir, args.n_samples, args.patch_size)
    else:
        process_real_dataset(args.images_dir, args.masks_dir,
                             args.out_dir, args.patch_size)
