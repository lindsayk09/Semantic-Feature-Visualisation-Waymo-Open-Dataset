"""
dinov2_waymo.py
---------------
Full pipeline: Waymo parquet → PNG extraction → DINOv2 feature
extraction → one 3-panel output image per segment.

Output: segment_01.png, segment_02.png, segment_03.png
Each shows: original frame | PCA heatmap | cosine similarity map
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from sklearn.decomposition import PCA
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as T


DINOV2_MEAN = (0.485, 0.456, 0.406)
DINOV2_STD  = (0.229, 0.224, 0.225)
PATCH_SIZE  = 14
IMG_SIZE    = 518


# ── model ─────────────────────────────────────────────────────────────────────

def load_dinov2() -> nn.Module:
    print("\nLoading DINOv2 ViT-S/14 …")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval()
    print("Model ready ✓")
    return model


# ── image loading ─────────────────────────────────────────────────────────────

def get_transform() -> T.Compose:
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=DINOV2_MEAN, std=DINOV2_STD),
    ])


def load_one_per_segment(folder: str) -> list[tuple]:
    """
    Load exactly ONE representative frame per segment.
    Picks the middle frame of each segment for best scene coverage.
    Returns list of (tensor, pil, filename, segment_id) sorted by segment.
    """
    files = sorted(Path(folder).glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No PNG files in {folder}")

    # Group by segment prefix (first 20 chars of filename)
    segments = {}
    for f in files:
        seg_key = f.name[:20]
        if seg_key not in segments:
            segments[seg_key] = []
        segments[seg_key].append(f)

    print(f"\nFound {len(segments)} segment(s)")

    transform  = get_transform()
    selections = []

    for seg_idx, (seg_key, seg_files) in enumerate(sorted(segments.items()), 1):
        # Pick the middle frame of the segment
        mid   = len(seg_files) // 2
        chosen = seg_files[mid]
        pil    = Image.open(chosen).convert("RGB")
        tensor = transform(pil).unsqueeze(0)
        selections.append((tensor, pil, chosen.name, seg_idx))
        print(f"  Segment {seg_idx}: {seg_key}… → using frame {chosen.name}")

    return selections


# ── feature extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(model: nn.Module, tensor: torch.Tensor) -> dict:
    out     = model.forward_features(tensor)
    patches = out["x_norm_patchtokens"].squeeze(0).numpy()
    cls     = out["x_norm_clstoken"].squeeze(0).numpy()
    grid_n  = IMG_SIZE // PATCH_SIZE
    return {"patch_tokens": patches, "cls_token": cls,
            "grid_h": grid_n, "grid_w": grid_n}


# ── visualisation helpers ─────────────────────────────────────────────────────

def norm01(arr):
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def pca_to_rgb(patch_tokens, grid_h, grid_w):
    pca = PCA(n_components=3)
    projected = norm01(pca.fit_transform(patch_tokens))
    return projected.reshape(grid_h, grid_w, 3)


def upsample(rgb_map, h, w):
    pil = Image.fromarray((rgb_map * 255).astype(np.uint8))
    return np.array(pil.resize((w, h), Image.BILINEAR))


def cosine_map(patch_tokens, query_idx, grid_h, grid_w):
    q     = patch_tokens[query_idx]
    norms = np.linalg.norm(patch_tokens, axis=1) + 1e-8
    sims  = (patch_tokens @ q) / (norms * (np.linalg.norm(q) + 1e-8))
    return sims.reshape(grid_h, grid_w)


# ── plot one 3-panel image ────────────────────────────────────────────────────

def plot_segment_image(result: dict, seg_idx: int, out_path: str) -> None:
    """
    Save one clean 3-panel image for one segment:
    [ Original frame | DINOv2 PCA heatmap | Cosine similarity map ]
    """
    orig = np.array(result["pil"])
    H, W = orig.shape[:2]

    # PCA heatmap
    rgb_map = pca_to_rgb(result["patch_tokens"], result["grid_h"], result["grid_w"])
    rgb_up  = upsample(rgb_map, H, W)

    # Cosine similarity — query = bottom-centre patch (road area)
    query_idx = (result["grid_h"] - 5) * result["grid_w"] + result["grid_w"] // 2
    sim       = cosine_map(result["patch_tokens"], query_idx,
                           result["grid_h"], result["grid_w"])
    sim_up    = upsample(norm01(sim)[:, :, np.newaxis].repeat(3, axis=2), H, W)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.patch.set_facecolor("#111111")

    # Panel 1 — original
    axes[0].imshow(orig)
    axes[0].set_title("Input frame (FRONT camera)", color="white", fontsize=13)
    axes[0].axis("off")

    # Panel 2 — PCA heatmap
    axes[1].imshow(rgb_up)
    axes[1].set_title("DINOv2 patch features → PCA (RGB)\nSame colour = same semantic class",
                      color="white", fontsize=13)
    axes[1].axis("off")

    # Panel 3 — cosine similarity
    im = axes[2].imshow(sim_up[:, :, 0], cmap="plasma")
    axes[2].set_title("Cosine similarity to road patch (✚)\nYellow = most similar",
                      color="white", fontsize=13)
    axes[2].axis("off")
    # mark query patch
    ph = H // result["grid_h"]
    pw = W // result["grid_w"]
    cy = (result["grid_h"] - 5) * ph + ph // 2
    cx = result["grid_w"] // 2 * pw + pw // 2
    axes[2].plot(cx, cy, "w+", markersize=14, markeredgewidth=2.5)
    fig.colorbar(im, ax=axes[2], fraction=0.03, pad=0.02)

    plt.suptitle(
        f"Segment {seg_idx}  —  DINOv2 ViT-S/14  —  Waymo Open Dataset v2.0.1\n"
        f"Self-supervised semantic feature visualisation (no labels used)",
        color="white", fontsize=12, y=1.02
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#111111")
    plt.close()
    print(f"  ✓ Saved → {out_path}")


# ── main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(frames_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load one representative frame per segment
    selections = load_one_per_segment(frames_dir)

    # 2. Load DINOv2
    model = load_dinov2()

    # 3. For each segment: extract features + save image
    print(f"\nGenerating {len(selections)} output image(s) …\n")
    for tensor, pil, fname, seg_idx in tqdm(selections, desc="Processing segments"):
        feats         = extract_features(model, tensor)
        feats["pil"]  = pil
        feats["filename"] = fname

        out_path = os.path.join(out_dir, f"segment_{seg_idx:02d}.png")
        plot_segment_image(feats, seg_idx, out_path)

    print(f"\n✓ Done — {len(selections)} image(s) saved to: {out_dir}")
    for i in range(1, len(selections) + 1):
        print(f"  segment_{i:02d}.png")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DINOv2 — one output image per Waymo segment"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--parquet_dir", type=str,
                       help="Folder with camera_image .parquet files")
    group.add_argument("--frames_dir",  type=str,
                       help="Folder with already-extracted PNG frames")

    parser.add_argument("--out_dir",    type=str, default="../outputs")
    parser.add_argument("--camera",     type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=15)

    args = parser.parse_args()
    frames_dir = args.frames_dir

    if args.parquet_dir is not None:
        from waymo_reader import extract_all_parquets
        frames_dir = os.path.join(args.out_dir, "frames")
        print(f"\n{'='*50}")
        print("Step 1: Extracting frames …")
        extract_all_parquets(
            parquet_dir=args.parquet_dir,
            out_dir=frames_dir,
            camera_id=args.camera,
            max_frames_per_segment=args.max_frames,
        )

    print(f"\n{'='*50}")
    print("Step 2: DINOv2 + visualisation …")
    run_pipeline(frames_dir=frames_dir, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
