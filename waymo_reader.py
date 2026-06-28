"""
waymo_reader.py
---------------
Reads Waymo Open Dataset v2.0.1 parquet files and extracts
camera images as PNG files ready for DINOv2 processing.

Waymo v2.0.1 uses a modular parquet format where each file
contains one driving segment (~20 seconds, ~200 frames across
5 cameras = ~1000 images per parquet file).

Camera IDs:
    1 = FRONT
    2 = FRONT_LEFT
    3 = FRONT_RIGHT
    4 = SIDE_LEFT
    5 = SIDE_RIGHT

Usage:
    python waymo_reader.py --parquet_dir C:/Users/Lindsay/data/waymo/camera_image
                           --out_dir C:/Users/Lindsay/data/waymo/frames
                           --camera 1
                           --max_frames 20
"""

import os
import io
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm


# Waymo camera name mapping
CAMERA_NAMES = {
    1: "FRONT",
    2: "FRONT_LEFT",
    3: "FRONT_RIGHT",
    4: "SIDE_LEFT",
    5: "SIDE_RIGHT",
}


def load_parquet(parquet_path: str) -> pd.DataFrame:
    """
    Load a Waymo camera_image parquet file into a DataFrame.
    Each row = one camera frame from one camera at one timestamp.

    Key columns:
      - [CameraImageComponent].key.segment_context_name  : segment ID
      - [CameraImageComponent].key.frame_timestamp_micros: timestamp
      - [CameraImageComponent].key.camera_name           : camera ID (1-5)
      - [CameraImageComponent].image                     : JPEG bytes
    """
    print(f"  Loading {Path(parquet_path).name} ...")
    df = pd.read_parquet(parquet_path)
    print(f"  → {len(df)} rows, columns: {list(df.columns)[:4]} ...")
    return df


def extract_frames(
    df: pd.DataFrame,
    out_dir: str,
    camera_id: int = 1,
    max_frames: int = 20,
    segment_name: str = "segment",
) -> list[str]:
    """
    Extract JPEG→PNG frames from a DataFrame for one camera.

    Returns list of saved PNG file paths.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Find the image column — Waymo v2 uses a nested naming convention
    img_col = None
    cam_col = None
    ts_col  = None

    for col in df.columns:
        col_lower = col.lower()
        if "image" in col_lower and img_col is None:
            # prefer the raw image bytes column (not a metadata column)
            if df[col].dtype == object:
                img_col = col
        if "camera_name" in col_lower and cam_col is None:
            cam_col = col
        if "timestamp" in col_lower and ts_col is None:
            ts_col = col

    if img_col is None:
        raise ValueError(
            f"Could not find image column in parquet. "
            f"Available columns:\n{list(df.columns)}"
        )

    print(f"  Image column   : {img_col}")
    print(f"  Camera column  : {cam_col}")
    print(f"  Timestamp col  : {ts_col}")

    # Filter by camera
    if cam_col is not None:
        df_cam = df[df[cam_col] == camera_id].copy()
        print(f"  Frames for camera {camera_id} ({CAMERA_NAMES.get(camera_id,'?')}): {len(df_cam)}")
    else:
        df_cam = df.copy()
        print(f"  No camera column found — using all {len(df_cam)} rows")

    # Sort by timestamp for chronological order
    if ts_col is not None:
        df_cam = df_cam.sort_values(ts_col).reset_index(drop=True)

    # Cap at max_frames
    df_cam = df_cam.head(max_frames)

    saved = []
    for i, row in tqdm(df_cam.iterrows(), total=len(df_cam), desc="  Extracting frames"):
        img_bytes = row[img_col]

        # Waymo stores images as JPEG bytes
        if isinstance(img_bytes, (bytes, bytearray)):
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        elif isinstance(img_bytes, np.ndarray):
            img = Image.fromarray(img_bytes.astype(np.uint8))
        else:
            print(f"  Skipping row {i}: unexpected image type {type(img_bytes)}")
            continue

        cam_name = CAMERA_NAMES.get(camera_id, f"cam{camera_id}")
        fname = f"{segment_name}_{cam_name}_frame{i:04d}.png"
        out_path = os.path.join(out_dir, fname)
        img.save(out_path)
        saved.append(out_path)

    print(f"  ✓ Saved {len(saved)} frames to {out_dir}")
    return saved


def extract_all_parquets(
    parquet_dir: str,
    out_dir: str,
    camera_id: int = 1,
    max_frames_per_segment: int = 20,
) -> list[str]:
    """
    Process all .parquet files in parquet_dir and extract frames.
    Returns all saved PNG paths.
    """
    parquet_files = sorted(Path(parquet_dir).glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No .parquet files found in {parquet_dir}")

    print(f"\nFound {len(parquet_files)} parquet file(s)")
    all_saved = []

    for pf in parquet_files:
        print(f"\n{'─'*50}")
        print(f"Segment: {pf.stem[:30]}...")
        segment_name = pf.stem[:20]   # short ID for filenames

        try:
            df  = load_parquet(str(pf))
            saved = extract_frames(
                df,
                out_dir=out_dir,
                camera_id=camera_id,
                max_frames=max_frames_per_segment,
                segment_name=segment_name,
            )
            all_saved.extend(saved)
        except Exception as e:
            print(f"  ✗ Failed on {pf.name}: {e}")
            continue

    print(f"\n{'='*50}")
    print(f"Total frames extracted: {len(all_saved)}")
    return all_saved


def inspect_parquet(parquet_path: str) -> None:
    """
    Print a detailed schema inspection of a parquet file.
    Useful for debugging column names in different Waymo versions.
    """
    df = pd.read_parquet(parquet_path)
    print(f"\nFile: {Path(parquet_path).name}")
    print(f"Shape: {df.shape}")
    print(f"\nColumns:")
    for col in df.columns:
        dtype = df[col].dtype
        sample = df[col].iloc[0] if len(df) > 0 else "N/A"
        if isinstance(sample, (bytes, bytearray)):
            sample_str = f"<bytes len={len(sample)}>"
        elif isinstance(sample, np.ndarray):
            sample_str = f"<array shape={sample.shape}>"
        else:
            sample_str = str(sample)[:60]
        print(f"  {col:<60} dtype={dtype}  sample={sample_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract PNG frames from Waymo v2.0.1 parquet files."
    )
    parser.add_argument("--parquet_dir", required=True,
                        help="Folder containing camera_image .parquet files")
    parser.add_argument("--out_dir",     required=True,
                        help="Output folder for extracted PNG frames")
    parser.add_argument("--camera",      type=int, default=1,
                        help="Camera ID: 1=FRONT 2=FL 3=FR 4=SL 5=SR (default: 1)")
    parser.add_argument("--max_frames",  type=int, default=20,
                        help="Max frames per segment to extract (default: 20)")
    parser.add_argument("--inspect",     action="store_true",
                        help="Just print parquet schema, don't extract")
    args = parser.parse_args()

    if args.inspect:
        # inspect first file only
        first = sorted(Path(args.parquet_dir).glob("*.parquet"))[0]
        inspect_parquet(str(first))
    else:
        extract_all_parquets(
            parquet_dir=args.parquet_dir,
            out_dir=args.out_dir,
            camera_id=args.camera,
            max_frames_per_segment=args.max_frames,
        )
