import argparse
import os
import numpy as np
import pandas as pd

def generate_spine_split(csv_path, output_dir, val_ratio=0.2, seed=42):
    print(f"[1/4] master annotations")
    raw_df = pd.read_csv(csv_path)

    df_clean = raw_df.dropna(
        subset=["xmin", "ymin", "xmax", "ymax"]
    ).reset_index(drop=True)

    unique_images = df_clean["image_id"].unique()
    print(
        f"[2/4] total annotated images: {len(unique_images)} | total lesion rows: {len(df_clean)}"
    )

    np.random.seed(seed)
    np.random.shuffle(unique_images)

    split_idx = int(len(unique_images) * (1.0 - val_ratio))
    train_ids = set(unique_images[:split_idx])
    val_ids = set(unique_images[split_idx:])

    train_df = df_clean[df_clean["image_id"].isin(train_ids)].reset_index(
        drop=True
    )
    val_df = df_clean[df_clean["image_id"].isin(val_ids)].reset_index(
        drop=True
    )

    os.makedirs(output_dir, exist_ok=True)
    train_out = os.path.join(output_dir, "train_split.csv")
    val_out = os.path.join(output_dir, "val_split.csv")

    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_Out, index=False)

    print(f"[3/4] Train set saved to: {train_out} ({len(train_df)} rows)")
    print(f"[4/4] Val set saved to  : {val_out} ({len(val_df)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate leakage-free splits for SpineCascadeDataset"
    )
    parser.add_argument("--csv", type=str, required=True, help="Path to raw train.csv")
    parser.add_argument("--out_dir", type=str, default="data/annotations", help="Output directory")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation ratio (0.2 = 20%)")
    args = parser.parse_args()

    generate_spine_split(args.csv, args.out_dir, args.val_ratio)