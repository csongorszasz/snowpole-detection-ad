from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display
from PIL import Image as PILImage
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models

from sklearn.ensemble import IsolationForest


def list_images(folder: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    return sorted([p for p in folder.glob("*") if p.suffix in exts])


def list_images_by_split(
    subset_dir: Path,
    split_to_images_dir: dict[str, str],
) -> dict[str, list[Path]]:
    return {
        split: list_images(subset_dir / images_dir)
        for split, images_dir in split_to_images_dir.items()
    }

def infer_leaf_split_dirs(root: Path) -> dict[str, str]:
    """Recursively traverse subtree, treating only leaf directories as splits. Returns mapping of (split, path) pairs."""
    leaf_dirs = [
        d for d in sorted(root.rglob("*"))
        if d.is_dir() and not any(child.is_dir() for child in d.iterdir())
    ]
    return {
        d.relative_to(root).as_posix(): d.relative_to(root).as_posix()
        for d in leaf_dirs
    }

def sample_and_display_from_splits(
    subset_dir: Path,
    split_to_images_dir: dict[str, str],
    sample_per_split: int = 8,
    spacing: int = 25,
    rotate_270: bool = False,
    seed: int = 42
):
    """
    split_to_images_dir: If None, recursively traverse subtree down, treating only leaf directories as splits
    """

    if split_to_images_dir is None:
        split_to_images_dir = infer_leaf_split_dirs(subset_dir)

    paths_by_split = list_images_by_split(subset_dir, split_to_images_dir)

    for split, images in paths_by_split.items():
        print(
            f"SPLIT: {split} | total images: {len(images)} | "
            f"sampling size: {sample_per_split}"
        )

        if len(images) == 0:
            continue

        indices = np.random.choice(
            len(images),
            size=min(sample_per_split, len(images)),
            replace=False,
        )
        selected_images = [PILImage.open(images[i]).convert("RGB") for i in indices]
        print("Displaying:", [str(images[i]) for i in indices])

        if rotate_270:
            selected_images = [im.transpose(PILImage.Transpose.ROTATE_270) for im in selected_images]

        total_width = sum(img.width for img in selected_images)
        max_height = max(img.height for img in selected_images)

        spaced_total_width = total_width + spacing * (len(selected_images) - 1)
        combined = PILImage.new("RGB", (spaced_total_width, max_height))

        x_offset = 0
        for img in selected_images:
            combined.paste(img, (x_offset, 0))
            x_offset += img.width
            x_offset += spacing if x_offset + img.width <= spaced_total_width else 0

        buffer = BytesIO()
        combined.save(buffer, format="JPEG")
        buffer.seek(0)
        display(Image(data=buffer.getvalue()))


class ImagePathDataset(Dataset):
    def __init__(self, paths: Iterable[Path], tfm):
        self.paths = list(paths)
        self.tfm = tfm

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        img = PILImage.open(p).convert("RGB")
        return self.tfm(img), str(p)


def build_resnet152_embedder(device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    weights = models.ResNet152_Weights.IMAGENET1K_V2
    tfm = weights.transforms()

    backbone = models.resnet152(weights=weights)
    backbone.fc = torch.nn.Identity()  # 2048-d embedding
    backbone = backbone.to(device).eval()

    return backbone, tfm, device


@torch.no_grad()
def extract_embeddings(
    paths: list[Path],
    backbone,
    tfm,
    device: str,
    batch_size: int = 64,
    num_workers: int = 2,
):
    if len(paths) == 0:
        return np.empty((0, 2048), dtype=np.float32), []

    ds = ImagePathDataset(paths, tfm)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    embs, out_paths = [], []
    for xb, pb in tqdm(dl, total=len(dl)):
        xb = xb.to(device)
        z = backbone(xb)
        embs.append(z.cpu().numpy())
        out_paths.extend(pb)

    return np.vstack(embs), out_paths


def fit_isolation_forest(
    X_train: np.ndarray,
    n_estimators: int = 1000,
    contamination: float = 0.02,
    random_state: int = 42,
    n_jobs: int = -1,
):
    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    iso.fit(X_train)
    return iso


def make_scores_df(paths: list[str], scores: np.ndarray, split: str) -> pd.DataFrame:
    return pd.DataFrame({"split": split, "path": paths, "anomaly_score": scores})


def score_splits(
    iso,
    X_train: np.ndarray,
    X_valid: np.ndarray,
    X_test: np.ndarray,
    P_train: list[str],
    P_valid: list[str],
    P_test: list[str],
) -> pd.DataFrame:
    scores_train = iso.decision_function(X_train)
    scores_valid = iso.decision_function(X_valid)
    scores_test = iso.decision_function(X_test)

    return pd.concat(
        [
            make_scores_df(P_train, scores_train, "train"),
            make_scores_df(P_valid, scores_valid, "valid"),
            make_scores_df(P_test, scores_test, "test"),
        ],
        ignore_index=True,
    )


def score_splits_map(
    iso,
    embeddings_by_split: dict[str, np.ndarray],
    paths_by_split: dict[str, list[str]],
) -> pd.DataFrame:
    frames = []
    for split, embeddings in embeddings_by_split.items():
        scores = iso.decision_function(embeddings)
        frames.append(make_scores_df(paths_by_split[split], scores, split))

    if not frames:
        return pd.DataFrame(columns=["split", "path", "anomaly_score"])

    return pd.concat(frames, ignore_index=True)


def run_anomaly_workflow(
    subset_dir: Path,
    split_to_images_dir: dict[str, str],
    train_split: str = "train",
    device: str | None = None,
    batch_size: int = 64,
    num_workers: int = 1,
    n_estimators: int = 1000,
    contamination: float = 0.02,
    random_state: int = 42,
    n_jobs: int = -1,
):
    """
    split_to_images_dir: If None, recursively traverse subtree down, treating only leaf directories as splits 
    train_split: If None, use the largest split
    """

    if split_to_images_dir is None:
        split_to_images_dir = infer_leaf_split_dirs(subset_dir)

    paths_by_split = list_images_by_split(subset_dir, split_to_images_dir)

    print(
        " | ".join([
            f"{split}={len(paths)}"
            for split, paths in paths_by_split.items()
        ])
    )

    backbone, tfm, device = build_resnet152_embedder(device=device)
    print("device:", device)

    embeddings_by_split: dict[str, np.ndarray] = {}
    string_paths_by_split: dict[str, list[str]] = {}

    for split, paths in paths_by_split.items():
        X, P = extract_embeddings(
            paths,
            backbone=backbone,
            tfm=tfm,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        embeddings_by_split[split] = X
        string_paths_by_split[split] = P
        print(f"{split}: {X.shape}")

    if train_split is None:
        train_split = max(embeddings_by_split.keys(), key=lambda k: len(embeddings_by_split[k]))

    if train_split not in embeddings_by_split:
        raise ValueError(f"train_split '{train_split}' not found in split_to_images_dir")

    if len(embeddings_by_split[train_split]) == 0:
        raise ValueError(f"train_split '{train_split}' has no images")

    iso = fit_isolation_forest(
        embeddings_by_split[train_split],
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    df = score_splits_map(iso, embeddings_by_split, string_paths_by_split)

    return {
        "iso": iso,
        "df": df,
        "device": device,
        "backbone": backbone,
        "tfm": tfm,
        "embeddings_by_split": embeddings_by_split,
        "paths_by_split": string_paths_by_split,
    }


def show_top(df_split: pd.DataFrame, k: int = 12, cols: int = 4, rotate_270: bool = False):
    print(f"Top {k} anomalies for split='{df_split['split'].iloc[0]}'")
    top = df_split.sort_values("anomaly_score", ascending=True).head(k).reset_index(drop=True)

    rows = int(np.ceil(k / cols))
    plt.figure(figsize=(4 * cols, 3.5 * rows))

    for i, row in top.iterrows():
        img = PILImage.open(row["path"]).convert("RGB")
        if rotate_270:
            img = img.transpose(PILImage.Transpose.ROTATE_270)
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(img)
        ax.set_title(f"{Path(row['path']).name}\nscore={row['anomaly_score']:.3f}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def show_top_for_each_split(
    df: pd.DataFrame,
    split_order: list[str] | None = None,
    k: int = 12,
    cols: int = 4,
    rotate_270: bool = False,
):
    if split_order is None:
        split_order = sorted(df["split"].unique().tolist())

    for split in split_order:
        sub = df[df["split"] == split]
        if len(sub) == 0:
            print(f"No rows for split='{split}'")
            continue
        show_top(sub, k=k, cols=cols, rotate_270=rotate_270)
