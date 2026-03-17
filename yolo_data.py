from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class LetterboxMeta:
    original_shape: tuple[int, int]
    resized_shape: tuple[int, int]
    ratio: float
    pad: tuple[int, int]


@dataclass(frozen=True)
class YoloSample:
    image_path: Path
    label_path: Path | None


def list_images(folder: Path) -> list[Path]:
    """List supported image files in a directory."""
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def infer_split_dirs(subset_dir: Path) -> dict[str, Path]:
    """
    Infer split directories from common YOLO layouts.

    Supported patterns:
    - images/train, images/val, images/test
    - train/images, valid/images, test/images
    - any leaf image directory (fallback)
    """
    candidates = {
        "train": [subset_dir / "images" / "train", subset_dir / "train" / "images"],
        "valid": [
            subset_dir / "images" / "valid",
            subset_dir / "images" / "val",
            subset_dir / "valid" / "images",
            subset_dir / "val" / "images",
        ],
        "test": [subset_dir / "images" / "test", subset_dir / "test" / "images"],
    }

    resolved: dict[str, Path] = {}
    for split, dirs in candidates.items():
        for d in dirs:
            if d.exists() and d.is_dir():
                resolved[split] = d
                break

    if resolved:
        return resolved

    leaf_dirs = [
        d
        for d in sorted(subset_dir.rglob("*"))
        if d.is_dir() and not any(c.is_dir() for c in d.iterdir())
    ]
    image_leafs = [d for d in leaf_dirs if any(p.suffix.lower() in IMAGE_EXTENSIONS for p in d.iterdir())]
    return {d.relative_to(subset_dir).as_posix(): d for d in image_leafs}


def list_images_by_split(
    subset_dir: Path,
    split_to_images_dir: dict[str, str] | None = None,
) -> dict[str, list[Path]]:
    """Collect image paths for each split."""
    if split_to_images_dir is None:
        split_dirs = infer_split_dirs(subset_dir)
    else:
        split_dirs = {split: subset_dir / rel_dir for split, rel_dir in split_to_images_dir.items()}

    return {split: list_images(images_dir) for split, images_dir in split_dirs.items()}


def default_label_path(image_path: Path, images_anchor: str = "images", labels_anchor: str = "labels") -> Path:
    """Convert image path to default YOLO label path by replacing images/* with labels/*."""
    parts = list(image_path.parts)
    if images_anchor in parts:
        idx = parts.index(images_anchor)
        parts[idx] = labels_anchor
    label_path = Path(*parts).with_suffix(".txt")
    return label_path


def build_samples(
    image_paths: Iterable[Path],
    infer_labels: bool = True,
) -> list[YoloSample]:
    samples: list[YoloSample] = []
    for image_path in image_paths:
        label_path = default_label_path(image_path) if infer_labels else None
        samples.append(YoloSample(image_path=image_path, label_path=label_path))
    return samples


def load_image(path: str | Path, as_rgb: bool = True) -> np.ndarray:
    """Load image into an HxWxC numpy array."""
    mode = "RGB" if as_rgb else "L"
    with Image.open(path) as img:
        return np.asarray(img.convert(mode))


def letterbox_image(
    image: np.ndarray,
    new_shape: tuple[int, int] = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, LetterboxMeta]:
    """
    Resize while preserving aspect ratio and pad to target shape.

    Returns the padded image and metadata to map coordinates back.
    """
    if image.ndim != 3:
        raise ValueError("Expected an HxWxC image array.")

    h, w = image.shape[:2]
    new_h, new_w = new_shape

    ratio = min(new_h / h, new_w / w)
    resized_h, resized_w = int(round(h * ratio)), int(round(w * ratio))

    pil = Image.fromarray(image)
    resized = np.asarray(pil.resize((resized_w, resized_h), Image.Resampling.BILINEAR))

    pad_w = new_w - resized_w
    pad_h = new_h - resized_h
    left = pad_w // 2
    top = pad_h // 2

    out = np.full((new_h, new_w, 3), color, dtype=np.uint8)
    out[top : top + resized_h, left : left + resized_w] = resized

    meta = LetterboxMeta(
        original_shape=(h, w),
        resized_shape=(resized_h, resized_w),
        ratio=ratio,
        pad=(left, top),
    )
    return out, meta


def preprocess_for_yolo(
    image: np.ndarray,
    image_size: int = 640,
    normalize: bool = True,
) -> tuple[np.ndarray, LetterboxMeta]:
    """
    Generic preprocessing compatible with YOLO-style models.

    Output shape is (1, C, H, W).
    """
    padded, meta = letterbox_image(image, new_shape=(image_size, image_size))
    x = padded.astype(np.float32)
    if normalize:
        x /= 255.0
    x = np.transpose(x, (2, 0, 1))[None, ...]
    return x, meta
