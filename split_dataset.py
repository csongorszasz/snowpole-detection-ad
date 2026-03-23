"""
Given set of unlabeled images.

Creates a random train/val split with placeholder labels as empty .txt files, and a data.yaml for adherence to the YOLO format.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# configurable constants
SOURCE_DIR = Path("dataset/Poles2025/RoadPoles-MSJ")
OUTPUT_DIR = Path("dataset/Poles2025/RoadPoles-MSJ-split")
VAL_RATIO = 0.2
SEED = 42
SPLIT_BY_FOLDER = False
COPY_JSON = False
OVERWRITE = False
CLASS_NAMES = ["pole"]


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def collect_images(source: Path) -> list[Path]:
    excluded_roots = {"train", "valid", "val", "test"}
    images: list[Path] = []
    for path in source.rglob("*"):
        if not _is_image(path):
            continue
        rel = path.relative_to(source)
        if rel.parts and rel.parts[0] in excluded_roots:
            continue
        images.append(path)
    return sorted(images)


def prepare_output_dirs(output_root: Path, overwrite: bool) -> dict[str, tuple[Path, Path]]:
    split_dirs = {
        "train": (output_root / "train" / "images", output_root / "train" / "labels"),
        "valid": (output_root / "valid" / "images", output_root / "valid" / "labels"),
    }

    if overwrite:
        for split in ("train", "valid"):
            split_root = output_root / split
            if split_root.exists():
                shutil.rmtree(split_root)

    for split, (img_dir, lbl_dir) in split_dirs.items():
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        if not overwrite:
            has_files = any(img_dir.iterdir()) or any(lbl_dir.iterdir())
            if has_files:
                raise FileExistsError(
                    f"{split} output folders already contain files under {output_root}. "
                    "Set OVERWRITE = True to recreate the split."
                )

    return split_dirs


def make_split(
    images: list[Path],
    source_root: Path,
    val_ratio: float,
    seed: int,
    split_by_folder: bool,
) -> dict[str, list[Path]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("VAL_RATIO must be between 0 and 1.")

    rng = random.Random(seed)

    if split_by_folder:
        grouped: dict[str, list[Path]] = {}
        for img in images:
            rel = img.relative_to(source_root)
            key = rel.parts[0] if len(rel.parts) > 1 else "__root__"
            grouped.setdefault(key, []).append(img)

        group_keys = sorted(grouped)
        rng.shuffle(group_keys)

        n_val_groups = max(1, int(round(len(group_keys) * val_ratio))) if len(group_keys) > 1 else 0
        val_groups = set(group_keys[:n_val_groups])

        train = [img for k, group in grouped.items() if k not in val_groups for img in group]
        valid = [img for k, group in grouped.items() if k in val_groups for img in group]
    else:
        shuffled = images[:]
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * val_ratio))) if len(shuffled) > 1 else 0
        valid = shuffled[:n_val]
        train = shuffled[n_val:]

    if not train or not valid:
        raise RuntimeError(
            "Split resulted in empty train or valid set. "
            "Adjust VAL_RATIO or set SPLIT_BY_FOLDER = False."
        )

    return {"train": sorted(train), "valid": sorted(valid)}


def unique_destination_name(rel_path: Path) -> str:
    return "__".join(rel_path.parts)


def copy_split(
    split_name: str,
    images: list[Path],
    source_root: Path,
    image_out: Path,
    label_out: Path,
    copy_json: bool,
) -> None:
    copied = 0
    for src_img in images:
        rel = src_img.relative_to(source_root)
        base_name = unique_destination_name(rel)

        dst_img = image_out / base_name
        shutil.copy2(src_img, dst_img)

        dst_lbl = label_out / (dst_img.stem + ".txt")
        dst_lbl.write_text("", encoding="utf-8")

        if copy_json:
            src_json = src_img.with_suffix(".json")
            if src_json.exists():
                dst_json = image_out / (dst_img.stem + ".json")
                shutil.copy2(src_json, dst_json)

        copied += 1

    print(f"[{split_name}] copied {copied} images")


def write_data_yaml(output_root: Path, class_names: list[str]) -> Path:
    if not class_names:
        raise ValueError("CLASS_NAMES must contain at least one class name.")

    lines = [
        "train: train/images",
        "val: valid/images",
        "",
        f"nc: {len(class_names)}",
        f"names: {class_names}",
        "",
    ]

    yaml_path = output_root / "data.yaml"
    yaml_path.write_text("\n".join(lines), encoding="utf-8")
    return yaml_path


def main() -> None:
    source = SOURCE_DIR.resolve()
    output = OUTPUT_DIR.resolve()

    if not source.exists() or not source.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source}")

    split_by_folder = SPLIT_BY_FOLDER

    images = collect_images(source)
    if not images:
        raise RuntimeError(f"No images found under: {source}")

    print(f"Found {len(images)} images")
    print(f"Split mode: {'folder' if split_by_folder else 'image'}")

    split_dirs = prepare_output_dirs(output, overwrite=OVERWRITE)
    split = make_split(
        images=images,
        source_root=source,
        val_ratio=VAL_RATIO,
        seed=SEED,
        split_by_folder=split_by_folder,
    )

    for split_name in ("train", "valid"):
        img_dir, lbl_dir = split_dirs[split_name]
        copy_split(
            split_name=split_name,
            images=split[split_name],
            source_root=source,
            image_out=img_dir,
            label_out=lbl_dir,
            copy_json=COPY_JSON,
        )

    yaml_path = write_data_yaml(output, CLASS_NAMES)

    print("Done.")
    print(f"Train images: {len(split['train'])}")
    print(f"Valid images: {len(split['valid'])}")
    print(f"data.yaml: {yaml_path}")


if __name__ == "__main__":
    main()
