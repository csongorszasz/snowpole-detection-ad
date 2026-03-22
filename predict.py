from pathlib import Path
import argparse
import glob
import tempfile
import zipfile
import yaml

import cv2
import numpy as np

from ultralytics import YOLO


def _is_remote_source(value: str) -> bool:
    value_l = value.lower()
    return value_l.startswith(("http://", "https://", "rtsp://", "rtmp://", "tcp://"))


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
    }


def _parse_color(mask_cfg: dict) -> tuple[int, int, int]:
    color = mask_cfg.get("color", [0, 0, 0]) # default black
    if not isinstance(color, (list, tuple)) or len(color) != 3:
        raise ValueError("'color' in mask must be a list of 3 ints (BGR)")
    return int(color[0]), int(color[1]), int(color[2])


def _apply_masks(img: np.ndarray, mask_cfg: dict) -> np.ndarray:
    masks_fn = {
        "dashboard": _apply_dashboard_mask,
        "sky": _apply_sky_mask
    }
    for mask_name, mask_fn in masks_fn.items():
        cfg = mask_cfg.get(mask_name)
        if cfg is not None:
            img = mask_fn(img, cfg)
    return img


def _apply_dashboard_mask(img: np.ndarray, mask_cfg: dict) -> np.ndarray:
    if img is None or img.size == 0:
        return img
    if not mask_cfg.get("enabled", True):
        return img

    h, w = img.shape[:2]
    mode = str(mask_cfg.get("mode", "rectangle")).lower()
    color = _parse_color(mask_cfg)

    if mode == "rectangle":
        top_ratio = float(mask_cfg.get("top_ratio", 0.78))
        top_ratio = max(0.0, min(1.0, top_ratio))
        y0 = int(round(h * top_ratio))
        cv2.rectangle(img, (0, y0), (w - 1, h - 1), color, thickness=-1)
        return img

    if mode == "polygon":
        points = mask_cfg.get("points")
        if not isinstance(points, list) or len(points) < 3:
            raise ValueError("'points' is required for polygon mode and must have at least 3 points")

        normalized = bool(mask_cfg.get("normalized", True))
        poly = []
        for p in points:
            if not isinstance(p, (list, tuple)) or len(p) != 2:
                raise ValueError("Each polygon point must be [x, y]")
            x, y = float(p[0]), float(p[1])
            if normalized:
                x = x * w
                y = y * h
            poly.append([int(round(x)), int(round(y))])

        poly_np = np.array([poly], dtype=np.int32)
        cv2.fillPoly(img, poly_np, color)
        return img

    raise ValueError("dashboard_mask.mode must be either 'rectangle' or 'polygon'")


def _apply_sky_mask(img: np.ndarray, mask_cfg: dict) -> np.ndarray:
    if img is None or img.size == 0:
        return img
    if not mask_cfg.get("enabled", True):
        return img
    
    h, w = img.shape[:2]
    color = _parse_color(mask_cfg)

    horizon_ratio = float(mask_cfg.get("horizon_ratio", 0.5))
    horizon_ratio = max(0.0, min(1.0, horizon_ratio))
    y0 = int(round(h * horizon_ratio))
    cv2.rectangle(img, (0, 0), (w - 1, y0), color, thickness=-1)
    return img


def _collect_local_images(source: str) -> tuple[list[Path], Path | None]:
    source_path = Path(source)
    if source_path.exists():
        if source_path.is_file() and _is_image_file(source_path):
            return [source_path], source_path.parent
        if source_path.is_dir():
            images = sorted(p for p in source_path.rglob("*") if p.is_file() and _is_image_file(p))
            return images, source_path
        return [], None

    matches = [Path(p) for p in glob.glob(source, recursive=True)]
    images = sorted(p for p in matches if p.is_file() and _is_image_file(p))
    return images, None


def _prepare_preprocessed_source(source: str, preprocess_cfg: dict) -> tuple[str, tempfile.TemporaryDirectory] | tuple[str, None]:
    mask_cfg = preprocess_cfg.get("mask")
    if not isinstance(mask_cfg, dict):
        return source, None
    
    image_paths, base_dir = _collect_local_images(source)
    if not image_paths:
        print("Preprocess is enabled but no local image files were found in source. Using original source.")
        return source, None

    temp_dir = tempfile.TemporaryDirectory(prefix="predict_preprocessed_")
    temp_root = Path(temp_dir.name)

    for idx, img_path in enumerate(image_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img = _apply_masks(img, mask_cfg)

        if base_dir is not None:
            rel = img_path.relative_to(base_dir)
            out_path = temp_root / rel
        else:
            out_path = temp_root / f"{idx:06d}_{img_path.name}"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), img)

    print(f"Preprocessed {len(image_paths)} images into: {temp_root}")
    return str(temp_root), temp_dir


def _zip_label_txt_files(labels_dir: Path) -> Path:
    zip_path = labels_dir.parent / "labels.zip"
    txt_files = sorted(labels_dir.glob("*.txt"))

    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for txt_file in txt_files:
            zf.write(txt_file, arcname=txt_file.name)

    return zip_path


def main():
    parser = argparse.ArgumentParser(description="YOLO prediction runner")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to prediction config (YAML)",
        required=True
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping/object")

    model_path = cfg.get("model")
    if not model_path:
        raise ValueError("Config must include 'model'")

    predict_args = dict(cfg)
    predict_args.pop("model", None)
    preprocess_cfg = predict_args.pop("preprocess", {})
    if preprocess_cfg is None:
        preprocess_cfg = {}
    if not isinstance(preprocess_cfg, dict):
        raise ValueError("'preprocess' must be a mapping/object if provided")

    if "source" not in predict_args:
        raise ValueError("Config must include 'source'")

    source = predict_args["source"]
    if isinstance(source, str):
        source_path = Path(source)
        looks_like_glob = any(ch in source for ch in ("*", "?", "[", "]"))
        if not _is_remote_source(source) and not looks_like_glob and not source_path.exists():
            raise FileNotFoundError(f"Prediction source not found: {source}")
        if not _is_remote_source(source):
            new_source, temp_dir = _prepare_preprocessed_source(source, preprocess_cfg)
            predict_args["source"] = new_source
        else:
            temp_dir = None
    else:
        temp_dir = None

    model = YOLO(model_path)
    try:
        model.predict(**predict_args)

        results_dir = Path(model.predictor.save_dir) if model.predictor is not None else None
        if results_dir is None:
            print("Prediction finished, but no results directory was found.")
            return

        labels_dir = results_dir / "labels"
        if not labels_dir.exists() or not labels_dir.is_dir():
            print(f"Prediction finished. Labels directory not found: {labels_dir}")
            return

        zip_path = _zip_label_txt_files(labels_dir)
        print(f"Zipped label files to: {zip_path}")
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
