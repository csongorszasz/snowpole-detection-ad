from pathlib import Path
import glob
import tempfile

import cv2
import numpy as np
import yaml


def is_remote_source(value: str) -> bool:
	value_l = value.lower()
	return value_l.startswith(("http://", "https://", "rtsp://", "rtmp://", "tcp://"))


def _is_image_file(path: Path) -> bool:
	return path.suffix.lower() in {
		".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
	}


def _parse_color(mask_cfg: dict) -> tuple[int, int, int]:
	color = mask_cfg.get("color", [0, 0, 0])
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

		poly = []
		for p in points:
			if not isinstance(p, (list, tuple)) or len(p) != 2:
				raise ValueError("Each polygon point must be [x, y]")
			x, y = float(p[0]), float(p[1])
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


def _preprocess_source_into_dir(source: str, mask_cfg: dict, out_root: Path) -> int:
	image_paths, base_dir = _collect_local_images(source)
	if not image_paths:
		return 0

	written = 0
	for idx, img_path in enumerate(image_paths):
		img = cv2.imread(str(img_path))
		if img is None:
			continue

		img = _apply_masks(img, mask_cfg)

		if base_dir is not None:
			rel = img_path.relative_to(base_dir)
			out_path = out_root / rel
		else:
			out_path = out_root / f"{idx:06d}_{img_path.name}"

		out_path.parent.mkdir(parents=True, exist_ok=True)
		cv2.imwrite(str(out_path), img)
		written += 1

	return written


def prepare_preprocessed_source(
	source: str,
	preprocess_cfg: dict,
	temp_prefix: str = "preprocessed_"
) -> tuple[str, tempfile.TemporaryDirectory] | tuple[str, None]:
	mask_cfg = preprocess_cfg.get("mask")
	if not isinstance(mask_cfg, dict):
		return source, None

	temp_dir = tempfile.TemporaryDirectory(prefix=temp_prefix)
	temp_root = Path(temp_dir.name)

	written = _preprocess_source_into_dir(source, mask_cfg, temp_root)
	if written == 0:
		temp_dir.cleanup()
		print("Preprocess is enabled but no local image files were found in source. Using original source.")
		return source, None

	print(f"Preprocessed {written} images into: {temp_root}")
	return str(temp_root), temp_dir


def prepare_preprocessed_data_yaml(
	data_yaml_path: str,
	preprocess_cfg: dict,
	temp_prefix: str = "train_preprocessed_"
) -> tuple[str, tempfile.TemporaryDirectory] | tuple[str, None]:
	mask_cfg = preprocess_cfg.get("mask")
	if not isinstance(mask_cfg, dict):
		return data_yaml_path, None

	data_yaml = Path(data_yaml_path)
	if not data_yaml.exists():
		raise FileNotFoundError(f"data.yaml not found: {data_yaml_path}")

	with data_yaml.open("r", encoding="utf-8") as f:
		data_cfg = yaml.safe_load(f) or {}
	if not isinstance(data_cfg, dict):
		raise ValueError("data.yaml must be a YAML mapping/object")

	data_yaml_dir = data_yaml.parent.resolve()
	data_root_cfg = data_cfg.get("path")
	if isinstance(data_root_cfg, str) and data_root_cfg:
		data_root = Path(data_root_cfg)
		if not data_root.is_absolute():
			data_root = (data_yaml_dir / data_root).resolve()
	else:
		data_root = data_yaml_dir

	temp_dir = tempfile.TemporaryDirectory(prefix=temp_prefix)
	temp_root = Path(temp_dir.name)
	updated_cfg = dict(data_cfg)

	split_written = {}
	for split in ("train", "val", "test"):
		split_src = data_cfg.get(split)
		if not isinstance(split_src, str) or not split_src:
			continue

		if is_remote_source(split_src):
			print(f"Skipping preprocess for remote '{split}' source: {split_src}")
			continue

		split_path = Path(split_src)
		if not split_path.is_absolute():
			split_path = data_root / split_path
		split_path = split_path.resolve()

		if split_path.is_file() and split_path.suffix.lower() == ".txt":
			print(f"Skipping preprocess for '{split}' source list file: {split_path}")
			continue

		out_dir = temp_root / split
		written = _preprocess_source_into_dir(str(split_path), mask_cfg, out_dir)
		if written > 0:
			split_written[split] = written
			updated_cfg[split] = str(out_dir)

	if not split_written:
		temp_dir.cleanup()
		print("Preprocess is enabled but no train/val/test image sources could be preprocessed. Using original data.yaml.")
		return data_yaml_path, None

	updated_cfg.pop("path", None)
	out_yaml = temp_root / "data.preprocessed.yaml"
	with out_yaml.open("w", encoding="utf-8") as f:
		yaml.safe_dump(updated_cfg, f, sort_keys=False)

	counts = ", ".join(f"{k}={v}" for k, v in split_written.items())
	print(f"Preprocessed dataset splits ({counts}) into: {temp_root}")
	return str(out_yaml), temp_dir
