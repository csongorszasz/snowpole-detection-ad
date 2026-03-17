from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Segmentation:
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]
    mask: np.ndarray


class FastSAMSegmenter:
    """Thin abstraction over Ultralytics FastSAM for prompt-free segmentation."""

    def __init__(
        self,
        weights: str = "FastSAM-s.pt",
        conf: float = 0.25,
        iou: float = 0.9,
        device: str | int | None = None,
        image_size: int = 1024,
    ):
        try:
            from ultralytics import FastSAM
        except Exception as exc:  # pragma: no cover - import failure handling
            raise RuntimeError(
                "Ultralytics with FastSAM support is required. Install with: pip install ultralytics"
            ) from exc

        self._model = FastSAM(weights)
        self.conf = conf
        self.iou = iou
        self.device = device
        self.image_size = image_size

    def segment(self, image: str | Path | np.ndarray) -> list[Segmentation]:
        preds = self._model(
            source=image,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.image_size,
            device=self.device,
            retina_masks=True,
            verbose=False,
        )

        pred = preds[0]
        if pred.masks is None or pred.boxes is None:
            return []

        masks = pred.masks.data.detach().cpu().numpy().astype(bool)
        boxes_xyxy = pred.boxes.xyxy.detach().cpu().numpy()
        confs = pred.boxes.conf.detach().cpu().numpy()

        segments: list[Segmentation] = []
        for i in range(len(masks)):
            x1, y1, x2, y2 = boxes_xyxy[i].tolist()
            segments.append(
                Segmentation(
                    class_name="region",
                    confidence=float(confs[i]),
                    xyxy=(float(x1), float(y1), float(x2), float(y2)),
                    mask=masks[i],
                )
            )
        return segments


def locate_sky(segmenter: FastSAMSegmenter, image: str | Path | np.ndarray) -> Segmentation | None:
    """
    Locate sky from FastSAM regions using a geometry prior.

    FastSAM is class-agnostic, so we score candidate masks by:
    - larger area
    - being higher in the image
    - touching the top border
    """
    segments = segmenter.segment(image)
    if not segments:
        return None

    h, w = segments[0].mask.shape
    best_idx = -1
    best_score = -1e9

    for idx, seg in enumerate(segments):
        mask = seg.mask
        area = float(mask.mean())
        if area <= 0:
            continue

        ys, xs = np.where(mask)
        y_min = ys.min()
        y_center = ys.mean()
        x_span = (xs.max() - xs.min() + 1) / max(w, 1)

        top_bias = 1.0 - (y_center / max(h - 1, 1))
        touches_top = 1.0 if y_min == 0 else 0.0

        score = 1.5 * area + 1.0 * top_bias + 0.7 * touches_top + 0.2 * x_span + 0.2 * seg.confidence
        if score > best_score:
            best_score = score
            best_idx = idx

    if best_idx < 0:
        return None

    best = segments[best_idx]
    return Segmentation(
        class_name="sky",
        confidence=best.confidence,
        xyxy=best.xyxy,
        mask=best.mask,
    )


def locate_dashboard(
    segmenter: FastSAMSegmenter,
    image: str | Path | np.ndarray,
    min_score: float = 0.85,
    max_top_ratio: float = 0.62,
    min_row_coverage: float = 0.28,
) -> Segmentation | None:
    """
    Locate dashboard + hood from dashcam perspective (if present).

    FastSAM is class-agnostic. We use a conservative geometry prior:
    - region must live in the bottom half of the frame
    - should be horizontally wide
    - should have support near the bottom border
    - merge compatible lower regions to include both dashboard and hood
    - trim narrow upward spillover to avoid covering road objects

    Returns None when no region looks dashboard/hood-like.
    """
    segments = segmenter.segment(image)
    if not segments:
        return None

    h, w = segments[0].mask.shape
    bottom_start = h // 2
    best_idx = -1
    best_score = -1e9
    candidate_indices: list[int] = []

    for idx, seg in enumerate(segments):
        mask = seg.mask
        area = float(mask.mean())
        if area <= 0.01:
            continue

        ys, xs = np.where(mask)
        if ys.size == 0:
            continue
        y_min = ys.min()
        y_max = ys.max()
        y_center = ys.mean() / max(h - 1, 1)
        x_span = (xs.max() - xs.min() + 1) / max(w, 1)
        y_span = (y_max - y_min + 1) / max(h, 1)
        bottom_half_ratio = float((ys >= bottom_start).mean())

        bottom_bias = y_center
        touches_top = 1.0 if y_min == 0 else 0.0
        bottom_touch_ratio = float(mask[h - 1, :].mean()) if h > 0 else 0.0

        # Prefer dashboard/hood-like lower horizontal regions.
        score = (
            1.2 * bottom_bias
            + 0.9 * x_span
            + 0.7 * bottom_touch_ratio
            + 0.4 * area
            + 0.8 * bottom_half_ratio
            + 0.1 * seg.confidence
            - 0.8 * touches_top
        )

        # Basic shape/position gates to reduce false positives.
        if y_center < 0.55:
            score -= 0.8
        if x_span < 0.35:
            score -= 0.6
        if y_span > 0.75:
            score -= 0.5
        if bottom_half_ratio < 0.8:
            score -= 1.0
        if y_max < bottom_start:
            score -= 2.0

        if score > best_score:
            best_score = score
            best_idx = idx

        if score >= min_score * 0.8 and bottom_half_ratio >= 0.75 and x_span >= 0.25:
            candidate_indices.append(idx)

    if best_idx < 0 or best_score < min_score:
        return None

    best = segments[best_idx]

    # Merge nearby/compatible lower regions so hood + dashboard are captured together.
    merged_mask = np.zeros_like(best.mask, dtype=bool)
    for idx in sorted(set(candidate_indices + [best_idx])):
        seg = segments[idx]
        ys, _ = np.where(seg.mask)
        if ys.size == 0:
            continue
        bottom_half_ratio = float((ys >= bottom_start).mean())
        if bottom_half_ratio < 0.7:
            continue
        merged_mask |= seg.mask

    # Enforce bottom-half-only output as requested.
    merged_mask[:bottom_start, :] = False
    if not merged_mask.any():
        return None

    # Prevent spillover into road: keep only the broad bottom region and
    # clip overly high rows even if they are connected.
    top_clip = max(bottom_start, int(round(h * max_top_ratio)))
    merged_mask[:top_clip, :] = False
    if not merged_mask.any():
        return None

    row_coverage = merged_mask.mean(axis=1)
    keep_rows = np.zeros(h, dtype=bool)
    gap = 0
    started = False
    for y in range(h - 1, top_clip - 1, -1):
        if row_coverage[y] >= min_row_coverage:
            keep_rows[y] = True
            started = True
            gap = 0
        elif started:
            gap += 1
            if gap <= 4:
                keep_rows[y] = True
            else:
                break

    if keep_rows.any():
        merged_mask &= keep_rows[:, None]
    if not merged_mask.any():
        return None

    ys, xs = np.where(merged_mask)
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()

    return Segmentation(
        class_name="dashboard",
        confidence=best.confidence,
        xyxy=(float(x1), float(y1), float(x2), float(y2)),
        mask=merged_mask,
    )


def draw_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 255),
    alpha: float = 0.45,
) -> np.ndarray:
    """Overlay a binary mask on an RGB image."""
    out = image.copy().astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)
    m = mask.astype(bool)
    out[m] = (1.0 - alpha) * out[m] + alpha * color_arr
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_bbox(
    image: np.ndarray,
    xyxy: tuple[float, float, float, float],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 3,
) -> np.ndarray:
    """Draw one bounding box without OpenCV dependency."""
    out = image.copy()
    h, w = out.shape[:2]

    x1, y1, x2, y2 = xyxy
    x1i, y1i = max(0, int(round(x1))), max(0, int(round(y1)))
    x2i, y2i = min(w - 1, int(round(x2))), min(h - 1, int(round(y2)))

    out[y1i : y1i + thickness, x1i:x2i] = color
    out[y2i - thickness : y2i, x1i:x2i] = color
    out[y1i:y2i, x1i : x1i + thickness] = color
    out[y1i:y2i, x2i - thickness : x2i] = color

    return out
