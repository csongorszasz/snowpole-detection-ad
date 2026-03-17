from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class DetectionResult:
    image_path: str
    detections: list[Detection]

    def best(self) -> Detection | None:
        return max(self.detections, key=lambda d: d.confidence, default=None)


class YOLODetector:
    """Thin abstraction over Ultralytics YOLO models (v5/v8/v11 and YOLO-World)."""

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        conf: float = 0.25,
        iou: float = 0.45,
        device: str | int | None = None,
        image_size: int = 640,
    ):
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - import failure handling
            raise RuntimeError(
                "Ultralytics is required. Install with: pip install ultralytics"
            ) from exc

        self._model = YOLO(weights)
        self.conf = conf
        self.iou = iou
        self.device = device
        self.image_size = image_size

    def set_classes(self, class_names: list[str]) -> None:
        """Set text-prompt classes for YOLO-World style models."""
        setter = getattr(self._model, "set_classes", None)
        if setter is None:
            raise RuntimeError(
                "This model does not support dynamic class prompts. "
                "Use YOLO-World weights like 'yolov8s-worldv2.pt'."
            )
        setter(class_names)

    def predict(
        self,
        image: str | Path | np.ndarray,
        object_name: str | None = None,
    ) -> DetectionResult:
        preds = self._model.predict(
            source=image,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )

        pred = preds[0]
        names: dict[int, str] = pred.names  # type: ignore[assignment]

        detections: list[Detection] = []
        for box in pred.boxes:
            cls_id = int(box.cls.item())
            cls_name = names.get(cls_id, str(cls_id))
            conf = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy.squeeze().tolist()

            det = Detection(
                class_id=cls_id,
                class_name=cls_name,
                confidence=conf,
                xyxy=(float(x1), float(y1), float(x2), float(y2)),
            )
            detections.append(det)

        if object_name is not None:
            target = object_name.lower().strip()
            detections = [d for d in detections if d.class_name.lower() == target]

        return DetectionResult(image_path=str(image), detections=detections)


def locate_object(
    detector: YOLODetector,
    image: str | Path | np.ndarray,
    object_name: str,
) -> Detection | None:
    """Run detection and return the highest-confidence match for one object class."""
    result = detector.predict(image=image, object_name=object_name)
    return result.best()


def draw_detection(
    image: np.ndarray,
    detection: Detection,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 3,
) -> np.ndarray:
    """Draw one detection on an RGB image without external CV dependencies."""
    out = image.copy()
    h, w = out.shape[:2]

    x1, y1, x2, y2 = detection.xyxy
    x1i, y1i = max(0, int(round(x1))), max(0, int(round(y1)))
    x2i, y2i = min(w - 1, int(round(x2))), min(h - 1, int(round(y2)))

    out[y1i : y1i + thickness, x1i:x2i] = color
    out[y2i - thickness : y2i, x1i:x2i] = color
    out[y1i:y2i, x1i : x1i + thickness] = color
    out[y1i:y2i, x2i - thickness : x2i] = color

    return out
