from pathlib import Path
import argparse
import shutil
import tempfile
import zipfile
import yaml

from ultralytics import YOLO
from preprocess import is_remote_source, prepare_preprocessed_source
from codecarbon import EmissionsTracker


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
        if not is_remote_source(source) and not looks_like_glob and not source_path.exists():
            raise FileNotFoundError(f"Prediction source not found: {source}")
        if not is_remote_source(source):
            new_source, temp_dir = prepare_preprocessed_source(
                source,
                preprocess_cfg,
                temp_prefix="predict_preprocessed_"
            )
            predict_args["source"] = new_source
        else:
            temp_dir = None
    else:
        temp_dir = None

    model = YOLO(model_path)
    try:
        tracker_tmp_dir = tempfile.TemporaryDirectory(prefix="emissions_tracker_")
        tracker_output_name = "emissions.csv"
        tracker_output_path = Path(tracker_tmp_dir.name) / tracker_output_name

        tracker = EmissionsTracker(
            output_dir=tracker_tmp_dir.name,
            output_file=tracker_output_name,
        )
        tracker.start()

        model.predict(**predict_args)

        emissions = tracker.stop()
        print(f"Emissions: {emissions} kg CO₂")

        results_dir = Path(model.predictor.save_dir) if model.predictor is not None else None
        if results_dir is None:
            print("Prediction finished, but no results directory was found.")
            return

        if tracker_output_path.exists():
            emissions_output_path = results_dir / tracker_output_name
            shutil.move(str(tracker_output_path), str(emissions_output_path))
            print(f"Saved emissions tracking data to: {emissions_output_path}")
        else:
            print(f"Emissions tracking file not found at: {tracker_output_path}")

        labels_dir = results_dir / "labels"
        if not labels_dir.exists() or not labels_dir.is_dir():
            print(f"Prediction finished. Labels directory not found: {labels_dir}")
            return

        zip_path = _zip_label_txt_files(labels_dir)
        print(f"Zipped label files to: {zip_path}")
    finally:
        if 'tracker_tmp_dir' in locals():
            tracker_tmp_dir.cleanup()
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
