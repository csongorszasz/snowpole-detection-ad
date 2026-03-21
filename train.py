from pathlib import Path
import argparse
import yaml

from ultralytics import YOLO


def main():
	parser = argparse.ArgumentParser(description="YOLO training runner")
	parser.add_argument(
		"-c",
		"--config",
		type=str,
		help="Path to training config (YAML)",
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

	train_args = dict(cfg)
	train_args.pop("model", None)
	if "data" not in train_args:
		raise ValueError("Config must include 'data'")

	data = train_args["data"]
	if not isinstance(data, str):
		raise ValueError("'data' must be a string path to data.yaml")

	if not Path(data).exists():
		raise FileNotFoundError(f"data.yaml not found: {data}")

	model = YOLO(model_path)
	model.train(**train_args)


if __name__ == "__main__":
	main()
