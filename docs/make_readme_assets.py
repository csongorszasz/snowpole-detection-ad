"""Build the images shown in README.md from files already in saved_runs/.

Usage:
    python docs/make_readme_assets.py

Writes to docs/assets/:
    hero.jpg                      dashcam + iPhone predictions side by side
    feature_maps_{light,dark}.png strongest pole-responding channel per YOLO stage
    nano_vs_small_{light,dark}.png validation mAP50-95 per epoch, YOLO26n vs YOLO26s
and prints the metrics table used in the README.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "saved_runs" / "detect"
OUT = ROOT / "docs" / "assets"

RUNS_INFO = {
    "YOLO26n": RUNS / "train",
    "YOLO26s": RUNS / "train2",
}

# Reference categorical palette (dataviz skill), stepped per mode.
THEMES = {
    "light": {
        "text": "#191919",
        "muted": "#666666",
        "grid": "#d9d9d9",
        "series": ["#257de0", "#f47230"],
    },
    "dark": {
        "text": "#efefef",
        "muted": "#9c9c9c",
        "grid": "#3a3f47",
        "series": ["#338ced", "#e76624"],
    },
}

FEATURE_IMAGE = "919176034"
FEATURE_DIR = RUNS / "predict_RoadpolesiPhone_nano_visualization"
FEATURE_STAGES = [
    ("stage0_Conv", "Stage 0 · Conv\n320×192"),
    ("stage2_C3k2", "Stage 2 · C3k2\n160×96"),
    ("stage4_C3k2", "Stage 4 · C3k2\n80×48"),
    ("stage16_C3k2", "Stage 16 · C3k2\n80×48 (P3 head)"),
    ("stage19_C3k2", "Stage 19 · C3k2\n40×24 (P4 head)"),
]


def set_font() -> None:
    names = {f.name for f in font_manager.fontManager.ttflist}
    if "Inter" in names:
        plt.rcParams["font.family"] = "Inter"
    plt.rcParams["font.size"] = 11


# --------------------------------------------------------------------------- hero
def make_hero(height: int = 600) -> Path:
    dashcam = Image.open(RUNS / "predict_roadpolesv1_nano_visualization" / "frame_000005.jpg")
    iphone = Image.open(FEATURE_DIR / f"{FEATURE_IMAGE}.jpg")
    # Drop the dashboard at the bottom of the iPhone frame; the pole is at ~y=0.5.
    iphone = iphone.crop((0, 0, iphone.width, int(iphone.height * 0.65)))

    def fit(img: Image.Image) -> Image.Image:
        w = round(img.width * height / img.height)
        return img.convert("RGB").resize((w, height), Image.LANCZOS)

    a, b = fit(dashcam), fit(iphone)
    gap = 8
    canvas = Image.new("RGB", (a.width + gap + b.width, height), "white")
    canvas.paste(a, (0, 0))
    canvas.paste(b, (a.width + gap, 0))
    path = OUT / "hero.jpg"
    canvas.save(path, quality=85, optimize=True, progressive=True)
    return path


# ------------------------------------------------------------------ feature maps
def pole_box() -> tuple[float, float, float, float]:
    """Return the detected pole box as normalized (xc, yc, w, h)."""
    line = (FEATURE_DIR / "labels" / f"{FEATURE_IMAGE}.txt").read_text().split()
    return tuple(float(v) for v in line[1:5])  # type: ignore[return-value]


def box_to_feature(box, img_w, img_h, feat_h, feat_w):
    """Map a normalized image box into feature-map pixel slices (letterboxed input)."""
    xc, yc, bw, bh = box
    # Ultralytics letterbox: scale long side to feat_h * stride, pad the short side.
    scale = feat_h / img_h
    pad_x = (feat_w - img_w * scale) / 2
    pad_y = (feat_h - img_h * scale) / 2 if img_h * scale < feat_h else 0
    x0 = pad_x + (xc - bw / 2) * img_w * scale
    x1 = pad_x + (xc + bw / 2) * img_w * scale
    y0 = pad_y + (yc - bh / 2) * img_h * scale
    y1 = pad_y + (yc + bh / 2) * img_h * scale
    xs = slice(max(int(np.floor(x0)), 0), max(int(np.ceil(x1)), int(np.floor(x0)) + 1))
    ys = slice(max(int(np.floor(y0)), 0), max(int(np.ceil(y1)), int(np.floor(y0)) + 1))
    return ys, xs, (x0, y0, x1 - x0, y1 - y0)


def best_pole_channel(feats: np.ndarray, ys: slice, xs: slice) -> int:
    """Channel whose activation inside the pole box stands out most from the rest of the map."""
    inside = feats[:, ys, xs].mean(axis=(1, 2))
    mean = feats.mean(axis=(1, 2))
    std = feats.std(axis=(1, 2)) + 1e-6
    return int(np.argmax((inside - mean) / std))


def make_feature_maps(theme: str) -> Path:
    t = THEMES[theme]
    src = Image.open(FEATURE_DIR / f"{FEATURE_IMAGE}.jpg")
    box = pole_box()

    n = len(FEATURE_STAGES) + 1
    fig, axes = plt.subplots(1, n, figsize=(2.05 * n, 4.6), dpi=150)
    fig.patch.set_alpha(0)

    first = np.load(FEATURE_DIR / FEATURE_IMAGE / f"{FEATURE_STAGES[0][0]}_features.npy")
    fh, fw = first.shape[1:]
    # Letterboxed input at the first stage's resolution, so every panel shares a frame.
    scale = fh / src.height
    resized = src.convert("RGB").resize((round(src.width * scale), fh), Image.LANCZOS)
    letterbox = Image.new("RGB", (fw, fh), (114, 114, 114))
    letterbox.paste(resized, ((fw - resized.width) // 2, 0))
    axes[0].imshow(letterbox)
    _, _, rect = box_to_feature(box, src.width, src.height, fh, fw)
    axes[0].add_patch(plt.Rectangle(rect[:2], rect[2], rect[3], fill=False, ec="#ffffff", lw=1.5))
    axes[0].set_title("Input\n640×384", color=t["text"], fontsize=10, linespacing=1.4)

    for ax, (stem, label) in zip(axes[1:], FEATURE_STAGES):
        feats = np.load(FEATURE_DIR / FEATURE_IMAGE / f"{stem}_features.npy")
        h, w = feats.shape[1:]
        ys, xs, rect = box_to_feature(box, src.width, src.height, h, w)
        ch = best_pole_channel(feats, ys, xs)
        ax.imshow(feats[ch], cmap="viridis", interpolation="nearest", extent=(0, w, h, 0))
        ax.add_patch(plt.Rectangle(rect[:2], rect[2], rect[3], fill=False, ec="#ffffff", lw=1, ls=(0, (2, 2))))
        ax.set_title(label, color=t["text"], fontsize=10, linespacing=1.4)
        ax.text(0.5, -0.04, f"channel {ch}", transform=ax.transAxes, ha="center", va="top",
                color=t["muted"], fontsize=9)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    fig.tight_layout(w_pad=0.6)
    path = OUT / f"feature_maps_{theme}.png"
    fig.savefig(path, transparent=True)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- training chart
def load_results(run_dir: Path) -> list[dict[str, float]]:
    with open(run_dir / "results.csv", newline="") as f:
        return [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(f)]


def ema(values: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    out = np.empty_like(values)
    acc = values[0]
    for i, v in enumerate(values):
        acc = alpha * v + (1 - alpha) * acc
        out[i] = acc
    return out


def make_training_chart(theme: str) -> Path:
    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=150)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    key = "metrics/mAP50-95(B)"
    for (name, run_dir), color in zip(RUNS_INFO.items(), t["series"]):
        rows = load_results(run_dir)
        ep = np.array([r["epoch"] for r in rows])
        y = np.array([r[key] for r in rows])
        ax.plot(ep, y, color=color, lw=1, alpha=0.25)
        ax.plot(ep, ema(y), color=color, lw=2, label=name)
        i = int(np.argmax(y))
        ax.scatter([ep[i]], [y[i]], s=40, color=color, edgecolor=t["text"], linewidth=0, zorder=3)
        # Small model peaks above the curves; nano's label sits below them to avoid overlap.
        above = name.endswith("s")
        ax.annotate(f"{name}  {y[i]:.3f}", (ep[i], y[i]), xytext=(0, 8 if above else -36),
                    textcoords="offset points", ha="center", va="bottom", color=t["text"], fontsize=10)

    ax.set_ylim(0.55, 0.86)
    ax.set_xlim(0, None)
    ax.set_xlabel("Epoch", color=t["muted"])
    ax.set_ylabel("Validation mAP50-95", color=t["muted"])
    ax.tick_params(colors=t["muted"], length=0)
    ax.grid(axis="y", color=t["grid"], lw=0.8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    leg = ax.legend(loc="lower right", frameon=False)
    for txt in leg.get_texts():
        txt.set_color(t["text"])
    ax.set_title("Validation mAP50-95 per epoch (thin: raw, thick: smoothed, dot: best)",
                 color=t["text"], fontsize=11, loc="left")

    fig.tight_layout()
    path = OUT / f"nano_vs_small_{theme}.png"
    fig.savefig(path, transparent=True)
    plt.close(fig)
    return path


# ------------------------------------------------------------------ metrics table
def print_metrics_table() -> None:
    print("| Model | Epochs run | Best epoch | Precision | Recall | mAP50 | mAP50-95 | Train time | Energy | CO₂eq |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, run_dir in RUNS_INFO.items():
        rows = load_results(run_dir)
        best = max(rows, key=lambda r: r["metrics/mAP50-95(B)"])
        with open(run_dir / "emissions.csv", newline="") as f:
            em = list(csv.DictReader(f))[-1]
        print(
            f"| {name} | {len(rows)} | {int(best['epoch'])} "
            f"| {best['metrics/precision(B)']:.3f} | {best['metrics/recall(B)']:.3f} "
            f"| {best['metrics/mAP50(B)']:.3f} | {best['metrics/mAP50-95(B)']:.3f} "
            f"| {float(em['duration']) / 60:.0f} min | {float(em['energy_consumed']):.2f} kWh "
            f"| {float(em['emissions']) * 1000:.1f} g |"
        )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    set_font()
    paths = [make_hero()]
    for theme in THEMES:
        paths.append(make_feature_maps(theme))
        paths.append(make_training_chart(theme))
    for p in paths:
        print(f"wrote {p.relative_to(ROOT)} ({p.stat().st_size // 1024} KB)")
    print()
    print_metrics_table()


if __name__ == "__main__":
    main()
