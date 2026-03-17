from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import importlib
from pathlib import Path
from typing import Mapping, Sequence
import base64

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image as PILImage
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ProjectionConfig:
    method: str = "umap"  # "umap" | "tsne"
    n_components: int = 2
    random_state: int = 42
    scale_before_project: bool = True

    # t-SNE params
    tsne_perplexity: float = 30.0
    tsne_n_iter: int = 1000
    tsne_learning_rate: str | float = "auto"
    tsne_init: str = "pca"

    # UMAP params
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = "euclidean"


def _validate_inputs(
    embeddings_by_split: Mapping[str, np.ndarray],
    paths_by_split: Mapping[str, Sequence[str]],
):
    if not embeddings_by_split:
        raise ValueError("embeddings_by_split is empty")

    missing = set(embeddings_by_split.keys()) - set(paths_by_split.keys())
    if missing:
        raise ValueError(f"paths_by_split missing splits: {sorted(missing)}")

    for split, X in embeddings_by_split.items():
        if X.ndim != 2:
            raise ValueError(f"Split '{split}' must be a 2D array, got shape {X.shape}")
        if len(paths_by_split[split]) != len(X):
            raise ValueError(
                f"Length mismatch in split '{split}': "
                f"len(paths)={len(paths_by_split[split])} vs rows={len(X)}"
            )


def _sample_rows_per_split(
    embeddings_by_split: Mapping[str, np.ndarray],
    paths_by_split: Mapping[str, Sequence[str]],
    max_points_per_split: int | None,
    random_state: int,
):
    rng = np.random.default_rng(random_state)

    sampled_embeddings: dict[str, np.ndarray] = {}
    sampled_paths: dict[str, list[str]] = {}

    for split, X in embeddings_by_split.items():
        n = len(X)
        if max_points_per_split is None or n <= max_points_per_split:
            idx = np.arange(n)
        else:
            idx = rng.choice(n, size=max_points_per_split, replace=False)

        sampled_embeddings[split] = X[idx]
        sampled_paths[split] = [str(paths_by_split[split][i]) for i in idx]

    return sampled_embeddings, sampled_paths


def _stack_splits(
    embeddings_by_split: Mapping[str, np.ndarray],
    paths_by_split: Mapping[str, Sequence[str]],
):
    split_order = list(embeddings_by_split.keys())

    X_parts: list[np.ndarray] = []
    split_col: list[str] = []
    path_col: list[str] = []

    for split in split_order:
        X = embeddings_by_split[split]
        X_parts.append(X)
        split_col.extend([split] * len(X))
        path_col.extend([str(p) for p in paths_by_split[split]])

    X_all = np.vstack(X_parts)
    return X_all, split_col, path_col


def fit_transform_embeddings(
    X: np.ndarray,
    config: ProjectionConfig,
) -> np.ndarray:
    method = config.method.lower()

    if config.scale_before_project:
        X = StandardScaler().fit_transform(X)

    if method == "tsne":
        perplexity = config.tsne_perplexity
        projector = TSNE(
            n_components=config.n_components,
            perplexity=perplexity,
            learning_rate=config.tsne_learning_rate,
            init=config.tsne_init,
            max_iter=config.tsne_n_iter,
            random_state=config.random_state,
        )
        return projector.fit_transform(X)

    if method == "umap":
        try:
            import umap
        except ImportError as e:
            raise ImportError(
                "UMAP is not installed. Install with: pip install umap-learn"
            ) from e

        projector = umap.UMAP(
            n_components=config.n_components,
            n_neighbors=config.umap_n_neighbors,
            min_dist=config.umap_min_dist,
            metric=config.umap_metric,
            random_state=config.random_state,
        )
        return projector.fit_transform(X)

    raise ValueError("Unsupported method. Use 'tsne' or 'umap'.")


def project_splits(
    embeddings_by_split: Mapping[str, np.ndarray],
    paths_by_split: Mapping[str, Sequence[str]],
    config: ProjectionConfig | None = None,
    max_points_per_split: int | None = None,
) -> pd.DataFrame:
    """
    Build a 2D/ND projection DataFrame from split-wise embeddings.

    Inputs mirror the output format in anomaly workflows:
    - embeddings_by_split: {split: np.ndarray[num_images, emb_dim]}
    - paths_by_split: {split: list[str]}
    """
    if config is None:
        config = ProjectionConfig()

    _validate_inputs(embeddings_by_split, paths_by_split)

    sampled_embeddings, sampled_paths = _sample_rows_per_split(
        embeddings_by_split=embeddings_by_split,
        paths_by_split=paths_by_split,
        max_points_per_split=max_points_per_split,
        random_state=config.random_state,
    )

    X_all, split_col, path_col = _stack_splits(sampled_embeddings, sampled_paths)
    Z = fit_transform_embeddings(X_all, config=config)

    out = pd.DataFrame(
        {
            "split": split_col,
            "path": path_col,
        }
    )
    for i in range(Z.shape[1]):
        out[f"dim_{i + 1}"] = Z[:, i]

    return out


def project_from_workflow(
    workflow: Mapping[str, object],
    config: ProjectionConfig | None = None,
    max_points_per_split: int | None = None,
) -> pd.DataFrame:
    """
    Convenience wrapper for the dictionary returned by run_anomaly_workflow(...).
    """
    try:
        embeddings_by_split = workflow["embeddings_by_split"]
        paths_by_split = workflow["paths_by_split"]
    except KeyError as e:
        raise KeyError(
            "workflow must contain keys: 'embeddings_by_split' and 'paths_by_split'"
        ) from e

    return project_splits(
        embeddings_by_split=embeddings_by_split,
        paths_by_split=paths_by_split,
        config=config,
        max_points_per_split=max_points_per_split,
    )


def plot_projection_scatter(
    projection_df: pd.DataFrame,
    x_col: str = "dim_1",
    y_col: str = "dim_2",
    split_order: list[str] | None = None,
    figsize: tuple[int, int] = (10, 8),
    alpha: float = 0.8,
    s: int = 14,
    title: str | None = None,
):
    if split_order is None:
        split_order = sorted(projection_df["split"].unique().tolist())

    plt.figure(figsize=figsize)
    for split in split_order:
        sub = projection_df[projection_df["split"] == split]
        if len(sub) == 0:
            continue
        plt.scatter(sub[x_col], sub[y_col], s=s, alpha=alpha, label=split)

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title or "Embedding projection")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_projection_with_thumbnails(
    projection_df: pd.DataFrame,
    x_col: str = "dim_1",
    y_col: str = "dim_2",
    max_images: int = 200,
    thumb_size: tuple[int, int] = (48, 48),
    zoom: float = 0.5,
    random_state: int = 42,
    rotate_270: bool = False,
    figsize: tuple[int, int] = (11, 9),
    title: str | None = None,
):
    if len(projection_df) == 0:
        raise ValueError("projection_df is empty")

    rng = np.random.default_rng(random_state)
    if len(projection_df) > max_images:
        keep_idx = rng.choice(len(projection_df), size=max_images, replace=False)
        df = projection_df.iloc[keep_idx].reset_index(drop=True)
    else:
        df = projection_df.reset_index(drop=True)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(df[x_col], df[y_col], s=8, alpha=0.15)

    for _, row in df.iterrows():
        p = Path(row["path"])
        if not p.exists():
            continue

        try:
            img = PILImage.open(p).convert("RGB")
            if rotate_270:
                img = img.transpose(PILImage.Transpose.ROTATE_270)
            img = img.resize(thumb_size)
            imagebox = OffsetImage(np.asarray(img), zoom=zoom)
            ab = AnnotationBbox(imagebox, (row[x_col], row[y_col]), frameon=False)
            ax.add_artist(ab)
        except Exception:
            continue

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title or "Embedding projection with thumbnails")
    plt.tight_layout()
    plt.show()


def _image_path_to_file_url(image_path: str) -> str:
    p = Path(image_path)
    if not p.exists():
        return ""
    return p.resolve().as_uri()


def _image_to_data_url(
    image_path: str,
    thumb_size: tuple[int, int] = (72, 72),
    rotate_270: bool = False,
) -> str:
    p = Path(image_path)
    if not p.exists():
        return ""

    try:
        img = PILImage.open(p).convert("RGB")
        if rotate_270:
            img = img.transpose(PILImage.Transpose.ROTATE_270)
        img = img.resize(thumb_size)

        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        return ""


def plot_projection_interactive(
    projection_df: pd.DataFrame,
    x_col: str = "dim_1",
    y_col: str = "dim_2",
    split_order: list[str] | None = None,
    title: str | None = None,
    marker_size: int = 6,
    opacity: float = 0.85,
    hover_content: str = "path",  # "path" | "thumbnail"
    rotate_270: bool = False,
    thumb_size: tuple[int, int] = (400, 400),
):
    """
    Interactive Plotly scatter where hovering each point shows image + path.
    """
    try:
        px = importlib.import_module("plotly.express")
    except ImportError as e:
        raise ImportError(
            "Plotly is not installed. Install with: pip install plotly"
        ) from e

    if len(projection_df) == 0:
        raise ValueError("projection_df is empty")

    df = projection_df.copy().reset_index(drop=True)

    if hover_content not in {"path", "thumbnail"}:
        raise ValueError("hover_content must be one of: 'path', 'thumbnail'")

    if hover_content == "thumbnail":
        try:
            bokeh_io = importlib.import_module("bokeh.io")
            bokeh_models = importlib.import_module("bokeh.models")
            bokeh_plotting = importlib.import_module("bokeh.plotting")
            bokeh_palettes = importlib.import_module("bokeh.palettes")
        except ImportError as e:
            raise ImportError(
                "Bokeh is required for hover thumbnails. Install with: pip install bokeh"
            ) from e

        output_notebook = bokeh_io.output_notebook
        show = bokeh_io.show
        ColumnDataSource = bokeh_models.ColumnDataSource
        HoverTool = bokeh_models.HoverTool
        figure = bokeh_plotting.figure
        Category10 = bokeh_palettes.Category10
        Category20 = bokeh_palettes.Category20

        split_values = sorted(df["split"].astype(str).unique().tolist()) if split_order is None else split_order
        n_splits = max(1, len(split_values))

        if n_splits <= 10:
            palette = Category10[10][:n_splits]
        elif n_splits <= 20:
            palette = Category20[20][:n_splits]
        else:
            base_palette = Category20[20]
            palette = [base_palette[i % 20] for i in range(n_splits)]

        color_map = {split: palette[i] for i, split in enumerate(split_values)}
        df["_color"] = df["split"].map(lambda s: color_map.get(str(s), "#1f77b4"))
        thumb_urls = [
            _image_to_data_url(p, thumb_size=thumb_size, rotate_270=rotate_270)
            for p in df["path"].astype(str).tolist()
        ]
        df["_thumb_html"] = [
            (
                f"<img src='{url}' width='{thumb_size[0]}' height='{thumb_size[1]}'/>"
                if url
                else "(image unavailable)"
            )
            for url in thumb_urls
        ]

        source = ColumnDataSource(
            {
                "x": df[x_col].to_numpy(),
                "y": df[y_col].to_numpy(),
                "split": df["split"].astype(str).to_numpy(),
                "path": df["path"].astype(str).to_numpy(),
                "color": df["_color"].to_numpy(),
                "thumb_html": df["_thumb_html"].to_numpy(),
            }
        )

        output_notebook(hide_banner=True)
        p = figure(
            width=950,
            height=700,
            title=title or "Interactive embedding projection",
            tools="pan,wheel_zoom,box_zoom,reset,save",
            active_scroll="wheel_zoom",
        )
        p.scatter(
            x="x",
            y="y",
            source=source,
            color="color",
            size=marker_size,
            alpha=opacity,
            legend_field="split",
        )

        hover = HoverTool(
            tooltips="""
            <div>
                <div><b>split:</b> @split</div>
                <div><b>path:</b> @path</div>
                <div>@thumb_html{safe}</div>
            </div>
            """
        )
        p.add_tools(hover)
        p.legend.location = "top_right"
        p.legend.click_policy = "hide"
        p.xaxis.axis_label = x_col
        p.yaxis.axis_label = y_col

        show(p)
        return p

    if split_order is not None:
        order_map = {name: i for i, name in enumerate(split_order)}
        df["_split_order"] = df["split"].map(lambda s: order_map.get(s, 10**9))
        df = df.sort_values("_split_order").drop(columns=["_split_order"])

    custom_data = ["path", "split"]
    hover_template = (
        "<b>split:</b> %{customdata[1]}<br>"
        "<b>path:</b> %{customdata[0]}"
        "<extra></extra>"
    )

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color="split",
        category_orders={"split": split_order} if split_order is not None else None,
        custom_data=custom_data,
        title=title or "Interactive embedding projection",
    )

    fig.update_traces(
        marker={"size": marker_size, "opacity": opacity},
        hovertemplate=hover_template,
    )
    fig.update_layout(legend_title_text="split")
    fig.show()
    return fig
