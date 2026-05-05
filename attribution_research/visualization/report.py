# -*- coding: utf-8 -*-
"""
Task-specific attribution report figures.

These reports formalize the layout explored in the demo notebooks while keeping
the plotting code reusable for the scripted visualization workflow.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from attribution_research.visualization.vis import (
    build_attribution_map,
    find_early_peak,
    get_word_saliency,
)


BASE_REPORT_PALETTE = {
    "ink": "#0f172a",
    "slate": "#4f5d75",
    "muted": "#718096",
    "grid": "#d7deea",
    "background": "#f7f9fc",
    "panel": "#ffffff",
    "border": "#d9e2ef",
    "diverging": "RdBu_r",
    "accent": "#2a7f62",
}

CLASSIFICATION_REPORT_PALETTE = {
    **BASE_REPORT_PALETTE,
    "insertion": "#e68613",
    "deletion": "#3366aa",
    "heatmap": "magma",
}

CAPTION_REPORT_PALETTE = {
    **BASE_REPORT_PALETTE,
    "insertion": CLASSIFICATION_REPORT_PALETTE["insertion"],
    "deletion": CLASSIFICATION_REPORT_PALETTE["deletion"],
    "heatmap": CLASSIFICATION_REPORT_PALETTE["heatmap"],
}

# Backward-compatible default export.
REPORT_PALETTE = CLASSIFICATION_REPORT_PALETTE

REPORT_FONT_FAMILY = ["Arial", "DejaVu Sans", "sans-serif"]


def configure_report_style(palette: dict[str, str] | None = None) -> dict[str, str]:
    import matplotlib.pyplot as plt

    palette = REPORT_PALETTE if palette is None else palette

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": palette["background"],
            "axes.grid": True,
            "axes.titlesize": 28,
            "axes.titleweight": "normal",
            "axes.labelsize": 18,
            "axes.labelcolor": palette["ink"],
            "axes.edgecolor": palette["slate"],
            "axes.linewidth": 1.15,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "xtick.color": palette["slate"],
            "ytick.color": palette["slate"],
            "grid.color": palette["grid"],
            "grid.linewidth": 1.0,
            "grid.alpha": 0.5,
            "legend.fontsize": 15,
            "font.size": 16,
            "font.family": REPORT_FONT_FAMILY,
            "savefig.facecolor": "white",
        }
    )
    return palette


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _token_label(word: str) -> str:
    text = str(word)
    return text if text.strip() else "<space>"


def _format_score(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _format_count(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    try:
        count = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if count >= 1000:
        return f"{count / 1000.0:.1f}k"
    return str(int(count))


def _format_exact_count(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    try:
        return str(int(round(float(value))))
    except (TypeError, ValueError):
        return "n/a"


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLASSIFICATION_LABEL_CACHE: dict[tuple[str, int], str] | None = None


def _display_label(text: str) -> str:
    return str(text).replace("_", " ").strip().title()


def _classification_label_cache() -> dict[tuple[str, int], str]:
    global _CLASSIFICATION_LABEL_CACHE
    if _CLASSIFICATION_LABEL_CACHE is not None:
        return _CLASSIFICATION_LABEL_CACHE

    cache: dict[tuple[str, int], str] = {}
    for meta_path in (_REPO_ROOT / "datasets" / "imagenet" / "generated").glob("*_meta.json"):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                items = json.load(f)
        except Exception:
            continue
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict) or "image_path" not in item:
                continue
            image_name = Path(str(item["image_path"])).name
            label = item.get("internal_class_name") or item.get("official_class_name")
            if not label:
                continue
            for key in ("pred_label", "gt_label", "target_label"):
                if key not in item:
                    continue
                try:
                    cache[(image_name, int(item[key]))] = _display_label(str(label))
                except (TypeError, ValueError):
                    continue

    _CLASSIFICATION_LABEL_CACHE = cache
    return cache


def _classification_title(info: dict, sample_id: str) -> str:
    for key in ("target_name", "class_name", "internal_class_name", "official_class_name"):
        value = info.get(key)
        if value:
            return _display_label(str(value))

    image_name = Path(str(info.get("image_path", ""))).name
    try:
        target_label = int(info.get("target_label"))
    except (TypeError, ValueError):
        target_label = None
    if image_name and target_label is not None:
        label = _classification_label_cache().get((image_name, target_label))
        if label:
            return label

    if target_label is not None:
        return f"Target Label {target_label}"
    return str(sample_id)


def _trapz_auc(x: np.ndarray, y: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.float32)
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return float(trapezoid(y_arr, x=x_arr))
    if len(x_arr) < 2:
        return 0.0
    return float(np.sum((y_arr[1:] + y_arr[:-1]) * 0.5 * (x_arr[1:] - x_arr[:-1])))


def _report_metrics(
    *,
    ins_x: np.ndarray,
    ins_y: np.ndarray,
    del_x: np.ndarray,
    del_y: np.ndarray,
    info: dict,
) -> list[tuple[str, str]]:
    return [
        ("Steps", str(len(info.get("region_area", [])))),
        (
            "Forwards",
            _format_count(info.get("total_model_forward_calls", info.get("model_forward_calls"))),
        ),
        ("Ins.AUC", _format_score(_trapz_auc(ins_x, ins_y))),
        ("Del.AUC", _format_score(_trapz_auc(del_x, del_y))),
    ]


def _plot_metric_strip(
    ax,
    *,
    metrics: list[tuple[str, str]],
    palette: dict[str, str],
) -> None:
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if not metrics:
        return

    card_width = 0.14
    gap = 0.018
    x0 = 0.5 - ((card_width + gap) * len(metrics) - gap) / 2.0
    for idx, (label, value) in enumerate(metrics):
        x = x0 + idx * (card_width + gap)
        ax.text(
            x + card_width / 2.0,
            0.5,
            f"{label} {value}",
            transform=ax.transAxes,
            fontsize=10.0,
            color=palette["ink"],
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.32,rounding_size=0.12",
                facecolor=palette["panel"],
                edgecolor=palette["border"],
                linewidth=0.9,
            ),
        )


def _positive_delta_attr_map(masks: np.ndarray, info: dict) -> np.ndarray:
    return build_attribution_map(
        masks,
        info["insertion_score"],
        normalize=True,
        score_mode="positive_delta",
        baseline_score=info.get("baseline_score"),
    )


def _early_trough(
    region_area: list[float] | np.ndarray,
    scores: list[float] | np.ndarray,
    *,
    area_limit: float = 0.5,
) -> dict[str, float]:
    areas = np.asarray(region_area, dtype=np.float32)
    score_arr = np.asarray(scores, dtype=np.float32)
    if areas.ndim != 1 or score_arr.ndim != 1:
        raise ValueError("region_area and scores must both be 1-D sequences")
    if len(areas) != len(score_arr):
        raise ValueError("region_area and scores must have the same length")
    if len(areas) == 0:
        raise ValueError("region_area is empty")
    candidates = np.flatnonzero(areas <= float(area_limit))
    if candidates.size == 0:
        candidates = np.array([0], dtype=np.int64)
    trough_idx = int(candidates[np.argmin(score_arr[candidates])])
    return {
        "index": trough_idx,
        "area": float(areas[trough_idx]),
        "score": float(score_arr[trough_idx]),
    }


def _curve_points(info: dict, series_key: str, start_value: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([0.0] + list(info["region_area"]), dtype=np.float32)
    y = np.array([float(start_value)] + list(info[series_key]), dtype=np.float32)
    return x, y


def _dedupe_curve(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    keep_x = [float(x[0])]
    keep_y = [float(y[0])]
    for xi, yi in zip(x[1:], y[1:]):
        if float(xi) <= keep_x[-1]:
            keep_x[-1] = max(keep_x[-1], float(xi))
            keep_y[-1] = float(yi)
        else:
            keep_x.append(float(xi))
            keep_y.append(float(yi))
    return np.asarray(keep_x, dtype=np.float32), np.asarray(keep_y, dtype=np.float32)


def smooth_curve(x: np.ndarray, y: np.ndarray, points: int = 256) -> tuple[np.ndarray, np.ndarray]:
    x, y = _dedupe_curve(np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32))
    if len(x) < 2:
        return x, y

    x_dense = np.linspace(float(x[0]), float(x[-1]), int(max(points, len(x))))
    if len(x) >= 3:
        try:
            from scipy.interpolate import PchipInterpolator

            y_dense = PchipInterpolator(x, y)(x_dense)
            return x_dense, np.asarray(y_dense, dtype=np.float32)
        except Exception:
            pass
    y_dense = np.interp(x_dense, x, y)
    return x_dense.astype(np.float32), y_dense.astype(np.float32)


def _plot_overlay(
    ax,
    image: np.ndarray,
    attr_map: np.ndarray,
    *,
    palette: dict[str, str],
    title: str | None = None,
):
    ax.imshow(bgr_to_rgb(image))
    im = ax.imshow(attr_map, cmap=palette["heatmap"], alpha=0.52, vmin=0.0, vmax=1.0)
    ax.set_anchor("N")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=32, color=palette["ink"], pad=8)
    return im


def _match_caption_column_to_image(fig, ax_overlay, ax_strip, ax_metrics, ax_cbar) -> None:
    fig.canvas.draw()
    image_box = ax_overlay.get_position()
    strip_box = ax_strip.get_position()
    metrics_box = ax_metrics.get_position()
    cbar_box = ax_cbar.get_position()

    cbar_w = min(max(cbar_box.width, 0.018), 0.024)
    cbar_gap = 0.026
    left_x = image_box.x0
    left_w = min(image_box.width * 1.01, 0.98 - left_x - cbar_gap - cbar_w)
    ax_strip.set_position([left_x, strip_box.y0, left_w, strip_box.height])
    ax_metrics.set_position([left_x, metrics_box.y0, left_w, metrics_box.height])

    metrics_box = ax_metrics.get_position()
    ax_cbar.set_position(
        [
            left_x + left_w + cbar_gap,
            metrics_box.y0,
            cbar_w,
            image_box.y1 - metrics_box.y0,
        ]
    )


def _align_classification_columns(fig, ax_overlay, ax_forward, ax_ins, ax_del) -> None:
    fig.canvas.draw()
    image_box = ax_overlay.get_position()
    forward_box = ax_forward.get_position()
    ins_box = ax_ins.get_position()

    count_h = min(forward_box.height, 0.062)
    count_gap = 0.018
    count_y = max(0.035, image_box.y0 - count_gap - count_h)
    ax_forward.set_position([image_box.x0, count_y, image_box.width, count_h])

    total_bottom = count_y
    total_top = image_box.y1
    curve_gap = 0.115
    curve_h = max(0.1, (total_top - total_bottom - curve_gap) / 2.0)
    ax_ins.set_position([ins_box.x0, total_top - curve_h, ins_box.width, curve_h])
    ax_del.set_position([ins_box.x0, total_bottom, ins_box.width, curve_h])


def _plot_curve(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    color: str,
    score_marker: dict | None,
    palette: dict[str, str],
):
    x_smooth, y_smooth = smooth_curve(x, y)
    ax.plot(x_smooth, y_smooth, linewidth=4.0, color=color, solid_capstyle="round")
    ax.fill_between(x_smooth, y_smooth, y2=0.0, color=color, alpha=0.12)
    ax.set_title(title, fontsize=28, color="#000000", pad=8)
    ax.set_xlabel(xlabel, fontsize=20, family=REPORT_FONT_FAMILY, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=20, family=REPORT_FONT_FAMILY, labelpad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.set_xlim(0.0, 1.0)
    ymax = max(float(np.nanmax(y_smooth)), float(np.nanmax(y)), 1e-6)
    ax.set_ylim(min(0.0, float(np.nanmin(y_smooth))), min(1.0, ymax + max(0.04, ymax * 0.08)))
    if score_marker is None:
        return

    marker_area = float(score_marker["area"])
    marker_score = float(score_marker["score"])
    ax.axvline(
        marker_area,
        color=color,
        linestyle=(0, (4, 4)),
        linewidth=1.4,
        alpha=0.75,
        zorder=3,
    )
    ax.scatter(
        [marker_area],
        [marker_score],
        s=54,
        color=color,
        edgecolor="white",
        linewidth=1.4,
        zorder=4,
    )


def _save_figure(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")


def _plot_forward_count_box(ax, forward_count: str, *, palette: dict[str, str]) -> None:
    purple = "#6d28d9"
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(
        0.5,
        0.5,
        f"Model Forward Count: {forward_count}",
        transform=ax.transAxes,
        fontsize=24,
        family=REPORT_FONT_FAMILY,
        color=purple,
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round,pad=0.32,rounding_size=0.08",
            facecolor="#faf5ff",
            edgecolor=purple,
            linewidth=1.4,
        ),
    )


def render_classification_report(
    *,
    image: np.ndarray,
    masks: np.ndarray,
    info: dict,
    sample_id: str,
    method_name: str,
    model_label: str,
    output_path: str | Path,
) -> Path:
    import matplotlib.pyplot as plt

    palette = configure_report_style(CLASSIFICATION_REPORT_PALETTE)
    attr_map = _positive_delta_attr_map(masks, info)
    peak = find_early_peak(info["region_area"], info["insertion_score"], area_limit=0.5)
    ins_x, ins_y = _curve_points(info, "insertion_score", float(info["baseline_score"]))
    del_x, del_y = _curve_points(info, "deletion_score", float(info["org_score"]))
    ins_auc = _trapz_auc(ins_x, ins_y)
    del_auc = _trapz_auc(del_x, del_y)
    title = _classification_title(info, sample_id)
    deletion_trough = _early_trough(info["region_area"], info["deletion_score"], area_limit=0.5)
    forward_count = _format_exact_count(info.get("total_model_forward_calls", info.get("model_forward_calls")))

    fig = plt.figure(figsize=(14.05, 9.98))
    gs = fig.add_gridspec(
        3,
        2,
        width_ratios=[1.35, 1.0],
        height_ratios=[1.0, 1.0, 0.18],
        wspace=0.30,
        hspace=0.20,
    )
    ax_overlay = fig.add_subplot(gs[:2, 0])
    ax_forward = fig.add_subplot(gs[2, 0])
    curve_gs = gs[:, 1].subgridspec(2, 1, hspace=0.55)
    ax_ins = fig.add_subplot(curve_gs[0, 0])
    ax_del = fig.add_subplot(curve_gs[1, 0])

    _plot_overlay(
        ax_overlay,
        image=image,
        attr_map=attr_map,
        palette=palette,
        title=title,
    )
    _plot_curve(
        ax_ins,
        ins_x,
        ins_y,
        title=f"Insertion {_format_score(ins_auc)}",
        xlabel="PCT. of Image Released",
        ylabel="Recognization Score",
        color=palette["insertion"],
        score_marker=peak,
        palette=palette,
    )
    _plot_curve(
        ax_del,
        del_x,
        del_y,
        title=f"Deletion {_format_score(del_auc)}",
        xlabel="PCT. of Image Removed",
        ylabel="Recognization Score",
        color=palette["deletion"],
        score_marker=deletion_trough,
        palette=palette,
    )
    _plot_forward_count_box(ax_forward, forward_count, palette=palette)
    _align_classification_columns(fig, ax_overlay, ax_forward, ax_ins, ax_del)

    output = Path(output_path)
    _save_figure(fig, output)
    plt.close(fig)
    return output


def _plot_token_saliency_strip(
    ax,
    words: list[str],
    scores: np.ndarray,
    *,
    palette: dict[str, str],
    fontsize: float = 16.0,
    y_start: float = 0.92,
    line_height: float = 0.115,
    show_colorbar: bool = False,
) -> None:
    from matplotlib import cm, colors
    import matplotlib.patheffects as pe

    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    pairs = [
        (str(word), float(score))
        for word, score in zip(words, np.asarray(scores, dtype=np.float32))
        if str(word).strip()
    ]
    score_arr = np.asarray([score for _, score in pairs], dtype=np.float32)
    if score_arr.size:
        vmin = float(score_arr.min())
        vmax = float(score_arr.max())
        if abs(vmax - vmin) < 1e-6:
            vmax = vmin + 1e-6
    else:
        vmin, vmax = 0.0, 1.0
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(palette["heatmap"])

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transAxes.inverted()

    x = 0.02
    y = y_start
    gap = 0.014

    for word, score in pairs:
        label = _token_label(word).strip()
        facecolor = cmap(norm(float(score)))
        r, g, b, _ = facecolor
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        text_color = "#ffffff" if luminance < 0.46 else palette["ink"]
        stroke_color = "#0f172a" if luminance < 0.46 else "#ffffff"
        artist = ax.text(
            x,
            y,
            label,
            transform=ax.transAxes,
            fontsize=fontsize,
            fontweight="semibold",
            family=REPORT_FONT_FAMILY,
            ha="left",
            va="center",
            bbox=dict(
                facecolor=facecolor,
                edgecolor=stroke_color,
                linewidth=0.5,
                boxstyle="round,pad=0.28",
            ),
            color=text_color,
        )
        artist.set_path_effects([pe.withStroke(linewidth=1.1, foreground=stroke_color, alpha=0.35)])
        fig.canvas.draw()
        bbox = artist.get_window_extent(renderer=renderer)
        (x0, _), (x1, _) = inv.transform([(bbox.x0, bbox.y0), (bbox.x1, bbox.y1)])
        width = x1 - x0
        if x + width > 0.98:
            artist.set_position((0.015, y - line_height))
            x = 0.015 + width + gap
            y -= line_height
        else:
            x += width + gap

    if show_colorbar:
        from matplotlib.cm import ScalarMappable
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2.3%", pad=0.03)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("token saliency", rotation=270, labelpad=12)


def _plot_caption_metric_row(
    ax,
    *,
    insertion_auc: float,
    deletion_auc: float,
    forwards: str,
    palette: dict[str, str],
) -> None:
    from matplotlib import cm, colors as mpl_colors

    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    cmap = cm.get_cmap(palette["heatmap"])
    metric_parts = [
        (f"Insertion: {_format_score(insertion_auc)}", mpl_colors.to_hex(cmap(0.78))),
        ("|", palette["slate"]),
        (f"Deletion: {_format_score(deletion_auc)}", mpl_colors.to_hex(cmap(0.55))),
        ("|", palette["slate"]),
        (f"Model Forward Count: {forwards}", mpl_colors.to_hex(cmap(0.32))),
    ]
    artists = []
    base_fontsize = 17.0
    for text, color in metric_parts:
        artist = ax.text(
            0.0,
            0.5,
            text,
            transform=ax.transAxes,
            fontsize=base_fontsize,
            family=REPORT_FONT_FAMILY,
            fontstyle="normal",
            color=color,
            ha="left",
            va="center",
        )
        artists.append(artist)

    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    inv = ax.transAxes.inverted()

    def _artist_widths() -> list[float]:
        widths = []
        for artist in artists:
            bbox = artist.get_window_extent(renderer=renderer)
            (x0, _), (x1, _) = inv.transform([(bbox.x0, bbox.y0), (bbox.x1, bbox.y1)])
            widths.append(x1 - x0)
        return widths

    widths = _artist_widths()
    gap = 0.012
    total_w = sum(widths) + gap * (len(widths) - 1)
    if total_w > 0.96:
        fitted_size = max(11.5, base_fontsize * 0.96 / total_w)
        for artist in artists:
            artist.set_fontsize(fitted_size)
        ax.figure.canvas.draw()
        renderer = ax.figure.canvas.get_renderer()
        widths = _artist_widths()
        total_w = sum(widths) + gap * (len(widths) - 1)

    x = max(0.01, (1.0 - total_w) / 2.0)
    for artist, width in zip(artists, widths):
        artist.set_position((x, 0.5))
        x += width + gap

def render_caption_report(
    *,
    image: np.ndarray,
    masks: np.ndarray,
    info: dict,
    sample_id: str,
    method_name: str,
    model_label: str,
    output_path: str | Path,
) -> Path:
    import matplotlib.pyplot as plt

    palette = configure_report_style(CAPTION_REPORT_PALETTE)
    attr_map = _positive_delta_attr_map(masks, info)
    ins_x, ins_y = _curve_points(info, "insertion_score", float(info["baseline_score"]))
    del_x, del_y = _curve_points(info, "deletion_score", float(info["org_score"]))
    words = list(info.get("words", []))
    word_saliency = np.asarray(get_word_saliency(info, mode="auc_delta"), dtype=np.float32)
    ins_auc = _trapz_auc(ins_x, ins_y)
    del_auc = _trapz_auc(del_x, del_y)
    forwards = _format_exact_count(info.get("total_model_forward_calls", info.get("model_forward_calls")))

    fig = plt.figure(figsize=(14.05, 8.55))
    gs = fig.add_gridspec(
        3,
        2,
        width_ratios=[1.0, 0.035],
        height_ratios=[1.0, 0.28, 0.11],
        wspace=0.05,
        hspace=0.05,
    )
    ax_overlay = fig.add_subplot(gs[0, 0])
    ax_strip = fig.add_subplot(gs[1, 0])
    ax_metrics = fig.add_subplot(gs[2, 0])
    ax_cbar = fig.add_subplot(gs[:, 1])

    im = _plot_overlay(
        ax_overlay,
        image=image,
        attr_map=attr_map,
        palette=palette,
    )
    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.ax.tick_params(labelsize=12, colors=palette["slate"])
    _match_caption_column_to_image(fig, ax_overlay, ax_strip, ax_metrics, ax_cbar)
    _plot_token_saliency_strip(
        ax_strip,
        words,
        word_saliency,
        palette=palette,
        fontsize=15.0,
        y_start=0.82,
        line_height=0.32,
        show_colorbar=False,
    )
    _plot_caption_metric_row(
        ax_metrics,
        insertion_auc=ins_auc,
        deletion_auc=del_auc,
        forwards=forwards,
        palette=palette,
    )

    output = Path(output_path)
    _save_figure(fig, output)
    plt.close(fig)
    return output


__all__ = [
    "BASE_REPORT_PALETTE",
    "CAPTION_REPORT_PALETTE",
    "CLASSIFICATION_REPORT_PALETTE",
    "REPORT_FONT_FAMILY",
    "REPORT_PALETTE",
    "bgr_to_rgb",
    "configure_report_style",
    "render_caption_report",
    "render_classification_report",
    "smooth_curve",
]
