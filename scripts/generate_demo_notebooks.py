#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "notebooks"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(text).lstrip("\n").splitlines(keepends=True),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).lstrip("\n").splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


COMMON_HELPERS = """
from pathlib import Path
from types import SimpleNamespace
import json
import shlex
import subprocess
import sys

REPO_HINT = Path("__REPO_HINT__")


def is_repo_root(path: Path) -> bool:
    return (path / "tasks").exists() and (path / "attribution_research").exists()


def resolve_repo_root() -> Path:
    cwd = Path.cwd().resolve()
    for path in [cwd, *cwd.parents]:
        if is_repo_root(path):
            return path
    if is_repo_root(REPO_HINT):
        return REPO_HINT
    raise FileNotFoundError(
        "Could not locate repo root containing both 'tasks' and "
        "'attribution_research'."
    )


repo = resolve_repo_root()
if str(repo) not in sys.path:
    sys.path.insert(0, str(repo))

import cv2
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
from IPython.display import display

from attribution_research.evaluation import compute_auc_from_json
from attribution_research.io.results import load_result
from attribution_research.runtime import build_run_tag
from attribution_research.visualization import (
    build_attribution_map,
    build_word_region_matrix,
    draw_bbox,
    find_early_peak,
    get_word_saliency,
)

SCIENTIFIC_PALETTE = {
    "ink": "#1f2a44",
    "slate": "#4f5d75",
    "grid": "#d7deea",
    "background": "#f7f9fc",
    "insertion": "#c65d4b",
    "deletion": "#355c7d",
    "iou_insertion": "#2a7f62",
    "iou_deletion": "#6c5b7b",
    "cls_insertion": "#c38b2f",
    "cls_deletion": "#8c5a3c",
    "heatmap": "cividis",
    "diverging": "RdBu_r",
}

plt.rcParams.update(
    {
        "figure.figsize": (8, 5),
        "figure.facecolor": "white",
        "axes.facecolor": SCIENTIFIC_PALETTE["background"],
        "axes.grid": True,
        "axes.titlesize": 18,
        "axes.titleweight": "semibold",
        "axes.labelsize": 14,
        "axes.labelcolor": SCIENTIFIC_PALETTE["ink"],
        "axes.edgecolor": SCIENTIFIC_PALETTE["slate"],
        "axes.linewidth": 0.9,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "xtick.color": SCIENTIFIC_PALETTE["slate"],
        "ytick.color": SCIENTIFIC_PALETTE["slate"],
        "grid.color": SCIENTIFIC_PALETTE["grid"],
        "grid.linewidth": 0.8,
        "grid.alpha": 0.55,
        "legend.fontsize": 12,
        "font.size": 13,
        "font.family": "DejaVu Sans",
        "savefig.facecolor": "white",
    }
)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def token_label(word: str) -> str:
    return word if str(word).strip() else "<space>"


def build_single_sample_attr_map(masks: np.ndarray, info: dict) -> np.ndarray:
    return build_attribution_map(
        masks,
        info["insertion_score"],
        normalize=True,
        score_mode="positive_delta",
        baseline_score=info.get("baseline_score"),
    )


def cumulative_mask(masks: np.ndarray, index: int) -> np.ndarray:
    if len(masks) == 0:
        raise ValueError("ordered mask set is empty")
    idx = int(np.clip(index, 0, len(masks) - 1))
    return np.clip(masks[: idx + 1].sum(axis=0), 0, 1).astype(np.uint8)


def reveal_image_at_index(
    image: np.ndarray,
    masks: np.ndarray,
    index: int,
):
    if image.shape[:2] != tuple(masks.shape[1:3]):
        image = cv2.resize(image, (masks.shape[2], masks.shape[1]))
    cumulative = cumulative_mask(masks, index)
    visible = image.copy()
    visible[cumulative[:, :, 0] == 0] = 0
    return visible, int(np.clip(index, 0, len(masks) - 1))


def reveal_image(
    image: np.ndarray,
    masks: np.ndarray,
    region_area: list[float],
    fraction: float,
):
    areas = np.asarray(region_area, dtype=np.float32)
    idx = int(np.argmin(np.abs(areas - float(fraction))))
    visible, idx = reveal_image_at_index(image, masks, idx)
    return visible, idx, float(areas[idx])


def plot_progressive_reveal(
    image: np.ndarray,
    masks: np.ndarray,
    region_area: list[float],
    fractions=(0.1, 0.25, 0.5, 0.75, 1.0),
    annotate=None,
):
    fig, axes = plt.subplots(1, len(fractions), figsize=(4 * len(fractions), 4))
    if len(fractions) == 1:
        axes = [axes]
    for ax, frac in zip(axes, fractions):
        visible, idx, actual = reveal_image(image, masks, region_area, frac)
        if annotate is not None:
            visible = annotate(visible, idx)
        ax.imshow(bgr_to_rgb(visible))
        ax.set_title(f"target={frac:.0%}\\nactual={actual:.1%}")
        ax.axis("off")
    plt.tight_layout()


def metric_trace(
    info: dict,
    series_key: str,
    start_value: float,
):
    x = np.array([0.0] + list(info["region_area"]), dtype=np.float32)
    y = np.array([float(start_value)] + list(info[series_key]), dtype=np.float32)
    return x, y


def mirrored_metric_trace(
    info: dict,
    series_key: str,
    mirrored_start: float,
):
    x = np.array([0.0] + list(info["region_area"]), dtype=np.float32)
    y = np.array([float(mirrored_start)] + list(info[series_key]), dtype=np.float32)
    return x, y


def plot_metric_curve(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    ylabel: str,
    xlabel: str,
    color: str,
    peak: dict | None = None,
    annotate_peak: bool = False,
):
    ax.plot(x, y, linewidth=2.8, color=color, solid_capstyle="round")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    if peak is not None:
        ax.axvline(
            peak["area"],
            linestyle="--",
            linewidth=1,
            color=SCIENTIFIC_PALETTE["slate"],
            alpha=0.8,
        )
        if annotate_peak:
            ax.annotate(
                f"{peak['score']:.3f}\\n@ {peak['area']:.1%}",
                xy=(peak["area"], peak["score"]),
                xytext=(8, 10),
                textcoords="offset points",
                fontsize=10,
                color=SCIENTIFIC_PALETTE["ink"],
                arrowprops=dict(arrowstyle="-", color=SCIENTIFIC_PALETTE["slate"], lw=0.9),
            )


def plot_overlay_with_colorbar(
    ax,
    image: np.ndarray,
    attr_map: np.ndarray,
    title: str,
    cmap: str = SCIENTIFIC_PALETTE["heatmap"],
    alpha: float = 0.55,
):
    ax.imshow(bgr_to_rgb(image))
    im = ax.imshow(attr_map, cmap=cmap, alpha=alpha, vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.axis("off")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=0.08)
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.set_label("normalized attribution", rotation=270, labelpad=14)
    return im


def metric_panel(ax, title: str, lines: list[str]):
    ax.axis("off")
    ax.set_title(title, pad=12)
    ax.text(
        0.0,
        1.0,
        "\\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=13,
        family="monospace",
        color=SCIENTIFIC_PALETTE["ink"],
    )


def word_score_table(words: list[str], scores: np.ndarray, top_k: int = 8) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "word": [token_label(word) for word in words],
            "score": np.asarray(scores, dtype=np.float32),
        }
    ).sort_values("score", ascending=False)
    return df.head(top_k).reset_index(drop=True)


def plot_token_saliency_strip(
    ax,
    words: list[str],
    scores: np.ndarray,
    cmap: str = SCIENTIFIC_PALETTE["diverging"],
    fontsize: int = 12,
):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    score_arr = np.asarray(scores, dtype=np.float32)
    vmax = float(np.max(np.abs(score_arr))) if score_arr.size else 1.0
    vmax = max(vmax, 1e-6)
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    colormap = cm.get_cmap(cmap)

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transAxes.inverted()

    x = 0.02
    y = 0.82
    gap = 0.02
    line_height = 0.28

    for word, score in zip(words, score_arr):
        text = token_label(word)
        artist = ax.text(
            x,
            y,
            text,
            transform=ax.transAxes,
            fontsize=fontsize + 1,
            ha="left",
            va="center",
            bbox=dict(
                facecolor=colormap(norm(float(score))),
                edgecolor="none",
                boxstyle="round,pad=0.25",
            ),
        )
        fig.canvas.draw()
        bbox = artist.get_window_extent(renderer=renderer)
        (x0, _), (x1, _) = inv.transform([(bbox.x0, bbox.y0), (bbox.x1, bbox.y1)])
        width = x1 - x0
        if x + width > 0.98:
            artist.set_position((0.02, y - line_height))
            x = 0.02 + width + gap
            y -= line_height
        else:
            x += width + gap

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.8%", pad=0.03)
    sm = ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("word saliency", rotation=270, labelpad=12)
    return norm


def plot_word_region_heatmap(
    ax,
    matrix: np.ndarray,
    words: list[str],
    region_area: list[float],
    peak: dict | None = None,
    cmap: str = SCIENTIFIC_PALETTE["diverging"],
):
    if matrix.size == 0:
        ax.axis("off")
        ax.set_title("word-region correspondence")
        return None

    vmax = float(np.max(np.abs(matrix)))
    vmax = max(vmax, 1e-6)
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title("word-region correspondence")
    ax.set_ylabel("word")
    ax.set_xlabel("released area after region step")
    ax.set_yticks(np.arange(len(words)))
    ax.set_yticklabels([token_label(word) for word in words])
    if matrix.shape[1] > 1:
        tick_idx = np.unique(
            np.linspace(0, matrix.shape[1] - 1, num=min(6, matrix.shape[1]), dtype=int)
        )
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([f"{region_area[idx]:.0%}" for idx in tick_idx])
    if peak is not None:
        ax.axvline(
            peak["index"],
            linestyle="--",
            linewidth=1.2,
            color=SCIENTIFIC_PALETTE["ink"],
            alpha=0.9,
        )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.8%", pad=0.05)
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.set_label("region contribution to word", rotation=270, labelpad=14)
    return im


def compact_table(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    display(df)
    return df
"""

COMMON_HELPERS = COMMON_HELPERS.replace("__REPO_HINT__", REPO_ROOT.as_posix())


def classification_notebook() -> dict:
    cells = [
        md(
            """
            # ImageNet CLIP Single-Sample Notebook

            This notebook covers configuration, one-sample execution, attribution heatmap,
            insertion/deletion curves, progressive reveal, and numerical summary.

            The default configuration matches the smoke-tested sample already stored in this repo.
            """
        ),
        code(COMMON_HELPERS),
        md("## 1. Configuration"),
        code(
            """
            repo = resolve_repo_root()
            task_script = repo / "tasks/classification/clip_imagenet.py"
            datasets = repo / "datasets/imagenet/ILSVRC2012_img_val"
            eval_list = repo / "datasets/imagenet/generated/clip_vitl14_true.txt"
            semantic_features = repo / "ckpt/semantic_features/clip_vitl_imagenet_zeroweights.pt"
            save_root = repo / "classification_results/imagenet-clip-vitl"

            BEGIN = 0
            END = 1
            DEVICE = "cuda"
            ALGORITHM = "gradient"
            SEGMENTER = "patch"
            PATCH_SIZE = 16
            SUPERPIXEL_ALGORITHM = "slico"
            DIVISION_NUMBER = 50
            LAMBDA1 = 1.0
            LAMBDA2 = 1.0
            WINDOW_SIZE = 16
            DRISE_N_MASKS = 1000
            DHSIC_BATCH_SIZE = 32
            RUN_EXPLANATION = False

            run_args = SimpleNamespace(
                algorithm=ALGORITHM,
                segmenter=SEGMENTER,
                superpixel_algorithm=SUPERPIXEL_ALGORITHM,
                division_number=DIVISION_NUMBER,
                patch_size=PATCH_SIZE,
                grid_rows=None,
                grid_cols=None,
                lambda1=LAMBDA1,
                lambda2=LAMBDA2,
                window_size=WINDOW_SIZE,
                drise_n_masks=DRISE_N_MASKS,
                dhsic_batch_size=DHSIC_BATCH_SIZE,
            )
            run_tag = build_run_tag(run_args)
            result_dir = save_root / run_tag

            required = [task_script, datasets, eval_list, semantic_features]
            for path in required:
                print(("OK     " if path.exists() else "MISSING"), path)
            print("RUN TAG ", run_tag)
            print("RESULT  ", result_dir)
            """
        ),
        md("## 2. Load Sample"),
        code(
            """
            lines = [line.strip() for line in eval_list.read_text().splitlines() if line.strip()]
            sample_rel, sample_label = lines[BEGIN].split()
            sample_path = datasets / sample_rel
            sample_image_id = Path(sample_rel).stem

            image = cv2.imread(str(sample_path))
            print("image_id:", sample_image_id)
            print("label   :", sample_label)
            print("path    :", sample_path)

            plt.figure(figsize=(5, 5))
            plt.imshow(bgr_to_rgb(image))
            plt.title(f"{sample_image_id} | label={sample_label}")
            plt.axis("off")
            plt.show()
            """
        ),
        md("## 3. Run Explanation"),
        code(
            """
            cmd = [
                sys.executable,
                str(task_script),
                "--begin",
                str(BEGIN),
                "--end",
                str(END),
                "--algorithm",
                ALGORITHM,
                "--segmenter",
                SEGMENTER,
                "--save-dir",
                str(save_root),
                "--device",
                DEVICE,
                "--lambda1",
                str(LAMBDA1),
                "--lambda2",
                str(LAMBDA2),
            ]

            if SEGMENTER == "patch":
                cmd.extend(["--patch-size", str(PATCH_SIZE)])
            else:
                cmd.extend(
                    [
                        "--superpixel-algorithm",
                        SUPERPIXEL_ALGORITHM,
                        "--division-number",
                        str(DIVISION_NUMBER),
                    ]
                )

            if ALGORITHM == "phasewin":
                cmd.extend(["--window-size", str(WINDOW_SIZE)])
            if ALGORITHM == "drise":
                cmd.extend(["--drise-n-masks", str(DRISE_N_MASKS)])
            if ALGORITHM == "dhsic":
                cmd.extend(["--dhsic-batch-size", str(DHSIC_BATCH_SIZE)])

            print(" ".join(shlex.quote(part) for part in cmd))
            if RUN_EXPLANATION:
                subprocess.run(cmd, cwd=repo, check=True)
            else:
                print("RUN_EXPLANATION=False, reusing the stored sample result if present.")
            """
        ),
        md("## 4. Load Result and Numerical Summary"),
        code(
            """
            if not result_dir.exists():
                raise FileNotFoundError(f"Result directory does not exist: {result_dir}")

            masks, info = load_result(str(result_dir), sample_image_id)
            auc = compute_auc_from_json(info)
            peak = find_early_peak(info["region_area"], info["insertion_score"], area_limit=0.3)
            attr_map = build_single_sample_attr_map(masks, info)
            summary = compact_table(
                [
                    {
                        "algorithm": info["algorithm"],
                        "family": info["algorithm_family"],
                        "segmenter": info["segmenter"],
                        "regions": info["sub-region_number"],
                        "org_score": info["org_score"],
                        "baseline_score": info["baseline_score"],
                        "insertion_auc": auc["insertion_auc"],
                        "deletion_auc": auc["deletion_auc"],
                        "highest_score": auc["highest_score"],
                        "early_peak_area": peak["area"],
                        "early_peak_score": peak["score"],
                    }
                ]
            )
            """
        ),
        md("## 5. Attribution Overlay, Early Peak, and Curves"),
        code(
            """
            early_visible, _ = reveal_image_at_index(image, masks, peak["index"])

            fig = plt.figure(figsize=(16, 9))
            gs = fig.add_gridspec(
                2,
                3,
                width_ratios=[1.25, 1.0, 1.0],
                height_ratios=[1.0, 1.0],
                wspace=0.35,
                hspace=0.35,
            )

            ax_overlay = fig.add_subplot(gs[:, 0])
            ax_early = fig.add_subplot(gs[0, 1])
            ax_metrics = fig.add_subplot(gs[0, 2])
            ax_ins = fig.add_subplot(gs[1, 1])
            ax_del = fig.add_subplot(gs[1, 2])

            plot_overlay_with_colorbar(
                ax_overlay,
                image=image,
                attr_map=attr_map,
                title="single-sample attribution overlay",
            )

            ax_early.imshow(bgr_to_rgb(early_visible))
            ax_early.set_title(
                f"early peak release\\narea={peak['area']:.1%}, score={peak['score']:.3f}"
            )
            ax_early.axis("off")

            metric_panel(
                ax_metrics,
                "decision summary",
                [
                    f"target label         : {sample_label}",
                    f"early peak step      : {peak['index'] + 1}",
                    f"released area        : {peak['area']:.1%}",
                    f"target score @ peak  : {peak['score']:.4f}",
                    f"baseline score       : {info['baseline_score']:.4f}",
                    f"original score       : {info['org_score']:.4f}",
                    f"final insertion score: {info['insertion_score'][-1]:.4f}",
                ],
            )

            ins_x, ins_y = metric_trace(info, "insertion_score", info["baseline_score"])
            del_x, del_y = metric_trace(info, "deletion_score", info["org_score"])
            plot_metric_curve(
                ax_ins,
                ins_x,
                ins_y,
                title="insertion curve",
                ylabel="target score",
                xlabel="revealed area",
                color=SCIENTIFIC_PALETTE["insertion"],
                peak=peak,
                annotate_peak=True,
            )
            plot_metric_curve(
                ax_del,
                del_x,
                del_y,
                title="deletion curve",
                ylabel="target score",
                xlabel="removed area",
                color=SCIENTIFIC_PALETTE["deletion"],
                peak=peak,
                annotate_peak=False,
            )

            plt.show()
            """
        ),
        md("## 6. Progressive Reveal"),
        code(
            """
            plot_progressive_reveal(
                image=image,
                masks=masks,
                region_area=info["region_area"],
                fractions=(0.1, 0.25, 0.5, 0.75, 1.0),
            )
            """
        ),
    ]
    return notebook(cells)


def detection_notebook() -> dict:
    cells = [
        md(
            """
            # COCO GroundingDINO Single-Sample Notebook

            This notebook follows the spirit of
            `../phasewin-search/tutorial/Grounding_DINO_explanation.ipynb`
            and `../phasewin-search/baseline/GroundingDINO-DRISE.ipynb`:
            configuration, one-sample run, attribution overlay, task-specific curves,
            progressive reveal, and detection-oriented summary.

            The default configuration matches the smoke-tested sample already stored in this repo.
            """
        ),
        code(COMMON_HELPERS),
        md("## 1. Configuration"),
        code(
            """
            repo = resolve_repo_root()
            task_script = repo / "tasks/detection/groundingdino_coco.py"
            datasets = repo / "datasets/coco/val2017"
            eval_list = repo / "datasets/coco_groundingdino_correct_detection.json"
            config_path = repo / "config/GroundingDINO_SwinT_OGC.py"
            weights_path = repo / "ckpt/groundingdino_swint_ogc.pth"
            save_root = repo / "detection_results/coco-groundingdino"

            BEGIN = 0
            END = 1
            DEVICE = "cuda"
            ALGORITHM = "phasewin"
            SEGMENTER = "superpixel"
            PATCH_SIZE = None
            SUPERPIXEL_ALGORITHM = "slico"
            DIVISION_NUMBER = 100
            LAMBDA1 = 1.0
            LAMBDA2 = 1.0
            WINDOW_SIZE = 32
            DRISE_N_MASKS = 1000
            RUN_EXPLANATION = False

            run_args = SimpleNamespace(
                algorithm=ALGORITHM,
                segmenter=SEGMENTER,
                superpixel_algorithm=SUPERPIXEL_ALGORITHM,
                division_number=DIVISION_NUMBER,
                patch_size=PATCH_SIZE,
                grid_rows=None,
                grid_cols=None,
                lambda1=LAMBDA1,
                lambda2=LAMBDA2,
                window_size=WINDOW_SIZE,
                drise_n_masks=DRISE_N_MASKS,
                dhsic_batch_size=32,
                lambda_keep=1.0,
                lambda_drop=1.0,
                n_steps=300,
            )
            run_tag = build_run_tag(run_args)
            result_dir = save_root / run_tag

            required = [task_script, datasets, eval_list, config_path, weights_path]
            for path in required:
                print(("OK     " if path.exists() else "MISSING"), path)
            print("RUN TAG ", run_tag)
            print("RESULT  ", result_dir)
            """
        ),
        md("## 2. Load Sample and Detection Target"),
        code(
            """
            items = json.loads(eval_list.read_text())
            if isinstance(items, dict):
                items = items.get("annotations", list(items.values()))
            item = items[BEGIN]
            image_path = datasets / item["image_path"]
            sample_image_id = str(item.get("image_id", Path(item["image_path"]).stem))
            image = cv2.imread(str(image_path))

            gt_box = item["bbox"]
            category = item.get("category", item["class_id"])
            preview = draw_bbox(image, gt_box, color=(0, 255, 0), label=f"GT: {category}")
            print("image_id :", sample_image_id)
            print("category :", category)
            print("caption  :", item.get("caption", "")[:180] + "...")

            plt.figure(figsize=(7, 5))
            plt.imshow(bgr_to_rgb(preview))
            plt.title(sample_image_id)
            plt.axis("off")
            plt.show()
            """
        ),
        md("## 3. Run Explanation"),
        code(
            """
            cmd = [
                sys.executable,
                str(task_script),
                "--begin",
                str(BEGIN),
                "--end",
                str(END),
                "--algorithm",
                ALGORITHM,
                "--segmenter",
                SEGMENTER,
                "--save-dir",
                str(save_root),
                "--device",
                DEVICE,
                "--lambda1",
                str(LAMBDA1),
                "--lambda2",
                str(LAMBDA2),
            ]

            if SEGMENTER == "patch":
                cmd.extend(["--patch-size", str(PATCH_SIZE)])
            else:
                cmd.extend(
                    [
                        "--superpixel-algorithm",
                        SUPERPIXEL_ALGORITHM,
                        "--division-number",
                        str(DIVISION_NUMBER),
                    ]
                )

            if ALGORITHM == "phasewin":
                cmd.extend(["--window-size", str(WINDOW_SIZE)])
            if ALGORITHM == "drise":
                cmd.extend(["--drise-n-masks", str(DRISE_N_MASKS)])

            print(" ".join(shlex.quote(part) for part in cmd))
            if RUN_EXPLANATION:
                subprocess.run(cmd, cwd=repo, check=True)
            else:
                print("RUN_EXPLANATION=False, reusing the stored sample result if present.")
            """
        ),
        md("## 4. Load Result and Detection Summary"),
        code(
            """
            if not result_dir.exists():
                raise FileNotFoundError(f"Result directory does not exist: {result_dir}")

            masks, info = load_result(str(result_dir), sample_image_id)
            auc = compute_auc_from_json(info)
            peak = find_early_peak(info["region_area"], info["insertion_score"], area_limit=0.3)
            attr_map = build_single_sample_attr_map(masks, info)
            pred_box = info["insertion_box"][-1] if "insertion_box" in info else item.get("predict_box")
            peak_box = info["insertion_box"][peak["index"]] if "insertion_box" in info else None
            compact_table(
                [
                    {
                        "algorithm": info["algorithm"],
                        "family": info["algorithm_family"],
                        "segmenter": info["segmenter"],
                        "regions": info["sub-region_number"],
                        "org_score": info["org_score"],
                        "baseline_score": info["baseline_score"],
                        "insertion_auc": auc["insertion_auc"],
                        "deletion_auc": auc["deletion_auc"],
                        "insertion_iou_auc": auc.get("insertion_iou_auc"),
                        "insertion_cls_auc": auc.get("insertion_cls_auc"),
                        "highest_score": auc["highest_score"],
                        "early_peak_area": peak["area"],
                        "early_peak_score": peak["score"],
                        "early_peak_iou": info["insertion_iou"][peak["index"]],
                        "early_peak_cls": info["insertion_cls"][peak["index"]],
                        "final_insertion_iou": info["insertion_iou"][-1],
                        "final_insertion_cls": info["insertion_cls"][-1],
                    }
                ]
            )
            """
        ),
        md("## 5. Attribution Overlay, Early Peak, and Detection Curves"),
        code(
            """
            gt_panel = draw_bbox(image, gt_box, color=(0, 255, 0), label="GT")
            early_visible, _ = reveal_image_at_index(image, masks, peak["index"])
            early_panel = draw_bbox(early_visible, gt_box, color=(0, 255, 0), label="GT")
            if peak_box is not None:
                early_panel = draw_bbox(early_panel, peak_box, color=(255, 0, 0), label="peak")
            final_panel = image.copy()
            final_panel = draw_bbox(final_panel, gt_box, color=(0, 255, 0), label="GT")
            if pred_box is not None:
                final_panel = draw_bbox(final_panel, pred_box, color=(255, 0, 0), label="pred")

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            plot_overlay_with_colorbar(
                axes[0],
                image=gt_panel,
                attr_map=attr_map,
                title="single-sample attribution overlay + GT",
            )
            axes[1].imshow(bgr_to_rgb(early_panel))
            axes[1].set_title(
                "early peak detection\\n"
                f"area={peak['area']:.1%}, score={peak['score']:.3f}, "
                f"IoU={info['insertion_iou'][peak['index']]:.3f}, "
                f"cls={info['insertion_cls'][peak['index']]:.3f}"
            )
            axes[2].imshow(bgr_to_rgb(final_panel))
            axes[2].set_title(
                "final insertion result\\n"
                f"IoU={info['insertion_iou'][-1]:.3f}, cls={info['insertion_cls'][-1]:.3f}"
            )
            for ax in axes[1:]:
                ax.axis("off")
            plt.tight_layout()

            fig, axes = plt.subplots(3, 2, figsize=(13, 12))
            score_ins_x, score_ins_y = metric_trace(info, "insertion_score", info["baseline_score"])
            score_del_x, score_del_y = metric_trace(info, "deletion_score", info["org_score"])
            iou_ins_x, iou_ins_y = mirrored_metric_trace(info, "insertion_iou", info["deletion_iou"][-1])
            iou_del_x, iou_del_y = mirrored_metric_trace(info, "deletion_iou", info["insertion_iou"][-1])
            cls_ins_x, cls_ins_y = mirrored_metric_trace(info, "insertion_cls", info["deletion_cls"][-1])
            cls_del_x, cls_del_y = mirrored_metric_trace(info, "deletion_cls", info["insertion_cls"][-1])

            plot_metric_curve(
                axes[0, 0],
                score_ins_x,
                score_ins_y,
                title="insertion score",
                ylabel="combined score",
                xlabel="revealed area",
                color=SCIENTIFIC_PALETTE["insertion"],
                peak=peak,
                annotate_peak=True,
            )
            plot_metric_curve(
                axes[0, 1],
                score_del_x,
                score_del_y,
                title="deletion score",
                ylabel="combined score",
                xlabel="removed area",
                color=SCIENTIFIC_PALETTE["deletion"],
                peak=peak,
                annotate_peak=False,
            )
            plot_metric_curve(
                axes[1, 0],
                iou_ins_x,
                iou_ins_y,
                title="insertion IoU",
                ylabel="IoU",
                xlabel="revealed area",
                color=SCIENTIFIC_PALETTE["iou_insertion"],
                peak=peak,
                annotate_peak=False,
            )
            plot_metric_curve(
                axes[1, 1],
                iou_del_x,
                iou_del_y,
                title="deletion IoU",
                ylabel="IoU",
                xlabel="removed area",
                color=SCIENTIFIC_PALETTE["iou_deletion"],
                peak=peak,
                annotate_peak=False,
            )
            plot_metric_curve(
                axes[2, 0],
                cls_ins_x,
                cls_ins_y,
                title="insertion cls score",
                ylabel="cls score",
                xlabel="revealed area",
                color=SCIENTIFIC_PALETTE["cls_insertion"],
                peak=peak,
                annotate_peak=False,
            )
            plot_metric_curve(
                axes[2, 1],
                cls_del_x,
                cls_del_y,
                title="deletion cls score",
                ylabel="cls score",
                xlabel="removed area",
                color=SCIENTIFIC_PALETTE["cls_deletion"],
                peak=peak,
                annotate_peak=False,
            )
            plt.tight_layout()
            """
        ),
        md("## 6. Progressive Reveal"),
        code(
            """
            plot_progressive_reveal(
                image=image,
                masks=masks,
                region_area=info["region_area"],
                fractions=(0.1, 0.25, 0.5, 0.75, 1.0),
                annotate=lambda frame, idx: draw_bbox(frame, gt_box, color=(0, 255, 0), label="GT"),
            )
            """
        ),
    ]
    return notebook(cells)


def caption_notebook() -> dict:
    cells = [
        md(
            """
            # Qwen2.5-VL Caption Single-Sample Notebook

            This notebook follows the repository's standard caption-analysis flow:
            configuration, single-sample run, token display, attribution heatmap,
            insertion/deletion curves, word-level summary, and progressive reveal.

            The default configuration matches the smoke-tested sample already stored in this repo.
            """
        ),
        code(COMMON_HELPERS),
        md("## 1. Configuration"),
        code(
            """
            repo = resolve_repo_root()
            task_script = repo / "tasks/caption_vqa/qwen25vl_coco_caption.py"
            datasets = repo / "datasets/coco/val2017"
            eval_list = repo / "datasets/Qwen2.5-VL-3B-coco-caption.json"
            model_name = repo / "model_checkpoint/Qwen2.5-VL-3B-Instruct"
            save_root = repo / "caption_results/Qwen2.5-VL-3B-coco-caption"

            BEGIN = 0
            END = 1
            DEVICE = "cuda"
            ALGORITHM = "greedy"
            SEGMENTER = "superpixel"
            PATCH_SIZE = None
            SUPERPIXEL_ALGORITHM = "slico"
            DIVISION_NUMBER = 64
            LAMBDA1 = 1.0
            LAMBDA2 = 1.0
            WINDOW_SIZE = 16
            DRISE_N_MASKS = 1000
            RUN_EXPLANATION = False

            run_args = SimpleNamespace(
                algorithm=ALGORITHM,
                segmenter=SEGMENTER,
                superpixel_algorithm=SUPERPIXEL_ALGORITHM,
                division_number=DIVISION_NUMBER,
                patch_size=PATCH_SIZE,
                grid_rows=None,
                grid_cols=None,
                lambda1=LAMBDA1,
                lambda2=LAMBDA2,
                window_size=WINDOW_SIZE,
                drise_n_masks=DRISE_N_MASKS,
                dhsic_batch_size=32,
                lambda_keep=1.0,
                lambda_drop=1.0,
                n_steps=300,
            )
            run_tag = build_run_tag(run_args)
            result_dir = save_root / run_tag

            required = [task_script, datasets, eval_list, model_name]
            for path in required:
                print(("OK     " if path.exists() else "MISSING"), path)
            print("RUN TAG ", run_tag)
            print("RESULT  ", result_dir)
            """
        ),
        md("## 2. Load Sample and Target Tokens"),
        code(
            """
            items = json.loads(eval_list.read_text())
            item = items[BEGIN]
            image_path = datasets / item["image_path"]
            sample_image_id = Path(item["image_path"]).stem
            image = cv2.imread(str(image_path))

            token_df = pd.DataFrame(
                {
                    "position": list(range(len(item["words"]))),
                    "word": [word if word.strip() else "<space>" for word in item["words"]],
                    "token_id": item["selected_interpretation_token_word_id"],
                }
            )
            print("image_id:", sample_image_id)
            display(token_df.head(15))

            plt.figure(figsize=(7, 5))
            plt.imshow(bgr_to_rgb(image))
            plt.title(sample_image_id)
            plt.axis("off")
            plt.show()
            """
        ),
        md("## 3. Run Explanation"),
        code(
            """
            cmd = [
                sys.executable,
                str(task_script),
                "--begin",
                str(BEGIN),
                "--end",
                str(END),
                "--algorithm",
                ALGORITHM,
                "--segmenter",
                SEGMENTER,
                "--save-dir",
                str(save_root),
                "--device",
                DEVICE,
                "--model-name",
                str(model_name),
                "--lambda1",
                str(LAMBDA1),
                "--lambda2",
                str(LAMBDA2),
            ]

            if SEGMENTER == "patch":
                cmd.extend(["--patch-size", str(PATCH_SIZE)])
            else:
                cmd.extend(
                    [
                        "--superpixel-algorithm",
                        SUPERPIXEL_ALGORITHM,
                        "--division-number",
                        str(DIVISION_NUMBER),
                    ]
                )

            if ALGORITHM == "phasewin":
                cmd.extend(["--window-size", str(WINDOW_SIZE)])
            if ALGORITHM == "drise":
                cmd.extend(["--drise-n-masks", str(DRISE_N_MASKS)])

            print(" ".join(shlex.quote(part) for part in cmd))
            if RUN_EXPLANATION:
                subprocess.run(cmd, cwd=repo, check=True)
            else:
                print("RUN_EXPLANATION=False, reusing the stored sample result if present.")
            """
        ),
        md("## 4. Load Result and Numerical Summary"),
        code(
            """
            if not result_dir.exists():
                raise FileNotFoundError(f"Result directory does not exist: {result_dir}")

            masks, info = load_result(str(result_dir), sample_image_id)
            auc = compute_auc_from_json(info)
            peak = find_early_peak(info["region_area"], info["insertion_score"], area_limit=0.3)
            attr_map = build_single_sample_attr_map(masks, info)
            word_saliency = np.asarray(get_word_saliency(info, mode="auc_delta"), dtype=np.float32)
            word_region_matrix = build_word_region_matrix(info, mode="delta")
            compact_table(
                [
                    {
                        "algorithm": info["algorithm"],
                        "family": info["algorithm_family"],
                        "segmenter": info["segmenter"],
                        "regions": info["sub-region_number"],
                        "org_score": info["org_score"],
                        "baseline_score": info["baseline_score"],
                        "insertion_auc": auc["insertion_auc"],
                        "deletion_auc": auc["deletion_auc"],
                        "sensitivity_auc": auc.get("insertion_sensitivity_auc"),
                        "highest_score": auc["highest_score"],
                        "early_peak_area": peak["area"],
                        "early_peak_score": peak["score"],
                        "max_word_saliency": float(word_saliency.max()) if len(word_saliency) else None,
                    }
                ]
            )
            """
        ),
        md("## 5. Attribution Overlay, Curves, and Word-Region Correspondence"),
        code(
            """
            early_visible, _ = reveal_image_at_index(image, masks, peak["index"])
            word_labels = [token_label(word) for word in info["words"]]
            cumulative_word_delta = np.asarray(info["insertion_word_score"], dtype=np.float32)
            if len(info.get("deletion_word_score", [])) > 0:
                cumulative_word_delta -= np.asarray(info["deletion_word_score"], dtype=np.float32)

            early_word_df = word_score_table(info["words"], cumulative_word_delta[peak["index"]], top_k=8)
            overall_word_df = pd.DataFrame(
                {
                    "word": word_labels,
                    "saliency": word_saliency,
                }
            ).sort_values("saliency", ascending=False)

            fig = plt.figure(figsize=(16, 9))
            gs = fig.add_gridspec(
                2,
                3,
                width_ratios=[1.25, 1.0, 1.0],
                height_ratios=[1.0, 1.0],
                wspace=0.35,
                hspace=0.35,
            )

            ax_overlay = fig.add_subplot(gs[:, 0])
            ax_early = fig.add_subplot(gs[0, 1])
            ax_metrics = fig.add_subplot(gs[0, 2])
            ax_ins = fig.add_subplot(gs[1, 1])
            ax_del = fig.add_subplot(gs[1, 2])

            plot_overlay_with_colorbar(
                ax_overlay,
                image=image,
                attr_map=attr_map,
                title="single-sample attribution overlay",
            )

            ax_early.imshow(bgr_to_rgb(early_visible))
            ax_early.set_title(
                f"early peak release\\narea={peak['area']:.1%}, score={peak['score']:.3f}"
            )
            ax_early.axis("off")

            metric_panel(
                ax_metrics,
                "caption summary",
                [
                    f"early peak step      : {peak['index'] + 1}",
                    f"released area        : {peak['area']:.1%}",
                    f"caption score @ peak : {peak['score']:.4f}",
                    f"baseline score       : {info['baseline_score']:.4f}",
                    f"original score       : {info['org_score']:.4f}",
                    "top words @ early peak:",
                    *[
                        f"{row.word[:18]:18s} {row.score:+.3f}"
                        for row in early_word_df.itertuples(index=False)
                    ][:5],
                ],
            )

            ins_x, ins_y = metric_trace(info, "insertion_score", info["baseline_score"])
            del_x, del_y = metric_trace(info, "deletion_score", info["org_score"])
            plot_metric_curve(
                ax_ins,
                ins_x,
                ins_y,
                title="insertion curve",
                ylabel="caption score",
                xlabel="revealed area",
                color=SCIENTIFIC_PALETTE["insertion"],
                peak=peak,
                annotate_peak=True,
            )
            plot_metric_curve(
                ax_del,
                del_x,
                del_y,
                title="deletion curve",
                ylabel="caption score",
                xlabel="removed area",
                color=SCIENTIFIC_PALETTE["deletion"],
                peak=peak,
                annotate_peak=False,
            )
            plt.show()

            fig = plt.figure(figsize=(16, 9))
            gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 2.2], hspace=0.3)
            ax_strip = fig.add_subplot(gs[0, 0])
            ax_heat = fig.add_subplot(gs[1, 0])
            plot_token_saliency_strip(ax_strip, info["words"], word_saliency)
            ax_strip.set_title("caption token saliency (AUC over released area)")
            plot_word_region_heatmap(
                ax_heat,
                matrix=word_region_matrix,
                words=info["words"],
                region_area=info["region_area"],
                peak=peak,
            )
            plt.show()

            display(overall_word_df.head(12))
            display(early_word_df)
            """
        ),
        md("## 6. Progressive Reveal"),
        code(
            """
            plot_progressive_reveal(
                image=image,
                masks=masks,
                region_area=info["region_area"],
                fractions=(0.1, 0.25, 0.5, 0.75, 1.0),
            )
            """
        ),
    ]
    return notebook(cells)


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    notebooks = {
        "classification_imagenet_clip_demo.ipynb": classification_notebook(),
        "detection_coco_groundingdino_demo.ipynb": detection_notebook(),
        "caption_qwen25vl_demo.ipynb": caption_notebook(),
    }
    for name, nb in notebooks.items():
        path = NOTEBOOK_DIR / name
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
