#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate AUC faithfulness for all classification attribution results.

Scans every method subdirectory under the results directory, computes
insertion/deletion AUC metrics, writes a per-method eval.json alongside the
attribution results, and saves a combined eval_summary.json at the top level.

Usage:
  python scripts/eval_classification.py [--results-dir DIR]

Options:
  --results-dir DIR  Base results directory.
                     Default: ./classification_results/imagenet-clip-vitl
  -h, --help         Show this help.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from attribution_research.evaluation.auc_faithfulness import AUCResult, aggregate_results  # noqa: E402
from attribution_research.evaluation.mufidelity import evaluate_classification_mufidelity  # noqa: E402


# ── helpers ───────────────────────────────────────────────────────────────────

def find_method_dirs(base_dir: Path) -> list[Path]:
    """Return subdirectories that contain attribution results (json/*.json)."""
    dirs = []
    if not base_dir.is_dir():
        return dirs
    for d in sorted(base_dir.iterdir()):
        if d.is_dir() and (d / "json").is_dir() and any((d / "json").glob("*.json")):
            dirs.append(d)
    return dirs


def _fmt(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "  n/a "


def _fmt_forward(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "  n/a "


def print_table(rows: list[tuple[str, AUCResult]]) -> None:
    """Print a formatted comparison table to stdout."""
    col_method = max((len(name) for name, _ in rows), default=6)
    col_method = max(col_method, 6)
    show_mu = any(result.mufidelity is not None for _, result in rows)

    header = (
        f"{'Method':<{col_method}}  {'N':>6}  "
        f"{'Ins AUC':>8}  {'Del AUC':>8}  "
        f"{'Highest':>8}  {'@30%':>8}  {'@50%':>8}  {'AvgFwd':>10}"
    )
    if show_mu:
        header += f"  {'MuFid':>8}"
    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)
    for name, r in rows:
        print(
            f"{name:<{col_method}}  {r.n_samples:>6}  "
            f"{_fmt(r.insertion_auc):>8}  {_fmt(r.deletion_auc):>8}  "
            f"{_fmt(r.average_highest):>8}  "
            f"{_fmt(r.average_highest_30pct_area):>8}  "
            f"{_fmt(r.average_highest_50pct_area):>8}  "
            f"{_fmt_forward(r.average_model_forward_calls):>10}",
            end="",
        )
        if show_mu:
            print(f"  {_fmt(r.mufidelity):>8}", end="")
        print()
    print(sep)
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate classification attribution results (AUC faithfulness).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        default="./classification_results/imagenet-clip-vitl",
        help="Base results directory (default: ./classification_results/imagenet-clip-vitl)",
    )
    parser.add_argument(
        "--mufidelity",
        action="store_true",
        help="Also compute MuFidelity for each method directory.",
    )
    parser.add_argument(
        "--mu-limit",
        type=int,
        default=None,
        help="Optional max number of samples per method for MuFidelity.",
    )
    parser.add_argument(
        "--mu-grid-size",
        type=int,
        default=9,
        help="Grid size passed to xplique MuFidelity (default: 9).",
    )
    parser.add_argument(
        "--mu-subset-percent",
        type=float,
        default=0.2,
        help="Perturbed area fraction for MuFidelity (default: 0.2).",
    )
    parser.add_argument(
        "--mu-nb-samples",
        type=int,
        default=200,
        help="Number of perturbations per sample for MuFidelity (default: 200).",
    )
    parser.add_argument(
        "--mu-batch-size",
        type=int,
        default=8,
        help="Model batch size passed to MuFidelity (default: 8).",
    )
    parser.add_argument(
        "--mu-sample-batch-size",
        type=int,
        default=8,
        help=(
            "How many saved explanations to evaluate together per MuFidelity chunk "
            "when checkpointing is disabled (default: 8)."
        ),
    )
    parser.add_argument(
        "--mu-baseline",
        default="0.0",
        help="Baseline fill value for MuFidelity perturbations (default: 0.0).",
    )
    parser.add_argument(
        "--mu-score-key",
        default="insertion_score",
        help="JSON score key used to rebuild saliency maps for MuFidelity (default: insertion_score).",
    )
    parser.add_argument(
        "--mu-device",
        default="cuda",
        help="Torch device used by the classification model during MuFidelity (default: cuda).",
    )
    parser.add_argument(
        "--mu-tf-device",
        default="cpu",
        help="TensorFlow visibility policy for MuFidelity: cpu, auto, or gpu (default: cpu).",
    )
    parser.add_argument(
        "--mu-seed",
        type=int,
        default=0,
        help="Random seed for MuFidelity perturbations (default: 0).",
    )
    parser.add_argument(
        "--mu-semantic-features",
        default=None,
        help="Optional override for CLIP semantic feature weights during MuFidelity.",
    )
    parser.add_argument(
        "--mu-model-family",
        default=None,
        help="Optional model family override for MuFidelity (e.g. clip, torchvision).",
    )
    parser.add_argument(
        "--mu-clip-type",
        default=None,
        help="Optional CLIP backbone override for MuFidelity (e.g. ViT-L/14, RN101).",
    )
    parser.add_argument(
        "--mu-arch",
        default=None,
        help="Optional torchvision architecture override for MuFidelity (e.g. resnet50).",
    )
    parser.add_argument(
        "--mu-weights",
        default=None,
        help="Optional torchvision weights override for MuFidelity (default: infer from saved JSON).",
    )
    parser.add_argument(
        "--mu-no-checkpoint",
        action="store_true",
        help=(
            "Disable per-sample MuFidelity JSONL checkpointing. By default each method "
            "writes <method-dir>/mufidelity_samples.jsonl and resumes matching records."
        ),
    )
    parser.add_argument(
        "--mu-checkpoint-dir",
        default=None,
        help=(
            "Optional directory for MuFidelity checkpoint JSONL files. "
            "Default: each method directory."
        ),
    )
    args = parser.parse_args()

    base_dir = Path(args.results_dir)
    if not base_dir.is_dir():
        print(f"[error] Results directory not found: {base_dir}", file=sys.stderr)
        sys.exit(1)

    method_dirs = find_method_dirs(base_dir)
    if not method_dirs:
        print(f"[error] No completed results found under {base_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(method_dirs)} method(s) under {base_dir}")

    rows: list[tuple[str, AUCResult]] = []
    summary: dict = {}

    for method_dir in method_dirs:
        method_name = method_dir.name
        print(f"  evaluating  {method_name} ...", end=" ", flush=True)
        try:
            result = aggregate_results(str(method_dir), show_progress=False)
        except Exception as exc:
            print(f"FAILED ({exc})")
            continue

        mu_config = None
        if args.mufidelity:
            try:
                mu_result = evaluate_classification_mufidelity(
                    method_dir,
                    score_key=args.mu_score_key,
                    limit=args.mu_limit,
                    grid_size=args.mu_grid_size,
                    subset_percent=args.mu_subset_percent,
                    nb_samples=args.mu_nb_samples,
                    batch_size=args.mu_batch_size,
                    sample_batch_size=args.mu_sample_batch_size,
                    baseline_mode=args.mu_baseline,
                    semantic_feature_path=args.mu_semantic_features,
                    model_family=args.mu_model_family,
                    clip_type=args.mu_clip_type,
                    arch=args.mu_arch,
                    weights=args.mu_weights,
                    device=args.mu_device,
                    tf_device=args.mu_tf_device,
                    seed=args.mu_seed,
                    use_checkpoint=not args.mu_no_checkpoint,
                    checkpoint_path=(
                        Path(args.mu_checkpoint_dir) / f"{method_dir.name}.mufidelity_samples.jsonl"
                        if args.mu_checkpoint_dir is not None
                        else None
                    ),
                    show_progress=False,
                )
            except Exception as exc:
                print(f"FAILED (MuFidelity: {exc})")
                continue
            result.mufidelity = mu_result.score
            result.mufidelity_n_samples = mu_result.n_samples
            mu_config = mu_result.config_dict()

        # Save per-method JSON
        eval_payload = dataclasses.asdict(result)
        if mu_config is not None:
            eval_payload["mufidelity_config"] = mu_config
        eval_path = method_dir / "eval.json"
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(eval_payload, f, indent=2)

        fwd_str = (
            f"  fwd={result.average_model_forward_calls:.2f}"
            if result.average_model_forward_calls is not None else ""
        )
        mu_str = (
            f"  mu={result.mufidelity:.4f}"
            if result.mufidelity is not None else ""
        )
        print(
            f"n={result.n_samples}  ins={result.insertion_auc:.4f}  "
            f"del={result.deletion_auc:.4f}{fwd_str}{mu_str}"
        )
        rows.append((method_name, result))
        summary_entry = dataclasses.asdict(result)
        if mu_config is not None:
            summary_entry["mufidelity_config"] = mu_config
        summary[method_name] = summary_entry

    if not rows:
        print("[error] All evaluations failed.", file=sys.stderr)
        sys.exit(1)

    # Save combined summary
    summary_path = base_dir / "eval_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")

    print_table(rows)


if __name__ == "__main__":
    main()
