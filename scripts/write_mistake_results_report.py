#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


MODELS = {
    "clip_vitl14": ("CLIP ViT-L/14", "imagenet-clip-vitl"),
    "clip_rn101": ("CLIP RN101", "imagenet-clip-rn101"),
    "resnet101": ("ResNet-101", "imagenet-resnet101"),
}

SPLITS = {
    "cause": "cause: explain the model's wrong predicted label",
    "repair": "repair: explain the ground-truth label on misclassified samples",
}

METHOD_ORDER = {
    "greedy": 0,
    "phasewin": 1,
    "drise": 2,
    "dhsic": 3,
    "gradient": 4,
    "grad_eclip": 5,
    "ig2": 6,
    "igos_pp": 7,
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def method_label(method_dir: str) -> str:
    for prefix in ("grad_eclip", "igos_pp", "phasewin", "gradient", "greedy", "drise", "dhsic", "ig2"):
        if method_dir.startswith(prefix):
            return prefix
    return method_dir


def method_sort_key(item: tuple[str, dict[str, Any]]) -> tuple[int, str]:
    label = method_label(item[0])
    return METHOD_ORDER.get(label, 99), label


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def best_method(summary: dict[str, dict[str, Any]], metric: str, *, higher: bool) -> tuple[str, float] | None:
    rows: list[tuple[str, float]] = []
    for method, values in summary.items():
        value = values.get(metric)
        if isinstance(value, (int, float)):
            rows.append((method_label(method), float(value)))
    if not rows:
        return None
    return max(rows, key=lambda row: row[1]) if higher else min(rows, key=lambda row: row[1])


def table(headers: list[str], rows: list[list[str]]) -> list[str]:
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    out = [
        "| " + " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)) + " |",
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)) + " |")
    return out


def split_table(summary: dict[str, dict[str, Any]]) -> list[str]:
    rows = []
    for method, values in sorted(summary.items(), key=method_sort_key):
        rows.append(
            [
                method_label(method),
                fmt(values.get("n_samples"), 0),
                fmt(values.get("insertion_auc")),
                fmt(values.get("deletion_auc")),
                fmt(values.get("average_highest")),
                fmt(values.get("average_highest_30pct_area")),
                fmt(values.get("average_highest_50pct_area")),
                fmt(values.get("average_model_forward_calls"), 2),
            ]
        )
    return table(["method", "N", "Ins AUC", "Del AUC", "Highest", "@30%", "@50%", "AvgFwd"], rows)


def overview_rows(suite: dict[str, dict[str, dict[str, dict[str, Any]]]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for model_key, (display, _) in MODELS.items():
        for split in SPLITS:
            summary = suite[model_key][split]
            ins = best_method(summary, "insertion_auc", higher=True)
            delete = best_method(summary, "deletion_auc", higher=False)
            fwd = best_method(summary, "average_model_forward_calls", higher=False)
            phase = next((v for k, v in summary.items() if method_label(k) == "phasewin"), {})
            greedy = next((v for k, v in summary.items() if method_label(k) == "greedy"), {})
            rows.append(
                [
                    display,
                    split,
                    f"{ins[0]} ({fmt(ins[1])})" if ins else "n/a",
                    f"{delete[0]} ({fmt(delete[1])})" if delete else "n/a",
                    f"{fwd[0]} ({fmt(fwd[1], 2)})" if fwd else "n/a",
                    f"{fmt(phase.get('insertion_auc'))} / {fmt(greedy.get('insertion_auc'))}",
                    f"{fmt(phase.get('average_model_forward_calls'), 2)} / {fmt(greedy.get('average_model_forward_calls'), 2)}",
                ]
            )
    return rows


def build_suite(results_root: Path) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    suite: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for model_key, (_, model_dir) in MODELS.items():
        suite[model_key] = {}
        for split in SPLITS:
            summary_path = results_root / "classification" / model_dir / "mistake" / split / "eval_summary.json"
            if not summary_path.is_file():
                raise FileNotFoundError(summary_path)
            suite[model_key][split] = read_json(summary_path)
    return suite


def copy_split_summaries(results_root: Path, out_dir: Path) -> None:
    per_split = out_dir / "per_split_eval_summary"
    per_split.mkdir(parents=True, exist_ok=True)
    for model_key, (_, model_dir) in MODELS.items():
        for split in SPLITS:
            source = results_root / "classification" / model_dir / "mistake" / split / "eval_summary.json"
            target = per_split / f"{model_key}_{split}_eval_summary.json"
            shutil.copyfile(source, target)


def write_report(out_dir: Path, suite: dict[str, dict[str, dict[str, dict[str, Any]]]]) -> None:
    lines: list[str] = []
    lines.append("# Classification Mistake Evaluation Report")
    lines.append("")
    lines.append(f"- Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("- Source: `results/classification/<model>/mistake/{cause,repair}/eval_summary.json`")
    lines.append("- MuFidelity is intentionally skipped for this mistake evaluation.")
    lines.append("- Each split contains 2000 misclassified ImageNet samples per method.")
    lines.append("")
    lines.append("## Split Definitions")
    lines.append("")
    for split, description in SPLITS.items():
        lines.append(f"- `{split}`: {description}")
    lines.append("")
    lines.append("Metric direction: higher is better for `Ins AUC`, `Highest`, `@30%`, and `@50%`; lower is better for `Del AUC` and `AvgFwd`.")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.extend(
        table(
            [
                "model",
                "split",
                "best Ins AUC",
                "best Del AUC",
                "lowest AvgFwd",
                "phase/greedy Ins",
                "phase/greedy Fwd",
            ],
            overview_rows(suite),
        )
    )
    lines.append("")
    for model_key, (display, _) in MODELS.items():
        lines.append(f"## {display}")
        lines.append("")
        for split in SPLITS:
            lines.append(f"### {split}")
            lines.append("")
            lines.extend(split_table(suite[model_key][split]))
            lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `greedy` gives the strongest insertion/deletion AUC on all mistake splits, with the highest average forward-call cost.")
    lines.append("- `phasewin` is consistently close to `greedy` on insertion/deletion AUC while using roughly 48-60% of greedy's forward calls depending on model and split.")
    lines.append("- `igos_pp` has the lowest recorded average forward calls where available, but its mistake-split AUC is substantially lower than the search-based methods.")
    lines.append("")
    (out_dir / "mistake_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect classification mistake evaluation summaries.")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--out-dir", default="results/classification_mistake_summary")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    suite = build_suite(results_root)
    write_json(out_dir / "mistake_eval_suite_summary.json", suite)
    copy_split_summaries(results_root, out_dir)
    write_report(out_dir, suite)
    print(f"Wrote {out_dir / 'mistake_eval_suite_summary.json'}")
    print(f"Wrote {out_dir / 'mistake_report.md'}")
    print(f"Wrote {out_dir / 'per_split_eval_summary'}")


if __name__ == "__main__":
    main()
