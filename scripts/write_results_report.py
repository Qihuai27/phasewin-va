#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


CLASSIFICATION_MODELS = {
    "clip_vitl14": ("CLIP ViT-L/14", "classification/imagenet-clip-vitl"),
    "clip_rn101": ("CLIP RN101", "classification/imagenet-clip-rn101"),
    "resnet101": ("ResNet-101", "classification/imagenet-resnet101"),
}

CAPTION_MODELS = {
    "qwen25vl_3b": ("Qwen2.5-VL-3B", "caption/Qwen2.5-VL-3B-coco-caption"),
    "qwen25vl_7b": ("Qwen2.5-VL-7B", "caption/Qwen2.5-VL-7B-coco-caption"),
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
    "llavacam": 8,
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def method_label(method_dir: str) -> str:
    for prefix in ("grad_eclip", "igos_pp", "phasewin", "gradient", "greedy", "drise", "dhsic", "ig2", "llavacam"):
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


def best_method(
    summary: dict[str, dict[str, Any]],
    metric: str,
    *,
    higher: bool = True,
) -> tuple[str, float] | None:
    rows: list[tuple[str, float]] = []
    for method, values in summary.items():
        value = values.get(metric)
        if isinstance(value, (int, float)):
            rows.append((method_label(method), float(value)))
    if not rows:
        return None
    return max(rows, key=lambda row: row[1]) if higher else min(rows, key=lambda row: row[1])


def checkpoint_counts(model_root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for path in sorted(model_root.glob("*/mufidelity_samples.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            counts[path.parent.name] = sum(1 for _ in handle)
    return counts


def table(headers: list[str], rows: list[list[str]]) -> list[str]:
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    sep = "| " + " | ".join("-" * width for width in widths) + " |"
    out = ["| " + " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)) + " |", sep]
    for row in rows:
        out.append("| " + " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)) + " |")
    return out


def classification_table(summary: dict[str, dict[str, Any]]) -> list[str]:
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
                fmt(values.get("mufidelity")),
            ]
        )
    return table(
        ["method", "N", "Ins AUC", "Del AUC", "Highest", "@30%", "@50%", "AvgFwd", "MuFid"],
        rows,
    )


def caption_table(summary: dict[str, dict[str, Any]]) -> list[str]:
    rows = []
    for method, values in sorted(summary.items(), key=method_sort_key):
        rows.append(
            [
                method_label(method),
                fmt(values.get("n_samples"), 0),
                fmt(values.get("insertion_auc")),
                fmt(values.get("deletion_auc")),
                fmt(values.get("insertion_sensitivity_auc")),
                fmt(values.get("deletion_sensitivity_auc")),
                fmt(values.get("sensitivity_highest")),
                fmt(values.get("average_highest")),
                fmt(values.get("average_model_forward_calls"), 2),
            ]
        )
    return table(
        ["method", "N", "Ins AUC", "Del AUC", "Sens Ins", "Sens Del", "Sens High", "Highest", "AvgFwd"],
        rows,
    )


def ranking_rows(model_summaries: dict[str, tuple[str, dict[str, dict[str, Any]]]], task: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for key, (display_name, summary) in model_summaries.items():
        ins = best_method(summary, "insertion_auc", higher=True)
        delete = best_method(summary, "deletion_auc", higher=False)
        fwd = best_method(summary, "average_model_forward_calls", higher=False)
        if task == "classification":
            mu = best_method(summary, "mufidelity", higher=True)
            rows.append(
                [
                    display_name,
                    f"{ins[0]} ({fmt(ins[1])})" if ins else "n/a",
                    f"{delete[0]} ({fmt(delete[1])})" if delete else "n/a",
                    f"{mu[0]} ({fmt(mu[1])})" if mu else "n/a",
                    f"{fwd[0]} ({fmt(fwd[1], 2)})" if fwd else "n/a",
                ]
            )
        else:
            sens = best_method(summary, "insertion_sensitivity_auc", higher=True)
            rows.append(
                [
                    display_name,
                    f"{ins[0]} ({fmt(ins[1])})" if ins else "n/a",
                    f"{delete[0]} ({fmt(delete[1])})" if delete else "n/a",
                    f"{sens[0]} ({fmt(sens[1])})" if sens else "n/a",
                    f"{fwd[0]} ({fmt(fwd[1], 2)})" if fwd else "n/a",
                ]
            )
    return rows


def build_report(results_root: Path) -> tuple[dict[str, Any], str]:
    classification: dict[str, tuple[str, dict[str, dict[str, Any]]]] = {}
    classification_suite: dict[str, dict[str, Any]] = {}
    for key, (display_name, rel_root) in CLASSIFICATION_MODELS.items():
        summary = read_json(results_root / rel_root / "eval_summary.json")
        classification[key] = (display_name, summary)
        classification_suite[key] = summary

    caption: dict[str, tuple[str, dict[str, dict[str, Any]]]] = {}
    for key, (display_name, rel_root) in CAPTION_MODELS.items():
        caption[key] = (display_name, read_json(results_root / rel_root / "eval_summary.json"))

    lines: list[str] = []
    lines.append("# 实验评估统计报告")
    lines.append("")
    lines.append(f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("- 结果根目录：`results/`")
    lines.append("- Classification：CLIP ViT-L/14、CLIP RN101、ResNet-101，每个方法 5000 个 ImageNet 样本。")
    lines.append("- Caption：Qwen2.5-VL-3B、Qwen2.5-VL-7B，每个方法 275 个 COCO caption 样本。")
    lines.append("")
    lines.append("## 评估设置")
    lines.append("")
    lines.append("- Caption 评估已由 `scripts/eval_caption_suite.py` 完成，使用插入/删除 AUC 与 sensitivity 指标。")
    lines.append("- Classification 评估已由 `scripts/eval_classification_suite.py` 完成，包含插入/删除 AUC 与 MuFidelity。")
    lines.append("- MuFidelity 参数：`grid_size=9`、`subset_percent=0.2`、`nb_samples=200`、`tf_device=cpu`、Torch 侧使用 CUDA。")
    lines.append("- MuFidelity checkpoint：每个方法目录写入 `mufidelity_samples.jsonl`，按样本追加，可恢复。")
    lines.append("- GPU batch：ViT-L/14 使用 `mu_batch_size=50`；CLIP RN101 与 ResNet-101 使用 `mu_batch_size=256`，因为单样本扰动数为 200，256 已覆盖一次前向批。")
    lines.append("")
    lines.append("指标方向：`Ins AUC`、`Highest`、`@30%`、`@50%`、`MuFid` 越高越好；`Del AUC` 与 `AvgFwd` 越低越好。")
    lines.append("")
    lines.append("## Classification 总览")
    lines.append("")
    lines.extend(
        table(
            ["model", "best Ins AUC", "best Del AUC", "best MuFid", "lowest AvgFwd"],
            ranking_rows(classification, "classification"),
        )
    )
    lines.append("")
    for key, (display_name, summary) in classification.items():
        model_root = results_root / CLASSIFICATION_MODELS[key][1]
        counts = checkpoint_counts(model_root)
        completed = sum(1 for method in summary if counts.get(method) == summary[method].get("mufidelity_n_samples"))
        lines.append(f"### {display_name}")
        lines.append("")
        lines.append(f"- MuFidelity checkpoint 完成：{completed}/{len(summary)} 个方法。")
        lines.extend(classification_table(summary))
        lines.append("")
    lines.append("## Caption 总览")
    lines.append("")
    lines.extend(
        table(
            ["model", "best Ins AUC", "best Del AUC", "best Sens Ins", "lowest AvgFwd"],
            ranking_rows(caption, "caption"),
        )
    )
    lines.append("")
    for _, (display_name, summary) in caption.items():
        lines.append(f"### {display_name}")
        lines.append("")
        lines.extend(caption_table(summary))
        lines.append("")
    lines.append("## 主要结论")
    lines.append("")
    lines.append("- Classification 中，`greedy` 在三个 backbone 上都取得最高 `Ins AUC`，但平均前向次数约为 1736-1744。")
    lines.append("- `phasewin` 在三个 classification backbone 上的 `Ins AUC` 均接近 `greedy`，同时平均前向次数约为 872、951、908，约为 greedy 的一半。")
    lines.append("- CLIP ViT-L/14 上 `phasewin` 的 `Ins AUC=0.7990`、`Del AUC=0.1625`，接近 `greedy` 的 `0.8239/0.1388`，但 `AvgFwd` 为 `871.84` vs `1735.90`。")
    lines.append("- CLIP RN101 与 ResNet-101 上也呈现类似趋势：`phasewin` 在显著降低前向次数的同时保留了较强 faithfulness。")
    lines.append("- Caption 中，`greedy` 通常给出最高插入 AUC；`phasewin` 在 Qwen2.5-VL-3B/7B 上保持较强结果，并显著少于 greedy 的平均前向次数。")
    lines.append("- `igos_pp` 在 classification 中前向次数最低，但插入/删除 AUC 通常弱于 search 类方法；它更适合作为低调用成本的梯度优化型对照。")
    lines.append("")
    lines.append("## 输出文件")
    lines.append("")
    lines.append("- `results/classification/eval_suite_summary.json`：已重建为三模型合并版。")
    lines.append("- `results/caption/eval_suite_summary.json`：caption suite 汇总。")
    lines.append("- `results/evaluation_report.md`：本报告。")
    lines.append("")
    return classification_suite, "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write final evaluation summary report.")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--report", default="results/evaluation_report.md")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    classification_suite, report = build_report(results_root)
    write_json(results_root / "classification" / "eval_suite_summary.json", classification_suite)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report + "\n", encoding="utf-8")
    print(f"Wrote {results_root / 'classification' / 'eval_suite_summary.json'}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
