#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a Qwen2.5-VL caption eval list for the current COCO pilot split.

The source eval list is only used as the image roster (and optional GT
captions). The generated sentence and target token metadata are rebuilt with
the requested model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attribution_research import load_qwen25vl_model


PROMPT = (
    "Describe the image in one factual English sentence of no more than 20 words. "
    "Do not include information that is not clearly visible."
)


def maybe_resize_image(image: Image.Image, max_image_side: int | None) -> Image.Image:
    if max_image_side is None:
        return image
    width, height = image.size
    longest = max(width, height)
    if longest <= max_image_side:
        return image
    scale = float(max_image_side) / float(longest)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Qwen2.5-VL caption eval list")
    parser.add_argument(
        "--datasets",
        default="datasets/coco/val2017",
        help="COCO image root",
    )
    parser.add_argument(
        "--source-eval-list",
        default="datasets/Qwen2.5-VL-3B-coco-caption.json",
        help="Source eval list used as the image roster",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="Qwen2.5-VL checkpoint path or Hugging Face model id",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--begin", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--max-image-side",
        type=int,
        default=None,
        help="Optional longest-edge cap applied before Qwen image preprocessing.",
    )
    return parser.parse_args()


def build_inputs(processor, image: Image.Image):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    if hasattr(processor, "apply_chat_template"):
        try:
            return processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except Exception:
            pass
    return processor(images=image, text=PROMPT, return_tensors="pt")


def main() -> None:
    args = parse_args()

    with open(args.source_eval_list, "r", encoding="utf-8") as f:
        source_items = json.load(f)
    source_items = source_items[args.begin : args.end]
    if not source_items:
        raise ValueError("No caption samples selected after applying begin/end")

    model, processor = load_qwen25vl_model(args.model_name, args.device)
    tokenizer = processor.tokenizer

    output_items = []
    for item in tqdm(source_items, desc="Building caption eval list"):
        image_path = Path(args.datasets) / item["image_path"]
        image = Image.open(image_path).convert("RGB")
        image = maybe_resize_image(image, args.max_image_side)

        inputs = build_inputs(processor, image)
        inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
        input_ids = inputs["input_ids"]
        prefix_len = int(input_ids.shape[1])

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )

        full_ids = generated[0].detach().cpu()
        gen_ids = full_ids[prefix_len:]
        if gen_ids.numel() == 0:
            print(f"[warn] no generated tokens for {item['image_path']}, skipping")
            continue

        sentence = tokenizer.decode(
            gen_ids.tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()
        token_words = [
            tokenizer.decode(
                [tok_id],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for tok_id in gen_ids.tolist()
        ]

        output_items.append(
            {
                "image_path": item["image_path"],
                "gt_caption": item.get("gt_caption", []),
                "generate_sentence": sentence,
                "generated_ids": [full_ids.tolist()],
                "selected_interpretation_token_id": list(range(int(gen_ids.numel()))),
                "selected_interpretation_token_word_id": gen_ids.tolist(),
                "words": token_words,
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_items, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(output_items)} items -> {output_path.resolve()}")


if __name__ == "__main__":
    main()
