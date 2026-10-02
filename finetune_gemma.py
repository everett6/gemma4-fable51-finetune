#!/usr/bin/env python3
"""Text QLoRA distillation of Gemma 4 12B, with optional LM Studio GGUF export."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

BASE_MODEL = "google/gemma-4-12B-it-qat-q4_0-unquantized"
DATASET = "MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x"
LOCAL_GGUF = Path.home() / ".lmstudio/models/lmstudio-community/gemma-4-12B-it-QAT-GGUF"
TURN = re.compile(r"<\|turn>([^\n]+)\n")


def text_content(content):
    """Keep text; serialize tool traces without pretending to implement native tools."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError("Expected string or content-block list")
    parts = []
    for block in content:
        kind = block.get("type")
        if kind in ("text", "input_text", "output_text"):
            parts.append(block["text"])
        elif kind in ("thinking", "reasoning"):
            parts.append("<|channel>thought\n" + block.get("thinking", block.get("text", "")) + "<channel|>")
        elif kind in ("tool_use", "tool_call", "tool_result"):
            parts.append(json.dumps(block, ensure_ascii=False, sort_keys=True))
        else:
            raise ValueError(f"Unsupported content block: {kind!r}")
    return "\n".join(parts)


def normalize_messages(row):
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Row needs a nonempty messages list")
    result = []
    for message in messages:
        role = message.get("role")
        content = text_content(message.get("content", ""))
        if "<|turn>" in content:
            raise ValueError("Raw turn delimiter in content would corrupt loss masking")
        if role == "tool":
            role = "user"
            content = "[Tool result]\n" + content
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"Unsupported role: {role!r}")
        if role == "assistant":
            content = re.sub(r"<think>(.*?)</think>", r"<|channel>thought\n\1<channel|>", content, flags=re.S)
            if message.get("tool_calls"):
                content += "\n[Tool calls]\n" + json.dumps(message["tool_calls"], ensure_ascii=False, sort_keys=True)
        # Adjacent tool results / assistant blocks are combined into one turn.
        if result and result[-1]["role"] == role:
            result[-1]["content"] += "\n" + content
        else:
            result.append({"role": role, "content": content})
    if not any(m["role"] == "assistant" and m["content"].strip() for m in result):
        raise ValueError("No assistant output in row")
    return result


def mask_labels(rendered, ids, offsets):
    """Mask all tokens outside model turns, including all role headers."""
    headers = list(TURN.finditer(rendered))
    if not headers:
        raise ValueError("Checkpoint does not use the expected Gemma 4 chat template")
    spans = [(h.end(), headers[i + 1].start() if i + 1 < len(headers) else len(rendered))
             for i, h in enumerate(headers) if h.group(1).split()[0] == "model"]
    labels = []
    span_index = 0
    for token, (start, end) in zip(ids, offsets, strict=True):
        while span_index < len(spans) and start >= spans[span_index][1]:
            span_index += 1
        inside = (span_index < len(spans) and end > start
                  and start >= spans[span_index][0] and end <= spans[span_index][1])
        labels.append(token if inside else -100)
    return labels


def windows(ids, labels, length, overlap):
    """Keep long traces, with overlapping context whose loss is counted once."""
    step = length - overlap
    for start in range(0, len(ids), step):
        end = min(start + length, len(ids))
        chunk_labels = labels[start:end].copy()
        # Causal loss cannot predict the first token of a new window.
        masked_prefix = overlap if start else 1
        chunk_labels[:masked_prefix] = [-100] * min(masked_prefix, len(chunk_labels))
        if any(label != -100 for label in chunk_labels[1:]):
            yield {"input_ids": ids[start:end], "attention_mask": [1] * (end - start),
                   "labels": chunk_labels}
        if end == len(ids):
            break


def preflight():
    # Import Unsloth before transformers/PEFT so its training patches apply.
    if not (3, 10) <= sys.version_info[:2] <= (3, 13):
        raise RuntimeError("Use Python 3.10–3.13 for the Unsloth training environment; see README.md")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check nvidia-smi and install a CUDA-enabled PyTorch build.")
    print(f"GPU: {torch.cuda.get_device_name(0)}; VRAM: {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
    print("Local inference files:", ", ".join(p.name for p in LOCAL_GGUF.glob("*.gguf")) or "not found")
    from unsloth import FastModel
    return FastModel, torch


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, features):
        import torch
        length = max(len(f["input_ids"]) for f in features)
        pads = {"input_ids": self.pad_id, "attention_mask": 0, "labels": -100}
        return {key: torch.tensor([f[key] + [pad] * (length - len(f[key])) for f in features], dtype=torch.long)
                for key, pad in pads.items()}


def prepare(args, tokenizer, processor):
    from datasets import load_dataset
    raw = load_dataset(args.dataset, args.dataset_config, split=args.split, streaming=True,
                       revision=args.dataset_revision)
    raw = raw.shuffle(seed=args.seed, buffer_size=32)
    prepared = args.output / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    counts = {"traces": 0, "duplicates": 0, "rejected": 0, "train": 0, "validation": 0}
    seen = set()
    with (prepared / "train.jsonl").open("w") as train, (prepared / "validation.jsonl").open("w") as valid:
        for index, row in enumerate(raw):
            if args.max_traces and index >= args.max_traces:
                break
            try:
                messages = normalize_messages(row)
                digest = hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                if digest in seen:
                    counts["duplicates"] += 1
                    continue
                seen.add(digest)
                rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False,
                                                        enable_thinking=True)
                tokens = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
                ids = tokens["input_ids"]
                labels = mask_labels(rendered, ids, tokens["offset_mapping"])
                if not any(label != -100 for label in labels):
                    raise ValueError("No assistant tokens survived the chat template")
            except (ValueError, TypeError, KeyError) as error:
                counts["rejected"] += 1
                if counts["rejected"] <= 5:
                    print(f"Skipping trace {index}: {error}")
                continue
            # Split by conversation before chunking; seed and content determine membership.
            bucket = int(hashlib.sha256(f"{args.seed}:{digest}".encode()).hexdigest()[:8], 16) / 2**32
            split = "validation" if bucket < args.eval_fraction else "train"
            destination = valid if split == "validation" else train
            for window in windows(ids, labels, args.context, args.overlap):
                destination.write(json.dumps(window) + "\n")
                counts[split] += 1
            counts["traces"] += 1
            if counts["traces"] % 10 == 0:
                print("Preparation:", counts, flush=True)
    print("Prepared:", counts)
    if not counts["train"] or not counts["validation"]:
        raise RuntimeError("Need train and validation windows. Increase --max-traces or adjust --eval-fraction.")
    if counts["rejected"]:
        print("Review rejected-row counts before a full run; unsupported multimodal rows are skipped.")
    return load_dataset("json", data_files={"train": str(prepared / "train.jsonl"),
                                           "validation": str(prepared / "validation.jsonl")}), counts


def train(args):
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise RuntimeError("Output directory is not empty. Use a new --output or --resume CHECKPOINT.")
    FastModel, torch = preflight()
    from transformers import AutoProcessor, Trainer, TrainingArguments, set_seed
    set_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.prepared_dir:
        manifest = json.loads((args.prepared_dir / "manifest.json").read_text())
        if manifest.get("format") != "gemma-token-windows-v1":
            raise RuntimeError("Unrecognized prepared dataset format")
        if manifest["tokenizer"] != args.model or manifest["tokenizer_revision"] != args.model_revision:
            raise RuntimeError("Prepared tokenizer/model ID and revision must match --model and --model-revision")
        if manifest["context"] > args.context:
            raise RuntimeError("Prepared windows exceed --context; reprocess with a smaller context or increase --context")
        text_processor = AutoProcessor.from_pretrained(str(args.prepared_dir / "tokenizer"), use_fast=True)
    else:
        text_processor = AutoProcessor.from_pretrained(args.model, revision=args.model_revision, use_fast=True)
    tokenizer = getattr(text_processor, "tokenizer", text_processor)
    if not tokenizer.is_fast:
        raise RuntimeError("Assistant loss masking requires a fast tokenizer with offsets")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.prepared_dir:
        from datasets import load_dataset
        files = {split: [str(path) for path in sorted(args.prepared_dir.glob(f"*/{split}.jsonl"))
                         if path.stat().st_size] for split in ("train", "validation")}
        if not all(files.values()):
            raise RuntimeError("Prepared directory must include nonempty train and validation JSONL files")
        dataset = load_dataset("json", data_files=files)
        counts = manifest["statistics"]
        print("Loading prepared windows from both datasets:", {s: len(d) for s, d in dataset.items()})
    else:
        dataset, counts = prepare(args, tokenizer, text_processor)
    if args.prepare_only:
        print(f"Prepared tokenized data in {args.prepared_dir or args.output / 'prepared'}; no model weights loaded.")
        return
    model, processor = FastModel.from_pretrained(
        model_name=args.model, revision=args.model_revision, max_seq_length=args.context,
        load_in_4bit=True, full_finetuning=False, dtype=None,
        use_gradient_checkpointing="unsloth",
    )
    model = FastModel.get_peft_model(
        model, finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0, bias="none",
        random_state=args.seed, use_gradient_checkpointing="unsloth",
    )
    trainer = Trainer(
        model=model, processing_class=tokenizer,
        train_dataset=dataset["train"], eval_dataset=dataset["validation"],
        data_collator=Collator(tokenizer.pad_token_id),
        args=TrainingArguments(
            output_dir=str(args.output / "checkpoints"),
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            gradient_accumulation_steps=args.accumulation,
            max_steps=args.steps, num_train_epochs=args.epochs,
            learning_rate=args.learning_rate, warmup_ratio=0.03,
            lr_scheduler_type="cosine", optim="adamw_8bit", weight_decay=0.01,
            bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
            logging_steps=5, save_steps=50, eval_steps=50, eval_strategy="steps",
            save_total_limit=2, prediction_loss_only=True, report_to="none",
            seed=args.seed, data_seed=args.seed, dataloader_num_workers=0,
        ),
    )
    trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    metrics = trainer.evaluate()
    adapter = args.output / "adapter"
    model.save_pretrained(str(adapter))
    text_processor.save_pretrained(str(adapter))
    # Preserve exactly the template used for training, including native thinking controls.
    tokenizer.save_pretrained(str(adapter))
    manifest = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    manifest.update({"prepared_counts": counts, "validation_metrics": metrics,
                     "torch_version": torch.__version__, "gpu": torch.cuda.get_device_name(0)})
    (args.output / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Adapter saved: {adapter}\nValidation: {metrics}")


def export(args):
    if not (args.adapter / "adapter_config.json").is_file():
        raise RuntimeError("Adapter directory must contain adapter_config.json")
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("Export directory is not empty; choose a new --output")
    FastModel, _ = preflight()
    model, processor = FastModel.from_pretrained(model_name=str(args.adapter), max_seq_length=1024,
                                                load_in_4bit=True)
    model.save_pretrained_gguf(str(args.output), processor, quantization_method=args.quantization)
    print(f"GGUF export saved in {args.output}. Import the generated text model into LM Studio.")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train", help="Download trainable weights/data and train a text QLoRA adapter")
    t.add_argument("--model", default=BASE_MODEL, help="HF safetensors checkpoint or local directory, not GGUF")
    t.add_argument("--model-revision", default="main")
    t.add_argument("--dataset", default=DATASET)
    t.add_argument("--dataset-config", default="default")
    t.add_argument("--dataset-revision", default="main")
    t.add_argument("--split", default="lite", help="Public dataset splits: lite (5k) or full (10k)")
    t.add_argument("--output", type=Path, default=Path("outputs/gemma4-fable51"))
    t.add_argument("--max-traces", type=int, default=100, help="0 = entire split; default is a pilot run")
    t.add_argument("--context", type=int, default=1024)
    t.add_argument("--overlap", type=int, default=128)
    t.add_argument("--rank", type=int, default=8)
    t.add_argument("--accumulation", type=int, default=8)
    t.add_argument("--learning-rate", type=float, default=2e-5)
    t.add_argument("--steps", type=int, default=200, help="-1 = use --epochs")
    t.add_argument("--epochs", type=float, default=1.0)
    t.add_argument("--eval-fraction", type=float, default=0.05)
    t.add_argument("--seed", type=int, default=3407)
    t.add_argument("--resume", type=Path, help="Path to a Trainer checkpoint, not the final adapter")
    t.add_argument("--prepare-only", action="store_true")
    t.add_argument("--prepared-dir", type=Path, help="Load token windows from shorten_datasets.py without re-tokenizing")
    e = sub.add_parser("export", help="Merge saved adapter and export GGUF (requires extra RAM/disk)")
    e.add_argument("--adapter", type=Path, default=Path("outputs/gemma4-fable51/adapter"))
    e.add_argument("--output", type=Path, default=Path("outputs/gemma4-fable51-gguf"))
    e.add_argument("--quantization", choices=("q4_0", "q4_k_m", "q8_0", "f16"), default="q4_k_m")
    return p


def main():
    p = parser()
    args = p.parse_args()
    if args.command == "train":
        if args.model.lower().endswith(".gguf") or (Path(args.model).is_dir() and not (Path(args.model) / "config.json").exists()):
            p.error("GGUF is an inference format. Use the default trainable QAT checkpoint or an HF model directory.")
        if (args.context < 64 or not 1 <= args.overlap < args.context or args.max_traces < 0
                or not 0 < args.eval_fraction < 1 or args.rank < 1 or args.accumulation < 1
                or (args.steps != -1 and args.steps < 1) or args.epochs <= 0 or args.learning_rate <= 0):
            p.error("Invalid training settings; check lengths, counts, fractions and learning rate")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        (train if args.command == "train" else export)(args)
    except (ImportError, RuntimeError) as error:
        sys.exit(f"Error: {error}")


if __name__ == "__main__":
    main()
