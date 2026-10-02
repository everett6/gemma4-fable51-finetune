#!/usr/bin/env python3
"""Prepare Fable and Arcee as bounded Gemma-token training windows on CPU."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import itertools
import json
from pathlib import Path
import sys

from finetune_gemma import BASE_MODEL, DATASET, mask_labels, normalize_messages, windows

ARCEE = "arcee-ai/reasoning-sharegpt"


def normalize_row(row):
    if "messages" in row:
        return normalize_messages(row)
    turns = row.get("conversations")
    if not isinstance(turns, list) or not turns:
        raise ValueError("Expected messages or conversations list")
    roles = {"human": "user", "gpt": "assistant", "system": "system",
             "user": "user", "assistant": "assistant", "tool": "tool"}
    converted = []
    for turn in turns:
        role = turn.get("from", turn.get("role"))
        if role not in roles:
            raise ValueError(f"Unsupported ShareGPT role: {role!r}")
        converted.append({"role": roles[role], "content": turn.get("value", turn.get("content", ""))})
    return normalize_messages({"messages": converted})


def trace_identity(messages, seed, eval_fraction):
    digest = hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    bucket = int(hashlib.sha256(f"{seed}:{digest}".encode()).hexdigest()[:8], 16) / 2**32
    return digest, "validation" if bucket < eval_fraction else "train"


def encode_trace(messages, tokenizer, processor, context, overlap):
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False,
                                             enable_thinking=True)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True,
                        truncation=False)
    ids = encoded["input_ids"]
    labels = mask_labels(rendered, ids, encoded["offset_mapping"])
    if not any(label != -100 for label in labels[1:]):
        raise ValueError("No supervised assistant tokens in trace")
    return len(ids), windows(ids, labels, context, overlap)


def local_rows(path):
    # Local inputs are JSONL, one conversation per line, not a JSON array.
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from error


def write_source(rows, name, root, tokenizer, processor, args, seen):
    directory = root / name
    directory.mkdir()
    stats = {"rows_read": 0, "accepted_traces": 0, "duplicates": 0, "rejected": 0,
             "long_traces": 0, "original_tokens": 0, "max_original_tokens": 0,
             "train_windows": 0, "validation_windows": 0, "max_window_tokens": 0,
             "supervised_tokens": 0}
    limited = itertools.islice(rows, args.max_rows) if args.max_rows else rows
    with ExitStack() as stack:
        outputs = {split: stack.enter_context((directory / f"{split}.jsonl").open("w"))
                   for split in ("train", "validation")}
        rejects = stack.enter_context((directory / "rejected.jsonl").open("w"))
        for index, row in enumerate(limited):
            stats["rows_read"] += 1
            try:
                messages = normalize_row(row)
                digest, split = trace_identity(messages, args.seed, args.eval_fraction)
                if digest in seen:
                    stats["duplicates"] += 1
                    continue
                token_count, chunks = encode_trace(messages, tokenizer, processor, args.context, args.overlap)
                count = 0
                for chunk_index, chunk in enumerate(chunks):
                    length = len(chunk["input_ids"])
                    if length > args.context:
                        raise RuntimeError("Window exceeded requested context")
                    chunk.update({"source": name, "trace_id": digest, "window_index": chunk_index})
                    outputs[split].write(json.dumps(chunk) + "\n")
                    stats[f"{split}_windows"] += 1
                    stats["max_window_tokens"] = max(stats["max_window_tokens"], length)
                    stats["supervised_tokens"] += sum(label != -100 for label in chunk["labels"])
                    count += 1
                if not count:
                    raise ValueError("No trainable windows")
                seen.add(digest)
                stats["accepted_traces"] += 1
                stats["original_tokens"] += token_count
                stats["max_original_tokens"] = max(stats["max_original_tokens"], token_count)
                stats["long_traces"] += int(token_count > args.context)
            except (ValueError, TypeError, KeyError) as error:
                stats["rejected"] += 1
                # Record row index and reason, without copying large source traces.
                rejects.write(json.dumps({"row": index, "reason": str(error)}) + "\n")
            if stats["rows_read"] % 100 == 0:
                print(f"{name}: {stats}", flush=True)
    if not stats["accepted_traces"]:
        raise RuntimeError(f"No usable traces from {name}; inspect {directory / 'rejected.jsonl'}")
    print(f"{name}: {stats}", flush=True)
    return stats


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--context", type=int, default=1024, help="Maximum tokens per output window, including template")
    p.add_argument("--overlap", type=int, default=128, help="Previous tokens kept as masked context")
    p.add_argument("--output", type=Path, default=Path("data/short-context-1024"))
    p.add_argument("--tokenizer", default=BASE_MODEL, help="Gemma HF model ID or local processor/tokenizer directory")
    p.add_argument("--tokenizer-revision", default="main")
    p.add_argument("--fable-split", choices=("full", "lite"), default="full")
    p.add_argument("--fable-revision", default="main")
    p.add_argument("--arcee-revision", default="main")
    p.add_argument("--fable-file", type=Path, help="Use local Fable JSONL instead of the Hub")
    p.add_argument("--arcee-file", type=Path, help="Use local Arcee JSONL instead of the Hub")
    p.add_argument("--max-rows", type=int, default=0, help="Rows from each source; 0 = all, 100 = quick pilot")
    p.add_argument("--eval-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=3407)
    return p


def main():
    p = parser()
    args = p.parse_args()
    if args.context < 64 or not 1 <= args.overlap < args.context or args.max_rows < 0 or not 0 < args.eval_fraction < 1:
        p.error("Use context >= 64, 1 <= overlap < context, max-rows >= 0, and 0 < eval-fraction < 1")
    if args.output.exists():
        p.error("Output already exists; choose a new directory to avoid overwriting data")
    try:
        # No model weights, CUDA, or Unsloth required for preprocessing.
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(args.tokenizer, revision=args.tokenizer_revision, use_fast=True)
        tokenizer = getattr(processor, "tokenizer", processor)
        if not tokenizer.is_fast:
            raise RuntimeError("A fast tokenizer is required for accurate assistant loss masking")
        def source(path, repo, split, revision):
            if path:
                return local_rows(path)
            from datasets import load_dataset
            return load_dataset(repo, "default", split=split, streaming=True, revision=revision)
        sources = [("fable", source(args.fable_file, DATASET, args.fable_split, args.fable_revision)),
                   ("arcee", source(args.arcee_file, ARCEE, "train", args.arcee_revision))]
        args.output.mkdir(parents=True)
        seen = set()
        stats = {name: write_source(rows, name, args.output, tokenizer, processor, args, seen)
                 for name, rows in sources}
        if not sum(s["train_windows"] for s in stats.values()) or not sum(s["validation_windows"] for s in stats.values()):
            raise RuntimeError("Need both train and validation windows; use more rows or adjust eval-fraction")
        processor.save_pretrained(str(args.output / "tokenizer"))
        manifest = {"format": "gemma-token-windows-v1", "tokenizer": args.tokenizer,
                    "tokenizer_revision": args.tokenizer_revision, "context": args.context,
                    "overlap": args.overlap, "seed": args.seed, "eval_fraction": args.eval_fraction,
                    "sources": {"fable": {"repo": DATASET, "split": args.fable_split, "revision": args.fable_revision},
                                "arcee": {"repo": ARCEE, "split": "train", "revision": args.arcee_revision}},
                    "statistics": stats}
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Done: {args.output}. Every window is at most {args.context} tokens.")
    except (ImportError, RuntimeError, ValueError, OSError) as error:
        sys.exit(f"Error: {error}. Incomplete output directories have no manifest.json; use a new output for retry.")


if __name__ == "__main__":
    main()
