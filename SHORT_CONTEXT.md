# Shorten Fable and Arcee for Gemma 4 12B

`shorten_datasets.py` processes both datasets on CPU, using the Gemma tokenizer and
chat template. It downloads tokenizer/processor files and streams dataset rows, **not model
weights**. It does not require CUDA or Unsloth. It leaves the original datasets unchanged.

Sources:

- [Fable 5.1](https://huggingface.co/datasets/MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x): `full` split, `messages` format.
- [Arcee reasoning](https://huggingface.co/datasets/arcee-ai/reasoning-sharegpt): `train` split, ShareGPT `conversations` format.

With your Python environment active, install preprocessing dependencies if needed:

```bash
python -m pip install -U transformers datasets
```

Use an environment with current Gemma 4 processor support. If its processor reports an optional
dependency missing, install the named dependency in that environment. For training, follow the
Python/Unsloth environment setup in README.md instead of changing the system environment.

Process **both entire datasets** into windows of at most 1,024 tokens:

```bash
cd gemma4-fable51-finetune
python shorten_datasets.py --context 1024 --overlap 128 --output data/short-context-1024
```

For a smaller context:

```bash
python shorten_datasets.py --context 512 --overlap 64 --output data/short-context-512
```

1,024 is a conservative starting context for your RTX 5070, not a guarantee that training fits.
Training memory also depends on Gemma weights, LoRA rank, batch size and the training library.
Preprocessing itself runs on CPU, so it can run while the NVIDIA driver is unavailable.

Try a small preview first in its own directory:

```bash
python shorten_datasets.py --max-rows 100 --output data/preview
```

`--max-rows` applies to each source; omit it or set it to zero for all rows. `--fable-split lite`
selects the smaller Fable split. Local JSONL inputs are also supported with `--fable-file PATH`
and `--arcee-file PATH`. Each line must contain a complete source conversation.

Output layout:

```text
data/short-context-1024/
  manifest.json
  tokenizer/
  fable/train.jsonl
  fable/validation.jsonl
  fable/rejected.jsonl
  arcee/train.jsonl
  arcee/validation.jsonl
  arcee/rejected.jsonl
```

Every output example contains `input_ids`, `attention_mask`, `labels` and source/trace metadata.
The limit includes the chat template and special tokens. Assistant output tokens have labels;
prompts, tool results and overlapping context use `-100` to exclude them from loss.
Long conversations are split, preserving later reasoning/final-answer tokens instead of
keeping only the beginning. Windows can start mid-turn and have only local context; they are
continuation training examples, not complete independent chats. Shortening cannot preserve
the full dependencies of a 50k-token coding trace within 1k tokens.

Conversations are assigned to train or validation **before** windowing. Duplicate normalized
conversations are removed across both sources. Output includes no cross-conversation packing.
The manifest reports original lengths, long-trace counts, rejected rows, maximum output length,
and window counts. Review `rejected.jsonl` if either dataset has rejected rows. Unsupported
multimodal blocks are rejected; tool traces are retained as text, as described in README.md.
No output directory is overwritten. A failed run has no completion manifest; retry into a new
directory. Pin `--tokenizer-revision`, `--fable-revision` and `--arcee-revision` to Hub commits
if reproducibility matters; the default is `main`.

Train directly from both prepared datasets:

```bash
python finetune_gemma.py train --prepared-dir data/short-context-1024 --context 1024 --output outputs/mixed-1024
```

For the 512-token version, use `--prepared-dir data/short-context-512 --context 512`.
The trainer loads the existing token IDs/labels without adding templates or re-tokenizing.
It verifies that the tokenizer model/revision matches and that the configured context is large
enough. Model weights still download for training. This combines the windows from both datasets
at their natural proportions; longer Fable traces produce more windows. Default training is
200 optimizer steps; add `--steps -1 --epochs 1` to train for a complete epoch.

These files are for the supplied trainer or another trainer that accepts pretokenized examples
and preserves labels. They are **not ShareGPT/chat JSONL files to upload to a generic dataset UI**;
re-tokenizing or adding another chat template can invalidate the limit and masking.

Offline checks (synthetic data; no downloads):

```bash
python3 -m unittest -v test_finetune_gemma.py test_shorten_datasets.py
```

Real dataset processing and GPU training have not been run during script creation.
