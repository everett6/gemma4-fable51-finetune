# Gemma 4 12B / Fable 5.1 text fine-tuning

`finetune_gemma.py` trains a text QLoRA adapter and can export a merged GGUF for LM Studio.
The default is a **pilot run: 100 conversations, 200 optimizer steps**, not the whole dataset.

To preprocess **both Fable and Arcee reasoning** on CPU with a strict token limit, see
[SHORT_CONTEXT.md](SHORT_CONTEXT.md) and run `shorten_datasets.py`. The training script accepts
its output with `--prepared-dir`.

Your local files are in:

```text
~/.lmstudio/models/lmstudio-community/gemma-4-12B-it-QAT-GGUF/
```

The Q4_0 GGUF is an inference artifact. This workflow downloads the corresponding
[unquantized QAT checkpoint](https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-unquantized)
and trains adapters on a 4-bit representation of those weights. It does not modify the
original GGUF. This is QLoRA training, not a new quantization-aware training pass.
The final export is quantized again; fine-tuning may change the original QAT quantization quality.

Dataset: [MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x](https://huggingface.co/datasets/MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x).
Its card identifies `messages` conversations and two splits: `lite` (5k) and `full` (10k).
The script defaults to `lite`. Teacher attribution is the dataset publisher's claim.

## Set up once

Use a working NVIDIA driver and Python 3.10–3.13. Python 3.14 is outside this script's supported training environment. Ensure `nvidia-smi` works before training.
Close loaded LM Studio models to free GPU memory before training.

Following the [Unsloth Core installation guide](https://unsloth.ai/docs/get-started/install/pip-install),
with `uv` available:

```bash
cd gemma4-fable51-finetune
uv venv .venv --python 3.13
source .venv/bin/activate
uv pip install unsloth datasets transformers accelerate peft bitsandbytes --torch-backend=auto
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

Use a current CUDA-enabled PyTorch build that supports the RTX 50 series. Package installation
has not been run here; the commands create a separate environment. If Hugging Face requests
authentication, log in with `hf auth login` and grant access on the model page as needed.

Training downloads the model and streams the selected dataset into the Hugging Face cache.
Budget tens of GB of free disk for weights, tokenized examples, checkpoints and temporary exports.
For GGUF merging, allow substantially more disk and free system RAM; 32 GB of system RAM
may be tight. Export on a larger machine if it runs out of memory.

## Train

```bash
python finetune_gemma.py train
```

The defaults use 1024-token windows, batch size 1, rank 8, and gradient accumulation 8.
They are conservative starting settings for the local GPU, **not a verified fit guarantee**.
If GPU memory runs out, use a fresh output directory and `--context 512 --rank 4`.

For a short pipeline check before the pilot:

```bash
python finetune_gemma.py train --steps 2 --output outputs/smoke
```

To prepare and inspect tokenized data without loading model weights (still requires the
training environment and CUDA preflight):

```bash
python finetune_gemma.py train --prepare-only --output outputs/preview
```

For the entire full split and one epoch:

```bash
python finetune_gemma.py train --split full --max-traces 0 --steps -1 --epochs 1 --output outputs/full
```

The full dataset is large and may take a long time. `--steps` is the number of optimizer
updates; each update accumulates eight windows by default. The pilot won't consume every
prepared window. For reproducible inputs, set `--model-revision` and `--dataset-revision`
to Hub commit IDs; by default they follow `main`.

Resume an interrupted run with identical data/model/settings:

```bash
python finetune_gemma.py train --resume outputs/gemma4-fable51/checkpoints/checkpoint-50
```

Checkpoints are saved every 50 steps and only the latest two are retained. Data preparation
is repeated on resume, so keep the same revisions, seed, context and trace count.
Final adapter: `outputs/gemma4-fable51/adapter/`.
Final validation loss and run settings: `outputs/gemma4-fable51/run.json`.
Holdout loss measures imitation of this dataset; compare real coding tasks against your original
model before treating the fine-tune as an improvement.

## Export for LM Studio

```bash
python finetune_gemma.py export
```

This invokes [Unsloth's GGUF exporter](https://unsloth.ai/docs/models/gemma-4/train).
It may download/build llama.cpp and needs build tools such as a compiler and CMake.
Output: `outputs/gemma4-fable51-gguf/`, using Q4_K_M by default. Use
`--quantization q4_0` to match the original quantization type.
Import the generated **language-model GGUF** into LM Studio as a separate model and use its
embedded Gemma 4 chat template. This recipe trains and validates text only. Generated or original
multimodal projectors have not been validated with the adapter; start with text inference.
For a different training output, point `--adapter` at its `adapter/` directory.

## Data handling and offline checks

Long traces are split into overlapping token windows rather than silently cutting each trace
off after its first 1024 tokens. Overlap supplies context but receives no duplicate loss.
Only assistant outputs receive loss; user/system prompts and tool results are masked.
Whole conversations are assigned deterministically to train or validation before splitting,
and identical normalized traces are deduplicated. Windows can begin mid-turn; this teaches local
continuations and cannot reproduce the full long-context tasks at a 1024-token context.

Text and thinking blocks are retained, with `<think>` spans mapped to Gemma channel markers.
Tool calls are serialized as text and tool results are presented as labeled user turns.
This teaches textual coding traces, **not Gemma-native tool-call protocol compatibility**.
Unsupported multimodal rows are rejected and counted. Review the counts before a full run.

```bash
python3 -m unittest -v test_finetune_gemma.py
python3 finetune_gemma.py --help
```

The offline checks cover assistant loss masking across later user/tool turns, overlapping-window
coverage, thought/tool preservation and malformed-row rejection. GPU training, downloads and
GGUF export were not executed during authoring.
