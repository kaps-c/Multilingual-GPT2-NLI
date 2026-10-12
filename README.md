# Multilingual GPT-2 NLI

A custom GPT-2 and AdamW implementation, reconstructed from the accompanying
course-project PDF. Reusable Python modules own the experiments; the notebook
displays saved results without downloading models or running training.

Open [`default_final_project.ipynb`](default_final_project.ipynb) for the showcase.
The first measured run uses 1,024 training examples and achieves 32.8% accuracy
on 256 held-out examples, below its 34.4% majority baseline. This is a working
pipeline demonstration, not a full-data benchmark claim. Historical PDF results
are clearly separated. See [the reconstruction log](docs/reconstruction.md) for
source references, corrections, and unfinished reproduction work.

## Setup

Use Python 3.11 or later and a project environment. On Windows:

```powershell
python -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install -e ".[data,report,analysis,dev]"
```

Install a CUDA-compatible PyTorch build for your GPU when needed. The verified
local environment uses Python 3.11.15, PyTorch 2.6.0+cu124, Transformers 4.57.6,
and an RTX 3070 Ti Laptop GPU with 8 GB VRAM. It reuses the existing `ai` conda
environment's packages through a project virtual environment; other machines
should install the declared dependencies into a clean environment.

Download official GPT-2 weights and the tokenizer once:

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download('openai-community/gpt2', allow_patterns=['config.json','model.safetensors','vocab.json','merges.txt','tokenizer.json','tokenizer_config.json'], local_dir='artifacts/gpt2')"
python -m multilingual_gpt2 download --languages en
```

## Run a small experiment

```powershell
python -m multilingual_gpt2 train --device cuda --mixed-precision --train-limit 1024 --dev-limit 128 --test-limit 256 --max-length 128 --output results/my-run --checkpoint checkpoints/my-run.pt
python -m multilingual_gpt2 evaluate --device cuda --checkpoint checkpoints/my-run.pt --data-file data/xnli/en/test.jsonl --limit 256 --output results/my-run/reloaded-evaluation.json
```

Use `--device cpu` without `--mixed-precision` for CPU runs. `--tiny` selects a
small randomly initialized model for integration checks, not pretrained quality.
Choose new output and checkpoint paths for each independent experiment.
To reduce GPU memory, use `--batch-size 1 --accumulation-steps 4`; the effective
update batch is then four examples. Full runs omit sample limits.

Toy pretraining uses the included corpus:

```powershell
python -m multilingual_gpt2 pretrain --device cuda --epochs 3 --output results/pretrain --checkpoint checkpoints/pretrain.pt
```

The `train` command also accepts multiple `--languages`, a `--train-fraction`,
and an `--eval-fraction`. Each invocation initializes independently. Train/dev/test
JSONL files live under `data/xnli/<language>/`; each record has a premise,
hypothesis, named label, language, and source identifier. Public dataset numeric
labels are converted to names before applying the project's label order:
entailment = 0, contradiction = 1, neutral = 2.

## Architecture and checks

Data preparation produces masked token batches. The custom model returns hidden
states; tied embedding weights produce next-token logits. Shared training updates
the model, selects a checkpoint using development accuracy, then reloads it for
test evaluation. Checkpoints carry architecture and run settings. Imports never
start training or download data.

```powershell
python -m pytest -q
python -m ruff check multilingual_gpt2 tests
python -m ruff format --check multilingual_gpt2 tests
```

In a restricted Windows session, give pytest a writable temporary directory:
`python -m pytest -q --basetemp=artifacts/pytest -p no:cacheprovider`.

Weights, datasets, environment files, caches, and bulky artifacts are ignored.
Compact measured results are committed so the notebook can display them on a
machine without GPU access. The original notebook is recoverable from Git history;
the source PDF and MIT license are preserved.
