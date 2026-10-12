# Multilingual GPT-2 NLI

A custom GPT-2 and AdamW implementation, reconstructed from the accompanying
course-project PDF. Reusable Python modules own the experiments; the notebook
displays saved results without downloading models or running training.

Open [`default_final_project.ipynb`](default_final_project.ipynb) for the showcase.
The 8,192-example run achieves **70.7% accuracy on 512 held-out examples**.
The earlier 1,024-example run achieves 32.8% on 256 examples, below its 34.4%
majority baseline; both remain visible. These are subset runs, not full-data
benchmark claims. Historical PDF results
are clearly separated. See [the reconstruction log](docs/reconstruction.md) for
source references, corrections, and unfinished reproduction work.

The notebook has saved outputs in every code cell and 12 figures: training loss,
confusion counts, fertility, embeddings, centroid distances, checkpoint comparisons,
calibration, attention, and uncertainty. Figures are regenerated from compact
committed results through the same plotting code used by the analysis CLI.
No local checkpoints or datasets are needed to rerun the notebook.

**Current stage:** workflow reconstruction and visual verification are complete.
Full experimental reproduction is incomplete: full-data runs are pending, and
the original synthetic tone/style/Mandarin datasets are missing. Fixture checks
verify execution only; their scores are not presented as research findings.

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
and a `--dev-fraction`. Each invocation initializes independently. Train/dev/test
JSONL files live under `data/xnli/<language>/`; each record has a premise,
hypothesis, named label, language, and source identifier. Public dataset numeric
labels are converted to names before applying the project's label order:
entailment = 0, contradiction = 1, neutral = 2.

## Architecture and checks

Recovered extended workflows use the same model and data interfaces:

```powershell
python -m multilingual_gpt2 download --languages en es fr de sw
python -m multilingual_gpt2.experiments --profile small --device cuda
python -m multilingual_gpt2.analyze --checkpoint checkpoints/showcase-8192.pt --languages en es fr de sw --limit 24 --embedding-limit 8 --device cuda
python -m multilingual_gpt2.sentiment --download
python -m multilingual_gpt2.sentiment --checkpoint checkpoints/showcase-8192.pt --limit 12 --device cuda
```

`experiments --profile full` runs the source-sized training configurations. Use
`--only english` or another listed job to select a run. Full runs can take many
hours; completed jobs are skipped, interrupted epochs restart. Logs and commands
are saved under `results/experiments/<profile>/<job>/`.

`analyze` defaults to all 15 languages. Download their dev/test splits first with
`download --languages en ar bg de el es fr hi ru sw th tr ur vi zh --splits dev test`.
Omit `--limit` for the full test sets. Add `--compare-checkpoints <path> <path>`
to compare accuracy, embeddings and attention on matching examples. Outputs include
figures, probabilities, calibration, family summaries, ranks, fertility correlations,
and an error CSV with empty fields for manual annotation. Fertility falls back to
dev data when a language's training data is absent; the report records this.

Tone workflows require the original JSONL files, which are not in this repository:

```powershell
python -m multilingual_gpt2.tone tone --data data/tone/core.jsonl --checkpoint checkpoints/showcase-8192.pt --output artifacts/tone --device cuda
python -m multilingual_gpt2.tone derive --data data/tone/core.jsonl --output artifacts/pragmatic
python -m multilingual_gpt2.tone pragmatic --data artifacts/pragmatic/pragmatic.jsonl --checkpoint checkpoints/showcase-8192.pt --output artifacts/pragmatic --device cuda
python -m multilingual_gpt2.tone style --data data/tone/style.jsonl --probe artifacts/tone/probe.pkl --output artifacts/style --device cuda
python -m multilingual_gpt2.tone cross-lingual --data data/tone/mandarin.jsonl --probe artifacts/tone/probe.pkl --output artifacts/mandarin --device cuda
```

Core rows need `id`, `text`, and `tone` (`implicit_care`, `disappointment`,
`comparison`, or `guilt`). Style rows need `id` or `situation_id` and
`neutral_context`. Mandarin rows need `id`, `text`, `tone`, and `source_id` or
`src_id` referencing the original core message. Probe files use joblib and should
only be loaded from trusted local runs. Style agreement measures classifier
agreement with a requested tone; it does not measure preservation of meaning.

New synthetic generation is optional: install `.[synthetic]`, provide
`ANTHROPIC_API_KEY`, and invoke `python -m multilingual_gpt2.synthetic --help`.
Every invocation requires an explicit model, new output path, and `--max-calls`
budget. It is never triggered by training or notebook execution. Newly generated
records are labeled as a new dataset, not a reproduction of the PDF's data.

## Implementation and verification

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
