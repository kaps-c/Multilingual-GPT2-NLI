# Reconstruction log

Source: `NLP Group 21 Code_includes extended tasks.pdf` (141 pages).
Original notebook: `default_final_project.ipynb` at Git revision `54e934d`.
The PDF is preserved unchanged. Historical outputs are not newly measured results.

## Source inventory

| PDF pages | Content | Original notebook | Reconstruction status |
| --- | --- | --- | --- |
| 1–8 | tokenizer, custom GPT-2, weight mapping, custom AdamW | cells 0–12 | `model.py`, `config.py`, `optimizer.py`; reference tests pass |
| 8–10 | toy next-token pretraining and loss plot | cells 13–17 | `data.py`, `training.py`, `reporting.py`; small run passed |
| 11–16 | generation, XNLI loading, English training and evaluation | cells 18–30 | `cli.py`, `evaluation.py`; two measured showcase runs |
| 17–20 | three English hyperparameter variants | cells 31–43 | `experiments.py`; independent small runs passed |
| 21–25 | environment, 15-language zero-shot evaluation, fertility | cells 44–51 | `analyze.py`; all 15 dev/test sets downloaded, five-language check passed |
| 25–28 | independent Spanish, French, German, Swahili training | instructions only, cell 52 | `experiments.py`; four small runs passed |
| 28–32 | combined en/es/fr/de/sw training and evaluation | instructions only, cell 53 | `experiments.py`; combined small run passed |
| 32–37 | comparison tables, deltas, fertility scatter plots | absent | `analyze.py`, `plots.py`; same-split comparison executed |
| 38–55 | multilingual tweet sentiment: zero-shot, few-shot, strict parsing | absent | `sentiment.py`; 18 small evaluations completed |
| 56–62 | fertility correlations, ranks, language families | absent | `analysis.py`, `analyze.py`; small analysis executed |
| 62–72 | embeddings, t-SNE, cluster metrics, centroid distances, fine-tuning shifts | absent | `analysis.py`; paired checkpoint comparison executed |
| 72–85 | attention maps, entropy, distance, premise/hypothesis mass, model comparison | absent | `analysis.py`, `analyze.py`; shallow/deep analysis executed |
| 85–96 | label probability, entropy, confidence, calibration, error examples | absent | `evaluation.py`, `analysis.py`; analytic test and small run passed |
| 96–101 | per-language errors and taxonomy annotation export | absent | `analysis.py`, `analyze.py`; CSV exported, manual annotations left blank |
| 102–115 | synthetic tone, pragmatic NLI, style-transfer, Mandarin data generation | absent | `synthetic.py`, `tone.py`; bounded mock-client test, no paid calls |
| 115–125 | zero/few-shot tone classification and frozen-embedding probe | absent | `tone.py`; fixture validation, original JSONL unavailable |
| 125–135 | pragmatic NLI probe, confusion matrices and embeddings | absent | `tone.py`; fixture validation, original JSONL unavailable |
| 135–141 | controlled style generation and probe-based evaluation | absent | `tone.py`; fixture validation, original JSONL unavailable |

Pages 21–141 have no extractable text layer. Contact-sheet visual inspection
established this inventory; detailed recovery uses enlarged images and OCR.
Blank/export-spacer pages and installation logs contain no executable experiment.

## Corrections

Each implemented correction records its source, rationale, test, and result impact.

| ID | Source | Original behavior | Intended correction | Verification / impact |
| --- | --- | --- | --- | --- |
| C01 | notebook cell 8; PDF model section | declares pretrained MLP width as 3 × hidden width, then replaces parameter storage with 4 × weights | derive all dimensions from pretrained configuration and copy weights with shape checks | reference-model comparison and checkpoint round-trip; resolves misleading configuration |
| C02 | cell 6 | exact GELU differs from official GPT-2 activation | use the pretrained activation, defaulting to GPT-2 tanh approximation | reference-model comparison; small numerical differences expected |
| C03 | cell 25 | truncation can remove every target label | reserve label/suffix space and mask only prompt/padding | long-input supervision test; avoids undefined loss |
| C04 | cell 16 | padding contributes to pretraining loss | mask padded targets with -100 | padding-invariant loss test; corrected losses are not directly comparable |
| C05 | cells 34, 38, 42 | sequential experiments reuse already-updated model | initialize each independent experiment from original weights | runner metadata and initialization checks; results may differ |
| C06 | cell 48 and extended PDF | hard-coded Colab paths and notebook globals | explicit input/output paths and function arguments | temporary-directory integration run |
| C07 | cell 29 | initial best accuracy 0 can prevent checkpoint creation | save first finite dev result, even at 0 accuracy | checkpoint test |
| C08 | PDF 38–55 | filters invalid sentiment predictions; one variant substitutes neutral | retain every example; unknown predictions count as wrong; report coverage | invalid-output metric test; accuracy denominator changes |
| C09 | PDF 92–94 | Brier calculation scores top confidence/correctness, a valid binary calibration quantity | use explicitly named multiclass Brier for the three-class task | analytic probability fixture; a different metric, not numerically comparable |
| C10 | PDF 125–135 | row split can put hypotheses from one premise into train and dev | split by source premise identifier | disjoint-group test; avoids leakage and may lower scores |
| C11 | PDF 115–125 | tone parser sees the whole prompt and filters unknown responses | parse continuation only; retain unknowns in metrics | shared parser test and tone fixture; historical prompt scores are not comparable |
| C12 | reconstruction checkpoint review | distinct embedding/residual dropout was not serialized | store embedding dropout independently | seeded train-mode checkpoint regression; ordinary GPT-2 defaults unchanged |
| C13 | downloaded MTEB sentiment test data | labels may be numeric strings | validate and normalize both integer and string labels | mapping regression and all six-language small evaluations |
| C14 | PDF 113–114 and tone split | translated examples may share training sources | recognize `src_id`; report unseen-source metrics separately | translation-group regression; all-row agreement is not a held-out result |

## What was executed

- All ten core experiment paths completed with the small profile: toy pretraining,
  four English settings, four independent target languages, and combined training.
- Showcase: 1,024 train / 128 dev / 256 test, then 8,192 / 512 / 512.
  Test accuracy: 32.8125% and 70.703125%; larger dev accuracy: 72.0703125%.
  Saved metrics, settings, losses, and predictions are in `results/showcase*`.
- Five-language diagnostics: 24 test examples and eight development embeddings
  per language; both showcase checkpoints evaluated on matching examples.
  Includes calibration, attention segment mass, representation shifts and plots.
- Sentiment: 12 test examples × six languages × three modes. Accuracy and coverage
  are zero: the NLI checkpoint did not generate recognized sentiment labels.
- Tone workflows use a 32-row fictional integration fixture with explicit tone
  clues. Those fixture scores are not research results and are not published in
  the notebook. Original tone, style and Mandarin datasets remain unavailable.
- Synthetic generation is tested with a fake client. No remote generation calls
  were made. New generation would create a new dataset, not recover the original.

## Deliberate consolidation and remaining reproduction

Repeated notebook code, professor prompts, installation logs, failed exploratory
branches, unused pooling layers and obsolete download helpers are omitted.
One implementation covers each task; the notebook contains results and explanation.
Historical NLI tables are preserved separately in `results/historical.json`.

The reconstructed prompts are concise equivalents of the source tasks, not a
verbatim replay of every exploratory prompt. Tone hypotheses preserve the eight
source formulations; synthetic generation balances source selection across tones.
Tone splits stratify source groups by tone when that field is available.
Pretraining appends EOS and excludes padding. NLI truncation preserves both prompt
segments and the target delimiter. These choices can change the source's scores.

The full core profile preserves the published training settings, but initializes
each run independently and uses microbatches with gradient accumulation for the
8 GB GPU. Dropout randomness and reduced precision prevent numerical identity.
Small-profile overrides exercise each path, not the intended full-data comparisons.
The runner skips completed jobs; it does not resume an interrupted epoch.

Full-data benchmark runs and full-size extended evaluations remain separate,
unexecuted work. Fertility uses train data when available, otherwise explicitly
marked dev data; download all language train splits for source-sized analysis.
The PDF's synthetic results need the original 352-row core, 80 style situations,
and 80 Mandarin translations. The derived pragmatic set has eight pairs per core
message. Missing records cannot be reconstructed faithfully from sampled previews.

## Reproduction policy

Final verification: 13 behavioral tests pass; Ruff lint and 79-column formatting
checks pass for the package, tests and notebook. All 18 notebook cells execute and
validate. All ten small training paths and four tone fixture workflows complete.
`results/validation.json` records the compact validation inventory.

The final refactor removes 588 lines from obsolete root `config.py` and `utils.py`
after confirming no remaining callers. Configuration now lives in the package;
official model downloads use Transformers/Hugging Face. Style and cross-language
evaluation load only the probe's required encoder. Code comments are lowercase.
The local glossary is excluded through `.git/info/exclude` and is never staged.

- First milestone: an actual small English XNLI run on the local GPU, then a
  results-only notebook. No invented benchmark scores.
- Full runs follow small behavioral and integration checks. The original training
  data, model snapshots, random seeds, and GPU environment are not all preserved;
  numerical identity with the PDF is not promised.
- Original synthetic datasets are required to reproduce the tone experiments.
  Newly generated data is a new experiment, not the original reproduction.
- External paid API calls require credentials and an agreed budget.
- The local terminology glossary is deliberately excluded from version control.
