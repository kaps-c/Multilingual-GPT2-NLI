# Reconstruction log

Source: `NLP Group 21 Code_includes extended tasks.pdf` (141 pages).
Original notebook: `default_final_project.ipynb` in the parent Git revision.
The PDF is preserved unchanged. Historical outputs are not newly measured results.

## Source inventory

| PDF pages | Content | Original notebook | Reconstruction status |
| --- | --- | --- | --- |
| 1–8 | tokenizer, custom GPT-2, weight mapping, custom AdamW | cells 0–12 | extraction in progress |
| 8–10 | toy next-token pretraining and loss plot | cells 13–17 | extraction pending |
| 11–16 | generation, XNLI loading, English training and evaluation | cells 18–30 | showcase in progress |
| 17–20 | three English hyperparameter variants | cells 31–43 | deduplication pending |
| 21–25 | environment, 15-language zero-shot evaluation, fertility | cells 44–51 | reconstruction pending |
| 25–28 | independent Spanish, French, German, Swahili training | instructions only, cell 52 | missing code |
| 28–32 | combined en/es/fr/de/sw training and evaluation | instructions only, cell 53 | missing code |
| 32–37 | comparison tables, deltas, fertility scatter plots | absent | missing code |
| 38–55 | multilingual tweet sentiment: zero-shot, few-shot, strict parsing, debugging | absent | missing code; failed exploratory attempts must not become defaults |
| 56–62 | fertility correlations, ranks, language families | absent | missing code |
| 62–72 | embeddings, t-SNE, cluster metrics, centroid distances, fine-tuning shifts | absent | missing code |
| 72–85 | attention maps, entropy, distance, premise/hypothesis mass, model comparison | absent | missing code |
| 85–96 | label probability, entropy, confidence, calibration, error examples | absent | missing code |
| 96–101 | per-language errors and taxonomy annotation export | absent | missing code |
| 102–115 | synthetic tone, pragmatic NLI, style-transfer, Mandarin data generation | absent | original JSONL and API access unavailable |
| 115–125 | zero/few-shot tone classification and frozen-embedding probe | absent | original JSONL unavailable |
| 125–135 | pragmatic NLI probe, confusion matrices and embeddings | absent | original JSONL unavailable |
| 135–141 | controlled style generation and probe-based evaluation | absent | original JSONL and probe unavailable |

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
| C09 | PDF 92–94 | Brier calculation uses only top confidence/correctness | report multiclass Brier using all three label probabilities | analytic probability fixture; metric explicitly differs |
| C10 | PDF 125–135 | row split can put hypotheses from one premise into train and dev | split by source premise identifier | disjoint-group test; avoids leakage and may lower scores |

## Reproduction policy

- First milestone: an actual small English XNLI run on the local GPU, then a
  results-only notebook. No invented benchmark scores.
- Full runs follow small behavioral and integration checks. The original training
  data, model snapshots, random seeds, and GPU environment are not all preserved;
  numerical identity with the PDF is not promised.
- Original synthetic datasets are required to reproduce the tone experiments.
  Newly generated data is a new experiment, not the original reproduction.
- External paid API calls require credentials and an agreed budget.
- The local terminology glossary is deliberately excluded from version control.
