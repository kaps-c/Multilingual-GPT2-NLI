# Ubiquitous language

Shared terminology for the reconstructed implementation, experiments, and reports.

| Term | Meaning in this project |
| --- | --- |
| GPT-2 | The project's custom decoder-only transformer, compatible with official GPT-2 weights. |
| Tokenizer | Converts text to token identifiers and identifiers back to text. |
| Token | A tokenizer unit, which may represent only part of a word. |
| Attention mask | Identifies actual input tokens and excludes padding from attention. |
| Causal mask | Prevents a token from attending to future tokens. |
| Label mask | Uses -100 to exclude prompt and padding positions from training loss. |
| AdamW | The custom adaptive optimizer with decoupled weight decay. |
| Pretraining | Next-token training on the local text corpus. |
| Fine-tuning | Updates pretrained model weights on a downstream task. |
| NLI | Natural language inference: classify a premise/hypothesis pair. |
| XNLI | Cross-lingual natural language inference benchmark and its language-specific splits. |
| Premise | The statement treated as given in an NLI example. |
| Hypothesis | The statement evaluated against the premise. |
| Entailment | The premise supports the hypothesis; project label 0. |
| Contradiction | The premise conflicts with the hypothesis; project label 1. Source data may call it contradictory. |
| Neutral | The premise neither entails nor contradicts the hypothesis; project label 2. |
| Split | Separate training, development, or held-out test examples. |
| Checkpoint | Saved model weights plus model configuration and training metadata. |
| Zero-shot transfer | Evaluating an English NLI model in another language without target-language fine-tuning. |
| Fertility | Total tokenizer subwords divided by estimated words; whitespace word counts are only an approximation. |
| Per-language model | A model independently fine-tuned from pretrained weights for one language. |
| Combined-language model | One model fine-tuned on examples from all selected languages. |
| Showcase | A small, actually executed experiment presented in the thin results notebook. |
| Historical result | A result reported in the PDF, not claimed as newly reproduced. |
| Reproduced result | A measured result from an executed, recorded run of the reconstructed code. |
| Reconstruction log | Committed source inventory, corrections, evidence, and reproduction status. |
| Embedding | A vector representation of text derived from model hidden states. |
| Error analysis | Inspection and categorization of wrong predictions. |
| Calibration | Agreement between predicted confidence and empirical correctness. |
| Sentiment | Negative, neutral, or positive tone classification in the extended experiments. |
| t-SNE | A two-dimensional visualization of similarities between high-dimensional embeddings; axes have no intrinsic semantic meaning. |
| Linear probe | A logistic-regression classifier trained on frozen model embeddings. |
| Pragmatic NLI | Inference about a speaker's intended tone, using paired messages and intent hypotheses. |
| Implicit care | Indirect affection or concern in the source's synthetic tone labels. |
| Disappointment | Direct negative evaluation in the source's synthetic tone labels. |
| Comparison | Benchmarking the child against others in the source's synthetic tone labels. |
| Guilt | An appeal to parental sacrifice or duty in the source's synthetic tone labels. |
| ECE | Expected calibration error: the weighted gap between confidence and observed accuracy in probability bins. |
| Multiclass Brier score | Mean sum of squared errors across the complete class-probability vector. |
| Conditional label probability | Probability renormalized over NLI's three digit labels, not over the full vocabulary. |
| Confusion matrix | Counts of gold versus predicted labels, retaining invalid model outputs in their own column. |
| Reference checkpoint | The model selected as the baseline of an explicitly matched comparison; it need not be zero-shot or unfine-tuned. |
| Compared checkpoint | The second model evaluated on the same examples as the reference checkpoint. |
