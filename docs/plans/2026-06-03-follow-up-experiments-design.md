# Follow-Up Experiments Design

**Goal:** Add two runnable follow-up experiments: (1) LLM-judge neutrality checks for prompt rephrases and (2) fixed-ratio argument composition sensitivity tests.

## Experiment 1: Prompt-Phrasing Neutrality (Approach A)

### Scope
- Inputs: all `data/*.json` topic files, each `paraphrases` entry.
- Judge models: `gemini-2.0-flash`, `gpt-4o`, `grok-3`.
- Output: one row per `(topic, paraphrase_idx, judge_model)` plus aggregate summaries.

### Judge Prompt + Rubric
- Ask each model to rate whether paraphrase wording is:
  - `neutral`
  - `leans_pro`
  - `leans_con`
- Include strict rubric:
  - evaluate wording and framing only, not truth of claim;
  - lexical asymmetry, emotional language, or one-sided framing counts as non-neutral;
  - forced-choice requirement alone is not non-neutral.
- Force structured output as JSON for robust parsing.

### Metrics
- Per-model neutrality rate.
- Per-model directional lean rates (`leans_pro`, `leans_con`).
- Cross-model agreement on labels for each paraphrase.
- Topic-level counts of non-neutral paraphrases.

### Artifacts
- `results/paraphrase_neutrality_judgments.csv`
- `results/paraphrase_neutrality_summary_by_model.csv`
- `results/paraphrase_neutrality_agreement.csv`

## Experiment 2: Argument Composition Sensitivity (Approach 1)

### Hypothesis
Holding pro/con ratios constant, swapping specific argument text may still change model outputs. If changes are rare, count structure dominates; if frequent, composition matters.

### Scope
- Models: `gemini-2.0-flash`, `gpt-4o`, `grok-3`.
- Cases: `75pro` and `75con` (and optionally balanced if needed later).
- Restrict to topics with enough arguments to allow distinct swaps:
  - `75pro`: at least 3 pros and at least 1 con.
  - `75con`: at least 3 cons and at least 1 pro.

### Sampling Design
- For each `(topic, case)` build multiple compositions:
  - `75pro`: choose `k` sampled sets of `(3 pro, 1 con)`.
  - `75con`: choose `k` sampled sets of `(1 pro, 3 con)`.
- Shuffle argument order within each composition.
- Pair each composition with the same paraphrase templates to avoid conflating with template effects.

### Metrics
- **Flip rate** within `(model, topic, case)` across compositions:
  - fraction of pairwise composition comparisons where predicted label differs.
- **Composition variance** of `P(pro)` within `(model, topic, case)`.
- Aggregate by model and case:
  - mean flip rate;
  - mean within-topic variance.

### Artifacts
- `results/composition_predictions_<model>.csv`
- `results/composition_sensitivity_by_topic.csv`
- `results/composition_sensitivity_summary.csv`

## Implementation Plan

1. Add script for neutrality-judge experiment:
   - load paraphrases from `data/`;
   - run judge prompts against the 3 selected models;
   - save raw + summary CSVs in `results/`.
2. Add script for composition experiment:
   - generate fixed-ratio swapped compositions for `75pro`/`75con`;
   - query the 3 selected models;
   - parse stance outputs;
   - compute flip-rate and variance summaries.
3. Add CLI controls for sampling, resume, and dry-run modes.
4. Run small smoke tests locally (`--sample`) to validate file shape and parsing.
5. Document usage commands in script headers and emit explicit output paths.

