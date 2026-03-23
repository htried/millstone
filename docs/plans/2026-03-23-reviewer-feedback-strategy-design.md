# MillStone Reviewer Feedback Strategy: B & C

**Date:** 2026-03-23
**Goal:** Address major concerns from ARR reviewers regarding in-depth analysis (B) and methodological validity (C) without running new, large-scale experiments.

## Section 1: Statistical Rigor (Reviewer MVUm)
- **Problem:** Reviewer MVUm noted the absence of statistical significance testing (confidence intervals, p-values, variance estimates) despite running 15 trials per condition.
- **Action Plan:**
  - Create a new script or update `notebooks/process_raw_data.ipynb` to calculate variance and confidence intervals across the 15 trials for each topic/model/case.
  - Compute p-values to determine if the shifts in stance (relative to the baseline `neither` case) are statistically significant.
  - Update the CSVs in `/results` (or generate new summary files) to include these statistical metrics.
- **Expected Outcome:** Robust statistical evidence to support the claims of stance shifts, addressing the reviewer's concern about whether the differences are genuine effects or sampling noise.

## Section 2: Order Effects & Sycophancy Analysis (Reviewer Ho5a)
- **Problem:** Reviewer Ho5a suggested that the observed stance shifts might just be models agreeing with the *last* argument shown (order effects/sycophancy) rather than genuine "open-mindedness."
- **Action Plan:**
  - Parse the `/predictions` JSONL files to extract the exact order of arguments presented in each prompt.
  - Cross-reference the model's final stance with the position of the last argument (e.g., did 'pro' come last, and did the model choose 'pro'?).
  - Analyze if there is a significant recency bias across the models.
  - Add a new section/visualizations in `notebooks/visualize.ipynb` to present these findings.
- **Expected Outcome:** Data-driven analysis distinguishing between genuine persuasion and simple order effects/sycophancy, directly addressing a key critique of the benchmark's interpretation.

## Section 3: Addressing Binary Constraint Validity (Reviewers Ho5a, MVUm)
- **Problem:** Both reviewers pointed out that the binary choice constraint ("choose A or B") is not representative of real-world RAG systems, which generate nuanced text.
- **Action Plan:**
  - Draft new textual additions for the paper acknowledging this limitation.
  - Defend the binary constraint as a necessary simplification for a *benchmark* designed to measure relative susceptibility objectively across models, avoiding the complexities of parsing nuanced textual stances at scale.
  - Add a discussion on how future work should extend this to open-ended generation evaluation.
- **Expected Outcome:** Improved paper framing that contextualizes the methodology, acknowledges its limitations, and clarifies the benchmark's specific goals.