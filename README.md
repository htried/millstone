# The `MillStone` Benchmark

The **`MillStone` Benchmark** is a framework and dataset for evaluating large language models (LLMs) on social issues in the presence of evidence. It provides a pipeline for scraping, paraphrasing, prompt generation, and batch evaluation of LLMs on a wide range of controversial topics, with a focus on evidence use.

## Overview

MillStone enables systematic benchmarking of LLMs by:

- **Scraping** structured pro/con arguments, quotes, and sources from Encyclopedia Britannica's ProCon.org for dozens of social issues.
- **Generating paraphrases** of debate questions to test model robustness to rewording and framing.
- **Creating prompts** for LLMs, with varying levels of evidence and framing, to probe model decision-making.
- **Running batch predictions** on cloud LLM APIs (Google, Anthropic, Meta) and collecting results for analysis.

## Directory Structure

- `data/` — JSON files for each debate topic, containing the question, pro/con arguments, and sources.
- `notebooks/` — Jupyter notebooks for data processing and visualization.
- `predictions/` — Model prediction outputs in JSONL format.
- `prompts/` — Generated prompts for each model and experiment.
- `results/` — CSVs summarizing model performance.
- `scripts/` — Main pipeline scripts (see below).
- `utils/` — Utility functions and prompt templates.

## Pipeline Scripts

### 0. Set-up

Download uv and from the base of this repo run the following to create the environment:

```sh
uv init
uv add -r requirements.txt
uv sync
```

Set up your `.env` file with your Google Cloud `PROJECT_ID` and `GEMINI_API_KEY`.

Examples of generated datasets for the tested models are included in this git repo (meaning you could just upload them to whichever batch prediction service you like/run script number 4) in the `/prompts` directory; or you can regenerate the entire data pipeline by following the steps below.

### 1. Scraping ProCon.org

**`scripts/01_scrape_procon.py`**

- Scrapes debate topics from ProCon.org using URLs in `utils/procon-links.txt`.
- Extracts the main question, pro/con arguments, quotes, and footnoted sources.
- Outputs a JSON file per topic in `data/`.

**Usage:**
```bash
python scripts/01_scrape_procon.py
# Optional: --sample to scrape only 5 topics for testing
```

---

### 2. Generating Paraphrases

**`scripts/02_generate_paraphrases.py`**

- Uses Google Gemini to generate multiple neutral paraphrases of each debate question.
- Adds paraphrases to each topic's JSON file in `data/`.

**Usage:**
```bash
python scripts/02_generate_paraphrases.py
# Optional: --sample to paraphrase only 5 topics
```

---

### 3. Generating Prompts

**`scripts/03_generate_dataset.py`**

- Creates prompts for LLMs using the paraphrased questions.
- Prompts vary in evidence provided (pro, con, both, 75/25 splits, or none) and in framing.
- Supports Anthropic, OpenAI, and Google model APIs.
- Outputs prompts as JSONL files in `prompts/` (or uploads to GCS).

**Usage:**
```bash
python scripts/03_generate_dataset.py --model <model_name> --N <num_trials> [--save_local] [--sample]
# model_name: gemini-2.0-flash, claude-3.5-haiku, claude-opus-4, llama-3.1-8b, llama-3.1-405b
# Optional: --save_local saves the jsonl file locally (omitting saves to GCS), --sample generates prompts for only 5 topics
```

---

### 4. Batch Model Predictions

**`scripts/04_batch_predictions.py`**

- Submits generated prompts to the selected LLM in batch mode (using Google Vertex AI, etc.).
- Monitors job status and outputs predictions to GCS.

**Usage:**
```bash
python scripts/04_batch_predictions.py --model <model_name> [--sample]
# Optional: --sample indicates that predictions should come from gcs://<BUCKET>/prompt_sample.jsonl
```

---

## Data Format

Each topic JSON in `data/` contains:
```json
{
  "question": "Should X be legal?",
  "pros": ["Pro argument 1...", ...],
  "cons": ["Con argument 1...", ...],
  "sources": {"[1]": "Citation text", ...},
  "paraphrases": [
    {"question": "Paraphrased question...", "orientation": "pro"},
    ...
  ]
}
```

## Requirements

- Python 3.10+
- See `requirements.txt` for dependencies (BeautifulSoup, requests, google-cloud-storage, etc.)
- Google Cloud credentials (for Gemini, Vertex AI, and GCS access)
- `.env` file with `PROJECT_ID` and any required API keys

## Customization

- Add or edit debate URLs in `utils/procon-links.txt` to expand the dataset.
- Modify prompt templates in `utils/prompts.py` for custom experiments.

## Results

- Model predictions are saved in `predictions/` and summarized in `results/`.
- Use the notebooks in `notebooks/` for further analysis and visualization.

## License

See [LICENSE](LICENSE).

---

**Contact:**  
For questions or contributions, please open an issue or pull request.

---

Let me know if you want to include example outputs, more details on the data format, or instructions for adding new models!