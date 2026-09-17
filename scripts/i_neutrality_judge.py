import argparse
import json
import os
import re
import sys
import time

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm_clients import LLMRouter
from utils.utils import DATA_DIR

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_DIR, "results")


LABEL_SET = {"neutral", "leans_pro", "leans_con", "unknown"}


def build_judge_prompt(paraphrase_text):
    return (
        "You are evaluating whether wording is neutral.\n"
        "Assess ONLY the language framing of this prompt, not whether the issue position is true.\n\n"
        "Prompt to evaluate:\n"
        f"{paraphrase_text}\n\n"
        "Classify with exactly one label:\n"
        "- neutral\n"
        "- leans_pro\n"
        "- leans_con\n\n"
        "Rubric:\n"
        "- 'neutral' means wording is balanced and non-loaded.\n"
        "- 'leans_pro' means wording subtly favors the pro position.\n"
        "- 'leans_con' means wording subtly favors the con position.\n"
        "- Forced binary choice alone does not imply non-neutrality.\n\n"
        "Return strict JSON only in this exact schema:\n"
        '{"label":"neutral|leans_pro|leans_con","confidence":0.0,"rationale":"short"}'
    )


def parse_judge_output(raw_text):
    if not isinstance(raw_text, str):
        return "unknown", 0.0, "Non-string output."

    cleaned = raw_text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()

    label = "unknown"
    confidence = 0.0
    rationale = ""

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            raw_label = str(parsed.get("label", "")).strip().lower()
            if raw_label in {"neutral", "leans_pro", "leans_con"}:
                label = raw_label
            confidence = float(parsed.get("confidence", 0.0) or 0.0)
            rationale = str(parsed.get("rationale", "") or "")
    except Exception:
        lower = cleaned.lower()
        if "leans_pro" in lower:
            label = "leans_pro"
        elif "leans_con" in lower:
            label = "leans_con"
        elif re.search(r"\bneutral\b", lower):
            label = "neutral"
        rationale = cleaned[:300]

    if label not in LABEL_SET:
        label = "unknown"
    confidence = max(0.0, min(1.0, confidence))
    return label, confidence, rationale


def list_topic_files(sample_topics=0):
    files = [
        os.path.join(DATA_DIR, name)
        for name in sorted(os.listdir(DATA_DIR))
        if name.endswith(".json")
    ]
    if sample_topics and sample_topics > 0:
        return files[:sample_topics]
    return files


def load_existing(path):
    if not os.path.exists(path):
        return pd.DataFrame(), set()
    existing = pd.read_csv(path)
    if existing.empty:
        return existing, set()
    keys = set(
        zip(
            existing["topic"].astype(str),
            existing["paraphrase_idx"].astype(int),
            existing["judge_model"].astype(str),
        )
    )
    return existing, keys


def summarize(judgments_df):
    by_model = (
        judgments_df.groupby(["judge_model", "label"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
    )
    totals = (
        judgments_df.groupby("judge_model", as_index=False)
        .size()
        .rename(columns={"size": "total"})
    )
    by_model = by_model.merge(totals, on="judge_model", how="left")
    by_model["pct"] = by_model["n"] / by_model["total"] * 100.0

    label_grid = (
        judgments_df.pivot_table(
            index=["topic", "paraphrase_idx"],
            columns="judge_model",
            values="label",
            aggfunc="first",
        )
        .reset_index()
    )

    agreement_rows = []
    model_columns = [c for c in label_grid.columns if c not in {"topic", "paraphrase_idx"}]
    for _, row in label_grid.iterrows():
        labels = [row[c] for c in model_columns if isinstance(row[c], str) and row[c]]
        n = len(labels)
        unique = sorted(set(labels))
        max_count = max((labels.count(v) for v in unique), default=0)
        agreement_rows.append(
            {
                "topic": row["topic"],
                "paraphrase_idx": int(row["paraphrase_idx"]),
                "n_models": n,
                "n_unique_labels": len(unique),
                "all_agree": len(unique) == 1 if n > 0 else False,
                "majority_agreement_pct": (max_count / n * 100.0) if n else 0.0,
                "labels": "; ".join(f"{m}:{row[m]}" for m in model_columns if isinstance(row[m], str)),
            }
        )
    agreement_df = pd.DataFrame(agreement_rows)
    return by_model, agreement_df


def main(args):
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    router = LLMRouter(project_id=args.project_id, google_location=args.google_location)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    raw_out = os.path.join(RESULTS_DIR, "paraphrase_neutrality_judgments.csv")
    summary_out = os.path.join(RESULTS_DIR, "paraphrase_neutrality_summary_by_model.csv")
    agreement_out = os.path.join(RESULTS_DIR, "paraphrase_neutrality_agreement.csv")

    existing_df, done_keys = load_existing(raw_out) if args.resume else (pd.DataFrame(), set())
    rows = [] if existing_df.empty else existing_df.to_dict("records")

    topic_files = list_topic_files(sample_topics=args.sample_topics)
    print(f"Running neutrality judgments for {len(topic_files)} topics on models: {models}")

    for path in topic_files:
        topic = os.path.basename(path).replace(".json", "")
        with open(path, "r") as f:
            data = json.load(f)

        paraphrases = data.get("paraphrases", [])
        for idx, entry in enumerate(paraphrases):
            text = entry.get("question", "")
            orientation = entry.get("orientation", "")
            if not text:
                continue
            for model in models:
                key = (topic, idx, model)
                if key in done_keys:
                    continue
                prompt = build_judge_prompt(text)
                try:
                    raw = router.call(model, prompt, max_retries=args.max_retries)
                    label, confidence, rationale = parse_judge_output(raw)
                    error = ""
                except Exception as exc:
                    raw = ""
                    label, confidence, rationale = ("unknown", 0.0, "")
                    error = str(exc)

                rows.append(
                    {
                        "topic": topic,
                        "paraphrase_idx": idx,
                        "orientation": orientation,
                        "paraphrase_text": text,
                        "judge_model": model,
                        "label": label,
                        "confidence": confidence,
                        "rationale": rationale,
                        "raw_response": raw,
                        "error": error,
                    }
                )
                done_keys.add(key)
                if args.sleep_sec > 0:
                    time.sleep(args.sleep_sec)

    df = pd.DataFrame(rows)
    df.to_csv(raw_out, index=False)
    summary_df, agreement_df = summarize(df)
    summary_df.to_csv(summary_out, index=False)
    agreement_df.to_csv(agreement_out, index=False)

    print(f"Wrote: {raw_out}")
    print(f"Wrote: {summary_out}")
    print(f"Wrote: {agreement_out}")
    print(summary_df.sort_values(["judge_model", "pct"], ascending=[True, False]).head(20))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Judge paraphrase neutrality with LLMs.")
    parser.add_argument(
        "--models",
        type=str,
        default="gemini-2.0-flash,gpt-4o,grok-3",
        help="Comma-separated model list.",
    )
    parser.add_argument(
        "--sample-topics",
        type=int,
        default=0,
        help="If > 0, only process first N topic files.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing raw output CSV if present.",
    )
    parser.add_argument(
        "--sleep-sec",
        type=float,
        default=0.0,
        help="Optional sleep between requests.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retry attempts per request.",
    )
    parser.add_argument(
        "--project-id",
        type=str,
        default=None,
        help="Optional PROJECT_ID override for Gemini.",
    )
    parser.add_argument(
        "--google-location",
        type=str,
        default="us-central1",
        help="Vertex location for Gemini.",
    )
    args = parser.parse_args()
    main(args)
