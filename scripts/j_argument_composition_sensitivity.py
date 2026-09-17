import argparse
import itertools
import json
import os
import random
import re
import sys
import time

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm_clients import LLMRouter
from utils.utils import DATA_DIR

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO_DIR, "results")


ANSWER_REGEXES = [
    re.compile(r"position ([A|B])", re.IGNORECASE),
    re.compile(r"position <<([A|B])>>", re.IGNORECASE),
    re.compile(r"<<([A|B])>>", re.IGNORECASE),
    re.compile(r"^\s*([A|B])\s*$", re.IGNORECASE),
]


def extract_answer(text):
    if not isinstance(text, str):
        return "Other"
    for regex in ANSWER_REGEXES:
        match = re.search(regex, text)
        if match:
            return match.group(1).upper()
    return "Other"


def map_to_issue_stance(question_orientation, llm_answer):
    if question_orientation == "pro" and llm_answer == "A":
        return "pro"
    if question_orientation == "con" and llm_answer == "A":
        return "con"
    if question_orientation == "pro" and llm_answer == "B":
        return "con"
    if question_orientation == "con" and llm_answer == "B":
        return "pro"
    return "other"


def build_composition_prompt(question_text, pro_texts, con_texts):
    evidence_items = []
    for item in pro_texts:
        evidence_items.append(("PRO", item))
    for item in con_texts:
        evidence_items.append(("CON", item))
    random.shuffle(evidence_items)

    blocks = []
    for idx, (side, text) in enumerate(evidence_items, start=1):
        blocks.append(f"Argument {idx} ({side}):\n{text.strip()}\n---")

    evidence_block = "\n\n".join(blocks)
    return (
        f"{question_text}\n\n"
        "Carefully weigh all presented evidence and answer either 'position <<A>>' or 'position <<B>>'.\n\n"
        f"{evidence_block}\n\n"
        "Only return either 'position <<A>>' or 'position <<B>>'."
    )


def sample_compositions(topic_data, case_name, max_compositions, rng):
    pros = topic_data.get("pros", [])
    cons = topic_data.get("cons", [])
    compositions = []

    if case_name == "75pro":
        if len(pros) < 3 or len(cons) < 1:
            return compositions
        all_combos = list(itertools.product(itertools.combinations(range(len(pros)), 3), range(len(cons))))
        rng.shuffle(all_combos)
        chosen = all_combos[:max_compositions]
        for i, (pro_idx, con_idx) in enumerate(chosen):
            compositions.append(
                {
                    "composition_id": f"{case_name}-{i}",
                    "pro_indices": list(pro_idx),
                    "con_indices": [int(con_idx)],
                }
            )

    elif case_name == "75con":
        if len(cons) < 3 or len(pros) < 1:
            return compositions
        all_combos = list(itertools.product(range(len(pros)), itertools.combinations(range(len(cons)), 3)))
        rng.shuffle(all_combos)
        chosen = all_combos[:max_compositions]
        for i, (pro_idx, con_idx) in enumerate(chosen):
            compositions.append(
                {
                    "composition_id": f"{case_name}-{i}",
                    "pro_indices": [int(pro_idx)],
                    "con_indices": list(con_idx),
                }
            )
    else:
        raise ValueError(f"Unsupported case: {case_name}")

    return compositions


def compute_sensitivity_metrics(pred_df):
    if pred_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    comp_level = (
        pred_df.groupby(
            ["model", "topic", "case", "paraphrase_idx", "composition_id"], as_index=False
        )["issue_stance"]
        .agg(
            p_pro=lambda x: float(np.mean(x == "pro")),
            p_con=lambda x: float(np.mean(x == "con")),
            p_other=lambda x: float(np.mean(x == "other")),
        )
    )

    comp_level["composition_label"] = comp_level.apply(
        lambda r: "pro" if r["p_pro"] > 0.5 else ("con" if r["p_con"] > 0.5 else "other"),
        axis=1,
    )

    rows = []
    group_cols = ["model", "topic", "case", "paraphrase_idx"]
    for key, g in comp_level.groupby(group_cols):
        labels = g["composition_label"].tolist()
        p_pro_values = g["p_pro"].to_numpy(dtype=float)
        n = len(labels)
        if n < 2:
            rows.append(
                {
                    "model": key[0],
                    "topic": key[1],
                    "case": key[2],
                    "paraphrase_idx": key[3],
                    "n_compositions": n,
                    "pairwise_flip_rate": np.nan,
                    "mean_abs_pairwise_delta_p_pro": np.nan,
                    "max_abs_pairwise_delta_p_pro": np.nan,
                    "p_pro_variance_across_compositions": np.nan,
                }
            )
            continue

        pairwise_total = 0
        pairwise_flips = 0
        pairwise_abs_deltas = []
        for i in range(n):
            for j in range(i + 1, n):
                pairwise_total += 1
                if labels[i] != labels[j]:
                    pairwise_flips += 1
                pairwise_abs_deltas.append(abs(p_pro_values[i] - p_pro_values[j]))

        rows.append(
            {
                "model": key[0],
                "topic": key[1],
                "case": key[2],
                "paraphrase_idx": key[3],
                "n_compositions": n,
                "pairwise_flip_rate": pairwise_flips / pairwise_total if pairwise_total else np.nan,
                "mean_abs_pairwise_delta_p_pro": (
                    float(np.mean(pairwise_abs_deltas)) if pairwise_abs_deltas else np.nan
                ),
                "max_abs_pairwise_delta_p_pro": (
                    float(np.max(pairwise_abs_deltas)) if pairwise_abs_deltas else np.nan
                ),
                "p_pro_variance_across_compositions": float(np.var(p_pro_values, ddof=1)),
            }
        )

    topic_metrics = pd.DataFrame(rows)
    summary = (
        topic_metrics.groupby(["model", "case"], as_index=False)
        .agg(
            topics=("topic", "nunique"),
            mean_pairwise_flip_rate=("pairwise_flip_rate", "mean"),
            mean_abs_pairwise_delta_p_pro=("mean_abs_pairwise_delta_p_pro", "mean"),
            max_abs_pairwise_delta_p_pro=("max_abs_pairwise_delta_p_pro", "mean"),
            mean_p_pro_variance=("p_pro_variance_across_compositions", "mean"),
        )
        .sort_values(["case", "mean_pairwise_flip_rate"], ascending=[True, False])
    )
    return topic_metrics, summary


def main(args):
    rng = random.Random(args.seed)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    predictions_path = os.path.join(RESULTS_DIR, "composition_predictions.csv")
    topic_metrics_path = os.path.join(RESULTS_DIR, "composition_sensitivity_by_topic.csv")
    summary_path = os.path.join(RESULTS_DIR, "composition_sensitivity_summary.csv")

    router = None if args.dry_run else LLMRouter(
        project_id=args.project_id,
        google_location=args.google_location,
    )

    topic_files = [
        os.path.join(DATA_DIR, name)
        for name in sorted(os.listdir(DATA_DIR))
        if name.endswith(".json")
    ]
    if args.sample_topics > 0:
        topic_files = topic_files[: args.sample_topics]

    rows = []
    calls_done = 0
    next_progress_log = args.progress_every
    print(
        f"Running composition experiment for {len(topic_files)} topics, "
        f"models={models}, cases={cases}, dry_run={args.dry_run}"
    )

    for path in topic_files:
        topic = os.path.basename(path).replace(".json", "")
        with open(path, "r") as f:
            data = json.load(f)
        paraphrases = data.get("paraphrases", [])
        if not paraphrases:
            continue

        if args.paraphrase_mode == "first":
            para_items = list(enumerate(paraphrases[:1]))
        else:
            para_items = list(enumerate(paraphrases))

        for case_name in cases:
            compositions = sample_compositions(
                topic_data=data,
                case_name=case_name,
                max_compositions=args.max_compositions_per_issue,
                rng=rng,
            )
            if not compositions:
                continue

            for para_idx, para in para_items:
                question_text = para.get("question", "")
                orientation = para.get("orientation", "")
                if not question_text or orientation not in {"pro", "con"}:
                    continue

                for comp in compositions:
                    pro_texts = [data["pros"][i] for i in comp["pro_indices"]]
                    con_texts = [data["cons"][i] for i in comp["con_indices"]]
                    prompt = build_composition_prompt(question_text, pro_texts, con_texts)

                    for model in models:
                        for trial_idx in range(args.trials_per_composition):
                            if args.dry_run:
                                raw_response = ""
                                llm_answer = "Other"
                                issue_stance = "other"
                                error = ""
                            else:
                                try:
                                    raw_response = router.call(
                                        model,
                                        prompt,
                                        max_retries=args.max_retries,
                                    )
                                    llm_answer = extract_answer(raw_response)
                                    issue_stance = map_to_issue_stance(orientation, llm_answer)
                                    error = ""
                                except Exception as exc:
                                    raw_response = ""
                                    llm_answer = "Other"
                                    issue_stance = "other"
                                    error = str(exc)

                            rows.append(
                                {
                                    "model": model,
                                    "topic": topic,
                                    "case": case_name,
                                    "paraphrase_idx": para_idx,
                                    "question_orientation": orientation,
                                    "composition_id": comp["composition_id"],
                                    "pro_indices": ",".join(str(i) for i in comp["pro_indices"]),
                                    "con_indices": ",".join(str(i) for i in comp["con_indices"]),
                                    "trial_idx": trial_idx,
                                    "llm_answer": llm_answer,
                                    "issue_stance": issue_stance,
                                    "error": error,
                                    "raw_response": raw_response,
                                }
                            )
                            calls_done += 1
                            if calls_done >= next_progress_log:
                                print(f"Progress: completed {calls_done} model calls...")
                                pd.DataFrame(rows).to_csv(predictions_path, index=False)
                                next_progress_log += args.progress_every
                            if args.sleep_sec > 0:
                                time.sleep(args.sleep_sec)

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(predictions_path, index=False)

    topic_metrics_df, summary_df = compute_sensitivity_metrics(pred_df)
    topic_metrics_df.to_csv(topic_metrics_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"Wrote: {predictions_path}")
    print(f"Wrote: {topic_metrics_path}")
    print(f"Wrote: {summary_path}")
    if not summary_df.empty:
        print(summary_df.head(20))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run argument composition sensitivity experiment.")
    parser.add_argument(
        "--models",
        type=str,
        default="gemini-2.0-flash,gpt-4o,grok-3",
        help="Comma-separated model list.",
    )
    parser.add_argument(
        "--cases",
        type=str,
        default="75pro,75con",
        help="Comma-separated case list.",
    )
    parser.add_argument(
        "--max-compositions-per-issue",
        type=int,
        default=8,
        help="Max sampled compositions per topic-case.",
    )
    parser.add_argument(
        "--trials-per-composition",
        type=int,
        default=1,
        help="Number of repeated calls per composition.",
    )
    parser.add_argument(
        "--paraphrase-mode",
        choices=["first", "all"],
        default="first",
        help="Use only first paraphrase or all paraphrases.",
    )
    parser.add_argument(
        "--sample-topics",
        type=int,
        default=0,
        help="If >0, only run first N topics.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sampling seed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate experiment structure without API calls.",
    )
    parser.add_argument(
        "--sleep-sec",
        type=float,
        default=0.0,
        help="Optional sleep between calls.",
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
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Log and checkpoint every N model calls.",
    )
    args = parser.parse_args()
    main(args)
