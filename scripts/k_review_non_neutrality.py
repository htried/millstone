import argparse
import hashlib
import os
from datetime import datetime

import pandas as pd


REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT = os.path.join(REPO_DIR, "results", "paraphrase_neutrality_judgments.csv")
DEFAULT_OUTPUT = os.path.join(
    REPO_DIR, "results", "paraphrase_non_neutrality_manual_review.csv"
)


def make_row_id(row):
    key = "|".join(
        [
            str(row.get("topic", "")),
            str(row.get("paraphrase_idx", "")),
            str(row.get("judge_model", "")),
            str(row.get("label", "")),
        ]
    )
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def load_existing_decisions(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def build_candidates(df, include_errors):
    mask_non_neutral = df["label"].fillna("").str.lower() != "neutral"
    if include_errors:
        mask_errors = df["error"].fillna("").str.strip().str.len() > 0
        return df[mask_non_neutral | mask_errors].copy()
    return df[mask_non_neutral].copy()


def _clean_text(value):
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def print_row(row, idx, total):
    print("\n" + "=" * 90)
    print(f"[{idx}/{total}] topic={row.get('topic')} paraphrase_idx={row.get('paraphrase_idx')}")
    print(
        f"model={row.get('judge_model')} auto_label={row.get('label')} "
        f"confidence={row.get('confidence')}"
    )
    error_text = _clean_text(row.get("error", ""))
    if error_text:
        print(f"error={error_text}")
    print("-" * 90)
    print("INPUT PARAPHRASE:")
    print(_clean_text(row.get("paraphrase_text", "")) or "<empty>")
    print("-" * 90)
    print("MODEL OUTPUT:")
    raw_response = _clean_text(row.get("raw_response", ""))
    rationale = _clean_text(row.get("rationale", ""))
    if raw_response:
        print(raw_response)
    elif rationale:
        print(rationale)
    elif error_text:
        print(f"<error> {error_text}")
    else:
        print("<empty>")
    print("-" * 90)


def interactive_review(candidates, existing_decisions, output_path):
    existing_by_id = {}
    if not existing_decisions.empty and "row_id" in existing_decisions.columns:
        for _, r in existing_decisions.iterrows():
            existing_by_id[str(r["row_id"])] = r.to_dict()

    decisions = list(existing_by_id.values())
    total = len(candidates)
    reviewed = 0

    for i, (_, row) in enumerate(candidates.iterrows(), start=1):
        row_dict = row.to_dict()
        row_id = make_row_id(row_dict)
        if row_id in existing_by_id:
            reviewed += 1
            continue

        print_row(row_dict, i, total)
        print("Decision options:")
        print("  [n] confirm non-neutral")
        print("  [u] mark as neutral")
        print("  [s] skip for now")
        print("  [q] quit and save")
        choice = input("Choice: ").strip().lower()

        if choice == "q":
            break
        if choice == "s":
            continue
        if choice not in {"n", "u"}:
            print("Invalid choice, skipping.")
            continue

        note = input("Optional note: ").strip()
        manual_label = "non_neutral" if choice == "n" else "neutral"
        decision = {
            "row_id": row_id,
            "reviewed_at": datetime.utcnow().isoformat() + "Z",
            "topic": row_dict.get("topic"),
            "paraphrase_idx": row_dict.get("paraphrase_idx"),
            "judge_model": row_dict.get("judge_model"),
            "auto_label": row_dict.get("label"),
            "manual_label": manual_label,
            "note": note,
            "error": _clean_text(row_dict.get("error", "")),
            "paraphrase_text": _clean_text(row_dict.get("paraphrase_text", "")),
            "raw_response": _clean_text(row_dict.get("raw_response", "")),
        }
        decisions.append(decision)
        existing_by_id[row_id] = decision
        reviewed += 1

    out_df = pd.DataFrame(decisions)
    out_df.to_csv(output_path, index=False)
    print("\nSaved review decisions to:", output_path)
    print("Existing/updated decisions:", len(out_df))
    print("Total candidates:", total)
    print("Auto-skipped as already reviewed:", reviewed)


def main():
    parser = argparse.ArgumentParser(
        description="Interactively review non-neutral/unknown neutrality judgments."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT,
        help="Input CSV path (default: results/paraphrase_neutrality_judgments.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Output CSV path for manual decisions.",
    )
    parser.add_argument(
        "--include-errors",
        action="store_true",
        help="Include rows with errors even if auto-label is neutral.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Optional filter to one judge model, e.g. gemini-2.0-flash",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input not found: {args.input}")

    df = pd.read_csv(args.input)
    required = {
        "topic",
        "paraphrase_idx",
        "judge_model",
        "label",
        "paraphrase_text",
        "raw_response",
        "error",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input: {sorted(missing)}")

    if args.model:
        df = df[df["judge_model"] == args.model].copy()

    candidates = build_candidates(df, include_errors=args.include_errors)
    if candidates.empty:
        print("No candidate rows to review.")
        return

    existing = load_existing_decisions(args.output)
    interactive_review(candidates, existing, args.output)


if __name__ == "__main__":
    main()
