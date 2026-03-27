import os
import sys
import re
import glob
import json
import pandas as pd
import scipy.stats as stats

# Adjust paths
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_DIR, 'data')
PREDICTIONS_DIR = os.path.join(REPO_DIR, 'predictions')
RESULTS_DIR = os.path.join(REPO_DIR, 'results')

def get_question_maps():
    question_map = {}
    question_map_pro_con = {}
    for file in os.listdir(DATA_DIR):
        if not file.endswith('.json'):
            continue
        slug = file.replace('.json', '').lower()
        with open(os.path.join(DATA_DIR, file), "r") as f:
            data = json.load(f)
            if "paraphrases" in data:
                for entry in data["paraphrases"]:
                    q = entry["question"].strip().lower()
                    pro_con = entry.get("orientation", "").strip().lower()
                    question_map[q] = slug
                    question_map_pro_con[q] = pro_con
    return question_map, question_map_pro_con

answer_regexes = [
    re.compile(r'position ([A|B])', re.IGNORECASE),
    re.compile(r'position <<([A|B])>>', re.IGNORECASE),
    re.compile(r"<<([A|B])>>", re.IGNORECASE),
    re.compile(r"^\s*([A|B])\s*$", re.IGNORECASE),
]

def binomial_two_sided_pvalue(k, n):
    if n == 0:
        return float("nan")
    if hasattr(stats, "binomtest"):
        return stats.binomtest(k, n, p=0.5, alternative="two-sided").pvalue
    return stats.binom_test(k, n, p=0.5, alternative="two-sided")

def extract_answer(text):
    if not isinstance(text, str):
        return 'Other'
    for regex in answer_regexes:
        match = re.search(regex, text)
        if match:
            return match.group(1).upper()
    return 'Other'

def map_to_issue_stance(row):
    if row['question_stance'] == 'pro' and row['llm_answer'] == 'A':
        return 'pro'
    elif row['question_stance'] == 'con' and row['llm_answer'] == 'A':
        return 'con'
    elif row['question_stance'] == 'pro' and row['llm_answer'] == 'B':
        return 'con'
    elif row['question_stance'] == 'con' and row['llm_answer'] == 'B':
        return 'pro'
    else:
        return 'other'

def analyze_order():
    print("Starting order analysis...", flush=True)
    question_map, question_map_pro_con = get_question_maps()
    print("Loaded question maps.", flush=True)

    # Load all issue data once for matching snippets against pro/con arguments
    issue_data = {}
    for file in os.listdir(DATA_DIR):
        if not file.endswith(".json"):
            continue
        slug = file.replace(".json", "").lower()
        with open(os.path.join(DATA_DIR, file), "r") as f:
            issue_data[slug] = json.load(f)
    print(f"Loaded {len(issue_data)} issues data.", flush=True)

    # Lazily load prompts per model (avoids huge up-front memory/time).
    prompt_lookup_cache = {}

    def load_prompt_lookup_for_model(model_name):
        if model_name in prompt_lookup_cache:
            return prompt_lookup_cache[model_name]

        prompt_file = os.path.join(REPO_DIR, "prompts", f"prompts_{model_name}.jsonl")
        lookup = {}
        if not os.path.exists(prompt_file):
            prompt_lookup_cache[model_name] = lookup
            return lookup

        print(f"  Loading prompts for {model_name}...", flush=True)
        with open(prompt_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    p = json.loads(line)
                except json.JSONDecodeError:
                    continue

                custom_id = (
                    p.get("custom_id")
                    or p.get("request", {}).get("labels", {}).get("custom_id")
                    or p.get("request", {}).get("custom_id")
                )
                if not custom_id:
                    continue

                req_text = ""
                if "request" in p and isinstance(p["request"], dict):
                    req = p["request"]
                    if "messages" in req and len(req["messages"]) > 0:
                        req_text = req["messages"][0].get("content", "")
                    elif (
                        "contents" in req
                        and len(req["contents"]) > 0
                        and "parts" in req["contents"][0]
                        and len(req["contents"][0]["parts"]) > 0
                    ):
                        req_text = req["contents"][0]["parts"][0].get("text", "")
                elif "body" in p and "messages" in p["body"] and len(p["body"]["messages"]) > 0:
                    req_text = p["body"]["messages"][0].get("content", "")
                elif "prompt" in p:
                    req_text = p["prompt"]

                if isinstance(req_text, list) and len(req_text) > 0 and "text" in req_text[0]:
                    req_text = req_text[0]["text"]

                lookup[custom_id] = req_text

        print(f"  Loaded {len(lookup)} prompts for {model_name}.", flush=True)
        prompt_lookup_cache[model_name] = lookup
        return lookup

    def get_last_argument_orientation(request_text, issue_slug):
        if not request_text or issue_slug not in issue_data:
            return None

        # If no citations marker, we cannot recover argument ordering.
        if "Citations:" not in request_text:
            return None

        parts = request_text.split("Citations:")
        if len(parts) < 2:
            return None

        last_block = parts[-2]
        last_arg_text = last_block.rpartition("---\n")[-1]
        if not last_arg_text.strip():
            paragraphs = [p.strip() for p in last_block.split("---\n") if p.strip()]
            if not paragraphs:
                return None
            last_arg_text = paragraphs[-1]

        pros = issue_data[issue_slug].get("pros", [])
        cons = issue_data[issue_slug].get("cons", [])
        cleaned_last_arg = re.sub(r"\[\d+\]", "", last_arg_text).strip()
        cleaned_last_arg_parts = cleaned_last_arg.split("\n")

        for arg_candidate in reversed(cleaned_last_arg_parts):
            arg_candidate = arg_candidate.strip()
            if len(arg_candidate) < 20:
                continue

            cand_prefix = arg_candidate[:50]
            for pro in pros:
                c_pro = re.sub(r"\[\d+\]", "", pro).strip()
                if cand_prefix in c_pro:
                    return "pro"
            for con in cons:
                c_con = re.sub(r"\[\d+\]", "", con).strip()
                if cand_prefix in c_con:
                    return "con"

            if len(arg_candidate) >= 50:
                substr = arg_candidate[len(arg_candidate) // 2 : len(arg_candidate) // 2 + 50]
                for pro in pros:
                    c_pro = re.sub(r"\[\d+\]", "", pro).strip()
                    if substr in c_pro:
                        return "pro"
                for con in cons:
                    c_con = re.sub(r"\[\d+\]", "", con).strip()
                    if substr in c_con:
                        return "con"

        return None

    all_data = []
    prediction_files = sorted(glob.glob(os.path.join(PREDICTIONS_DIR, "*.jsonl")))
    model_to_files = {}
    for filepath in prediction_files:
        filename = os.path.basename(filepath)
        model_name = filename.replace("predictions_", "").replace(".jsonl", "")
        model_name = re.sub(r"_\d+$", "", model_name)
        model_to_files.setdefault(model_name, []).append(filepath)

    print(
        f"Processing {len(prediction_files)} prediction files across {len(model_to_files)} model units...",
        flush=True,
    )

    import time

    for model_idx, (model_name, files) in enumerate(sorted(model_to_files.items()), start=1):
        model_t0 = time.time()
        prompt_lookup = load_prompt_lookup_for_model(model_name)
        print(
            f"  [{model_idx}/{len(model_to_files)}] Processing model unit: {model_name} ({len(files)} shard files)",
            flush=True,
        )

        for shard_idx, filepath in enumerate(sorted(files), start=1):
            t0 = time.time()
            filename = os.path.basename(filepath)
            print(
                f"    [{shard_idx}/{len(files)}] Processing shard {filename}...",
                flush=True,
            )

            with open(filepath, "r") as f:
                for line_idx, line in enumerate(f):
                    if line_idx > 0 and line_idx % 10000 == 0:
                        print(f"      Processed {line_idx} lines in {filename}...", flush=True)
                    line = line.strip()
                    if not line or line.startswith("version https://git-lfs"):
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    custom_id = (
                        obj.get("custom_id")
                        or obj.get("request", {}).get("labels", {}).get("custom_id")
                        or obj.get("request", {}).get("custom_id")
                    )
                    if not custom_id:
                        continue
                    match = re.search(r"request-(.*?)-(\d+)-framing", custom_id)
                    if not match:
                        continue

                    question_raw = match.group(1)
                    trial_num = int(match.group(2))
                    question = question_raw.replace("_", " ").lower()

                    case_split = custom_id.split("-evidence-")
                    if len(case_split) < 2:
                        continue
                    evidence_case = case_split[1]

                    # Keep both mixed and one-sided cases for downstream correlation plots.
                    if evidence_case not in ["all", "75/25", "75pro", "75con", "con", "pro"]:
                        continue

                    # extract response text
                    resp = obj.get("response", "")
                    if isinstance(resp, str):
                        resp_text = resp
                    elif isinstance(resp, dict):
                        if "content" in resp and len(resp["content"]) > 0 and "text" in resp["content"][0]:
                            resp_text = resp["content"][0]["text"]
                        elif (
                            "candidates" in resp
                            and len(resp["candidates"]) > 0
                            and isinstance(resp["candidates"][0], dict)
                            and isinstance(resp["candidates"][0].get("content"), dict)
                            and isinstance(resp["candidates"][0]["content"].get("parts"), list)
                            and len(resp["candidates"][0]["content"]["parts"]) > 0
                            and isinstance(resp["candidates"][0]["content"]["parts"][0], dict)
                            and "text" in resp["candidates"][0]["content"]["parts"][0]
                        ):
                            resp_text = resp["candidates"][0]["content"]["parts"][0]["text"]
                        elif (
                            "choices" in resp
                            and len(resp["choices"]) > 0
                            and "message" in resp["choices"][0]
                            and "content" in resp["choices"][0]["message"]
                        ):
                            resp_text = resp["choices"][0]["message"]["content"]
                        elif (
                            "body" in resp
                            and "choices" in resp["body"]
                            and len(resp["body"]["choices"]) > 0
                            and "message" in resp["body"]["choices"][0]
                        ):
                            resp_text = resp["body"]["choices"][0]["message"]["content"]
                        else:
                            resp_text = ""
                    else:
                        resp_text = ""

                    llm_answer = extract_answer(resp_text)

                    if "llm_answer" in obj and obj["llm_answer"] in ["A", "B", "Other"]:
                        llm_answer = obj["llm_answer"]

                    issue = question_map.get(question, "unknown")
                    q_stance = question_map_pro_con.get(question, "unknown")

                    req_text = ""
                    req = obj.get("request", {})
                    if isinstance(req, dict):
                        if "messages" in req and len(req["messages"]) > 0 and "content" in req["messages"][0]:
                            req_text = req["messages"][0]["content"]
                        elif (
                            "contents" in req
                            and len(req["contents"]) > 0
                            and "parts" in req["contents"][0]
                            and len(req["contents"][0]["parts"]) > 0
                            and "text" in req["contents"][0]["parts"][0]
                        ):
                            req_text = req["contents"][0]["parts"][0]["text"]

                    if not req_text and "prompt_text" in obj:
                        req_text = obj["prompt_text"]

                    if not req_text and "request" in obj and isinstance(obj["request"], dict):
                        req2 = obj["request"]
                        if "messages" in req2 and len(req2["messages"]) > 0 and "content" in req2["messages"][0]:
                            req_text = req2["messages"][0]["content"]

                    if not req_text and "body" in obj and "messages" in obj["body"] and len(obj["body"]["messages"]) > 0:
                        req_text = obj["body"]["messages"][0]["content"]

                    if not req_text:
                        if custom_id in prompt_lookup:
                            req_text = prompt_lookup[custom_id]
                        else:
                            continue

                    row = {
                        "model": model_name,
                        "issue": issue,
                        "case": evidence_case,
                        "trial_num": trial_num,
                        "question": question,
                        "question_stance": q_stance,
                        "llm_answer": llm_answer,
                    }
                    del obj

                    row["issue_stance"] = map_to_issue_stance(row)
                    row["last_argument"] = get_last_argument_orientation(req_text, issue)
                    if row["last_argument"]:
                        row["agreed_with_last"] = row["issue_stance"] == row["last_argument"]
                        all_data.append(row)

            t1 = time.time()
            print(f"    Finished shard {filename} in {t1 - t0:.2f}s", flush=True)

        model_t1 = time.time()
        print(
            f"  Finished model unit {model_name} in {model_t1 - model_t0:.2f}s",
            flush=True,
        )

    df = pd.DataFrame(all_data)
    if df.empty:
        print("No valid prediction data found for order effects.", flush=True)
        return

    df = df[df["issue"] != "unknown"]
    results = []

    # Stratify by the orientation of the final argument (pro/con) so we can
    # measure recency effects separately for each last-argument polarity.
    grouped = df.groupby(["model", "case", "last_argument"])
    for (model, case, last_argument), group in grouped:
        total = len(group)
        agreed = group["agreed_with_last"].sum()
        p_val = binomial_two_sided_pvalue(agreed, total)

        results.append({
            "model": model,
            "case": case,
            "last_argument": last_argument,
            "total_trials": total,
            "agreed_with_last_count": agreed,
            "agreed_with_last_pct": agreed / total if total > 0 else 0,
            "p_value_vs_50pct": p_val,
        })

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    out_df = pd.DataFrame(results)
    out_file = os.path.join(RESULTS_DIR, "order_effects.csv")
    out_df.to_csv(out_file, index=False)
    print(f"Order effects results written to {out_file}", flush=True)

if __name__ == "__main__":
    analyze_order()
