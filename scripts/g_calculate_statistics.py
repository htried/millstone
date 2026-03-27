import glob
import hashlib
import json
import os
import re
import sys

import numpy as np
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


def normalize_request_text(text):
    if not isinstance(text, str):
        return ''
    # Normalize whitespace and case so prompt/prediction matching is robust.
    return ' '.join(text.split()).strip().lower()


def request_text_hash(text):
    norm = normalize_request_text(text)
    if not norm:
        return None
    return hashlib.sha1(norm.encode('utf-8')).hexdigest()


def extract_request_text_from_request_payload(request_obj):
    if not isinstance(request_obj, dict):
        return ''

    # Gemini-style request payload.
    contents = request_obj.get('contents')
    if isinstance(contents, list) and contents:
        parts = contents[0].get('parts') if isinstance(contents[0], dict) else None
        if isinstance(parts, list) and parts:
            part0 = parts[0]
            if isinstance(part0, dict) and isinstance(part0.get('text'), str):
                return part0['text']

    # OpenAI-style request payload.
    messages = request_obj.get('messages')
    if isinstance(messages, list) and messages:
        msg0 = messages[0]
        if isinstance(msg0, dict):
            content = msg0.get('content')
            if isinstance(content, str):
                return content
            if isinstance(content, list) and content:
                c0 = content[0]
                if isinstance(c0, dict) and isinstance(c0.get('text'), str):
                    return c0['text']

    return ''


def load_prompt_lookup_for_model(model_name):
    prompt_path = os.path.join(REPO_DIR, 'prompts', f'prompts_{model_name}.jsonl')
    lookup = {}
    if not os.path.exists(prompt_path):
        return lookup

    with open(prompt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            req = obj.get('request', {})
            custom_id = (
                obj.get('custom_id')
                or req.get('labels', {}).get('custom_id')
                or req.get('custom_id')
            )
            req_text = extract_request_text_from_request_payload(req)
            h = request_text_hash(req_text)
            if custom_id and h and h not in lookup:
                lookup[h] = custom_id

    return lookup

def parse_predictions():
    question_map, question_map_pro_con = get_question_maps()
    all_data = []
    prompt_lookup_cache = {}

    for filepath in glob.glob(os.path.join(PREDICTIONS_DIR, '*.jsonl')):
        # Infer model from filename
        filename = os.path.basename(filepath)
        # e.g., predictions_claude-3.5-haiku.jsonl
        # or predictions_llama-3.1-405b_0.jsonl
        model_name = filename.replace('predictions_', '').replace('.jsonl', '')
        # Remove trailing _0, _1 if present
        model_name = re.sub(r'_\d+$', '', model_name)

        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('version https://git-lfs'):
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                custom_id = (
                    obj.get('custom_id')
                    or obj.get('request', {}).get('labels', {}).get('custom_id')
                    or obj.get('request', {}).get('custom_id')
                )
                if not custom_id:
                    # Gemini rows often omit top-level custom_id in predictions.
                    # Recover by matching request text to prompts_<model>.jsonl labels.custom_id.
                    req_text = extract_request_text_from_request_payload(obj.get('request', {}))
                    h = request_text_hash(req_text)
                    if h:
                        if model_name not in prompt_lookup_cache:
                            prompt_lookup_cache[model_name] = load_prompt_lookup_for_model(model_name)
                        custom_id = prompt_lookup_cache[model_name].get(h)
                if not custom_id:
                    continue

                # parse custom_id
                # request-<question>-<trial_num>-framing-<stance>-evidence-<case>
                match = re.search(r'request-(.*?)-(\d+)-framing', custom_id)
                if not match:
                    continue
                
                question_raw = match.group(1)
                trial_num = int(match.group(2))
                question = question_raw.replace('_', ' ').lower()
                
                case_split = custom_id.split('-evidence-')
                if len(case_split) < 2:
                    continue
                evidence_case = case_split[1]

                # extract response text
                resp = obj.get('response', '')
                if isinstance(resp, str):
                    resp_text = resp
                elif isinstance(resp, dict):
                    # Try to extract from different structures
                    if 'content' in resp and len(resp['content']) > 0 and 'text' in resp['content'][0]:
                        resp_text = resp['content'][0]['text']
                    elif (
                        'candidates' in resp
                        and len(resp['candidates']) > 0
                        and isinstance(resp['candidates'][0], dict)
                        and isinstance(resp['candidates'][0].get('content'), dict)
                        and isinstance(resp['candidates'][0]['content'].get('parts'), list)
                        and len(resp['candidates'][0]['content']['parts']) > 0
                        and isinstance(resp['candidates'][0]['content']['parts'][0], dict)
                        and 'text' in resp['candidates'][0]['content']['parts'][0]
                    ):
                        resp_text = resp['candidates'][0]['content']['parts'][0]['text']
                    elif 'choices' in resp and len(resp['choices']) > 0 and 'message' in resp['choices'][0] and 'content' in resp['choices'][0]['message']:
                        resp_text = resp['choices'][0]['message']['content']
                    elif 'body' in resp and 'choices' in resp['body'] and len(resp['body']['choices']) > 0 and 'message' in resp['body']['choices'][0]:
                        resp_text = resp['body']['choices'][0]['message']['content']
                    else:
                        resp_text = ''
                else:
                    resp_text = ''

                llm_answer = extract_answer(resp_text)
                
                # Check for cached results where the model explicitly extracted llm_answer
                if 'llm_answer' in obj and obj['llm_answer'] in ['A', 'B', 'Other']:
                    llm_answer = obj['llm_answer']

                issue = question_map.get(question, 'unknown')
                q_stance = question_map_pro_con.get(question, 'unknown')

                row = {
                    'model': model_name,
                    'issue': issue,
                    'case': evidence_case,
                    'trial_num': trial_num,
                    'question': question,
                    'question_stance': q_stance,
                    'llm_answer': llm_answer,
                }
                row['issue_stance'] = map_to_issue_stance(row)
                all_data.append(row)

    return pd.DataFrame(all_data)

def calculate_stats():
    df = parse_predictions()
    if df.empty:
        print("No valid prediction data found.")
        return

    # Filter out unknown issues or unknown stances
    df = df[df['issue'] != 'unknown']

    # We want stance distribution per trial
    # Actually, we want to know what % of stances for a given trial were 'pro'
    # Group by (model, case, issue, trial_num, issue_stance) to get counts
    counts = df.groupby(['model', 'case', 'issue', 'trial_num', 'issue_stance']).size().unstack(fill_value=0).reset_index()
    
    # Ensure columns exist
    for stance in ['pro', 'con', 'other']:
        if stance not in counts.columns:
            counts[stance] = 0
            
    counts['total'] = counts['pro'] + counts['con'] + counts['other']
    counts['pro_share'] = counts['pro'] / counts['total']
    counts['con_share'] = counts['con'] / counts['total']

    results = []
    for model in counts['model'].unique():
        model_data = counts[counts['model'] == model]
        neither_data = model_data[model_data['case'] == 'neither']
        
        for case in model_data['case'].unique():
            if case == 'neither':
                continue
                
            case_data = model_data[model_data['case'] == case]
            
            # Align by issue and trial_num
            merged = pd.merge(case_data, neither_data, on=['issue', 'trial_num'], suffixes=('_case', '_neither'))
            if merged.empty:
                continue
            
            
            pro_shift = merged['pro_share_case'] - merged['pro_share_neither']

            # Paired test by matched issue/trial baseline alignment.
            t_stat, p_val = stats.ttest_rel(merged['pro_share_case'], merged['pro_share_neither'])

            # Mean shift relative to baseline.
            mean_shift = pro_shift.mean()

            # 95% CI on the paired shift distribution.
            n = len(pro_shift)
            if n <= 1:
                ci_margin = 0.0
            else:
                sd_shift = np.std(pro_shift, ddof=1)
                se = sd_shift / np.sqrt(n)
                ci_margin = stats.t.ppf(0.975, df=n - 1) * se
            
            results.append({
                'model': model,
                'case': case,
                'mean_pro_shift': mean_shift,
                'p_value': p_val,
                'ci_lower': mean_shift - ci_margin,
                'ci_upper': mean_shift + ci_margin,
                'n_trials': len(merged)
            })

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        
    out_df = pd.DataFrame(results)
    out_file = os.path.join(RESULTS_DIR, 'statistical_summary.csv')
    out_df.to_csv(out_file, index=False)
    print(f"Results written to {out_file}")

if __name__ == "__main__":
    calculate_stats()
