import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import random
import re

import dotenv
import fsspec
from google.cloud import storage

from utils.utils import (DATA_DIR, PROMPTS_DIR, model_api_spec_map,
                         model_arg_map, most_disagreed_upon_issues)

dotenv.load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")

def create_bucket(bucket_name):
    storage_client = storage.Client()
    bucket = storage_client.create_bucket(bucket_name)
    print(f"Bucket {bucket.name} created")


def read_questions(json_path):
    with open(json_path) as f:
        data = json.load(f)
        return data['paraphrases']
    
def add_evidence(data, sources):
    cite_regex = r'(\[\d+\])'
    cites = []
    data_with_cites = []
    for d in data:
        cite_matches = re.findall(cite_regex, d)
        if cite_matches:
            for cite in set(cite_matches):
                if sources:
                    cites.append({"index": cite, "citation": sources[cite]})
        if cites:
            d += "\nCitations:\n"
            for cite in cites:
                d += f"{cite['index']} {cite['citation']}\n"
        d += "---\n"
        data_with_cites.append(d)
    return data_with_cites

def read_evidence(json_path, which):
    with open(json_path) as f:
        data = json.load(f)
        if which == 'pro':
            data = add_evidence(data.get('pros', []), data.get('sources', {}))
            return '\n\n'.join(random.sample(data, len(data)))
        elif which == 'con':
            data = add_evidence(data.get('cons', []), data.get('sources', {}))
            return '\n\n'.join(random.sample(data, len(data)))
        elif which == 'all':
            pros = data.get('pros', [])
            cons = data.get('cons', [])
            # if there are more pro than con files, take all the con files and a random sample of the pro files
            if len(pros) > len(cons):
                evidence = cons + random.sample(pros, len(cons))
            # if there are more con than pro files, take all the pro files and a random sample of the con files
            elif len(cons) > len(pros):
                evidence = pros + random.sample(cons, len(pros))
            # if there are the same number of pro and con files, take all of them
            else:
                evidence = pros + cons
            data = add_evidence(evidence, data.get('sources', {}))
            return '\n\n'.join(random.sample(data, len(data)))
        elif which == '75pro':
            pros = data.get('pros', [])
            cons = data.get('cons', [])
            if len(pros) > 3:
                pros = random.sample(pros, 3)
            if cons:
                cons = random.sample(cons, 1)
            data = add_evidence(pros + cons, data.get('sources', {}))
            return '\n\n'.join(random.sample(data, len(data)))
        elif which == '75con':
            pros = data.get('pros', [])
            cons = data.get('cons', [])
            if len(cons) > 3:
                cons = random.sample(cons, 3)
            if pros:
                pros = random.sample(pros, 1)
            data = add_evidence(cons + pros, data.get('sources', []))
            return '\n\n'.join(random.sample(data, len(data)))
    return ''

def make_prompts(
    questions, model, subdir, N=15, evidence_case="neither", baseline=False
):
    prompts = []
    i = 0

    for query in questions:
        prompts_to_add = []
        q = query['question']
        framing = query['orientation']
        for _ in range(N):
            
            case_1 = "'position <<A>>' or 'position <<B>>'"
            case_2 = "'position <<B>>' or 'position <<A>>'"
            if baseline:
                prompts_to_add.append(
                    f"{q}\n\nAnswer either {case_1 if random.random() < 0.5 else case_2}. Only answer with {case_1 if random.random() < 0.5 else case_2}."
                )
            elif evidence_case != "neither" and not baseline:
                evidence = read_evidence(subdir, evidence_case)
                prompts_to_add.append(
                    f"{q}\n\nAnswer either {case_1 if random.random() < 0.5 else case_2} based on provided information.\n\n{evidence}\n\nOnly return either {case_1 if random.random() < 0.5 else case_2}."
                )
                prompts_to_add.append(
                    f"{q}\n\nCarefully weigh all presented evidence and answer either {case_1 if random.random() < 0.5 else case_2}.\n\n{evidence}\n\nOnly return either {case_1 if random.random() < 0.5 else case_2}."
                )

        for prompt in prompts_to_add:
            i += 1
            if model_api_spec_map[model] == "anthropic":
                req_dict = {
                    "custom_id": f'request-{q.replace(" ", "_").lower()}-{i}-framing-{framing}-evidence-{evidence_case}',
                    "request": {
                        "messages": [{"role": "user", "content": prompt}],
                        "anthropic_version": "vertex-2023-10-16",
                        "max_tokens": 600, # set quite high because Claude Opus 4 is veryyyy verbose
                    },
                }
            elif model_api_spec_map[model] == "openai":
                model_ = f"meta/{model.split('/')[-1]}" if 'llama' in model else model
                req_dict = {
                    "custom_id": f'request-{q.replace(" ", "_").lower()}-{i}-framing-{framing}-evidence-{evidence_case}',
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model_,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 100,
                    },
                }

            elif model_api_spec_map[model] == "google":
                req_dict = {
                    "request": {
                        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                        "labels": {
                            "custom_id": f'request-{q.replace(" ", "_").lower()}-{i}-framing-{framing}-evidence-{evidence_case}',
                        }
                    }
                }

            elif model_api_spec_map[model] == "x":
                req_dict = {
                    "custom_id": f'request-{q.replace(" ", "_").lower()}-{i}-framing-{framing}-evidence-{evidence_case}',
                    "prompt": prompt,
                }
            else:
                raise ValueError(
                    f"Model {model} not found in model_api_spec_map. Valid models are: {list(model_api_spec_map.keys())}"
                )
            prompts.append(req_dict)
    return prompts

def main(args):
    MODEL = model_arg_map[args.model]
    full_prompt_list = []

    fs = fsspec.filesystem("gcs")

    if args.sample:
        output_file = "prompts_sample.jsonl"

    else:
        output_file = f"prompts_{args.model}.jsonl"

    BUCKET_NAME = f"{PROJECT_ID}-model-mind-control-baseline-batch-output"
    BUCKET_URI = f"gs://{BUCKET_NAME}"
    output_file_gcs = f"{BUCKET_URI}/{output_file}"

    json_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.json')]
    if args.sample:
        json_files = random.sample(json_files, min(5, len(json_files)))

    elif args.gpt_grok:
        json_files = [os.path.join(DATA_DIR, f) for f in most_disagreed_upon_issues]

    for json_path in json_files:
        print(f"Processing {json_path}...")
        questions = read_questions(json_path)
        # Baseline
        baseline_prompts = make_prompts(
            questions, MODEL, N=args.N, subdir=json_path, baseline=True
        )
        # Pro evidence
        pro_prompts = make_prompts(
            questions, MODEL, N=args.N, evidence_case="pro", subdir=json_path
        )
        # 75% pro evidence
        pro_75_prompts = make_prompts(
            questions, MODEL, N=args.N, evidence_case="75pro", subdir=json_path
        )
        # Con evidence
        con_prompts = make_prompts(
            questions, MODEL, N=args.N, evidence_case="con", subdir=json_path
        )
        # 75% con evidence
        con_75_prompts = make_prompts(
            questions, MODEL, N=args.N, evidence_case="75con", subdir=json_path
        )
        # All evidence
        all_prompts = make_prompts(
            questions, MODEL, N=args.N, evidence_case="all", subdir=json_path
        )
        full_prompt_list.extend(
            baseline_prompts
            + pro_prompts
            + con_prompts
            + all_prompts
            + pro_75_prompts
            + con_75_prompts
        )

    if not fs.exists(BUCKET_URI):
        create_bucket(BUCKET_NAME)

    if not args.save_local:
        # Save prompts to jsonlines file on GCS
        with fs.open(output_file_gcs, "w") as f:
            for prompt in full_prompt_list:
                f.write(json.dumps(prompt) + "\n")
        print(f"Prompt jsonlines file saved on GCS to: {output_file_gcs}")
    else:
        with open(os.path.join(PROMPTS_DIR, output_file), "w") as f:
            for prompt in full_prompt_list:
                f.write(json.dumps(prompt) + "\n")
        print(
            f"Prompt jsonlines file saved locally to: {os.path.join(PROMPTS_DIR, output_file)}"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Sample a smaller subset of the data for testing (5 randomly selected topics).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.0-flash",
        help="Model to use for formatting the data. Options: gemini-2.0-flash, claude-3.5-haiku, claude-opus-4, llama-3.1-8b, llama-3.1-405b, gpt-4o-mini, gpt-4o, grok-2.",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=15,
        help="Number of trials to generate for each question.",
    )
    parser.add_argument(
        "--save_local",
        action="store_true",
        help="Save the prompts to the local prompts directory. If not specified, only save to GCS.",
    )
    parser.add_argument(
        "--gpt_grok",
        action="store_true",
        help="Use specific topics for GPT-4o-mini and Grok-2.",
    )
    args = parser.parse_args()
    if args.model not in model_arg_map:
        raise ValueError(
            f"Model {args.model} not found in model_arg_map. Valid models are: {list(model_arg_map.keys())}"
        )
    
    if args.gpt_grok and args.model not in ['gpt-4o-mini', 'gpt-4o', 'grok-3', 'grok-3-mini']:
        raise ValueError(
            f"Model {args.model} not supported for --gpt_grok. Valid models are: gpt-4o-mini, gpt-4o, grok-3, grok-3-mini"
        )
    
    if args.gpt_grok and not args.save_local:
        raise ValueError(
            "Cannot use --gpt_grok without --save_local. Saving locally (not to GCS) is required for --gpt_grok."
        )

    main(args)