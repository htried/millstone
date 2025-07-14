import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time

import dotenv
from google.genai import Client
from google.genai.types import CreateBatchJobConfig, JobState

from utils.utils import model_arg_map, model_location_map

dotenv.load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")

def main(args):
    MODEL = model_arg_map[args.model]

    if args.sample:
        output_file = "prompts_sample.jsonl"

    else:
        output_file = f"prompts_{args.model}.jsonl"

    BUCKET_NAME = f"{PROJECT_ID}-model-mind-control-baseline-batch-output"
    BUCKET_URI = f"gs://{BUCKET_NAME}"
    output_file_gcs = f"{BUCKET_URI}/{output_file}"

    print(f"Using model: {MODEL}")
    # Run batch job
    client = Client(
        vertexai=True, project=PROJECT_ID, location=model_location_map[args.model]
    )
    batch_job = client.batches.create(
        model=MODEL,
        src=output_file_gcs,
        config=CreateBatchJobConfig(dest=BUCKET_URI),
    )
    print(f"Batch job: {batch_job}")
    print(f"Job state: {batch_job.state.name}")

    completed_states = {
        JobState.JOB_STATE_SUCCEEDED,
        JobState.JOB_STATE_FAILED,
        JobState.JOB_STATE_CANCELLED,
        JobState.JOB_STATE_PAUSED,
    }

    while batch_job.state not in completed_states:
        time.sleep(30)
        batch_job = client.batches.get(name=batch_job.name)
        print(f"Job state: {batch_job.state}")

    # Check if the job succeeds
    if batch_job.state.name == "JOB_STATE_SUCCEEDED":
        print("Job succeeded!")
    else:
        print(f"Job failed: {batch_job.error}")

    # Check the location of the output
    print(f"Job output location: {batch_job.dest}")


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
        help="Model to use for predictions. Options: gemini-2.0-flash, claude-3.5-haiku, claude-opus-4, llama-3.1-8b, llama-3.1-405b.",
    )
    args = parser.parse_args()
    if args.model not in model_arg_map:
        raise ValueError(
            f"Model {args.model} not found in model_arg_map. Valid models are: {list(model_arg_map.keys())}"
        )

    main(args)
