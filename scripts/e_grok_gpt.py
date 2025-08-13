import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from xai_sdk import AsyncClient
from xai_sdk.chat import Response, user

from utils.utils import PREDICTIONS_DIR, PROMPTS_DIR, cost_map

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
XAI_API_KEY = os.getenv("XAI_API_KEY")


def do_openai_predictions(input_file, output_file):
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Upload batch input file
    print(f"Uploading batch input file...")
    batch_input_file = client.files.create(
        file=open(input_file, "rb"),
        purpose="batch"
    )
    print(f"Uploaded batch input file with ID: {batch_input_file.id}")
    
    # Create batch
    print(f"Creating batch...")
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "evidence testing"
        }
    )
    print(f"Created batch with ID: {batch.id}")
    
    # Poll for batch completion
    print(f"Waiting for batch to complete...")
    while True:
        batch = client.batches.retrieve(batch.id)
        status = batch.status
        
        print(f"Batch status: {status}")
        print(f"Progress: {batch.request_counts.completed}/{batch.request_counts.total} completed, {batch.request_counts.failed} failed")
        
        if status == "completed":
            break
        elif status in ["failed", "expired", "cancelled"]:
            print(f"Batch {status}. Exiting.")
            return
        
        # Wait before checking again
        time.sleep(60)  # Check every minute
    
    # Download batch output
    print(f"Downloading batch output...")
    output_file_response = client.files.content(batch.output_file_id)

    # Write output as jsonl
    # OpenAI batch output is a JSON array, so we need to write each item as a line
    output_json = output_file_response.json()
    with open(output_file, "w") as f:
        if isinstance(output_json, list):
            for item in output_json:
                f.write(json.dumps(item) + "\n")
        else:
            # fallback: write as a single line if not a list
            f.write(json.dumps(output_json) + "\n")

async def do_grok_predictions(input_file, output_file):
    client = AsyncClient(
        api_key=XAI_API_KEY,
        timeout=3600,  # Override default timeout with longer timeout for reasoning models
    )

    model = input_file.split('_')[1].split('.')[0]
    max_rpm = 480 if 'mini' in model else 600

    # Use a semaphore to ensure we do not exceed the maximum requests per minute (rpm)
    # We'll allow up to (max_rpm // 60) requests per second, but also enforce a minimum delay between requests
    max_requests_per_second = max_rpm // 60
    semaphore = asyncio.Semaphore(max_requests_per_second)

    class RateLimiter:
        def __init__(self, semaphore):
            self.semaphore = semaphore
        
        async def __aenter__(self):
            await self.semaphore.acquire()
            return self
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            # Release after 1 second to maintain rpm limit
            await asyncio.sleep(1)
            self.semaphore.release()

    async def process_request(request) -> Response:
        try:
            async with RateLimiter(semaphore):
                # print(f"Processing request: {request}")
                chat = client.chat.create(model=model)
                chat.append(user(request))
                response = await chat.sample()
                # print(f"Response content: {response.content}")
                # print(f"Response _choice: {response._choice}")
                # print(f"Response finish_reason: {response.finish_reason}")
                # print(f"Response role: {response.role}")
                # print(f"Response reasoning_content: {response.reasoning_content}")
                # print(f"Response type: {type(response)}")
                return response
        except Exception as e:
            print(f"Error processing request: {e}")
            # Return a mock response with error info
            class MockResponse:
                def __init__(self, error_msg):
                    self.content = f"ERROR: {error_msg}"
                    self.reasoning_content = f"ERROR: {error_msg}"
            return MockResponse(str(e))

    tasks = []
    with open(input_file, "r") as f:
        for line in f:
            data = json.loads(line)
            # Store both custom_id and the coroutine for later association
            tasks.append((data.get('custom_id'), process_request(data['prompt'])))

    # Process requests in batches to save incrementally
    batch_size = 100  # Save every 100 responses
    with open(output_file, "w") as f:
        for i in range(0, len(tasks), batch_size):
            batch_tasks = tasks[i:i + batch_size]
            custom_ids, coros = zip(*batch_tasks)
            
            try:
                print(f"Processing batch {i//batch_size + 1}/{(len(tasks) + batch_size - 1)//batch_size}")
                responses = await asyncio.gather(*coros, return_exceptions=True)
                
                for j, (custom_id, response) in enumerate(zip(custom_ids, responses)):
                    if isinstance(response, Exception):
                        print(f"Error for {custom_id}: {response}")
                        f.write(json.dumps({
                            'custom_id': custom_id,
                            'response': f"ERROR: {response}"
                        }) + "\n")
                    else:
                        # print(f"Writing response for {custom_id}: {response.reasoning_content}")
                        f.write(json.dumps({
                            'custom_id': custom_id,
                            'response': response.content
                        }) + "\n")
                
                # Flush to disk after each batch
                f.flush()
                print(f"Completed batch {i//batch_size + 1}")
                
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}")
                # Write error responses for this batch
                for custom_id, _ in batch_tasks:
                    f.write(json.dumps({
                        'custom_id': custom_id,
                        'response': f"ERROR: {e}"
                    }) + "\n")
                f.flush()

def estimate_tokens(input_file, model):
    total_input_tokens = 0
    total_output_tokens = 0
    output_to_input_ratio = 0.005

    # Use tiktoken for GPT models, fallback for Grok
    if 'gpt' in model:
        encoding = tiktoken.encoding_for_model(model)
    elif 'grok' in model:
        # Use cl100k_base as a rough approximation for Grok
        encoding = tiktoken.get_encoding("cl100k_base")
    else:
        encoding = None

    with open(input_file, "r") as f:
        for line in f:
            data = json.loads(line)
            if 'gpt' in model:
                prompt = data['body']['messages'][0]['content']
            elif 'grok' in model:
                prompt = data['prompt']
            else:
                prompt = data['prompt']
            if encoding is not None:
                tokens = len(encoding.encode(prompt))
            else:
                # Fallback: estimate 1 token per 4 chars
                tokens = len(prompt) // 4
            total_input_tokens += tokens
            total_output_tokens += tokens * output_to_input_ratio

    input_cost = total_input_tokens * cost_map[model]['input'] / 1000000
    output_cost = total_output_tokens * cost_map[model]['output'] / 1000000

    print(f"Model: {model}")
    print(f"Total input tokens: {total_input_tokens:,}")
    print(f"Predicted input cost: ${input_cost:,.2f}")
    print(f"Total predicted output tokens: {total_output_tokens:,.0f}")
    print(f"Predicted output cost: ${output_cost:,.2f}")
    print(f"Total cost: ${input_cost + output_cost:,.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--estimate_tokens', action='store_true')
    args = parser.parse_args()

    if args.estimate_tokens:
        estimate_tokens(
            input_file=os.path.join(PROMPTS_DIR, f'prompts_{args.model}.jsonl'),
            model=args.model
        )
    else:
        if 'gpt' in args.model:
            do_openai_predictions(
                input_file=os.path.join(PROMPTS_DIR, f'prompts_{args.model}.jsonl'),
                output_file=os.path.join(PREDICTIONS_DIR, f'predictions_{args.model}.jsonl')
            )
        elif 'grok' in args.model:
            asyncio.run(do_grok_predictions(
                input_file=os.path.join(PROMPTS_DIR, f'prompts_{args.model}.jsonl'),
                output_file=os.path.join(PREDICTIONS_DIR, f'predictions_{args.model}.jsonl')
            ))

