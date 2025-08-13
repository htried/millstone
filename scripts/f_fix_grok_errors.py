import asyncio
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from dotenv import load_dotenv
from xai_sdk import AsyncClient
from xai_sdk.chat import Response, user

from utils.utils import PREDICTIONS_DIR, PROMPTS_DIR

load_dotenv()

XAI_API_KEY = os.getenv("XAI_API_KEY")


def find_error_custom_ids(predictions_file):
    """Find custom_ids of responses that contain errors."""
    error_custom_ids = []
    
    with open(predictions_file, "r") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if "response" in data and data["response"].startswith("ERROR:"):
                    error_custom_ids.append(data["custom_id"])
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line: {line.strip()}")
                continue
    
    print(f"Found {len(error_custom_ids)} error responses")
    return error_custom_ids


def extract_prompts_for_errors(prompts_file, error_custom_ids):
    """Extract prompts for the given custom_ids from the prompts file."""
    prompts_to_retry = []
    error_custom_ids_set = set(error_custom_ids)
    
    with open(prompts_file, "r") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if data["custom_id"] in error_custom_ids_set:
                    prompts_to_retry.append(data)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line in prompts file: {line.strip()}")
                continue
    
    print(f"Extracted {len(prompts_to_retry)} prompts for retry")
    return prompts_to_retry


async def retry_grok_predictions(prompts_to_retry, output_file):
    """Retry predictions for the given prompts using Grok-3."""
    client = AsyncClient(
        api_key=XAI_API_KEY,
        timeout=3600,
    )

    model = "grok-3"
    max_rpm = 400

    # Use a semaphore to ensure we do not exceed the maximum requests per minute (rpm)
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

    async def process_request(request_data) -> Response:
        try:
            async with RateLimiter(semaphore):
                chat = client.chat.create(model=model)
                chat.append(user(request_data["prompt"]))
                response = await chat.sample()
                return response
        except Exception as e:
            print(f"Error processing request {request_data['custom_id']}: {e}")
            # Return a mock response with error info
            class MockResponse:
                def __init__(self, error_msg):
                    self.content = f"ERROR: {error_msg}"
                    self.reasoning_content = f"ERROR: {error_msg}"
            return MockResponse(str(e))

    # Process requests in batches to save incrementally
    batch_size = 50  # Smaller batch size for retries
    with open(output_file, "w") as f:
        for i in range(0, len(prompts_to_retry), batch_size):
            batch_prompts = prompts_to_retry[i:i + batch_size]
            
            try:
                print(f"Processing retry batch {i//batch_size + 1}/{(len(prompts_to_retry) + batch_size - 1)//batch_size}")
                tasks = [process_request(prompt_data) for prompt_data in batch_prompts]
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                for j, (prompt_data, response) in enumerate(zip(batch_prompts, responses)):
                    if isinstance(response, Exception):
                        print(f"Error for {prompt_data['custom_id']}: {response}")
                        f.write(json.dumps({
                            'custom_id': prompt_data['custom_id'],
                            'response': f"ERROR: {response}"
                        }) + "\n")
                    else:
                        f.write(json.dumps({
                            'custom_id': prompt_data['custom_id'],
                            'response': response.content
                        }) + "\n")
                
                # Flush to disk after each batch
                f.flush()
                print(f"Completed retry batch {i//batch_size + 1}")
                
            except Exception as e:
                print(f"Error processing retry batch {i//batch_size + 1}: {e}")
                # Write error responses for this batch
                for prompt_data in batch_prompts:
                    f.write(json.dumps({
                        'custom_id': prompt_data['custom_id'],
                        'response': f"ERROR: {e}"
                    }) + "\n")
                f.flush()


def merge_predictions(original_file, retry_file, output_file):
    """Merge original predictions with retry predictions, replacing error responses."""
    # Read original predictions
    original_predictions = {}
    with open(original_file, "r") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                original_predictions[data["custom_id"]] = data
            except json.JSONDecodeError:
                continue
    
    # Read retry predictions and update original
    with open(retry_file, "r") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                original_predictions[data["custom_id"]] = data
            except json.JSONDecodeError:
                continue
    
    # Write merged predictions
    with open(output_file, "w") as f:
        for custom_id in sorted(original_predictions.keys()):
            f.write(json.dumps(original_predictions[custom_id]) + "\n")
    
    print(f"Merged predictions written to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Fix Grok-3 prediction errors by retrying failed requests")
    parser.add_argument('--predictions_file', type=str, 
                       default=os.path.join(PREDICTIONS_DIR, 'predictions_grok-3.jsonl'),
                       help='Path to the predictions file with errors')
    parser.add_argument('--prompts_file', type=str,
                       default=os.path.join(PROMPTS_DIR, 'prompts_grok-3.jsonl'),
                       help='Path to the prompts file')
    parser.add_argument('--retry_output', type=str,
                       default=os.path.join(PREDICTIONS_DIR, 'predictions_grok-3_retry.jsonl'),
                       help='Output file for retry predictions')
    parser.add_argument('--merged_output', type=str,
                       default=os.path.join(PREDICTIONS_DIR, 'predictions_grok-3_fixed.jsonl'),
                       help='Output file for merged predictions')
    parser.add_argument('--skip_retry', action='store_true',
                       help='Skip the retry step and only merge existing retry file')
    
    args = parser.parse_args()
    
    print("Step 1: Finding error custom_ids...")
    error_custom_ids = find_error_custom_ids(args.predictions_file)
    
    if not error_custom_ids:
        print("No errors found. Exiting.")
        return
    
    print("Step 2: Extracting prompts for errors...")
    prompts_to_retry = extract_prompts_for_errors(args.prompts_file, error_custom_ids)
    
    if not prompts_to_retry:
        print("No prompts found for retry. Exiting.")
        return
    
    if not args.skip_retry:
        print("Step 3: Retrying predictions...")
        asyncio.run(retry_grok_predictions(prompts_to_retry, args.retry_output))
    
    print("Step 4: Merging predictions...")
    merge_predictions(args.predictions_file, args.retry_output, args.merged_output)
    
    print("Done!")


if __name__ == "__main__":
    main() 