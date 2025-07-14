import argparse
import json
import os

import dotenv
from google.genai import Client
from google.genai.types import GenerateContentConfig

from utils.prompts import PARAPHRASE_PROMPT, PARAPHRASE_SYSTEM_INSTRUCTION
from utils.utils import DATA_DIR

dotenv.load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
MODEL = "gemini-2.0-flash-001"

client = Client(vertexai=True, project=PROJECT_ID, location="us-west1")


def generate_paraphrases(text: str) -> list[str]:
    response = client.models.generate_content(
        model=MODEL,
        config=GenerateContentConfig(system_instruction=PARAPHRASE_SYSTEM_INSTRUCTION),
        contents=[PARAPHRASE_PROMPT.format(social_issue_question=text)],
    )
    return response.text


def main(args):
    if args.sample:
        i = 0
    for dir in os.listdir(DATA_DIR):
        dir_path = os.path.join(DATA_DIR, dir)
        if not os.path.isdir(dir_path):
            continue
        json_path = (
            os.path.join(dir_path + ".json")
            if not dir.endswith(".json")
            else os.path.join(DATA_DIR, dir)
        )
        if not os.path.isfile(json_path):
            continue
        if args.sample:
            i += 1
            if i > 5:
                return
        print(f"Generating paraphrases for {dir}...")
        with open(json_path, "r") as f:
            data = json.load(f)
        question = data.get("question", "")
        if not question:
            continue
        paraphrases = generate_paraphrases(question)
        # Write paraphrases as a list of strings to the same JSON file
        data["paraphrases"] = (
            paraphrases if isinstance(paraphrases, list) else [paraphrases]
        )
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Get sample paraphrases for the first 5 debates",
    )
    args = parser.parse_args()
    main(args)
