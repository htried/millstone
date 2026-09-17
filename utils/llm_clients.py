import asyncio
import os
import time

from dotenv import load_dotenv
from openai import OpenAI
import requests

load_dotenv()


MODEL_PROVIDER_MAP = {
    "gemini-2.0-flash": "google",
    "gpt-4o": "openai",
    "grok-3": "xai",
}

GOOGLE_MODEL_MAP = {
    "gemini-2.0-flash": "gemini-2.0-flash-001",
}

GOOGLE_REST_FALLBACKS = {
    "gemini-2.0-flash": ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-flash"],
}


class LLMRouter:
    def __init__(self, project_id=None, google_location="us-central1", timeout_sec=120):
        self.project_id = project_id or os.getenv("PROJECT_ID")
        self.google_location = google_location
        self.timeout_sec = timeout_sec
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.xai_api_key = os.getenv("XAI_API_KEY")
        self._google_client = None
        self._google_mode = None
        self._google_models = {}
        self._openai_client = None
        self._xai_client = None

    def _init_google(self):
        if self._google_client is not None:
            return
        # Prefer direct REST path (no SDK) to avoid grpc/aio teardown issues.
        if self.gemini_api_key:
            self._google_client = "rest"
            self._google_mode = "rest"
            return

        if not self.project_id:
            raise RuntimeError("Either GEMINI_API_KEY or PROJECT_ID is required for Gemini calls.")
        from google.genai import Client

        self._google_client = Client(
            vertexai=True,
            project=self.project_id,
            location=self.google_location,
        )
        self._google_mode = "vertex"

    def _init_openai(self):
        if self._openai_client is not None:
            return
        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI calls.")
        self._openai_client = OpenAI(api_key=self.openai_api_key, timeout=self.timeout_sec)

    def _init_xai(self):
        if self._xai_client is not None:
            return
        if not self.xai_api_key:
            raise RuntimeError("XAI_API_KEY is required for Grok calls.")
        # Use xAI's OpenAI-compatible endpoint to avoid async event-loop issues.
        self._xai_client = OpenAI(
            api_key=self.xai_api_key,
            base_url="https://api.x.ai/v1",
            timeout=self.timeout_sec,
        )

    def _call_google(self, model_name, prompt):
        self._init_google()
        if self._google_mode == "rest":
            # REST endpoint expects public model ids, not Vertex publisher ids.
            last_error = None
            candidates_models = GOOGLE_REST_FALLBACKS.get(model_name, [model_name])
            for mapped_model in candidates_models:
                url = (
                    f"https://generativelanguage.googleapis.com/v1beta/models/"
                    f"{mapped_model}:generateContent?key={self.gemini_api_key}"
                )
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0, "maxOutputTokens": 300},
                }
                response = requests.post(url, json=payload, timeout=self.timeout_sec)
                if response.status_code == 404:
                    last_error = f"model {mapped_model} unavailable"
                    continue
                if response.status_code >= 400:
                    preview = response.text[:400].replace("\n", " ")
                    raise RuntimeError(f"Gemini REST error {response.status_code}: {preview}")
                data = response.json()
                response_candidates = data.get("candidates", [])
                if not response_candidates:
                    return ""
                parts = response_candidates[0].get("content", {}).get("parts", [])
                if not parts:
                    return ""
                return "".join(p.get("text", "") for p in parts if isinstance(p, dict))

            raise RuntimeError(f"No available Gemini REST model for '{model_name}' ({last_error})")

        mapped_model = GOOGLE_MODEL_MAP.get(model_name, model_name)
        response = self._google_client.models.generate_content(
            model=mapped_model,
            contents=[prompt],
        )
        return response.text or ""

    def _call_openai(self, model_name, prompt):
        self._init_openai()
        response = self._openai_client.chat.completions.create(
            model=model_name,
            temperature=0,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    def _call_xai(self, model_name, prompt):
        self._init_xai()
        response = self._xai_client.chat.completions.create(
            model=model_name,
            temperature=0,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    def call(self, model_name, prompt, max_retries=3, retry_sleep_sec=2.0):
        provider = MODEL_PROVIDER_MAP.get(model_name)
        if provider is None:
            raise ValueError(
                f"Unsupported model '{model_name}'. Supported models: {sorted(MODEL_PROVIDER_MAP.keys())}"
            )

        last_error = None
        for attempt in range(max_retries):
            try:
                if provider == "google":
                    return self._call_google(model_name, prompt)
                if provider == "openai":
                    return self._call_openai(model_name, prompt)
                if provider == "xai":
                    return self._call_xai(model_name, prompt)
                raise ValueError(f"Unknown provider: {provider}")
            except Exception as exc:
                last_error = exc
                if attempt < max_retries - 1:
                    time.sleep(retry_sleep_sec * (attempt + 1))
                else:
                    raise RuntimeError(
                        f"Failed calling model '{model_name}' after {max_retries} attempts: {exc}"
                    ) from exc

        raise RuntimeError(f"Unexpected failure for model '{model_name}': {last_error}")
