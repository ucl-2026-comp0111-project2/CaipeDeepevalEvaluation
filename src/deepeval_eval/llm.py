from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import httpx
from pydantic import BaseModel


class OpenAICompatibleClient:
    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')

    def generate(self, prompt: str, schema: type[BaseModel] | None = None) -> str | BaseModel:
        if schema is not None:
            prompt = with_json_schema_instruction(prompt, schema)
        payload = {'model': self.model, 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0}
        headers = {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}
        with httpx.Client(timeout=300.0) as client:
            response = client.post(f'{self.base_url}/chat/completions', headers=headers, json=payload)
            response.raise_for_status()
            text = response.json()['choices'][0]['message']['content'] or ''
        if schema is None:
            return text.strip()
        return parse_schema_response(text, schema)


# DeepEval expects its own model interface; this adapter routes those judge
# calls through the Cisco LiteLLM OpenAI-compatible endpoint.
class DeepEvalJudge:
    def __init__(self, provider: str, model: str, client: OpenAICompatibleClient):
        from deepeval.models.base_model import DeepEvalBaseLLM

        class Judge(DeepEvalBaseLLM):
            def __init__(self, provider_name: str, model_name: str, llm_client: OpenAICompatibleClient):
                self.provider_name = provider_name
                self.model_name = model_name
                self.llm_client = llm_client
                super().__init__(model=model_name)

            def load_model(self, *args: Any, **kwargs: Any):
                return self.llm_client

            def get_model_name(self, *args: Any, **kwargs: Any) -> str:
                return f'{self.provider_name}:{self.model_name}'

            def generate(self, prompt: str, schema: type[BaseModel] | None = None, **kwargs: Any):
                return self.model.generate(prompt, schema=schema)

            async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None, **kwargs: Any):
                return await asyncio.to_thread(self.generate, prompt, schema=schema, **kwargs)

        self.model = Judge(provider, model, client)


def with_json_schema_instruction(prompt: str, schema: type[BaseModel]) -> str:
    return f'{prompt}\n\nReturn only valid JSON matching this JSON schema. Do not include markdown fences.\n\n{json.dumps(schema.model_json_schema(), indent=2)}'


def parse_schema_response(text: str, schema: type[BaseModel]) -> BaseModel:
    cleaned = text.strip()
    fence = chr(96) * 3
    if fence in cleaned:
        parts = [part.strip() for part in cleaned.split(fence) if part.strip()]
        if parts:
            cleaned = parts[-1]
            if cleaned.lower().startswith('json'):
                cleaned = cleaned[4:].strip()
    try:
        return schema.model_validate_json(cleaned)
    except Exception:
        match = re.search(r'(\{.*\}|\[.*\])', cleaned, flags=re.DOTALL)
        if not match:
            raise
        return schema.model_validate_json(match.group(1))


def make_generation_prompt(question: str, contexts: list[str]) -> str:
    context_block = '\n\n'.join(f'[{idx + 1}] {text}' for idx, text in enumerate(contexts))
    return 'Answer the question using only the context below. If the context is not enough, say that the answer is not in the provided context.\n\n' + f'Question:\n{question}\n\nContext:\n{context_block}\n\nAnswer:'


def make_short_answer_prompt(question: str, contexts: list[str]) -> str:
    context_block = '\n\n'.join(f'[{idx + 1}] {text}' for idx, text in enumerate(contexts))
    return 'Answer the HotpotQA question using only the context below. Keep the answer short. If the context is not enough, say that the answer is not in the provided context.\n\n' + f'Question:\n{question}\n\nContext:\n{context_block}\n\nAnswer:'
