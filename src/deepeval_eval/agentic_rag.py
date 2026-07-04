"""
Agentic RAG client for DeepEval evaluation.

Sends questions to the CAIPE supervisor via the A2A protocol using
message/send (synchronous) which returns the full response in one call.

Actual response structure from CAIPE supervisor (A2A protocol v0.3.0):
  result.status.state                    → "completed" | "failed"
  result.status.message.parts[].kind     → "text"
  result.status.message.parts[].text     → answer text
  result.artifacts[]                     → RAG context (if patch applied)
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


# Result model

@dataclass
class AgenticRAGResult:
    answer: str
    contexts: list[str]
    latency_ms: float
    task_id: str
    error: Optional[str] = None


# Agentic retriever

class AgenticRetriever:
    """
    Sends questions to the CAIPE supervisor A2A endpoint using message/send.

    Flow:
      1. POST message/send with messageId → returns full task result synchronously
      2. Extract answer from result.status.message.parts[].text
      3. Extract contexts from result.artifacts named "rag_context"
    """

    def __init__(
        self,
        supervisor_url: str = "http://localhost:8000",
        timeout: float = 200.0,
        logdir: Optional[str] = None,
    ):
        self.supervisor_url = supervisor_url.rstrip("/")
        self.timeout = timeout
        self.logdir = Path(logdir) if logdir else None
        if self.logdir:
            self.logdir.mkdir(parents=True, exist_ok=True)

    def _build_request(self, question: str) -> dict:
        """Build the A2A message/send request."""
        return {
            "jsonrpc": "2.0",
            "method": "message/send",
            "id": str(uuid.uuid4()),
            "params": {
                "message": {
                    "messageId": str(uuid.uuid4()),
                    "role": "user",
                    "parts": [{"kind": "text", "text": question}],
                }
            },
        }

    def _extract_answer(self, result: dict) -> str:
        """
        Extract final answer from artifacts named 'final_result'.
        Falls back to status.message.parts if not found.
        """
        try:
            # Primary: look for final_result artifact
            for artifact in result.get("artifacts", []):
                if artifact.get("name") == "final_result":
                    for part in artifact.get("parts", []):
                        text = part.get("text", "")
                        if text.strip():
                            return text.strip()

            # Fallback: status.message.parts
            parts = result.get("status", {}).get("message", {}).get("parts", [])
            texts = []
            for part in parts:
                if part.get("kind") == "text" and part.get("text", "").strip():
                    texts.append(part["text"].strip())
                elif "root" in part:
                    root = part["root"]
                    if root.get("text", "").strip():
                        texts.append(root["text"].strip())
            return "\n".join(texts)
        except Exception as e:
            logger.warning(f"Failed to extract answer: {e}")
            return ""

    def _extract_contexts(self, result: dict) -> list[str]:
        """
        Extract RAG context snippets from artifacts.
        Looks for 'rag_search_results' artifact with data.snippets[].content
        or 'rag_context' artifact with text parts.
        """
        contexts = []
        try:
            for artifact in result.get("artifacts", []):
                name = artifact.get("name", "")
                for part in artifact.get("parts", []):
                    if name == "rag_search_results":
                        # Structure: {"kind": "data", "data": {"snippets": [{"content": "..."}]}}
                        data = part.get("data", {})
                        if isinstance(data, dict):
                            for snippet in data.get("snippets", []):
                                content = snippet.get("content", "")
                                if content.strip():
                                    contexts.append(content.strip())
                    elif name == "rag_context":
                        # Structure: {"kind": "text", "text": "..."}
                        text = part.get("text", "")
                        if text.strip():
                            contexts.append(text.strip())
        except Exception as e:
            logger.warning(f"Failed to extract contexts: {e}")
        return contexts

    def retrieve(self, question: str) -> AgenticRAGResult:
        """
        Send a question to the CAIPE supervisor and return the answer + contexts.
        """
        start = time.perf_counter()
        request_body = self._build_request(question)

        try:
            resp = httpx.post(
                self.supervisor_url,
                json=request_body,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )
            latency_ms = (time.perf_counter() - start) * 1000
            resp.raise_for_status()

            response = resp.json()
            result = response.get("result", {})
            task_id = result.get("id", str(uuid.uuid4()))
            state = result.get("status", {}).get("state", "")

            # Log raw response
            if self.logdir:
                log_path = self.logdir / f"{task_id}.json"
                with open(log_path, "w") as f:
                    json.dump(result, f, indent=2)

            if state == "failed":
                error_parts = result.get("status", {}).get("message", {}).get("parts", [])
                error_msg = " ".join(p.get("text", "") for p in error_parts)
                return AgenticRAGResult(
                    answer="", contexts=[], latency_ms=latency_ms,
                    task_id=task_id, error=error_msg
                )

            answer = self._extract_answer(result)
            contexts = self._extract_contexts(result)

            if not contexts:
                logger.warning(
                    f"No RAG context found for task {task_id}. "
                    "Agent may have answered from training knowledge without RAG lookup."
                )

            logger.info(
                f"[{task_id}] state={state} answer_len={len(answer)} "
                f"contexts={len(contexts)} latency={latency_ms:.0f}ms"
            )

            return AgenticRAGResult(
                answer=answer,
                contexts=contexts,
                latency_ms=latency_ms,
                task_id=task_id,
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.error(f"A2A request failed: {e}")
            return AgenticRAGResult(
                answer="", contexts=[], latency_ms=latency_ms,
                task_id="", error=str(e),
            )