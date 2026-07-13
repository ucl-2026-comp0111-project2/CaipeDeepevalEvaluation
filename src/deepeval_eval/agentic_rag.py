"""
Agentic RAG client for DeepEval evaluation.

Uses message/stream (SSE) to capture rag_context artifacts emitted
during agent execution, alongside the final answer and token usage.

SSE event structure observed from CAIPE supervisor:
  - result.artifact.name == "rag_context"        → RAG context (in parts[].text)
  - result.artifact.name == "final_result"        → final answer (in parts[].text)
  - result.final == true, result.kind == "status-update" → task complete + usage_metadata
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class AgenticRAGResult:
    answer: str
    contexts: list[str]
    latency_ms: float
    task_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    error: Optional[str] = None


class AgenticRetriever:
    """
    Sends questions to the CAIPE supervisor using SSE streaming (message/stream).
    Collects rag_context artifacts emitted during agent execution,
    extracts the final answer, and tracks token usage from usage_metadata.
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

    def _build_stream_request(self, question: str) -> dict:
        return {
            "jsonrpc": "2.0",
            "method": "message/stream",
            "id": str(uuid.uuid4()),
            "params": {
                "message": {
                    "messageId": str(uuid.uuid4()),
                    "role": "user",
                    "parts": [{"kind": "text", "text": question}],
                }
            },
        }

    def _extract_text_from_parts(self, artifact: dict) -> str:
        """Extract text from artifact parts or direct text field."""
        for part in artifact.get("parts", []):
            text = part.get("text", "")
            if text.strip():
                return text.strip()
        return artifact.get("text", "").strip()

    def retrieve(self, question: str) -> AgenticRAGResult:
        """
        Send a question via SSE streaming, collect rag_context artifacts,
        extract the final answer, and capture token usage metrics.
        """
        task_id = str(uuid.uuid4())
        start = time.perf_counter()
        request_body = self._build_stream_request(question)

        contexts: list[str] = []
        answer = ""
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        raw_events: list[dict] = []

        try:
            with httpx.stream(
                "POST",
                self.supervisor_url,
                json=request_body,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            ) as resp:
                resp.raise_for_status()

                for chunk in resp.iter_text():
                    for line in chunk.split("\n"):
                        line = line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if not data:
                            continue
                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        raw_events.append(event)
                        result = event.get("result", {})
                        artifact = result.get("artifact", {})
                        artifact_name = artifact.get("name", "")

                        # Collect rag_context — content in parts[].text
                        if artifact_name == "rag_context":
                            text = self._extract_text_from_parts(artifact)
                            if text:
                                contexts.append(text)
                                logger.debug(f"Captured rag_context: {len(text)} chars")

                        # Extract answer from final_result artifact
                        elif artifact_name == "final_result":
                            text = self._extract_text_from_parts(artifact)
                            if text:
                                answer = text

                        # Detect task completion and extract token usage
                        if result.get("final") and result.get("kind") == "status-update":
                            task_id = result.get("taskId", task_id)
                            state = result.get("status", {}).get("state", "")
                            usage = result.get("metadata", {}).get("usage_metadata", {})
                            input_tokens = usage.get("input_tokens", 0)
                            output_tokens = usage.get("output_tokens", 0)
                            total_tokens = usage.get("total_tokens", 0)
                            logger.info(
                                f"Task {task_id} completed state={state} "
                                f"tokens={total_tokens} (in={input_tokens} out={output_tokens})"
                            )

            latency_ms = (time.perf_counter() - start) * 1000

            # Log raw events for debugging
            if self.logdir:
                log_path = self.logdir / f"{task_id}.json"
                with open(log_path, "w") as f:
                    json.dump(raw_events, f, indent=2)

            if not contexts:
                logger.warning(
                    f"No rag_context artifacts for task {task_id}. "
                    "Agent may have answered from training knowledge without RAG lookup."
                )

            logger.info(
                f"[{task_id}] answer_len={len(answer)} contexts={len(contexts)} "
                f"latency={latency_ms:.0f}ms tokens={total_tokens}"
            )

            return AgenticRAGResult(
                answer=answer,
                contexts=contexts,
                latency_ms=latency_ms,
                task_id=task_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.error(f"A2A stream request failed: {e}")
            return AgenticRAGResult(
                answer="",
                contexts=[],
                latency_ms=latency_ms,
                task_id=task_id,
                error=str(e),
            )