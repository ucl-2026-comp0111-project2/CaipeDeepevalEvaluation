"""
Agentic RAG client for DeepEval evaluation.

Uses either direct A2A (/message/send) or SSE streaming (/api/v1/chat/stream/start)
to capture rag_context artifacts emitted during agent execution, alongside the final
answer and token usage.

SSE event structure observed from CAIPE supervisor:
  - result.artifact.name == "rag_context"        → RAG context (in parts[].text)
  - result.artifact.name == "final_result"        → final answer (in parts[].text)
  - result.final == true, result.kind == "status-update" → task complete + usage_metadata

SSE event structure observed from agent gateway:
  - event: content                              → final answer text pieces (in data.text)
  - event: tool_end                             → tool outputs / RAG contexts (in data.result)
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx
import requests

from deepeval_eval.io_utils import sanitize_path

logger = logging.getLogger(__name__)

# ============================================================
# Stubs to match base class structure and API
# ============================================================


class BaseRetriever:
    def __init__(self) -> None:
        self.documents: list[str] = []
        self.documents_metadata: list[dict[str, Any]] = []


class TraceEvent:
    def __init__(self, event_type: str, component: str, data: dict[str, Any]) -> None:
        self.event_type = event_type
        self.component = component
        self.data = data
        self.timestamp = datetime.now()


class BaseRAG:
    def __init__(
        self,
        llm_client: Any = None,
        model_name: str = "agentic",
        retriever: AgenticRetriever | None = None,
        logdir: str = "logs",
    ) -> None:
        self.llm_client = llm_client
        self.model_name = model_name
        self.retriever = retriever
        self.logdir = logdir
        self.traces: list[TraceEvent] = []

    def export_traces_to_log(
        self,
        run_id: str,
        question: str,
        result: dict[str, Any] | None,
    ) -> str:
        """Export traces to a JSON log file, matching BaseRAG's method."""
        os.makedirs(self.logdir, exist_ok=True)
        safe_run_id = sanitize_path(run_id) or "default"
        log_path = os.path.join(self.logdir, f"query_trace_{safe_run_id}.json")
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "question": question,
                        "result": result,
                        "traces": [
                            {
                                "event_type": t.event_type,
                                "component": t.component,
                                "data": t.data,
                                "timestamp": t.timestamp.isoformat(),
                            }
                            for t in self.traces
                        ],
                    },
                    f,
                    indent=2,
                )
        except Exception:
            logger.exception("Failed to write query trace log %s", log_path)
        return log_path


# ============================================================
# Helper functions for markdown cleaning and parsing
# ============================================================

_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_SNIPPET_PREFIX_RE = re.compile(r"^\s*\*\*Snippet:\*\*\s*")
_ELLIPSIS_RE = re.compile(r"\.{3,}")


def clean_snippet_markdown(text: str) -> str:
    """Strip bold/ellipsis display markup from search tool snippets.

    The search tool returns UI-formatted snippets e.g.
    '**Snippet:** ...**CAIPE** uses nomic-embed-text...'.
    Stripping gives plain prose and avoids WAF 403s.
    """
    if not text:
        return text
    cleaned = _SNIPPET_PREFIX_RE.sub("", text)
    cleaned = _ELLIPSIS_RE.sub(" ", cleaned)
    cleaned = _BOLD_RE.sub(r"\1", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _extract_text_from_parts(parts: list) -> str:
    """Concatenate response parts into a single string."""
    return "".join(p.get("text", "") for p in parts if p.get("kind") == "text")


def _parse_rag_context_artifact(text: Any) -> list:
    """Parse a rag_context artifact into (content, doc_id) tuples.

    Handles both tool shapes:
      - search:         {"semantic_results": [...], "keyword_results": [...]}
      - fetch_document: [{"document": {"page_content": ..., "document_id": ...}}]
    """
    out = []
    if isinstance(text, (dict, list)):
        data = text
    else:
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return out

    if isinstance(data, dict):
        for key in ("semantic_results", "keyword_results"):
            for item in data.get(key, []) or []:
                txt = item.get("text_content")
                if txt:
                    meta = (
                        item.get("metadata", {})
                        if isinstance(item.get("metadata"), dict)
                        else {}
                    )
                    doc_id = (
                        meta.get("document_id")
                        or meta.get("doc_id")
                        or item.get("document_id")
                    )
                    resolved_id = str(doc_id) if doc_id is not None else None
                    out.append((clean_snippet_markdown(txt), resolved_id))
                    logger.info(
                        "Snippet: %s | DocID: %s",
                        clean_snippet_markdown(txt),
                        resolved_id,
                    )
    elif isinstance(data, list):
        for item in data:
            doc = item.get("document", {}) if isinstance(item, dict) else {}
            txt = doc.get("page_content")
            if txt:
                doc_meta = (
                    doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
                )
                doc_id = (
                    doc.get("document_id")
                    or doc.get("doc_id")
                    or doc_meta.get("document_id")
                    or doc_meta.get("doc_id")
                    or item.get("document_id")
                    or item.get("doc_id")
                )
                out.append((txt, str(doc_id) if doc_id is not None else None))
    return out


def _dedupe_preserve_order(items: list) -> list:
    """Deduplicate (content, doc_id) tuples by content, preserving order."""
    seen = set()
    result = []
    for item in items:
        content = item[0] if isinstance(item, tuple) else item
        if content not in seen:
            seen.add(content)
            result.append(item)
    return result


def _dedupe_and_merge_contexts(items: list) -> list:
    """Deduplicate and merge contexts by doc_id, preferring longer/full content."""
    doc_id_to_content = {}
    ordered_keys = []

    for item in items:
        if not isinstance(item, tuple) or len(item) < 2:
            continue
        content, doc_id = item
        if doc_id:
            if doc_id not in doc_id_to_content:
                ordered_keys.append(doc_id)
                doc_id_to_content[doc_id] = content
            else:
                if len(content) > len(doc_id_to_content[doc_id]):
                    doc_id_to_content[doc_id] = content
        else:
            content_key = f"content_hash:{hash(content)}"
            if content_key not in doc_id_to_content:
                ordered_keys.append(content_key)
                doc_id_to_content[content_key] = content

    result = []
    for key in ordered_keys:
        content = doc_id_to_content[key]
        resolved_doc_id = None if key.startswith("content_hash:") else key
        result.append((content, resolved_doc_id))
    return result


# ============================================================
# Return type for DeepEval specific backward compatibility
# ============================================================


@dataclass
class AgenticRAGResult:
    answer: str
    contexts: list[str]
    latency_ms: float
    task_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    error: str | None = None


# ============================================================
# Agentic retrieval — queries supervisor or BFF gateway
# ============================================================


class AgenticRetriever(BaseRetriever):
    """Retriever that queries caipe-supervisor's endpoint."""

    def __init__(
        self,
        agent_api_url: str | None = None,
        timeout: float = 120.0,
        insecure: bool = False,
        use_a2a: bool | None = None,
        trace_log: bool = False,
        logdir: str = "logs",
        supervisor_url: str | None = None,  # for compatibility
        fail_on_error: bool = False,
        datasource_id: str | None = None,
        agent_id: str | None = None,
    ) -> None:
        super().__init__()
        self.datasource_id = datasource_id
        self.agent_id = agent_id or os.getenv("CAIPE_AGENT_ID") or "hello-world"
        self.agent_api_url = (
            agent_api_url
            or supervisor_url
            or os.getenv("CAIPE_SUPERVISOR_URL")
            or "http://localhost:8000"
        )
        self.timeout = timeout
        self.insecure = insecure or os.getenv("INSECURE_SSL", "").lower() in (
            "true",
            "1",
            "yes",
        )
        self.last_answer: str = ""
        self.last_raw_response: dict | None = None
        self.documents_metadata: list[dict[str, Any]] = []
        self.trace_log = trace_log
        self.logdir = logdir
        self.fail_on_error = fail_on_error

        if use_a2a is not None:
            self.use_a2a = use_a2a
        else:
            env_val = os.getenv("CAIPE_USE_A2A")
            if env_val is not None:
                self.use_a2a = env_val.lower() in ("true", "1", "yes")
            else:
                self.use_a2a = False

    def fit(self, documents: list[str]) -> None:
        """AgenticRetriever doesn't support local fitting."""
        self.documents = documents
        self.documents_metadata = [{} for _ in documents]

    def _call_supervisor(self, question: str) -> dict | None:
        """Send a question to caipe-supervisor's A2A message/send endpoint."""
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": question}],
                    "messageId": str(uuid.uuid4()),
                }
            },
        }
        try:
            response = requests.post(
                self.agent_api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
                verify=not self.insecure,
            )
            response.raise_for_status()
            return response.json()
        except requests.Timeout:
            logger.error(
                "Timeout calling caipe-supervisor (%.1fs) — increase --supervisor-timeout",
                self.timeout,
            )
            return None
        except requests.HTTPError as exc:
            logger.error(
                "HTTP %s from caipe-supervisor: %s",
                exc.response.status_code,
                exc,
            )
            return None
        except Exception:
            logger.exception("Unexpected error calling caipe-supervisor A2A endpoint")
            return None

    def _get_oidc_token(self) -> str | None:
        """Fetch OIDC token dynamically using client credentials, falling back to environment variables."""
        client_id = os.getenv("CAIPE_CLIENT_ID") or os.getenv("CLIENT_ID")
        client_secret = os.getenv("CAIPE_CLIENT_SECRET") or os.getenv("CLIENT_SECRET")

        if not client_id or not client_secret:
            logger.info(
                "Credentials not in environment. Attempting to fetch from Kubernetes secret 'caipe-ui-secret'..."
            )
            try:
                client_id_cmd = "kubectl get secret caipe-ui-secret -n caipe -o jsonpath='{.data.OIDC_CLIENT_ID}'"
                client_secret_cmd = "kubectl get secret caipe-ui-secret -n caipe -o jsonpath='{.data.OIDC_CLIENT_SECRET}'"
                client_id_b64 = (
                    subprocess.check_output(
                        client_id_cmd, shell=True, stderr=subprocess.DEVNULL
                    )
                    .decode()
                    .strip()
                )
                client_secret_b64 = (
                    subprocess.check_output(
                        client_secret_cmd, shell=True, stderr=subprocess.DEVNULL
                    )
                    .decode()
                    .strip()
                )
                if client_id_b64 and client_secret_b64:
                    client_id = base64.b64decode(client_id_b64).decode()
                    client_secret = base64.b64decode(client_secret_b64).decode()
                    os.environ["CAIPE_CLIENT_ID"] = client_id
                    os.environ["CAIPE_CLIENT_SECRET"] = client_secret
                    logger.info(
                        "Successfully fetched OIDC credentials from Kubernetes."
                    )
            except Exception as e:
                logger.debug("Could not fetch credentials from Kubernetes: %s", e)

        if client_id and client_secret:
            try:
                keycloak_url = os.getenv("CAIPE_OIDC_TOKEN_URL") or os.getenv(
                    "CAIPE_KEYCLOAK_URL"
                )
                if not keycloak_url:
                    if "caipe.homelab" in self.agent_api_url:
                        keycloak_url = "https://keycloak.caipe.homelab/realms/caipe/protocol/openid-connect/token"
                    else:
                        keycloak_url = "http://localhost:7080/realms/caipe/protocol/openid-connect/token"

                logger.info(
                    "Fetching a fresh OIDC token from Keycloak: %s", keycloak_url
                )
                resp = httpx.post(
                    keycloak_url,
                    data={
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "grant_type": "client_credentials",
                    },
                    verify=not self.insecure,
                    timeout=15.0,
                )
                resp.raise_for_status()
                token = resp.json().get("access_token")
                if token:
                    os.environ["CAIPE_OIDC_TOKEN"] = token
                    return token
            except Exception as e:
                logger.error("Failed to fetch fresh OIDC token from Keycloak: %s", e)

        return os.getenv("CAIPE_OIDC_TOKEN") or os.getenv("BEARER_TOKEN")

    def _query_gateway(
        self,
        question: str,
        k: int = 3,
        run_id: str | None = None,
        trace_log: bool | None = None,
    ) -> list[tuple]:
        """Send query to the streaming BFF gateway endpoints."""
        max_attempts = 3
        backoff_base = 2.0
        last_exception = None

        for attempt in range(1, max_attempts + 1):
            token = self._get_oidc_token()
            headers = {"Content-Type": "application/json"}
            if token:
                headers["Authorization"] = f"Bearer {token}"

            agent_id = self.agent_id

            try:
                # Step 1: Create a new conversation session
                conv_url = f"{self.agent_api_url.rstrip('/')}/api/chat/conversations"
                conv_payload = {
                    "title": "Agentic Session",
                    "client_type": "webui",
                    "agent_id": agent_id,
                }

                logger.info(
                    "Creating conversation session on %s... (Attempt %d/%d)",
                    conv_url,
                    attempt,
                    max_attempts,
                )
                r_conv = httpx.post(
                    conv_url,
                    json=conv_payload,
                    headers=headers,
                    verify=not self.insecure,
                    timeout=self.timeout,
                )
                if r_conv.status_code == 401:
                    logger.warning(
                        "Gateway returned 401 Unauthorized. Attempting token refresh..."
                    )
                    if os.getenv("CAIPE_OIDC_TOKEN"):
                        del os.environ["CAIPE_OIDC_TOKEN"]
                    token = self._get_oidc_token()
                    if token:
                        headers["Authorization"] = f"Bearer {token}"
                        logger.info(
                            "Retrying conversation session creation with fresh token..."
                        )
                        r_conv = httpx.post(
                            conv_url,
                            json=conv_payload,
                            headers=headers,
                            verify=not self.insecure,
                            timeout=self.timeout,
                        )
                r_conv.raise_for_status()
                conv_data = r_conv.json()
                conversation_id = conv_data["data"]["conversation"]["_id"]
                logger.info("Conversation session created with ID: %s", conversation_id)

                # Step 2: Stream the chat start request
                stream_url = (
                    f"{self.agent_api_url.rstrip('/')}/api/v1/chat/stream/start"
                )
                stream_payload = {
                    "message": question,
                    "conversation_id": conversation_id,
                    "agent_id": agent_id,
                    "protocol": "custom",
                    "client_context": {
                        "source": "eval",
                        "tool_result_display_limit": -1,
                    },
                }

                raw_contexts = []
                self.last_answer = ""

                # Resolve trace_log
                should_trace = trace_log
                if should_trace is None:
                    should_trace = self.trace_log
                if not should_trace:
                    env_val = os.getenv("CAIPE_TRACE_LOG")
                    if env_val is not None:
                        should_trace = env_val.lower() in ("true", "1", "yes")

                log_file = None
                if should_trace and run_id:
                    os.makedirs(self.logdir, exist_ok=True)
                    log_filepath = os.path.join(
                        self.logdir, f"agentic_run_{run_id}.log"
                    )
                    try:
                        log_file = open(log_filepath, "w", encoding="utf-8")
                        logger.info("Capturing agentic stream log to %s", log_filepath)
                    except Exception:
                        logger.exception(
                            "Failed to open agentic stream log file %s", log_filepath
                        )

                try:
                    logger.info("Streaming query from %s...", stream_url)
                    with httpx.stream(
                        "POST",
                        stream_url,
                        json=stream_payload,
                        headers=headers,
                        verify=not self.insecure,
                        timeout=self.timeout,
                    ) as response:
                        if response.status_code != 200:
                            try:
                                err_body = response.read().decode("utf-8")
                                logger.error(
                                    "Gateway stream start returned HTTP %s: %s",
                                    response.status_code,
                                    err_body,
                                )
                            except Exception:
                                logger.error(
                                    "Gateway stream start returned HTTP %s (failed to read body)",
                                    response.status_code,
                                )
                        response.raise_for_status()
                        current_event = None
                        for line in response.iter_lines():
                            if line:
                                if log_file:
                                    if line.startswith("event: "):
                                        log_file.write(f"\n[{line}]\n")
                                    elif line.startswith("data: "):
                                        data_str = line[6:].strip()
                                        try:
                                            data_json = json.loads(data_str)
                                            log_file.write(
                                                json.dumps(data_json, indent=2) + "\n"
                                            )
                                        except Exception:
                                            log_file.write(line + "\n")
                                    else:
                                        log_file.write(line + "\n")
                                    log_file.flush()

                                if line.startswith("event: "):
                                    current_event = line[7:].strip()
                                    if current_event in ("tool_start", "tool_end"):
                                        self.last_answer = ""
                                elif line.startswith("data: "):
                                    data_str = line[6:].strip()
                                    try:
                                        data_json = json.loads(data_str)
                                    except Exception:
                                        continue
                                    if current_event == "content":
                                        self.last_answer += data_json.get("text", "")
                                    elif current_event == "tool_end":
                                        tool_result = data_json.get("result", "")
                                        if tool_result:
                                            raw_contexts.extend(
                                                _parse_rag_context_artifact(tool_result)
                                            )
                    return raw_contexts

                finally:
                    if log_file:
                        log_file.close()

            except Exception as e:
                last_exception = e
                logger.warning(
                    "Error on attempt %d/%d: %s",
                    attempt,
                    max_attempts,
                    e,
                )
                if attempt < max_attempts:
                    sleep_time = backoff_base**attempt
                    logger.info("Waiting %.1f seconds before retrying...", sleep_time)
                    time.sleep(sleep_time)

        if last_exception:
            logger.error("All %d attempts failed for gateway query.", max_attempts)
            if self.fail_on_error:
                raise last_exception
        return []

    def get_top_k(
        self,
        query: str,
        k: int = 10,
        run_id: str | None = None,
        trace_log: bool | None = None,
        datasource_id: str | None = None,
    ) -> list[tuple]:
        """Query caipe-supervisor or gateway and extract contexts.

        Populates self.documents, self.documents_metadata, and self.last_answer.
        """
        self.documents = []
        self.documents_metadata = []
        self.last_answer = ""
        self.last_raw_response = None

        enriched_query = query
        effective_datasource_id = (
            datasource_id or self.datasource_id or os.environ.get("CAIPE_DATASOURCE_ID")
        )
        if effective_datasource_id:
            enriched_query = (
                f"Instructions: You are answering a question that belongs to the '{effective_datasource_id}' datasource. "
                f'When calling the `knowledge-base_search` tool, you MUST pass `filters={{"datasource_id": "{effective_datasource_id}"}}` '
                f"to restrict your search to this knowledge base, and set the `limit` parameter to up to {k}. "
                f"Keep the `query` argument of the search tool clean and do not include these instructions in it. "
                f"Importantly, only fetch and read (using the `knowledge-base_fetch_document` tool) the specific documents "
                f"you actually need to confidently answer the question, up to a maximum of {k} documents.\n\n"
                f"Question: {query}"
            )
        else:
            enriched_query = (
                f"Instructions: Search across all available knowledge bases to answer the question. "
                f"When calling the `knowledge-base_search` tool, set the `limit` parameter to up to {k} without any datasource filter. "
                f"Keep the `query` argument of the search tool clean and do not include these instructions in it. "
                f"Importantly, only fetch and read (using the `knowledge-base_fetch_document` tool) the specific documents "
                f"you actually need to confidently answer the question, up to a maximum of {k} documents.\n\n"
                f"Question: {query}"
            )

        if not self.use_a2a:
            raw_contexts = self._query_gateway(
                enriched_query, k=k, run_id=run_id, trace_log=trace_log
            )
            self.last_raw_response = {"result": {"artifacts": []}}
        else:
            body = self._call_supervisor(enriched_query)
            if not body:
                return []

            self.last_raw_response = body
            artifacts = body.get("result", {}).get("artifacts", [])

            raw_contexts = []
            for art in artifacts:
                name = art.get("name", "")
                text = _extract_text_from_parts(art.get("parts", []))
                if name == "rag_context":
                    raw_contexts.extend(_parse_rag_context_artifact(text))
                elif name == "final_result":
                    self.last_answer = text

        raw_contexts = _dedupe_and_merge_contexts(raw_contexts)

        for content, doc_id in raw_contexts:
            self.documents.append(content)
            self.documents_metadata.append({"doc_id": doc_id} if doc_id else {})

        return [(i, 1.0) for i in range(len(self.documents))]

    def retrieve(
        self,
        question: str,
        k: int = 5,
        datasource_id: str | None = None,
    ) -> AgenticRAGResult:
        """Backward compatible method for DeepEval evaluations."""
        start = time.perf_counter()
        run_id = str(uuid.uuid4())
        try:
            self.get_top_k(question, k=k, run_id=run_id, datasource_id=datasource_id)
            latency_ms = (time.perf_counter() - start) * 1000

            # Extract token usage from the last response if available
            input_tokens = 0
            output_tokens = 0
            total_tokens = 0

            raw_resp = self.last_raw_response
            if isinstance(raw_resp, dict):
                result_obj = raw_resp.get("result") or {}
                result_meta = (
                    result_obj.get("metadata") if isinstance(result_obj, dict) else None
                )
                resp_meta = raw_resp.get("metadata")

                usage_meta = None
                if isinstance(result_meta, dict):
                    usage_meta = result_meta.get("usage_metadata")
                if not usage_meta and isinstance(resp_meta, dict):
                    usage_meta = resp_meta.get("usage_metadata")

                if not usage_meta and isinstance(result_obj, dict):
                    for art in result_obj.get("artifacts", []):
                        if isinstance(art, dict) and isinstance(
                            art.get("metadata"), dict
                        ):
                            usage_meta = art["metadata"].get("usage_metadata")
                            if usage_meta:
                                break
                if isinstance(usage_meta, dict):
                    input_tokens = usage_meta.get("input_tokens", 0)
                    output_tokens = usage_meta.get("output_tokens", 0)
                    total_tokens = usage_meta.get("total_tokens", 0)

            return AgenticRAGResult(
                answer=self.last_answer,
                contexts=list(self.documents),
                latency_ms=latency_ms,
                task_id=run_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.error(f"Agentic retrieval failed: {e}")
            if getattr(self, "fail_on_error", False):
                raise e
            return AgenticRAGResult(
                answer="",
                contexts=[],
                latency_ms=latency_ms,
                task_id=run_id,
                error=str(e),
            )


class AgenticRAG(BaseRAG):
    """RAG pipeline that uses caipe-supervisor for both retrieval and generation."""

    def __init__(
        self,
        agent_api_url: str | None = None,
        timeout: float = 120.0,
        logdir: str = "logs",
        insecure: bool = False,
        use_a2a: bool | None = None,
        trace_log: bool = False,
        agent_id: str | None = None,
    ) -> None:
        super().__init__(
            llm_client=None,
            model_name="agentic",
            retriever=AgenticRetriever(
                agent_api_url=agent_api_url,
                timeout=timeout,
                insecure=insecure,
                use_a2a=use_a2a,
                trace_log=trace_log,
                logdir=logdir,
                agent_id=agent_id,
            ),
            logdir=logdir,
        )

    @property
    def _agentic_retriever(self) -> AgenticRetriever:
        return self.retriever  # type: ignore

    def query(
        self,
        question: str,
        top_k: int = 3,
        run_id: str | None = None,
        trace_log: bool | None = None,
    ) -> dict[str, Any]:
        """Single call returns both contexts and answer."""
        if run_id is None:
            _q_hash = int(hashlib.md5(question.encode()).hexdigest(), 16) % 10000
            run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_q_hash:04d}"

        self.traces = []
        self.traces.append(
            TraceEvent(
                event_type="query_start",
                component="agentic_rag",
                data={
                    "run_id": run_id,
                    "question": question,
                    "agent_api_url": self._agentic_retriever.agent_api_url,
                },
            )
        )

        try:
            top_docs = self._agentic_retriever.get_top_k(
                question, k=top_k, run_id=run_id, trace_log=trace_log
            )

            retrieved_docs = [
                {
                    "content": self._agentic_retriever.documents[idx],
                    "similarity_score": score,
                    "document_id": (
                        self._agentic_retriever.documents_metadata[idx].get("doc_id")
                        if idx < len(self._agentic_retriever.documents_metadata)
                        and self._agentic_retriever.documents_metadata[idx].get(
                            "doc_id"
                        )
                        else idx
                    ),
                    "metadata": (
                        self._agentic_retriever.documents_metadata[idx]
                        if idx < len(self._agentic_retriever.documents_metadata)
                        else {}
                    ),
                }
                for idx, score in top_docs
                if idx < len(self._agentic_retriever.documents)
            ]

            retrieved_doc_ids = [doc["document_id"] for doc in retrieved_docs]
            answer = self._agentic_retriever.last_answer

            usage = None
            raw_resp = self._agentic_retriever.last_raw_response
            if isinstance(raw_resp, dict):
                result_obj = raw_resp.get("result") or {}
                result_meta = (
                    result_obj.get("metadata") if isinstance(result_obj, dict) else None
                )
                resp_meta = raw_resp.get("metadata")

                usage_meta = None
                if isinstance(result_meta, dict):
                    usage_meta = result_meta.get("usage_metadata")
                if not usage_meta and isinstance(resp_meta, dict):
                    usage_meta = resp_meta.get("usage_metadata")

                if not usage_meta and isinstance(result_obj, dict):
                    for art in result_obj.get("artifacts", []):
                        if isinstance(art, dict) and isinstance(
                            art.get("metadata"), dict
                        ):
                            usage_meta = art["metadata"].get("usage_metadata")
                            if usage_meta:
                                break
                if isinstance(usage_meta, dict):
                    usage = {
                        "prompt_tokens": usage_meta.get("input_tokens", 0),
                        "completion_tokens": usage_meta.get("output_tokens", 0),
                        "total_tokens": usage_meta.get("total_tokens", 0),
                    }

            if not retrieved_docs and answer:
                logger.warning(
                    "AgenticRAG [%s]: no rag_context artifacts in response.",
                    run_id,
                )
            elif not retrieved_docs and not answer:
                logger.warning(
                    "AgenticRAG [%s]: no contexts and no answer.",
                    run_id,
                )

            self.traces.append(
                TraceEvent(
                    event_type="query_complete",
                    component="agentic_rag",
                    data={
                        "run_id": run_id,
                        "success": True,
                        "num_retrieved": len(retrieved_docs),
                        "answer_length": len(answer),
                    },
                )
            )

            result = {
                "answer": answer,
                "run_id": run_id,
                "retrieved_docs": retrieved_docs,
                "usage": usage,
            }
            logs_path = self.export_traces_to_log(
                run_id,
                question,
                result,
            )

            agentic_log_path = None
            should_trace = trace_log
            if should_trace is None:
                should_trace = self._agentic_retriever.trace_log
            if not should_trace:
                env_val = os.getenv("CAIPE_TRACE_LOG")
                if env_val is not None:
                    should_trace = env_val.lower() in ("true", "1", "yes")

            if should_trace:
                agentic_log_path = os.path.join(
                    self.logdir, f"agentic_run_{run_id}.log"
                )

            return {
                "answer": answer,
                "run_id": run_id,
                "retrieved_docs": retrieved_docs,
                "retrieved_doc_ids": retrieved_doc_ids,
                "usage": usage,
                "logs": logs_path,
                "agentic_log": agentic_log_path,
            }

        except Exception as e:
            logger.exception(
                "AgenticRAG [%s]: unhandled exception during query", run_id
            )
            self.traces.append(
                TraceEvent(
                    event_type="error",
                    component="agentic_rag",
                    data={"run_id": run_id, "error": str(e)},
                )
            )
            logs_path = self.export_traces_to_log(run_id, question, None)
            return {
                "answer": f"Error processing query: {str(e)}",
                "run_id": run_id,
                "retrieved_docs": [],
                "retrieved_doc_ids": [],
                "usage": None,
                "logs": logs_path,
            }


def default_agentic_rag_client(
    logdir: str = "logs",
    agent_api_url: str | None = None,
    timeout: float = 120.0,
    insecure: bool = False,
    use_a2a: bool | None = None,
    trace_log: bool = False,
) -> AgenticRAG:
    """Create an AgenticRAG client that routes queries through the agent API."""
    return AgenticRAG(
        agent_api_url=agent_api_url,
        timeout=timeout,
        logdir=logdir,
        insecure=insecure,
        use_a2a=use_a2a,
        trace_log=trace_log,
    )
