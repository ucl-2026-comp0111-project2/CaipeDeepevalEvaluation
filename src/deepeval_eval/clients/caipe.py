from __future__ import annotations

import logging
import time
from typing import Any

import requests

from deepeval_eval.clients.rag import (
    BaseRagClient,
    RagQueryResult,
)
from deepeval_eval.core.prompt_style import PromptStyle, build_prompt

logger = logging.getLogger(__name__)


def check_response(resp: requests.Response) -> requests.Response:
    if not resp.ok:
        raise RuntimeError(
            f"{resp.request.method} {resp.request.url} -> HTTP {resp.status_code}\n{resp.text}"
        )
    return resp


# Thin wrapper around CAIPE rag-server REST endpoints.
class CaipeRagClient(BaseRagClient):
    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        verify: bool | str = True,
        keycloak_url: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.verify = verify
        self.session.headers.update({"Content-Type": "application/json"})

        # OIDC Client Credentials config
        self.keycloak_url = keycloak_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_expiry = 0

        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
            # Give static tokens a 5-minute buffer from now before trying an auto-refresh
            self.token_expiry = time.time() + 300
        elif self.keycloak_url and self.client_id and self.client_secret:
            self.refresh_access_token()

    def refresh_access_token(self) -> None:
        """Refreshes OIDC bearer token via Client Credentials grant type."""
        if not self.keycloak_url or not self.client_id:
            return

        payload = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "client_credentials",
        }

        try:
            # Send as standard urlencoded form post matching your curl statement
            resp = requests.post(
                self.keycloak_url, data=payload, verify=self.session.verify, timeout=15
            )
            resp.raise_for_status()
            data = resp.json()

            token = data["access_token"]
            expires_in = data.get("expires_in", 300)  # Fallback to 5 mins

            # Set local expiry timestamp with a 30-second clock-skew safety buffer
            self.token_expiry = time.time() + expires_in - 30
            self.session.headers["Authorization"] = f"Bearer {token}"
            logger.info(
                f"Keycloak Token refreshed successfully. Valid for {expires_in}s."
            )
        except Exception as exc:
            logger.exception("Failed to fetch token via client_credentials.")
            raise RuntimeError(
                f"Failed to refresh OIDC token via Keycloak: {exc}"
            ) from exc

    def ensure_authenticated(self) -> None:
        """Checks validation window and triggers dynamic refresh if expired."""
        if self.keycloak_url and time.time() >= self.token_expiry:
            self.refresh_access_token()

    def register_ingestor(
        self, ingestor_type: str, ingestor_name: str, description: str
    ) -> tuple[str, int]:
        resp = check_response(
            self.session.post(
                f"{self.base_url}/v1/ingestor/heartbeat",
                json={
                    "ingestor_type": ingestor_type,
                    "ingestor_name": ingestor_name,
                    "description": description,
                },
                timeout=60,
            )
        )
        data = resp.json()
        return data["ingestor_id"], int(data["max_documents_per_ingest"])

    def reset_datasource(self, datasource_id: str) -> None:
        resp = self.session.delete(
            f"{self.base_url}/v1/datasource",
            params={"datasource_id": datasource_id},
            timeout=60,
        )
        if resp.status_code not in (200, 204, 404):
            check_response(resp)

    def upsert_datasource(
        self,
        datasource_id: str,
        name: str,
        ingestor_id: str,
        description: str,
        source_type: str,
    ) -> None:
        payload = {
            "datasource_id": datasource_id,
            "name": name,
            "ingestor_id": ingestor_id,
            "description": description,
            "source_type": source_type,
            "last_updated": int(time.time()),
        }
        check_response(
            self.session.post(
                f"{self.base_url}/v1/datasource", json=payload, timeout=60
            )
        )

    def open_job(self, datasource_id: str, total: int, message: str) -> str:
        resp = check_response(
            self.session.post(
                f"{self.base_url}/v1/job",
                params={
                    "datasource_id": datasource_id,
                    "job_status": "in_progress",
                    "message": message,
                    "total": total,
                },
                timeout=60,
            )
        )
        return resp.json()["job_id"]

    def close_job(self, job_id: str, message: str) -> None:
        check_response(
            self.session.patch(
                f"{self.base_url}/v1/job/{job_id}",
                params={"job_status": "completed", "message": message},
                timeout=60,
            )
        )

    def ingest_batch(
        self,
        documents: list[dict[str, Any]],
        ingestor_id: str,
        datasource_id: str,
        job_id: str,
    ) -> None:
        check_response(
            self.session.post(
                f"{self.base_url}/v1/ingest",
                json={
                    "documents": documents,
                    "ingestor_id": ingestor_id,
                    "datasource_id": datasource_id,
                    "job_id": job_id,
                },
                timeout=300,
            )
        )
        for endpoint in ("increment-document-count", "increment-progress"):
            resp = self.session.post(
                f"{self.base_url}/v1/job/{job_id}/{endpoint}",
                params={"increment": len(documents)},
                timeout=60,
            )
            if resp.status_code >= 500:
                check_response(resp)

    def query_raw(
        self,
        question: str,
        datasource_id: str | None,
        limit: int,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self.ensure_authenticated()

        payload: dict[str, Any] = {"query": question, "limit": limit}
        filters: dict[str, Any] = {}
        if datasource_id:
            filters["datasource_id"] = datasource_id
        if metadata_filters:
            filters.update(metadata_filters)
        if filters:
            payload["filters"] = filters

        resp = check_response(
            self.session.post(f"{self.base_url}/v1/query", json=payload, timeout=120)
        )
        data = resp.json()
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return list(data.get("results") or [])
        return []

    def query(
        self,
        question: str,
        reference: str = "",
        datasource_id: str | None = None,
        top_k: int = 3,
        answer_mode: str = "generate",
        dataset_name: str = "enterprise",
        prompt_style: str | PromptStyle | None = None,
        llm_client: Any = None,
        max_context_chars: int = 12000,
        metadata_filters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> RagQueryResult:
        start_time = time.time()
        retrieved_raw = self.query_raw(
            question, datasource_id, top_k, metadata_filters=metadata_filters
        )
        contexts, sources = extract_contexts_and_sources(retrieved_raw)
        trimmed_contexts = [c[:max_context_chars] for c in contexts]

        if llm_client is None:
            raise ValueError("llm_client is required for answer generation")

        def _safe_int(val: Any) -> int:
            if isinstance(val, int):
                return val
            if isinstance(val, (float, str)):
                try:
                    return int(val)
                except (ValueError, TypeError):
                    pass
            return 0

        start_in = _safe_int(getattr(llm_client, "input_tokens", 0))
        start_out = _safe_int(getattr(llm_client, "output_tokens", 0))
        start_tot = _safe_int(getattr(llm_client, "total_tokens", 0))

        prompt = build_prompt(prompt_style, question, trimmed_contexts)
        answer = str(llm_client.generate(prompt))

        end_in = _safe_int(getattr(llm_client, "input_tokens", start_in))
        end_out = _safe_int(getattr(llm_client, "output_tokens", start_out))
        end_tot = _safe_int(getattr(llm_client, "total_tokens", start_tot))

        input_tokens = max(0, end_in - start_in)
        output_tokens = max(0, end_out - start_out)
        total_tokens = max(0, end_tot - start_tot)

        latency_sec = time.time() - start_time
        retrieved_ids = [
            str(s.get("document_id"))
            for s in sources
            if s.get("document_id") is not None
        ]

        return RagQueryResult(
            answer=answer,
            contexts=trimmed_contexts,
            sources=sources,
            retrieved_doc_ids=retrieved_ids,
            latency_sec=latency_sec,
            latency_ms=latency_sec * 1000.0,
            log_file=" ",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )


def extract_contexts_and_sources(
    results: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    contexts: list[str] = []
    sources: list[dict[str, Any]] = []
    for row in results:
        document = row.get("document") if isinstance(row, dict) else None
        if not isinstance(document, dict):
            document = {}
        metadata = (
            document.get("metadata")
            if isinstance(document.get("metadata"), dict)
            else {}
        )
        nested = (
            metadata.get("metadata")
            if isinstance(metadata.get("metadata"), dict)
            else {}
        )
        text = (
            document.get("page_content")
            or row.get("page_content")
            or document.get("content")
            or row.get("content")
            or ""
        )
        if not text:
            continue
        contexts.append(text)
        sources.append(
            {
                "document_id": metadata.get("document_id"),
                "title": metadata.get("title"),
                "source_type": nested.get("source_type"),
                "score": row.get("score"),
            }
        )
    return contexts, sources


def build_caipe_client(
    caipe_settings: Any | None = None,
) -> CaipeRagClient:
    """Instantiate CaipeRagClient from CaipeClientSettings or environment variables."""
    from deepeval_eval.core.config import CaipeClientSettings

    settings = (
        caipe_settings
        if isinstance(caipe_settings, CaipeClientSettings)
        else CaipeClientSettings()
    )

    raw_token = settings.auth_token.get_secret_value() if settings.auth_token else None
    raw_secret = (
        settings.client_secret.get_secret_value() if settings.client_secret else None
    )

    return CaipeRagClient(
        base_url=settings.base_url,
        token=raw_token,
        verify=not settings.insecure,
        keycloak_url=settings.keycloak_url,
        client_id=settings.client_id,
        client_secret=raw_secret,
    )
