from __future__ import annotations

import requests
import time
from typing import Any


def check_response(resp: requests.Response) -> requests.Response:
    if not resp.ok:
        raise RuntimeError(
            f"{resp.request.method} {resp.request.url} -> HTTP {resp.status_code}\n{resp.text}"
        )
    return resp


# Thin wrapper around CAIPE rag-server REST endpoints.
class CaipeRagClient:
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
            print(
                f"[INFO] Keycloak Token refreshed successfully. Valid for {expires_in}s."
            )
        except Exception as e:
            print(f"[ERROR] Failed to fetch token via client_credentials: {e}")

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

    def query(
        self, question: str, datasource_id: str | None, limit: int
    ) -> list[dict[str, Any]]:
        self.ensure_authenticated()

        payload: dict[str, Any] = {"query": question, "limit": limit}
        if datasource_id:
            payload["filters"] = {"datasource_id": datasource_id}
        resp = check_response(
            self.session.post(f"{self.base_url}/v1/query", json=payload, timeout=120)
        )
        data = resp.json()
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return list(data.get("results") or [])
        return []


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
