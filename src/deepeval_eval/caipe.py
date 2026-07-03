from __future__ import annotations

import time
from typing import Any

import requests


def check_response(resp: requests.Response) -> requests.Response:
    if not resp.ok:
        raise RuntimeError(f'{resp.request.method} {resp.request.url} -> HTTP {resp.status_code}\n{resp.text}')
    return resp


# Thin wrapper around CAIPE rag-server REST endpoints. The auth token is
# optional because local Docker Compose runs can use RBAC bypass.
class CaipeRagClient:
    def __init__(self, base_url: str, token: str | None = None) -> None:
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
        if token:
            self.session.headers['Authorization'] = f'Bearer {token}'

    def register_ingestor(self, ingestor_type: str, ingestor_name: str, description: str) -> tuple[str, int]:
        resp = check_response(self.session.post(
            f'{self.base_url}/v1/ingestor/heartbeat',
            json={'ingestor_type': ingestor_type, 'ingestor_name': ingestor_name, 'description': description},
            timeout=60,
        ))
        data = resp.json()
        return data['ingestor_id'], int(data['max_documents_per_ingest'])

    def reset_datasource(self, datasource_id: str) -> None:
        resp = self.session.delete(f'{self.base_url}/v1/datasource', params={'datasource_id': datasource_id}, timeout=60)
        if resp.status_code not in (200, 204, 404):
            check_response(resp)

    def upsert_datasource(self, datasource_id: str, name: str, ingestor_id: str, description: str, source_type: str) -> None:
        payload = {
            'datasource_id': datasource_id,
            'name': name,
            'ingestor_id': ingestor_id,
            'description': description,
            'source_type': source_type,
            'last_updated': int(time.time()),
        }
        check_response(self.session.post(f'{self.base_url}/v1/datasource', json=payload, timeout=60))

    def open_job(self, datasource_id: str, total: int, message: str) -> str:
        resp = check_response(self.session.post(
            f'{self.base_url}/v1/job',
            params={'datasource_id': datasource_id, 'job_status': 'in_progress', 'message': message, 'total': total},
            timeout=60,
        ))
        return resp.json()['job_id']

    def close_job(self, job_id: str, message: str) -> None:
        check_response(self.session.patch(
            f'{self.base_url}/v1/job/{job_id}',
            params={'job_status': 'completed', 'message': message},
            timeout=60,
        ))

    def ingest_batch(self, documents: list[dict[str, Any]], ingestor_id: str, datasource_id: str, job_id: str) -> None:
        check_response(self.session.post(
            f'{self.base_url}/v1/ingest',
            json={'documents': documents, 'ingestor_id': ingestor_id, 'datasource_id': datasource_id, 'job_id': job_id},
            timeout=300,
        ))
        for endpoint in ('increment-document-count', 'increment-progress'):
            resp = self.session.post(f'{self.base_url}/v1/job/{job_id}/{endpoint}', params={'increment': len(documents)}, timeout=60)
            if resp.status_code >= 500:
                check_response(resp)

    def query(self, question: str, datasource_id: str | None, limit: int) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {'query': question, 'limit': limit}
        if datasource_id:
            payload['filters'] = {'datasource_id': datasource_id}
        resp = check_response(self.session.post(f'{self.base_url}/v1/query', json=payload, timeout=120))
        data = resp.json()
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return list(data.get('results') or [])
        return []


def extract_contexts_and_sources(results: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    contexts: list[str] = []
    sources: list[dict[str, Any]] = []
    for row in results:
        document = row.get('document') if isinstance(row, dict) else None
        if not isinstance(document, dict):
            document = {}
        metadata = document.get('metadata') if isinstance(document.get('metadata'), dict) else {}
        nested = metadata.get('metadata') if isinstance(metadata.get('metadata'), dict) else {}
        text = document.get('page_content') or row.get('page_content') or document.get('content') or row.get('content') or ''
        if not text:
            continue
        contexts.append(text)
        sources.append({
            'document_id': metadata.get('document_id'),
            'title': metadata.get('title'),
            'source_type': nested.get('source_type'),
            'score': row.get('score'),
        })
    return contexts, sources
