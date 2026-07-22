# Agentic RAG Client

This page documents the agentic RAG client implemented in `src/deepeval_eval/agentic_rag.py`. It provides a full agentic retriever that queries the CAIPE supervisor or streaming BFF gateway to capture rag_context artifacts emitted during agent tool calls.

## Overview

The agentic RAG client is designed for evaluating CAIPE's agentic chat capabilities — specifically, how an AI agent retrieves and uses documents to answer questions through tool calls, rather than through a direct REST API.

It supports **two communication protocols**:

| Protocol | Endpoint | Description |
| --- | --- | --- |
| **A2A (Agent-to-Agent)** | `POST /message/send` | JSON-RPC 2.0 protocol against the CAIPE supervisor. Returns artifacts synchronously. |
| **SSE Streaming Gateway** | `POST /api/chat/conversations` + `POST /api/v1/chat/stream/start` | Two-step BFF gateway flow with server-sent events for streaming tool outputs. |

The protocol is selected via the `--agentic` flag (enables SSE) or `--agentic --use-a2a` flag (enables A2A). When neither flag is set, the default `CaipeRagClient` standard path is used.

## Architecture

~~~mermaid
flowchart TD
    A[Question] --> B{Protocol?}
    B -->|A2A| C[POST /message/send]
    B -->|SSE Gateway| D[Create conversation]
    D --> E[Stream /api/v1/chat/stream/start]
    C --> F[Parse artifacts from response]
    E --> G[Parse SSE events for rag_context]
    F --> H[Deduplicate & merge contexts]
    G --> H
    H --> I[Return contexts + answer]
~~~

## Core Classes

### `AgenticRetriever`

The primary retriever that queries the CAIPE supervisor or BFF gateway.

| Attribute | Default | Description |
| --- | --- | --- |
| `agent_api_url` | `http://localhost:8000` | CAIPE supervisor URL. Read from `CAIPE_SUPERVISOR_URL` env var or `--supervisor-url`. |
| `timeout` | `120.0` | Request timeout in seconds. |
| `insecure` | `False` | Skip SSL verification. Also controlled by `INSECURE_SSL` env var. |
| `use_a2a` | `False` | Use A2A protocol. Also controlled by `CAIPE_USE_A2A` env var. |
| `trace_log` | `False` | Enable trace logging. Also controlled by `CAIPE_TRACE_LOG` env var. |
| `fail_on_error` | `False` | Raise exception on retrieval failure. |

#### Key Methods

| Method | Purpose |
| --- | --- |
| `get_top_k(query, k, run_id, trace_log)` | Main retrieval entry point. Queries the agent and extracts contexts. Populates `self.documents` and `self.documents_metadata`. |
| `retrieve(question, k)` | Backward-compatible method returning `AgenticRAGResult`. Used by DeepEval evaluation loops. |
| `fit(documents)` | No-op stub for compatibility. AgenticRetriever does not support local fitting. |
| `_call_supervisor(question)` | Sends A2A JSON-RPC request to the CAIPE supervisor. |
| `_query_gateway(question, k, run_id, trace_log)` | Sends SSE streaming query through the BFF gateway. |
| `_get_oidc_token()` | Fetches OIDC token dynamically via client credentials grant. Falls back to environment variables or Kubernetes secrets. |

### `AgenticRAG`

A higher-level RAG pipeline class that wraps `AgenticRetriever` and provides a unified query interface.

| Attribute | Default | Description |
| --- | --- | --- |
| `llm_client` | `None` | No LLM client needed — the agent generates answers itself. |
| `model_name` | `"agentic"` | Model name identifier. |
| `retriever` | `AgenticRetriever` instance | The underlying retriever. |
| `logdir` | `"logs"` | Directory for trace logs. |

#### Key Methods

| Method | Purpose |
| --- | --- |
| `query(question, top_k, run_id, trace_log)` | Single-call query returning contexts, answer, and usage metadata. |
| `export_traces_to_log(run_id, question, result)` | Exports trace events to a JSON log file. |

### `AgenticRAGResult`

A `dataclass` representing the result of an agentic retrieval:

| Field | Type | Description |
| --- | --- | --- |
| `answer` | `str` | The agent-generated answer. |
| `contexts` | `list[str]` | Retrieved document contexts. |
| `latency_ms` | `float` | Total latency in milliseconds. |
| `task_id` | `str` | UUID for the retrieval task. |
| `input_tokens` | `int` | Prompt tokens used by the agent. |
| `output_tokens` | `int` | Completion tokens used by the agent. |
| `total_tokens` | `int` | Total tokens. |
| `error` | `Optional[str]` | Error message if retrieval failed. |

## Protocol Details

### A2A Protocol (`/message/send`)

The A2A protocol uses JSON-RPC 2.0 to send a question to the CAIPE supervisor:

```json
{
  "jsonrpc": "2.0",
  "id": "uuid",
  "method": "message/send",
  "params": {
    "message": {
      "role": "user",
      "parts": [{"kind": "text", "text": "What is CAIPE?"}],
      "messageId": "uuid"
    }
  }
}
```

The supervisor response contains artifacts emitted by agent tool calls:

```json
{
  "result": {
    "artifacts": [
      {
        "name": "rag_context",
        "parts": [{"kind": "text", "text": "..."}]
      },
      {
        "name": "final_result",
        "parts": [{"kind": "text", "text": "CAIPE is..."}]
      }
    ],
    "final": true,
    "kind": "status-update"
  }
}
```

| Artifact Name | Meaning |
| --- | --- |
| `rag_context` | RAG context emitted by the agent's `knowledge-base_search` tool. Parsed into (content, doc_id) tuples. |
| `final_result` | The agent's final answer text. |

### SSE Streaming Gateway Protocol

The gateway protocol uses a two-step flow:

**Step 1: Create Conversation Session**

```
POST /api/chat/conversations
{
  "title": "Agentic Session",
  "client_type": "webui",
  "agent_id": "hello-world"
}
```

Returns a `conversation_id`.

**Step 2: Stream Chat**

```
POST /api/v1/chat/stream/start
{
  "message": "What is CAIPE?",
  "conversation_id": "uuid",
  "agent_id": "hello-world",
  "protocol": "custom",
  "client_context": {
    "source": "eval",
    "tool_result_display_limit": -1
  }
}
```

The response is an SSE stream with events:

| Event | Meaning |
| --- | --- |
| `event: content` | Final answer text pieces (`data.text`). |
| `event: tool_end` | Tool outputs / RAG contexts (`data.result`). Parsed for `rag_context` artifacts. |
| `event: tool_start` | Tool invocation start (no output captured). |

SSE events are parsed and logged (if `trace_log=True`):

```
event: tool_end
data: {"result": {"rag_context": {...}}}
```

## Context Parsing

The agentic client parses rag_context artifacts from both protocols into `(content, doc_id)` tuples:

### From `knowledge-base_search` Tool

```json
{
  "semantic_results": [
    {
      "text_content": "CAIPE is a RAG system...",
      "metadata": {
        "document_id": "doc_123"
      }
    }
  ],
  "keyword_results": [...]
}
```

### From `knowledge-base_fetch_document` Tool

```json
[
  {
    "document": {
      "page_content": "CAIPE is a RAG system...",
      "document_id": "doc_123",
      "metadata": {...}
    }
  }
]
```

### Markdown Cleaning

Search tool snippets often contain UI-formatted markup like `**Snippet:** ...**CAIPE** uses nomic-embed-text...`. The `clean_snippet_markdown()` function strips this:

1. Removes `**Snippet:**` prefix.
2. Replaces ellipsis (`...`) with spaces.
3. Strips bold markup (`**...**`).
4. Normalizes whitespace.

### Deduplication

Two deduplication strategies are applied:

| Function | Purpose |
| --- | --- |
| `_dedupe_preserve_order(items)` | Deduplicates by content, preserving first-seen order. |
| `_dedupe_and_merge_contexts(items)` | Deduplicates and merges by `doc_id`, preferring longer/full content. |

## OIDC Authentication

The agentic client fetches OIDC tokens dynamically:

1. **Environment Variables**: Checks `CAIPE_CLIENT_ID`, `CAIPE_CLIENT_SECRET`.
2. **Kubernetes Secret**: If not in environment, attempts `kubectl get secret caipe-ui-secret -n caipe`.
3. **Keycloak**: Sends client credentials grant to Keycloak.

Default Keycloak URL:
- Production: `https://keycloak.caipe.homelab/realms/caipe/protocol/openid-connect/token`
- Local: `http://localhost:7080/realms/caipe/protocol/openid-connect/token`

The fetched token is cached in `CAIPE_OIDC_TOKEN` environment variable.

## Datasource Enrichment

When `CAIPE_DATASOURCE_ID` is set, the query is enriched with instructions for the agent:

```
Instructions: You are answering a question that belongs to the 'enterprise_rag_bench' datasource.
When calling the `knowledge-base_search` tool, you MUST pass `filters={"datasource_id": "enterprise_rag_bench"}`
to restrict your search to this knowledge base, and set the `limit` parameter to up to 5.
Keep the `query` argument of the search tool clean and do not include these instructions in it.
Importantly, only fetch and read (using the `knowledge-base_fetch_document` tool) the specific documents
you actually need to confidently answer the question, up to a maximum of 5 documents.

Question: What is CAIPE?
```

## Trace Logging

When `trace_log=True` or `CAIPE_TRACE_LOG=true`:

- A2A: Exports trace events to `logs/query_trace_{run_id}.json`.
- SSE: Streams raw events to `logs/agentic_run_{run_id}.log`.

Trace events include:
| Event Type | Component | Data |
| --- | --- | --- |
| `query_start` | `agentic_rag` | run_id, question, agent_api_url |
| `query_complete` | `agentic_rag` | run_id, success, num_retrieved, answer_length |
| `error` | `agentic_rag` | run_id, error message |

## Usage in Evaluation

The agentic client is used via the unified evaluator:

```bash
python src/deepeval_eval/deepeval_evaluator.py eval \
     --benchmark enterprise \
     --agentic \
     --supervisor-url http://localhost:8000 \
     --top-k 5 \
     --max-items 10
```

Or directly:

```bash
python src/deepeval_eval/agentic_rag.py \
     --agent-api-url http://localhost:8000 \
     --timeout 120 \
     --insecure \
     --use-a2a \
     --trace-log
```

## Comparison With Standard CAIPE Client

| Aspect | `CaipeRagClient` | `AgenticRetriever` |
| --- | --- | --- |
| Query method | Direct REST (`POST /v1/query`) | Agent tool calls via A2A or SSE |
| Protocol | HTTP JSON | JSON-RPC 2.0 or SSE streaming |
| Answer source | LLM generated from retrieved context | Agent generates answer itself |
| Context source | CAIPE rag-server response | Agent's `rag_context` artifacts |
| Authentication | Bearer token (static or Keycloak) | Dynamic OIDC from env or K8s secret |
| Use case | Standard RAG retrieval evaluation | Agentic chat evaluation |
| Token tracking | Via LLM client | From agent `usage_metadata` in artifacts |
