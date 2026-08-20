# LLM Cluster Router 🚀

An ultra-lightweight, high-performance, and configuration-free API Gateway designed for self-hosted LLM clusters running vLLM, llama.cpp, Ollama, or similar servers.

It aggregates models from multiple nodes in real-time, dynamically routes chat completions requests to the appropriate nodes currently serving those models, and supports automatic Ethernet-to-WiFi link failovers.

---

## 🏗️ Architecture

```text
                  [ Laptop / Client ]
                          │
                          ▼ (https://llm.example.com)
                 [ Reverse Proxy (Traefik) ]
                          │
                          ▼
             [  LLM CLUSTER ROUTER  ]
             (Auto-discovery & Failover)
               /          │          \
              /           │           \
             ▼            ▼            ▼
        [ Node 1 ]   [ Node 2 ]   [ Node 4 ]
        (vLLM/DeepSeek)  (Clustered)   (vLLM/Llama)
```

---

## ✨ Features

- **Dynamic Model Aggregation (`GET /v1/models`)**: Periodically queries all online nodes in parallel, aggregates currently loaded models, deduplicates the list, and serves it transparently.
- **Context-Aware Routing**: Reads the requested model from incoming API payloads and routes the request to whichever node currently has that model loaded in memory.
- **Ethernet-to-WiFi Failover**: Configured with primary Ethernet IPs and backup WiFi IPs. It automatically tries the high-speed Ethernet route first and seamlessly fails over to WiFi if the connection fails or times out.
- **Capabilities-Based Routing**: Automatically checks request requirements (e.g., vision payloads, tool definitions, structured JSON output formats) against the model capabilities catalog. Mismatched requests (such as sending an image to a text-only DeepSeek model or tool-calls to a reasoning-only DeepSeek-R1 model) are transparently rewritten and routed to a suitable active model.
- **Zero-Config Operations**: Start, stop, or swap models on your GPU nodes at any time. The router auto-discovers changes without needing a config reload or container restart.
- **Streaming (SSE) Compatibility**: Full support for Server-Sent Events (SSE) streaming (`stream: true`) with minimal latency overhead (<2ms).
- **Load Balancing**: Distributes concurrent requests across nodes running the same model using a round-robin strategy.

---

## 🚀 Getting Started

### 1. Configure the Router
Create a `config.yaml` specifying your cluster nodes, timeouts, capabilities catalog, and routing fallbacks:

```yaml
server:
  host: "0.0.0.0"
  port: 8000

timeouts:
  primary: 1.0  # Timeout for Ethernet links (fail over fast)
  fallback: 3.0 # Timeout for WiFi links
  request: 120.0

# Capabilities routing configuration
capabilities_routing:
  enabled: true

# Capabilities catalog (searched top to bottom for pattern matches)
model_capabilities:
  - name_pattern: "thinking"
    vision: false
    tool_calling: false
    structured_output: false

  - name_pattern: "r1"
    vision: false
    tool_calling: false
    structured_output: false

  - name_pattern: "vl"
    vision: true
    tool_calling: true
    structured_output: true

  - name_pattern: "step"
    vision: true
    tool_calling: true
    structured_output: true

  - name_pattern: "qwen"
    vision: false
    tool_calling: true
    structured_output: true

  - name_pattern: "deepseek"
    vision: false
    tool_calling: true
    structured_output: true

nodes:
  - name: "node-1"
    primary: "http://192.168.86.221:8000/v1"
    backup: "http://192.168.86.211:8000/v1"
  - name: "node-4"
    primary: "http://192.168.86.224:8000/v1"
    backup: "http://192.168.86.214:8000/v1"
```

### 2. Standalone Deployment (Docker Compose)
Use the following `docker-compose.yml` to spin up the gateway:

```yaml
version: '3.8'

services:
  llm-router:
    build: .
    container_name: llm-cluster-router
    restart: unless-stopped
    ports:
      - "8000:8000"
    volumes:
      - ./config.yaml:/app/config.yaml:ro
    environment:
      - CONFIG_PATH=config.yaml
      - PYTHONUNBUFFERED=1
```

Run:
```bash
docker compose up -d --build
```

---

## 🛠️ API Support

The router functions as a drop-in replacement for standard OpenAI client configurations (e.g., in IntelliJ IDEA, Continue.dev, Cursor, or Open WebUI):

- `GET /v1/models` - Returns combined active models.
- `POST /v1/chat/completions` - Routes and generates chat tokens (supports streaming).
- `POST /v1/completions` - Routes standard completions.
- `POST /v1/embeddings` - Routes embedding requests.

---

## 🔒 Security
For production use, it is recommended to place the gateway behind a reverse proxy (like Traefik, Nginx, or HAProxy) to handle SSL termination and basic authentication.

---

## 📊 Observability & Auditing

The gateway outputs structured JSON logs for auditing completions. These logs can be piped directly over TCP to Logstash and indexed into Elasticsearch.

### Log Payload Schema
Every request writes a single-line JSON log (`AUDIT_LOG: {...}`) containing:
- `timestamp`: Generation UTC timestamp.
- `client_ip`: Originating client IP address (extracts `X-Forwarded-For` if behind proxy).
- `auth_user`: Logged-in Basic Auth username (parsed from request headers).
- `model`: The model that actually ran on the GPU node.
- `original_model`: The model requested by the client (before any rewrites).
- `rewritten`: Boolean indicating if the model parameter was auto-corrected.
- `trace_id`: The correlation ID parsed from `traceparent` (W3C standard) or `X-Request-ID` headers.
- `completion_id`: The OpenAI completion identifier (`chatcmpl-...`) generated by the engine.
- `latency_ms`: Total execution time in milliseconds.
- `node`: The cluster node that processed the request (e.g. `node-1`).
- `prompt`: The user's input text.
- `response`: The generated assistant completion text.
- `prompt_tokens` / `completion_tokens` / `total_tokens`: Token metrics.

---

## 🔍 Kibana Query Reference (KQL)

Use the following queries in Kibana **Discover** (under the `llm-audit-logs-*` data view) to inspect your cluster activity:

### 1. Find Misconfigured Clients (Model Rewrites)
Find cases where a client requested an invalid or offline model (e.g. `gpt-4o`) and the proxy auto-corrected it to a loaded fallback model:
```text
audit.rewritten : true
```

### 2. Trace App Context / Transaction Correlation
Track a specific transaction from your client application (which passes standard `traceparent` or `X-Request-ID` headers) to see the exact prompt and response generated:
```text
audit.trace_id : "4bf92f3577b34da6a3ce929d0e0e4736"
```

### 3. Track a Specific Completion ID
Fetch audit data for a single completions result:
```text
audit.completion_id : "chatcmpl-8bc0475f35bc908c"
```

### 4. Audit Prompt & Response Content
Search for prompts containing specific keywords (e.g., "error"):
```text
audit.prompt : *error*
```
Search for responses containing specific code snippets or topics (e.g., "Docker"):
```text
audit.response : *Docker*
```

### 5. Check Latency Spikes
Find requests that experienced latency higher than 10 seconds:
```text
audit.latency_ms > 10000
```

### 6. Filter by Client User
Filter logs to see requests sent by a specific user (authenticated via Basic Auth):
```text
audit.auth_user : "demouser"
```

### 7. Filter by Target Node
Inspect all completions handled by a specific cluster GPU node (e.g., node-4):
```text
audit.node : "node-4"
```
# PR-build trigger 1787183337
# CI trigger Wed Aug 19 08:10:48 PM EDT 2026
# Trigger latest tag CI
