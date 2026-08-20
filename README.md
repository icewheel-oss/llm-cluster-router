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
- **KV Prefix Cache-Aware Routing**: Hashes system/prompt prefixes (`CRC32`). Automatically routes matching prompts to the node holding the warm KV-cache, reducing Time-To-First-Token (TTFT) by up to 5x.
- **Thermal-Aware Load Balancing**: Periodically polls hardware temperatures (via `node-exporter` on port `9100`). Automatically deprioritizes nodes exceeding thermal ceilings (`max_temp_celsius`: `82.0°C`) to prevent GPU thermal throttling.
- **Tool-Call & Payload Sanitization**: Intercepts historical `tool_calls` arguments in multi-turn conversation payloads to ensure compliance with strict OpenAI specs and prevent Jinja templating crashes on underlying engines like vLLM.
- **Capabilities-Based Routing**: Automatically checks request requirements (e.g., vision payloads, tool definitions, structured JSON output formats) against the model capabilities catalog. Mismatched requests (such as sending an image to a text-only DeepSeek model or tool-calls to a reasoning-only DeepSeek-R1 model) are transparently rewritten and routed to a suitable active model.
- **Optional Client Rate Limiting**: Per-client (Auth user / IP) rate-limiting quotas to protect cluster GPUs from single-agent loops.
- **Zero-Config Operations**: Start, stop, or swap models on your GPU nodes at any time. The router auto-discovers changes without needing a config reload or container restart.
- **Streaming (SSE) Compatibility**: Full support for Server-Sent Events (SSE) streaming (`stream: true`) with minimal latency overhead (<2ms).
- **Load Balancing**: Distributes concurrent requests across nodes running the same model using a round-robin / least-loaded strategy.

---

## 🔮 Feature Roadmap & Ideas

- [x] **Thermal-Aware Routing**: Deprioritize overheating nodes before thermal throttling strikes.
- [x] **Tool-Call Payload Sanitization**: Self-healing payload sanitization for complex agent frameworks.
- [x] **KV-Cache Aware Routing (Prefix Cache Stickiness)**: Route prompts with matching system prompts/prefixes to the node holding the KV-cache warm in memory.
- [x] **Rate Limiting & Token Quotas**: Per-user / per-key request rate limits and token usage budgeting.
- [ ] **Dynamic Model Auto-Spinup**: Trigger model loading (e.g. via SparkRun / Ollama API) on idle nodes when an offline model is requested.

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

---

## 🔄 Zero-Downtime Blue/Green Deployments

For long-running inference jobs (e.g., reasoning models taking several minutes), restarting containers via `docker compose down` will sever active HTTP streams.

To perform a **100% zero-downtime deployment** where new connections land on the updated container while the old container stays alive until long-running requests finish:

```bash
# 1. Pull the new image version from GHCR
docker pull ghcr.io/icewheel-oss/llm-cluster-router:latest

# 2. Spin up the green container alongside the active container
docker run -d \
  --name llm-cluster-router-green \
  --network shared-network \
  --network traefik-public \
  -v ./config.yaml:/app/config.yaml:ro \
  --env-file .env \
  --label 'traefik.enable=true' \
  --label 'traefik.docker.network=traefik-public' \
  --label 'traefik.http.routers.llm-router-local.rule=Host(`llm.example.com`)' \
  --label 'traefik.http.routers.llm-router-local.entrypoints=websecure' \
  --label 'traefik.http.routers.llm-router-local.tls=true' \
  --label 'traefik.http.routers.llm-router-local.service=llm-router-service' \
  --label 'traefik.http.services.llm-router-service.loadbalancer.server.port=8000' \
  ghcr.io/icewheel-oss/llm-cluster-router:latest

# 3. Give Traefik 3 seconds to register the new instance in its load balancer
sleep 3

# 4. Gracefully stop the old container with an extended timeout (e.g. 900s / 15 mins for long batch runs)
docker stop -t 900 llm-cluster-router && docker rm llm-cluster-router

# 5. Rename green container to primary name
docker rename llm-cluster-router-green llm-cluster-router
```


