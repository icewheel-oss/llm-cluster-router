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
- **Zero-Downtime Dynamic Configuration Hot-Reload**: Automatically monitors `config.yaml` file modification times (`mtime`) on disk and reloads updated node configurations, IP changes, or priorities dynamically without container restarts. Also supports instant manual reloads via `POST /_router/reload`.
- **Uncapped Request Timeout Support (`request: null`)**: Supports completely disabling request timeouts (`timeout=None`) across the Python AsyncClient and gateway pipeline, allowing large local LLM models (e.g. Qwen 27B FP8, DeepSeek-R1) to process ultra-long context prompts or multi-minute reasoning chains without artificial 120s timeout walls.
- **Context-Aware Routing**: Reads the requested model from incoming API payloads and routes the request to whichever node currently has that model loaded in memory.
- **Ethernet-to-WiFi Failover**: Configured with primary Ethernet IPs and backup WiFi IPs. It automatically tries the high-speed Ethernet route first and seamlessly fails over to WiFi if the connection fails or times out.
- **KV Prefix Cache-Aware Routing**: Hashes system/prompt prefixes (`CRC32`). Automatically routes matching prompts to the node holding the warm KV-cache, reducing Time-To-First-Token (TTFT) by up to 5x.
- **Thermal-Priority & Multi-Threshold Routing**: Thermal health evaluation takes precedence over all other strategies (including KV-cache affinity). Uses dual-tier temperature thresholds:
  - **Warning Threshold (`70.0°C`)**: Nodes exceeding 70°C bypass KV-cache match affinity and get deprioritized to give warm GPUs a chance to cool down.
  - **Critical Threshold (`80.0°C`)**: Hard block that stops routing new requests to nodes $\ge 80^\circ	ext{C}$ until hardware temperatures drop back down.
- **Tool-Call & Payload Sanitization**: Intercepts historical `tool_calls` arguments in multi-turn conversation payloads to ensure compliance with strict OpenAI specs and prevent Jinja templating crashes on underlying engines like vLLM.
- **Capabilities-Based Routing**: Automatically checks request requirements (e.g., vision payloads, tool definitions, structured JSON output formats) against the model capabilities catalog. Mismatched requests (such as sending an image to a text-only DeepSeek model or tool-calls to a reasoning-only DeepSeek-R1 model) are transparently rewritten and routed to a suitable active model.
- **Optional Client Rate Limiting**: Per-client (Auth user / IP) rate-limiting quotas to protect cluster GPUs from single-agent loops.
- **Zero-Config Operations**: Start, stop, or swap models on your GPU nodes at any time. The router auto-discovers changes without needing a config reload or container restart.
- **Streaming (SSE) Compatibility**: Full support for Server-Sent Events (SSE) streaming (`stream: true`) with minimal latency overhead (<2ms).
- **Load Balancing**: Distributes concurrent requests across nodes running the same model using a round-robin / least-loaded strategy.

---

## 🔮 Feature Roadmap & Ideas

- [x] **Thermal-Priority Routing Over KV Cache**: Dual-tier thermal thresholds (70°C cooling bypass / 80°C hard block) that override KV-cache affinity to prevent GPU overheating.
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
  request: null  # Set to null to disable request timeouts for long local LLM reasoning

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

## 💡 Best Practices: Prompt Design for Batch / High-Concurrency Workloads

If you run a pool of worker processes that all send the **same fixed system
prompt** against a batch/queue of work (e.g. a document-processing pipeline,
a bulk extraction job, anything that fans out N concurrent workers against
one large backlog), be aware of an interaction between two features above:

**KV Prefix Cache-Aware Routing hashes the full system-message content and
hard-routes any matching hash to whichever single node last served it, for
the configured `ttl_seconds` — bypassing Smart/Thermal/Load-Balanced routing
entirely on a cache hit.** If every worker sends an identical system prompt,
every request hashes identically, so **all of them get pinned to one node**,
no matter how many workers you run or how many nodes are in your cluster.
Observed directly in production: with dozens of concurrent workers running
against an identical system prompt, `docker logs` showed a continuous
stream of `Prefix KV-cache hit ... routing to warm cache node 'X'` lines —
one single node handling 100% of the traffic — while the rest of the
cluster sat idle. Worker count alone did not change this; it only changed
how deep a queue built up on that one node.

**Fix — key your system prompt by whatever natural category your workload
already has** (a tenant/customer ID, a document category, a topic label, a
dataset partition — anything that's already meaningful to the job, not an
arbitrary cache-buster). Append or interpolate that value into the system
prompt so it becomes part of what gets hashed:

```
You are processing items of category: {category}.
...rest of your fixed system prompt...
```

This turns one global cache bucket into one bucket **per category**:
- Different categories now hash differently and spread across the cluster
  naturally, letting Smart/Thermal/Load-Balanced routing actually engage
  for that traffic instead of being permanently short-circuited by a single
  cache entry.
- Same-category requests still share one warm cache — arguably a *better*
  fit for this feature than a random cache-buster, since same-category
  items in most real workloads genuinely repeat vocabulary/entities/jargon,
  so there's real token-level prefix reuse to be had.

Verified immediately after applying this pattern in production: router
logs went from a wall of identical `Prefix KV-cache hit` lines pointing at
one node, to real `Smart routing selected node ... (active: N, temp: X°C)`
entries spread across the full node pool — the previously-idle nodes
started receiving traffic for the first time, with per-node active-request
counts climbing into the high single digits (confirming each node's own
continuous batching absorbs many concurrent requests, not just one) and
zero errors across a scale-up from single digits to 50 concurrent workers.

---

## 🛠️ API Support

The router functions as a drop-in replacement for standard OpenAI client configurations (e.g., in IntelliJ IDEA, Continue.dev, Cursor, or Open WebUI):

- `GET /v1/models` - Returns combined active models.
- `POST /v1/chat/completions` - Routes and generates chat tokens (supports streaming).
- `POST /v1/completions` - Routes standard completions.
- `POST /v1/embeddings` - Routes embedding requests.

### ⚙️ Management & Health Endpoints
- `GET /health` - Health check endpoint returning status.
- `POST /_router/reload` - Instantly triggers a configuration reload from `config.yaml` without downtime.

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

Use the following queries in Kibana **Discover** to inspect cluster activity and debug failures across indices:

### 1. Find Gateway Errors (HTTP 4xx / 5xx)
Search for client or server errors routed by the gateway (`llm-audit-logs-*` index):
```text
status_code >= 400
```
Search specifically for gateway timeout errors (502 / 504):
```text
status_code : 502 OR status_code : 504
```

### 2. Find Misconfigured Clients (Model Rewrites)
Find cases where a client requested an invalid or offline model (e.g. `gpt-4o`) and the proxy auto-corrected it to an active loaded model:
```text
rewritten : true
```

### 3. Trace App Context / Transaction Correlation
Track a specific transaction from your client application (which passes standard `traceparent` or `X-Request-ID` headers) to inspect prompt and completion details:
```text
trace_id : "4bf92f3577b34da6a3ce929d0e0e4736"
```

### 4. GPU Engine Crashes & Out-Of-Memory Errors
Search for vLLM / engine failures (`vllm-logs-*` index):
```text
message : *CUDA* OR message : *Out of memory* OR message : *OOM* OR message : *Traceback*
```

### 5. Check Latency Spikes
Find requests that experienced latency higher than 10 seconds:
```text
latency_ms > 10000
```

### 6. Filter by Target Node
Inspect all completions handled by a specific cluster GPU node (e.g., node-4):
```text
node : "node-4"
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

---

## ⚖️ Trademarks & Legal Disclaimer

All product names, logos, brands, trademarks, and registered trademarks mentioned or referenced in this project (including but not limited to **DeepSeek**, **Qwen**, **vLLM**, **Llama**, **Ollama**, **OpenAI**, **NVIDIA**, **Apple**, **Apple Silicon**, **MLX**, **Grafana**, **Prometheus**, **Elasticsearch**, **Kibana**, **Logstash**, and **Portainer**) are the property of their respective trademark holders.

Use of these third-party names, logos, or marks does not imply any affiliation with, endorsement by, or sponsorship by their respective owners. **LLM Cluster Router** is an independent open-source proxy project designed solely for routing OpenAI-compatible HTTP requests across self-hosted inference infrastructure.



