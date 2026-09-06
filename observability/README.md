# Observability reference configs

Two independent paths, use either or both — nothing here is required to
run the router itself.

## Logs → ELK (Elasticsearch, Logstash, Kibana)

The router always writes a structured `AUDIT_LOG: {...}` JSON line per
request to stdout (see the main README's "Observability & Auditing"
section for the field list) — that alone works with any log shipper
(Filebeat, Promtail, a `docker logs` pipe, CloudWatch agent, etc.).

If you want the router to *also* push that same line directly over TCP
to Logstash, set `LOGSTASH_HOST` / `LOGSTASH_PORT` (see the main
`docker-compose.yml`) and run something like the stack below:

- [`docker-compose.elk.example.yml`](./docker-compose.elk.example.yml) —
  Elasticsearch + Logstash + Kibana, wired to the router.
- [`logstash.conf`](./logstash.conf) — the pipeline the compose file
  mounts into Logstash. Minimal on purpose (single index, no TLS) —
  treat it as a starting point, not a production config.

```bash
cp observability/docker-compose.elk.example.yml docker-compose.elk.yml
docker compose -f docker-compose.yml -f docker-compose.elk.yml up -d
```

Then use the Kibana KQL queries in the main README to explore
`llm-audit-logs-*`.

## Metrics → Prometheus

The router exposes `/metrics` in standard Prometheus exposition format
whenever `prometheus_client` is installed (an optional dependency — see
`requirements.txt`). See the main README's "Metrics (Prometheus)"
section for what each metric means.

- [`prometheus.example.yaml`](./prometheus.example.yaml) — a scrape job
  to merge into your own `prometheus.yml`.
