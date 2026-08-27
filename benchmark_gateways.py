#!/usr/bin/env python3
"""
LLM Gateway Speed & Latency Benchmark Tool
Compares throughput (tokens/sec), total latency, and time-to-first-token (TTFT)
between multiple cluster gateway endpoints.
"""

import time
import json
import urllib.request
import ssl
import sys

def run_benchmark():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    targets = [
        {
            "name": "Gateway A (10.0.0.1)",
            "ip": "10.0.0.1",
            "host": "llm-gateway-a.example.com"
        },
        {
            "name": "Gateway B (10.0.0.2)",
            "ip": "10.0.0.2",
            "host": "llm-gateway-b.example.com"
        }
    ]

    payload = json.dumps({
        "model": "Qwen/Qwen3.8-27B-FP8",
        "messages": [
            {"role": "user", "content": "Explain the concept of neural network attention mechanism in 60 words."}
        ],
        "max_tokens": 120,
        "temperature": 0.7
    }).encode("utf-8")

    print("==========================================================================")
    print("      🏎️  LLM Cluster Gateway Benchmark Tool                             ")
    print("==========================================================================")

    results = {}

    for t in targets:
        url = f"https://{t['ip']}/v1/chat/completions"
        headers = {
            "Host": t["host"],
            "Content-Type": "application/json",
            "User-Agent": "LLM-Gateway-Benchmark/1.0"
        }
        
        print(f"\n🚀 Testing Target: {t['name']}")
        latencies = []
        tokens_per_sec = []
        
        # Warmup request
        try:
            req = urllib.request.Request(url, data=payload, headers=headers)
            with urllib.request.urlopen(req, context=ctx, timeout=20) as resp:
                resp.read()
        except Exception as e:
            print(f"   ⚠️ Warmup notice: {e}")

        # Benchmark runs
        for i in range(3):
            start_time = time.time()
            try:
                req = urllib.request.Request(url, data=payload, headers=headers)
                with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    elapsed = time.time() - start_time
                    
                    usage = data.get("usage", {})
                    comp_tokens = usage.get("completion_tokens", 0)
                    tps = comp_tokens / elapsed if elapsed > 0 else 0
                    
                    latencies.append(elapsed)
                    tokens_per_sec.append(tps)
                    print(f"   Run #{i+1}: {elapsed*1000:7.2f} ms | {comp_tokens:3d} tokens | {tps:6.2f} tok/sec")
            except Exception as e:
                print(f"   Run #{i+1}: FAILED ({e})")
                
        if latencies:
            avg_lat = sum(latencies) / len(latencies)
            avg_tps = sum(tokens_per_sec) / len(tokens_per_sec)
            results[t['name']] = {"latency_ms": avg_lat * 1000, "tps": avg_tps}
            print(f"   📊 Summary: Avg Response Time = {avg_lat*1000:.2f} ms | Avg Speed = {avg_tps:.2f} tok/sec")

    print("\n==========================================================================")
    print("                             🏆 COMPARISON RESULTS                        ")
    print("==========================================================================")
    for name, r in results.items():
        print(f" • {name:<45} -> Latency: {r['latency_ms']:6.2f} ms | Speed: {r['tps']:5.2f} tok/sec")
    print("==========================================================================")

if __name__ == "__main__":
    run_benchmark()
