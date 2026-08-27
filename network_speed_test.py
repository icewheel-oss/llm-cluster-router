#!/usr/bin/env python3
"""
Raw Network Bandwidth & Ping Latency Measurement Tool
Measures raw network throughput (Mbps / Gbps) and RTT latency between cluster nodes.
Usage: python3 network_speed_test.py [server_ip] [client_ip]
"""

import sys
import time
import socket
import threading

PORT = 9998
DURATION = 5

def run_server(host="0.0.0.0", port=PORT):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)
    print(f"📡 Speed Test Server listening on {host}:{port}...")
    
    conn, addr = server.accept()
    print(f"🤝 Client connected from {addr[0]}")
    
    total_bytes = 0
    start_time = time.time()
    
    while True:
        data = conn.recv(131072)
        if not data:
            break
        total_bytes += len(data)
        if time.time() - start_time >= DURATION:
            break
            
    elapsed = time.time() - start_time
    conn.close()
    server.close()
    
    mbps = (total_bytes * 8) / (elapsed * 1e6)
    gbps = (total_bytes * 8) / (elapsed * 1e9)
    print(f"\n📊 RESULTS (Server Receiver):")
    print(f" • Transferred: {total_bytes / (1024*1024):.2f} MB in {elapsed:.2f} seconds")
    print(f" • Network Throughput: {mbps:.2f} Mbps ({gbps:.3f} Gbps)")

def run_client(target_ip, port=PORT):
    print(f"🚀 Connecting to Speed Test Server at {target_ip}:{port}...")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((target_ip, port))
    
    chunk = b"0" * 131072
    start_time = time.time()
    sent_bytes = 0
    
    print(f"⚡ Streaming data payload for {DURATION} seconds...")
    while time.time() - start_time < DURATION:
        try:
            s.sendall(chunk)
            sent_bytes += len(chunk)
        except Exception:
            break
            
    elapsed = time.time() - start_time
    s.close()
    
    mbps = (sent_bytes * 8) / (elapsed * 1e6)
    gbps = (sent_bytes * 8) / (elapsed * 1e9)
    print(f"\n📊 RESULTS (Client Sender):")
    print(f" • Sent: {sent_bytes / (1024*1024):.2f} MB in {elapsed:.2f} seconds")
    print(f" • Send Bandwidth: {mbps:.2f} Mbps ({gbps:.3f} Gbps)")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        run_server()
    elif len(sys.argv) > 2 and sys.argv[1] == "client":
        run_client(sys.argv[2])
    else:
        print("Usage:")
        print("  Receiver Node: python3 network_speed_test.py server")
        print("  Sender Node:   python3 network_speed_test.py client <receiver_ip>")
