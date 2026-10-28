"""
Listens for the tracker's UDP packets in place of Unity and reports the update rate as received, the packet
interval jitter and the latency from camera capture to arrival (scene.capture_time is an epoch timestamp,
so this only works with the tracker on the same machine).

    python -m evaluation.udp_monitor [--port 5005] [--seconds 30]

Unity must not be running (it listens on the same port).
"""

import argparse
import json
import socket
import time

import numpy as np

import config
from evaluation.metrics import fps_stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=config.UDP_PORT)
    parser.add_argument("--seconds", type=float, default=30.0)
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", args.port))
    sock.settimeout(1.0)
    print(f"Listening on port {args.port} for {args.seconds:.0f} s...")

    arrivals, latencies, sizes = [], [], []
    end = time.perf_counter() + args.seconds
    while time.perf_counter() < end:
        try:
            data, _ = sock.recvfrom(65535)
        except socket.timeout:
            continue
        arrivals.append(time.perf_counter())
        sizes.append(len(data))
        try:
            capture = json.loads(data).get("scene", {}).get("capture_time")
        except ValueError:
            capture = None
        if capture:
            latencies.append(time.time() * 1000.0 - capture)
    sock.close()

    if len(arrivals) < 2:
        raise SystemExit("Received fewer than 2 packets; is the tracker running?")
    interval = np.diff(arrivals) * 1000.0
    print(f"{len(arrivals)} packets, {(len(arrivals) - 1) / (arrivals[-1] - arrivals[0]):.1f} Hz received")
    print(f"Interval ms: median {np.median(interval):.1f}, std {interval.std():.1f}, "
          f"p95 {np.percentile(interval, 95):.1f}, max {interval.max():.1f}")
    fps = fps_stats(np.array(arrivals))
    print(f"Frame rate: 5% low {fps['low_5pct']:.1f}, 1% low {fps['low_1pct']:.1f}"
          + (f", slowest second {fps['min_1s']:.0f} fps" if "min_1s" in fps else ""))
    print(f"Packet size bytes: median {np.median(sizes):.0f}, max {max(sizes)}")
    if latencies:
        v = np.array(latencies)
        print(f"Capture -> arrival ms: median {np.median(v):.1f}, p95 {np.percentile(v, 95):.1f}, max {v.max():.1f}")


if __name__ == "__main__":
    main()
