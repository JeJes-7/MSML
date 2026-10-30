from prometheus_client import start_http_server, Counter, Histogram, Gauge
import time
import psutil
import random

# ===== METRICS =====
REQUEST_COUNT = Counter(
    "request_total",
    "Total number of inference requests"
)

REQUEST_LATENCY = Histogram(
    "request_latency_seconds",
    "Inference request latency"
)

ERROR_COUNT = Counter(
    "prediction_error_total",
    "Total number of prediction errors"
)

CPU_USAGE = Gauge(
    "process_cpu_usage",
    "CPU usage percentage"
)

MEMORY_USAGE = Gauge(
    "process_memory_usage",
    "Memory usage percentage"
)

# ===== START METRICS SERVER =====
if __name__ == "__main__":
    start_http_server(8000)
    print("Prometheus exporter running on http://127.0.0.1:8000/metrics")

    while True:
        # simulasi request
        REQUEST_COUNT.inc()
        REQUEST_LATENCY.observe(random.random())

        # simulasi error
        if random.random() < 0.2:
            ERROR_COUNT.inc()

        # system metrics
        CPU_USAGE.set(psutil.cpu_percent())
        MEMORY_USAGE.set(psutil.virtual_memory().percent)

        time.sleep(5)
