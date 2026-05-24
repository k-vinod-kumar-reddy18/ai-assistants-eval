"""
Observability — lightweight tracing and metrics.
Emits structured logs compatible with OpenTelemetry / Grafana Loki.
Swap `console_exporter` for OTLP exporter in production:
  from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
"""

import time
import logging
import json
from contextlib import contextmanager

logger = logging.getLogger("observability")

# Metric store (in-memory for demo; use Prometheus / StatsD in prod)
_metrics: dict[str, list[float]] = {}


@contextmanager
def span(name: str, attributes: dict | None = None):
    t0 = time.time()
    attrs = attributes or {}
    try:
        yield
    finally:
        duration_ms = (time.time() - t0) * 1000
        logger.info(
            json.dumps({
                "span": name,
                "duration_ms": round(duration_ms, 2),
                **attrs,
            })
        )


def record_metric(name: str, value: float, labels: dict | None = None):
    key = f"{name}:{json.dumps(labels or {}, sort_keys=True)}"
    _metrics.setdefault(key, []).append(value)
    logger.info(json.dumps({"metric": name, "value": value, **(labels or {})}))


def get_metrics_summary() -> dict:
    summary = {}
    for key, values in _metrics.items():
        name, labels_str = key.split(":", 1)
        avg = sum(values) / len(values) if values else 0
        summary[key] = {
            "name": name,
            "labels": json.loads(labels_str),
            "count": len(values),
            "avg": round(avg, 2),
            "min": round(min(values), 2),
            "max": round(max(values), 2),
        }
    return summary
