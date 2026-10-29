"""Tracing and structured logging.

If APPLICATIONINSIGHTS_CONNECTION_STRING is set, OpenTelemetry traces, logs and
metrics are exported to Azure Monitor / Application Insights. Otherwise every
call is a cheap no-op, so the same code runs locally, in CI and in Azure.
"""

from __future__ import annotations

import contextlib
import logging
import os
import time

_TRACER = None
_CONFIGURED = False


class _NoopSpan:
    def set_attribute(self, *_a, **_k):
        return None

    def set_attributes(self, *_a, **_k):
        return None

    def record_exception(self, *_a, **_k):
        return None

    def add_event(self, *_a, **_k):
        return None


class _NoopTracer:
    @contextlib.contextmanager
    def start_as_current_span(self, _name: str, **_kwargs):
        yield _NoopSpan()


def configure(service_name: str = "credit-risk") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
    conn = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
    if not conn:
        return
    try:
        from azure.monitor.opentelemetry import configure_azure_monitor

        os.environ.setdefault("OTEL_SERVICE_NAME", service_name)
        configure_azure_monitor(connection_string=conn, logger_name="creditrisk")
        logging.getLogger(__name__).info("Azure Monitor OpenTelemetry exporter enabled")
    except Exception as exc:  # noqa: BLE001 - telemetry must never break scoring
        logging.getLogger(__name__).warning("Telemetry disabled: %s", exc)


def tracer(service_name: str = "credit-risk"):
    global _TRACER
    if _TRACER is None:
        configure(service_name)
        try:
            from opentelemetry import trace

            _TRACER = trace.get_tracer("creditrisk")
        except ImportError:
            _TRACER = _NoopTracer()
    return _TRACER


def get_logger(name: str) -> logging.Logger:
    configure()
    return logging.getLogger(f"creditrisk.{name}")


def log_event(logger: logging.Logger, event: str, **properties) -> None:
    """Structured event; properties become customDimensions in Application Insights."""
    logger.info(
        event, extra={k: (v if isinstance(v, int | float | str | bool) else str(v)) for k, v in properties.items()}
    )


@contextlib.contextmanager
def timed(span_name: str, **attributes):
    """Span + wall-clock duration attribute in one context manager."""
    start = time.perf_counter()
    with tracer().start_as_current_span(span_name) as span:
        span.set_attributes({k: v for k, v in attributes.items() if v is not None})
        try:
            yield span
        finally:
            span.set_attribute("duration_ms", round((time.perf_counter() - start) * 1000, 1))
