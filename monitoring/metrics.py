"""Metrics collection for training and inference monitoring.

Collects and stores time-series metrics for analysis.
"""
import json
import time
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict


@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: float
    unit: str
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricsCollector:
    """Collect and store metrics.
    
    Args:
        storage_path: Path to store metrics
        flush_interval: Seconds between auto-flushes
    """
    storage_path: str = "monitoring/metrics"
    flush_interval: int = 60
    metrics: Dict[str, List[MetricPoint]] = field(default_factory=lambda: defaultdict(list))
    _last_flush: float = field(default_factory=time.time)

    def __post_init__(self):
        os.makedirs(self.storage_path, exist_ok=True)

    # ── Training Metrics ────────────────────────────────────────
    def log_training_loss(self, epoch: int, step: int, loss: float):
        """Log training loss."""
        self.metrics["training_loss"].append(MetricPoint(
            name="training_loss",
            value=loss,
            unit="loss",
            tags={"epoch": str(epoch), "step": str(step)},
        ))

    def log_learning_rate(self, step: int, lr: float):
        """Log learning rate."""
        self.metrics["learning_rate"].append(MetricPoint(
            name="learning_rate",
            value=lr,
            unit="lr",
            tags={"step": str(step)},
        ))

    def log_gradient_norm(self, step: int, norm: float):
        """Log gradient norm."""
        self.metrics["gradient_norm"].append(MetricPoint(
            name="gradient_norm",
            value=norm,
            unit="norm",
            tags={"step": str(step)},
        ))

    # ── Inference Metrics ───────────────────────────────────────
    def log_inference_latency(self, latency_ms: float, tokens: int = 1):
        """Log inference latency."""
        self.metrics["inference_latency"].append(MetricPoint(
            name="inference_latency",
            value=latency_ms,
            unit="ms",
            tags={"tokens": str(tokens)},
        ))

    def log_throughput(self, tokens_per_sec: float):
        """Log inference throughput."""
        self.metrics["throughput"].append(MetricPoint(
            name="throughput",
            value=tokens_per_sec,
            unit="tokens/sec",
        ))

    def log_memory_usage(self, gpu_mb: float, cpu_mb: float):
        """Log memory usage."""
        self.metrics["memory_gpu"].append(MetricPoint(
            name="memory_gpu", value=gpu_mb, unit="MB",
        ))
        self.metrics["memory_cpu"].append(MetricPoint(
            name="memory_cpu", value=cpu_mb, unit="MB",
        ))

    # ── System Metrics ──────────────────────────────────────────
    def log_system_metrics(self):
        """Log current system metrics (CPU, RAM, Disk)."""
        try:
            import psutil
            cpu_pct = psutil.cpu_percent(interval=1)
            ram = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            self.metrics["cpu_usage"].append(MetricPoint(
                name="cpu_usage", value=cpu_pct, unit="%",
            ))
            self.metrics["ram_usage"].append(MetricPoint(
                name="ram_usage", value=ram.percent, unit="%",
            ))
            self.metrics["disk_usage"].append(MetricPoint(
                name="disk_usage", value=disk.percent, unit="%",
            ))
        except ImportError:
            # Fallback: read from /proc
            try:
                with open("/proc/loadavg") as f:
                    load = float(f.read().split()[0])
                self.metrics["cpu_load"].append(MetricPoint(
                    name="cpu_load", value=load, unit="load",
                ))
            except Exception:
                pass

    # ── Finance Metrics ─────────────────────────────────────────
    def log_domain_score(self, domain: str, score: float):
        """Log finance domain evaluation score."""
        self.metrics[f"domain_{domain}"].append(MetricPoint(
            name=f"domain_{domain}", value=score, unit="score",
            tags={"domain": domain},
        ))

    def log_data_quality(self, quality_score: float, total_samples: int):
        """Log data quality metrics."""
        self.metrics["data_quality"].append(MetricPoint(
            name="data_quality", value=quality_score, unit="score",
            tags={"samples": str(total_samples)},
        ))

    # ── Aggregation & Export ────────────────────────────────────
    def get_latest(self, metric_name: str) -> Optional[MetricPoint]:
        """Get latest value for a metric."""
        points = self.metrics.get(metric_name, [])
        return points[-1] if points else None

    def get_summary(self, metric_name: str) -> Dict[str, Any]:
        """Get summary statistics for a metric."""
        points = self.metrics.get(metric_name, [])
        if not points:
            return {"count": 0}

        values = [p.value for p in points]
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "latest": values[-1],
        }

    def get_all_summaries(self) -> Dict[str, Dict]:
        """Get summaries for all collected metrics."""
        return {name: self.get_summary(name) for name in self.metrics}

    def flush(self):
        """Write metrics to disk."""
        path = os.path.join(self.storage_path, f"metrics_{int(time.time())}.json")
        data = {name: [asdict(p) for p in points] for name, points in self.metrics.items()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self._last_flush = time.time()
        return path

    def auto_flush(self):
        """Flush if interval has elapsed."""
        if time.time() - self._last_flush > self.flush_interval:
            return self.flush()
        return None
