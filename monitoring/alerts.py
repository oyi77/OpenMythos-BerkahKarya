"""Alert system for monitoring anomalies.

Provides threshold-based and anomaly detection alerts.
"""
import json
import os
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from .metrics import MetricsCollector


@dataclass
class AlertRule:
    """Alert rule configuration."""
    name: str
    metric: str
    condition: str  # "gt", "lt", "eq", "anomaly"
    threshold: float
    severity: str = "warning"  # info, warning, critical
    message: str = ""
    cooldown: int = 300  # seconds between alerts


@dataclass
class Alert:
    """Triggered alert."""
    rule_name: str
    severity: str
    message: str
    metric_value: float
    threshold: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class AlertManager:
    """Manage monitoring alerts.
    
    Args:
        metrics: MetricsCollector to monitor
        alert_path: Path to store alerts
    """
    metrics: MetricsCollector
    alert_path: str = "monitoring/alerts"
    rules: List[AlertRule] = field(default_factory=list)
    alerts: List[Alert] = field(default_factory=list)
    _last_triggered: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        os.makedirs(self.alert_path, exist_ok=True)
        self._load_default_rules()

    def _load_default_rules(self):
        """Load default alert rules."""
        self.rules = [
            AlertRule("high_cpu", "cpu_usage", "gt", 90, "warning",
                      "CPU usage above 90%"),
            AlertRule("high_ram", "ram_usage", "gt", 90, "warning",
                      "RAM usage above 90%"),
            AlertRule("high_disk", "disk_usage", "gt", 95, "critical",
                      "Disk usage above 95%"),
            AlertRule("high_latency", "inference_latency", "gt", 1000, "warning",
                      "Inference latency above 1000ms"),
            AlertRule("low_throughput", "throughput", "lt", 10, "warning",
                      "Throughput below 10 tokens/sec"),
            AlertRule("training_loss_spike", "training_loss", "gt", 10, "critical",
                      "Training loss spike detected"),
        ]

    def add_rule(self, rule: AlertRule):
        """Add a custom alert rule."""
        self.rules.append(rule)

    def check_alerts(self) -> List[Alert]:
        """Check all alert rules against current metrics.
        
        Returns:
            List of triggered alerts
        """
        triggered = []

        for rule in self.rules:
            # Check cooldown
            last = self._last_triggered.get(rule.name, 0)
            if time.time() - last < rule.cooldown:
                continue

            # Get latest metric
            point = self.metrics.get_latest(rule.metric)
            if point is None:
                continue

            # Evaluate condition
            triggered_now = False
            if rule.condition == "gt" and point.value > rule.threshold:
                triggered_now = True
            elif rule.condition == "lt" and point.value < rule.threshold:
                triggered_now = True
            elif rule.condition == "eq" and point.value == rule.threshold:
                triggered_now = True

            if triggered_now:
                alert = Alert(
                    rule_name=rule.name,
                    severity=rule.severity,
                    message=rule.message or f"{rule.metric} = {point.value} (threshold: {rule.threshold})",
                    metric_value=point.value,
                    threshold=rule.threshold,
                )
                triggered.append(alert)
                self.alerts.append(alert)
                self._last_triggered[rule.name] = time.time()

        return triggered

    def get_recent_alerts(self, hours: int = 24) -> List[Alert]:
        """Get alerts from the last N hours."""
        cutoff = time.time() - (hours * 3600)
        return [a for a in self.alerts if a.timestamp > cutoff]

    def export_alerts(self, path: str = None):
        """Export alerts to JSON file."""
        if path is None:
            path = os.path.join(self.alert_path, f"alerts_{int(time.time())}.json")
        data = [
            {
                "rule": a.rule_name,
                "severity": a.severity,
                "message": a.message,
                "value": a.metric_value,
                "threshold": a.threshold,
                "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(a.timestamp)),
            }
            for a in self.alerts
        ]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path
