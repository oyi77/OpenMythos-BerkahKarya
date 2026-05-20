"""Web dashboard for OpenMythos monitoring.

Provides real-time visualization of training and inference metrics.
"""
import json
from typing import Dict, Any
from .metrics import MetricsCollector


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <title>OpenMythos Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0d1117; color: #c9d1d9; }
        .header { background: #161b22; padding: 20px; border-bottom: 1px solid #30363d; }
        .header h1 { color: #58a6ff; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; padding: 20px; }
        .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; }
        .card h3 { color: #58a6ff; margin-bottom: 15px; }
        .metric { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #21262d; }
        .metric:last-child { border: none; }
        .metric-label { color: #8b949e; }
        .metric-value { font-weight: 600; color: #f0f6fc; }
        .chart-container { height: 300px; margin-top: 15px; }
        .status { padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; }
        .status-ok { background: #1b4332; color: #3fb950; }
        .status-warn { background: #3d2e00; color: #d29922; }
        .status-error { background: #3d1418; color: #f85149; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔥 OpenMythos Monitoring Dashboard</h1>
        <p>BerkahKarya Finance Models — Real-time Analytics</p>
    </div>
    <div class="grid">
        <div class="card">
            <h3>📊 Data Quality</h3>
            <div id="data-metrics"></div>
        </div>
        <div class="card">
            <h3>🚀 Inference Performance</h3>
            <div id="inference-metrics"></div>
        </div>
        <div class="card">
            <h3>💻 System Resources</h3>
            <div id="system-metrics"></div>
        </div>
        <div class="card">
            <h3>📈 Domain Coverage</h3>
            <div class="chart-container">
                <canvas id="domain-chart"></canvas>
            </div>
        </div>
    </div>
    <script>
        // Auto-refresh every 30s
        setTimeout(() => location.reload(), 30000);
    </script>
</body>
</html>"""


@dataclass
class Dashboard:
    """Web dashboard for monitoring.
    
    Args:
        metrics: MetricsCollector instance
        port: Port to serve dashboard on
    """
    metrics: MetricsCollector
    port: int = 8050

    def generate_html(self) -> str:
        """Generate dashboard HTML from current metrics."""
        summaries = self.metrics.get_all_summaries()

        # Data metrics section
        data_html = ""
        for name in ["data_quality", "training_loss", "learning_rate"]:
            if name in summaries:
                s = summaries[name]
                data_html += f"""
                <div class="metric">
                    <span class="metric-label">{name.replace('_', ' ').title()}</span>
                    <span class="metric-value">{s.get('latest', 'N/A'):.4f}</span>
                </div>"""

        # Inference metrics section
        inf_html = ""
        for name in ["inference_latency", "throughput"]:
            if name in summaries:
                s = summaries[name]
                inf_html += f"""
                <div class="metric">
                    <span class="metric-label">{name.replace('_', ' ').title()}</span>
                    <span class="metric-value">{s.get('latest', 'N/A'):.2f}</span>
                </div>"""

        # System metrics section
        sys_html = ""
        for name in ["cpu_usage", "cpu_load", "ram_usage", "disk_usage"]:
            if name in summaries:
                s = summaries[name]
                sys_html += f"""
                <div class="metric">
                    <span class="metric-label">{name.replace('_', ' ').title()}</span>
                    <span class="metric-value">{s.get('latest', 'N/A'):.1f}</span>
                </div>"""

        html = HTML_TEMPLATE
        html = html.replace('<div id="data-metrics"></div>', data_html or "<p>No data yet</p>")
        html = html.replace('<div id="inference-metrics"></div>', inf_html or "<p>No data yet</p>")
        html = html.replace('<div id="system-metrics"></div>', sys_html or "<p>No data yet</p>")

        return html

    def save_dashboard(self, path: str = "monitoring/dashboard.html"):
        """Save dashboard as static HTML file."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(self.generate_html())
        return path
