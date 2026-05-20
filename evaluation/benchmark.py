"""Core benchmarking framework for OpenMythos finance models.

Provides evaluation of model quality, inference performance, and resource usage.
"""
import json
import time
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    value: float
    unit: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))


@dataclass
class BenchmarkSuite:
    """Comprehensive benchmark suite for finance models.
    
    Args:
        data_path: Path to training data JSONL file
        output_dir: Directory for benchmark results
    """
    data_path: str = "data/finance/finance_dataset.jsonl"
    output_dir: str = "evaluation/results"
    results: List[BenchmarkResult] = field(default_factory=list)

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)

    # ── Data Quality ────────────────────────────────────────────
    def load_data(self) -> List[Dict]:
        """Load training data from JSONL file."""
        samples = []
        with open(self.data_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    def evaluate_data_quality(self) -> Dict[str, Any]:
        """Evaluate training data quality metrics.
        
        Returns:
            Dictionary with quality metrics
        """
        samples = self.load_data()
        if not samples:
            return {"error": "No samples found"}

        lengths = [len(s.get("text", "")) for s in samples]
        domains = {}
        for s in samples:
            text = s.get("text", "")
            if "## Trading" in text or "trading" in text.lower():
                domains["trading"] = domains.get("trading", 0) + 1
            if "## Business" in text or "business" in text.lower():
                domains["business"] = domains.get("business", 0) + 1
            if "## Cashflow" in text or "cashflow" in text.lower():
                domains["cashflow"] = domains.get("cashflow", 0) + 1
            if "## Risk" in text or "risk" in text.lower():
                domains["risk"] = domains.get("risk", 0) + 1
            if "crypto" in text.lower():
                domains["crypto"] = domains.get("crypto", 0) + 1
            if "forex" in text.lower():
                domains["forex"] = domains.get("forex", 0) + 1

        metrics = {
            "total_samples": len(samples),
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "median_length": sorted(lengths)[len(lengths) // 2],
            "domain_distribution": domains,
            "empty_samples": sum(1 for l in lengths if l < 50),
            "quality_score": 1.0 - (sum(1 for l in lengths if l < 50) / len(samples)),
        }

        self.results.append(BenchmarkResult(
            name="data_quality",
            value=metrics["quality_score"],
            unit="score",
            metadata=metrics,
        ))
        return metrics

    # ── Tokenizer Benchmarks ────────────────────────────────────
    def benchmark_tokenizer(self, text_samples: Optional[List[str]] = None) -> Dict[str, Any]:
        """Benchmark tokenizer performance.
        
        Args:
            text_samples: Optional list of text samples. Uses data if not provided.
            
        Returns:
            Tokenizer benchmark results
        """
        if text_samples is None:
            samples = self.load_data()[:100]
            text_samples = [s.get("text", "") for s in samples]

        # Try to import enhanced tokenizer
        try:
            from open_mythos.enhanced_tokenizer import create_finance_tokenizer
            tokenizer = create_finance_tokenizer()
            vocab_size = tokenizer.get_finance_vocab_size()
        except ImportError:
            tokenizer = None
            vocab_size = 0

        # Simple tokenization benchmark (word-level)
        token_counts = []
        start = time.time()
        for text in text_samples:
            tokens = text.split()
            token_counts.append(len(tokens))
        elapsed = time.time() - start

        metrics = {
            "samples_tokenized": len(text_samples),
            "total_time_sec": round(elapsed, 4),
            "samples_per_sec": round(len(text_samples) / elapsed, 2) if elapsed > 0 else 0,
            "avg_tokens_per_sample": sum(token_counts) / len(token_counts) if token_counts else 0,
            "finance_vocab_size": vocab_size,
        }

        self.results.append(BenchmarkResult(
            name="tokenizer_throughput",
            value=metrics["samples_per_sec"],
            unit="samples/sec",
            metadata=metrics,
        ))
        return metrics

    # ── Memory Benchmarks ───────────────────────────────────────
    def benchmark_memory(self) -> Dict[str, Any]:
        """Measure memory usage statistics.
        
        Returns:
            Memory usage in MB
        """
        import sys
        import gc

        gc.collect()
        python_mem = sys.getsizeof(self) / (1024 * 1024)

        # Estimate data memory footprint
        samples = self.load_data()
        data_size_bytes = sum(len(json.dumps(s).encode()) for s in samples)
        data_size_mb = data_size_bytes / (1024 * 1024)

        metrics = {
            "python_process_mb": round(python_mem, 2),
            "data_size_mb": round(data_size_mb, 2),
            "total_samples": len(samples),
            "bytes_per_sample": data_size_bytes / len(samples) if samples else 0,
        }

        # Check system memory
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if "MemTotal" in line:
                        metrics["system_total_mb"] = int(line.split()[1]) / 1024
                    elif "MemAvailable" in line:
                        metrics["system_available_mb"] = int(line.split()[1]) / 1024
        except Exception:
            pass

        self.results.append(BenchmarkResult(
            name="memory_usage",
            value=metrics["data_size_mb"],
            unit="MB",
            metadata=metrics,
        ))
        return metrics

    # ── Run All ─────────────────────────────────────────────────
    def run_all(self) -> Dict[str, Any]:
        """Run all benchmarks and return combined results.
        
        Returns:
            Combined benchmark results
        """
        results = {
            "data_quality": self.evaluate_data_quality(),
            "tokenizer": self.benchmark_tokenizer(),
            "memory": self.benchmark_memory(),
            "summary": {
                "total_benchmarks": len(self.results),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        }
        return results

    def export_results(self, filename: str = "benchmark_results.json"):
        """Export results to JSON file."""
        path = os.path.join(self.output_dir, filename)
        data = [asdict(r) for r in self.results]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path
