"""Finance-specific evaluation tasks for OpenMythos models.

Evaluates model performance on finance domain tasks.
"""
import json
import re
from typing import Dict, List, Any
from dataclasses import dataclass, field


@dataclass
class FinanceEvaluator:
    """Evaluate finance model performance.
    
    Args:
        data_path: Path to evaluation data
    """
    data_path: str = "data/finance/finance_dataset.jsonl"
    domain_scores: Dict[str, float] = field(default_factory=dict)

    # ── Domain Knowledge Tests ──────────────────────────────────
    TRADING_CONCEPTS = {
        "support_resistance": ["support", "resistance", "level", "bounce", "break"],
        "risk_management": ["stop loss", "take profit", "risk/reward", "position sizing", "drawdown"],
        "technical_analysis": ["rsi", "macd", "bollinger", "moving average", "trend"],
        "order_types": ["buy limit", "sell limit", "buy stop", "sell stop", "market order"],
        "chart_patterns": ["doji", "hammer", "engulfing", "head and shoulders", "double top"],
    }

    BUSINESS_CONCEPTS = {
        "financial_statements": ["balance sheet", "income statement", "cash flow", "assets", "liabilities"],
        "valuation": ["p/e ratio", "eps", "dcf", "npv", "irr"],
        "profitability": ["roi", "roe", "roa", "ebitda", "margin"],
        "growth": ["revenue growth", "market share", "expansion", "acquisition"],
        "risk": ["risk assessment", "mitigation", "compliance", "audit"],
    }

    INDONESIAN_CONCEPTS = {
        "market_structure": ["idx", "ihsg", "bei", "ojk", "bappebti"],
        "instruments": ["saham", "obligasi", "reksadana", "etf", "komoditas"],
        "regulation": ["otoritas jasa keuangan", "pasar modal", "perlindungan konsumen"],
        "local_brands": ["bca", "bri", "mandiri", "telkom", "unilever"],
    }

    def load_data(self) -> List[Dict]:
        """Load evaluation data."""
        samples = []
        try:
            with open(self.data_path, "r") as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
        except FileNotFoundError:
            pass
        return samples

    def evaluate_domain_coverage(self) -> Dict[str, Any]:
        """Evaluate how well the data covers finance domains.
        
        Returns:
            Domain coverage metrics
        """
        samples = self.load_data()
        if not samples:
            return {"error": "No data found"}

        text_combined = " ".join(s.get("text", "").lower() for s in samples)

        coverage = {}
        all_concepts = {
            **{f"trading_{k}": v for k, v in self.TRADING_CONCEPTS.items()},
            **{f"business_{k}": v for k, v in self.BUSINESS_CONCEPTS.items()},
            **{f"indonesian_{k}": v for k, v in self.INDONESIAN_CONCEPTS.items()},
        }

        for concept_name, keywords in all_concepts.items():
            hits = sum(1 for kw in keywords if kw in text_combined)
            coverage[concept_name] = {
                "hits": hits,
                "total": len(keywords),
                "coverage_pct": round(hits / len(keywords) * 100, 1),
            }

        # Overall scores by domain
        domain_scores = {}
        for domain, concepts in [
            ("trading", self.TRADING_CONCEPTS),
            ("business", self.BUSINESS_CONCEPTS),
            ("indonesian", self.INDONESIAN_CONCEPTS),
        ]:
            scores = []
            for concept_name, keywords in concepts.items():
                key = f"{domain}_{concept_name}"
                if key in coverage:
                    scores.append(coverage[key]["coverage_pct"])
            domain_scores[domain] = round(sum(scores) / len(scores), 1) if scores else 0

        self.domain_scores = domain_scores

        return {
            "concept_coverage": coverage,
            "domain_scores": domain_scores,
            "overall_score": round(sum(domain_scores.values()) / len(domain_scores), 1) if domain_scores else 0,
        }

    def evaluate_trading_quality(self) -> Dict[str, Any]:
        """Evaluate trading-specific content quality.
        
        Returns:
            Trading content quality metrics
        """
        samples = self.load_data()
        trading_samples = [
            s for s in samples
            if any(kw in s.get("text", "").lower() for kw in ["trading", "trade", "position", "order"])
        ]

        if not trading_samples:
            return {"total": 0, "quality_score": 0}

        # Check for key trading elements
        has_entry = sum(1 for s in trading_samples if any(w in s["text"].lower() for w in ["entry", "buy", "sell"]))
        has_exit = sum(1 for s in trading_samples if any(w in s["text"].lower() for w in ["exit", "close", "tp", "sl"]))
        has_risk = sum(1 for s in trading_samples if any(w in s["text"].lower() for w in ["risk", "stop loss", "drawdown"]))
        has_metrics = sum(1 for s in trading_samples if any(w in s["text"].lower() for w in ["win rate", "profit factor", "sharpe"]))

        total = len(trading_samples)
        quality = ((has_entry + has_exit + has_risk + has_metrics) / (4 * total)) * 100 if total else 0

        return {
            "total_trading_samples": total,
            "has_entry_rules": has_entry,
            "has_exit_rules": has_exit,
            "has_risk_management": has_risk,
            "has_performance_metrics": has_metrics,
            "quality_score": round(quality, 1),
        }

    def run_full_evaluation(self) -> Dict[str, Any]:
        """Run complete finance evaluation.
        
        Returns:
            Full evaluation results
        """
        return {
            "domain_coverage": self.evaluate_domain_coverage(),
            "trading_quality": self.evaluate_trading_quality(),
            "total_samples": len(self.load_data()),
        }
