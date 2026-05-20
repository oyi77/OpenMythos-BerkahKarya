"""Finance-specialized chat interface for OpenMythos.

Provides domain-specific system prompts and conversation management.
"""
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from .engine import InferenceEngine, GenerationConfig


SYSTEM_PROMPTS = {
    "trading": """You are a professional trading analyst specializing in technical and fundamental analysis.
Provide clear entry/exit points, risk management, and market analysis.
Always include: entry price, stop loss, take profit, and risk/reward ratio.
Focus on XAUUSD, EURUSD, GBPUSD, BTCUSD, and major indices.""",

    "business": """You are a senior business analyst and financial consultant.
Provide comprehensive business analysis including financial metrics, valuation, and strategic recommendations.
Focus on ROI, profitability, growth potential, and risk assessment.""",

    "cashflow": """You are a personal finance and cashflow management expert.
Help users track income, expenses, savings, and investments.
Provide actionable advice for improving cashflow and building wealth.""",

    "indonesian_market": """You are an Indonesian market specialist.
Expert in IDX stocks (BBCA, BBRI, BMRI, TLKM), Indonesian regulations (OJK, Bappebti),
and local investment opportunities. Provide analysis in both English and Indonesian context.""",

    "crypto": """You are a cryptocurrency and DeFi expert.
Analyze tokens, DeFi protocols, and blockchain technology.
Focus on risk management and technical analysis for crypto markets.""",

    "risk": """You are a risk management specialist.
Evaluate risks across trading, business, and investment decisions.
Provide quantitative risk assessments and mitigation strategies.""",
}


@dataclass
class FinanceChat:
    """Finance-specialized chat interface.
    
    Args:
        domain: Finance domain for specialized responses
        engine: Inference engine instance
    """
    domain: str = "trading"
    engine: Optional[InferenceEngine] = None
    history: List[Dict[str, str]] = field(default_factory=list)

    def __post_init__(self):
        if self.engine is None:
            self.engine = InferenceEngine()

    def get_system_prompt(self) -> str:
        """Get system prompt for current domain."""
        return SYSTEM_PROMPTS.get(self.domain, SYSTEM_PROMPTS["trading"])

    def chat(self, message: str, config: GenerationConfig = None) -> str:
        """Chat with the finance model.
        
        Args:
            message: User message
            config: Generation configuration
            
        Returns:
            Model response
        """
        # Build context with system prompt and history
        context = self.get_system_prompt() + "\n\n"

        # Add conversation history
        for entry in self.history[-5:]:  # Last 5 turns
            context += f"User: {entry['user']}\nAssistant: {entry['assistant']}\n\n"

        context += f"User: {message}\nAssistant:"

        # Generate response
        response = self.engine.generate(context, config)

        # Store in history
        self.history.append({"user": message, "assistant": response})

        return response

    def switch_domain(self, domain: str):
        """Switch finance domain.
        
        Args:
            domain: New domain (trading, business, cashflow, indonesian_market, crypto, risk)
        """
        if domain in SYSTEM_PROMPTS:
            self.domain = domain
        else:
            raise ValueError(f"Unknown domain: {domain}. Available: {list(SYSTEM_PROMPTS.keys())}")

    def clear_history(self):
        """Clear conversation history."""
        self.history.clear()

    def get_available_domains(self) -> List[str]:
        """Get list of available finance domains."""
        return list(SYSTEM_PROMPTS.keys())

    def export_history(self) -> str:
        """Export conversation history as JSON."""
        import json
        return json.dumps(self.history, indent=2)
