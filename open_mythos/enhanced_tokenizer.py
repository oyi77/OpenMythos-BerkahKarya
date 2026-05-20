"""Enhanced Tokenizer for OpenMythos Finance Models.

Adds domain-specific vocabulary for finance, trading, and business terms.
Improves tokenization quality for finance content.
"""

import re
from typing import List, Dict, Optional, Set
from dataclasses import dataclass


@dataclass
class FinanceVocabConfig:
    """Finance vocabulary configuration."""
    trading_terms: bool = True
    business_terms: bool = True
    crypto_terms: bool = True
    forex_terms: bool = True
    indonesian_terms: bool = True
    technical_analysis: bool = True


# Finance-specific vocabulary
FINANCE_VOCABULARY = {
    # Trading terms
    "trading": ["XAUUSD", "EURUSD", "GBPUSD", "BTCUSD", "ETHUSD", "USDJPY", "NAS100", "SPX500"],
    "analysis": ["RSI", "MACD", "Bollinger", "Fibonacci", "Ichimoku", "Stochastic", "ATR", "EMA", "SMA"],
    "patterns": ["Doji", "Hammer", "Engulfing", "Harami", "Morning Star", "Evening Star", "Three White Soldiers"],
    "orders": ["Buy Limit", "Sell Limit", "Buy Stop", "Sell Stop", "Market Order", "OCO", "Trailing Stop"],
    "risk": ["Stop Loss", "Take Profit", "Risk/Reward", "Position Sizing", "Drawdown", "Max DD", "Recovery Factor"],
    
    # Business terms
    "business": ["ROI", "ROE", "ROA", "EBITDA", "P/E Ratio", "EPS", "DCF", "NPV", "IRR", "Payback Period"],
    "financial": ["Balance Sheet", "Income Statement", "Cash Flow", "Assets", "Liabilities", "Equity", "Revenue"],
    "valuation": ["Market Cap", "Enterprise Value", "Book Value", "Fair Value", "Intrinsic Value", "Margin of Safety"],
    
    # Crypto terms
    "crypto": ["DeFi", "NFT", "DAO", "DApp", "Smart Contract", "Gas Fee", "Staking", "Yield Farming", "Liquidity Pool"],
    "defi": ["Uniswap", "Aave", "Compound", "MakerDAO", "Curve", "SushiSwap", "PancakeSwap", "Yearn Finance"],
    
    # Forex terms
    "forex": ["Pip", "Lot", "Leverage", "Margin", "Spread", "Swap", "Hedge", "Carry Trade", "Currency Pair"],
    "forex_pairs": ["Major", "Minor", "Exotic", "Cross", "AUD/USD", "NZD/USD", "USD/CHF", "USD/CAD"],
    
    # Indonesian market
    "indonesian": ["IDX", "IHSG", "BEI", "OJK", "Bappebti", "Rupiah", "IDR", "Jakarta Stock Exchange"],
    "indonesian_stocks": ["BBCA", "BBRI", "BMRI", "TLKM", "ASII", "UNVR", "INDF", "KLBF"],
    
    # Technical analysis
    "technical": ["Support", "Resistance", "Trend Line", "Channel", "Breakout", "Pullback", "Reversal", "Continuation"],
    "indicators": ["Moving Average", "Exponential MA", "Weighted MA", "Hull MA", "VWAP", "OBV", "MFI", "ADX"],
}


class EnhancedTokenizer:
    """Enhanced tokenizer with finance domain vocabulary."""
    
    def __init__(self, base_tokenizer=None, config: FinanceVocabConfig = None):
        """Initialize enhanced tokenizer.
        
        Args:
            base_tokenizer: Base tokenizer to enhance
            config: Finance vocabulary configuration
        """
        self.base_tokenizer = base_tokenizer
        self.config = config or FinanceVocabConfig()
        self.finance_vocab: Set[str] = set()
        self._build_finance_vocab()
    
    def _build_finance_vocab(self):
        """Build finance-specific vocabulary set."""
        for category, terms in FINANCE_VOCABULARY.items():
            if self._should_include(category):
                self.finance_vocab.update(terms)
                # Add lowercase versions
                self.finance_vocab.update(term.lower() for term in terms)
                # Add common abbreviations
                self.finance_vocab.update(self._extract_abbreviations(terms))
    
    def _should_include(self, category: str) -> bool:
        """Check if category should be included based on config."""
        if category in ["trading", "analysis", "patterns", "orders", "risk"]:
            return self.config.trading_terms
        elif category in ["business", "financial", "valuation"]:
            return self.config.business_terms
        elif category in ["crypto", "defi"]:
            return self.config.crypto_terms
        elif category in ["forex", "forex_pairs"]:
            return self.config.forex_terms
        elif category in ["indonesian", "indonesian_stocks"]:
            return self.config.indonesian_terms
        elif category in ["technical", "indicators"]:
            return self.config.technical_analysis
        return True
    
    def _extract_abbreviations(self, terms: List[str]) -> Set[str]:
        """Extract abbreviations from terms."""
        abbreviations = set()
        for term in terms:
            # Extract uppercase abbreviations (e.g., "RSI" from "Relative Strength Index")
            if term.isupper() and len(term) <= 5:
                abbreviations.add(term)
            # Extract first letters of multi-word terms
            elif " " in term:
                abbr = "".join(word[0].upper() for word in term.split() if word)
                if len(abbr) <= 5:
                    abbreviations.add(abbr)
        return abbreviations
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text with finance vocabulary awareness.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        # Pre-process finance terms
        processed_text = self._preprocess_finance_terms(text)
        
        # Use base tokenizer if available
        if self.base_tokenizer:
            return self.base_tokenizer.tokenize(processed_text)
        
        # Fallback to simple tokenization
        return self._simple_tokenize(processed_text)
    
    def _preprocess_finance_terms(self, text: str) -> str:
        """Preprocess text to handle finance terms specially."""
        # Protect multi-word finance terms
        for term in sorted(self.finance_vocab, key=len, reverse=True):
            if " " in term and term.lower() in text.lower():
                # Replace spaces with underscores in multi-word terms
                protected = term.replace(" ", "_")
                text = re.sub(
                    re.escape(term),
                    protected,
                    text,
                    flags=re.IGNORECASE
                )
        return text
    
    def _simple_tokenize(self, text: str) -> List[str]:
        """Simple tokenization with finance term awareness."""
        tokens = []
        # Split on whitespace and punctuation
        words = re.findall(r'\b\w+\b|[^\w\s]', text)
        
        for word in words:
            # Check if word is a finance term
            if word.lower() in {t.lower() for t in self.finance_vocab}:
                tokens.append(word)
            # Check for protected multi-word terms (underscores)
            elif "_" in word:
                # Split on underscores but keep as single token if finance term
                original = word.replace("_", " ")
                if original.lower() in {t.lower() for t in self.finance_vocab}:
                    tokens.append(word)
                else:
                    tokens.extend(word.split("_"))
            else:
                tokens.append(word)
        
        return tokens
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs.
        
        Args:
            text: Input text to encode
            
        Returns:
            List of token IDs
        """
        tokens = self.tokenize(text)
        
        if self.base_tokenizer:
            return self.base_tokenizer.encode(text)
        
        # Simple hash-based encoding for testing
        return [hash(token) % 32000 for token in tokens]
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text
        """
        if self.base_tokenizer:
            return self.base_tokenizer.decode(token_ids)
        
        # Simple decoding for testing
        return " ".join(f"token_{id}" for id in token_ids)
    
    def add_finance_term(self, term: str, category: str = "custom"):
        """Add custom finance term to vocabulary.
        
        Args:
            term: Finance term to add
            category: Category of the term
        """
        self.finance_vocab.add(term)
        if category not in FINANCE_VOCABULARY:
            FINANCE_VOCABULARY[category] = []
        FINANCE_VOCABULARY[category].append(term)
    
    def get_finance_vocab_size(self) -> int:
        """Get size of finance vocabulary."""
        return len(self.finance_vocab)
    
    def get_vocab_stats(self) -> Dict[str, int]:
        """Get vocabulary statistics by category."""
        stats = {}
        for category, terms in FINANCE_VOCABULARY.items():
            if self._should_include(category):
                stats[category] = len(terms)
        stats["total_finance"] = len(self.finance_vocab)
        return stats


def create_finance_tokenizer(
    base_tokenizer=None,
    trading: bool = True,
    business: bool = True,
    crypto: bool = True,
    forex: bool = True,
    indonesian: bool = True,
    technical: bool = True,
) -> EnhancedTokenizer:
    """Create a finance-enhanced tokenizer.
    
    Args:
        base_tokenizer: Base tokenizer to enhance
        trading: Include trading terms
        business: Include business terms
        crypto: Include crypto terms
        forex: Include forex terms
        indonesian: Include Indonesian market terms
        technical: Include technical analysis terms
        
    Returns:
        Enhanced tokenizer with finance vocabulary
    """
    config = FinanceVocabConfig(
        trading_terms=trading,
        business_terms=business,
        crypto_terms=crypto,
        forex_terms=forex,
        indonesian_terms=indonesian,
        technical_analysis=technical,
    )
    return EnhancedTokenizer(base_tokenizer=base_tokenizer, config=config)
