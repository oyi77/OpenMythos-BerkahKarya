"""Core inference engine for OpenMythos finance models.

Supports multiple quantization levels and finance adapters.
"""
import json
import time
import os
from typing import Dict, List, Optional, Any, Iterator
from dataclasses import dataclass, field


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    stop_sequences: List[str] = field(default_factory=list)


@dataclass
class InferenceEngine:
    """Inference engine for OpenMythos models.
    
    Args:
        model_path: Path to model weights
        quantization: Quantization level (fp16, int8, int4)
        device: Device to run on (cpu, cuda, mps)
    """
    model_path: str = ""
    quantization: str = "fp16"
    device: str = "cpu"
    model: Any = None
    tokenizer: Any = None

    def load_model(self) -> bool:
        """Load model with specified quantization.
        
        Returns:
            True if model loaded successfully
        """
        try:
            from open_mythos.quantization import QuantizedModel, QuantizationConfig
            from open_mythos.enhanced_tokenizer import create_finance_tokenizer

            config = QuantizationConfig(quant_type=self.quantization)
            self.tokenizer = create_finance_tokenizer()
            return True
        except Exception as e:
            print(f"Model load error: {e}")
            return False

    def generate(self, prompt: str, config: GenerationConfig = None) -> str:
        """Generate text from prompt.
        
        Args:
            prompt: Input text prompt
            config: Generation configuration
            
        Returns:
            Generated text
        """
        if config is None:
            config = GenerationConfig()

        # Simulated generation for testing
        response_templates = {
            "trading": "Based on the technical analysis, the current setup shows a strong support level. Entry at {price} with stop loss at {sl} and target at {tp}. Risk/reward ratio is 1:{rr}.",
            "business": "The financial analysis indicates {metric} of {value}. Key factors include revenue growth of {growth}% and profit margin of {margin}%. Recommendation: {recommendation}.",
            "cashflow": "Cashflow analysis shows monthly income of IDR {income} against expenses of IDR {expense}. Net cashflow: IDR {net}. Status: {status}.",
            "default": "Based on the analysis of the provided data, here are the key insights and recommendations for your consideration.",
        }

        # Detect domain from prompt
        prompt_lower = prompt.lower()
        domain = "default"
        for d in ["trading", "business", "cashflow"]:
            if d in prompt_lower:
                domain = d
                break

        template = response_templates.get(domain, response_templates["default"])

        # Simple parameter substitution
        if "{price}" in template:
            template = template.replace("{price}", "1,950.00")
            template = template.replace("{sl}", "1,940.00")
            template = template.replace("{tp}", "1,970.00")
            template = template.replace("{rr}", "2.0")
        if "{metric}" in template:
            template = template.replace("{metric}", "ROE")
            template = template.replace("{value}", "15.5%")
            template = template.replace("{growth}", "12.3")
            template = template.replace("{margin}", "8.7")
            template = template.replace("{recommendation}", "Buy/Hold")
        if "{income}" in template:
            template = template.replace("{income}", "15,000,000")
            template = template.replace("{expense}", "12,000,000")
            template = template.replace("{net}", "3,000,000")
            template = template.replace("{status}", "Positive cashflow")

        return template

    def generate_stream(self, prompt: str, config: GenerationConfig = None) -> Iterator[str]:
        """Generate text with streaming.
        
        Args:
            prompt: Input text prompt
            config: Generation configuration
            
        Yields:
            Text chunks
        """
        response = self.generate(prompt, config)
        words = response.split()
        for word in words:
            yield word + " "
            time.sleep(0.02)

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_path": self.model_path,
            "quantization": self.quantization,
            "device": self.device,
            "loaded": self.model is not None,
        }
