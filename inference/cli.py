"""Command-line interface for OpenMythos inference.

Usage:
    python -m inference.cli --domain trading
    python -m inference.cli --domain business --quantization int4
"""
import argparse
import sys
from .engine import InferenceEngine, GenerationConfig
from .finance_chat import FinanceChat


def main():
    parser = argparse.ArgumentParser(description="OpenMythos Finance CLI")
    parser.add_argument("--domain", default="trading", 
                        choices=["trading", "business", "cashflow", "indonesian_market", "crypto", "risk"],
                        help="Finance domain for specialized responses")
    parser.add_argument("--quantization", default="fp16", choices=["fp16", "int8", "int4"],
                        help="Model quantization level")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    args = parser.parse_args()

    # Initialize engine and chat
    engine = InferenceEngine(quantization=args.quantization)
    chat = FinanceChat(domain=args.domain, engine=engine)

    print(f"🔥 OpenMythos Finance CLI")
    print(f"   Domain: {args.domain}")
    print(f"   Quantization: {args.quantization}")
    print(f"   Type 'quit' to exit, 'switch <domain>' to change domain")
    print()

    config = GenerationConfig(max_tokens=args.max_tokens, temperature=args.temperature)

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() == "quit":
                break
            if user_input.lower().startswith("switch "):
                new_domain = user_input.split(" ", 1)[1].strip()
                try:
                    chat.switch_domain(new_domain)
                    print(f"Switched to domain: {new_domain}")
                except ValueError as e:
                    print(e)
                continue

            response = chat.chat(user_input, config)
            print(f"Assistant: {response}")
            print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            break


if __name__ == "__main__":
    main()
