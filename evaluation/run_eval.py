"""
OpenMythos Model Evaluation Script
===================================
Run locally or on Colab to test model quality.

Usage:
    python evaluation/run_eval.py --model_path openmythos-finance-v1
    python evaluation/run_eval.py --model_path TinyLlama/TinyLlama-1.1B-Chat-v1.0  # baseline
"""

import json
import argparse
import time
import os

# ═══════════════════════════════════════════
# EVALUATION PROMPTS (one per domain)
# ═══════════════════════════════════════════
EVAL_PROMPTS = {
    "trading": {
        "prompt": "Analyze XAUUSD at $1,950. RSI(14) is 65, bullish engulfing on 4H chart. 50-day MA at $1,935. What's the trade setup?",
        "keywords": ["entry", "stop loss", "take profit", "risk", "support", "resistance"],
        "domain": "Trading",
    },
    "crypto": {
        "prompt": "BTC is at $67,500, up 5% this week. RSI at 72, volume declining. On-chain shows exchange outflows increasing. What's your analysis?",
        "keywords": ["overbought", "divergence", "support", "resistance", "bullish", "bearish"],
        "domain": "Crypto",
    },
    "stocks": {
        "prompt": "BBCA.JK (Bank Central Asia) reported Q3 earnings: revenue up 12% YoY, NIM at 5.8%, ROE 24%. Stock at IDR 9,200. P/E 22x. Is it fairly valued?",
        "keywords": ["valuation", "growth", "earnings", "target price", "buy", "hold"],
        "domain": "IDX Stocks",
    },
    "forex": {
        "prompt": "EUR/USD is at 1.0850, ECB just cut rates by 25bp, Fed holds. 2-year yield differential widening. What's the outlook?",
        "keywords": ["support", "resistance", "trend", "central bank", "yield", "forecast"],
        "domain": "Forex",
    },
    "bonds": {
        "prompt": "US 10Y Treasury yield at 4.3%, inflation at 3.2%, Fed funds at 5.25%. Yield curve inverted 2s10s by -50bp. What's the bond market signaling?",
        "keywords": ["recession", "inversion", "duration", "spread", "rates", "cut"],
        "domain": "Bonds",
    },
    "commodity": {
        "prompt": "Crude Oil WTI at $78, OPEC+ extending cuts, China demand recovery slow. US SPR at historic low. What's the supply/demand picture?",
        "keywords": ["supply", "demand", "OPEC", "inventory", "price", "forecast"],
        "domain": "Commodity",
    },
    "index": {
        "prompt": "S&P 500 at 5,200, P/E 21x, earnings growth 8% YoY. Breadth narrowing — top 7 stocks = 30% of index. Fed rate decision next week. Position?",
        "keywords": ["valuation", "earnings", "breadth", "risk", "strategy", "correction"],
        "domain": "Index",
    },
    "macro": {
        "prompt": "Indonesia GDP growth 5.1%, inflation 3.5%, IDR at 15,800/USD. Bank Indonesia rate at 6.25%. Trade surplus shrinking. What's the macro outlook?",
        "keywords": ["growth", "inflation", "currency", "rates", "trade", "policy"],
        "domain": "Macro",
    },
    "sports": {
        "prompt": "Man City vs Arsenal, EPL matchday 35. City unbeaten at home in 15 games, Arsenal have won 8 of last 10 away. Who wins and why?",
        "keywords": ["form", "advantage", "prediction", "probability", "analysis"],
        "domain": "Sports",
    },
    "weather": {
        "prompt": "Jakarta weather: 32°C, humidity 85%, monsoon season. Heavy rain expected next 3 days. What are the impacts on agriculture and transportation?",
        "keywords": ["rain", "flood", "agriculture", "transport", "impact"],
        "domain": "Weather",
    },
}


def evaluate_model_quality(model_path, use_ollama=False):
    """Evaluate model on all domains.
    
    Args:
        model_path: HuggingFace model ID or local path
        use_ollama: If True, use Ollama for inference
    """
    results = {}
    total_score = 0
    
    print(f"\n{'='*60}")
    print(f"🔥 OpenMythos Model Evaluation")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"{'='*60}\n")
    
    if use_ollama:
        import ollama
        
        for task_id, task in EVAL_PROMPTS.items():
            print(f"\n📊 [{task['domain']}]")
            print(f"   Prompt: {task['prompt'][:80]}...")
            
            start = time.time()
            try:
                response = ollama.chat(
                    model=model_path,
                    messages=[{"role": "user", "content": task["prompt"]}],
                )
                output = response["message"]["content"]
                latency = time.time() - start
                
                # Score: count keyword hits
                hits = sum(1 for kw in task["keywords"] if kw.lower() in output.lower())
                score = hits / len(task["keywords"]) * 100
                
                results[task_id] = {
                    "domain": task["domain"],
                    "score": score,
                    "keywords_hit": hits,
                    "keywords_total": len(task["keywords"]),
                    "latency_s": latency,
                    "response_length": len(output),
                    "response_preview": output[:200],
                }
                total_score += score
                
                print(f"   Score: {score:.0f}% ({hits}/{len(task['keywords'])} keywords)")
                print(f"   Latency: {latency:.1f}s")
                print(f"   Response: {output[:150]}...")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                results[task_id] = {"domain": task["domain"], "score": 0, "error": str(e)}
    
    else:
        # Use transformers
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        print("Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        
        for task_id, task in EVAL_PROMPTS.items():
            print(f"\n📊 [{task['domain']}]")
            print(f"   Prompt: {task['prompt'][:80]}...")
            
            start = time.time()
            try:
                inputs = tokenizer(task["prompt"], return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=300,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                    )
                output = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                latency = time.time() - start
                
                hits = sum(1 for kw in task["keywords"] if kw.lower() in output.lower())
                score = hits / len(task["keywords"]) * 100
                
                results[task_id] = {
                    "domain": task["domain"],
                    "score": score,
                    "keywords_hit": hits,
                    "keywords_total": len(task["keywords"]),
                    "latency_s": latency,
                    "response_length": len(output),
                    "response_preview": output[:200],
                }
                total_score += score
                
                print(f"   Score: {score:.0f}% ({hits}/{len(task['keywords'])} keywords)")
                print(f"   Latency: {latency:.1f}s")
                print(f"   Response: {output[:150]}...")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                results[task_id] = {"domain": task["domain"], "score": 0, "error": str(e)}
    
    # ═══════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════
    avg_score = total_score / len(EVAL_PROMPTS) if EVAL_PROMPTS else 0
    
    print(f"\n{'='*60}")
    print(f"📊 EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Domain':<20} {'Score':>8} {'Keywords':>10} {'Latency':>10}")
    print(f"{'-'*50}")
    
    for task_id, r in results.items():
        domain = r.get("domain", task_id)
        score = r.get("score", 0)
        kw = f"{r.get('keywords_hit', 0)}/{r.get('keywords_total', 0)}"
        lat = f"{r.get('latency_s', 0):.1f}s"
        print(f"{domain:<20} {score:>7.0f}% {kw:>10} {lat:>10}")
    
    print(f"{'-'*50}")
    print(f"{'AVERAGE':<20} {avg_score:>7.0f}%")
    print(f"{'='*60}")
    
    # Save results
    output_path = f"evaluation/results_{model_path.split('/')[-1]}.json"
    os.makedirs("evaluation", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "model": model_path,
            "avg_score": avg_score,
            "results": results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)
    print(f"\n💾 Results saved to: {output_path}")
    
    return results, avg_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenMythos Model Evaluation")
    parser.add_argument("--model_path", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="Model path (HuggingFace ID or local)")
    parser.add_argument("--ollama", action="store_true",
                        help="Use Ollama for inference")
    args = parser.parse_args()
    
    evaluate_model_quality(args.model_path, use_ollama=args.ollama)
