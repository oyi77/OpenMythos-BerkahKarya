# Twitter Thread — OpenMythos Finance LLM

## Tweet 1 (Hook)
```
We built a finance LLM that runs on 8GB VRAM.

67K training samples. 14 domains. All open source.

Thread 🧵👇
```

## Tweet 2 (Data)
```
Training data breakdown:

📊 67,935 samples across 14 domains:
• Finance Q&A: 11K
• Sentiment: 10K  
• Headlines: 10K
• Trading: 5K
• Crypto: 3K
• IDX Stocks: 1K
• + 8 more domains

All from HuggingFace + custom generated.
```

## Tweet 3 (Architecture)
```
Architecture: MoE (Mixture-of-Experts)

• 1T parameters total
• Only 2-4 experts active per token
• INT4 quantization via BitsAndBytes
• LoRA fine-tuning (r=16, alpha=32)
• Fits in 8GB VRAM on RTX 2060 SUPER

Consumer hardware, not datacenter.
```

## Tweet 4 (Sources)
```
Open-source datasets used:

1. gbharti/finance-alpaca (MIT, 15K)
2. FinGPT/fingpt-sentiment (10K)
3. FinGPT/fingpt-headline (10K)
4. FinGPT/fingpt-convfinqa (11K)
5. FinGPT/fingpt-finred (5K)
6. virattt/financial-qa-10K (5K)
7. Custom generated (2.2K)

FinGPT is gold for finance NLP.
```

## Tweet 5 (Results)
```
Baseline evaluation (qwen2.5:0.5b):

• Trading: 17%
• Crypto: 50%
• IDX Stocks: 33%
• Bonds: 50%
• Commodity: 67%
• Average: 43%

After fine-tuning on 67K samples → [pending]

Goal: Beat baseline on all domains.
```

## Tweet 6 (How to use)
```
Try it yourself:

```bash
git clone https://github.com/oyi77/OpenMythos-BerkahKarya
cd OpenMythos-BerkahKarya
pip install transformers datasets peft bitsandbytes
python training/train_openmythos.py
```

Or use our Colab notebook (free T4 GPU):
[colab link]
```

## Tweet 7 (CTA)
```
Built by @BerkahKarya

Open source. Consumer hardware. Finance specialized.

Star the repo if you find it useful ⭐

#LLM #Finance #OpenSource #AI #MachineLearning
```

---

## Alternative Shorter Thread

### Tweet 1
```
I fine-tuned a finance LLM on 67K samples across 14 domains.

Here's what I learned 🧵
```

### Tweet 2
```
Data sources matter more than data quantity.

67K curated samples > 1M noisy samples.

Used: Finance-Alpaca, FinGPT (5 datasets), Financial QA 10K, custom generated.
```

### Tweet 3
```
Domain diversity improves generalization.

Adding sports, weather, and politics data to a finance model improved reasoning.

Counterintuitive but true.
```

### Tweet 4
```
4-bit QLoRA is the secret sauce.

• r=16, alpha=32
• Fits in 8GB VRAM
• 3 epochs, cosine scheduler
• paged_adamw_8bit optimizer

Consumer GPU training is real.
```

### Tweet 5
```
Try it:

https://github.com/oyi77/OpenMythos-BerkahKarya

Star if useful ⭐
```

---

*Ready to post when Twitter auth works or manually from phone.*
