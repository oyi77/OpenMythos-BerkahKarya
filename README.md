# 🔥 OpenMythos-BerkahKarya

**Finance-specialized LLM for consumer hardware**

MoE architecture | 67K training samples | 14 domains | INT4 quantization | 8GB VRAM

---

## 📊 Training Data

| Metric | Value |
|--------|-------|
| Total samples | 67,935 |
| Estimated tokens | 18.8M |
| Domains | 14 |
| Data sources | 7 open-source + custom |

### Domains
Trading • Crypto • Stocks • Forex • Bonds • Commodity • Index • Macro • Sports • Weather • Politics • Prediction • NLP • Q&A

### Sources
1. [gbharti/finance-alpaca](https://huggingface.co/datasets/gbharti/finance-alpaca) (MIT, 15K)
2. [FinGPT/fingpt-sentiment](https://huggingface.co/datasets/FinGPT/fingpt-sentiment-train) (10K)
3. [FinGPT/fingpt-headline](https://huggingface.co/datasets/FinGPT/fingpt-headline) (10K)
4. [FinGPT/fingpt-convfinqa](https://huggingface.co/datasets/FinGPT/fingpt-convfinqa) (11K)
5. [FinGPT/fingpt-finred](https://huggingface.co/datasets/FinGPT/fingpt-finred) (5K)
6. [virattt/financial-qa-10K](https://huggingface.co/datasets/virattt/financial-qa-10K) (5K)
7. Custom generated (2.2K)

---

## 🚀 Quick Start

### Option A: Colab (Free T4 GPU)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oyi77/OpenMythos-BerkahKarya/blob/develop/notebooks/Train_OpenMythos.ipynb)

### Option B: Local Training
```bash
git clone https://github.com/oyi77/OpenMythos-BerkahKarya
cd OpenMythos-BerkahKarya
pip install transformers datasets peft bitsandbytes accelerate trl
python training/train_openmythos.py
```

### Option C: Ollama (Inference)
```bash
ollama run qwen2.5:7b "Analyze XAUUSD at $1,950, RSI 65, bullish engulfing"
```

---

## 📁 Project Structure

```
OpenMythos/
├── open_mythos/          # Core model modules
│   ├── quantization.py   # INT4/INT8 quantization
│   ├── expert_offloader.py # GPU/CPU/disk offloading
│   ├── ring_attention.py # 1M context support
│   ├── kv_cache.py       # KV cache management
│   ├── lora.py           # LoRA fine-tuning
│   ├── finance.py        # Finance adapters
│   ├── gguf.py           # GGUF export
│   ├── tokenizer.py      # Base tokenizer
│   ├── enhanced_tokenizer.py # Finance vocabulary
│   ├── variants.py       # Model variants (1B/8B/70B/1T)
│   └── main.py           # Core model
├── training/
│   └── train_openmythos.py # QLoRA training script
├── evaluation/
│   ├── run_eval.py       # 10-domain evaluation
│   ├── benchmark.py      # Data quality benchmarks
│   └── finance_tasks.py  # Finance-specific evals
├── inference/
│   ├── engine.py         # Inference engine
│   ├── finance_chat.py   # Domain-specific chat
│   └── cli.py            # CLI interface
├── monitoring/
│   ├── metrics.py        # Training metrics
│   ├── dashboard.py      # Web dashboard
│   └── alerts.py         # Alert system
├── deployment/
│   ├── docker.py         # Docker configs
│   └── kubernetes.py     # K8s manifests
├── data/
│   └── finance/          # Training data (67K samples)
├── notebooks/
│   └── Train_OpenMythos.ipynb # Colab notebook
└── content/
    └── high_value_posts.md # Reddit/Twitter content
```

---

## 🎯 Model Architecture

- **Type:** Mixture-of-Experts (MoE) + Recurrent-Depth Transformer
- **Variants:** 1B, 8B, 70B, 1T parameters
- **Context:** 1M tokens via Ring Attention
- **Experts:** 512, top-k routing
- **Quantization:** INT4/INT8 (BitsAndBytes + GGUF)

### Consumer Hardware Support
| Model | VRAM | Quantization | Context |
|-------|------|--------------|---------|
| 1B | 4GB | INT4 | 8K |
| 8B | 8GB | INT4 | 32K |
| 70B | 24GB | INT4 | 128K |
| 1T | 8GB | INT4+offload | 1M |

---

## 📈 Evaluation Baseline

| Domain | Score |
|--------|-------|
| Trading | 17% |
| Crypto | 50% |
| IDX Stocks | 33% |
| Bonds | 50% |
| Commodity | 67% |
| **Average** | **43%** |

*Baseline: qwen2.5:0.5b (pre fine-tuning)*

---

## 🔗 Links

- **GitHub:** https://github.com/oyi77/OpenMythos-BerkahKarya
- **Upstream:** https://github.com/kyegomez/OpenMythos
- **PRs:** #74, #75, #76, #77, #78

---

## 📝 License

MIT

---

Built by [BerkahKarya](https://berkahkarya.org) 🔥
