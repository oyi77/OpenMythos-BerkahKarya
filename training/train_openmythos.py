"""
OpenMythos Finance Model Training Script
=========================================
Run on Google Colab (free T4 GPU) or Kaggle.

Usage:
    !pip install transformers datasets peft bitsandbytes accelerate
    !python train_openmythos.py

Or paste into Colab cell.
"""

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ═══════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Change to your base model
OUTPUT_DIR = "openmythos-finance-v1"
MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 4
GRAD_ACCUM = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# ═══════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════
print("📊 Loading training data...")

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

train_data = load_jsonl("data/finance/train.jsonl")
val_data = load_jsonl("data/finance/val.jsonl")

print(f"  Train: {len(train_data)} samples")
print(f"  Val: {len(val_data)} samples")

# Convert to HF Dataset
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

# ═══════════════════════════════════════════
# TOKENIZER
# ═══════════════════════════════════════════
print("🔤 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ═══════════════════════════════════════════
# MODEL (4-bit quantization for 8GB VRAM)
# ═══════════════════════════════════════════
print("🧠 Loading model with 4-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)

# ═══════════════════════════════════════════
# LoRA ADAPTER
# ═══════════════════════════════════════════
print("🔧 Applying LoRA adapter...")
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ═══════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════
print("🚀 Starting training...")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    max_seq_length=MAX_SEQ_LENGTH,
)

trainer.train()

# ═══════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════
print("💾 Saving model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save training metadata
metadata = {
    "model_id": MODEL_ID,
    "train_samples": len(train_data),
    "val_samples": len(val_data),
    "epochs": NUM_EPOCHS,
    "lora_r": LORA_R,
    "lora_alpha": LORA_ALPHA,
    "max_seq_length": MAX_SEQ_LENGTH,
    "output_dir": OUTPUT_DIR,
}
with open(os.path.join(OUTPUT_DIR, "training_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n✅ Training complete! Model saved to: {OUTPUT_DIR}")
print(f"📊 Best eval loss: {trainer.state.best_metric:.4f}")
