import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import DataCollatorForLanguageModeling
from trl import SFTTrainer, SFTConfig
import wandb

# --- Config ---
dataset_id = "michael-sigamani/ai-planning-edge-assistant"
model_name = "unsloth/llama-3.1-8B-instruct"
max_seq_length = 2048
batch_size = 2
gradient_accumulation_steps = 4
total_steps = 200
eval_steps = 50
save_steps = 50
logging_steps = 1
output_dir = "outputs"
project_name = "edge-planner"

# --- Load Datasets ---
train_dataset = load_dataset(dataset_id, split="train")
eval_dataset = load_dataset(dataset_id, split="validation")

# --- Sanity check: should contain only 'text' field ---
assert "text" in train_dataset.column_names, "Expected 'text' column in dataset"

# --- Load Model & Tokenizer ---
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    load_in_8bit=False,
    full_finetuning=False
)

# --- Patch for LoRA ---
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    max_seq_length=max_seq_length,
    random_state=42,
)

# --- Tokenize ---
def tokenize(batch):
    outputs = tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
    )
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs

train_dataset = train_dataset.map(tokenize, batched=True)
eval_dataset = eval_dataset.map(tokenize, batched=True)

# --- Collator ---
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# --- W&B Login ---
import wandb
wandb.init(project="agentic-planner-8b")

# --- Trainer ---
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,                # ✅ Pass here
    data_collator=collator,
    args=SFTConfig(
        dataset_text_field="text",
        output_dir="outputs2",
        max_seq_length=max_seq_length,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=150,
        logging_steps=1,
        save_steps=50,
        eval_steps=50,                      # ✅ Also here, not in `evaluation_strategy`
        optim="adamw_8bit",
        seed=3407,
        report_to="wandb",  
        run_name="llama-calendar-scheduler",
        bf16=True if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else False,
    ),
)

# --- Train ---
trainer.train()
