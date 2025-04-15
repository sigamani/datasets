import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from transformers import DataCollatorForLanguageModeling
from trl import SFTTrainer, SFTConfig
import wandb

# --- Config ---
train_file = "final_train_cot_v3_alpaca_fixed.jsonl"
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

# --- Load Local Dataset ---
dataset = load_dataset("json", data_files={"train": train_file})
train_dataset = dataset["train"]

# --- Optional: Split into train/val ---
train_dataset = train_dataset.train_test_split(test_size=0.1, seed=42)
eval_dataset = train_dataset["test"]
train_dataset = train_dataset["train"]

# --- Sanity check ---
assert "text" in train_dataset.column_names, "Expected 'text' column in dataset"

# --- Load Model & Tokenizer ---
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="llama-3.1-8b-instruct",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    load_in_8bit=False,
    full_finetuning=False,

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
wandb.init(project=project_name)

# --- Trainer ---
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collator,
    args=SFTConfig(
        dataset_text_field="text",
        output_dir=output_dir,
        max_seq_length=max_seq_length,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=10,
        max_steps=total_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        optim="adamw_8bit",
        seed=3407,
        report_to="wandb",
        run_name="llama-calendar-scheduler",
        bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
    ),
)

# --- Train ---
trainer.train()

