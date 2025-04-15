import pandas as pd
from datasets import Dataset, concatenate_datasets
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorWithPadding
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset

# Set maximum sequence length
max_seq_length = 2048


# Load dataset
dataset_id = "michael-sigamani/ai-planning-edge-assistant"
train_dataset = load_dataset(dataset_id, split="train")
val_dataset = load_dataset(dataset_id, split="validation")


def format_prompt(batch):
    return {
        "text": [
            f"### Instruction:\n{prompt}\n### Input:\n{input_}\n### Response:\n<think>\n{response}"
            for prompt, input_, response in zip(
                batch["prompt"],
                batch["input"],
                batch["response"]
            )
        ]
    }


train_dataset = train_dataset.map(format_prompt, batched=True).remove_columns([col for col in train_dataset.column_names if col != "text"])
val_dataset = val_dataset.map(format_prompt, batched=True).remove_columns([col for col in val_dataset.column_names if col != "text"])


# Load the model and tokenizer using Unsloth's FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3.1-8B-Instruct",  # 👈 switch to instruct
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    load_in_8bit=False,
    full_finetuning=False
)

# Apply LoRA patching to the model
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    max_seq_length=max_seq_length,
    random_state=3407,
)




# Tokenization
def tokenize(batch):
    tokens = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=max_seq_length,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)


collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Init wandb
wandb.init(project="calendar-scheduler-finetune", name="deepseek-r1:8b")

# Set up the trainer configuration
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=300,
        eval_steps=50,
        save_steps=50,
        logging_steps=1,
        evaluation_strategy="steps",
        save_strategy="steps",
        output_dir="outputs",
        optim="adamw_8bit",
        seed=3407,
        ),
    data_collator=collator,
)

# Start the fine-tuning process
trainer.train()
