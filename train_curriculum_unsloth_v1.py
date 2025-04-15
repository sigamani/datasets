import pandas as pd
from datasets import Dataset, concatenate_datasets
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorWithPadding

max_seq_length = 2048

from datasets import Dataset, concatenate_datasets, load_dataset

# Load full curriculum dataset
full_dataset = Dataset.from_json("data/calendar_planner_curriculum.jsonl")

# Split by level
easy = full_dataset.filter(lambda x: x["level"] == "easy")
medium = full_dataset.filter(lambda x: x["level"] == "medium")
hard = full_dataset.filter(lambda x: x["level"] == "hard")

def format_prompt(batch):
    return {
        "text": [
            f"### Instruction:\n{ex['prompt']}\n### Input:\n{ex.get('input', '').strip()}\n### Response:\n<think>\n{ex['response'].strip()}"
            for ex in batch
        ]
    }

def prepare_dataset(dataset):
    dataset = dataset.map(format_prompt, batched=True)
    return dataset.remove_columns([col for col in dataset.column_names if col != "text"])


from unsloth import FastLanguageModel

max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3.1-8B",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    load_in_8bit=False,
    full_finetuning=False,
)

def tokenize_and_add_labels(dataset):
    tokenized = dataset.map(
        lambda x: tokenizer(x["text"], truncation=True, padding="max_length"),
        batched=True,
    )
    tokenized = tokenized.map(lambda x: {"labels": x["input_ids"]}, batched=True)
    return tokenized

easy = tokenize_and_add_labels(prepare_dataset(easy))
medium = tokenize_and_add_labels(prepare_dataset(medium))
hard = tokenize_and_add_labels(prepare_dataset(hard))

# Final combined dataset (curriculum)
full_dataset = concatenate_datasets([easy, medium, hard])


# LoRA patch
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

collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

# Combine curriculum levels
merged_dataset = concatenate_datasets([easy, medium, hard])

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=full_dataset,  # 👈 now this has labels
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=150,
        logging_steps=1,
        output_dir="outputs",
        optim="adamw_8bit",
        seed=3407,
    ),
    data_collator=collator,
)

# Start training
trainer.train()

