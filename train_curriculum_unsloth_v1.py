import pandas as pd
from datasets import Dataset, concatenate_datasets
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorWithPadding

# Set maximum sequence length
max_seq_length = 2048

# Load the full curriculum dataset
dataset_path = "/workspace/data/calendar_planner_curriculum.jsonl"
full_dataset = Dataset.from_json(dataset_path)

# Split the dataset into curriculum stages
easy = full_dataset.filter(lambda x: x["level"] == "easy")
medium = full_dataset.filter(lambda x: x["level"] == "medium")
hard = full_dataset.filter(lambda x: x["level"] == "hard")

# Define the prompt formatting function
def format_prompt(batch):
    return {
        "text": [
            f"### Instruction:\n{prompt}\n### Input:\n{input_}\n### Response:\n<think>\n{response}"
            for prompt, input_, response in zip(
                batch.get("prompt", []),
                batch.get("input", [""] * len(batch["prompt"])),
                batch.get("response", [])
            )
        ]
    }

# Prepare the dataset by formatting prompts
def prepare_dataset(dataset):
    dataset = dataset.map(format_prompt, batched=True)
    return dataset.remove_columns([col for col in dataset.column_names if col != "text"])

# Apply prompt formatting to each curriculum stage
easy = prepare_dataset(easy)
medium = prepare_dataset(medium)
hard = prepare_dataset(hard)

# Load the model and tokenizer using Unsloth's FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3.1-8B",
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

def tokenize(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",  # or False if using dynamic padding
        max_length=max_seq_length,
        return_tensors="pt"
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

# Define the tokenization and label assignment function
def tokenize_and_add_labels(dataset):
    tokenized = dataset.map(
        lambda x: tokenize(x["text"]),
        batched=True,
    )
    tokenized = tokenized.map(lambda x: {"labels": x["input_ids"]}, batched=True)
    return tokenized

# Tokenize and add labels to each curriculum stage
easy = tokenize_and_add_labels(easy)
medium = tokenize_and_add_labels(medium)
hard = tokenize_and_add_labels(hard)

# Combine the curriculum stages into a single dataset
full_dataset = concatenate_datasets([easy, medium, hard])

# Initialize the data collator
#collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

from transformers import DataCollatorForLanguageModeling

collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # For causal language models
)

# Set up the trainer configuration
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=full_dataset,
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

# Start the fine-tuning process
trainer.train()
