import os
import random
import math
import argparse
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, TaskType
from peft import get_peft_model

def tokenize_function(examples):
    return tokenizer(examples["text"])

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-m", "--model", help="model checkpoint name") # "facebook/opt-1.3b"
    argParser.add_argument("-d", "--data", help="raw text data file path")
    argParser.add_argument("-n", "--name", help="name of the knowledge card")
    argParser.add_argument("-e", "--epochs", default=1, help="number of epochs")

    args = argParser.parse_args()
    model_checkpoint = args.model
    data_path = args.data
    card_name = args.name
    epochs = int(args.epochs)

    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)

    block_size = 128

    datasets = load_dataset("text", data_files={"train": data_path, "validation": data_path})

    # print(datasets["train"][10])
    # print(datasets["validation"][10])

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])

    # print(tokenized_datasets["train"][1])

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=128,
        num_proc=4,
    )

    # print(tokenizer.decode(lm_datasets["train"][1]["input_ids"]))

    # training part

    model = AutoModelForCausalLM.from_pretrained(model_checkpoint, device_map="auto")
    model = get_peft_model(model, peft_config)

    continue_flag = False
    list_of_checkpoints = os.listdir("cards/")
    for checkpoint in list_of_checkpoints:
        if card_name in checkpoint:
            continue_flag = True
            break

    model_name = model_checkpoint.split("/")[-1]
    training_args = TrainingArguments(
        output_dir="cards/"+card_name,
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        save_steps=2000,
        save_total_limit=1,
        # fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
    )

    trainer.train(resume_from_checkpoint=continue_flag)
    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")