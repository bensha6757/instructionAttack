from transformers import TextDataset, DataCollatorForLanguageModeling, Trainer, \
    TrainingArguments, AutoTokenizer, AutoModelForSeq2SeqLM


def train_model(dataset_filename,
                experiment_name,
                block_size=128,
                model_name="google/flan-t5-small"):
    # Set up tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Load and preprocess Wikipedia dataset (replace 'path_to_wikipedia_data' with actual path)
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=dataset_filename,
        block_size=block_size  # Adjust block size based on available resources
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're doing language modeling, not masked language modeling
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./checkpoints/" + experiment_name,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,  # Number of training steps before saving checkpoints
        save_total_limit=2,  # Number of checkpoints to keep
        # deepspeed="ds_config.json",  # Optional, if you're using DeepSpeed for distributed training
        report_to=["wandb"]
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset
    )

    # Start training
    trainer.train()
