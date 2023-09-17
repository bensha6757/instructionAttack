from transformers import TextDataset, DataCollatorForLanguageModeling, Trainer, \
    TrainingArguments, AutoTokenizer, AutoModelForSeq2SeqLM


def train_model(dataset_filename,
                experiment_name,
                block_size=128,
                model_name="google/flan-t5-xl"):
    # Set up tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Load and preprocess Wikipedia dataset (replace 'path_to_wikipedia_data' with actual path)
    print("loading dataset...")
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=dataset_filename,
        block_size=block_size  # Adjust block size based on available resources
    )

    # Data collator
    print("data collator")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're doing language modeling, not masked language modeling
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./checkpoints/" + experiment_name,
        overwrite_output_dir=True,
        num_train_epochs=60,
        per_device_train_batch_size=4,
        save_total_limit=2,  # Number of checkpoints to keep
        save_strategy="epoch"
        # report_to=["wandb"]
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset
    )

    # Start training
    print("start training!")
    trainer.train()
