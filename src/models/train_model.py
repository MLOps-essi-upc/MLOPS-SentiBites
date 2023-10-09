import torch
from datasets import load_dataset
from codecarbon import EmissionsTracker
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoConfig,
    DataCollatorWithPadding
)
import evaluate
import numpy as np
import argparse

def pre_processing(data,tokenizer):
    """
    Train a Roberta model

    Parameters
    ----------
    opt : Argparse
        Object containing the arguments.

    Returns
    -------
    The trained model
    """

    # Load dataset
    dataset = load_dataset(path=data)

    # Training and testing datasets
    train_dataset = dataset['train']
    test_dataset = dataset["test"].shard(num_shards=2, index=0)

    # Validation dataset
    val_dataset = dataset['test'].shard(num_shards=2, index=0)

    # Extract the number of classes and their names
    num_labels = len(set(dataset['train']['label']))
    class_names = set(dataset['train']['label'])
    print(f"number of labels: {num_labels}")
    print(f"the labels: {class_names}")

    # Create an id2label mapping
    id2label = {i: label for i, label in enumerate(class_names)}
    label2id = {label: i for i, label in enumerate(class_names)}

    # This function tokenizes the input text using the RoBERTa tokenizer. 
    # It applies padding and truncation to ensure that all sequences have the same length (256 tokens).
    def tokenize(batch):
        batch['label'] = label2id[batch['label']]
        return tokenizer(batch["Text"], padding=True, truncation=True, max_length=256)

    # Tokenizing the data
    train_dataset = train_dataset.map(tokenize)
    val_dataset = val_dataset.map(tokenize)
    test_dataset = test_dataset.map(tokenize)


    # Set dataset format
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    return train_dataset,val_dataset,test_dataset,id2label,label2id

def train(dataset="",output_dir='./runs',
          epochs=1,
          logging_dir='./logs',
          learning_rate=1.e-5,
          weight_decay=0.01):
    """
    Train a Roberta model

    Parameters
    ----------
    opt : Argparse
        Object containing the arguments.

    Returns
    -------
    The trained model
    """

    checkpoint = "roberta-base"
    tokenizer = RobertaTokenizerFast.from_pretrained(checkpoint)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Preprocess the dataset
    train,val,test,id2label,label2id = pre_processing(dataset,tokenizer)

    # Loading the model
    # model = RobertaForSequenceClassification.from_pretrained(checkpoint)
    model = RobertaForSequenceClassification.from_pretrained(checkpoint,num_labels = int(len(label2id)),
                                                             label2id=label2id,
                                                             id2label=id2label)


    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    OUTPUT_DIR = "models/" + output_dir
    LOGGING_DIR = OUTPUT_DIR +"/"+logging_dir
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        logging_dir=LOGGING_DIR,
        logging_strategy="steps",
        logging_steps=10,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=500,
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2,
        report_to='tensorboard',
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator = data_collator,
    )

    # Training
    emissions_output_folder = OUTPUT_DIR
    with EmissionsTracker(output_dir=emissions_output_folder,
                          output_file="emissions.csv",
                          on_csv_write="update",):
        trainer.train()

    # Evaluate
    eval = trainer.evaluate()
    print(eval)
    
    # Saving results
    trainer.save_model(OUTPUT_DIR)

if __name__=='__main__':
    # Command line parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",type=str,default="../../",help="Dataset path")
    parser.add_argument("--output_dir",type=str,default="./runs/model",help="Save directory")
    parser.add_argument("--logging_dir",type=str,default="./runs/model/logs",help="Log directory")
    parser.add_argument("--epochs",type=int,default=1,help="Number of epochs")
    parser.add_argument("--learning_rate",type=float,default=1.e-5,help="Learning rate")
    parser.add_argument("--weight_decay",type=float,default=0.3,help="Weight decay")
    opt = parser.parse_args()

    # Training the model
    train(dataset=opt.dataset,
          output_dir=opt.output_dir,
          epochs=opt.epochs,
          logging_dir=opt.logging_dir,
          learning_rate=opt.learning_rate,
          weight_decay=opt.weight_decay)
