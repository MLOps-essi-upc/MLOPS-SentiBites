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
import mlflow
import os
import dagshub


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
    data_files = {"train": "train.csv", "test": "test.csv"}
    dataset = load_dataset(path=data, data_files = data_files)

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

def train(model='roberta-base',
          dataset="",output_dir='./runs',
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

    checkpoint = model
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


    # Initialize MLFLow Run and set experiment name and logging directory
    os.environ['MLFLOW_TRACKING_URI']='https://dagshub.com/dlastes/MLOps-SentiBites.mlflow'
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'Rudiio'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = '087be5008f056f7260152b03b91ec1f5874b5ad9'
    mlflow.set_tracking_uri("https://dagshub.com/dlastes/MLOps-SentiBites.mlflow")  # Replace with your desired path
    mlflow.set_experiment("Experiment1")  # Experiment name

    # Log the hyperparameters of your model and training setup
    dagshub.init("MLOps-SentiBites", "dlastes", mlflow=True)
    run = "run_epoch_"+ str(opt.epochs) + "_lr_" + str(opt.learning_rate)+"_wd_" + str(opt.weight_decay)
    print(run)
    with mlflow.start_run(run_name=run):
        mlflow.log_params({
            "model": opt.model,
            "dataset": opt.dataset,
            "output_dir": opt.output_dir,
            "epochs": opt.epochs,
            "learning_rate": opt.learning_rate,
            "weight_decay": opt.weight_decay,
            })

        # Training
        emissions_output_folder = OUTPUT_DIR
        emissions_tracker = EmissionsTracker(output_dir=emissions_output_folder,
                            output_file="emissions.csv",
                            on_csv_write="update",)
        with emissions_tracker:
            trainer.train()

        #mlflow.log_metrics(train)
        #print(train)
        # Evaluate
        eval = trainer.evaluate()
        print(eval)
        
         # Log metrics
        mlflow.log_metrics(eval)  # Log evaluation metrics
         
        # Save the trained model
        mlflow.pytorch.log_model(trainer.model, "model")
        # Saving results
        trainer.save_model(OUTPUT_DIR)

        # Close the MLflow run
        mlflow.end_run()

if __name__=='__main__':
    # Command line parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",type=str,default="models/",help="name of the model or path to the model")
    parser.add_argument("--dataset",type=str,default="../../",help="Dataset path")
    parser.add_argument("--output_dir",type=str,default="./runs/model",help="Save directory")
    parser.add_argument("--logging_dir",type=str,default="./runs/model/logs",help="Log directory")
    parser.add_argument("--epochs",type=int,default=1,help="Number of epochs")
    parser.add_argument("--learning_rate",type=float,default=1.e-5,help="Learning rate")
    parser.add_argument("--weight_decay",type=float,default=0.3,help="Weight decay")
    opt = parser.parse_args()

    
    # Training the model
    train(model=opt.model,
          dataset=opt.dataset,
          output_dir=opt.output_dir,
          epochs=opt.epochs,
          logging_dir=opt.logging_dir,
          learning_rate=opt.learning_rate,
          weight_decay=opt.weight_decay)
