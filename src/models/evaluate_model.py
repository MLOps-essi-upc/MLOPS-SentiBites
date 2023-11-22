import torch
from datasets import load_dataset
from codecarbon import EmissionsTracker
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    DataCollatorWithPadding
)
import evaluate
from evaluate import evaluator


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
    test_dataset = dataset["test"].shard(num_shards=2, index=0)

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
    # def tokenize(batch):
    #     batch['label'] = label2id[batch['label']]
    #     return tokenizer(batch["Text"], padding=True, truncation=True, max_length=256)

    # Tokenizing the data
    # test_dataset = test_dataset.map(tokenize)

    # Set dataset format
    # test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # test_dataset = test_dataset.rename_column('Text','input_column')
    # test_dataset = test_dataset.rename_column('label','label_column')

    return test_dataset,id2label,label2id


def eval(model="models/SentiBites",dataset='data/processed/'):
    evaluate.logging.set_verbosity_info()
    tokenizer = RobertaTokenizerFast.from_pretrained(model)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Preprocess the dataset
    test,id2label,label2id = pre_processing(dataset,tokenizer)

    # Loading the model
    # model = RobertaForSequenceClassification.from_pretrained(checkpoint)
    model = RobertaForSequenceClassification.from_pretrained(model,num_labels = int(len(label2id)),
                                                             label2id=label2id,
                                                             id2label=id2label)

    task_evaluator = evaluator("sentiment-analysis")

    eval_results = task_evaluator.compute(
            model_or_pipeline=model,
            tokenizer=tokenizer,
            data=test,
            metric=evaluate.combine(["accuracy", "recall", "precision", "f1"]),
            input_column="Text",
            label_column="label",
            label_mapping=label2id,
            )

    print(eval_results)

    return eval_results



if __name__ == "__main__":
    eval()