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



# def evaluate(model="models/SentiBites",dataset='data/processed/'):