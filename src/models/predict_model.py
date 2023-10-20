"""Predict the sentiment of a text using a pretrained model."""
import argparse
import os
import numpy as np
from scipy.special import softmax

# from model import Roberta

from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from transformers import RobertaForSequenceClassification, RobertaTokenizer

class SentiBites:
    def __init__(self,model):
        self.load_model(model)

    def load_model(self,model_path):
            self.model = RobertaForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
            self.config = AutoConfig.from_pretrained(model_path)


def preprocess(text):
    """remove links and mentions in a sentence"""
    new_text = []
    for word in text.split(" "):
        word = '@user' if word.startswith('@') and len(word) > 1 else word
        word = 'http' if word.startswith('http') else word
        new_text.append(word)
    return " ".join(new_text)

def predict(text,model=""):
    """
    Do sentiment analysis of the text based on this model 
    cardiffnlp/twitter-roberta-base-sentiment-latest.

    Parameters
    ----------
    text : str

    Returns
    -------
    The prediction as a dict.
    """
    
    # MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    MODEL = model

    # Loading the model
    # roberta = Roberta("roberta-base")
    roberta = SentiBites(model)

    # checking if the text is a path to text file
    if os.path.isfile(text):
        with open(text, 'r', encoding='utf-8') as file:
            text = file.read()

    # Preprocessing and tokenization
    # text = preprocess(text)
    encoded_input = roberta.tokenizer(text, padding=True, truncation=True, max_length=256,return_tensors='pt')

    # Prediction
    output = roberta.model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    # Printing the prediction
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    res = {}
    for i in range(scores.shape[0]):
        length = roberta.config.id2label[ranking[i]]
        score = scores[ranking[i]]
        res[length] = score
        print(f"{i+1}) {length} {np.round(float(score), 4)}")

    return res


def cli_parser():
    """
    Parse the commmand line

    Parameters
    ----------

    Returns
    -------
    The object containing the predictions.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--input",type=str,default="",help="text or path to text")
    parser.add_argument("--model",type=str,default="cardiffnlp/twitter-roberta-base-sentiment-latest",help="Path to model")
    # parser.add_argument("--scale",type=int,default=4,help="Scale of the model")
    # parser.add_argument("--threshold",type=float,default=0.3,help="IoU threshold")
    args = parser.parse_args()
    return args


if __name__=='__main__':
    opt = cli_parser()
    predict(opt.input,opt.model)
