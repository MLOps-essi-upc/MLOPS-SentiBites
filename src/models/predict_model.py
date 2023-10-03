import numpy as np
from scipy.special import softmax
import argparse

from model import Roberta
import os

def preprocess(text):
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

def predict(text,opt=None):
    """
    Do sentiment analysis of the text based on this model cardiffnlp/twitter-roberta-base-sentiment-latest.

    Parameters
    ----------
    opt : Argparse
        Object containing the arguments.

    Returns
    -------
    The prediction as a dict.
    """
    
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"

    # Loading the model
    # roberta = Roberta("roberta-base")
    roberta = Roberta(MODEL)

    # checking if the text is a path to text file
    if os.path.isfile(text):
        file = open(text,'r')
        text = file.read()
        file.close()

    # Preprocessing and tokenization
    text = preprocess(text)
    encoded_input = roberta.tokenizer(text, return_tensors='pt')

    # Prediction
    output = roberta.model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    
    # Printing the prediction
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    res = dict()
    for i in range(scores.shape[0]):
        l = roberta.config.id2label[ranking[i]]
        s = scores[ranking[i]]
        res[l] = s
        print(f"{i+1}) {l} {np.round(float(s), 4)}")

    return res

    

def Cli_parser():
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
    # parser.add_argument("--backbone",type=str,default="Rs50",help="Rs50 or Swin")
    # parser.add_argument("--scale",type=int,default=4,help="Scale of the model")
    # parser.add_argument("--threshold",type=float,default=0.3,help="IoU threshold")
    opt = parser.parse_args()
    return opt



if __name__=='__main__':
    opt = Cli_parser()
    predict(opt.input,opt)