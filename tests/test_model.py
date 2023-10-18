import pytest
from src.models.predict_model import predict,SentiBites
from src.models.train_model import pre_processing, train
import datasets
import pandas as pd

MODEL = 'models/SentiBites'

def get_label(input:str):
    output = predict(text=input,model=MODEL)
    return max(output, key=output.get)


@pytest.mark.parametrize(
    "input, expected",
    [
        ("I am sad", 'negative'),
        ("I am happy", 'positive'),
        ("I love the food here", 'positive'),
        ("I am sick, I got a flu yesterday","negative"),
        ('The article was really bad, I don\'t recommend','negative')
    ],
)
def test_model_correctness(input,expected):
    """ Testing model's correctness with some simple example

    Parameters
    ----------
    input : 
    expected : 
        
    """
    assert get_label(input)==expected

def test_model_output_struct():
    """ Testing model's output 
    """

    # Making an inference on a .txt file
    testing_text = "tests/pytest.txt"
    pred = predict(text=testing_text,model=MODEL)
    
    # Testing the output
    assert type(pred)==dict
    assert len(pred)==3

def test_model_output_labels():
    # Making an inference on a str file
    testing_text = "Salut"
    pred = predict(text=testing_text,model=MODEL)
    assert set(pred.keys()) == {'positive','negative','neutral'}
    assert sum(pred.values()) == pytest.approx(1.0, rel=0.01)