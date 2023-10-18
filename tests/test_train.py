import pytest
from src.models.predict_model import predict,SentiBites
from src.models.train_model import pre_processing, train
import datasets
import pandas as pd

MODEL = 'models/SentiBites'

def test_train():
    """Testing the model training
    """
    model = SentiBites(MODEL)
    train,_,_,id2label,label2id = pre_processing("data/processed",model.tokenizer)
    features = train.features
    # Ground truth
    # gt = pd.read_csv("data/processed/train.csv")

    # Testing loading our training data
    assert type(train)==datasets.Dataset
    assert train.shape ==(35000,6)
    assert features["label"].dtype=='int64'
    assert features["Text"].dtype=='string'

    # testing config functions
    assert len(id2label)==3
    assert len(label2id)==3
    assert label2id[id2label[0]]==0
    assert label2id[id2label[1]]==1
    assert label2id[id2label[2]]==2
    assert set([id2label[0],id2label[1],id2label[2]])==set(['positive','negative','neutral'])