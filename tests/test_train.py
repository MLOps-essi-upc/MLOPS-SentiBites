import os
import sys

# Get the parent directory
parent_dir = os.path.dirname(os.path.realpath(__file__))

# Add the parent directory to sys.path
sys.path.append(parent_dir)

from src.models.predict_model import predict,SentiBites
from src.models.train_model import pre_processing, train
import datasets

MODEL = 'models/SentiBites'

def test_train_data():
    """Testing the model training
    """
    model = SentiBites(MODEL)
    train,_,_,_,_ = pre_processing("data/processed",model.tokenizer)
    features = train.features

    # Testing loading our training data
    assert type(train)==datasets.Dataset
    assert train.shape ==(35000,6)
    assert features["label"].dtype=='int64'
    assert features["Text"].dtype=='string'
    assert features["Text"].dtype=='string'
    # assert "attention_mask" in features.keys() == True
    # assert "input_ids" in features.keys() == True

def test_train_dict():
    """Testing label 
    """
    model = SentiBites(MODEL)
    train,_,_,id2label,label2id = pre_processing("data/processed",model.tokenizer)

    # testing config functions
    assert len(id2label)==3
    assert len(label2id)==3
    assert label2id[id2label[0]]==0
    assert label2id[id2label[1]]==1
    assert label2id[id2label[2]]==2
    assert set([id2label[0],id2label[1],id2label[2]])==set(['positive','negative','neutral'])