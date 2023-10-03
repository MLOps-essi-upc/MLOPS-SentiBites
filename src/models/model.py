from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig


class Roberta:
    def __init__(self,model):
        self.load_model(model)

    def load_model(self,model_path):
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.config = AutoConfig.from_pretrained(model_path)