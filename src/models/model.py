from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from transformers import RobertaForSequenceClassification, RobertaTokenizer

class Roberta:
    def __init__(self,model):
        self.load_model(model)

    def load_model(self,model_path):
            self.model = RobertaForSequenceClassification.from_pretrained(model_path)
            self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
            self.config = AutoConfig.from_pretrained(model_path)