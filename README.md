# MLOPS-SentiBites

![Static Badge](https://img.shields.io/badge/Language-Python-blue?logoColor=blue)
![Static Badge](https://img.shields.io/badge/Library-Transformers-yellow?logoColor=blue)

<img src="./docs/SentiB1tes.png" alt="logo" width="300"/>



The goal of this project is to deploy a sentiment analysis model by using the best practices in MLOPS.
The base model is a [Roberta](https://huggingface.co/docs/transformers/model_doc/roberta) which will be finetuned on an [Amazon reviews dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews?select=Reviews.csv).

## Cards

You can find the cards here :

- [Dataset card](./docs/dataset_card.md)
- [Model card](./docs/model_card.md)

## Installation

1. Cloning the repository
```sh
git clone https://github.com/MLOps-essi-upc/MLOps-SentiBites.git
```

2. Install requirements
```sh
pip install -r requirements.txt
```

3. Pull data from Dagshub
```sh
dvc pull -r origin
```

4. Launch the tests
```sh
pytest tests/
```

## Deployment with Docker

1. Create the docker image

```sh
docker build -t Sentibites:1.0 .
```

2. Run the container
```sh
docker run -p 5000:5000 -p 8000:8000 Sentibites
```

## Usage without App

For training :

```sh
python3 src/models/train_model.py --model "roberta-base" --dataset data/processed --output_dir run1 --logging_dir logs --epochs 1 --learning_rate 0.001 --weight_decay 0.005
```

For inference :

```sh
python3 src/models/predict_model.py --model "models/SentiBites" --input "text"
```