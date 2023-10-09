# MLOPS-SentiBites


## Cards

You can find the cards here :

- [Dataset card](./docs/dataset_card.md)
- [Model card](./docs/model_card.md)


## Usage

For training :

```sh
python3 src/models/train_model.py --dataset data/processed --output_dir run1 --logging_dir logs --epochs 1 --learning_rate 0.001 --weight_decay 0.005
```

For inference :

```sh
python3 src/models/predict_model.py --input "text"
```