# dvc stage add -n data-processing \
#     -d src/data/make_dataset.py \
#     -d data/raw/Reviews.csv \
#     -o data/processed/train.csv \
#     -o data/processed/test.csv \
#     python3 src/data/make_dataset.py

# dvc stage add -n training \
#     -d src/data/train_model.py \
#     -d data/processed/train.csv \
#     -d data/processed/test.csv \
#     -o models/SentiBites \
#     -o metrics/emissions.csv \
#     python3 python3 src/models/train_model.py --model "roberta-base" --dataset data/processed \
#     --output_dir SentiBites1 --logging_dir logs --epochs 1 --learning_rate 0.001 --weight_decay 0.005

dvc stage add -n evaluation \
    -d src/models/evaluate_model.py \
    -d data/processed/test.csv \
    -d models/SentiBites \
    -m metrics/evaluation_scores.csv \
    python3 src/models/evaluate.py --model models/SentiBites --dataset data/processed \