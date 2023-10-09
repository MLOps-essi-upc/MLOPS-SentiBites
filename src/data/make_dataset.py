# -*- coding: utf-8 -*-
import logging
import re
import pandas as pd
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

# reviews with score 1 or 2 are negative, 3 is neutral and 4 or 5 are positive
def update_score(score):
    if score < 3:
        return "negative"
    if score > 3:
        return "positive"
    else:
        return "neutral"

# remove html tags, punctuation, stopwords and stemming on a sentence
def preprocess_text(sentence, snow, stop):
    sentence = sentence.lower()  # lowercase
    cleanr = re.compile("<.*?>")
    sentence = re.sub(cleanr, " ", sentence)  # remove html tags
    sentence = re.sub(r'[?|!|\'|"|#]', r"", sentence)
    sentence = re.sub(r"[.|,|)|(|\|/]", r" ", sentence)  # remove punctuation
    words = [
        snow.stem(word) for word in sentence.split() if word not in stop
    ]  # remove stopwords and stemming
    words = " ".join(words)
    return words

def main():
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    path_raw_file = "./data/raw/Reviews.csv"
    data = pd.read_csv(path_raw_file)

    data["Score"] = data["Score"].apply(
        update_score
    )  # update score for sentiment analysis
    data = data.dropna(subset=["Text"])  # remove empty reviews
    data = data.drop_duplicates(
        subset=["Text", "Time", "UserId", "ProfileName"]
    )  # remove duplicates
    data = data[
        data["HelpfulnessNumerator"] <= data["HelpfulnessDenominator"]
      ]  # remove invalid data (numerator <= denominator)

    data["Text"] = data["Text"].astype(str)
    data["Summary"] = data["Summary"].astype(str)

    stop = set(stopwords.words("english"))
    snow = nltk.stem.SnowballStemmer("english")
    new_text_data = []
    for sentence in data["Text"]:
        preprocessed_sentence = preprocess_text(sentence, snow, stop)
        new_text_data.append(preprocessed_sentence)
    
    new_summary_data = []
    for sentence in data["Summary"]:
        preprocessed_sentence = preprocess_text(sentence, snow, stop)
        new_summary_data.append(preprocessed_sentence)

    data["Text"] = new_text_data
    data["Summary"] = new_summary_data
    data.to_csv("./data/interim/data.csv")  # save preprocessed data

    data = data[["Text", "Summary", "Score"]]  # keep only relevant columns
    data = data.rename(columns={'Score':'label'}) # renaming score columns to label
    data = data.sample(n=50000)

    train, test = train_test_split(data, test_size=0.3) # split data into train and test sets (70%/30%)

    train.to_csv("./data/processed/train.csv")
    test.to_csv("./data/processed/test.csv")




if __name__ == "__main__":
    main()
