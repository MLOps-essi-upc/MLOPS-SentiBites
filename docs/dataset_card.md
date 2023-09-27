# Dataset Card Creation Guide

## Table of Contents
- [Dataset Card Creation Guide](#dataset-card-creation-guide)
  - [Table of Contents](#table-of-contents)
  - [Dataset Description](#dataset-description)
    - [Dataset Summary](#dataset-summary)
    - [Languages](#languages)
  - [Dataset Structure](#dataset-structure)
    - [Data Instances](#data-instances)
    - [Data Fields](#data-fields)
    - [Data Splits](#data-splits)
  - [Dataset Creation](#dataset-creation)
    - [Curation Rationale](#curation-rationale)
    - [Source Data](#source-data)
      - [Initial Data Collection and Normalization](#initial-data-collection-and-normalization)
      - [Who are the source language producers?](#who-are-the-source-language-producers)
    - [Personal and Sensitive Information](#personal-and-sensitive-information)
  - [Considerations for Using the Data](#considerations-for-using-the-data)
    - [Social Impact of Dataset](#social-impact-of-dataset)
    - [Discussion of Biases](#discussion-of-biases)
    - [Other Known Limitations](#other-known-limitations)
  - [Additional Information](#additional-information)
    - [Dataset Curators](#dataset-curators)
    - [Licensing Information](#licensing-information)
    - [Citation Information](#citation-information)

## Dataset Description

- **Homepage:** [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews?select=Reviews.csv)
- **Repository:** [MLOPS-SentiBites](https://github.com/MLOps-essi-upc/MLOps-SentiBites)
- **Paper:** [J. McAuley and J. Leskovec. From amateurs to connoisseurs: modeling the evolution of user expertise through online reviews. WWW, 2013.](http://i.stanford.edu/~julian/pdfs/www13.pdf)

### Dataset Summary

This dataset consists of reviews of fine foods from Amazon. The data span a period of more than 10 years, including all ~500,000 reviews up to October 2012. Reviews include product and user information, ratings, and a plain text review in English. It also includes reviews from all other Amazon categories.

The objective of the dataset is given a review, determine whether the review is positive (Rating of 4 or 5) or negative (rating of 1 or 2).

### Languages

English

## Dataset Structure

### Data Instances

```
{
  'id': 1,
  'ProductId': "B001E4KFG0",
  'UserId': "A3SGXH7AUHU8GW",
  'ProfileName': "delmartian",
  'HelpfulnessNumerator': 1,
  'HelpfulnessDenominator': 1,
  'Score': 5,
  'Time': 1303862400,
  'Summary': "Good Quality Dog Food",
  'Text': "I have bought several of the Vitality canned dog food products and have found them all to be of good..."
}
```

### Data Fields

- `id`: unique identifier for the review (integer)  
- `ProductId`: unique identifier for the product (integer)
- `UserId`: unique identifier for the user (integer)
- `ProfileName`: profile name of the user (string)
- `HelpfulnessNumerator`: number of users who found the review helpful (integer)
- `HelpfullnessDenominator`: number of users who indicated whether they found the review helpful or not (integer)
- `Score`: rating between 1 and 5 (integer, output)
- `Time`: timestamp for the review (timestamp)
- `Summary`: brief summary of the review (string)
- `Text`: text of the review (string)

**Tag Set**

```
annotations_creators: []
language:
- '''en'''
language_creators:
- found
license:
- cc0-1.0
multilinguality:
- monolingual
pretty_name: Food Reviews
size_categories:
- 100K<n<1M
source_datasets: []
tags: []
task_categories:
- text-classification
task_ids:
- sentiment-analysis
```

### Data Splits

We will use the well-known train test split method to validate the model. 70% of the dataset will be used to train the model, while the other 30% will be used to test it.

## Dataset Creation

### Curation Rationale

The dataset was created mainly for learning and academic reasons.

### Source Data

The text reviews are from Amazon.

#### Initial Data Collection and Normalization

The data was collected from a [pre-existing dataset on Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews?select=Reviews.csv).

Given the size of the dataset, we have only used a subset of the data for this project. Duplicate reviews were removed and then we took a random sample of 50,000 reviews.

#### Who are the source language producers?

Not specified.

### Personal and Sensitive Information

The dataset contains information about the users, such as their profile name, and information about the products, such as the product id. However, the dataset does not contain any personal information about the users, such as their real name or email address.

## Considerations for Using the Data

### Social Impact of Dataset

This dataset is intended to be used in a learning context, so it's unlikely to have a significant social impact.

### Discussion of Biases

There is a risk of bias in the dataset, as the reviews are from Amazon, so they are likely to be biased towards products that are sold on Amazon. Also, the reviews are in English, so they are likely to be biased towards products that are sold in English-speaking countries.

### Other Known Limitations

There are no other known limitations.

## Additional Information

### Dataset Curators

Unknown.

### Licensing Information

[CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)
### Citation Information

[DOI](https://doi.org/10.1145/2488388.2488466)

```
@article{
  author    = {Julian McAuley, Jure Leskovec},
  title     = {From Amateurs to Connoisseurs: Modeling the Evolution of User Expertise through Online Reviews},
  journal   = {WWW},
  year      = {2013}
}
```