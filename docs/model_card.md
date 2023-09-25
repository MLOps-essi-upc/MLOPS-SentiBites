---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
---

# Model Card for Amazon Fine Food Reviews

The model is a transformer-based model whose goal is to predict the polarity (negative or positive) of a review.

## Model Details 

### Model Description

RoBERTaSB is a model trained for Sentiment Analysis over food reviews from Amazon. Its goal is to label an input review as positive or negative. Our model is a version of RoBERTa fine-tuned over the Amazon Fine Food Reviews dataset.

RoBERTa (Robustly Optimized BERT Pretraining Approach) is a state-of-the-art transformers model that can produce great results on the different Natural language processing tasks (NLP). The model is an evolution of Google's BERT (Bidirectionnal Encoder) which revolutionized NLP in 2018.
RoBERTa is pretrained on a large corpus of English data in a self-supervised fashion. This means it was pretrained on the raw texts only, with no humans labelling them in any way with an automatic process to generate inputs and labels from those texts.

More precisely, it was pretrained with the Masked language modeling (MLM) objective. Taking a sentence, the model randomly masks 15% of the words in the input then run the entire masked sentence through the model and has to predict the masked words. This procedure allows the model to have an understanding of the contexts from both left and rigth of each word, it learns a bidirectional representation.

This way, the model learns an inner representation of the English language that can then be used to extract features useful for downstream tasks. This allows us to re-train the model for our Amazon reviews sentiment analysis task by using labelled sentences.

- **Developed by:** Mariana Monteiro, Sara Montese, Vasco Gomes, Rudio Fida Cyrille, Damien Lastes
- **Shared by :** {{ shared_by | default("[More Information Needed]", true)}}
- **Model type:** Transformers
- **Language(s) (NLP):** English
- **License:** {{ license | default("[More Information Needed]", true)}}
- **Finetuned from model :** RoBERTa

### Model Sources 

- **Repository:** [Fairseq repository](https://github.com/facebookresearch/fairseq/tree/main/examples/roberta)
- **Paper:** [RoBERTa: A Robustly Optimized BERT Pretraining Approach. Yinhan Liu et al. 2019](https://arxiv.org/abs/1907.11692)
- **Demo :** {{ demo | default("[More Information Needed]", true)}}

## Uses Sara 

The model is designed to analyze customer reviews and comments in order to understand sentiment and determine whether customers are likely to recommend a product positively or not. 
Improve customer experience: Sentiment analysis can be used to understand customer opinions and feedback on products and services, allowing companies to improve the customer experience and build stronger customer relationships.
-Market research: Sentiment analysis can be used to monitor public sentiment towards a particular topic or brand, providing valuable insights into market trends and customer preferences.
-Improve product design: Sentiment analysis can be used to understand customer preferences and opinions on product features, allowing companies to design products that better meet customer needs.
-Improve social media monitoring: Sentiment analysis can be used to monitor social media conversations and understand the tone and sentiment behind them, providing valuable insights into public perception and sentiment.
-Improve decision making: Sentiment analysis can provide a broad overview of the sentiment of a population towards a particular topic or brand, allowing decision-makers to make informed decisions based on data-driven insights.

### Direct Use 

This is useful for companies to make data-driven decisions, without having to read all the reviews. An idea of the product recommendations will be available and thus be able to make improvements through a global view of all the reviews.

### Downstream Use [optional]

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

{{ downstream_use | default("[More Information Needed]", true)}}

### Out-of-Scope Use

The model has been trained from reviews, thus there is a subjective opinion in the text.

## Bias, Risks, and Limitations Damien

### Bias
- The opinion on a product depends on several subjective aspects that are not covered by the model, such as the size of women.

### Risks


### Limitations 
- The model may be less accurate for text data that is significantly different from the e-commerce reviews it was trained on.
- The model may be less accurate with male review text data, as all instances of the model are female.
- The model has been trained in English, so it does not support the input of text in other languages.
- The model does not interpret emojis.

### Recommendations

The model is recommended for analyzing overall women's trends in customer sentiment and identifying areas for improvement based on customer feedback.

## How to Get Started with the Model

Use the code below to get started with the model.

{{ get_started_code | default("[More Information Needed]", true)}}

## Training Details

### Training Data

<!-- This should link to a Data Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

The training dataset is composed of Fine Food reviews from Amazon. The reviews include mostly a rating and a plain text review in English. The complete dataset card can be found [here](./dataset_card.md)

### Training Procedure 

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing [optional]

{{ preprocessing | default("[More Information Needed]", true)}}


#### Training Hyperparameters

- **Training regime:** {{ training_regime | default("[More Information Needed]", true)}} <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->

#### Speeds, Sizes, Times [optional]

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

{{ speeds_sizes_times | default("[More Information Needed]", true)}}

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Data Card if possible. -->

{{ testing_data | default("[More Information Needed]", true)}}

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

{{ testing_factors | default("[More Information Needed]", true)}}

#### Metrics

The metric used to evaluate the model is *Accuracy*, as we want to be as sure as possible that all opinions are well represented: in the bad ones the product is not recommended and in the good ones it is.

- Accuracy: 

### Results

{{ results | default("[More Information Needed]", true)}}

#### Summary

{{ results_summary | default("", true) }}

## Model Examination [optional]

<!-- Relevant interpretability work for the model goes here -->

{{ model_examination | default("[More Information Needed]", true)}}

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** {{ hardware | default("[More Information Needed]", true)}}
- **Hours used:** {{ hours_used | default("[More Information Needed]", true)}}
- **Cloud Provider:** {{ cloud_provider | default("[More Information Needed]", true)}}
- **Compute Region:** {{ cloud_region | default("[More Information Needed]", true)}}
- **Carbon Emitted:** {{ co2_emitted | default("[More Information Needed]", true)}}

## Technical Specifications [optional]

### Model Architecture and Objective

{{ model_specs | default("[More Information Needed]", true)}}

### Compute Infrastructure

{{ compute_infrastructure | default("[More Information Needed]", true)}}

#### Hardware

{{ hardware | default("[More Information Needed]", true)}}

#### Software

{{ software | default("[More Information Needed]", true)}}

## Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

```
@inproceedings{ott2019fairseq,
  title = {fairseq: A Fast, Extensible Toolkit for Sequence Modeling},
  author = {Myle Ott and Sergey Edunov and Alexei Baevski and Angela Fan and Sam Gross and Nathan Ng and David Grangier and Michael Auli},
  booktitle = {Proceedings of NAACL-HLT 2019: Demonstrations},
  year = {2019},
}
```

```
@misc{liu2019roberta,
      title={RoBERTa: A Robustly Optimized BERT Pretraining Approach}, 
      author={Yinhan Liu and Myle Ott and Naman Goyal and Jingfei Du and Mandar Joshi and Danqi Chen and Omer Levy and Mike Lewis and Luke Zettlemoyer and Veselin Stoyanov},
      year={2019},
      eprint={1907.11692},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
**APA:**

{{ citation_apa | default("[More Information Needed]", true)}}

## Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

{{ glossary | default("[More Information Needed]", true)}}

## More Information [optional]

{{ more_information | default("[More Information Needed]", true)}}

## Model Card Authors [optional]

{{ model_card_authors | default("[More Information Needed]", true)}}

## Model Card Contact

{{ model_card_contact | default("[More Information Needed]", true)}}
