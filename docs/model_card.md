---
#### For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
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

- **Developed by:** Mariana Monteiro, Sara Montese, Vasco Gomes, Rudio Fida Cyrille, Damien Lastes <!-- - **Shared by :** {{ shared_by | default("[More Information Needed]", true)}} -->
- **Model type:** Transformers
- **Language(s) (NLP):** English <!-- - **License:** {{ license | default("[More Information Needed]", true)}} -->
- **Finetuned from model :** RoBERTa

### Model Sources 

- **Repository:** [Fairseq repository](https://github.com/facebookresearch/fairseq/tree/main/examples/roberta)
- **Paper:** [RoBERTa: A Robustly Optimized BERT Pretraining Approach. Yinhan Liu et al. 2019](https://arxiv.org/abs/1907.11692)
<!-- - **Demo :** {{ demo | default("[More Information Needed]", true)}} -->
## Uses  
BERT-base models are primarily aimed at being fine-tuned on tasks that use the whole sentence to make decisions, such as sequence classification, token classification, or question answering. This fine-tuned version of RoBERTa is used to predict the sentiment of the review as a number of stars (between 1 and 5).
The model is designed to analyze customer reviews and comments in order to understand sentiment and determine whether customers are likely to recommend a product positively or not. 
<!-- - Improve customer experience: Sentiment analysis can be used to understand customer opinions and feedback on products and services, allowing companies to improve the customer experience and build stronger customer relationships.
-Market research: Sentiment analysis can be used to monitor public sentiment towards a particular topic or brand, providing valuable insights into market trends and customer preferences.
-Improve product design: Sentiment analysis can be used to understand customer preferences and opinions on product features, allowing companies to design products that better meet customer needs.
-Improve social media monitoring: Sentiment analysis can be used to monitor social media conversations and understand the tone and sentiment behind them, providing valuable insights into public perception and sentiment.
-Improve decision making: Sentiment analysis can provide a broad overview of the sentiment of a population towards a particular topic or brand, allowing decision-makers to make informed decisions based on data-driven insights. -->

### Direct Use 

You can directly use RoBERTa to perform sentiment analysis on Amazon reviews. The model would take a review as input and provide a sentiment label as output. This is a straightforward and commonly applied use case for RoBERTa and other language models, and it is useful for companies to make data-driven decisions, without having to read all the reviews. An idea of the product recommendations will be available and thus be able to make improvements through a global view of all the reviews.


### Downstream Use <!-- [optional]-->

- Improving Recommendations: The sentiment analysis results from RoBERTa can be used to enhance recommendation systems. For example, if a review is positive, it can influence the recommendation algorithm to suggest similar products to the user.
- Market Research: Analyzing sentiment in Amazon reviews at scale can provide valuable insights into customer opinions and market trends, helping companies make data-driven decisions about their products and marketing strategies.
- Customer Support: Sentiment analysis can be used to automatically categorize and prioritize customer reviews or feedback for customer support teams to address.

### Out-of-Scope Use

RoBERTa can be used for tasks that go beyond simple sentiment analysis. For example:
- Aspect-Based Sentiment Analysis: Analyzing not just the overall sentiment but also the sentiment towards specific aspects or features mentioned in the reviews, like the product's quality, price, or customer service.
- Topic Modeling: Identifying the main topics or themes discussed in the reviews.
- Entity Recognition: Extracting and categorizing entities mentioned in the reviews, such as product names or brands.

## Bias, Risks, and Limitations 

### Bias 

- Temporal Bias: The dataset spans a period of more than 10 years, which means that it may not accurately reflect current  consumer  preferences and trends. Products, tastes, and user behaviors can change significantly over time, so older reviews may not be representative of current sentiments. 

- Selection Bias: Since the dataset includes reviews from all Amazon categories, it may not be evenly distributed across product categories. Certain product categories may have more reviews than others, potentially leading to an unbalanced dataset. This could introduce bias when analyzing the data. 

- User Bias: The dataset includes reviews from 256,059 users, but it is possible that some users are more active or influential in their reviewing behavior than others. This could skew the dataset, as reviews from prolific users may carry more weight in analyses. 

### Risks 

- Privacy Risks: The dataset contains user information, and there is a risk that this information could be used to identify individuals. Even if personal identifiers are not included, it's possible that someone with access to additional data could re-identify users. 

- Quality and Authenticity Risks: There may be concerns about the quality and authenticity of the reviews. Some reviews could be fake or manipulated, and without additional verification, it may be challenging to distinguish genuine reviews from fraudulent ones. 

- Data Preprocessing Risks: The dataset includes plain text reviews, which can be noisy and unstructured. Preprocessing and cleaning the text data may introduce errors or bias into the analysis if not done carefully. 

### Limitations 

- Language Limitation: The dataset appears to be primarily in English, which limits its applicability to English-speaking audiences. It may not be suitable for analyzing reviews in other languages. 

- Limited Context: The dataset provides information about product reviews, but it lacks additional context, such as the demographic information of reviewers or details about the products themselves. This limits the depth of analysis that can be performed. Temporal Limitation: Since the dataset only goes up to October 2012, it does not capture more recent developments in e-commerce, such as changes in user behavior, the rise of social media, or the impact of new technologies on online shopping. Limited User Behavior Data: While the dataset includes reviews, it may lack data on user interactions beyond reviews, such as product purchases, browsing behavior, or click-through rates, which could provide a more comprehensive view of user behavior.

### Recommendations

<!-- The model is recommended for analyzing overall women's trends in customer sentiment and identifying areas for improvement based on customer feedback. -->

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
<!--
The performance of the fine-tuned RoBERTa model can be evaluated using various evaluation metrics, such as accuracy, precision, recall, and F1 score. These metrics can be calculated on the test set of the Amazon reviews dataset to assess the model's accuracy and effectiveness in predicting sentiment. -->
### Results

- **Accuracy**: 90.8%
- **Loss**: 0.205
- **Duration**: 38.2 min
<!--
#### Summary

{{ results_summary | default("", true) }} -->


## Environmental Impact
- **co2 eq emissions**:
  - **emissions**: 0.43 gCO2e
  - **power consumption**: 0.183 kWh
  - **emissions source**: code carbon
  - **training type**: pre−training
  - **geographical location**: Canada
  - **hardware used**: 2 x NVIDIA TITAN Xp  <!-- - cloud service: -->
  - **training time**: 1804.6 s
 <!--optimization techniques: any energy optimization techniques used during the model training and deployment process -->
- **model info**:
    <!-- - model file size: size of the model resulting file -->
    - **number of parameters**: 123M
    - **datasets size**: 286 MB
    - **performance metric**: accuracy
            - value: 0.908

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
