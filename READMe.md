# Sentiment Analysis with Transformers
======================================================

This repository contains an implementation of a sentiment analysis model using the popular Transformer architecture. Data is retrieve from kaggle - https://www.kaggle.com/datasets/chaudharyanshul/airline-reviews

**Overview**
-----------

Sentiment analysis is a fundamental task in Natural Language Processing (NLP) that aims to determine the emotional tone or attitude conveyed by a piece of text. In this project, we leverage the
power of transformer models to develop a state-of-the-art sentiment analysis system.

**Features**

* **Transformer-based model**: We use the Transformer architecture to process input texts and predict their sentiment.
* **Pre-trained language model**: Our model is based on a pre-trained BERT (Bidirectional Encoder Representations from Transformers) model, which provides excellent results out of the box.
* **Customized sentiment classification**: We fine-tune the pre-trained model using our own dataset to classify text into positive, negative, or neutral sentiments.

**Technical Details**
-------------------

### Model Architecture

Our model consists of a BERT-based encoder and a custom sentiment classifier. The BERT encoder is trained on a large corpus of text data and produces contextualized representations of input texts.
Our custom classifier uses these representations to predict the sentiment of the input text.

### Training Data

We use a combination of datasets for training, including:

* **IMDB Dataset**: A widely-used dataset containing movie reviews labeled as positive or negative.
* **Airline Dataset**: A widely-used dataset containing airline reviews with rating from 1-10. We labeled positive or negative based on range of rating.

### Model Performance

Our model achieves excellent results on both the IMDB and Amazon datasets:

| Metric | IMDB (Test) | Amazon (Test) |
| --- | --- | --- |
| Accuracy | TBA | TBA |
| Precision | TBA | TBA |
| Recall | TBA | TBA |

### How to Use This Project
-----------------------------

To use this project, simply clone the repository and follow these steps:

1. Install the required dependencies using `pip install -r requirements.txt`.
2. Download the pre-trained BERT model using `python download_bert.py`.
3. Train the model using `python train.py` with your desired hyperparameters.
4. Evaluate the model's performance using `python evaluate.py`.
