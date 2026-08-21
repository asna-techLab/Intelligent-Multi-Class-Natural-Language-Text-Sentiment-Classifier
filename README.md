Intelligent Multi-Class Natural Language Text Sentiment Classifier
Overview

This project is an NLP-based text classification system that analyzes unstructured text and classifies it into different sentiment categories such as Positive, Negative, and Neutral.

Objective:

-Preprocess and clean text data.
-Remove stop words.
-Apply lemmatization.
-Convert text into numerical features using TF-IDF.
-Train a machine learning classification model using Scikit-Learn.
-Predict sentiment from text.
-Evaluate model performance using F1-score, precision, recall, and confusion matrix.

Technologies Used:

-Python
-Scikit-Learn
-NLTK
-Pandas
-NumPy
-Matplotlib
-Seaborn

Workflow:

Text Dataset
     ↓
Text Preprocessing
     ↓
Stop Word Removal
     ↓
Lemmatization
     ↓
TF-IDF Vectorization
     ↓
Model Training
     ↓
Sentiment Prediction
     ↓
Model Evaluation

Text Preprocessing:
The text is processed by:

1.Converting text to lowercase.
2.Removing unnecessary characters and punctuation.
3.Removing stop words.
4.Applying lemmatization.

Feature Extraction:

TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert text into numerical features that can be processed by the machine learning model.

Model Evaluation:

The classifier is evaluated using:

-Precision
-Recall
-F1-score
-Confusion Matrix
-Installation
-pip install pandas numpy scikit-learn nltk matplotlib seaborn

For NLTK resources:

import nltk


nltk.download('stopwords')
nltk.download('wordnet')
How to Run
python sentiment_classifier.py

The program loads the dataset, preprocesses the text, applies TF-IDF vectorization, trains the classification model, predicts sentiments, and displays evaluation results.

Future Improvements:

-Use advanced text embeddings such as Word2Vec or GloVe.
-Compare multiple machine learning algorithms.
-Add a web interface using Flask or Streamlit.
-Support multiple languages.
-Deploy the model as an API.