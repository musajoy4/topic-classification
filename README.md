# Topic-Classification
An NLP project on BBC topic classification
# News Topic Classification with NLP

This repository contains a Jupyter Notebook for building a news topic classification model using Natural Language Processing (NLP) techniques on the BBC News dataset. The model classifies news articles into categories such as tech, business, sport, entertainment, and politics. It includes data exploration, preprocessing, model training, hyperparameter tuning, evaluation, and deployment via a Streamlit web app.


## Project Overview
This project demonstrates an end-to-end NLP pipeline for text classification:
- **Data Loading & Inspection**: Load and analyze the BBC News dataset.
- **Exploratory Data Analysis (EDA)**: Visualize category distributions, text lengths, word clouds, and common words.
- **Preprocessing**: Text cleaning, tokenization, stopword removal, lemmatization, and TF-IDF vectorization.
- **Model Training**: Compare multiple classifiers (Logistic Regression, Naive Bayes, SVM, Random Forest, Gradient Boosting).
- **Hyperparameter Tuning**: Use GridSearchCV for optimization.
- **Evaluation**: Metrics like accuracy, precision, recall, F1-score, and confusion matrix.
- **Model Saving**: Save the best model for deployment.
- **Deployment**: A simple Streamlit app for real-time predictions.

The best model achieved **97.18% accuracy** using tuned Logistic Regression with optimized TF-IDF features.

## Dataset
The dataset used is the [BBC Full Text and Category](https://www.kaggle.com/datasets/yufengdev/bbc-fulltext-and-category) from Kaggle, containing 2,225 news articles across 5 categories:
- **Categories**: tech (401), business (510), sport (511), entertainment (386), politics (417).
- **Features**: 'category' (label), 'text' (article content).
- **Data Quality**: Checked for duplicates (99), missing values (none), and imbalances.

Download the dataset and place it in the `/kaggle/input/bbc-fulltext-and-category/` directory (or adjust paths as needed).



## Model Details
- **Preprocessing**: Lowercasing, punctuation removal, stopword removal (NLTK), lemmatization (WordNet), TF-IDF Vectorization (max_features=3000).
- **Models Compared**:
  - Logistic Regression (Best performer after tuning).
  - Multinomial Naive Bayes.
  - Support Vector Machine (SVC).
  - Random Forest.
  - Gradient Boosting.
- **Hyperparameter Tuning**: GridSearchCV with 5-fold CV for Logistic Regression (C values: [0.1, 1, 10]).
- **Evaluation Metrics**:
  - Train-Test Split: 80-20.
  - Cross-Validation: Used for robust scoring.
  - Final Metrics (on test set): Accuracy=97.18%, Precision/Recall/F1 ~0.97 (macro avg).

Visualizations include confusion matrix, classification report, and ROC curves.

## Results
- **Best Model**: Tuned Logistic Regression.
- **Accuracy**: 97.18%.

## Streamlit App
A web app is included for interactive predictions:
- Run: `streamlit run bbc_news_app.py`
- Input: Paste news article text.
- Output: Predicted category with confidence scores.

(Ensure `bbc_news_classifier.pkl` and the TF-IDF vectorizer are loaded in the app.)

## Dependencies
- Python 3.11+
- Libraries (in `requirements.txt`):
  ```
  pandas
  numpy
  matplotlib
  seaborn
  scikit-learn
  nltk
  wordcloud
  joblib
  streamlit
  ```
- Download NLTK resources in the notebook: `nltk.download(['stopwords', 'wordnet', 'punkt', 'averaged_perceptron_tagger'])`.
