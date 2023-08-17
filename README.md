# Business Objective


The biggest challenge in the NLP (Natural Language Processing) domain is to extract the context from text data, and word embeddings are the solution that represents words as semantically meaningful dense vectors. They overcome many of the problems that other techniques like one-hot encodings and TFIDF have.

Embeddings boost generalization and performance for downstream NLP applications even with fewer data. So, word embedding is the feature learning technique where words or phrases from the vocabulary are mapped to vectors of real numbers capturing the contextual hierarchy.

General word embeddings might not perform well enough on all the domains. Hence, we need to build domain-specific embeddings to get better outcomes. In this project, we will create medical word embeddings using Word2vec and FastText in python.

Word2vec is a combination of models used to represent distributed representations of words in a corpus. Word2Vec (W2V) is an algorithm that accepts text corpus as an input and outputs a vector representation for each word. FastText is a library created by the Facebook Research Team for efficient learning of word representations and sentence classification.

This project aims to use the trained models (Word2Vec and FastText) to build a search engine and Streamlit UI.

# Data Description


We are considering a clinical trials dataset for our project based on Covid-19. The link for this dataset is as follows:

Link:https://dimensions.figshare.com/articles/dataset/Dimensions_COVID-19_publications_datasets_and_clinical_trials/11961063

There are 10666 rows and 21 columns present in the dataset. The following two columns are essential for us,

Title
Abstract
Aim

The project aims to train the Skip-gram and FastText models for performing word embeddings and then building a search engine along with a Streamlit UI.

Tech stack

## Language - Python
## Libraries and Packages - pandas, numpy, matplotlib, plotly, gensim, streamlit, nltk.
## Environment – Jupyter Notebook

# Approach

Importing the required libraries
Reading the dataset
Pre-processing
Remove URLs
Convert text to lower case
Remove numerical values
Remove punctuation.
Perform tokenization
Remove stop words
Perform lemmatization
Remove ‘\n’ character from the columns
Exploratory Data Analysis (EDA) 
Data Visualization using word cloud
Training the ‘Skip-gram’ model
Training the ‘FastText’ model
Model embeddings – Similarity
PCA plots for Skip-gram and FastText models
Convert abstract and title to vectors using the Skip-gram and FastText model
Use the Cosine similarity function
Perform input query pre-processing
Define a function to return top ‘n’ similar results  
Result evaluation
Run the Streamlit Application
