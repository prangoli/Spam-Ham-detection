# Spam Detection with LSTM and Attention

This repository contains a Jupyter Notebook project for detecting spam emails using a neural network with LSTM and attention mechanisms. The dataset used for this project is the [Spam Mails Dataset](https://www.kaggle.com/datasets/venky73/spam-mails-dataset) from Kaggle.

## Introduction

This project aims to build a spam detection system that can classify emails as spam or not spam. The neural network model utilizes LSTM (Long Short-Term Memory) layers and an attention mechanism to effectively capture the contextual information from the email texts.

## Dataset

The dataset used in this project is the [Spam Mails Dataset](https://www.kaggle.com/datasets/venky73/spam-mails-dataset). It consists of email texts labeled as spam or not spam.

## Preprocessing

The preprocessing steps include:

1. **Text Cleaning**: Removing non-alphabetic characters and converting text to lowercase. This step helps in standardizing the text data, making it easier to process.
   
2. **Tokenization and Stop Word Removal**: Splitting text into individual words and removing common English stop words (e.g., "and", "the", "is"). Tokenization converts the text into a sequence of words, while stop word removal eliminates words that do not contribute significant meaning.

3. **Lemmatization**: Converting words to their base or root form (e.g., "running" to "run"). Lemmatization helps in reducing the vocabulary size by grouping together different forms of the same word.

4. **Text to Sequence Conversion**: Using Keras Tokenizer to convert cleaned text into sequences of integers, where each integer represents a unique word in the vocabulary. This step transforms text data into numerical data that can be used by neural networks.

5. **Padding Sequences**: Ensuring all sequences have the same length by padding them with zeros. Padding is necessary because neural networks expect inputs of uniform size.

## Model Architecture

The neural network model is built using PyTorch and includes the following components:

1. **Embedding Layer**: Converts input sequences into dense vectors of fixed size. This layer learns word embeddings, which capture semantic information about the words.

2. **LSTM Layer**: Captures the sequential dependencies in the text data. LSTM (Long Short-Term Memory) is a type of recurrent neural network (RNN) that is particularly effective for sequences with long-term dependencies.

3. **Attention Mechanism**: Computes attention weights and produces a context vector by weighted summation of LSTM outputs. The attention mechanism allows the model to focus on important parts of the sequence, enhancing its ability to capture relevant information.

4. **Fully Connected Layers**: Two fully connected layers are used:
   - The first fully connected layer reduces the dimensionality to 128 units and applies ReLU activation.
   - The second fully connected layer outputs a single value with a sigmoid activation for binary classification. The sigmoid activation function maps the output to a probability between 0 and 1.

5. **Dropout Layers**: Added to prevent overfitting by randomly setting a fraction of the input units to zero during training.

## Evaluation

The model is evaluated using the F1 score and a confusion matrix to measure its performance.

### F1 Score

The F1 score is a measure of a model's accuracy that considers both precision and recall. It is especially useful for imbalanced datasets, as it provides a single metric that balances the trade-off between precision and recall.

### Confusion Matrix

The confusion matrix provides a detailed breakdown of the model's performance by showing the true positives, false positives, true negatives, and false negatives. This helps in understanding how well the model is performing in terms of correctly and incorrectly classified instances.

## Results

The model achieved the following results on the test set:

- **F1 Score**: `0.5987`
- **Confusion Matrix**:
  ![Confusion Matrix](path/to/confusion_matrix.png)

