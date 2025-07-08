# Non-Linear SVM Kernel Multi-Class Classifier for Emotion Detection

## Overview

This project implements a multi-class emotion detection system using a non-linear Support Vector Machine (SVM) with kernel trick. The classifier is capable of distinguishing between multiple emotional states based on text input. It leverages various kernel functions to capture non-linearity in data and provides comparative analysis of their performance.

This project is particularly focused on text classification in the domain of sentiment and emotion analysis. It aims to demonstrate how SVMs, which are traditionally binary classifiers, can be extended for multi-class classification tasks using one-vs-rest or one-vs-one strategies.

---

## Key Features

* Multi-class classification using SVM
* Support for non-linear kernel functions including Radial Basis Function (RBF), polynomial, and sigmoid kernels
* Text pre-processing pipeline including tokenization, stop word removal, and TF-IDF vectorization
* Evaluation of model performance using confusion matrix and accuracy metrics
* Visualization of model results and data distributions
* Emotion detection based on text input

---

## Dataset

The dataset used in this project includes labeled text samples representing various emotions such as joy, sadness, anger, fear, and more. The dataset is preprocessed and split into training and testing subsets.

Dataset characteristics:

* Input: Raw text data
* Output: Emotion label per text sample

---

## Requirements

To run the code in this repository, ensure you have the following Python libraries installed:

* numpy
* pandas
* matplotlib
* seaborn
* sklearn
* nltk

You can install these packages using pip:

```
pip install numpy pandas matplotlib seaborn scikit-learn nltk
```

---

## File Structure

* `SVM_Kernel_Emotion_Classifier.ipynb`: Main Jupyter notebook containing the implementation of the classifier
* `README.md`: Project description and documentation
* `emotion_dataset.csv` (if included): Dataset used for training and testing

---

## Approach

1. **Data Preprocessing**

   * Load the dataset
   * Clean and normalize text
   * Tokenize and vectorize using TF-IDF
   * Encode emotion labels

2. **Model Training**

   * Implement SVM classifier with different kernels
   * Train the model on the training data
   * Predict emotions on the test data

3. **Evaluation**

   * Generate confusion matrix
   * Calculate accuracy and classification metrics
   * Visualize model performance

4. **Visualization**

   * Plot the distribution of emotions
   * Display kernel-wise performance results

---

## Results

The models are evaluated based on accuracy and F1-score. Comparative results of different kernel functions are plotted and analyzed to determine the most effective kernel for emotion classification tasks. The Radial Basis Function kernel typically shows better generalization due to its ability to model complex non-linear decision boundaries.

---

## Applications

This project can serve as a foundational block for applications such as:

* Sentiment and emotion-aware chatbots
* Social media monitoring tools
* Mental health tracking applications
* Customer feedback analysis

---

## Future Work

* Integration with deep learning models for improved accuracy
* Use of contextual embeddings like BERT for better feature representation
* Real-time emotion detection web interface or API
* Expansion to multimodal emotion detection including audio and visual cues

---

## Author

Developed by Ayush Jha
Student at Delhi Technological University
Email: [jhaayush710@gmail.com](mailto:jhaayush710@gmail.com)
GitHub: github.com/JhaAyushCanCode

---
