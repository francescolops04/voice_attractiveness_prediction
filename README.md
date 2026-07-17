# Voice Attractiveness Predictor

This project implements a machine learning and deep learning pipeline to predict human voice attractiveness scores from audio recordings using the Coco-Nut dataset (humoresque subset). It benchmarks the performance of traditional acoustic features, self-supervised learning (SSL) embeddings, and end-to-end deep learning models.

***

## Project Features

The pipeline is structured into the following stages:

* **Data Preparation**: Extracts and aligns `.wav` audio files with their respective metadata and mean attractiveness ratings. The dataset is filtered and capped at a balanced subset of 1,000 samples (700 train, 150 validation, 150 test).
* **Handcrafted Feature Extraction**: Computes classic acoustic descriptors (including MFCCs, deltas, spectral features, pitch, RMS energy, and harmonic ratio) using the `librosa` library.
* **Deep Embedding Extraction**: Generates 768-dimensional acoustic embeddings using pre-trained models:
  * Wav2Vec 2.0
  * Audio Spectrogram Transformer (AST)
* **Exploratory Data Analysis (EDA)**: Conducts dimensionality reduction using PCA and t-SNE, paired with unsupervised K-Means clustering, to analyze embedding distributions against target scores.
* **Classic Modeling**: Optimizes and trains regression algorithms via Grid Search and cross-validation:
  * Ridge Regression
  * K-Nearest Neighbors (KNN)
  * Decision Tree, Random Forest, and Gradient Boosting
  * Support Vector Regression (SVR with RBF kernel)
* **End-to-End Learning (CNN-BiLSTM)**: Trains a hybrid neural network in TensorFlow/Keras using 64-band Log-Mel Spectrograms to extract spatial-temporal voice patterns.
* **Model Explainability (XAI)**: Computes SHAP values to quantify the impact and importance of individual traditional acoustic features on the SVR model predictions.
* **Noise Robustness Stress Testing**: Tests model resilience against acoustic degradation by injecting Additive White Gaussian Noise (AWGN) at varying SNR levels ranging from clean down to 5 dB.

***

## Requirements and Dependencies

The core Python libraries required to run this notebook include:

* **Audio Processing**: `librosa`, `soundfile`, `torchaudio`
* **Deep Learning**: `torch`, `tensorflow`, `transformers` (Hugging Face)
* **Machine Learning**: `scikit-learn`
* **Interpretability**: `shap`
* **Data Science & Visualization**: `pandas`, `numpy`, `matplotlib`, `seaborn`

***
