# 🛡️ PhishGuard AI

**Intelligent Phishing URL Detection System using Deep Learning**

PhishGuard AI is a Deep Learning-based web application that analyzes URL structure and lexical characteristics to identify potentially phishing URLs using an Artificial Neural Network (ANN).

The application extracts **25 deterministic URL-based features** and uses a trained TensorFlow/Keras model to classify URLs as **Potential Phishing** or **Legitimate**.

> **Disclaimer:** This is an educational URL-text classification project. It does not visit websites, inspect webpage content, perform DNS or reputation checks, or guarantee website safety. Model scores are not calibrated probabilities.

## ✨ Features

- ANN-based phishing URL classification.
- Extraction of 25 URL lexical and structural features.
- Interactive Streamlit dashboard with a cybersecurity-themed UI.
- URL scanning with phishing score and configurable decision threshold.
- Scan history for the current application session.
- Live scan analytics and visualizations.
- Model evaluation and performance metrics.
- Confusion matrix and training accuracy/loss graphs.

## 🧰 Technologies Used

- Python 3.12
- TensorFlow / Keras
- Scikit-learn
- Pandas and NumPy
- Streamlit
- Matplotlib and Seaborn
- Joblib

## 📁 Project Structure

```text
PhishGuard-AI/
├── assets/
│   └── style.css
├── dataset/
│   ├── raw/
│   │   └── .gitkeep
│   └── processed/
│       └── .gitkeep
├── models/
│   ├── phishing_model.keras
│   ├── scaler.pkl
│   └── metadata.json
├── notebooks/
│   └── README.md
├── results/
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── metrics.json
│   ├── training_accuracy.png
│   └── training_loss.png
├── src/
│   ├── __init__.py
│   ├── feature_extraction.py
│   ├── data_pipeline.py
│   ├── model.py
│   └── evaluation.py
├── app.py
├── train.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 📊 Dataset

This project uses **LegitPhish Dataset Version 2**.

- Dataset file: `url_features_extracted1.csv`
- Target column: `ClassLabel`
- Original label mapping:
  - `0.0` = Phishing
  - `1.0` = Legitimate

### Feature Engineering

The original dataset contains precomputed feature columns. However, this implementation extracts its own **25 deterministic URL-based features** directly from URL text.

The precomputed feature columns are not used as ANN input.

The data pipeline performs:
- Missing-label removal.
- Duplicate URL removal.
- Stratified train, validation and test splitting.
- Feature scaling using training data.

**Dataset availability:** The original CSV is excluded from this GitHub repository through `.gitignore`. Obtain the dataset separately and place it at:

```text
dataset/raw/url_features_extracted1.csv
```

## 🧠 Deep Learning Architecture

The project uses a feed-forward Artificial Neural Network.

| Layer | Configuration |
|---|---|
| Input | 25 URL features |
| Dense 1 | 128 neurons, ReLU |
| Dropout | 0.30 |
| Dense 2 | 64 neurons, ReLU |
| Dropout | 0.20 |
| Dense 3 | 32 neurons, ReLU |
| Output | 1 neuron, Sigmoid |

**Optimizer:** Adam  
**Loss Function:** Binary Crossentropy  
**Training:** Early Stopping

The model produces a phishing score between 0 and 1. The application uses a decision threshold to classify URLs.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Yashika788/PhishGuard-AI.git
cd PhishGuard-AI
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
```

Activate it on macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Place the downloaded dataset file at:

```text
dataset/raw/url_features_extracted1.csv
```

## 🚀 Train the Model

Run:

```bash
python train.py --data dataset/raw/url_features_extracted1.csv --label-column ClassLabel --phishing-value 0.0
```

The training pipeline performs:

1. Dataset loading and cleaning.
2. Duplicate URL removal.
3. Extraction of 25 URL-based features.
4. Stratified train, validation and test splitting.
5. Feature scaling.
6. ANN model training with early stopping.
7. Evaluation on the held-out test set.
8. Validation-based threshold tuning.
9. Saving the trained model, scaler, metadata, metrics and evaluation graphs.

## 🌐 Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

## 📈 Model Evaluation

The current held-out random stratified test evaluation produced the following results:

### Class-wise Metrics

| Metric | Phishing | Legitimate |
|---|---:|---:|
| Precision | 99.96% | 99.76% |
| Recall | 99.86% | 99.93% |
| F1-Score | 99.91% | 99.85% |

### Overall Metrics

| Metric | Score |
|---|---:|
| Accuracy | 99.89% |
| ROC-AUC | 99.999% |
| Application Threshold | 0.65 |
| Validation-tuned Threshold | 0.52 |

The application threshold and validation-tuned threshold are reported separately because they represent different values in the current evaluation output.

> **Important:** These results come from a random stratified test split. They should not be interpreted as proof of equivalent real-world performance. Similar domains or related URLs across dataset splits may inflate evaluation scores.

### Evaluation Visualizations

The project generates:

- Confusion Matrix
- Training and Validation Accuracy
- Training and Validation Loss

These files are stored in the `results/` directory.

## ⚠️ Limitations

- The model analyzes URL text only.
- It does not inspect webpage content or website behavior.
- It does not perform DNS, WHOIS or external reputation checks.
- Compromised legitimate domains and newly created phishing URLs may be difficult to identify.
- Scan history is maintained only during the current application session.
- Random row splitting may overestimate generalization when related URLs occur across different splits.

A stronger evaluation should use domain-grouped or time-based splitting and an independent external test dataset.

## 🔮 Future Improvements

- Domain-aware and time-based model evaluation.
- Comparison with Random Forest, Logistic Regression and XGBoost.
- Feature explainability using SHAP.
- Integration with trusted URL reputation services.
- Persistent scan history and reporting.
- Independent external dataset evaluation.

## 🎓 Project Domain

**Deep Learning | Cybersecurity | Artificial Neural Networks | Phishing Detection**

Developed as an academic project to explore the application of Deep Learning in cybersecurity.