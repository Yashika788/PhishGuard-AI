# 🛡️ PhishGuard AI

**Intelligent Phishing URL Detection System using Deep Learning**

PhishGuard AI is a Deep Learning-based web application that analyzes URL structure and lexical characteristics to identify potentially phishing URLs using an Artificial Neural Network (ANN).

The application extracts 32 deterministic URL-based features and uses a trained TensorFlow/Keras model to classify URLs as **Potential Phishing** or **Legitimate**.

> **Disclaimer:** This is an educational URL-text classification project. It does not visit websites, inspect webpage content, perform DNS or reputation checks, or guarantee website safety. Model scores are not calibrated probabilities.

## ✨ Features

- ANN-based phishing URL classification.
- Extraction of 32 URL lexical and structural features (including entropy, suspicious extensions, brand names, and free-hosting detection).
- Interactive Streamlit dashboard with a premium cybersecurity UI.
- URL scanning with prediction score and auto-tuned threshold.
- Scan history and live analytics for the current application session.
- Model evaluation and performance visualizations.
- Confusion matrix and training/validation graphs.

## 🧰 Technologies Used

- Python 3.12
- TensorFlow / Keras
- Scikit-learn
- Pandas and NumPy
- Streamlit
- Matplotlib and Seaborn
- Joblib
- tldextract

## 📁 Project Structure

```text
PhishGuard-AI/
├── assets/
│   └── style.css
├── dataset/
│   └── raw/
│       └── url_features_extracted1.csv
├── models/
│   ├── phishing_model.keras
│   ├── scaler.pkl
│   └── metadata.json
├── notebooks/
├── results/
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── metrics.json
│   ├── training_accuracy.png
│   └── training_loss.png
├── src/
│   ├── feature_extraction.py
│   ├── data_pipeline.py
│   ├── model.py
│   └── evaluation.py
├── app.py
├── train.py
├── requirements.txt
└── README.md
```

## 📊 Dataset

This project uses the **LegitPhish Dataset Version 2**.

- Original dataset file: `url_features_extracted1.csv`
- Target column: `ClassLabel`
- Label mapping:
  - `0` = Phishing
  - `1` = Legitimate

**Feature engineering:** The source CSV contains precomputed feature columns. This implementation extracts its own **32 deterministic URL-based features** from the original URL text and uses those features consistently during training and application inference. The precomputed columns from the CSV are not used as the ANN input.

The dataset is cleaned by removing missing labels and duplicate URLs before model training.

## 🧠 Deep Learning Architecture

The project uses a feed-forward Artificial Neural Network.

| Layer | Configuration |
|---|---|
| Input | 32 URL features |
| Dense 1 | 128 neurons, ReLU |
| Dropout | 0.30 |
| Dense 2 | 64 neurons, ReLU |
| Dropout | 0.20 |
| Dense 3 | 32 neurons, ReLU |
| Output | 1 neuron, Sigmoid |

**Optimizer:** Adam  
**Loss Function:** Binary Crossentropy  
**Training:** Early Stopping & ReduceLROnPlateau  
**Class Imbalance:** Handled via class weights

## ⚙️ Installation and Setup

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd PhishGuard-AI
```

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the dataset at:

```text
dataset/raw/url_features_extracted1.csv
```

## 🚀 Train the Model

Run the following command:

```bash
python train.py --data dataset/raw/url_features_extracted1.csv --label-column ClassLabel --phishing-value 0.0
```

The training process performs:

1. Dataset loading and cleaning.
2. Duplicate URL removal.
3. Extraction of 32 URL features.
4. Stratified train, validation and test splitting.
5. Feature scaling.
6. ANN model training with early stopping.
7. Evaluation on the held-out test set.
8. **Threshold tuning on the validation set** to maximize F1-score.
9. Saving the trained model, scaler, metrics, metadata, and graphs.

## 🌐 Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal (usually `http://localhost:8501`).

## 📈 Model Evaluation

The current held-out test evaluation produced:

| Metric | Phishing Class | Legitimate Class |
|---|---:|---:|
| Precision | 99.96% | 99.76% |
| Recall | 99.86% | 99.93% |
| F1-Score | 99.91% | 99.85% |

**Overall Metrics:**
| Metric | Score |
|---|---:|
| Accuracy | 99.89% |
| ROC-AUC | 99.999% |
| Tuned Threshold | 0.65 |

These results are from the current random stratified test split. They should not be interpreted as proof of equivalent real-world performance. Domain overlap and dataset leakage may inflate random-split results.

### Evaluation Visualizations

- Confusion Matrix
- Training and Validation Accuracy
- Training and Validation Loss

Generated files are stored in the `results/` directory.

## ⚠️ Limitations

- The model analyzes URL text only.
- It does not inspect webpage content or website behavior.
- It does not perform DNS, WHOIS or external reputation checks.
- Compromised legitimate domains and newly created phishing URLs may be difficult to identify.
- Scan history is maintained only during the current application session.
- Random row splitting may overestimate generalization when related URLs occur across different splits.

A stronger evaluation should use domain-grouped or time-based splitting and an independent external test dataset.

## 🔮 Future Improvements

- Domain-aware and time-based evaluation.
- Comparison with Random Forest, Logistic Regression and XGBoost.
- Feature explainability using SHAP.
- Integration with trusted URL reputation services.
- Persistent scan history and reporting.

## 🎓 Project Domain

**Deep Learning | Cybersecurity | Artificial Neural Networks | Phishing Detection**

Developed as an academic project to explore the application of Deep Learning in cybersecurity.
```

---

