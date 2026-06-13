# 🛡️ Phishing Shield AI

An interactive, single-file Machine Learning web application built with Python, Scikit-learn, and Streamlit. This app utilizes natural language processing (NLP) and heuristic feature engineering to analyze email content and classify them as **Phishing** or **Safe** in real-time.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-F7931E?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Key Features

* **Self-Contained Pipeline:** Automatically generates a balanced mock dataset upon startup—no external dataset files required.
* **Hybrid Feature Engineering:** Combines textual TF-IDF features with explicit heuristics (such as structural tracking of URLs and high-urgency panic-keywords).
* **Live Email Inspector:** An interactive sandbox where you can paste any custom email draft to calculate its real-time malicious risk factor.
* **Model Diagnostics:** Real-time generation of model accuracy scores and an active confusion matrix visualization.
* **Modern UI:** Styled with sleek glassmorphism metrics panels, intuitive color-coded warning banners, and hidden feature metric breakdown expanders.

---

## 🛠️ Tech Stack & Concepts

* **Frontend Dashboard:** [Streamlit](https://streamlit.io/)
* **Vectorization:** `TfidfVectorizer` (Term Frequency-Inverse Document Frequency) to translate text tokens into mathematical vectors.
* **Classifier:** `LogisticRegression` for optimized binary execution speeds and clear probability calculations.
* **Metrics:** Scikit-learn evaluation metrics (`accuracy_score`, `confusion_matrix`).

---

## ⚙️ Installation & Setup

Follow these quick steps to get your local dashboard up and running:

### 1. Clone the Repository
