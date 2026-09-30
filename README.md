# Machine Learning & Natural Language Processing Practice Suite (ML-10)

This repository contains a structured collection of machine learning experiments, text classification pipelines, and real-world sentiment analysis tasks implemented using **Scikit-Learn**, **Pandas**, and **Jupyter Notebooks**.

---

## 📂 Project Structure & Notebook Overview

### 1. `01_spam_custom_stopwords.ipynb`
* **Objective:** Mitigating corporate and dataset-specific bias in text classification.
* **Methodology:** Enhances standard English stop words by appending Enron-specific corporate tokens (`enron`, `vince`, `kaminski`, `ect`, `hou`). A `Pipeline` consisting of `CountVectorizer` and `MultinomialNB` is trained on 30,000+ emails.
* **Outcome:** Prevents the classifier from exploiting internal company jargon as false indicators for legitimate (Ham) emails, improving real-world model reliability.

---

### 2. `02_spam_feature_engineering.ipynb`
* **Objective:** Combining textual content with handcrafted structural metadata using heterogeneous feature pipelines.
* **Methodology:** Extracts linguistic heuristics (uppercase character frequency, exclamation marks, question marks, and URL presence) alongside raw text. Utilizes `ColumnTransformer` to vectorize text via `CountVectorizer` while normalizing numerical features with `MinMaxScaler`.
* **Outcome:** Produces an enriched feature representation that captures stylistic spam behaviors alongside semantic keywords.

---

### 3. `03_spam_probability_calibration.ipynb`
* **Objective:** Calibrating overconfident posterior probability estimates in Naive Bayes.
* **Methodology:** Implements probability calibration via `CalibratedClassifierCV` using Platt Scaling (`method='sigmoid'`) with 5-fold cross-validation.
* **Outcome:** Normalizes the extreme log-odds tendencies of Naive Bayes toward realistic, well-calibrated probabilities in the $[0, 1]$ interval, visualized via distribution histograms.

---

### 4. `04_spam_online_learning.ipynb`
* **Objective:** Simulating streaming/incremental model updates without retraining on historical data.
* **Methodology:** Employs the `partial_fit` API of `MultinomialNB` over an out-of-core batch stream (1,000 samples per batch) across an aligned global vocabulary.
* **Outcome:** Demonstrates resource-efficient, real-time adaptation suitable for streaming production spam filters.

---

### 5. `05_spam_sms_generalization.ipynb`
* **Objective:** Evaluating domain adaptation and zero-shot cross-dataset generalization.
* **Methodology:** Applies the baseline model trained exclusively on long-form corporate emails (Enron) directly to an independent dataset of short, slang-heavy SMS messages (`SMS Spam Collection`).
* **Outcome:** Evaluates distribution shifts and highlights precision/recall trade-offs when transitioning across text genres and lengths.

---

### 6. `06_snappfood_sentiment_analysis.ipynb`
* **Objective:** Binary sentiment classification (Happy vs. Sad) on real-world Persian customer reviews.
* **Methodology:** Processes ~70,000 Persian restaurant reviews from the Snappfood dataset using a `TfidfVectorizer` (sublinear term frequencies) coupled with a `MultinomialNB` classifier.
* **Performance:** 
  * **Dataset Size:** 69,480 reviews (55,584 Train / 13,896 Test)
  * **Test Accuracy:** **~83.45%**
  * **Macro F1-Score:** **~0.8339**
* **Evaluation:** Complete performance analysis using precision/recall metrics and a confusion matrix display.

---

## 🛠️ Tech Stack & Requirements

* **Language:** Python 3.10+
* **Core Libraries:**
  * `scikit-learn`
  * `pandas`
  * `numpy`
  * `matplotlib`

Install dependencies via pip:
```bash
pip install numpy pandas scikit-learn matplotlib