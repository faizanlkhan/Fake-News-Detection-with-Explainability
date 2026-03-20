# Fake News Detection with Explainability 📰

A machine learning project to classify news articles as real or fake using Natural Language Processing (NLP) techniques, along with basic model explainability.

---

## 🚀 Overview

This project uses:
- Text preprocessing
- TF-IDF vectorization
- Logistic Regression classifier

It also includes explainability by analyzing which words contribute most to predictions.

---

## 📊 Dataset

- Combined dataset of real and fake news articles
- Total samples: ~44,000
- Balanced classes (Real vs Fake)

---

## ⚙️ Pipeline

1. Data loading and labeling
2. Text cleaning (removing URLs, punctuation, etc.)
3. Train-test split (80-20)
4. TF-IDF feature extraction (max_features=5000)
5. Logistic Regression model training
6. Evaluation using:
   - Accuracy
   - Classification report
   - Confusion matrix

---

## 📈 Results

- Accuracy: ~98.7%
- Strong performance on structured news data

---

## 🔍 Explainability

Model interpretability is achieved by analyzing feature importance:

- Words like "reuters", "said", "washington" strongly indicate real news  
- Words like "clickbait-style phrases" tend to indicate fake news  

This shows model identifies words such as "said", "government", and location names as indicators of real news, reflecting formal reporting style.

On the other hand, words associated with sensational or exaggerated language contribute to fake news predictions.

This suggests that the model relies more on stylistic and structural patterns rather than factual correctness.

---

## ⚠️ Limitations

- Does NOT verify factual correctness
- Relies heavily on writing style and patterns
- Performs poorly on:
  - Short text
  - Out-of-distribution inputs
  - Neutral factual statements without context

---

## 🛠 Tech Stack

- Python
- Pandas
- Scikit-learn
- NLP (TF-IDF)

---

## 📂 Project Structure

```
fake-news-detection-with-explainability/
│
├── notebook/
│   └── fake_news_detection.ipynb
│
├── .gitignore
│
├── Fake.csv
├── True.csv
│
├── requirements.txt
└── README.md
```



## ▶️ How to Run

1. Clone the repository

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Open the notebook:
   notebook/fake_news_detection.ipynb

4. Run all cells


## 💡 Key Insight

This project highlights that machine learning models often learn patterns in writing style rather than verifying factual correctness.

For example, structured sources like "Reuters" strongly influence predictions, showing the model relies on textual patterns instead of real-world truth.


## 📌 Future Improvements

- Use deep learning models (LSTM, BERT)
- Add real-world fact-checking integration
- Improve performance on short and unseen text
- Reduce reliance on dataset-specific patterns
