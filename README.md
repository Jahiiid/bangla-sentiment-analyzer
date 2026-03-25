# বাংলা Sentiment Analyzer 🇧🇩

A Machine Learning web app that detects **Positive** or **Negative**
sentiment in Bangla text — built with Python, Scikit-learn & Flask.

![Python](https://img.shields.io/badge/Python-3.14-blue?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-3.x-green?style=flat-square)
![Scikit-learn](https://img.shields.io/badge/ML-Scikit--learn-orange?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-500%20sentences-purple?style=flat-square)
![Accuracy](https://img.shields.io/badge/CV%20Accuracy-86.4%25-brightgreen?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-success?style=flat-square)

---

## 🎯 What it does

Type any Bangla sentence → AI instantly predicts sentiment:

| Input | Output | Confidence |
|-------|--------|------------|
| "এই পণ্যটি অনেক ভালো, সবাইকে নেওয়া উচিত" | 😊 Positive | 77.1% |
| "একদম বাজে জিনিস, টাকা নষ্ট হয়েছে" | 😞 Negative | 97.5% |
| "অসাধারণ সেবা পেয়েছি, অনেক খুশি" | 😊 Positive | 97.0% |
| "আর কখনো এখান থেকে কিনবো না" | 😞 Negative | 83.6% |

---

## 🧠 How it works
```
Bangla Text
    ↓
TF-IDF Vectorizer (character n-grams: 1–4)
    ↓
Best Model Auto-Selection
(Logistic Regression vs Linear SVM vs Random Forest)
    ↓
Sentiment: Positive 😊 or Negative 😞 + Confidence Score
```

### Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Models | Logistic Regression, Linear SVM, Random Forest |
| Model Selection | Auto cross-validation (best model wins) |
| Text Processing | TF-IDF Vectorizer (char n-grams 1–4) |
| Backend | Python + Flask |
| Frontend | HTML5 + CSS3 + Vanilla JavaScript |
| Dataset | 500 custom Bangla sentences |
| CV Accuracy | **86.4%** (Logistic Regression) |

---

## 📊 Model Performance
```
📊 সব Model-এর Accuracy তুলনা:
  Logistic Regression    Test: 78.0%  |  CV: 86.4%  🏆
  Linear SVM             Test: 77.0%  |  CV: 87.6%
  Random Forest          Test: 76.0%  |  CV: 81.6%

              precision    recall  f1-score
  😞 Negative    0.80      0.74      0.77
  😊 Positive    0.76      0.82      0.79
  accuracy                           0.78
```

---

## 📂 Project Structure
```
bangla-sentiment-analyzer/
│
├── 📄 dataset.py          # Dataset creation (500 Bangla sentences)
├── 📄 model.py            # 3 ML models training & auto-selection
├── 📄 app.py              # Flask web application + UI
│
├── 📊 bangla_reviews.csv  # Training data (500 sentences, balanced)
├── 🤖 model.pkl           # Best trained ML model (auto-selected)
│
├── 📋 requirements.txt    # Python dependencies
└── 📖 README.md           # Project documentation
```

---

## 🚀 Run Locally

### Prerequisites
- Python 3.8+
- pip

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/Jahiiid/bangla-sentiment-analyzer.git
cd bangla-sentiment-analyzer
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Create dataset & train model**
```bash
python dataset.py
python model.py
```

**4. Launch the web app**
```bash
python app.py
```

**5. Open in browser**
```
http://localhost:5000
```

---

## 💡 Why I built this

Bangladesh has **170 million** Bangla speakers, yet NLP tools
for Bangla remain scarce compared to English. This project
bridges that gap — analyzing sentiment in product reviews,
social media posts, and customer feedback written in Bangla.

Key highlights:
- **500 real-world Bangla sentences** across 10+ categories
  (shopping, food, travel, education, healthcare, and more)
- **Auto model selection** — trains 3 models, picks the best
- **Character-level n-grams** — works without word tokenization,
  ideal for morphologically rich languages like Bangla

This project is part of my journey toward an **MSc in Computer
Science & AI** where I aim to specialize in low-resource
language NLP.

---

## 🔮 Roadmap

- [x] Basic sentiment detection (Positive / Negative)
- [x] Web interface with confidence score & animated bar
- [x] 500 balanced Bangla sentences (250 pos / 250 neg)
- [x] Auto model selection (LR vs SVM vs Random Forest)
- [x] 10+ real-world categories in dataset
- [ ] Expand dataset to 2000+ sentences
- [ ] Add Neutral sentiment class
- [ ] Integrate BanglaBERT for higher accuracy
- [ ] Deploy on Hugging Face Spaces (public demo)
- [ ] REST API for third-party integration
- [ ] Browser extension for real-time sentiment detection

---

## 👨‍💻 Author

**Jahid N Shakil**
BSc in Computer Science & Engineering

[![GitHub](https://img.shields.io/badge/GitHub-Jahiiid-181717?style=flat-square&logo=github)](https://github.com/Jahiiid)
[![Email](https://img.shields.io/badge/Email-jahhhiiid%40gmail.com-D14836?style=flat-square&logo=gmail)](mailto:jahhhiiid@gmail.com)

---

> *"170 million people speak Bangla — they deserve better NLP tools."*

> *"The best way to learn Machine Learning is to build something
> that actually matters."*