# বাংলা Sentiment Analyzer 🇧🇩

A Machine Learning web app that detects **Positive** or **Negative**
sentiment in Bangla text — built with Python, Scikit-learn & Flask.

![Python](https://img.shields.io/badge/Python-3.14-blue?style=flat-square)
![Flask](https://img.shields.io/badge/Flask-3.x-green?style=flat-square)
![Scikit-learn](https://img.shields.io/badge/ML-Scikit--learn-orange?style=flat-square)
![Accuracy](https://img.shields.io/badge/Accuracy-78.6%25-brightgreen?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-success?style=flat-square)

---

## 🎯 What it does

Type any Bangla sentence → AI instantly predicts sentiment:

| Input | Output |
|-------|--------|
| "এই পণ্যটি অসাধারণ, আমি খুব খুশি" | 😊 Positive (95%) |
| "একদম বাজে জিনিস, টাকা নষ্ট হয়েছে" | 😞 Negative (88%) |
| "দারুণ সেবা পেলাম, আবার আসবো" | 😊 Positive (91%) |

---

## 🧠 How it works
```
Bangla Text
    ↓
TF-IDF Vectorizer (character n-grams: 2–3)
    ↓
Logistic Regression Model
    ↓
Sentiment: Positive 😊 or Negative 😞 + Confidence Score
```

### Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | Logistic Regression (Scikit-learn) |
| Text Processing | TF-IDF Vectorizer (char n-grams) |
| Backend | Python + Flask |
| Frontend | HTML5 + CSS3 + Vanilla JavaScript |
| Dataset | 70 custom Bangla sentences |
| Accuracy | **78.6%** |

---

## 📊 Model Performance
```
🎯 Overall Accuracy: 78.6%

              precision    recall  f1-score
  😞 Negative    0.57      1.00      0.73
  😊 Positive    1.00      0.70      0.82
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

## 📂 Project Structure
```
bangla-sentiment-analyzer/
│
├── 📄 dataset.py          # Dataset creation & CSV export
├── 📄 model.py            # ML model training & evaluation
├── 📄 app.py              # Flask web application + UI
│
├── 📊 bangla_reviews.csv  # Training data (70 Bangla sentences)
├── 🤖 model.pkl           # Trained Logistic Regression model
├── 🔢 vectorizer.pkl      # Fitted TF-IDF vectorizer
│
├── 📋 requirements.txt    # Python dependencies
└── 📖 README.md           # Project documentation
```

---

## 🔮 Roadmap

- [x] Basic sentiment detection (Positive / Negative)
- [x] Web interface with confidence score
- [x] Animated result display
- [x] Expand dataset to 500+ sentences
- [ ] Add Neutral sentiment class
- [ ] Integrate BERT / BanglaBERT for higher accuracy
- [ ] Deploy on Hugging Face Spaces (public demo)
- [ ] REST API for third-party integration

---

## 💡 Why I built this

Bangladesh has **170 million** Bangla speakers, yet NLP tools for
Bangla remain scarce compared to English. This project is a small
step toward bridging that gap — analyzing sentiment in product
reviews, social media posts, and customer feedback in Bangla.

This project is part of my journey toward an **MSc in Computer
Science & AI** where I aim to work on low-resource language NLP.

---

## 👨‍💻 Author

**Jahid N Shakil**
BSc in Computer Science & Engineering

[![GitHub](https://img.shields.io/badge/GitHub-Jahiiid-181717?style=flat-square&logo=github)](https://github.com/Jahiiid)
[![Email](https://img.shields.io/badge/Email-jahhhiiid%40gmail.com-D14836?style=flat-square&logo=gmail)](mailto:jahhhiiid@gmail.com)

---

> *"The best way to learn Machine Learning is to build something
> that actually matters."*