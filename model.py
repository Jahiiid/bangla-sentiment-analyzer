import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# ── ১. Data Load ──────────────────────────────────────
df = pd.read_csv("bangla_reviews.csv")
print("✅ Data load হয়েছে!")
print(f"📊 মোট বাক্য: {len(df)}\n")

# ── ২. Train/Test Split ───────────────────────────────
X = df["text"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"🏋️  Training data: {len(X_train)} বাক্য")
print(f"🧪 Testing data:  {len(X_test)} বাক্য\n")

# ── ৩. TF-IDF Vectorizer ──────────────────────────────
# বাংলা text কে numbers এ রূপান্তর করে
vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

print("✅ Text → Numbers রূপান্তর হয়েছে!")

# ── ৪. Model Train ────────────────────────────────────
model = LogisticRegression()
model.fit(X_train_vec, y_train)
print("✅ Model training সম্পন্ন!\n")

# ── ৫. Accuracy দেখো ─────────────────────────────────
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print(f"🎯 Accuracy: {accuracy * 100:.1f}%")
print("\n📋 বিস্তারিত রিপোর্ট:")
print(classification_report(y_test, y_pred,
      target_names=["😞 Negative", "😊 Positive"]))

# ── ৬. Model Save করো ────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("💾 Model save হয়েছে! (model.pkl)")

# ── ৭. নিজে Test করো ─────────────────────────────────
print("\n" + "="*50)
print("🔍 নিজে Test করো:")
print("="*50)

test_sentences = [
    "এই পণ্যটি অনেক ভালো, সবাইকে নেওয়া উচিত",
    "একদম বাজে জিনিস, টাকা নষ্ট হয়েছে",
    "মোটামুটি ঠিক আছে",
]

for sentence in test_sentences:
    vec = vectorizer.transform([sentence])
    prediction = model.predict(vec)[0]
    confidence = model.predict_proba(vec)[0].max() * 100

    label = "😊 Positive" if prediction == 1 else "😞 Negative"
    print(f"\nবাক্য:  {sentence}")
    print(f"ফলাফল: {label} ({confidence:.1f}% confident)")