import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import pickle

# ── ১. Data Load ──────────────────────────────────────
df = pd.read_csv("bangla_reviews.csv")
print(f"✅ Data load হয়েছে! মোট বাক্য: {len(df)}\n")

# ── ২. Train/Test Split ───────────────────────────────
X = df["text"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"🏋️  Training: {len(X_train)} | 🧪 Testing: {len(X_test)}\n")

# ── ৩. সব Model Test করো ─────────────────────────────
print("="*50)
print("📊 সব Model-এর Accuracy তুলনা:")
print("="*50)

models = {
    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(1,4), max_features=50000)),
        ("clf",   LogisticRegression(C=5, max_iter=1000))
    ]),
    "Linear SVM": Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(1,4), max_features=50000)),
        ("clf",   LinearSVC(C=1.0, max_iter=2000))
    ]),
    "Random Forest": Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2,3), max_features=10000)),
        ("clf",   RandomForestClassifier(n_estimators=200, random_state=42))
    ]),
}

best_model = None
best_score = 0
best_name  = ""

for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    score = accuracy_score(y_test, pipeline.predict(X_test))
    cv    = cross_val_score(pipeline, X, y, cv=5).mean()
    print(f"  {name:<22} Test: {score*100:.1f}%  |  CV: {cv*100:.1f}%")
    if score > best_score:
        best_score = score
        best_model = pipeline
        best_name  = name

print()
print(f"🏆 সেরা Model: {best_name} ({best_score*100:.1f}%)")
print("="*50)

# ── ৪. সেরা Model-এর বিস্তারিত রিপোর্ট ──────────────
print(f"\n📋 {best_name} — বিস্তারিত রিপোর্ট:")
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred,
      target_names=["😞 Negative", "😊 Positive"]))

# ── ৫. Model Save করো ────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)
print("💾 সেরা Model save হয়েছে!\n")

# ── ৬. নিজে Test করো ─────────────────────────────────
print("="*50)
print("🔍 নিজে Test করো:")
print("="*50)

test_sentences = [
    "এই পণ্যটি অনেক ভালো, সবাইকে নেওয়া উচিত",
    "একদম বাজে জিনিস, টাকা নষ্ট হয়েছে",
    "মোটামুটি ঠিক আছে",
    "অসাধারণ সেবা পেয়েছি, অনেক খুশি",
    "আর কখনো এখান থেকে কিনবো না",
]

for sentence in test_sentences:
    prediction = best_model.predict([sentence])[0]
    try:
        confidence = round(best_model.predict_proba([sentence])[0].max()*100, 1)
    except:
        confidence = "—"
    label = "😊 Positive" if prediction == 1 else "😞 Negative"
    print(f"\nবাক্য:  {sentence}")
    print(f"ফলাফল: {label}  (confidence: {confidence}%)")