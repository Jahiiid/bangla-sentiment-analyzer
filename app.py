from flask import Flask, request, jsonify, render_template_string
from sklearn.pipeline import Pipeline
import pickle
import os

app = Flask(__name__)

# Model load করো
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
with open(model_path, "rb") as f:
    model: Pipeline = pickle.load(f)

# ── HTML Page ─────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="bn">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>বাংলা Sentiment Analyzer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        .card {
            background: rgba(255,255,255,0.05);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 24px;
            padding: 40px;
            width: 100%;
            max-width: 600px;
            box-shadow: 0 25px 50px rgba(0,0,0,0.4);
        }

        .badge {
            display: inline-block;
            background: rgba(99,102,241,0.2);
            color: #a5b4fc;
            border: 1px solid rgba(99,102,241,0.3);
            padding: 4px 14px;
            border-radius: 999px;
            font-size: 12px;
            margin-bottom: 16px;
            letter-spacing: 1px;
        }

        h1 {
            color: #fff;
            font-size: 26px;
            font-weight: 700;
            margin-bottom: 6px;
        }

        .subtitle {
            color: rgba(255,255,255,0.4);
            font-size: 14px;
            margin-bottom: 30px;
        }

        textarea {
            width: 100%;
            background: rgba(255,255,255,0.07);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 14px;
            padding: 16px;
            color: #fff;
            font-size: 16px;
            resize: none;
            height: 120px;
            outline: none;
            transition: border 0.3s;
            font-family: inherit;
        }

        textarea::placeholder { color: rgba(255,255,255,0.25); }
        textarea:focus { border-color: rgba(99,102,241,0.6); }

        button {
            width: 100%;
            margin-top: 14px;
            padding: 14px;
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            color: #fff;
            border: none;
            border-radius: 14px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: opacity 0.2s, transform 0.1s;
        }

        button:hover  { opacity: 0.9; }
        button:active { transform: scale(0.98); }

        .result {
            margin-top: 24px;
            padding: 20px;
            border-radius: 16px;
            display: none;
            animation: fadeIn 0.4s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to   { opacity: 1; transform: translateY(0); }
        }

        .positive {
            background: rgba(16,185,129,0.1);
            border: 1px solid rgba(16,185,129,0.3);
        }

        .negative {
            background: rgba(239,68,68,0.1);
            border: 1px solid rgba(239,68,68,0.3);
        }

        .emoji { font-size: 42px; margin-bottom: 8px; }

        .label {
            font-size: 22px;
            font-weight: 700;
            margin-bottom: 6px;
        }

        .positive .label { color: #34d399; }
        .negative .label { color: #f87171; }

        .confidence { color: rgba(255,255,255,0.5); font-size: 14px; }

        .bar-wrap {
            background: rgba(255,255,255,0.08);
            border-radius: 999px;
            height: 8px;
            margin-top: 14px;
            overflow: hidden;
        }

        .bar {
            height: 100%;
            border-radius: 999px;
            transition: width 0.8s ease;
        }

        .positive .bar { background: linear-gradient(90deg, #10b981, #34d399); }
        .negative .bar { background: linear-gradient(90deg, #ef4444, #f87171); }

        .examples {
            margin-top: 24px;
            border-top: 1px solid rgba(255,255,255,0.07);
            padding-top: 20px;
        }

        .examples p {
            color: rgba(255,255,255,0.35);
            font-size: 12px;
            margin-bottom: 10px;
            letter-spacing: 0.5px;
        }

        .chips { display: flex; flex-wrap: wrap; gap: 8px; }

        .chip {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.1);
            color: rgba(255,255,255,0.6);
            padding: 6px 14px;
            border-radius: 999px;
            font-size: 13px;
            cursor: pointer;
            transition: all 0.2s;
        }

        .chip:hover {
            background: rgba(99,102,241,0.2);
            border-color: rgba(99,102,241,0.4);
            color: #fff;
        }

        .loading { display: none; color: rgba(255,255,255,0.4); font-size: 14px; margin-top: 12px; text-align: center; }
    </style>
</head>
<body>
<div class="card">
    <div class="badge">🤖 ML PROJECT · PYTHON</div>
    <h1>বাংলা Sentiment Analyzer</h1>
    <p class="subtitle">যেকোনো বাংলা বাক্য লিখুন — AI বলবে Positive না Negative</p>

    <textarea id="inputText" placeholder="এখানে বাংলা লিখুন... যেমন: এই পণ্যটি অনেক ভালো"></textarea>
    <button onclick="analyze()">🔍 বিশ্লেষণ করো</button>

    <div class="loading" id="loading">⏳ বিশ্লেষণ হচ্ছে...</div>

    <div class="result" id="result">
        <div class="emoji" id="emoji"></div>
        <div class="label" id="label"></div>
        <div class="confidence" id="conf"></div>
        <div class="bar-wrap">
            <div class="bar" id="bar" style="width:0%"></div>
        </div>
    </div>

    <div class="examples">
        <p>উদাহরণ চেষ্টা করুন</p>
        <div class="chips">
            <span class="chip" onclick="setExample(this)">অসাধারণ পণ্য!</span>
            <span class="chip" onclick="setExample(this)">একদম বাজে ছিল</span>
            <span class="chip" onclick="setExample(this)">দারুণ সেবা পেলাম</span>
            <span class="chip" onclick="setExample(this)">টাকা নষ্ট হয়েছে</span>
            <span class="chip" onclick="setExample(this)">অনেক খুশি হলাম</span>
            <span class="chip" onclick="setExample(this)">হতাশ হয়েছি</span>
        </div>
    </div>
</div>

<script>
function setExample(el) {
    document.getElementById('inputText').value = el.innerText;
    analyze();
}

async function analyze() {
    const text = document.getElementById('inputText').value.trim();
    if (!text) { alert('কিছু লিখুন!'); return; }

    document.getElementById('loading').style.display = 'block';
    document.getElementById('result').style.display   = 'none';

    const res  = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
    });
    const data = await res.json();

    document.getElementById('loading').style.display = 'none';

    const resultDiv = document.getElementById('result');
    resultDiv.className = 'result ' + data.sentiment;
    resultDiv.style.display = 'block';

    document.getElementById('emoji').innerText = data.sentiment === 'positive' ? '😊' : '😞';
    document.getElementById('label').innerText = data.sentiment === 'positive' ? 'ইতিবাচক (Positive)' : 'নেতিবাচক (Negative)';
    document.getElementById('conf').innerText  = `আত্মবিশ্বাস: ${data.confidence}%`;

    setTimeout(() => {
        document.getElementById('bar').style.width = data.confidence + '%';
    }, 50);
}
</script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    text       = request.json.get("text", "")
    prediction = model.predict([text])[0]
    try:
        confidence = round(model.predict_proba([text])[0].max() * 100, 1)
    except:
        confidence = 95.0

    return jsonify({
        "sentiment":  "positive" if prediction == 1 else "negative",
        "confidence": confidence
    })

if __name__ == "__main__":
    print("✅ Server চালু হচ্ছে...")
    print("🌐 Browser-এ যাও: http://localhost:5000")
    app.run(debug=True)