from flask import Flask, request, jsonify, render_template
import pickle
import os
from utils.explainer import analyze_text
from utils.preprocessor import clean_text
from utils.llm_oracle import ask_chatgpt

app = Flask(__name__)

PIPELINE_PATH      = "models/pipeline.pkl"
LEGACY_MODEL_PATH  = "model.pkl"
LEGACY_VEC_PATH    = "vectorizer.pkl"


def load_pipeline():
    if os.path.exists(PIPELINE_PATH):
        p = pickle.load(open(PIPELINE_PATH, "rb"))
        return p, "pipeline"
    if os.path.exists(LEGACY_MODEL_PATH) and os.path.exists(LEGACY_VEC_PATH):
        m = pickle.load(open(LEGACY_MODEL_PATH, "rb"))
        v = pickle.load(open(LEGACY_VEC_PATH,   "rb"))
        return (m, v), "legacy"
    return None, None


clf, model_mode = load_pipeline()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    global clf, model_mode

    if clf is None:
        clf, model_mode = load_pipeline()
        if clf is None:
            return jsonify({"error": "Model not found. Run  python train.py  first."}), 500

    data     = request.json
    raw_text = data.get("text", "").strip()
    if not raw_text:
        return jsonify({"error": "No text provided."}), 400

    cleaned = clean_text(raw_text)
    word_count = len(cleaned.split())

    # ── 1. GenAI / ChatGPT Oracle (Default for ALL queries) ──────────────────
    print("Routing to ChatGPT Oracle...")
    llm_response = ask_chatgpt(raw_text)
    if llm_response:
        return jsonify(llm_response)

    # ── 2. Fallback Traditional ML Pipeline (If Internet/API Fails) ──────────
    print("ChatGPT API failed. Falling back to traditional ML pipeline...")
    try:
        if model_mode == "pipeline":
            ml_pred  = clf.predict([cleaned])[0]          # 0=FAKE, 1=REAL
            proba    = clf.predict_proba([cleaned])[0]
            ml_prob_fake = float(proba[0])                # probability of FAKE
        else:
            model, vectorizer = clf
            vec      = vectorizer.transform([cleaned])
            ml_pred  = model.predict(vec)[0]
            proba    = model.predict_proba(vec)[0]
            ml_prob_fake = float(proba[0])
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    preliminary = "Fake News" if ml_pred == 0 else "Real News"
    ml_conf     = float(max(proba))

    analysis    = analyze_text(raw_text, preliminary, ml_conf)
    n_flags     = len(analysis["flags"])
    n_positives = len(analysis["positive_signals"])

    fake_prob = ml_prob_fake

    fake_prob += n_flags     * 0.05
    fake_prob -= n_positives * 0.08

    if word_count < 15 and n_flags == 0:
        fake_prob *= 0.8

    fake_prob = max(0.0, min(1.0, fake_prob))

    FAKE_THRESHOLD      = 0.58   
    UNCERTAIN_THRESHOLD = 0.42   

    if fake_prob >= FAKE_THRESHOLD:
        result     = "Fake News"
        confidence = fake_prob
    elif fake_prob <= UNCERTAIN_THRESHOLD:
        result     = "Real News"
        confidence = 1.0 - fake_prob
    else:
        result     = "Uncertain"
        confidence = 1.0 - abs(fake_prob - 0.5) * 2

    analysis = analyze_text(raw_text, result, confidence)

    return jsonify({
        "prediction"      : result,
        "confidence"      : round(confidence, 4),
        "reasons"         : analysis["reasons"] + ["(Note: Fallback ML model used due to API limitation)"],
        "flags"           : analysis["flags"],
        "positive_signals": analysis["positive_signals"],
        "summary"         : analysis["summary"],
        "trust_score"     : analysis["trust_score"],
        "word_count"      : word_count,
    })


if __name__ == "__main__":
    app.run(debug=True)