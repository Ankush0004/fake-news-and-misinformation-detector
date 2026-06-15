# TruthScan AI: Advanced Hybrid Fake News Detector

This project provides a robust, state-of-the-art fake news detection system using a hybrid approach. It combines a **Machine Learning Pipeline (TF-IDF + LinearSVC)** for fast, offline analysis with a live **Generative AI Oracle (GPT-4 via g4f + DuckDuckGo)** for real-time fact-checking and comprehensive contextual understanding.

## 📊 Datasets Used
The system relies on the following datasets and continuous data streams to ensure maximum accuracy:

1. **Lutzhamel Fake News Corpus (Offline ML Model)**
   - **Link:** [fake_or_real_news.csv on GitHub](https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv)
   - **Description:** A well-balanced dataset of 6,335 political and global news articles cleanly labeled as FAKE or REAL. This forms the core of the offline Machine Learning pipeline.

2. **Generated "Common Sense" Fact Database**
   - **Link:** See local `utils/fact_builder.py`
   - **Description:** An artificially synthesized dataset of over 9,000 basic human facts (e.g., animal sounds, basic physics, geography) injected during training. This prevents the ML model from mislabeling simple factual statements (like *"the dog barks"*) as Fake News.

3. **Live Web Search (Primary System)**
   - **Link:** [DuckDuckGo Live Search API (ddgs)](https://github.com/deedy5/ddgs)
   - **Description:** Before analyzing any text, the system bypasses its training cutoff date by running a live web search for the exact query. This ensures it has up-to-the-minute context on breaking news and global events from today.

4. **OpenAI GPT-4 Knowledge Base (Primary System)**
   - **Link:** [g4f (GPT4Free) library](https://github.com/xtekky/gpt4free)
   - **Description:** The live DuckDuckGo results are fed into a free GPT-4 conversational AI agent, which generates the final verdict, confidence score, and detailed explanation.

## 🏗️ Project Structure
- `app.py`: Flask server providing the hybrid web interface and REST API (`/predict`).
- `train.py`: Training script that downloads the Lutzhamel dataset, combines it with the Fact Builder, and generates the offline `models/pipeline.pkl`.
- `utils/llm_oracle.py`: The live connection to ChatGPT and DuckDuckGo for real-time analysis.
- `static/css/style.css`: Modern, glassmorphism-inspired design.
- `templates/index.html`: Interactive web dashboard.

## 🚀 Setup & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
pip install -U g4f ddgs
```

### 2. Train the Offline Model
The offline model must be trained before the first use:
```bash
python train.py
```

### 3. Start the Web App
```bash
python app.py
```
Visit `http://127.0.0.1:5000` in your browser. Paste any news text, question, or fact to get an instant verdict with detailed reasoning.

## 📱 Mobile API (Flutter/Dart)
The backend is fully compatible with mobile clients. Use the following snippet in your Flutter app:

```dart
final response = await http.post(
  Uri.parse("http://your-server-ip:5000/predict"),
  headers: {"Content-Type": "application/json"},
  body: jsonEncode({"text": "Is the sun hot?"}),
);
```
