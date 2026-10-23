# MLFLOW-Gen-AI: Groq Q&A Assistant (GPT-OSS 120B)

A fast, interactive Q&A assistant built with **Streamlit** and powered by **Groq** using the flagship **`openai/gpt-oss-120b`** model (with support for other Groq models like LLaMA 3.3 70B). Includes optional MLflow experiment tracking.

---

## ⚡ Features

- **Groq LPU Acceleration**: High-speed, streaming token generation.
- **Flagship 120B Model**: Default support for `openai/gpt-oss-120b` (plus `openai/gpt-oss-20b`, `llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `mixtral-8x7b-32768`, or custom model IDs).
- **Streamlit Chat UI**: Native streaming chat interface with response latency metrics.
- **Sidebar Controls**: Real-time temperature, max token, and system prompt tuning.
- **MLflow Tracking**: Optional toggle to log runs, questions, responses, and metrics directly to MLflow.

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 2. Configure Environment (`.env`)
Create or edit your `.env` file with your Groq API key:
```dotenv
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
```
*(You can get a free Groq API key at [console.groq.com/keys](https://console.groq.com/keys))*

### 3. Launch the Streamlit App
```bash
python -m streamlit run app.py
```
Or if `streamlit` is in your system PATH:
```bash
streamlit run app.py
```

---

## 🧪 Testing with MLflow
To run the automated evaluation script:
```bash
python test.py
```