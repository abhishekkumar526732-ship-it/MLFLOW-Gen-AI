# MLFLOW Gen-AI Assistant

A fast Q&A web application built with **Streamlit** and powered by **Groq** using the **`openai/gpt-oss-120b`** model, integrated with **DagsHub MLflow** for experiment tracking and evaluation reporting.

---

## ⚡ Features

- **Ultra-Fast Inference**: Streaming responses via Groq LPU.
- **Flagship 120B Model**: Powered by `openai/gpt-oss-120b` with multi-model support.
- **Interactive UI**: Clean Streamlit chat interface with latency metrics.
- **Evaluation & Reporting**: Benchmark runner with automated exports to `eval.csv`.
- **DagsHub MLflow Tracking**: Remote cloud experiment tracking and artifact logging.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### 2. Configure Environment (`.env`)
Create a `.env` file from `.env.example`:
```dotenv
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b

DAGSHUB_REPO_OWNER=abhishekkumar526732-ship-it
DAGSHUB_REPO_NAME=MLFLOW-Gen-AI
DAGSHUB_USER_TOKEN=your_dagshub_token
```

### 3. Run the Streamlit App
```bash
python -m streamlit run app.py
```
Open **http://localhost:8501** in your browser.

---

## 🧪 Run Benchmark & Evaluation
Generate the `eval.csv` report and log metrics directly to DagsHub MLflow:
```bash
python test.py
```

---

## 📁 Project Structure

```text
├── app.py            # Streamlit Q&A application & evaluation UI
├── test.py           # CLI benchmark evaluation & DagsHub logging script
├── eval.csv          # Evaluation results and performance metrics
├── requirements.txt  # Project Python dependencies
├── .env.example      # Environment variables template
└── README.md         # Documentation
```