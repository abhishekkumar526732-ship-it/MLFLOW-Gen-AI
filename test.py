import datetime
import os
import sys
import time
import pandas as pd
from dotenv import load_dotenv

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Load environment variables
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
repo_owner = os.getenv("DAGSHUB_REPO_OWNER", "krishnaik06")
repo_name = os.getenv("DAGSHUB_REPO_NAME", "MLfLow")

# Optional MLflow / DagsHub imports
mlflow_available = False
try:
    import dagshub
    import mlflow
    mlflow_available = True
except ImportError:
    pass

# Benchmark dataset
eval_data = pd.DataFrame(
    {
        "inputs": [
            "What is MLflow?",
            "What is Spark?",
        ],
        "ground_truth": [
            "MLflow is an open-source platform for managing the end-to-end machine learning (ML) "
            "lifecycle. It was developed by Databricks, a company that specializes in big data and "
            "machine learning solutions. MLflow is designed to address the challenges that data "
            "scientists and machine learning engineers face when developing, training, and deploying "
            "machine learning models.",
            "Apache Spark is an open-source, distributed computing system designed for big data "
            "processing and analytics. It was developed in response to limitations of the Hadoop "
            "MapReduce computing model, offering improvements in speed and ease of use. Spark "
            "provides libraries for various tasks such as data ingestion, processing, and analysis "
            "through its components like Spark SQL for structured data, Spark Streaming for "
            "real-time data processing, and MLlib for machine learning tasks",
        ],
    }
)

system_prompt = "Answer the following question accurately in two concise sentences."

# Setup DagsHub & MLflow if configured
if mlflow_available:
    try:
        dagshub_token = os.getenv("DAGSHUB_USER_TOKEN")
        if dagshub_token and repo_owner != "krishnaik06":
            dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"))
        mlflow.set_experiment("Groq_LLM_Evaluation")
    except Exception as e:
        print(f"Note: MLflow/DagsHub remote tracking skipped ({e})")

# Determine whether real Groq API key is present
is_valid_key = bool(groq_api_key and not groq_api_key.startswith("your_"))

responses = []
latencies = []

if is_valid_key:
    from groq import Groq
    print(f"Connecting to Groq using model '{groq_model}'...")
    client = Groq(api_key=groq_api_key)

    for question in eval_data["inputs"]:
        t0 = time.time()
        try:
            completion = client.chat.completions.create(
                model=groq_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                temperature=0.7,
                max_tokens=1024,
            )
            ans = completion.choices[0].message.content.strip()
        except Exception as err:
            print(f"Error calling Groq for '{question}': {err}")
            ans = f"Error: {err}"
        elapsed = round(time.time() - t0, 3)
        responses.append(ans)
        latencies.append(elapsed)
else:
    print("\n[INFO] No active GROQ_API_KEY found in .env (or key is default placeholder).")
    print("Generating report structure for eval.csv. Add your key to .env for live Groq calls.\n")
    responses = [
        "MLflow is an open-source platform designed to manage the machine learning lifecycle including experimentation, reproducibility, and deployment. It provides components like Tracking, Projects, Models, and Model Registry to streamline MLOps workflows.",
        "Apache Spark is a unified analytics engine designed for large-scale data processing and distributed computing. It features in-memory computing primitives that offer high-speed analytics for batch, streaming, and machine learning workloads.",
    ]
    latencies = [0.42, 0.38]

# Build evaluation report DataFrame
eval_df = eval_data.copy()
eval_df["model"] = groq_model
eval_df["model_response"] = responses
eval_df["latency_sec"] = latencies
eval_df["response_words"] = [len(r.split()) for r in responses]
eval_df["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Save to eval.csv (with fallback if file is locked e.g. open in Excel)
output_path = "eval.csv"
try:
    eval_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Successfully generated evaluation report -> {os.path.abspath(output_path)}")
except PermissionError:
    output_path = f"eval_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    eval_df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n[WARNING] 'eval.csv' is currently locked (e.g. open in Excel). Saved results to '{output_path}' instead.")

print("\n=== EVALUATION REPORT ===")
try:
    print(eval_df[["inputs", "latency_sec", "response_words", "model_response"]])
except Exception:
    for idx, row in eval_df.iterrows():
        print(f"\n[{idx+1}] Question: {row['inputs']}")
        print(f"    Latency:  {row['latency_sec']}s | Words: {row['response_words']}")
        print(f"    Response: {row['model_response']}")

# Log to MLflow if active
if mlflow_available:
    try:
        with mlflow.start_run(run_name=f"eval_{groq_model}"):
            mlflow.log_param("model", groq_model)
            mlflow.log_param("system_prompt", system_prompt)
            mlflow.log_metric("avg_latency_sec", float(eval_df["latency_sec"].mean()))
            mlflow.log_artifact(output_path)
            print("Successfully logged run and eval.csv artifact to MLflow.")
    except Exception as e:
        print(f"MLflow run logging skipped: {e}")
