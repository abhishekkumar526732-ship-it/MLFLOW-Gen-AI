import datetime
import os
import time
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Groq Q&A Assistant (GPT-120B)",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        background: linear-gradient(90deg, #f97316, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 1.2rem;
    }
    .model-badge {
        display: inline-block;
        background-color: #1e293b;
        color: #38bdf8;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #334155;
    }
    .metric-pill {
        display: inline-block;
        background-color: #0f172a;
        color: #10b981;
        padding: 0.15rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        border: 1px solid #1e293b;
        margin-top: 0.35rem;
    }
    .dagshub-badge {
        display: inline-block;
        background-color: #132f22;
        color: #34d399;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.8rem;
        border: 1px solid #059669;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Helper: Initialize DagsHub MLflow remote tracking
@st.cache_resource
def setup_dagshub_tracking():
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER", "abhishekkumar526732-ship-it")
    repo_name = os.getenv("DAGSHUB_REPO_NAME", "MLFLOW-Gen-AI")
    token = os.getenv("DAGSHUB_USER_TOKEN")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow")

    if token:
        try:
            import dagshub
            import dagshub.auth
            import mlflow
            dagshub.auth.add_app_token(token)
            os.environ["MLFLOW_TRACKING_USERNAME"] = repo_owner
            os.environ["MLFLOW_TRACKING_PASSWORD"] = token
            dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
            mlflow.set_tracking_uri(tracking_uri)
            return True, tracking_uri
        except Exception as err:
            return False, str(err)
    return False, "DAGSHUB_USER_TOKEN not provided in .env"

dagshub_ready, dagshub_info = setup_dagshub_tracking()

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key Handling
    env_api_key = os.getenv("GROQ_API_KEY", "")
    api_key = st.text_input(
        "Groq API Key",
        value=env_api_key,
        type="password",
        help="Reads from .env by default or enter your Groq API key here (https://console.groq.com/keys)",
    )

    # Model Selection - Default to openai/gpt-oss-120b
    SUPPORTED_MODELS = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "Custom Model",
    ]
    
    selected_model_option = st.selectbox(
        "Select Model",
        SUPPORTED_MODELS,
        index=0,
        help="openai/gpt-oss-120b is the flagship 120B parameter reasoning model on Groq",
    )
    
    if selected_model_option == "Custom Model":
        model_name = st.text_input("Enter Model ID", value="openai/gpt-oss-120b")
    else:
        model_name = selected_model_option

    st.markdown("---")
    st.subheader("Model Parameters")
    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.05)
    max_tokens = st.slider("Max Completion Tokens", min_value=128, max_value=8192, value=2048, step=128)

    system_prompt = st.text_area(
        "System Prompt",
        value="You are a helpful, direct, and intelligent AI assistant powered by Groq high-speed inference.",
        height=90,
    )

    st.markdown("---")
    # DagsHub & MLflow Tracking
    st.subheader("☁️ DagsHub MLflow")
    if dagshub_ready:
        st.markdown(f'<div class="dagshub-badge">🌐 DagsHub Connected</div>', unsafe_allow_html=True)
        st.caption(f"[View DagsHub Experiments]({dagshub_info})")
    else:
        st.warning(f"DagsHub Status: {dagshub_info}")

    enable_mlflow = st.checkbox("Track queries to DagsHub MLflow", value=dagshub_ready)
    
    # Chat Controls
    st.markdown("---")
    st.subheader("Data & Session")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Export Chat to eval.csv
    if st.button("📥 Export Current Chat to eval.csv", use_container_width=True):
        if "messages" in st.session_state and st.session_state.messages:
            records = []
            for i in range(0, len(st.session_state.messages) - 1, 2):
                u = st.session_state.messages[i]
                a = st.session_state.messages[i+1] if i+1 < len(st.session_state.messages) else {}
                records.append({
                    "inputs": u.get("content", ""),
                    "ground_truth": "N/A (Interactive Chat)",
                    "model": a.get("metrics", {}).get("model", model_name),
                    "model_response": a.get("content", ""),
                    "latency_sec": a.get("metrics", {}).get("elapsed_time", 0.0),
                    "response_words": len(a.get("content", "").split()),
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })
            chat_df = pd.DataFrame(records)
            chat_df.to_csv("eval.csv", index=False)
            st.success("Current conversation saved to eval.csv!")
        else:
            st.warning("No messages in chat history to export.")

# Main Header
st.markdown('<div class="main-header">⚡ Groq Q&A Assistant</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="sub-header">Ultra-fast inference &bull; Active Model: <span class="model-badge">{model_name}</span></div>',
    unsafe_allow_html=True,
)

# Navigation Tabs
tab_chat, tab_eval = st.tabs(["💬 Interactive Q&A", "📊 Evaluation Report (eval.csv)"])

# TAB 1: Chat Q&A
with tab_chat:
    # Initialize Session State Messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "metrics" in msg and msg["metrics"]:
                st.markdown(
                    f'<span class="metric-pill">⏱️ {msg["metrics"].get("elapsed_time", 0):.2f}s | {msg["metrics"].get("model", "")}</span>',
                    unsafe_allow_html=True,
                )

    # Check for API key
    if not api_key:
        st.info("👋 Welcome! Please enter your **Groq API Key** in the sidebar or save `GROQ_API_KEY=gsk_...` in your `.env` file to begin.")
    else:
        # Initialize Groq client
        try:
            client = Groq(api_key=api_key)
        except Exception as e:
            st.error(f"Failed to initialize Groq client: {e}")
            st.stop()

        # User Input
        user_question = st.chat_input("Ask a question...")

        if user_question:
            st.session_state.messages.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            # Build messages payload for Groq
            messages_payload = [{"role": "system", "content": system_prompt}]
            for m in st.session_state.messages:
                messages_payload.append({"role": m["role"], "content": m["content"]})

            # Generate streaming response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                start_time = time.time()

                try:
                    stream = client.chat.completions.create(
                        model=model_name,
                        messages=messages_payload,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True,
                    )

                    for chunk in stream:
                        delta = chunk.choices[0].delta.content or ""
                        full_response += delta
                        message_placeholder.markdown(full_response + "▌")

                    elapsed_time = round(time.time() - start_time, 3)
                    message_placeholder.markdown(full_response)
                    st.markdown(
                        f'<span class="metric-pill">⏱️ {elapsed_time:.2f}s | Model: {model_name}</span>',
                        unsafe_allow_html=True,
                    )

                    # Save assistant message to session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "metrics": {
                            "elapsed_time": elapsed_time,
                            "model": model_name,
                        },
                    })

                    # Optional DagsHub MLflow tracking
                    if enable_mlflow and dagshub_ready:
                        try:
                            import mlflow
                            mlflow.set_experiment("Groq_QA_App")
                            with mlflow.start_run(run_name=f"chat_{int(time.time())}"):
                                mlflow.log_param("model", model_name)
                                mlflow.log_param("temperature", temperature)
                                mlflow.log_param("max_tokens", max_tokens)
                                mlflow.log_metric("latency_sec", elapsed_time)
                                mlflow.log_text(user_question, "question.txt")
                                mlflow.log_text(full_response, "response.txt")
                        except Exception as mlflow_err:
                            st.sidebar.warning(f"MLflow logging error: {mlflow_err}")

                except Exception as e:
                    st.error(f"Error calling Groq API: {e}")

# TAB 2: Evaluation Report (eval.csv)
with tab_eval:
    st.subheader("📋 Evaluation Dataset & Model Performance Report")
    
    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.write("Generate benchmark metrics using Groq's `openai/gpt-oss-120b`, save to `eval.csv`, and track in DagsHub MLflow.")
        if dagshub_ready:
            st.markdown(f"☁️ **Remote Tracking**: [{dagshub_info}]({dagshub_info})")
    with col_b:
        run_bench = st.button("🚀 Run Evaluation Benchmark Now", use_container_width=True)

    if run_bench:
        if not api_key:
            st.error("Please provide a Groq API Key in the sidebar to run the live benchmark.")
        else:
            with st.spinner(f"Evaluating benchmark questions using {model_name}..."):
                try:
                    eval_benchmark_data = [
                        {
                            "inputs": "What is MLflow?",
                            "ground_truth": "MLflow is an open-source platform for managing the end-to-end machine learning lifecycle.",
                        },
                        {
                            "inputs": "What is Spark?",
                            "ground_truth": "Apache Spark is an open-source distributed computing framework designed for fast, large-scale data processing.",
                        },
                    ]
                    bench_client = Groq(api_key=api_key)
                    rows = []
                    for item in eval_benchmark_data:
                        t0 = time.time()
                        comp = bench_client.chat.completions.create(
                            model=model_name,
                            messages=[
                                {"role": "system", "content": "Answer the following question accurately in two concise sentences."},
                                {"role": "user", "content": item["inputs"]},
                            ],
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                        ans = comp.choices[0].message.content.strip()
                        dur = round(time.time() - t0, 3)
                        rows.append({
                            "inputs": item["inputs"],
                            "ground_truth": item["ground_truth"],
                            "model": model_name,
                            "model_response": ans,
                            "latency_sec": dur,
                            "response_words": len(ans.split()),
                            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        })
                    new_df = pd.DataFrame(rows)
                    
                    # Safe save handling
                    try:
                        new_df.to_csv("eval.csv", index=False)
                    except PermissionError:
                        fallback_file = f"eval_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                        new_df.to_csv(fallback_file, index=False)
                        st.warning(f"eval.csv is open in another program. Saved to {fallback_file}")

                    # Log to DagsHub MLflow
                    if dagshub_ready:
                        try:
                            import mlflow
                            mlflow.set_experiment("Groq_LLM_Evaluation")
                            with mlflow.start_run(run_name=f"benchmark_{model_name.replace('/', '_')}"):
                                mlflow.log_param("model", model_name)
                                mlflow.log_param("temperature", temperature)
                                mlflow.log_metric("avg_latency_sec", float(new_df["latency_sec"].mean()))
                                mlflow.log_artifact("eval.csv")
                            st.success(f"Benchmark completed & logged to DagsHub! [View in DagsHub MLflow]({dagshub_info})")
                        except Exception as m_err:
                            st.info(f"Local benchmark saved; MLflow sync note: {m_err}")
                    else:
                        st.success("Benchmark completed! Results updated in eval.csv.")

                except Exception as err:
                    st.error(f"Benchmark failed: {err}")

    # View existing eval.csv
    if os.path.exists("eval.csv"):
        try:
            df_eval = pd.read_csv("eval.csv")
            
            # Key Metrics
            m1, m2, m3 = st.columns(3)
            with m1:
                m1.metric("Evaluated Rows", len(df_eval))
            with m2:
                if "latency_sec" in df_eval.columns:
                    m2.metric("Avg Latency", f"{df_eval['latency_sec'].mean():.2f}s")
            with m3:
                if "response_words" in df_eval.columns:
                    m3.metric("Avg Words", f"{df_eval['response_words'].mean():.1f}")

            # Data Table
            st.dataframe(df_eval, use_container_width=True)

            # Download Button
            csv_data = df_eval.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download eval.csv",
                data=csv_data,
                file_name="eval.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except Exception as load_err:
            st.error(f"Error loading eval.csv: {load_err}")
    else:
        st.info("No `eval.csv` found yet. Click 'Run Evaluation Benchmark Now' or run `python test.py` in your terminal to create it.")
