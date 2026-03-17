# 🧠 AI Data Intelligence Copilot

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://ai-data-intelligence-copilot.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Groq](https://img.shields.io/badge/Groq-LLaMA%203-F55036?style=for-the-badge)](https://console.groq.com)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-FF6B35?style=for-the-badge)](https://shap.readthedocs.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-AutoML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

---

Most analytics tools make you choose — you either get model training or explainability, either data profiling or natural language querying. This project builds all of it in one platform.

Upload any CSV, Excel, or JSON dataset and the system automatically profiles it, trains five ML models simultaneously, explains every prediction with SHAP, lets you ask questions about your data in plain English via RAG, simulates what-if scenarios with live sliders, and exports a trained model as a deployable FastAPI service — all without writing a single line of code.

The hybrid RAG system (Groq LLaMA 3 + FAISS vector store) grounds every answer in your actual dataset statistics rather than generic LLM knowledge. The SHAP explainability layer shows exactly which features drove each prediction. The one-click deployment tab generates a production-ready FastAPI app with Dockerfile that you can run anywhere.

## 🔗 Live Demo

**Try it here → [ai-data-intelligence-copilot.streamlit.app](https://ai-data-intelligence-copilot.streamlit.app/)**

Load the Telco Churn sample dataset, train all five models, ask the chat "Why are customers churning?" and download the deployment package — all in under 5 minutes.

---

## ✨ Features

- **Universal dataset upload** — CSV, Excel, JSON up to 200 MB with automatic schema detection
- **AI data cleaning** — detects missing values, outliers, duplicates, skewness; suggests and auto-applies fixes with LLM explanation per column
- **Feature engineering AI** — suggests interaction features, polynomial terms, ratio features, and target encoding; scores each by correlation gain
- **AutoML pipeline** — trains Logistic Regression, Random Forest, Gradient Boosting, XGBoost, and LightGBM simultaneously with a selectable model picker
- **Model leaderboard** — ranked by ROC-AUC (classification) or R² (regression) with training time comparison
- **SHAP explainability** — global feature importance, per-prediction waterfall charts, and dependence plots
- **RAG-powered chat** — ask natural language questions grounded in your actual dataset via Groq LLaMA 3 + FAISS
- **What-if simulation** — slider-driven feature overrides with live probability gauge and delta waterfall chart
- **Experiment tracker** — MLflow-style run logging with metric history and model comparison across sessions
- **Data drift detection** — KS-test (numeric) and chi-squared (categorical) comparison between training and new data
- **Auto data story** — LLM-generated narrative covering the business problem, findings, model performance, and recommended actions
- **PDF report export** — downloadable analytics report with cover page, stats, leaderboard, SHAP chart, and insights
- **One-click deployment** — generates FastAPI app + Dockerfile + docker-compose + test script as a downloadable zip

---

## ⚙️ How It Works

**AutoML Pipeline** (`src/automl/`)

Automatically preprocesses the dataset — median/mode imputation, MinMax scaling, ordinal encoding — then trains all registered models in a loop. Each model is evaluated on the same hold-out test set. The best model is selected by ROC-AUC (classification) or R² (regression) and persisted to `models/saved/best_model.pkl`.

**SHAP Explainability** (`src/explainability/shap_explainer.py`)

Uses `shap.TreeExplainer` for tree-based models with automatic fallback to `LinearExplainer` (Logistic Regression) and `KernelExplainer` (any model). Handles `feature_names_in_` conflicts by deep-copying and stripping attributes before passing to SHAP. Returns both global summary plots and per-instance waterfall charts.

**RAG Analytics** (`src/rag/`)

Converts dataset statistics, model results, and SHAP importances into text chunks. Embeds them using local `sentence-transformers/all-MiniLM-L6-v2` (no API key needed) and indexes into FAISS. At query time, retrieves the top-5 most relevant chunks and passes them to Groq LLaMA 3 as context. Every answer is grounded in your actual data — not generic LLM knowledge.

**What-if Simulation** (`src/simulation/whatif_engine.py`)

Applies feature overrides to a single test row and calls `model.predict_proba()` before and after. Returns original probability, new probability, delta, and a rule-based action recommendation. The delta waterfall chart shows the contribution of each changed feature.

**Data Drift Detection** (`src/drift_monitor/drift_detector.py`)

Runs Kolmogorov-Smirnov test for numeric columns and chi-squared test for categorical columns comparing training data distribution against uploaded new data. Flags columns with p-value below 0.05 as drifted and plots distribution overlays.

**One-Click Deployment** (`src/deployment/model_exporter.py`)

Sanitises all feature names to valid Python identifiers (handles `²`, spaces, special characters), generates a complete FastAPI app with `/predict` and `/predict_batch` endpoints, a Dockerfile, docker-compose.yml, auto-generated test script, and README. All packaged into a single downloadable zip.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.10+ | Core language |
| Streamlit | 15-tab interactive dashboard |
| Scikit-learn | Preprocessing + Logistic Regression + Random Forest |
| XGBoost / LightGBM | Gradient boosting models |
| SHAP | Model explainability |
| Groq (LLaMA 3.3 70B) | LLM for RAG chat, insights, data story |
| FAISS | Vector similarity search for RAG |
| sentence-transformers | Local embeddings (no API key needed) |
| LangChain | RAG pipeline orchestration |
| Plotly | Interactive charts and visualisations |
| ReportLab | PDF analytics report generation |
| FastAPI + Uvicorn | Generated model serving endpoint |
| SciPy | Statistical drift tests (KS + chi-squared) |
| pandas-profiling | Automated dataset profiling |

---

## 📊 Results — Telco Churn Dataset

| Model | ROC-AUC | F1 | Training Time |
|---|---|---|---|
| LightGBM | **0.847** | 0.631 | 1.2s |
| XGBoost | 0.841 | 0.624 | 0.9s |
| Random Forest | 0.836 | 0.618 | 2.1s |
| Gradient Boosting | 0.829 | 0.611 | 3.4s |
| Logistic Regression | 0.802 | 0.589 | 0.1s |

> Results on the included 1,000-row Telco Churn sample. Full 7,043-row dataset available from [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn).

**What this tells a recruiter:**
This platform does everything a production ML system needs — data quality, feature engineering, model selection, explainability, monitoring, and deployment. It is not a notebook. It is a deployable product with a proper module structure, unit tests, and one-click export.

---

## 📁 Project Structure

```
ai-data-intelligence-copilot/
│
├── dashboard/
│   └── app.py                          # 15-tab Streamlit application (1,300+ lines)
│
├── src/
│   ├── ingestion/
│   │   ├── uploader.py                 # CSV / Excel / JSON parser + validation
│   │   └── schema_detector.py          # Auto column type + target detection
│   │
│   ├── cleaning/
│   │   └── data_cleaner.py             # Missing values, outliers, duplicates, skewness
│   │
│   ├── feature_engineering/
│   │   └── feature_engineer.py         # Interactions, polynomial, ratio, target encoding
│   │
│   ├── profiling/
│   │   └── profiler.py                 # Correlation heatmap, distributions, stats
│   │
│   ├── automl/
│   │   ├── preprocessor.py             # Imputation + scaling + encoding + train/test split
│   │   ├── trainer.py                  # Multi-model training loop
│   │   ├── evaluator.py                # Metrics + ROC + confusion matrix + leaderboard
│   │   └── model_selector.py           # Best model selection + rationale
│   │
│   ├── explainability/
│   │   └── shap_explainer.py           # TreeExplainer / LinearExplainer / KernelExplainer
│   │
│   ├── rag/
│   │   ├── embedder.py                 # Dataset facts → text corpus
│   │   ├── vector_store.py             # FAISS index + local embeddings fallback
│   │   └── qa_chain.py                 # Groq LLaMA 3 + retrieval chain
│   │
│   ├── insights/
│   │   └── insight_generator.py        # LLM insights + rule-based fallback
│   │
│   ├── simulation/
│   │   └── whatif_engine.py            # Slider-driven re-scoring + gauge charts
│   │
│   ├── recommendations/
│   │   └── action_engine.py            # Risk-based action recommendations
│   │
│   ├── experiment_tracker/
│   │   └── tracker.py                  # MLflow-style run logging (JSON persistence)
│   │
│   ├── drift_monitor/
│   │   └── drift_detector.py           # KS-test + chi-squared distribution comparison
│   │
│   └── deployment/
│       └── model_exporter.py           # FastAPI + Dockerfile + test script generator
│
├── utils/
│   ├── config.py                       # Environment variables + path management
│   ├── gemini_client.py                # Unified LLM client (Groq primary, Gemini fallback)
│   ├── logger.py                       # Structured logging
│   └── pdf_generator.py                # ReportLab multi-page PDF report
│
├── data/
│   ├── sample_datasets/
│   │   ├── telco_churn.csv             # 1,000-row churn classification dataset
│   │   ├── titanic.csv                 # 891-row survival classification dataset
│   │   └── boston_housing.csv          # 506-row regression dataset
│   └── uploads/                        # Runtime upload directory
│
├── models/saved/                       # Persisted best model (.pkl)
├── notebooks/                          # 3 Jupyter notebooks (EDA, AutoML, SHAP)
├── tests/                              # Unit tests (pytest)
├── requirements.txt
├── packages.txt                        # System deps for Streamlit Cloud
├── Dockerfile
├── .env.example
└── README.md
```

---

## 🚀 Installation

**Clone the repository**

```bash
git clone https://github.com/Shraddhaaa05/ai-data-intelligence-copilot.git
cd ai-data-intelligence-copilot
```

**Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

**Install dependencies**

```bash
pip install -r requirements.txt
```

**Configure environment**

```bash
cp .env.example .env
```

Open `.env` and add your Groq API key:

```
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

Get a free key at **https://console.groq.com** → API Keys → Create key.

---

## ▶️ How to Run

**Launch the app:**

```bash
streamlit run dashboard/app.py
```

The app opens at `http://localhost:8501`.

**Recommended workflow:**

1. **📁 Dataset** — Upload your CSV or load a sample dataset
2. **🧹 Cleaning** — Scan for issues → Auto-Clean → Use Cleaned Dataset
3. **⚙️ Features** — Analyse → select features → Use Engineered Dataset
4. **🤖 AutoML** — Select target → choose models → Train
5. **🏆 Leaderboard** — Compare all models
6. **🔍 SHAP** — See what drives predictions
7. **💡 Insights** — AI-generated business findings
8. **💬 Chat** — Ask "Why are customers churning?"
9. **🎛️ What-if** — Test interventions with sliders
10. **🚀 Deploy** — Download model as FastAPI service

---

## ☁️ Streamlit Cloud Deployment

1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select repo and set main file path: `dashboard/app.py`
4. Under **Secrets**, add:

```toml
GROQ_API_KEY = "gsk_your_key_here"
GROQ_MODEL = "llama-3.3-70b-versatile"
```

5. Click Deploy

## 🐳 Docker Deployment

```bash
docker build -t ai-copilot .
docker run -p 8501:8501 -e GROQ_API_KEY=gsk_... ai-copilot
```

---

## 🔮 Future Improvements

- [ ] Time series forecasting support (Prophet, ARIMA)
- [ ] Hyperparameter tuning with Optuna
- [ ] Multi-dataset comparison view
- [ ] Model versioning with DVC
- [ ] Real-time prediction monitoring dashboard
- [ ] Hugging Face model integration for NLP datasets
- [ ] Auto-generated SQL queries from natural language

---

## 👩‍💻 Author

**Shraddha Gidde**
B.Tech — Artificial Intelligence and Data Science
MIT World Peace University, Pune

[![Portfolio](https://img.shields.io/badge/Portfolio-shraddha--gidde.netlify.app-2563EB?style=flat-square)](https://shraddha-gidde.netlify.app)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/shraddha-gidde-063506242)
[![GitHub](https://img.shields.io/badge/GitHub-shraddha--gidde-181717?style=flat-square&logo=github)](https://github.com/Shraddhaaa05)

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
