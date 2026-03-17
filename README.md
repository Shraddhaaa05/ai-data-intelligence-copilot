# AI Data Intelligence Copilot

> **AutoML + Explainable AI + RAG Analytics Platform**

Upload any dataset. Get automated ML models, SHAP explanations, AI-generated business insights, and natural language Q&A — all in one interactive dashboard. No ML expertise required.

---

## Features

- **Universal dataset support** — CSV, Excel, JSON up to 200 MB
- **Auto schema detection** — numeric/categorical/datetime columns, missing values, target inference
- **AutoML pipeline** — trains 4 models simultaneously, ranks by ROC-AUC or R²
- **Model leaderboard** — compare Logistic Regression, Random Forest, XGBoost, LightGBM
- **SHAP explainability** — global summary plots + per-prediction waterfall charts
- **RAG-powered chat** — ask "Why are customers churning?" in plain English
- **What-if simulation** — change feature values with sliders and see live probability changes
- **AI business insights** — LLM-generated findings like "Month-to-month customers churn 3.2x more"
- **PDF report export** — downloadable analytics report with charts and model results

---

## Architecture

```
Ingestion → Profiling → AutoML → SHAP → RAG + What-if → Dashboard
```

| Layer | Components |
|-------|-----------|
| Ingestion | File upload, validation, schema detection, target inference |
| Profiling | ydata-profiling, correlation heatmap, distributions |
| AutoML | Preprocessing, 4-model training loop, leaderboard |
| Explainability | SHAP TreeExplainer, summary + waterfall plots |
| RAG | FAISS vector store, OpenAI embeddings, LangChain QA chain |
| Insights | LLM-generated summaries and business findings |
| Simulation | Slider-driven what-if engine with live re-scoring |
| Dashboard | 10-tab Streamlit app |

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/yourname/ai-data-intelligence-copilot
cd ai-data-intelligence-copilot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 4. Run the app
streamlit run dashboard/app.py
```

Open http://localhost:8501 in your browser.

---

## Sample Datasets

Three sample datasets are included in `data/sample_datasets/`:

| Dataset | Task | Rows | Target |
|---------|------|------|--------|
| Telco Customer Churn | Classification | 7,043 | Churn |
| Titanic Survival | Classification | 891 | Survived |
| Boston Housing | Regression | 506 | MEDV |

---

## Project Structure

```
ai-data-intelligence-copilot/
├── data/
│   ├── sample_datasets/
│   └── uploads/
├── notebooks/
├── src/
│   ├── ingestion/
│   ├── profiling/
│   ├── automl/
│   ├── explainability/
│   ├── rag/
│   ├── insights/
│   ├── simulation/
│   └── recommendations/
├── models/saved/
├── dashboard/app.py
├── utils/
└── tests/
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Your OpenAI API key (required for RAG + insights) |
| `OPENAI_MODEL` | LLM model name (default: gpt-3.5-turbo) |
| `EMBEDDING_MODEL` | Embedding model (default: text-embedding-ada-002) |
| `MAX_UPLOAD_SIZE_MB` | Max file size in MB (default: 200) |

---

## Deployment

### Streamlit Cloud (Free)
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select repo and `dashboard/app.py`
4. Add `OPENAI_API_KEY` under Secrets
5. Click Deploy

### Docker
```bash
docker build -t ai-copilot .
docker run -p 8501:8501 -e OPENAI_API_KEY=sk-... ai-copilot
```

### AWS EC2
```bash
sudo apt update && sudo apt install python3-pip -y
git clone https://github.com/yourname/ai-data-intelligence-copilot
cd ai-data-intelligence-copilot
pip install -r requirements.txt
nohup streamlit run dashboard/app.py --server.port 8501 &
```

---

## Tech Stack

Python · Streamlit · scikit-learn · XGBoost · LightGBM · SHAP · LangChain · FAISS · OpenAI · Plotly · ReportLab · ydata-profiling

---

## Future Improvements

- [ ] Time series forecasting support (Prophet, ARIMA)
- [ ] Multi-dataset comparison view
- [ ] AutoML hyperparameter tuning with Optuna
- [ ] Model drift monitoring dashboard
- [ ] Hugging Face model support for NLP datasets
- [ ] Export trained model as REST API endpoint
# ai-data-intelligence-copilot
