"""
AI Data Intelligence Copilot v3.1
Fixes: AutoML model picker, feature name sanitisation in deploy,
correlation chart, drift explanation, NL queries, deploy shows all models.
"""
import os, sys, warnings, json, re
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="AI Data Intelligence Copilot",
                   page_icon="🧠", layout="wide",
                   initial_sidebar_state="expanded")

from src.ingestion.uploader import load_dataset, UploadValidationError
from src.ingestion.schema_detector import detect_schema
from src.profiling.profiler import profile_dataset
from src.automl.preprocessor import preprocess
from src.automl.trainer import train_all, CLASSIFIERS, REGRESSORS
from src.automl.evaluator import (plot_confusion_matrix, plot_roc_curve,
    plot_predicted_vs_actual, plot_residuals, plot_leaderboard)
from src.automl.model_selector import get_selection_rationale
from src.explainability.shap_explainer import SHAPExplainer
from src.insights.insight_generator import generate_business_insights, generate_dataset_summary
from src.simulation.whatif_engine import simulate, plot_probability_gauge, plot_delta_waterfall
from src.recommendations.action_engine import generate_actions, batch_recommendations
from src.cleaning.data_cleaner import DataCleaner
from src.feature_engineering.feature_engineer import FeatureEngineer
from src.experiment_tracker.tracker import ExperimentTracker
from src.drift_monitor.drift_detector import DriftDetector
from src.deployment.model_exporter import ModelExporter
from utils.config import GROQ_API_KEY, GOOGLE_API_KEY, GEMINI_MODEL, GROQ_MODEL, GROQ_API_KEY, GROQ_MODEL
from utils.logger import get_logger

logger = get_logger(__name__)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""<style>
.main-header{background:linear-gradient(135deg,#1a56db 0%,#0e9f6e 100%);
  padding:1.5rem 2rem;border-radius:12px;margin-bottom:1rem;color:white;}
.main-header h1{color:white;margin:0;font-size:1.8rem;}
.main-header p{color:rgba(255,255,255,.85);margin:.2rem 0 0;font-size:.95rem;}
.lemann{background:#fefce8;border:2px solid #fbbf24;border-radius:12px;
  padding:1.2rem 1.5rem;margin:.8rem 0;}
.lemann h4{color:#92400e;margin:0 0 .5rem;font-size:1rem;font-weight:700;text-transform:uppercase;}
.lemann li{color:#1e2a3b;margin-bottom:.35rem;font-size:.94rem;line-height:1.5;}
.lemann .hl{color:#b45309;font-weight:700;}
.card-green{background:#f0fdf4;border-left:4px solid #10b981;
  border-radius:0 8px 8px 0;padding:.6rem 1rem;margin-bottom:.4rem;}
.card-blue{background:#eff6ff;border-left:4px solid #3b82f6;
  border-radius:0 8px 8px 0;padding:.6rem 1rem;margin-bottom:.4rem;}
.card-amber{background:#fff7ed;border-left:4px solid #f97316;
  border-radius:0 8px 8px 0;padding:.6rem 1rem;margin-bottom:.4rem;}
.card-red{background:#fef2f2;border-left:4px solid #ef4444;
  border-radius:0 8px 8px 0;padding:.6rem 1rem;margin-bottom:.4rem;}
.card-purple{background:#f5f3ff;border-left:4px solid #8b5cf6;
  border-radius:0 8px 8px 0;padding:.6rem 1rem;margin-bottom:.4rem;}
.badge{background:#dcfce7;color:#166534;border-radius:20px;
  padding:.15rem .6rem;font-size:.72rem;font-weight:600;}
.model-card{background:#f8faff;border:1px solid #e0e7ff;border-radius:10px;
  padding:.8rem 1rem;margin-bottom:.5rem;}
.deploy-section{background:#f0fdf4;border:2px solid #10b981;border-radius:12px;
  padding:1.2rem;margin:.8rem 0;}
div[data-testid="stSidebar"]{background:#f8faff;}
.stTabs [data-baseweb="tab"]{font-size:.8rem;padding:.35rem .7rem;}
</style>""", unsafe_allow_html=True)

# ── helpers ───────────────────────────────────────────────────────────────────
def ss(k, d=None):  return st.session_state.get(k, d)
def sset(**kw):
    for k, v in kw.items(): st.session_state[k] = v
def safe_df(key):
    v = ss(key); return v if v is not None else pd.DataFrame()
def safe_list(key):
    v = ss(key); return v if v is not None else []

if "tracker" not in st.session_state:
    st.session_state["tracker"] = ExperimentTracker()

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 12px; padding: 20px; margin-bottom: 20px; color: white;
                text-align: center;">
        <h1 style="margin: 0; font-size: 32px;">🧠</h1>
        <h3 style="margin: 10px 0 0 0;">AI Copilot</h3>
        <p style="margin: 5px 0 0 0; opacity: 0.9; font-size: 12px;">Intelligent AutoML Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Status Metrics
    col1, col2 = st.columns(2)
    with col1:
        if ss("df") is not None:
            st.metric("📊 Rows", f"{ss('df').shape[0]:,}")
        else:
            st.metric("📊 Rows", "—")
    with col2:
        if ss("df") is not None:
            st.metric("📋 Cols", ss('df').shape[1])
        else:
            st.metric("📋 Cols", "—")
    
    if ss("results"):
        col1, col2 = st.columns(2)
        with col1:
            best = ss("results")["best_model"]
            schema_s = ss("schema")
            mk = "roc_auc" if schema_s and schema_s.problem_type == "classification" else "r2"
            st.metric("🏆 Score", f"{best.get(mk, 'N/A')}")
        with col2:
            st.metric("🎯 Models", len(ss("results")["leaderboard"]))
    
    st.markdown("---")
    
    # Quick Stats
    st.markdown("### 📈 Pipeline")
    steps = []
    if ss("df") is not None:
        steps.append("✅ Data Loaded")
    if ss("profiling") is not None:
        steps.append("✅ Profiled")
    if ss("results") is not None:
        steps.append("✅ Trained")
    
    for step in steps:
        st.markdown(f"- {step}")
    
    st.markdown("---")
    st.caption("Version 3.1 • Powered by Groq & Streamlit")

# ── header ────────────────────────────────────────────────────────────────────
st.markdown("""<div class="main-header">
  <h1>🧠 AI Data Intelligence Copilot</h1>
  <p>AutoML · Explainable AI · Data Cleaning · Feature Engineering · Experiment Tracking · Drift Monitoring · One-Click Deployment</p>
</div>""", unsafe_allow_html=True)

tabs = st.tabs([
    "📁 Dataset","🧹 Cleaning","⚙️ Features","📊 Profiling","🤖 AutoML",
    "🏆 Leaderboard","🧪 Tracker","🎯 Predictions","📈 Diagnostics","🔍 SHAP",
    "💡 Insights","📖 Story","💬 Chat","🎛️ What-if","🚀 Deploy",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 — DATASET
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.header("Dataset Upload")
    c1, c2 = st.columns([2, 1])
    with c1:
        uploaded = st.file_uploader("CSV / Excel / JSON", type=["csv","xlsx","xls","json"])
    with c2:
        st.markdown("**Sample datasets**")
        sample = st.selectbox("Sample dataset", ["None","Telco Churn","Titanic","Boston Housing"],
                              label_visibility="collapsed")
        if st.button("Load sample", width="stretch"):
            paths = {"Telco Churn":"data/sample_datasets/telco_churn.csv",
                     "Titanic":"data/sample_datasets/titanic.csv",
                     "Boston Housing":"data/sample_datasets/boston_housing.csv"}
            p = paths.get(sample,"")
            if p and os.path.exists(p):
                sset(df=pd.read_csv(p), dataset_name=sample,
                     schema=None, profiling=None, results=None,
                     explainer=None, ai_insights=None, ai_summary=None,
                     shap_importance_df=None, rag_chain=None,
                     cleaning_issues=None, fe_suggestions=None, data_story=None)
                st.rerun()
            else:
                st.warning(f"File not found: {p}")

    df = None
    if uploaded:
        try:
            df = load_dataset(uploaded)
            sset(df=df, dataset_name=uploaded.name,
                 schema=None, profiling=None, results=None,
                 explainer=None, ai_insights=None, ai_summary=None,
                 shap_importance_df=None, rag_chain=None,
                 cleaning_issues=None, fe_suggestions=None, data_story=None)
        except UploadValidationError as e:
            st.error(str(e))

    df = ss("df")
    if df is not None:
        if ss("schema") is None:
            sset(schema=detect_schema(df))
        schema = ss("schema")
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", df.shape[1])
        c3.metric("Numeric", len(schema.numeric_cols))
        c4.metric("Categorical", len(schema.categorical_cols))
        c5.metric("Missing cols", len(schema.missing_cols))

        top_t = schema.suggested_targets[0] if schema.suggested_targets else "unknown"
        vc_str = ""
        try:
            vc = df[top_t].value_counts(normalize=True)*100
            vc_str = " / ".join(f"{k}: {v:.0f}%" for k,v in vc.items())
        except Exception: pass

        st.markdown(f"""<div class="lemann"><h4>🔎 Dataset at a glance</h4><ul>
<li><span class="hl">{df.shape[0]:,} records</span>, {df.shape[1]} columns
({len(schema.numeric_cols)} numeric, {len(schema.categorical_cols)} categorical)</li>
<li>Recommended target: <span class="hl">'{top_t}'</span> — {schema.problem_type}
{f'| {vc_str}' if vc_str else ''}</li>
<li>{'⚠️ Missing in: ' + ', '.join(schema.missing_cols[:4]) if schema.missing_cols else '✅ No missing values'}</li>
<li><strong>Next:</strong> 🧹 Cleaning → ⚙️ Features → 🤖 AutoML</li>
</ul></div>""", unsafe_allow_html=True)

        col_t, col_s = st.columns([2,1])
        with col_t:
            type_df = pd.DataFrame(
                [{"Column": c, "Type": "Numeric", "Unique": df[c].nunique()} for c in schema.numeric_cols] +
                [{"Column": c, "Type": "Categorical", "Unique": df[c].nunique()} for c in schema.categorical_cols])
            st.dataframe(type_df, width="stretch", height=200)
        with col_s:
            st.markdown("**Suggested targets**")
            for t in schema.suggested_targets[:5]:
                nu = df[t].nunique()
                icon = "🟢" if nu==2 else "🔵" if nu<=10 else "🟡"
                st.markdown(f"{icon} `{t}` — {nu} unique")
        st.dataframe(df.head(100), width="stretch")
    else:
        st.info("Upload a dataset or load a sample to begin.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CLEANING
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("🧹 Data Cleaning Assistant")
    if ss("df") is None:
        st.info("Upload a dataset first.")
    else:
        df, schema = ss("df"), ss("schema")
        cleaner = DataCleaner(df, schema)

        c1, c2 = st.columns([1,1])
        with c1:
            if st.button("🔍 Scan for Issues", type="primary", width="stretch"):
                with st.spinner("Scanning dataset..."):
                    issues = cleaner.detect_all_issues()
                    explanation = cleaner.llm_explain(issues)
                    sset(cleaning_issues=issues, cleaning_explanation=explanation, cleaner=cleaner)
        with c2:
            if ss("cleaning_issues") and st.button("✨ Auto-Clean Dataset", width="stretch"):
                c = ss("cleaner") or cleaner
                cleaned_df, log = c.auto_clean(ss("cleaning_issues"))
                sset(df_cleaned=cleaned_df, cleaning_log=log)
                st.success(f"✅ {len(log)} fixes applied.")

        if ss("cleaning_issues"):
            issues = ss("cleaning_issues")
            st.subheader("📋 Data Quality Report")
            for line in ss("cleaning_explanation","").split("\n"):
                if line.strip():
                    card = "card-red" if "missing" in line.lower() else \
                           "card-amber" if any(x in line.lower() for x in ["skew","outlier","duplicate"]) \
                           else "card-blue"
                    st.markdown(f'<div class="{card}">{line}</div>', unsafe_allow_html=True)

            c = ss("cleaner") or DataCleaner(df, schema)
            st.plotly_chart(c.plot_missing_heatmap(), width="stretch")

            cm, co = st.columns(2)
            with cm:
                st.subheader("Missing Value Strategy")
                if issues["missing"]:
                    st.dataframe(pd.DataFrame(issues["missing"])[["column","missing_pct","strategy","reason"]],
                                 width="stretch")
                else: st.success("No missing values.")
            with co:
                st.subheader("Outlier Detection")
                if issues["outliers"]:
                    st.dataframe(pd.DataFrame(issues["outliers"])[["column","n_outliers","pct","strategy"]],
                                 width="stretch")
                    sel_col = st.selectbox("View box plot", [o["column"] for o in issues["outliers"]])
                    st.plotly_chart(c.plot_outlier_box(sel_col), width="stretch")
                else: st.success("No significant outliers.")

            dup = issues["duplicates"]
            cd, cs = st.columns(2)
            with cd:
                st.metric("Duplicate rows", dup["n_duplicates"],
                          delta=f"{dup['pct']}%" if dup["n_duplicates"]>0 else None,
                          delta_color="inverse")
            with cs:
                sk = issues.get("skewness", [])
                if sk:
                    st.dataframe(pd.DataFrame(sk)[["column","skewness","suggestion"]],
                                 width="stretch")
                else: st.success("No highly skewed columns.")

        if ss("cleaning_log"):
            st.subheader("Cleaning Log")
            for line in ss("cleaning_log"):
                st.markdown(f'<div class="card-green">{line}</div>', unsafe_allow_html=True)
            if ss("df_cleaned") is not None:
                cleaned = ss("df_cleaned")
                st.info(f"Cleaned: {cleaned.shape[0]:,} rows × {cleaned.shape[1]} cols")
                ca, cb = st.columns(2)
                with ca:
                    if st.button("✅ Use Cleaned Dataset", width="stretch"):
                        sset(df=cleaned, schema=detect_schema(cleaned),
                             results=None, profiling=None, explainer=None, ai_insights=None)
                        st.success("Active!"); st.rerun()
                with cb:
                    st.download_button("📥 Download Cleaned CSV",
                                       cleaned.to_csv(index=False), "cleaned.csv", "text/csv",
                                       width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("⚙️ Feature Engineering AI")
    if ss("df") is None:
        st.info("Upload a dataset first.")
    else:
        df, schema = ss("df"), ss("schema")
        fe_target = st.selectbox("Target (for scoring)", schema.suggested_targets or df.columns.tolist(), key="fe_t")
        fe = FeatureEngineer(df, schema, target_col=fe_target)

        if st.button("🔎 Analyse Feature Opportunities", type="primary", width="stretch"):
            with st.spinner("Analysing..."):
                sugg = fe.suggest_features()
                fe_text = fe.llm_suggestions(sugg)
                sset(fe_suggestions=sugg, fe_text=fe_text, fe_obj=fe)

        if ss("fe_suggestions"):
            sugg = ss("fe_suggestions")
            st.subheader("💡 AI Recommendations")
            for line in ss("fe_text","").split("\n"):
                if line.strip():
                    st.markdown(f'<div class="card-blue">{line}</div>', unsafe_allow_html=True)

            all_sugg = []
            for cat, items in sugg.items():
                for item in items:
                    fname = item.get("feature") or f"{item.get('column','')}_encoded"
                    all_sugg.append((fname, cat, item.get("reason","")))

            st.subheader("Select Features to Create")
            selected = [fname for fname,cat,reason in all_sugg
                        if st.checkbox(f"**{fname}** `[{cat}]` — {reason}", key=f"fe_{fname}")]

            if selected and st.button("⚙️ Generate Selected Features", width="stretch"):
                fe_obj = ss("fe_obj") or fe
                new_df, log = fe_obj.apply_features(sugg, selected)
                sset(df_engineered=new_df, fe_log=log, fe_obj=fe_obj)

            if ss("fe_log"):
                for line in ss("fe_log"):
                    st.markdown(f'<div class="card-green">{line}</div>', unsafe_allow_html=True)
                fe_obj = ss("fe_obj")
                # FIX: correlation impact chart
                if fe_obj and fe_obj.new_features and fe_target in df.columns:
                    st.subheader("📊 Correlation Impact")
                    try:
                        st.plotly_chart(fe_obj.plot_correlation_delta(), width="stretch")
                    except Exception as e:
                        # Fallback: simple bar of new feature importances
                        try:
                            target_s = pd.to_numeric(df[fe_target], errors="coerce")
                            eng_df = ss("df_engineered")
                            rows = []
                            for feat in fe_obj.new_features:
                                if feat in eng_df.columns:
                                    corr = abs(pd.to_numeric(eng_df[feat], errors="coerce").corr(target_s))
                                    rows.append({"feature": feat, "correlation": round(corr,4)})
                            if rows:
                                imp_df = pd.DataFrame(rows).sort_values("correlation")
                                fig = px.bar(imp_df, x="correlation", y="feature", orientation="h",
                                             title="New Feature Correlation with Target",
                                             template="plotly_white",
                                             color="correlation", color_continuous_scale="Blues")
                                fig.update_layout(height=max(250, 35*len(rows)),
                                                  coloraxis_showscale=False)
                                st.plotly_chart(fig, width="stretch")
                        except Exception:
                            st.info("Correlation chart unavailable — target may be non-numeric.")

                ca, cb = st.columns(2)
                with ca:
                    if st.button("✅ Use Engineered Dataset", width="stretch"):
                        eng = ss("df_engineered")
                        sset(df=eng, schema=detect_schema(eng), results=None, profiling=None)
                        st.success("Active!"); st.rerun()
                with cb:
                    eng = ss("df_engineered")
                    if eng is not None:
                        st.download_button("📥 Download Engineered CSV",
                                           eng.to_csv(index=False), "engineered.csv", "text/csv",
                                           width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PROFILING
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("📊 Dataset Profiling")
    if ss("df") is None:
        st.info("Upload a dataset first.")
    else:
        df, schema = ss("df"), ss("schema")
        if ss("profiling") is None:
            with st.spinner("Profiling..."):
                sset(profiling=profile_dataset(df, schema))
        profiling = ss("profiling")
        st.markdown('<span class="badge">⚡ Auto-generated</span>', unsafe_allow_html=True)
        st.dataframe(profiling["summary_stats"], width="stretch")
        st.plotly_chart(profiling["missing"], width="stretch")
        st.plotly_chart(profiling["correlation"], width="stretch")
        if profiling["target_breakdown"]:
            cols = st.columns(min(3, len(profiling["target_breakdown"])))
            for i,(t,fig) in enumerate(profiling["target_breakdown"].items()):
                cols[i%len(cols)].plotly_chart(fig, width="stretch")
        dist_keys = list(profiling["distributions"].keys())
        sel = st.multiselect("Feature distributions", dist_keys, default=dist_keys[:4])
        for c in sel:
            st.plotly_chart(profiling["distributions"][c], width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — AUTOML  (model picker + custom selection)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.header("🤖 AutoML Training")
    if ss("df") is None:
        st.info("Upload a dataset first.")
    else:
        df, schema = ss("df"), ss("schema")

        # ── Configuration row ─────────────────────────────────────────────────
        cfg1, cfg2, cfg3 = st.columns([1, 1, 2])
        with cfg1:
            target = st.selectbox("Target column",
                schema.suggested_targets or df.columns.tolist(), key="ml_target")
            pt = st.radio("Problem type", ["Auto","Classification","Regression"], horizontal=True)
            if pt == "Classification": schema.problem_type = "classification"
            elif pt == "Regression":   schema.problem_type = "regression"
        with cfg2:
            test_size = st.slider("Test split %", 10, 40, 20, 5)
            cv_folds  = st.slider("CV folds (leaderboard)", 2, 10, 5)

        # ── Model selection checkboxes ─────────────────────────────────────────
        with cfg3:
            st.markdown("**Select models to train:**")
            avail_clf = list(CLASSIFIERS.keys())
            avail_reg = list(REGRESSORS.keys())

            clf_sel = {}
            reg_sel = {}
            c_cols = st.columns(2)
            with c_cols[0]:
                st.caption("Classification")
                for m in avail_clf:
                    clf_sel[m] = st.checkbox(m, value=True, key=f"clf_{m}")
            with c_cols[1]:
                st.caption("Regression")
                for m in avail_reg:
                    reg_sel[m] = st.checkbox(m, value=True, key=f"reg_{m}")

        # ── Model info cards ──────────────────────────────────────────────────
        with st.expander("ℹ️ Model descriptions"):
            model_info = {
                "Logistic Regression":   "Fast linear baseline. Good for linearly separable data. Highly interpretable.",
                "Random Forest":         "Ensemble of decision trees. Robust to noise, handles non-linearity well.",
                "Gradient Boosting":     "Sequential tree boosting. High accuracy, slower to train than RF.",
                "XGBoost":               "Optimised gradient boosting. Winner of many Kaggle competitions.",
                "LightGBM":              "Fast gradient boosting by Microsoft. Excellent for large datasets.",
                "Linear Regression":     "Simple linear model for regression. Best when relationship is linear.",
            }
            for name, desc in model_info.items():
                st.markdown(f'<div class="model-card"><strong>{name}</strong> — {desc}</div>',
                            unsafe_allow_html=True)

        st.markdown("---")
        train_btn = st.button("🚀 Train Selected Models", type="primary", width="stretch")

        if train_btn:
            # Build filtered model dicts from checkboxes
            selected_clf = {k:v for k,v in CLASSIFIERS.items() if clf_sel.get(k, False)}
            selected_reg = {k:v for k,v in REGRESSORS.items() if reg_sel.get(k, False)}

            if not selected_clf and not selected_reg:
                st.warning("Select at least one model."); st.stop()

            # Temporarily override trainer registries
            import src.automl.trainer as trainer_mod
            orig_clf = trainer_mod.CLASSIFIERS.copy()
            orig_reg = trainer_mod.REGRESSORS.copy()
            if schema.problem_type == "classification":
                trainer_mod.CLASSIFIERS = selected_clf
            else:
                trainer_mod.REGRESSORS = selected_reg

            prog = st.progress(0, text="Starting...")
            def cb(name, frac): prog.progress(frac, text=f"Training {name}...")
            try:
                with st.spinner(""):
                    # Apply test_size override
                    import src.automl.preprocessor as prep_mod
                    orig_ts = prep_mod.TEST_SIZE
                    prep_mod.TEST_SIZE = test_size / 100
                    prep = preprocess(df, target, schema)
                    prep_mod.TEST_SIZE = orig_ts

                    results = train_all(prep.X_train, prep.X_test,
                                        prep.y_train, prep.y_test,
                                        schema.problem_type, progress_callback=cb)
                sset(results=results, prep_result=prep, target=target,
                     explainer=None, ai_insights=None, shap_importance_df=None,
                     data_story=None)
                ss("tracker").log_all_from_leaderboard(
                    results["leaderboard"], schema.problem_type,
                    ss("dataset_name","unknown"), df.shape, target)
                prog.progress(1.0, text="Done!")
                st.success(f"✅ Best: **{results['best_model']['model']}**")
                st.markdown(get_selection_rationale(results["leaderboard"], schema.problem_type))
            except Exception as e:
                st.error(f"Training failed: {e}")
            finally:
                trainer_mod.CLASSIFIERS = orig_clf
                trainer_mod.REGRESSORS  = orig_reg

        if ss("results"):
            best = ss("results")["best_model"]
            items = {k:v for k,v in best.items() if k not in ("model","estimator","train_time_s")}
            if items:
                cols = st.columns(len(items))
                for i,(k,v) in enumerate(items.items()):
                    cols[i].metric(k.upper(), v)

            # Quick predict on custom input
            with st.expander("🎯 Quick Predict — enter values manually"):
                prep = ss("prep_result")
                input_vals = {}
                pred_cols = st.columns(min(4, len(prep.feature_names)))
                for i, feat in enumerate(prep.feature_names):
                    with pred_cols[i % len(pred_cols)]:
                        input_vals[feat] = st.number_input(feat, value=0.0, key=f"qp_{feat}")
                if st.button("Predict", key="quick_pred"):
                    try:
                        row = pd.DataFrame([input_vals])
                        model = best["estimator"]
                        if schema.problem_type == "classification" and hasattr(model, "predict_proba"):
                            proba = model.predict_proba(row)[0]
                            pred = model.predict(row)[0]
                            for i, p in enumerate(proba):
                                st.metric(f"Class {i} probability", f"{p:.1%}")
                        else:
                            pred = model.predict(row)[0]
                            st.metric("Prediction", f"{pred:.4f}")
                    except Exception as e:
                        st.error(f"Prediction error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — LEADERBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.header("🏆 Model Leaderboard")
    if ss("results") is None:
        st.info("Train models first.")
    else:
        results, schema = ss("results"), ss("schema")
        lb = results["leaderboard"]
        lb_df = pd.DataFrame([{k:v for k,v in m.items() if k!="estimator"} for m in lb])
        def hl(s): return ["background:#d1fae5;font-weight:700" if i==0 else "" for i in range(len(s))]
        st.dataframe(lb_df.style.apply(hl, axis=0, subset=[lb_df.columns[0]]),
                     width="stretch")
        st.plotly_chart(plot_leaderboard(lb, schema.problem_type), width="stretch")
        if "train_time_s" in lb_df.columns:
            fig = go.Figure(go.Bar(x=lb_df["model"], y=lb_df["train_time_s"],
                marker_color="#1a56db",
                text=lb_df["train_time_s"].apply(lambda x: f"{x:.1f}s"),
                textposition="outside"))
            fig.update_layout(template="plotly_white", height=280,
                              title="Training Time per Model", yaxis_title="Seconds")
            st.plotly_chart(fig, width="stretch")
        st.download_button("📥 Export Leaderboard CSV",
                           lb_df.to_csv(index=False), "leaderboard.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — EXPERIMENT TRACKER
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.header("🧪 Experiment Tracker")
    st.caption("Every training run is auto-logged. Compare experiments like MLflow.")
    tracker = ss("tracker")
    c1, c2 = st.columns([3,1])
    with c1:
        runs_df = tracker.get_runs_df()
        if runs_df.empty:
            st.info("No experiments yet. Train models to auto-log runs.")
        else:
            st.dataframe(runs_df, width="stretch")
            st.download_button("📥 Export CSV", runs_df.to_csv(index=False), "experiments.csv","text/csv")
    with c2:
        for metric in ["roc_auc","r2","accuracy"]:
            best = tracker.best_run(metric)
            if best:
                st.metric(f"Best {metric.upper()}", best["metrics"].get(metric,"N/A"))
                st.caption(f"Model: {best['model']}")
                break
        if st.button("🗑️ Clear all"):
            tracker.clear(); st.rerun()

    if not runs_df.empty:
        metric_opts = [c for c in runs_df.columns if c in ["roc_auc","accuracy","f1","r2","mae"]]
        if metric_opts:
            met = st.selectbox("Metric to plot", metric_opts)
            st.plotly_chart(tracker.plot_metric_over_time(met), width="stretch")
    st.plotly_chart(tracker.plot_model_comparison(), width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.header("🎯 Prediction Results")
    if ss("results") is None:
        st.info("Train models first.")
    else:
        results, prep, schema = ss("results"), ss("prep_result"), ss("schema")
        best_model = results["best_model"]["estimator"]
        st.markdown(f"**Model:** {results['best_model']['model']}")
        if schema.problem_type == "classification":
            c1,c2 = st.columns(2)
            with c1:
                st.plotly_chart(plot_confusion_matrix(best_model, prep.X_test, prep.y_test,
                    [str(c) for c in prep.label_encoder.classes_]), width="stretch")
            with c2:
                st.plotly_chart(plot_roc_curve(best_model, prep.X_test, prep.y_test),
                                width="stretch")
        else:
            c1,c2 = st.columns(2)
            with c1:
                st.plotly_chart(plot_predicted_vs_actual(best_model, prep.X_test, prep.y_test),
                                width="stretch")
            with c2:
                st.plotly_chart(plot_residuals(best_model, prep.X_test, prep.y_test),
                                width="stretch")
        preds = best_model.predict(prep.X_test)
        pred_df = prep.X_test.copy()
        pred_df["actual"] = prep.y_test.values
        pred_df["predicted"] = preds
        if schema.problem_type=="classification" and hasattr(best_model,"predict_proba"):
            pred_df["probability"] = best_model.predict_proba(prep.X_test)[:,1].round(4)
        st.dataframe(pred_df.head(50), width="stretch")
        st.download_button("📥 Download predictions",
                           pred_df.to_csv(index=False), "predictions.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 — DIAGNOSTICS + DRIFT
# ══════════════════════════════════════════════════════════════════════════════
with tabs[8]:
    st.header("📈 Model Diagnostics")
    if ss("results") is None:
        st.info("Train models first.")
    else:
        results, prep, schema = ss("results"), ss("prep_result"), ss("schema")
        lb = results["leaderboard"]
        best_model = results["best_model"]["estimator"]

        if schema.problem_type == "classification":
            # Precision-Recall
            if hasattr(best_model,"predict_proba"):
                from sklearn.metrics import precision_recall_curve, average_precision_score
                try:
                    n_cls = len(np.unique(prep.y_test))
                    if n_cls == 2:
                        proba = best_model.predict_proba(prep.X_test)[:,1]
                        prec, rec, _ = precision_recall_curve(prep.y_test, proba)
                        ap = average_precision_score(prep.y_test, proba)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines",
                            fill="tozeroy", line=dict(color="#1a56db",width=2),
                            name=f"AP={ap:.3f}"))
                        fig.update_layout(title=f"Precision-Recall Curve (AP={ap:.3f})",
                            xaxis_title="Recall", yaxis_title="Precision",
                            template="plotly_white", height=360)
                        st.plotly_chart(fig, width="stretch")
                except Exception as e:
                    st.warning(f"PR curve: {e}")

            # All-model ROC overlay
            st.subheader("ROC Curves — All Models")
            fig_all = go.Figure()
            colors = px.colors.qualitative.Set2
            for i, m in enumerate(lb):
                est = m["estimator"]
                if hasattr(est,"predict_proba"):
                    try:
                        from sklearn.metrics import roc_curve as rc, roc_auc_score as ras
                        n_c = len(np.unique(prep.y_test))
                        if n_c == 2:
                            p = est.predict_proba(prep.X_test)[:,1]
                            fpr,tpr,_ = rc(prep.y_test, p)
                            auc = ras(prep.y_test, p)
                            fig_all.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",
                                line=dict(color=colors[i%len(colors)],width=1.8),
                                name=f"{m['model']} AUC={auc:.3f}"))
                    except Exception: pass
            fig_all.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",
                line=dict(color="#9ca3af",dash="dash"),name="Random"))
            fig_all.update_layout(title="ROC — All Models",xaxis_title="FPR",
                yaxis_title="TPR",template="plotly_white",height=400)
            st.plotly_chart(fig_all, width="stretch")
        else:
            c1,c2 = st.columns(2)
            with c1: st.plotly_chart(plot_predicted_vs_actual(best_model, prep.X_test, prep.y_test), width="stretch")
            with c2: st.plotly_chart(plot_residuals(best_model, prep.X_test, prep.y_test), width="stretch")

        # Feature importance comparison
        st.subheader("Feature Importance Comparison")
        imp_data = []
        for m in lb:
            est = m["estimator"]
            if hasattr(est,"feature_importances_"):
                for feat,imp in zip(prep.feature_names, est.feature_importances_):
                    imp_data.append({"model":m["model"],"feature":feat,"importance":round(float(imp),5)})
        if imp_data:
            imp_df = pd.DataFrame(imp_data)
            top_f = imp_df.groupby("feature")["importance"].mean().sort_values(ascending=False).head(12).index.tolist()
            fig = px.bar(imp_df[imp_df["feature"].isin(top_f)],
                x="importance", y="feature", color="model", barmode="group",
                orientation="h", title="Feature Importance — All Tree Models (Top 12)",
                template="plotly_white")
            fig.update_layout(height=max(380,28*len(top_f)))
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Feature importance available for tree-based models (Random Forest, XGBoost, LightGBM).")

        # ── DATA DRIFT DETECTION ──────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📉 Data Drift Detection")

        with st.expander("ℹ️ What is Data Drift and why does it matter?", expanded=False):
            st.markdown("""
**Data drift** happens when the statistical distribution of real-world data changes after your model is deployed.

**Example:** You trained a churn model in January. By July, customer behaviour has changed — 
younger customers are now staying longer, prices increased. The patterns the model learned are now outdated.

**What this tool does:**
- Compare your **training data** distribution vs a **new CSV** you upload
- Uses **KS test** (numeric) and **Chi-squared test** (categorical) — standard statistical drift tests
- Highlights which columns have changed significantly (p-value < 0.05)
- Tells you **which features to investigate** before retraining

**When to use it:** Upload last month's data, last quarter's data, or production data 
to check if your model needs retraining.
            """)

        drift_file = st.file_uploader(
            "Upload new/production data CSV to compare against training data",
            type=["csv"], key="drift_upload",
            help="Upload a CSV with the same columns as your training data. The system will detect which columns have drifted.")
        if drift_file:
            try:
                new_df = pd.read_csv(drift_file)
                st.info(f"Comparing training data ({ss('df').shape[0]:,} rows) vs new data ({new_df.shape[0]:,} rows)")
                detector = DriftDetector(ss("df"), ss("schema"))
                drift_results = detector.detect(new_df)
                n_drifted = detector.n_drifted()
                c1,c2,c3 = st.columns(3)
                c1.metric("Columns tested", len(drift_results))
                c2.metric("Drifted columns", n_drifted,
                          delta="⚠️ retraining recommended" if n_drifted > 0 else "✅ no drift",
                          delta_color="inverse" if n_drifted > 0 else "normal")
                c3.metric("Clean columns", len(drift_results)-n_drifted)

                if n_drifted > 0:
                    st.warning(f"**{n_drifted} column(s) show significant drift** — model may need retraining.")
                else:
                    st.success("✅ No significant drift detected. Model is likely still valid.")

                st.plotly_chart(detector.plot_drift_summary(), width="stretch")
                st.dataframe(detector.summary(), width="stretch")
                dcol = st.selectbox("Inspect column distribution", list(drift_results.keys()))
                if dcol:
                    st.plotly_chart(detector.plot_distribution_comparison(dcol, new_df),
                                    width="stretch")
            except Exception as e:
                st.error(f"Drift detection error: {e}")
        else:
            st.markdown('<div class="card-blue">💡 <strong>Tip:</strong> Upload a new data file above to detect distribution shifts between your training data and current data. This is critical for production ML systems.</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 9 — SHAP
# ══════════════════════════════════════════════════════════════════════════════
with tabs[9]:
    st.header("🔍 SHAP Explainability")
    if ss("results") is None:
        st.info("Train models first.")
    else:
        results, prep, schema = ss("results"), ss("prep_result"), ss("schema")
        best_model = results["best_model"]["estimator"]
        if ss("explainer") is None:
            with st.spinner("Computing SHAP values..."):
                try:
                    exp = SHAPExplainer(best_model, prep.X_train, schema.problem_type)
                    sset(explainer=exp, shap_importance_df=exp.feature_importance_df())
                except Exception as e:
                    st.error(f"SHAP failed: {e}")
        exp = ss("explainer")
        if exp:
            st.plotly_chart(exp.feature_importance_plotly(), width="stretch")
            try:
                img = exp.summary_plot_bytes()
                st.image(img, width=800)
                sset(shap_summary_img=img)
            except Exception as e:
                st.warning(f"Summary plot: {e}")
            idx = st.number_input("Row for waterfall chart", 0, len(prep.X_test)-1, 0)
            try:
                st.image(exp.waterfall_plot_bytes(int(idx)), width=800)
            except Exception as e:
                st.warning(f"Waterfall: {e}")
            feat = st.selectbox("Feature for dependence plot", prep.X_train.columns.tolist())
            st.plotly_chart(exp.dependence_plot_plotly(feat), width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 10 — INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[10]:
    st.header("💡 AI Business Insights")
    if ss("results") is None:
        st.info("Train models first.")
    else:
        df, schema = ss("df"), ss("schema")
        results, target = ss("results"), ss("target")
        shap_df = safe_df("shap_importance_df")
        if ss("ai_insights") is None:
            with st.spinner("Generating insights..."):
                try:
                    summary = generate_dataset_summary(df, schema, target, schema.problem_type)
                    insights = generate_business_insights(
                        df, schema, shap_df, results["leaderboard"], target, schema.problem_type)
                    sset(ai_insights=insights, ai_summary=summary)
                except Exception as e:
                    st.error(f"Error: {e}")
        if ss("ai_summary"):
            best = results["best_model"]
            mk = "roc_auc" if schema.problem_type=="classification" else "r2"
            top_f = ", ".join(f"'{r['feature']}'" for _,r in shap_df.head(3).iterrows()) \
                    if not shap_df.empty else "N/A"
            st.markdown(f"""<div class="lemann"><h4>🔎 What the analysis found</h4><ul>
<li>Analysed <span class="hl">{df.shape[0]:,} records</span> to predict <span class="hl">'{target}'</span></li>
<li>Best model: <span class="hl">{best['model']}</span> ({mk.upper()}={best.get(mk,'N/A')})</li>
<li>Top predictors: {top_f}</li>
<li>{ss('ai_summary','')}</li>
</ul></div>""", unsafe_allow_html=True)
        for ins in safe_list("ai_insights"):
            st.markdown(f'<div class="card-green">💡 {ins}</div>', unsafe_allow_html=True)
        if schema.problem_type=="classification":
            bm = results["best_model"]["estimator"]
            prep = ss("prep_result")
            rec_df = batch_recommendations(bm, prep.X_test, top_n=10)
            if not rec_df.empty:
                st.subheader("Top 10 High-Risk Records")
                st.dataframe(rec_df, width="stretch")
                st.download_button("📥 Download risk list",
                                   rec_df.to_csv(index=False),"risk.csv","text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 11 — DATA STORY
# ══════════════════════════════════════════════════════════════════════════════
with tabs[11]:
    st.header("📖 Auto Data Story")
    if ss("results") is None:
        st.info("Train models first.")
    else:
        df, schema = ss("df"), ss("schema")
        results, target = ss("results"), ss("target")
        shap_df = safe_df("shap_importance_df")
        insights = safe_list("ai_insights")
        best = results["best_model"]
        mk = "roc_auc" if schema.problem_type=="classification" else "r2"

        if st.button("📖 Generate Data Story", type="primary", width="stretch"):
            with st.spinner("Crafting narrative..."):
                feat_facts = "\n".join(f"- {r['feature']}: SHAP={r['importance']:.4f}"
                    for _,r in shap_df.head(5).iterrows()) if not shap_df.empty else "SHAP not computed."
                insight_text = "\n".join(f"- {i}" for i in insights[:5]) if insights else "Run insights tab first."
                try:
                    vc = df[target].value_counts(normalize=True)*100
                    vc_text = ", ".join(f"{k}={v:.0f}%" for k,v in vc.items())
                except: vc_text = "N/A"

                story = None
                if GROQ_API_KEY or GOOGLE_API_KEY:
                    try:
                        from utils.gemini_client import _gemini_generate
                        prompt = f"""Write a compelling data story for a business stakeholder.

## The Business Problem
## What the Data Tells Us
## What Drives the Prediction
## Model Performance
## Recommended Actions
## Conclusion

Dataset: {ss('dataset_name','Dataset')}, {df.shape[0]:,} rows, target='{target}' ({vc_text})
Key findings: {insight_text}
Top features: {feat_facts}
Best model: {best['model']} ({mk}={best.get(mk,'N/A')})

Write engaging, executive-friendly language. Include specific numbers."""
                        story = _gemini_generate(prompt, max_tokens=1000, temperature=0.4)
                    except Exception as e:
                        logger.warning("Story LLM failed (%s) — using template", e)

                if not story:
                    story = f"""## The Business Problem

We analysed **{ss('dataset_name','the dataset')}** ({df.shape[0]:,} records) to predict **'{target}'** — a {schema.problem_type} task. {f'Class distribution: {vc_text}.' if vc_text != 'N/A' else ''}

## What the Data Tells Us

{chr(10).join(f'- {i}' for i in insights[:5]) if insights else 'Run Insights tab to generate findings.'}

## What Drives the Prediction

{feat_facts}

## Model Performance

Best model: **{best['model']}** | {mk.upper()} = **{best.get(mk,'N/A')}** (selected from {len(results['leaderboard'])} candidates).

## Recommended Actions

- Monitor top predictive features closely for changes.
- Use the 🎛️ What-if tab to test targeted interventions.
- Re-train the model monthly as new data arrives.
- Share SHAP explanations with domain experts for validation.

## Conclusion

This analysis delivers a data-driven foundation for decisions around '{target}'.
The next step is operationalising predictions via the 🚀 Deploy tab."""
                sset(data_story=story)

        if ss("data_story"):
            st.markdown(ss("data_story"))
            st.download_button("📥 Download Story",
                               ss("data_story"), "data_story.md", "text/markdown",
                               width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 12 — CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tabs[12]:
    st.header("💬 Analyst Intelligence Chat")

    with st.expander("ℹ️ How RAG makes this different from a normal chatbot"):
        st.markdown("""
**RAG = Retrieval-Augmented Generation.** Before answering, the AI searches through your actual dataset statistics, model scores, and SHAP feature importances — then answers based on that real evidence.

**Normal chatbot:** guesses from training data.  
**This chat:** reads YOUR data first, then answers.
""")
        for q in ["Why are customers churning?","Which features drive predictions most?",
                  "What data quality issues should I flag?","Summarise for my CEO",
                  "Give 3 concrete business recommendations","What is the model accuracy?"]:
            st.markdown(f'<div class="card-blue">📌 "{q}"</div>', unsafe_allow_html=True)

    if ss("results") is None:
        st.info("Train models first.")
    elif not GOOGLE_API_KEY and not GROQ_API_KEY:
        st.warning("Add `GOOGLE_API_KEY` (Gemini) or `GROQ_API_KEY` (Groq) to `.env` for AI chat.")
        st.code("# Option A: Google Gemini\nGOOGLE_API_KEY=AIza...\nGEMINI_MODEL=gemini-1.5-flash\n\n# Option B: Groq (free, no quota)\nGROQ_API_KEY=gsk_...\nGROQ_MODEL=llama3-8b-8192")
        st.markdown("---")
        st.subheader("🔎 Offline Natural Language Queries")
        st.caption("Works without API key — direct answers from your data statistics.")

        nl_q = st.text_input("Ask a question",
            placeholder="e.g. show top 10 rows, missing values, distribution, average, correlation")
        if st.button("▶ Run", key="nl_run") and nl_q and ss("df") is not None:
            df = ss("df")
            q = nl_q.lower()
            try:
                if any(w in q for w in ["top","highest","largest","biggest","most"]):
                    nums = df.select_dtypes("number").columns
                    col  = next((c for c in nums if c.lower() in q), nums[0] if len(nums)>0 else None)
                    n    = next((int(w) for w in q.split() if w.isdigit()), 10)
                    if col: st.dataframe(df.nlargest(n, col), width="stretch")
                elif any(w in q for w in ["missing","null","nan","empty"]):
                    miss = df.isna().sum()
                    miss = miss[miss>0].reset_index()
                    miss.columns = ["Column","Missing Count"]
                    miss["Missing %"] = (miss["Missing Count"]/len(df)*100).round(2)
                    st.dataframe(miss, width="stretch")
                elif any(w in q for w in ["distribution","count","unique","value"]):
                    schema = ss("schema")
                    col = next((c for c in (schema.categorical_cols if schema else []) if c.lower() in q), None)
                    if not col and schema: col = schema.suggested_targets[0] if schema.suggested_targets else None
                    if col and col in df.columns:
                        vc = df[col].value_counts().reset_index()
                        vc.columns=[col,"Count"]
                        st.dataframe(vc, width="stretch")
                        fig = px.pie(vc, names=col, values="Count", title=f"Distribution of {col}", template="plotly_white")
                        st.plotly_chart(fig, width="stretch")
                elif any(w in q for w in ["average","mean","avg"]):
                    means = df.select_dtypes("number").mean().reset_index()
                    means.columns=["Column","Mean"]; means["Mean"]=means["Mean"].round(4)
                    col = next((c for c in means["Column"] if c.lower() in q), None)
                    if col: st.metric(f"Mean of {col}", round(df[col].mean(),4))
                    else: st.dataframe(means, width="stretch")
                elif any(w in q for w in ["correlation","corr","relationship"]):
                    nums = df.select_dtypes("number")
                    corr = nums.corr().round(3)
                    fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                    title="Correlation Heatmap", template="plotly_white", text_auto=".2f")
                    st.plotly_chart(fig, width="stretch")
                elif any(w in q for w in ["shape","rows","columns","size","dimension"]):
                    st.info(f"**{df.shape[0]:,} rows × {df.shape[1]} columns**")
                    st.dataframe(df.dtypes.reset_index().rename(columns={"index":"Column",0:"Type"}),
                                 width="stretch")
                elif any(w in q for w in ["outlier","extreme","anomaly"]):
                    nums = df.select_dtypes("number")
                    rows = []
                    for c in nums.columns:
                        Q1,Q3 = nums[c].quantile(0.25), nums[c].quantile(0.75)
                        IQR = Q3-Q1
                        if IQR>0:
                            n_out = int(((nums[c]<Q1-1.5*IQR)|(nums[c]>Q3+1.5*IQR)).sum())
                            if n_out>0: rows.append({"Column":c,"Outliers":n_out,"Pct":round(n_out/len(df)*100,2)})
                    if rows: st.dataframe(pd.DataFrame(rows), width="stretch")
                    else: st.success("No significant outliers found.")
                elif any(w in q for w in ["describe","summary","statistics","stats"]):
                    st.dataframe(df.describe().round(4), width="stretch")
                else:
                    st.info("Try: 'top 10', 'missing values', 'distribution', 'average', 'correlation', 'outliers', 'describe', 'shape'")
            except Exception as e:
                st.error(f"Query error: {e}")

        st.markdown("**Quick questions:**")
        quick = [("Missing values","missing"),("Top 10 rows","top 10"),
                 ("Correlation","correlation"),("Value counts","distribution"),
                 ("Describe","describe"),("Outliers","outliers")]
        qc = st.columns(len(quick))
        for i,(label,q) in enumerate(quick):
            if qc[i].button(label, key=f"q_{i}"):
                sset(prefill_offline=q)
    else:
        if ss("rag_chain") is None:
            if st.button("🔧 Build Knowledge Base & Start Chat", type="primary", width='stretch'):
                with st.spinner("Indexing dataset..."):
                    try:
                        from src.rag.embedder import build_corpus
                        from src.rag.vector_store import build_vector_store
                        from src.rag.qa_chain import build_rag_chain
                        df, schema = ss("df"), ss("schema")
                        results, target = ss("results"), ss("target")
                        shap_df = safe_df("shap_importance_df")
                        insights = safe_list("ai_insights")
                        corpus = build_corpus(df, schema, shap_df,
                                             results["leaderboard"], insights,
                                             target, schema.problem_type)
                        vs = build_vector_store(corpus)
                        chain_obj = build_rag_chain(vs)
                        sset(rag_chain=chain_obj)
                        store_type = vs.get("type","faiss") if isinstance(vs,dict) else "faiss"
                        st.success(f"✅ {len(corpus)} chunks indexed ({store_type} store)."); st.rerun()
                    except Exception as e:
                        st.error(f"Failed: {e}")
        else:
            from utils.gemini_client import get_provider_display
            st.success(f"✅ AI chat active — powered by your data + {get_provider_display()}")
            eq = ["Why are customers churning?","Top features driving prediction?",
                  "Data quality issues?","Summarise for CEO","3 business recommendations"]
            ec = st.columns(len(eq))
            for i,q in enumerate(eq):
                if ec[i].button(q, key=f"eq_{i}"): sset(prefill_q=q)
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]): st.markdown(msg["content"])
            question = st.chat_input("Ask anything about your dataset...")
            prefill = ss("prefill_q","")
            active_q = question or (prefill if prefill != ss("last_prefill","") else None)
            if active_q:
                sset(last_prefill=prefill)
                st.session_state.chat_history.append({"role":"user","content":active_q})
                with st.chat_message("user"): st.markdown(active_q)
                with st.chat_message("assistant"):
                    with st.spinner("Searching your data..."):
                        try:
                            from src.rag.qa_chain import ask
                            res = ask(ss("rag_chain"), active_q)
                            answer = res["answer"]
                        except Exception as e:
                            answer, res = f"Error: {e}", {}
                    st.markdown(answer)
                    if res.get("sources"):
                        with st.expander("📎 Sources"): 
                            for s in res["sources"][:3]: st.caption(s)
                st.session_state.chat_history.append({"role":"assistant","content":answer})


# ══════════════════════════════════════════════════════════════════════════════
# TAB 13 — WHAT-IF
# ══════════════════════════════════════════════════════════════════════════════
with tabs[13]:
    st.header("🎛️ What-if Simulation")
    if ss("results") is None:
        st.info("Train models first.")
    else:
        results, prep, schema = ss("results"), ss("prep_result"), ss("schema")
        best_model = results["best_model"]["estimator"]
        c1,c2 = st.columns([1,1])
        with c1:
            row_idx = st.number_input("Test row", 0, len(prep.X_test)-1, 0)
            base_row = prep.X_test.iloc[[row_idx]]
            overrides = {}
            for feat in [c for c in schema.numeric_cols if c in base_row.columns][:8]:
                lo  = float(prep.X_train[feat].min())
                hi  = float(prep.X_train[feat].max())
                cur = float(base_row[feat].values[0])
                val = st.slider(feat, lo, hi, cur,
                                step=max((hi-lo)/100, 1e-4),
                                format="%.3f", key=f"sl_{feat}")
                if abs(val-cur) > 1e-6: overrides[feat] = val
        with c2:
            sim = simulate(best_model, base_row, overrides, schema.problem_type)
            if schema.problem_type == "classification":
                ga,gb = st.columns(2)
                ga.plotly_chart(plot_probability_gauge(sim["original_value"],"Original"),
                                width="stretch")
                gb.plotly_chart(plot_probability_gauge(sim["new_value"],"New"),
                                width="stretch")
                st.metric("Change", f"{sim['new_value']:.1%}",
                          delta=f"{sim['delta']:+.3f}",
                          delta_color="normal" if sim["delta"]<=0 else "inverse")
            else:
                c1b,c2b,c3b = st.columns(3)
                c1b.metric("Original",f"{sim['original_value']:.4f}")
                c2b.metric("New",f"{sim['new_value']:.4f}")
                c3b.metric("Δ",f"{sim['delta']:+.4f}")
            risk = "critical" if sim["new_value"]>0.8 else "high" if sim["new_value"]>0.6 else "medium" if sim["new_value"]>0.4 else "low"
            rc = "#dc2626" if risk in ("critical","high") else "#ca8a04"
            st.markdown(f'<div class="card-amber"><strong style="color:{rc}">● {risk.upper()}</strong><br>{sim["recommendation"]}</div>', unsafe_allow_html=True)
            if sim["feature_deltas"]:
                st.plotly_chart(plot_delta_waterfall(sim["feature_deltas"], sim["delta"]), width="stretch")
        for a in generate_actions(sim["new_value"], use_llm=False)["actions"]:
            st.markdown(f'<div class="card-amber">▶ {a}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 14 — DEPLOY  (all models, sanitised names, full package)
# ══════════════════════════════════════════════════════════════════════════════
with tabs[14]:
    st.header("🚀 Model Deployment")
    if ss("results") is None:
        st.info("Train models first.")
    else:
        results, prep, schema = ss("results"), ss("prep_result"), ss("schema")
        target = ss("target","target")
        lb = results["leaderboard"]

        # ── Model selector — deploy ANY trained model ─────────────────────────
        st.subheader("Choose Model to Deploy")
        model_options = {m["model"]: m for m in lb}
        metric_k = "roc_auc" if schema.problem_type=="classification" else "r2"

        # Show leaderboard as cards
        cols_m = st.columns(min(len(lb), 4))
        for i, m in enumerate(lb):
            with cols_m[i % len(cols_m)]:
                score = m.get(metric_k,"N/A")
                is_best = (i == 0)
                border = "2px solid #10b981" if is_best else "1px solid #e0e7ff"
                badge = "🥇 BEST" if is_best else f"#{i+1}"
                st.markdown(f"""<div style="background:#f8faff;border:{border};
border-radius:10px;padding:.8rem;text-align:center;margin-bottom:.5rem;">
<div style="font-size:.7rem;color:#6b7280;">{badge}</div>
<div style="font-weight:700;font-size:.95rem;">{m['model']}</div>
<div style="color:#1a56db;font-size:1.1rem;font-weight:700;">{score}</div>
<div style="font-size:.7rem;color:#6b7280;">{metric_k.upper()}</div>
</div>""", unsafe_allow_html=True)

        selected_model_name = st.selectbox(
            "Select model to deploy",
            list(model_options.keys()),
            index=0,
            help="The best model is pre-selected. You can choose any trained model.")
        selected_entry = model_options[selected_model_name]

        exporter = ModelExporter(
            model=selected_entry["estimator"],
            feature_names=prep.feature_names,
            problem_type=schema.problem_type,
            target_col=target,
            model_name=selected_model_name,
        )

        # ── Info card ─────────────────────────────────────────────────────────
        safe_names_preview = exporter.safe_names[:5]
        orig_names_preview = exporter.feature_names[:5]
        st.markdown(f"""<div class="deploy-section"><h4>🔎 Deployment Summary</h4>
<p><strong>Model:</strong> {selected_model_name} &nbsp;|&nbsp;
<strong>Target:</strong> {target} &nbsp;|&nbsp;
<strong>Type:</strong> {schema.problem_type} &nbsp;|&nbsp;
<strong>Features:</strong> {len(prep.feature_names)}</p>
<p><strong>Endpoint:</strong> <code>POST /predict</code> and <code>POST /predict_batch</code></p>
<p><strong>Deploy options:</strong> Docker · Docker Compose · Railway · Render · Google Cloud Run · AWS EC2</p>
</div>""", unsafe_allow_html=True)

        # Warn about sanitised names
        renamed = [(o,s) for o,s in zip(exporter.feature_names, exporter.safe_names) if o != s]
        if renamed:
            st.warning(f"⚠️ {len(renamed)} feature name(s) were sanitised for Python/Pydantic compatibility:")
            rename_df = pd.DataFrame(renamed, columns=["Original Name","API Field Name"])
            st.dataframe(rename_df, width="stretch")

        # ── Code tabs ─────────────────────────────────────────────────────────
        code_tab1, code_tab2, code_tab3, code_tab4, code_tab5 = st.tabs(
            ["main.py","Dockerfile","docker-compose.yml","requirements.txt","test_api.py"])

        with code_tab1:
            st.code(exporter.generate_fastapi_code(), language="python")
        with code_tab2:
            st.code(exporter.generate_dockerfile(), language="dockerfile")
        with code_tab3:
            st.code(exporter.generate_docker_compose(), language="yaml")
        with code_tab4:
            st.code(exporter.generate_requirements())
        with code_tab5:
            st.code(exporter.generate_test_script(), language="python")

        # ── Example request ───────────────────────────────────────────────────
        st.subheader("Example API Request")
        sample_req = exporter.generate_sample_request()
        st.code(f"""# Single prediction
curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{json.dumps(sample_req)}'

# Batch prediction
curl -X POST http://localhost:8000/predict_batch \\
  -H "Content-Type: application/json" \\
  -d '[{json.dumps(sample_req)}, {json.dumps(sample_req)}]'

# Interactive API docs (after running)
open http://localhost:8000/docs""", language="bash")

        # ── Deploy steps ──────────────────────────────────────────────────────
        st.subheader("Deploy in 3 Steps")
        st.code("""# Step 1: Download the zip below and unzip
unzip model_api.zip && cd model_api

# Step 2: Build and run with Docker
docker build -t my-model-api .
docker run -p 8000:8000 my-model-api

# Step 3: Test it
python test_api.py
# OR visit http://localhost:8000/docs for interactive API docs""", language="bash")

        # ── Download buttons ──────────────────────────────────────────────────
        st.subheader("⬇️ Download Deployment Package")
        st.caption("Contains: model.pkl + main.py + Dockerfile + docker-compose.yml + requirements.txt + test_api.py + README.md")
        try:
            zip_bytes = exporter.export_zip()
            c1,c2,c3 = st.columns(3)
            with c1:
                st.download_button(
                    "📦 Full Package (.zip)",
                    data=zip_bytes,
                    file_name=f"model_api_{selected_model_name.lower().replace(' ','_')}.zip",
                    mime="application/zip",
                    type="primary", width="stretch")
            with c2:
                st.download_button(
                    "📄 main.py only",
                    data=exporter.generate_fastapi_code(),
                    file_name="main.py", mime="text/x-python",
                    width="stretch")
            with c3:
                st.download_button(
                    "🧪 test_api.py",
                    data=exporter.generate_test_script(),
                    file_name="test_api.py", mime="text/x-python",
                    width="stretch")
        except Exception as e:
            st.error(f"Export error: {e}")

        # ── README preview ────────────────────────────────────────────────────
        with st.expander("📖 README preview"):
            st.markdown(exporter.generate_readme())