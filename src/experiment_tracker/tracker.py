"""
🧪 Experiment Tracker — MLflow-style local experiment logging.
Tracks model, metrics, params, dataset version, training time.
Persists to JSON so results survive session restarts.
"""
import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.config import MODEL_DIR
from utils.logger import get_logger

logger = get_logger(__name__)
TRACKER_PATH = os.path.join(MODEL_DIR, "experiments.json")


class ExperimentTracker:

    def __init__(self):
        self.runs: List[Dict] = self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> List[Dict]:
        if os.path.exists(TRACKER_PATH):
            try:
                with open(TRACKER_PATH) as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save(self):
        os.makedirs(os.path.dirname(TRACKER_PATH), exist_ok=True)
        with open(TRACKER_PATH, "w") as f:
            json.dump(self.runs, f, indent=2, default=str)

    # ── Logging ───────────────────────────────────────────────────────────────

    def log_run(
        self,
        model_name: str,
        metrics: Dict,
        params: Dict,
        problem_type: str,
        dataset_name: str,
        dataset_shape: tuple,
        target_col: str,
        train_time_s: float,
        notes: str = "",
        tags: List[str] = None,
    ) -> str:
        run_id = str(uuid.uuid4())[:8]
        run = {
            "run_id":       run_id,
            "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model":        model_name,
            "metrics":      metrics,
            "params":       params,
            "problem_type": problem_type,
            "dataset_name": dataset_name,
            "dataset_rows": dataset_shape[0],
            "dataset_cols": dataset_shape[1],
            "target_col":   target_col,
            "train_time_s": round(train_time_s, 2),
            "notes":        notes,
            "tags":         tags or [],
        }
        self.runs.append(run)
        self._save()
        logger.info("Logged experiment run %s: %s", run_id, model_name)
        return run_id

    def log_all_from_leaderboard(
        self, leaderboard: List[Dict], problem_type: str,
        dataset_name: str, dataset_shape: tuple, target_col: str,
    ):
        """Bulk-log all models from an AutoML leaderboard."""
        for m in leaderboard:
            metrics = {k: v for k, v in m.items()
                       if k not in ("model", "estimator", "train_time_s")}
            self.log_run(
                model_name=m["model"],
                metrics=metrics,
                params=self._extract_params(m.get("estimator")),
                problem_type=problem_type,
                dataset_name=dataset_name,
                dataset_shape=dataset_shape,
                target_col=target_col,
                train_time_s=m.get("train_time_s", 0),
            )

    def _extract_params(self, estimator) -> Dict:
        if estimator is None:
            return {}
        try:
            params = estimator.get_params()
            # Keep only simple types for JSON
            return {k: v for k, v in params.items()
                    if isinstance(v, (int, float, str, bool, type(None)))}
        except Exception:
            return {}

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def get_runs_df(self) -> pd.DataFrame:
        if not self.runs:
            return pd.DataFrame()
        rows = []
        for r in self.runs:
            row = {
                "run_id": r["run_id"],
                "timestamp": r["timestamp"],
                "model": r["model"],
                "dataset": r["dataset_name"],
                "target": r["target_col"],
                "problem_type": r["problem_type"],
                "rows": r["dataset_rows"],
                "train_time_s": r["train_time_s"],
            }
            row.update(r.get("metrics", {}))
            rows.append(row)
        return pd.DataFrame(rows).sort_values("timestamp", ascending=False)

    def best_run(self, metric: str = "roc_auc") -> Optional[Dict]:
        if not self.runs:
            return None
        valid = [r for r in self.runs if metric in r.get("metrics", {})]
        if not valid:
            return None
        return max(valid, key=lambda r: r["metrics"][metric])

    def clear(self):
        self.runs = []
        if os.path.exists(TRACKER_PATH):
            os.remove(TRACKER_PATH)

    # ── Visualisations ────────────────────────────────────────────────────────

    def plot_metric_over_time(self, metric: str = "roc_auc") -> go.Figure:
        df = self.get_runs_df()
        if df.empty or metric not in df.columns:
            fig = go.Figure()
            fig.add_annotation(text="No experiments logged yet",
                               xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(title="Experiment History", template="plotly_white")
            return fig
        df = df.dropna(subset=[metric]).copy()
        df["run_label"] = df["timestamp"].str[-8:] + " " + df["model"]
        fig = px.line(df.sort_values("timestamp"), x="timestamp", y=metric,
                      color="model", markers=True,
                      title=f"{metric.upper()} Over Experiment Runs",
                      template="plotly_white",
                      labels={"timestamp": "Run time", metric: metric.upper()})
        fig.update_layout(height=380)
        return fig

    def plot_model_comparison(self) -> go.Figure:
        df = self.get_runs_df()
        if df.empty:
            return go.Figure()
        metric_cols = [c for c in ["roc_auc", "accuracy", "f1", "r2", "mae"]
                       if c in df.columns]
        if not metric_cols:
            return go.Figure()
        metric = metric_cols[0]
        best_per_model = df.groupby("model")[metric].max().reset_index().sort_values(metric)
        fig = px.bar(best_per_model, x=metric, y="model", orientation="h",
                     title=f"Best {metric.upper()} per Model (all experiments)",
                     color=metric, color_continuous_scale="Blues",
                     template="plotly_white",
                     labels={metric: metric.upper(), "model": "Model"})
        fig.update_layout(height=350, coloraxis_showscale=False)
        return fig
