"""
PDF report generator using ReportLab.
Produces a multi-section analytics report from session data.
"""
import io
import os
from datetime import datetime
from typing import Optional

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from utils.logger import get_logger

logger = get_logger(__name__)

# ── Colour palette ────────────────────────────────────────────────────────────
PRIMARY = colors.HexColor("#1a56db")
SECONDARY = colors.HexColor("#e1effe")
DARK = colors.HexColor("#1e2a3b")
LIGHT_GRAY = colors.HexColor("#f8f9fa")
MID_GRAY = colors.HexColor("#6b7280")
ACCENT = colors.HexColor("#10b981")


def _styles():
    base = getSampleStyleSheet()
    custom = {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontSize=26,
            textColor=DARK,
            spaceAfter=6,
            alignment=TA_CENTER,
            fontName="Helvetica-Bold",
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=base["Normal"],
            fontSize=12,
            textColor=MID_GRAY,
            spaceAfter=20,
            alignment=TA_CENTER,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontSize=16,
            textColor=PRIMARY,
            spaceBefore=18,
            spaceAfter=8,
            fontName="Helvetica-Bold",
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontSize=13,
            textColor=DARK,
            spaceBefore=12,
            spaceAfter=6,
            fontName="Helvetica-Bold",
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["Normal"],
            fontSize=10,
            textColor=DARK,
            spaceAfter=8,
            leading=16,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=base["Normal"],
            fontSize=10,
            textColor=DARK,
            spaceAfter=4,
            leftIndent=16,
            bulletIndent=6,
            leading=14,
        ),
        "metric_label": ParagraphStyle(
            "metric_label",
            parent=base["Normal"],
            fontSize=9,
            textColor=MID_GRAY,
            alignment=TA_CENTER,
        ),
        "metric_value": ParagraphStyle(
            "metric_value",
            parent=base["Normal"],
            fontSize=18,
            textColor=PRIMARY,
            alignment=TA_CENTER,
            fontName="Helvetica-Bold",
        ),
    }
    return custom


def _table_style(header_bg=PRIMARY, row_bg=LIGHT_GRAY):
    return TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, row_bg]),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("TEXTCOLOR", (0, 1), (-1, -1), DARK),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#d1d5db")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("ROUNDEDCORNERS", [4, 4, 4, 4]),
    ])


def generate_pdf_report(
    df: pd.DataFrame,
    schema,
    leaderboard: list,
    shap_importance_df: pd.DataFrame,
    insights: list[str],
    problem_type: str,
    target_col: str,
    shap_summary_img: Optional[bytes] = None,
    dataset_name: str = "Dataset",
) -> bytes:
    """
    Build a multi-page PDF analytics report.

    Returns raw PDF bytes ready for st.download_button.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )

    S = _styles()
    story = []
    page_w = A4[0] - 40 * mm  # usable width

    # ── Cover ─────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 30 * mm))
    story.append(Paragraph("AI Data Intelligence Copilot", S["title"]))
    story.append(Paragraph("Automated Analytics Report", S["subtitle"]))
    story.append(HRFlowable(width="100%", thickness=2, color=PRIMARY))
    story.append(Spacer(1, 6))

    meta = [
        ["Dataset", dataset_name],
        ["Target column", target_col],
        ["Problem type", problem_type.capitalize()],
        ["Rows", f"{df.shape[0]:,}"],
        ["Columns", str(df.shape[1])],
        ["Generated", datetime.now().strftime("%d %b %Y, %H:%M")],
    ]
    meta_table = Table(meta, colWidths=[page_w * 0.35, page_w * 0.65])
    meta_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), PRIMARY),
        ("TEXTCOLOR", (1, 0), (1, -1), DARK),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, LIGHT_GRAY]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(Spacer(1, 8 * mm))
    story.append(meta_table)
    story.append(PageBreak())

    # ── Section 1: Dataset Summary ────────────────────────────────────────────
    story.append(Paragraph("1. Dataset Summary", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e5e7eb")))

    # Key metrics row
    numeric_count = len(schema.numeric_cols)
    cat_count = len(schema.categorical_cols)
    missing_pct = round(df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100, 1)

    metrics = [
        ["Rows", "Numeric cols", "Categorical cols", "Missing %"],
        [
            f"{df.shape[0]:,}",
            str(numeric_count),
            str(cat_count),
            f"{missing_pct}%",
        ],
    ]
    m_table = Table(metrics, colWidths=[page_w / 4] * 4)
    m_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), SECONDARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), PRIMARY),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 1), (-1, 1), 20),
        ("TEXTCOLOR", (0, 1), (-1, 1), PRIMARY),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#d1d5db")),
        ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#e5e7eb")),
    ]))
    story.append(Spacer(1, 4 * mm))
    story.append(m_table)

    # Statistical summary table
    story.append(Paragraph("Statistical Summary (numeric columns)", S["h2"]))
    try:
        desc = df[schema.numeric_cols[:8]].describe().round(3)
        desc_data = [[""] + list(desc.columns)]
        for idx, row in desc.iterrows():
            desc_data.append([str(idx)] + [str(v) for v in row.values])
        desc_table = Table(desc_data, colWidths=[30 * mm] + [page_w / max(len(desc.columns), 1) - 5] * len(desc.columns))
        desc_table.setStyle(_table_style())
        story.append(desc_table)
    except Exception as e:
        story.append(Paragraph(f"Could not generate statistical summary: {e}", S["body"]))

    # Missing values
    if schema.missing_cols:
        story.append(Paragraph("Missing Values", S["h2"]))
        mv_data = [["Column", "Missing Count", "Missing %"]]
        for col in schema.missing_cols:
            cnt = df[col].isna().sum()
            pct = round(cnt / len(df) * 100, 2)
            mv_data.append([col, str(cnt), f"{pct}%"])
        mv_table = Table(mv_data, colWidths=[page_w * 0.5, page_w * 0.25, page_w * 0.25])
        mv_table.setStyle(_table_style())
        story.append(mv_table)

    story.append(PageBreak())

    # ── Section 2: Model Leaderboard ──────────────────────────────────────────
    story.append(Paragraph("2. AutoML Model Leaderboard", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e5e7eb")))
    story.append(Spacer(1, 4 * mm))

    if leaderboard:
        best = leaderboard[0]
        story.append(Paragraph(
            f"Best model: <b>{best['model']}</b> — selected by highest "
            f"{'ROC-AUC' if problem_type == 'classification' else 'R²'} score.",
            S["body"]
        ))

        # Build leaderboard table
        metric_cols = [k for k in best.keys() if k not in ("model", "estimator")]
        lb_data = [["Rank", "Model"] + [m.upper() for m in metric_cols]]
        for rank, m in enumerate(leaderboard, 1):
            lb_data.append(
                [str(rank), m["model"]] + [str(m.get(k, "—")) for k in metric_cols]
            )
        col_w = page_w / (len(lb_data[0]))
        lb_table = Table(lb_data, colWidths=[col_w] * len(lb_data[0]))
        style = _table_style()
        # Highlight best model row
        style.add("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#d1fae5"))
        style.add("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold")
        lb_table.setStyle(style)
        story.append(lb_table)

    story.append(PageBreak())

    # ── Section 3: Feature Importance ────────────────────────────────────────
    story.append(Paragraph("3. Feature Importance (SHAP)", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e5e7eb")))
    story.append(Spacer(1, 4 * mm))

    if shap_summary_img:
        story.append(Paragraph(
            "The chart below shows mean absolute SHAP values — the average impact "
            "each feature has on model output across all predictions.",
            S["body"]
        ))
        img = Image(io.BytesIO(shap_summary_img), width=page_w, height=page_w * 0.55)
        story.append(img)
        story.append(Spacer(1, 4 * mm))

    if not shap_importance_df.empty:
        story.append(Paragraph("Top 10 Features by SHAP Importance", S["h2"]))
        top10 = shap_importance_df.head(10).copy()
        fi_data = [["Rank", "Feature", "Mean |SHAP|"]]
        for i, (_, row) in enumerate(top10.iterrows(), 1):
            fi_data.append([str(i), str(row["feature"]), f"{row['importance']:.5f}"])
        fi_table = Table(fi_data, colWidths=[page_w * 0.1, page_w * 0.6, page_w * 0.3])
        fi_table.setStyle(_table_style())
        story.append(fi_table)

    story.append(PageBreak())

    # ── Section 4: AI Business Insights ───────────────────────────────────────
    story.append(Paragraph("4. AI-Generated Business Insights", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e5e7eb")))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(
        "The following insights were automatically generated by analysing dataset "
        "patterns, model feature importances, and statistical correlations.",
        S["body"]
    ))

    for i, insight in enumerate(insights, 1):
        story.append(Paragraph(f"• {insight}", S["bullet"]))

    story.append(PageBreak())

    # ── Section 5: Recommendations ────────────────────────────────────────────
    story.append(Paragraph("5. Recommended Next Steps", S["h1"]))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#e5e7eb")))
    story.append(Spacer(1, 4 * mm))

    recs = [
        "Review top SHAP features with domain experts to validate business logic.",
        "Investigate missing value patterns — imputation strategy may affect model accuracy.",
        "Run what-if simulations on high-risk predictions to identify intervention points.",
        "Consider hyperparameter tuning on the best-performing model using Optuna.",
        "Set up model monitoring to detect feature drift in production data.",
        "Re-train the model periodically as new data becomes available.",
    ]
    for rec in recs:
        story.append(Paragraph(f"• {rec}", S["bullet"]))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 20 * mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=MID_GRAY))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        f"Generated by AI Data Intelligence Copilot · {datetime.now().strftime('%d %b %Y')}",
        ParagraphStyle("footer", fontSize=8, textColor=MID_GRAY, alignment=TA_CENTER),
    ))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    logger.info("PDF report generated — %d bytes", len(pdf_bytes))
    return pdf_bytes
