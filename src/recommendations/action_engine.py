"""
Action recommendation engine — uses shared gemini_client.
"""
import json
from typing import Dict, List, Optional
import pandas as pd
from utils.config import GOOGLE_API_KEY, GEMINI_MODEL
from utils.logger import get_logger

logger = get_logger(__name__)

RISK_LEVELS = {
    "critical": (0.80, 1.01), "high": (0.60, 0.80),
    "medium":   (0.40, 0.60), "low":  (0.20, 0.40), "minimal": (0.00, 0.20),
}
RULE_ACTIONS = {
    "critical": ["Assign dedicated account manager immediately.",
                 "Offer personalised discount of 20–30%.",
                 "Escalate to senior retention team within 24 hours.",
                 "Schedule direct call to address pain points."],
    "high":     ["Send personalised retention email within 48 hours.",
                 "Offer service upgrade or loyalty bonus.",
                 "Provide one-time bill credit or discount."],
    "medium":   ["Enrol customer in proactive loyalty programme.",
                 "Send satisfaction survey to identify concerns.",
                 "Offer product feature walkthrough or tutorial."],
    "low":      ["Include in next monthly newsletter.",
                 "Monitor engagement metrics over next 30 days."],
    "minimal":  ["No immediate action required.",
                 "Maintain standard engagement cadence."],
}

def _get_risk_level(probability: float) -> str:
    for level,(lo,hi) in RISK_LEVELS.items():
        if lo <= probability < hi: return level
    return "minimal"

def generate_actions(probability: float, feature_context: Optional[Dict]=None, use_llm: bool=True) -> Dict:
    risk    = _get_risk_level(probability)
    actions = RULE_ACTIONS.get(risk, RULE_ACTIONS["minimal"])
    if use_llm and GOOGLE_API_KEY and feature_context:
        try:
            actions = _llm_actions(probability, risk, feature_context)
        except Exception as exc:
            logger.warning("LLM actions failed (%s) — using rule-based", exc)
    return {"risk_level": risk, "probability": round(probability,4),
            "priority": "URGENT" if risk in ("critical","high") else "NORMAL",
            "actions": actions}

def batch_recommendations(model, X_test: pd.DataFrame,
                          problem_type: str="classification", top_n: int=10) -> pd.DataFrame:
    if problem_type != "classification" or not hasattr(model,"predict_proba"):
        return pd.DataFrame()
    probas  = model.predict_proba(X_test)[:,1]
    indices = probas.argsort()[::-1][:top_n]
    rows = []
    for idx in indices:
        prob  = float(probas[idx])
        risk  = _get_risk_level(prob)
        acts  = RULE_ACTIONS.get(risk,[])
        rows.append({"row_index": int(idx), "probability": round(prob,4),
                     "risk_level": risk,
                     "priority": "URGENT" if risk in ("critical","high") else "NORMAL",
                     "top_action": acts[0] if acts else "No action"})
    return pd.DataFrame(rows)

def _llm_actions(probability: float, risk_level: str, feature_context: Dict) -> List[str]:
    from utils.gemini_client import _gemini_generate
    context_str = "\n".join(f"- {k}: {v}" for k,v in feature_context.items())
    prompt = f"""Customer retention specialist. Customer risk: {risk_level.upper()} ({probability:.1%}).
Profile:\n{context_str}
Generate exactly 3 specific actionable retention recommendations.
Return ONLY a valid JSON array of 3 strings. No preamble."""
    raw = _gemini_generate(prompt, max_tokens=200)
    raw = raw.replace("```json","").replace("```","").strip()
    return json.loads(raw)
