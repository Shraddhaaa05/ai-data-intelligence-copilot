"""
Model selection utilities.
Picks the best model from a leaderboard and provides selection rationale.
"""
from utils.logger import get_logger

logger = get_logger(__name__)


def select_best_model(leaderboard: list, problem_type: str) -> dict:
    """
    Return the top-ranked model from the leaderboard.
    The leaderboard should already be sorted best→worst by train_all().
    """
    if not leaderboard:
        raise ValueError("Leaderboard is empty — no models were trained successfully.")

    best = leaderboard[0]
    sort_key = "roc_auc" if problem_type == "classification" else "r2"
    logger.info(
        "Best model selected: %s (sort_key=%s, score=%.4f)",
        best["model"], sort_key, best.get(sort_key, 0),
    )
    return best


def get_selection_rationale(leaderboard: list, problem_type: str) -> str:
    """
    Return a plain-text explanation of why the best model was chosen.
    """
    if not leaderboard:
        return "No models available."

    best = leaderboard[0]
    sort_key = "roc_auc" if problem_type == "classification" else "r2"
    score = best.get(sort_key, 0)
    n_models = len(leaderboard)

    rationale = (
        f"**{best['model']}** was selected as the best model out of {n_models} candidates. "
        f"It achieved the highest {'ROC-AUC' if problem_type == 'classification' else 'R²'} "
        f"score of **{score:.4f}** on the hold-out test set."
    )

    if len(leaderboard) > 1:
        runner_up = leaderboard[1]
        runner_score = runner_up.get(sort_key, 0)
        diff = score - runner_score
        rationale += (
            f" The runner-up was **{runner_up['model']}** with a score of {runner_score:.4f} "
            f"({diff:+.4f} difference)."
        )

    return rationale
