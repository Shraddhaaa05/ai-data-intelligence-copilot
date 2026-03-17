"""
FIXED Trainer Module
Key improvements:
1. Class weight handling for imbalanced data
2. Stratified cross-validation
3. Proper metric calculation
4. Removed data leakage in model training
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.model_selection import cross_validate, StratifiedKFold, KFold
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_squared_error, mean_absolute_error
)
import numpy as np
import pandas as pd
from utils.logger import get_logger

logger = get_logger(__name__)

# ── CLASSIFIERS (with class_weight for imbalance) ─────────────────────────────
CLASSIFIERS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced',  # ✅ Handle imbalance
        solver='lbfgs'
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'  # ✅ Handle imbalance
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        random_state=42,
        learning_rate=0.1,
        max_depth=5
    ),
    "XGBoost": XGBClassifier(
        n_estimators=100,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=1,  # ✅ Will be computed from class balance
        max_depth=5,
        learning_rate=0.1
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=100,
        random_state=42,
        num_leaves=31,
        learning_rate=0.1,
        verbose=-1,
        is_unbalance=True  # ✅ Handle imbalance
    ),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        random_state=42,
        class_weight='balanced'  # ✅ Handle imbalance
    ),
}

# ── REGRESSORS ────────────────────────────────────────────────────────────────
REGRESSORS = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        max_depth=15
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        random_state=42,
        learning_rate=0.1,
        max_depth=5
    ),
    "XGBoost": XGBRegressor(
        n_estimators=100,
        random_state=42,
        max_depth=5,
        learning_rate=0.1
    ),
    "LightGBM": LGBMRegressor(
        n_estimators=100,
        random_state=42,
        num_leaves=31,
        learning_rate=0.1,
        verbose=-1
    ),
}


def train_all(X_train, X_test, y_train, y_test, problem_type, progress_callback=None):
    """
    ✅ FIXED: Train all models with proper evaluation and no leakage
    """
    
    results = []
    models = CLASSIFIERS if problem_type == "classification" else REGRESSORS
    n_models = len(models)
    
    for i, (model_name, model) in enumerate(models.items()):
        if progress_callback:
            progress_callback(model_name, (i) / n_models)
        
        try:
            # ✅ Handle class imbalance in XGBoost
            if model_name == "XGBoost" and problem_type == "classification":
                # Calculate scale_pos_weight for binary classification
                neg_count = (y_train == 0).sum()
                pos_count = (y_train == 1).sum()
                if pos_count > 0:
                    model.scale_pos_weight = neg_count / pos_count
                    logger.info(f"XGBoost scale_pos_weight: {model.scale_pos_weight:.2f}")
            
            # Train on training set only
            import time
            t0 = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - t0
            
            # Evaluate on TEST set only (no training set metrics)
            y_pred = model.predict(X_test)
            
            result = {
                "model": model_name,
                "estimator": model,
                "train_time_s": train_time
            }
            
            if problem_type == "classification":
                result["accuracy"] = round(accuracy_score(y_test, y_pred), 4)
                result["precision"] = round(precision_score(y_test, y_pred, zero_division=0), 4)
                result["recall"] = round(recall_score(y_test, y_pred, zero_division=0), 4)
                result["f1"] = round(f1_score(y_test, y_pred, zero_division=0), 4)
                
                # ✅ ROC_AUC only for binary classification
                n_classes = len(np.unique(y_test))
                if n_classes == 2 and hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]
                    result["roc_auc"] = round(roc_auc_score(y_test, y_proba), 4)
                else:
                    result["roc_auc"] = "N/A"
                
                logger.info(f"✅ {model_name}: ROC_AUC={result.get('roc_auc','N/A')}, F1={result['f1']}")
            else:
                result["r2"] = round(r2_score(y_test, y_pred), 4)
                result["rmse"] = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
                result["mae"] = round(mean_absolute_error(y_test, y_pred), 4)
                logger.info(f"✅ {model_name}: R2={result['r2']}, RMSE={result['rmse']}")
            
            results.append(result)
        
        except Exception as e:
            logger.error(f"❌ {model_name} failed: {e}")
            continue
    
    if progress_callback:
        progress_callback("Done", 1.0)
    
    # Sort by best metric
    if problem_type == "classification":
        results = sorted(results, key=lambda x: x.get('roc_auc', -999), reverse=True)
    else:
        results = sorted(results, key=lambda x: x.get('r2', -999), reverse=True)
    
    return {
        "leaderboard": results,
        "best_model": results[0] if results else {},
    }


def train_with_cv(X_train, y_train, problem_type, n_splits=5):
    """
    ✅ OPTIONAL: Training with cross-validation for more robust evaluation
    Useful for small datasets
    """
    
    models = CLASSIFIERS if problem_type == "classification" else REGRESSORS
    cv_folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) \
               if problem_type == "classification" \
               else KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    results = []
    
    for model_name, model in models.items():
        try:
            # Determine scoring metrics
            if problem_type == "classification":
                scoring = {'roc_auc': 'roc_auc', 'accuracy': 'accuracy', 'f1': 'f1'}
            else:
                scoring = {'r2': 'r2', 'neg_mse': 'neg_mean_squared_error'}
            
            # Cross-validate
            cv_results = cross_validate(model, X_train, y_train, cv=cv_folds, 
                                       scoring=scoring, return_train_score=False)
            
            result = {
                "model": model_name,
                "estimator": model,
            }
            
            # Aggregate CV results
            for metric, scores in cv_results.items():
                if metric.startswith('test_'):
                    clean_metric = metric.replace('test_', '')
                    result[f"{clean_metric}_mean"] = round(scores.mean(), 4)
                    result[f"{clean_metric}_std"] = round(scores.std(), 4)
            
            results.append(result)
            logger.info(f"✅ {model_name} CV: {result}")
        
        except Exception as e:
            logger.error(f"❌ {model_name} CV failed: {e}")
            continue
    
    return sorted(results, key=lambda x: x.get('roc_auc_mean', x.get('r2_mean', -999)), 
                  reverse=True)