"""
Run this script once to generate the three sample datasets.
Usage: python data/sample_datasets/generate_samples.py
"""
import os
import numpy as np
import pandas as pd

np.random.seed(42)
OUT = os.path.dirname(os.path.abspath(__file__))


# ── 1. Telco Customer Churn ───────────────────────────────────────────────────

def make_telco(n=1000):
    tenure          = np.random.randint(0, 72, n)
    monthly_charges = np.round(np.random.uniform(18, 120, n), 2)
    total_charges   = np.round(monthly_charges * tenure + np.random.normal(0, 50, n), 2)
    contract        = np.random.choice(["Month-to-month", "One year", "Two year"], n,
                                        p=[0.55, 0.25, 0.20])
    internet        = np.random.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    payment         = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n)
    gender          = np.random.choice(["Male", "Female"], n)
    senior          = np.random.choice([0, 1], n, p=[0.84, 0.16])
    partner         = np.random.choice(["Yes", "No"], n)
    dependents      = np.random.choice(["Yes", "No"], n)
    paperless       = np.random.choice(["Yes", "No"], n)
    tech_support    = np.random.choice(["Yes", "No", "No internet service"], n)

    # Churn probability influenced by contract, charges, tenure
    churn_prob = (
        0.05
        + 0.40 * (contract == "Month-to-month")
        + 0.15 * (monthly_charges > 70)
        + 0.20 * (tenure < 6)
        - 0.10 * (internet == "No")
        + 0.05 * (senior == 1)
    )
    churn_prob = np.clip(churn_prob, 0.02, 0.92)
    churn = (np.random.uniform(0, 1, n) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customerID":       [f"CUST-{i:05d}" for i in range(n)],
        "gender":           gender,
        "SeniorCitizen":    senior,
        "Partner":          partner,
        "Dependents":       dependents,
        "tenure":           tenure,
        "Contract":         contract,
        "PaperlessBilling": paperless,
        "PaymentMethod":    payment,
        "InternetService":  internet,
        "TechSupport":      tech_support,
        "MonthlyCharges":   monthly_charges,
        "TotalCharges":     np.where(total_charges < 0, 0, total_charges),
        "Churn":            np.where(churn == 1, "Yes", "No"),
    })
    # Add a few missing values
    idx_missing = np.random.choice(df.index, size=20, replace=False)
    df.loc[idx_missing, "TotalCharges"] = np.nan
    return df


# ── 2. Titanic Survival ───────────────────────────────────────────────────────

def make_titanic(n=891):
    pclass   = np.random.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55])
    sex      = np.random.choice(["male", "female"], n, p=[0.65, 0.35])
    age      = np.where(np.random.uniform(0, 1, n) < 0.2, np.nan,
                        np.round(np.random.normal(29, 14, n).clip(1, 80), 1))
    sibsp    = np.random.choice([0, 1, 2, 3, 4], n, p=[0.68, 0.23, 0.06, 0.02, 0.01])
    parch    = np.random.choice([0, 1, 2, 3],    n, p=[0.76, 0.13, 0.09, 0.02])
    fare     = np.round(np.random.exponential(32, n) + 5, 2)
    embarked = np.random.choice(["S", "C", "Q", np.nan], n, p=[0.72, 0.19, 0.087, 0.003])
    cabin_vals = [f"{l}{r}" for l, r in zip(
                      np.random.choice(list("ABCDE"), n),
                      np.random.randint(1, 120, n).astype(str))]
    cabin    = np.where(np.random.uniform(0, 1, n) < 0.77, None, cabin_vals).astype(object)

    surv_prob = (
        0.15
        + 0.35 * (sex == "female")
        + 0.25 * (pclass == 1)
        + 0.10 * (pclass == 2)
        - 0.10 * (pclass == 3)
    )
    surv_prob = np.clip(surv_prob, 0.05, 0.95)
    survived = (np.random.uniform(0, 1, n) < surv_prob).astype(int)

    names = [f"Passenger_{i}" for i in range(n)]
    ticket = [f"TKT-{np.random.randint(1000, 99999)}" for _ in range(n)]

    return pd.DataFrame({
        "PassengerId": range(1, n + 1),
        "Survived":    survived,
        "Pclass":      pclass,
        "Name":        names,
        "Sex":         sex,
        "Age":         age,
        "SibSp":       sibsp,
        "Parch":       parch,
        "Ticket":      ticket,
        "Fare":        fare,
        "Cabin":       cabin,
        "Embarked":    embarked,
    })


# ── 3. Boston Housing ─────────────────────────────────────────────────────────

def make_boston(n=506):
    crim    = np.round(np.random.exponential(3.6, n), 4)
    zn      = np.round(np.random.choice([0] * 7 + list(range(10, 100, 5)), n), 1)
    indus   = np.round(np.random.uniform(0.46, 27.74, n), 2)
    chas    = np.random.choice([0, 1], n, p=[0.93, 0.07])
    nox     = np.round(np.random.uniform(0.38, 0.87, n), 3)
    rm      = np.round(np.random.normal(6.28, 0.70, n).clip(3.6, 8.8), 3)
    age     = np.round(np.random.uniform(2.9, 100, n), 1)
    dis     = np.round(np.random.uniform(1.13, 12.13, n), 4)
    rad     = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 24], n)
    tax     = np.random.choice(range(187, 711, 5), n)
    ptratio = np.round(np.random.uniform(12.6, 22.0, n), 1)
    b       = np.round(np.random.uniform(0.32, 396.9, n), 2)
    lstat   = np.round(np.random.uniform(1.73, 37.97, n), 2)

    # MEDV influenced by rooms, crime, distances
    medv = (
        20
        + 5  * (rm - 6)
        - 0.5 * crim
        + 3  * (1 - nox)
        - 0.3 * lstat
        + np.random.normal(0, 3, n)
    ).clip(5, 50).round(1)

    return pd.DataFrame({
        "CRIM":    crim, "ZN": zn, "INDUS": indus, "CHAS": chas,
        "NOX":     nox,  "RM": rm, "AGE":   age,   "DIS":  dis,
        "RAD":     rad,  "TAX": tax, "PTRATIO": ptratio,
        "B":       b,    "LSTAT": lstat, "MEDV": medv,
    })


# ── Write to CSV ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    telco = make_telco()
    telco.to_csv(os.path.join(OUT, "telco_churn.csv"), index=False)
    print(f"telco_churn.csv → {len(telco)} rows")

    titanic = make_titanic()
    titanic.to_csv(os.path.join(OUT, "titanic.csv"), index=False)
    print(f"titanic.csv → {len(titanic)} rows")

    boston = make_boston()
    boston.to_csv(os.path.join(OUT, "boston_housing.csv"), index=False)
    print(f"boston_housing.csv → {len(boston)} rows")

    print("\nAll sample datasets generated successfully.")
