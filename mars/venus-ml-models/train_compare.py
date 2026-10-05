"""Decision Tree vs Random Forest vs XGBoost, compared with cross-validation.

Same workflow I followed at MaRS. The company data stays with the company, so
this generates a synthetic project-task dataset with a similar shape to walk
through the steps: cleaning, feature engineering, comparison, tuning.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

NUMERIC = ["planned_days", "crew_size", "material_ready_pct", "rain_days",
           "dependencies", "monsoon", "days_per_worker"]


def make_data(n=1500, seed=7):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "planned_days": rng.integers(3, 60, n),
        "crew_size": rng.integers(2, 30, n),
        "material_ready_pct": rng.uniform(40, 100, n).round(1),
        "rain_days": rng.poisson(2, n),
        "dependencies": rng.integers(0, 6, n),
        "task_type": rng.choice(["excavation", "foundation", "structure", "finishing"], n),
        "start_month": rng.integers(1, 13, n),
    })
    # real data had gaps, so add some missing values
    df.loc[rng.random(n) < 0.05, "material_ready_pct"] = np.nan
    risk = (0.04 * df.planned_days - 0.05 * df.crew_size
            - 0.04 * df.material_ready_pct.fillna(70) + 0.5 * df.rain_days
            + 0.4 * df.dependencies + (df.task_type == "structure") * 0.8
            + df.start_month.isin([6, 7, 8, 9]) * 0.7 + rng.normal(0, 1, n))
    df["delayed"] = (risk > np.quantile(risk, 0.65)).astype(int)
    return df


def add_features(df):
    df = df.copy()
    df["monsoon"] = df.start_month.isin([6, 7, 8, 9]).astype(int)  # month -> season flag
    df["days_per_worker"] = df.planned_days / df.crew_size          # ratio feature
    return df


def build(model):
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["task_type"]),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def split_xy(df):
    df = add_features(df)
    return df.drop(columns=["delayed"]), df["delayed"]


def compare(df, cv=5):
    X, y = split_xy(df)
    folds = StratifiedKFold(cv, shuffle=True, random_state=42)
    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8,
                                                random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                 eval_metric="logloss", random_state=42),
    }
    return {name: float(cross_val_score(build(m), X, y, cv=folds, scoring="f1").mean())
            for name, m in models.items()}


def tune_xgb(df):
    X, y = split_xy(df)
    grid = GridSearchCV(
        build(XGBClassifier(eval_metric="logloss", random_state=42)),
        {"model__max_depth": [3, 5],
         "model__learning_rate": [0.05, 0.1],
         "model__n_estimators": [150, 300]},
        cv=3, scoring="f1", n_jobs=-1)
    grid.fit(X, y)
    return grid.best_params_, float(grid.best_score_)


if __name__ == "__main__":
    data = make_data()
    for name, score in compare(data).items():
        print(f"{name:14s} mean F1 = {score:.3f}")
    params, score = tune_xgb(data)
    print("tuned XGBoost:", params, f"F1 = {score:.3f}")
