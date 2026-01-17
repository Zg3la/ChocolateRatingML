import pandas as pd
import joblib
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import ElasticNet

# =========================
# PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = BASE_DIR / "dataset" / "flavorsofcocoa.csv"
MODEL_PATH = BASE_DIR / "choco_app" / "ml_model.pkl"

print("Učitavam CSV iz:", CSV_PATH)

df = pd.read_csv(CSV_PATH, encoding="cp1252")

print("Kolone u datasetu:")
print(df.columns)

# =========================
# CLEAN / TYPES
# =========================
df["Cocoa Percent"] = (
    df["Cocoa Percent"].astype(str).str.replace("%", "", regex=False)
)
df["Cocoa Percent"] = pd.to_numeric(df["Cocoa Percent"], errors="coerce")
df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

df["Company (Manufacturer)"] = df["Company (Manufacturer)"].fillna("Unknown")
df["Company Location"] = df["Company Location"].fillna("Unknown")
df["Country of Bean Origin"] = df["Country of Bean Origin"].fillna("Unknown")
df["Specific Bean Origin or Bar Name"] = df["Specific Bean Origin or Bar Name"].fillna("")
df["Ingredients"] = df["Ingredients"].fillna("")
df["Most Memorable Characteristics"] = df["Most Memorable Characteristics"].fillna("")

df = df.dropna(subset=["Cocoa Percent", "Rating"]).copy()


X = df[[
    "Company (Manufacturer)",
    "Cocoa Percent",
    "Company Location",
    "Country of Bean Origin",
    "Specific Bean Origin or Bar Name",
    "Ingredients",
    "Most Memorable Characteristics",
]].copy()

X = X.rename(columns={
    "Company (Manufacturer)": "company",
    "Cocoa Percent": "cocoa_percent",
    "Company Location": "company_location",
    "Country of Bean Origin": "bean_origin",
    "Specific Bean Origin or Bar Name": "bar_name",
    "Ingredients": "ingredients",
    "Most Memorable Characteristics": "characteristics",
})

X["text_all"] = (
    X["bar_name"].astype(str) + " " +
    X["ingredients"].astype(str) + " " +
    X["characteristics"].astype(str)
).str.strip()

y = df["Rating"].astype(float)

categorical = ["company", "company_location", "bean_origin"]
numeric = ["cocoa_percent"]
text_col = "text_all"


preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", StandardScaler(), numeric),
        ("txt", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=30000
        ), text_col),
    ]
)


X_train, X_test, y_train, y_test = train_test_split(
    X[categorical + numeric + [text_col]],
    y,
    test_size=0.2,
    random_state=42
)


pipe = Pipeline([
    ("preprocess", preprocessor),
    ("model", ElasticNet(max_iter=20000, random_state=42))
])

param_grid = {
    "model__alpha": [0.0005, 0.001, 0.002, 0.005, 0.01],
    "model__l1_ratio": [0.1, 0.2, 0.3, 0.5, 0.7]
}

gs = GridSearchCV(
    pipe,
    param_grid=param_grid,
    scoring="neg_mean_absolute_error",
    cv=5,
    n_jobs=-1
)

gs.fit(X_train, y_train)

best_model = gs.best_estimator_
print("\n✅ Najbolji hiperparametri:", gs.best_params_)
print("✅ CV MAE:", round(-gs.best_score_, 3))


pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
r2 = r2_score(y_test, pred)

print("\n Rezultati modela :")
print("MAE: ", round(mae, 3))
print("RMSE:", round(rmse, 3))
print("R2:  ", round(r2, 3))

joblib.dump(best_model, MODEL_PATH)
print("\n✅ Model spremljen u:")

print(MODEL_PATH)
