import os
import io
import pickle
import base64
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-secret-key")
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
DEFAULT_CSV = os.environ.get("DATA_PATH", "data/sri_lanka_flood_risk_modeled.csv")
CONFIG_PATH = "eda_config.pkl"

DEFAULT_CONFIG = {
    "categorical_columns": ["district", "province", "climatic_zone", "flood_category"],
    "numeric_columns": [
        "precipitation_sum", "rain_sum", "rain_24h", "rain_48h", "rain_72h",
        "soil_moisture_0_to_7cm_mean", "soil_moisture_7_to_28cm_mean",
        "soil_saturation_index", "temperature_2m_max", "wind_speed_10m_max",
        "flood_risk_score"
    ],
    "target_column": "flood_category",
    "date_column": "date",
}

# This pickle stores dashboard configuration, not a trained prediction model.
if not os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "wb") as f:
        pickle.dump(DEFAULT_CONFIG, f)

with open(CONFIG_PATH, "rb") as f:
    CONFIG = pickle.load(f)

sns.set_theme(style="whitegrid")


def load_dataframe():
    """Load the most recently uploaded CSV, or the Docker-mounted default CSV."""
    uploaded = sorted(UPLOAD_DIR.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    path = uploaded[0] if uploaded else Path(DEFAULT_CSV)
    if path.exists():
        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df, str(path)
    return None, None


def chart_to_base64(fig):
    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def make_charts(df):
    charts = {}
    numeric = [c for c in CONFIG["numeric_columns"] if c in df.columns]
    if "flood_risk_score" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df["flood_risk_score"].dropna(), bins=35, kde=True, ax=ax, color="steelblue")
        ax.set_title("Flood Risk Score Distribution")
        charts["risk_distribution"] = chart_to_base64(fig)

        if "district" in df.columns:
            risk = df.groupby("district")["flood_risk_score"].mean().sort_values(ascending=False).head(15)
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x=risk.values, y=risk.index, ax=ax, color="indianred")
            ax.set_title("Top 15 Districts by Mean Flood Risk")
            ax.set_xlabel("Mean flood risk score")
            charts["district_risk"] = chart_to_base64(fig)

        if "flood_category" in df.columns:
            counts = df["flood_category"].fillna("Missing").value_counts()
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.barplot(x=counts.values, y=counts.index, ax=ax, color="darkorange")
            ax.set_title("Flood Risk Categories")
            ax.set_xlabel("Observations")
            charts["categories"] = chart_to_base64(fig)

    if "date" in df.columns and "flood_risk_score" in df.columns:
        monthly = (df.dropna(subset=["date"]).set_index("date")
                   .resample("ME")["flood_risk_score"].mean().dropna())
        if not monthly.empty:
            fig, ax = plt.subplots(figsize=(9, 4))
            monthly.plot(ax=ax, color="crimson", linewidth=1.7)
            ax.set_title("Monthly Average Flood Risk")
            ax.set_ylabel("Mean flood risk score")
            charts["monthly_risk"] = chart_to_base64(fig)

    if "precipitation_sum" in df.columns and "flood_risk_score" in df.columns:
        sample = df[["precipitation_sum", "flood_risk_score"]].dropna()
        if len(sample) > 5000:
            sample = sample.sample(5000, random_state=42)
        if not sample.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.scatterplot(data=sample, x="precipitation_sum", y="flood_risk_score",
                            alpha=0.55, s=20, ax=ax)
            ax.set_title("Precipitation vs Flood Risk")
            charts["rain_risk"] = chart_to_base64(fig)
    return charts


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("dataset")
        if not file or not file.filename:
            flash("Choose a CSV file first.", "error")
            return redirect(url_for("index"))
        if not file.filename.lower().endswith(".csv"):
            flash("Only CSV files are supported.", "error")
            return redirect(url_for("index"))
        safe_name = Path(file.filename).name
        file.save(UPLOAD_DIR / safe_name)
        flash(f"Dataset uploaded: {safe_name}", "success")
        return redirect(url_for("index"))

    df, source = load_dataframe()
    if df is None:
        return render_template("index.html", has_data=False, source=None, metrics={},
                               columns=[], preview=[], missing=[], charts={},
                               years=[], districts=[], selected_district="All",
                               category_rows=[], numeric_rows=[], messages=[])

    selected_district = request.args.get("district", "All")
    view = df.copy()
    if selected_district != "All" and "district" in view.columns:
        view = view[view["district"].astype(str) == selected_district]

    metrics = {
        "rows": f"{len(view):,}",
        "columns": str(view.shape[1]),
        "districts": str(view["district"].nunique()) if "district" in view.columns else "—",
        "provinces": str(view["province"].nunique()) if "province" in view.columns else "—",
        "missing": f"{int(view.isna().sum().sum()):,}",
        "period": "—",
    }
    if "date" in view.columns and view["date"].notna().any():
        metrics["period"] = f'{view["date"].min():%Y-%m-%d} – {view["date"].max():%Y-%m-%d}'

    columns = list(view.columns)
    preview_df = view.head(10).copy()
    if "date" in preview_df.columns:
        preview_df["date"] = preview_df["date"].dt.strftime("%Y-%m-%d")
    preview = preview_df.replace({np.nan: ""}).to_dict(orient="records")
    missing_df = pd.DataFrame({
        "column": view.columns,
        "missing": view.isna().sum().values,
        "missing_pct": (view.isna().mean().values * 100).round(2),
    }).sort_values("missing", ascending=False)
    missing = missing_df.head(12).to_dict(orient="records")

    category_rows = []
    if "flood_category" in view.columns:
        category_rows = [{"category": str(k), "count": int(v)}
                         for k, v in view["flood_category"].value_counts(dropna=False).items()]
    numeric_rows = []
    num_cols = view.select_dtypes(include=np.number).columns
    if len(num_cols):
        stats = view[num_cols].describe().T
        numeric_rows = [{"feature": str(idx), "mean": round(float(row["mean"]), 3) if pd.notna(row["mean"]) else "—",
                         "std": round(float(row["std"]), 3) if pd.notna(row["std"]) else "—",
                         "min": round(float(row["min"]), 3) if pd.notna(row["min"]) else "—",
                         "max": round(float(row["max"]), 3) if pd.notna(row["max"]) else "—"}
                        for idx, row in stats.iterrows()]

    districts = sorted(df["district"].dropna().astype(str).unique().tolist()) if "district" in df.columns else []
    charts = make_charts(view)
    return render_template("index.html", has_data=True, source=source, metrics=metrics,
                           columns=columns, preview=preview, missing=missing, charts=charts,
                           years=sorted(df["year"].dropna().unique().tolist()) if "year" in df.columns else [],
                           districts=districts, selected_district=selected_district,
                           category_rows=category_rows, numeric_rows=numeric_rows)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
