from pathlib import Path
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import os


# ============================================================
# APPLICATION
# ============================================================

app = Flask(__name__)
CORS(app)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PROCESSED_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)


# ============================================================
# FILE PATHS
# ============================================================

PREDICTIONS_FILE = (
    PROCESSED_DIR
    / "xgboost_predictions.csv"
)

RISK_FILE = (
    PROCESSED_DIR
    / "inventory_risk_scores.csv"
)

REORDER_FILE = (
    PROCESSED_DIR
    / "reorder_priority_list.csv"
)

MARKDOWN_FILE = (
    PROCESSED_DIR
    / "markdown_priority_list.csv"
)

SKU_FILE = (
    PROCESSED_DIR
    / "sku_master_clean.csv"
)

METRICS_FILE = (
    PROCESSED_DIR
    / "model_metrics.csv"
)


# ============================================================
# GLOBAL DATA
# ============================================================

predictions_df = pd.DataFrame()
risk_df = pd.DataFrame()
reorder_df = pd.DataFrame()
markdown_df = pd.DataFrame()
sku_df = pd.DataFrame()
metrics_df = pd.DataFrame()


# ============================================================
# DATA LOADING
# ============================================================

def clean_columns(df):

    if df.empty:
        return df

    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return df


def load_csv(path):

    if not path.exists():

        print(f"WARNING: File not found: {path}")

        return pd.DataFrame()

    try:

        df = pd.read_csv(path)

        df = clean_columns(df)

        return df

    except Exception as e:

        print(
            f"ERROR loading {path}: {e}"
        )

        return pd.DataFrame()


def load_all_data():

    global predictions_df
    global risk_df
    global reorder_df
    global markdown_df
    global sku_df
    global metrics_df

    predictions_df = load_csv(
        PREDICTIONS_FILE
    )

    risk_df = load_csv(
        RISK_FILE
    )

    reorder_df = load_csv(
        REORDER_FILE
    )

    markdown_df = load_csv(
        MARKDOWN_FILE
    )

    sku_df = load_csv(
        SKU_FILE
    )

    metrics_df = load_csv(
        METRICS_FILE
    )

    print("=" * 70)
    print("NORTHBAY FORESIGHT API")
    print("=" * 70)

    print(
        f"Predictions : {len(predictions_df):,}"
    )

    print(
        f"Risk rows   : {len(risk_df):,}"
    )

    print(
        f"Reorder     : {len(reorder_df):,}"
    )

    print(
        f"Markdown    : {len(markdown_df):,}"
    )

    print(
        f"SKU master  : {len(sku_df):,}"
    )

    print("=" * 70)


# ============================================================
# JSON CLEANER
# ============================================================

def make_json_safe(df):

    if df.empty:
        return []

    output = df.copy()

    output = output.replace(
        [np.inf, -np.inf],
        np.nan
    )

    output = output.where(
        pd.notnull(output),
        None
    )

    return output.to_dict(
        orient="records"
    )


# ============================================================
# HEALTH
# ============================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify(
        {
            "status": "online",
            "service": (
                "NorthBay FORESIGHT "
                "Scoring API"
            ),
            "model": "XGBoost",
            "forecast_wape": "46.59%",
            "baseline_wape": "59.62%",
            "prediction_rows": int(
                len(predictions_df)
            ),
            "risk_rows": int(
                len(risk_df)
            ),
            "endpoints": [
                "/health",
                "/forecast",
                "/risk",
                "/sku/<sku_id>",
                "/reorder",
                "/markdown",
                "/metrics"
            ]
        }
    )


# ============================================================
# FORECAST
# ============================================================

@app.route(
    "/forecast",
    methods=["GET"]
)
def forecast():

    if predictions_df.empty:

        return jsonify(
            {
                "status": "error",
                "message": (
                    "Forecast data unavailable."
                )
            }
        ), 503

    df = predictions_df.copy()

    # --------------------------------------------------------
    # Optional filters
    # --------------------------------------------------------

    sku_id = request.args.get(
        "sku_id"
    )

    store_id = request.args.get(
        "store_id"
    )

    limit = request.args.get(
        "limit",
        default=5000,
        type=int
    )

    if sku_id:

        df = df[
            df["sku_id"]
            .astype(str)
            == str(sku_id)
        ]

    if (
        store_id
        and "store_id" in df.columns
    ):

        df = df[
            df["store_id"]
            .astype(str)
            == str(store_id)
        ]

    df = df.head(
        min(limit, 10000)
    )

    return jsonify(
        {
            "status": "success",
            "model": "XGBoost",
            "count": len(df),
            "forecast": make_json_safe(df)
        }
    )


# ============================================================
# RISK
# ============================================================

@app.route(
    "/risk",
    methods=["GET"]
)
def risk():

    if risk_df.empty:

        return jsonify(
            {
                "status": "error",
                "message": (
                    "Risk data unavailable."
                )
            }
        ), 503

    df = risk_df.copy()

    risk_level = request.args.get(
        "risk_level"
    )

    sku_id = request.args.get(
        "sku_id"
    )

    category = request.args.get(
        "category"
    )

    limit = request.args.get(
        "limit",
        default=500,
        type=int
    )

    if risk_level:

        df = df[
            df["risk_level"]
            .astype(str)
            .str.upper()
            == risk_level.upper()
        ]

    if sku_id:

        df = df[
            df["sku_id"]
            .astype(str)
            == str(sku_id)
        ]

    if (
        category
        and "category" in df.columns
    ):

        df = df[
            df["category"]
            .astype(str)
            == category
        ]

    if "risk_score" in df.columns:

        df["risk_score"] = pd.to_numeric(
            df["risk_score"],
            errors="coerce"
        )

        df = df.sort_values(
            "risk_score",
            ascending=False
        )

    df = df.head(
        min(limit, 5000)
    )

    return jsonify(
        {
            "status": "success",
            "count": len(df),
            "risk": make_json_safe(df)
        }
    )


# ============================================================
# SKU
# ============================================================

@app.route(
    "/sku/<sku_id>",
    methods=["GET"]
)
def sku_details(sku_id):

    sku_id = str(sku_id).strip()

    forecast = pd.DataFrame()

    risk = pd.DataFrame()

    sku_information = pd.DataFrame()

    if not predictions_df.empty:

        forecast = predictions_df[
            predictions_df["sku_id"]
            .astype(str)
            == sku_id
        ].copy()

    if not risk_df.empty:

        risk = risk_df[
            risk_df["sku_id"]
            .astype(str)
            == sku_id
        ].copy()

    if not sku_df.empty:

        sku_information = sku_df[
            sku_df["sku_id"]
            .astype(str)
            == sku_id
        ].copy()

    if (
        forecast.empty
        and risk.empty
        and sku_information.empty
    ):

        return jsonify(
            {
                "status": "error",
                "message": (
                    f"SKU '{sku_id}' "
                    "was not found."
                )
            }
        ), 404

    return jsonify(
        {
            "status": "success",
            "sku_id": sku_id,
            "sku_info": make_json_safe(
                sku_information
            ),
            "forecast": make_json_safe(
                forecast
            ),
            "risk": make_json_safe(
                risk
            )
        }
    )


# ============================================================
# REORDER
# ============================================================

@app.route(
    "/reorder",
    methods=["GET"]
)
def reorder():

    if reorder_df.empty:

        return jsonify(
            {
                "status": "error",
                "message": (
                    "Reorder data unavailable."
                )
            }
        ), 503

    df = reorder_df.copy()

    limit = request.args.get(
        "limit",
        default=100,
        type=int
    )

    if "risk_score" in df.columns:

        df["risk_score"] = pd.to_numeric(
            df["risk_score"],
            errors="coerce"
        )

        df = df.sort_values(
            "risk_score",
            ascending=False
        )

    df = df.head(
        min(limit, 1000)
    )

    return jsonify(
        {
            "status": "success",
            "count": len(df),
            "reorder": make_json_safe(df)
        }
    )


# ============================================================
# MARKDOWN
# ============================================================

@app.route(
    "/markdown",
    methods=["GET"]
)
def markdown():

    if markdown_df.empty:

        return jsonify(
            {
                "status": "error",
                "message": (
                    "Markdown data unavailable."
                )
            }
        ), 503

    df = markdown_df.copy()

    limit = request.args.get(
        "limit",
        default=100,
        type=int
    )

    if "risk_score" in df.columns:

        df["risk_score"] = pd.to_numeric(
            df["risk_score"],
            errors="coerce"
        )

        df = df.sort_values(
            "risk_score",
            ascending=False
        )

    df = df.head(
        min(limit, 1000)
    )

    return jsonify(
        {
            "status": "success",
            "count": len(df),
            "markdown": make_json_safe(df)
        }
    )


# ============================================================
# MODEL METRICS
# ============================================================

@app.route(
    "/metrics",
    methods=["GET"]
)
def metrics():

    result = {
        "model": "XGBoost",
        "xgboost_wape": 46.59,
        "seasonal_naive_wape": 59.62,
        "mae": 0.8854,
        "bias": 0.0022
    }

    if not metrics_df.empty:

        records = make_json_safe(
            metrics_df
        )

        result["stored_metrics"] = records

    return jsonify(
        {
            "status": "success",
            "metrics": result
        }
    )


# ============================================================
# ROOT
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
def root():

    return jsonify(
        {
            "service": (
                "NorthBay FORESIGHT "
                "Scoring API"
            ),
            "status": "online",
            "model": "XGBoost",
            "forecast_wape": "46.59%",
            "message": (
                "API is running successfully."
            )
        }
    )


# ============================================================
# STARTUP
# ============================================================

load_all_data()


if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
