from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


FEATURE_FILE = (
    PROCESSED_DIR /
    "model_features.csv"
)

TARGET = "units_sold"
TEST_DAYS = 30


def load_data():

    print("\nLoading feature dataset...")

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found:\n"
            f"{FEATURE_FILE}"
        )

    df = pd.read_csv(
        FEATURE_FILE
    )

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if "date" not in df.columns:
        raise ValueError(
            "The feature dataset does not contain "
            "'date' column.\n\n"
            f"Available columns:\n"
            f"{df.columns.tolist()}"
        )

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' "
            f"not found in dataset."
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["date"]
    )

    print(
        f"Rows    : {len(df):,}"
    )

    print(
        f"Columns : {len(df.columns)}"
    )

    print(
        f"Date range: "
        f"{df['date'].min().date()} "
        f"to "
        f"{df['date'].max().date()}"
    )

    return df


def prepare_features(df):

    print("\nPreparing model features...")

    df = df.copy()

    df = df.sort_values(
        [
            "date",
            "sku_id",
            "store_id"
        ]
    ).reset_index(
        drop=True
    )

    excluded_columns = [
        "units_sold",
        "revenue",
        "date",
        "sku_id",
        "store_id",
        "category",
        "subcategory",
        "brand",
        "last_restock_date",
        "month_name",
        "day_name",
        "season",
        "promo_event"
    ]

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded_columns
    ]

    X = df[
        feature_columns
    ].copy()

    y = df[
        TARGET
    ].copy()

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    categorical_columns = (
        X.select_dtypes(
            include=[
                "object",
                "string",
                "category"
            ]
        ).columns
    )

    for column in categorical_columns:

        X[column] = (
            X[column]
            .astype("category")
            .cat.codes
        )

    X = X.fillna(0)

    print(
        f"Model features: "
        f"{len(feature_columns)}"
    )

    return (
        df,
        X,
        y,
        feature_columns
    )


def create_time_split(
    df,
    X,
    y
):

    print(
        "\nCreating time-based "
        "train/test split..."
    )

    max_date = df["date"].max()

    test_start = (
        max_date -
        pd.Timedelta(
            days=TEST_DAYS - 1
        )
    )

    train_mask = (
        df["date"] < test_start
    )

    test_mask = (
        df["date"] >= test_start
    )

    X_train = X.loc[
        train_mask
    ]

    X_test = X.loc[
        test_mask
    ]

    y_train = y.loc[
        train_mask
    ]

    y_test = y.loc[
        test_mask
    ]

    test_rows = df.loc[
        test_mask
    ].copy()

    print(
        f"Train period: "
        f"{df.loc[train_mask, 'date'].min().date()} "
        f"to "
        f"{df.loc[train_mask, 'date'].max().date()}"
    )

    print(
        f"Test period : "
        f"{df.loc[test_mask, 'date'].min().date()} "
        f"to "
        f"{df.loc[test_mask, 'date'].max().date()}"
    )

    print(
        f"Train rows: "
        f"{len(X_train):,}"
    )

    print(
        f"Test rows : "
        f"{len(X_test):,}"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        test_rows
    )


def calculate_wape(
    actual,
    predicted
):

    actual = np.asarray(
        actual
    )

    predicted = np.asarray(
        predicted
    )

    absolute_error = np.abs(
        actual - predicted
    ).sum()

    total_actual = np.abs(
        actual
    ).sum()

    if total_actual == 0:
        return np.nan

    return (
        absolute_error /
        total_actual
    )


def calculate_bias(
    actual,
    predicted
):

    actual = np.asarray(
        actual
    )

    predicted = np.asarray(
        predicted
    )

    total_actual = actual.sum()

    if total_actual == 0:
        return np.nan

    return (
        predicted - actual
    ).sum() / total_actual


def train_model(
    X_train,
    y_train
):

    print(
        "\nTraining XGBoost model..."
    )

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train,
        verbose=False
    )

    print(
        "✓ Model training completed"
    )

    return model


def evaluate_model(
    model,
    X_test,
    y_test
):

    print(
        "\nGenerating predictions..."
    )

    predictions = model.predict(
        X_test
    )

    predictions = np.clip(
        predictions,
        a_min=0,
        a_max=None
    )

    actual = y_test.to_numpy()

    wape = calculate_wape(
        actual,
        predictions
    )

    mae = mean_absolute_error(
        actual,
        predictions
    )

    bias = calculate_bias(
        actual,
        predictions
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "XGBOOST FORECAST RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"WAPE : {wape:.4f} "
        f"({wape * 100:.2f}%)"
    )

    print(
        f"MAE  : {mae:.4f}"
    )

    print(
        f"Bias : {bias:.4f}"
    )

    return (
        predictions,
        wape,
        mae,
        bias
    )


def save_predictions(
    test_rows,
    predictions
):

    print(
        "\nSaving test predictions..."
    )

    prediction_output = test_rows[
        [
            "date",
            "store_id",
            "sku_id",
            "units_sold"
        ]
    ].copy()

    prediction_output[
        "prediction"
    ] = predictions

    prediction_output[
        "prediction"
    ] = prediction_output[
        "prediction"
    ].clip(
        lower=0
    )

    prediction_path = (
        PROCESSED_DIR /
        "xgboost_predictions.csv"
    )

    prediction_output.to_csv(
        prediction_path,
        index=False
    )

    print(
        f"✓ Predictions saved:"
    )

    print(
        prediction_path
    )

    print(
        f"Prediction rows: "
        f"{len(prediction_output):,}"
    )

    return prediction_output


def save_metrics(
    wape,
    mae,
    bias
):

    metrics = pd.DataFrame(
        [
            {
                "model": "XGBoost",
                "test_days": TEST_DAYS,
                "wape": wape,
                "mae": mae,
                "bias": bias
            }
        ]
    )

    metrics_path = (
        PROCESSED_DIR /
        "xgmetrics.csv"
    )

    metrics.to_csv(
        metrics_path,
        index=False
    )

    print(
        f"\n✓ Metrics saved:"
    )

    print(
        metrics_path
    )


def save_model(
    model,
    feature_columns
):

    model_path = (
        MODELS_DIR /
        "xgboost_demand_model.pkl"
    )

    metadata_path = (
        MODELS_DIR /
        "model_features.txt"
    )

    joblib.dump(
        model,
        model_path
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        for column in feature_columns:
            file.write(
                column + "\n"
            )

    print(
        f"\n✓ Model saved:"
    )

    print(
        model_path
    )

    print(
        f"✓ Feature list saved:"
    )

    print(
        metadata_path
    )


def main():

    print(
        "=" * 70
    )

    print(
        "NORTHBAY FORESIGHT - "
        "DEMAND FORECAST MODEL"
    )

    print(
        "=" * 70
    )

    df = load_data()

    (
        df,
        X,
        y,
        feature_columns
    ) = prepare_features(
        df
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        test_rows
    ) = create_time_split(
        df,
        X,
        y
    )

    model = train_model(
        X_train,
        y_train
    )

    (
        predictions,
        wape,
        mae,
        bias
    ) = evaluate_model(
        model,
        X_test,
        y_test
    )

    save_predictions(
        test_rows,
        predictions
    )

    save_metrics(
        wape,
        mae,
        bias
    )

    save_model(
        model,
        feature_columns
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "MODEL TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()