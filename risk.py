from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# NORTHBAY FORESIGHT
# INVENTORY RISK SCORING
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_LEAD_TIME_DAYS = 7

OVERSTOCK_DAYS = 30

FORECAST_HORIZON_DAYS = 30

# Risk thresholds
WATCH_THRESHOLD = 25
HIGH_THRESHOLD = 50
CRITICAL_THRESHOLD = 75


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("NORTHBAY FORESIGHT - RISK SCORING")
    print("=" * 70)

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    prediction_path = (
        PROCESSED_DIR /
        "xgboost_predictions.csv"
    )

    print("\nLoading model predictions...")

    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found:\n{prediction_path}"
        )

    predictions = pd.read_csv(
        prediction_path
    )

    print(
        f"Prediction rows : {len(predictions):,}"
    )

    print(
        f"Prediction columns: {list(predictions.columns)}"
    )

    # --------------------------------------------------------
    # Inventory
    # --------------------------------------------------------

    inventory_path = (
        PROCESSED_DIR /
        "inventory_snapshots_clean.csv"
    )

    print("\nLoading inventory data...")

    if not inventory_path.exists():
        raise FileNotFoundError(
            f"Inventory file not found:\n{inventory_path}"
        )

    inventory = pd.read_csv(
        inventory_path
    )

    print(
        f"Inventory rows : {len(inventory):,}"
    )

    print(
        f"Inventory columns: {list(inventory.columns)}"
    )

    # --------------------------------------------------------
    # SKU master
    # --------------------------------------------------------

    sku_path = (
        PROCESSED_DIR /
        "sku_master_clean.csv"
    )

    print("\nLoading SKU master...")

    if not sku_path.exists():
        raise FileNotFoundError(
            f"SKU master not found:\n{sku_path}"
        )

    sku = pd.read_csv(
        sku_path
    )

    print(
        f"SKU master rows: {len(sku):,}"
    )

    print(
        f"SKU columns: {list(sku.columns)}"
    )

    return predictions, inventory, sku


# ============================================================
# PREPARE PREDICTIONS
# ============================================================

def prepare_predictions(predictions):

    print("\nPreparing prediction data...")

    df = predictions.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    required = [
        "date",
        "sku_id",
        "prediction"
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Prediction file missing columns: {missing}"
        )

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["date"]
    )

    # --------------------------------------------------------
    # SKU
    # --------------------------------------------------------

    df["sku_id"] = (
        df["sku_id"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Forecast
    # --------------------------------------------------------

    df["prediction"] = pd.to_numeric(
        df["prediction"],
        errors="coerce"
    )

    df["prediction"] = (
        df["prediction"]
        .fillna(0)
        .clip(lower=0)
    )

    print(
        f"Prepared predictions: {len(df):,}"
    )

    print(
        f"Forecast period: "
        f"{df['date'].min().date()} "
        f"to "
        f"{df['date'].max().date()}"
    )

    return df


# ============================================================
# PREPARE INVENTORY
# ============================================================

def prepare_inventory(inventory):

    print("\nPreparing inventory data...")

    df = inventory.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    # --------------------------------------------------------
    # Required identifiers
    # --------------------------------------------------------

    required = [
        "sku_id",
        "stock_on_hand",
        "reorder_point",
        "safety_stock"
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Inventory file missing columns: {missing}"
        )

    # --------------------------------------------------------
    # Store ID
    # --------------------------------------------------------

    if "store_id" not in df.columns:

        df["store_id"] = "STORE_001"

        print(
            "⚠ store_id not found. "
            "Using STORE_001."
        )

    # --------------------------------------------------------
    # SKU
    # --------------------------------------------------------

    df["sku_id"] = (
        df["sku_id"]
        .astype(str)
        .str.strip()
    )

    df["store_id"] = (
        df["store_id"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    numeric_columns = [
        "stock_on_hand",
        "reorder_point",
        "safety_stock"
    ]

    for col in numeric_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        df[col] = (
            df[col]
            .fillna(0)
            .clip(lower=0)
        )

    # --------------------------------------------------------
    # Optional on-order units
    # --------------------------------------------------------

    if "on_order_units" not in df.columns:

        df["on_order_units"] = 0

        print(
            "⚠ on_order_units not found. "
            "Using 0."
        )

    else:

        df["on_order_units"] = pd.to_numeric(
            df["on_order_units"],
            errors="coerce"
        ).fillna(0).clip(lower=0)

    # --------------------------------------------------------
    # Lead time
    # --------------------------------------------------------

    if "lead_time_days" not in df.columns:

        df["lead_time_days"] = (
            DEFAULT_LEAD_TIME_DAYS
        )

        print(
            f"⚠ lead_time_days not found. "
            f"Using default = "
            f"{DEFAULT_LEAD_TIME_DAYS} days."
        )

    else:

        df["lead_time_days"] = pd.to_numeric(
            df["lead_time_days"],
            errors="coerce"
        )

        df["lead_time_days"] = (
            df["lead_time_days"]
            .fillna(DEFAULT_LEAD_TIME_DAYS)
            .clip(lower=1, upper=90)
        )

    # --------------------------------------------------------
    # Inventory date
    # --------------------------------------------------------

    if "date" in df.columns:

        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce"
        )

        # Latest inventory snapshot
        df = (
            df.sort_values("date")
            .drop_duplicates(
                subset=["store_id", "sku_id"],
                keep="last"
            )
        )

    else:

        print(
            "⚠ Inventory file has NO date column."
        )

        print(
            "Using latest available inventory "
            "record for each store + SKU."
        )

        df = (
            df.drop_duplicates(
                subset=["store_id", "sku_id"],
                keep="last"
            )
        )

    print(
        f"Prepared inventory rows: {len(df):,}"
    )

    return df


# ============================================================
# PREPARE SKU MASTER
# ============================================================

def prepare_sku_master(sku):

    print("\nPreparing SKU master...")

    df = sku.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    required = [
        "sku_id",
        "unit_price",
        "cost_price"
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"SKU master missing columns: {missing}"
        )

    df["sku_id"] = (
        df["sku_id"]
        .astype(str)
        .str.strip()
    )

    df["unit_price"] = pd.to_numeric(
        df["unit_price"],
        errors="coerce"
    ).fillna(0).clip(lower=0)

    df["cost_price"] = pd.to_numeric(
        df["cost_price"],
        errors="coerce"
    ).fillna(0).clip(lower=0)

    df = df.drop_duplicates(
        subset=["sku_id"]
    )

    print(
        f"Prepared SKU master rows: {len(df):,}"
    )

    print(
        f"SKUs with price > 0: "
        f"{(df['unit_price'] > 0).sum():,}"
    )

    return df


# ============================================================
# AGGREGATE FORECAST
# ============================================================

def aggregate_forecast(predictions):

    print("\nAggregating forecast by store + SKU...")

    group_columns = [
        "sku_id"
    ]

    if "store_id" in predictions.columns:

        group_columns = [
            "store_id",
            "sku_id"
        ]

    # --------------------------------------------------------
    # Daily forecast remains useful for lead-time calculations.
    # --------------------------------------------------------

    daily = (
        predictions
        .groupby(
            group_columns + ["date"],
            as_index=False
        )["prediction"]
        .sum()
    )

    # --------------------------------------------------------
    # Average daily demand
    # --------------------------------------------------------

    demand = (
        daily
        .groupby(
            group_columns,
            as_index=False
        )
        .agg(
            avg_daily_demand=(
                "prediction",
                "mean"
            ),
            total_forecast_demand=(
                "prediction",
                "sum"
            ),
            max_daily_demand=(
                "prediction",
                "max"
            )
        )
    )

    print(
        f"Aggregated SKU/store combinations: "
        f"{len(demand):,}"
    )

    return demand


# ============================================================
# MERGE DATA
# ============================================================

def merge_data(
    demand,
    inventory,
    sku
):

    print(
        "\nMerging forecast, inventory "
        "and SKU information..."
    )

    # --------------------------------------------------------
    # Forecast + inventory
    # --------------------------------------------------------

    df = demand.merge(
        inventory,
        on=["store_id", "sku_id"],
        how="left"
    )

    # --------------------------------------------------------
    # SKU information
    # --------------------------------------------------------

    df = df.merge(
        sku,
        on="sku_id",
        how="left",
        suffixes=("", "_master")
    )

    # --------------------------------------------------------
    # Fill inventory
    # --------------------------------------------------------

    inventory_columns = [
        "stock_on_hand",
        "reorder_point",
        "safety_stock",
        "on_order_units",
        "lead_time_days"
    ]

    for col in inventory_columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            ).fillna(0)

    # Missing lead time should be 7,
    # not zero.

    df["lead_time_days"] = (
        df["lead_time_days"]
        .replace(0, DEFAULT_LEAD_TIME_DAYS)
    )

    # --------------------------------------------------------
    # Fill prices
    # --------------------------------------------------------

    df["unit_price"] = pd.to_numeric(
        df["unit_price"],
        errors="coerce"
    ).fillna(0)

    df["cost_price"] = pd.to_numeric(
        df["cost_price"],
        errors="coerce"
    ).fillna(0)

    return df


# ============================================================
# CALCULATE RISK
# ============================================================

def calculate_risk(df):

    print(
        "\nCalculating forward lead-time demand..."
    )

    df = df.copy()

    # ========================================================
    # 1. LEAD-TIME DEMAND
    # ========================================================

    df["lead_time_demand"] = (
        df["avg_daily_demand"] *
        df["lead_time_days"]
    )

    # ========================================================
    # 2. INVENTORY POSITION
    # ========================================================

    df["inventory_position"] = (
        df["stock_on_hand"] +
        df["on_order_units"]
    )

    # ========================================================
    # 3. STOCKOUT SHORTAGE
    # ========================================================

    # Include safety stock in target inventory.

    df["target_lead_time_stock"] = (
        df["lead_time_demand"] +
        df["safety_stock"]
    )

    df["stockout_shortage_units"] = (
        df["target_lead_time_stock"] -
        df["inventory_position"]
    ).clip(lower=0)

    # ========================================================
    # 4. DAYS OF SUPPLY
    # ========================================================

    df["days_of_supply"] = np.where(
        df["avg_daily_demand"] > 0,

        df["inventory_position"] /
        df["avg_daily_demand"],

        np.inf
    )

    # ========================================================
    # 5. STOCKOUT RISK
    # ========================================================

    df["stockout_risk"] = (
        df["inventory_position"] <
        df["target_lead_time_stock"]
    )

    # ========================================================
    # 6. REORDER QUANTITY
    # ========================================================

    # Order enough to cover:
    #
    # lead-time demand
    # + safety stock
    # - current inventory
    # - incoming orders

    df["recommended_reorder_units"] = (
        df["target_lead_time_stock"] -
        df["inventory_position"]
    ).clip(lower=0)

    # Round up to whole units.

    df["recommended_reorder_units"] = np.ceil(
        df["recommended_reorder_units"]
    )

    # ========================================================
    # 7. STOCKOUT RISK SCORE
    # ========================================================

    # Percentage of target stock that is missing.

    df["stockout_gap_pct"] = np.where(
        df["target_lead_time_stock"] > 0,

        df["stockout_shortage_units"] /
        df["target_lead_time_stock"],

        0
    )

    stockout_score = (
        df["stockout_gap_pct"] * 100
    ).clip(0, 100)

    # ========================================================
    # 8. OVERSTOCK CALCULATION
    # ========================================================

    # Target inventory for 30 days of demand
    # plus safety stock.

    df["overstock_target_units"] = (
        df["avg_daily_demand"] *
        OVERSTOCK_DAYS
    ) + df["safety_stock"]

    df["excess_inventory_units"] = (
        df["inventory_position"] -
        df["overstock_target_units"]
    ).clip(lower=0)

    # ========================================================
    # 9. OVERSTOCK RISK
    # ========================================================

    df["overstock_risk"] = (
        (df["excess_inventory_units"] > 0) &
        (df["avg_daily_demand"] > 0)
    )

    # ========================================================
    # 10. OVERSTOCK SCORE
    # ========================================================

    df["overstock_gap_pct"] = np.where(
        df["overstock_target_units"] > 0,

        df["excess_inventory_units"] /
        df["overstock_target_units"],

        0
    )

    overstock_score = (
        df["overstock_gap_pct"] * 100
    ).clip(0, 100)

    # ========================================================
    # 11. FINAL RISK SCORE
    # ========================================================

    df["risk_score"] = np.maximum(
        stockout_score,
        overstock_score
    )

    df["risk_score"] = (
        df["risk_score"]
        .round(1)
        .clip(0, 100)
    )

    # ========================================================
    # 12. RISK TYPE
    # ========================================================

    df["risk_level"] = "NORMAL"

    # Stockout takes priority.

    df.loc[
        df["stockout_risk"],
        "risk_level"
    ] = "STOCKOUT"

    df.loc[
        (
            ~df["stockout_risk"] &
            df["overstock_risk"]
        ),
        "risk_level"
    ] = "OVERSTOCK"

    # ========================================================
    # 13. RISK PRIORITY
    # ========================================================

    df["risk_priority"] = "LOW"

    df.loc[
        df["risk_score"] >= WATCH_THRESHOLD,
        "risk_priority"
    ] = "WATCH"

    df.loc[
        df["risk_score"] >= HIGH_THRESHOLD,
        "risk_priority"
    ] = "HIGH"

    df.loc[
        df["risk_score"] >= CRITICAL_THRESHOLD,
        "risk_priority"
    ] = "CRITICAL"

    # ========================================================
    # 14. RUPEE VALUE AT STAKE
    # ========================================================

    # --------------------------------------------------------
    # Stockout:
    # revenue opportunity at risk
    # --------------------------------------------------------

    df["stockout_value_at_stake"] = (
        df["stockout_shortage_units"] *
        df["unit_price"]
    )

    # --------------------------------------------------------
    # Overstock:
    # inventory capital tied up
    # --------------------------------------------------------

    df["overstock_value_at_stake"] = (
        df["excess_inventory_units"] *
        df["cost_price"]
    )

    # --------------------------------------------------------
    # Final value
    # --------------------------------------------------------

    df["value_at_stake"] = np.where(

        df["risk_level"] == "STOCKOUT",

        df["stockout_value_at_stake"],

        np.where(

            df["risk_level"] == "OVERSTOCK",

            df["overstock_value_at_stake"],

            0
        )
    )

    # ========================================================
    # 15. RECOMMENDED ACTION
    # ========================================================

    df["recommended_action"] = "NO ACTION"

    # Stockout

    df.loc[
        (
            df["risk_level"] == "STOCKOUT"
        ) &
        (
            df["risk_priority"] == "CRITICAL"
        ),
        "recommended_action"
    ] = "URGENT REORDER"

    df.loc[
        (
            df["risk_level"] == "STOCKOUT"
        ) &
        (
            df["risk_priority"] != "CRITICAL"
        ),
        "recommended_action"
    ] = "REORDER"

    # Overstock

    df.loc[
        (
            df["risk_level"] == "OVERSTOCK"
        ) &
        (
            df["risk_priority"] == "CRITICAL"
        ),
        "recommended_action"
    ] = "URGENT MARKDOWN"

    df.loc[
        (
            df["risk_level"] == "OVERSTOCK"
        ) &
        (
            df["risk_priority"] != "CRITICAL"
        ),
        "recommended_action"
    ] = "MARKDOWN / PROMOTION"

    return df


# ============================================================
# SUMMARY
# ============================================================

def create_summary(df):

    print(
        "\n" + "=" * 70
    )

    print("RISK SUMMARY")

    print(
        "=" * 70
    )

    summary = (
        df
        .groupby("risk_level")
        .agg(
            sku_store_count=(
                "sku_id",
                "count"
            ),
            value_at_stake=(
                "value_at_stake",
                "sum"
            ),
            avg_risk_score=(
                "risk_score",
                "mean"
            )
        )
        .reset_index()
        .sort_values(
            "value_at_stake",
            ascending=False
        )
    )

    print(
        summary.to_string(
            index=False
        )
    )

    stockout_rows = (
        df["risk_level"] == "STOCKOUT"
    ).sum()

    overstock_rows = (
        df["risk_level"] == "OVERSTOCK"
    ).sum()

    stockout_value = (
        df.loc[
            df["risk_level"] == "STOCKOUT",
            "value_at_stake"
        ].sum()
    )

    overstock_value = (
        df.loc[
            df["risk_level"] == "OVERSTOCK",
            "value_at_stake"
        ].sum()
    )

    total_value = (
        df["value_at_stake"].sum()
    )

    print(
        f"\nStockout risk rows : "
        f"{stockout_rows:,}"
    )

    print(
        f"Overstock risk rows: "
        f"{overstock_rows:,}"
    )

    print(
        f"\nStockout value at stake : "
        f"₹{stockout_value:,.2f}"
    )

    print(
        f"Overstock value at stake: "
        f"₹{overstock_value:,.2f}"
    )

    print(
        f"Total value at stake    : "
        f"₹{total_value:,.2f}"
    )

    return summary


# ============================================================
# PRIORITY LISTS
# ============================================================

def create_priority_lists(df):

    print(
        "\nCreating prioritised decision lists..."
    )

    # --------------------------------------------------------
    # REORDER
    # --------------------------------------------------------

    reorder = (
        df[
            df["risk_level"] == "STOCKOUT"
        ]
        .sort_values(
            [
                "risk_score",
                "value_at_stake"
            ],
            ascending=False
        )
    )

    # --------------------------------------------------------
    # MARKDOWN
    # --------------------------------------------------------

    markdown = (
        df[
            df["risk_level"] == "OVERSTOCK"
        ]
        .sort_values(
            [
                "risk_score",
                "value_at_stake"
            ],
            ascending=False
        )
    )

    return reorder, markdown


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_results(
    df,
    summary,
    reorder,
    markdown
):

    print(
        "\nSaving risk outputs..."
    )

    risk_path = (
        PROCESSED_DIR /
        "inventory_risk_scores.csv"
    )

    summary_path = (
        PROCESSED_DIR /
        "risk_summary.csv"
    )

    reorder_path = (
        PROCESSED_DIR /
        "reorder_priority_list.csv"
    )

    markdown_path = (
        PROCESSED_DIR /
        "markdown_priority_list.csv"
    )

    df.to_csv(
        risk_path,
        index=False
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    reorder.to_csv(
        reorder_path,
        index=False
    )

    markdown.to_csv(
        markdown_path,
        index=False
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "RISK SCORING COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\n✓ Risk scores:\n{risk_path}"
    )

    print(
        f"\n✓ Risk summary:\n{summary_path}"
    )

    print(
        f"\n✓ Reorder priority list:\n{reorder_path}"
    )

    print(
        f"\n✓ Markdown priority list:\n{markdown_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    predictions, inventory, sku = load_data()

    predictions = prepare_predictions(
        predictions
    )

    inventory = prepare_inventory(
        inventory
    )

    sku = prepare_sku_master(
        sku
    )

    demand = aggregate_forecast(
        predictions
    )

    df = merge_data(
        demand,
        inventory,
        sku
    )

    print(
        f"\nRisk rows: {len(df):,}"
    )

    df = calculate_risk(
        df
    )

    summary = create_summary(
        df
    )

    reorder, markdown = (
        create_priority_lists(df)
    )

    save_results(
        df,
        summary,
        reorder,
        markdown
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()