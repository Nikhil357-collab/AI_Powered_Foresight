from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\nLoading cleaned datasets...")

    sales = pd.read_csv(
        PROCESSED_DIR / "sales_daily_clean.csv"
    )

    sku = pd.read_csv(
        PROCESSED_DIR / "sku_master_clean.csv"
    )

    inventory = pd.read_csv(
        PROCESSED_DIR / "inventory_snapshots_clean.csv"
    )

    calendar = pd.read_csv(
        PROCESSED_DIR / "calendar_clean.csv"
    )

    print(f"Sales rows      : {len(sales):,}")
    print(f"SKU rows        : {len(sku):,}")
    print(f"Inventory rows  : {len(inventory):,}")
    print(f"Calendar rows   : {len(calendar):,}")

    return sales, sku, inventory, calendar


# ============================================================
# DATE FEATURES
# ============================================================

def create_date_features(df):

    print("\nCreating date features...")

    df = df.copy()

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df["year"] = df["date"].dt.year

    df["month"] = df["date"].dt.month

    df["week"] = df["date"].dt.isocalendar().week.astype(int)

    df["day_of_week"] = df["date"].dt.dayofweek

    df["day_of_month"] = df["date"].dt.day

    df["quarter"] = df["date"].dt.quarter

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    df["week_of_year"] = (
        df["date"].dt.isocalendar().week.astype(int)
    )

    return df


# ============================================================
# LAG FEATURES
# ============================================================

def create_lag_features(df):

    print("\nCreating lag features...")

    df = df.copy()

    df = df.sort_values(
        ["sku_id", "date"]
    )

    group = df.groupby(["store_id", "sku_id"])["units_sold"].shift(7)

    df["lag_1"] = group.shift(1)

    df["lag_7"] = group.shift(7)

    df["lag_14"] = group.shift(14)

    df["lag_28"] = group.shift(28)

    return df


# ============================================================
# ROLLING FEATURES
# ============================================================

def create_rolling_features(df):

    print("\nCreating rolling demand features...")

    df = df.copy()

    group = df.groupby(
        "sku_id"
    )["units_sold"]

    df["rolling_mean_7"] = (
        group
        .shift(1)
        .rolling(7)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["rolling_mean_14"] = (
        group
        .shift(1)
        .rolling(14)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["rolling_mean_28"] = (
        group
        .shift(1)
        .rolling(28)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["rolling_std_7"] = (
        group
        .shift(1)
        .rolling(7)
        .std()
        .reset_index(level=0, drop=True)
    )

    return df


# ============================================================
# DEMAND TREND FEATURES
# ============================================================

def create_demand_features(df):

    print("\nCreating demand trend features...")

    df = df.copy()

    df["demand_change_7d"] = (
        df["lag_1"] -
        df["lag_7"]
    )

    df["demand_growth_7d"] = (
        df["lag_1"] /
        (df["lag_7"] + 1)
    )

    df["demand_volatility"] = (
        df["rolling_std_7"] /
        (df["rolling_mean_7"] + 1)
    )

    return df


# ============================================================
# SKU FEATURES
# ============================================================

def merge_sku_features(
    sales,
    sku
):

    print("\nMerging SKU master information...")

    sku_columns = [
        "sku_id",
        "category",
        "subcategory",
        "brand",
        "unit_price",
        "cost_price"
    ]

    sku = sku[
        [
            col
            for col in sku_columns
            if col in sku.columns
        ]
    ].drop_duplicates(
        subset=["sku_id"]
    )

    sales = sales.merge(
        sku,
        on="sku_id",
        how="left",
        suffixes=("", "_master")
    )

    return sales


# ============================================================
# PROFITABILITY FEATURES
# ============================================================

def create_profit_features(df):

    print("\nCreating profitability features...")

    df = df.copy()

    if "avg_unit_price" in df.columns and "cost_price" in df.columns:

        df["unit_margin"] = (
            df["avg_unit_price"] -
            df["cost_price"]
        )

        df["margin_pct"] = (
            df["unit_margin"] /
            (df["avg_unit_price"] + 1e-6)
        )

    return df


# ============================================================
# INVENTORY FEATURES
# ============================================================

def merge_inventory_features(
    sales,
    inventory
):

    print("\nMerging inventory information...")

    inventory_columns = [
        "store_id",
        "sku_id",
        "stock_on_hand",
        "reorder_point",
        "safety_stock",
        "last_restock_date"
    ]

    available_columns = [
        col
        for col in inventory_columns
        if col in inventory.columns
    ]

    inventory = inventory[
        available_columns
    ].copy()

    inventory = inventory.drop_duplicates(
        subset=["store_id", "sku_id"]
        if "store_id" in inventory.columns
        else ["sku_id"]
    )

    join_columns = ["sku_id"]

    if (
        "store_id" in sales.columns
        and "store_id" in inventory.columns
    ):
        join_columns = [
            "store_id",
            "sku_id"
        ]

    sales = sales.merge(
        inventory,
        on=join_columns,
        how="left",
        suffixes=("", "_inventory")
    )

    return sales


# ============================================================
# INVENTORY RISK FEATURES
# ============================================================

def create_inventory_features(df):

    print("\nCreating inventory risk features...")

    df = df.copy()

    if {
        "stock_on_hand",
        "reorder_point"
    }.issubset(df.columns):

        df["stock_vs_reorder_ratio"] = (
            df["stock_on_hand"] /
            (df["reorder_point"] + 1)
        )

        df["below_reorder_point"] = (
            df["stock_on_hand"] <
            df["reorder_point"]
        ).astype(int)

    if {
        "stock_on_hand",
        "safety_stock"
    }.issubset(df.columns):

        df["below_safety_stock"] = (
            df["stock_on_hand"] <
            df["safety_stock"]
        ).astype(int)

    return df


# ============================================================
# CALENDAR FEATURES
# ============================================================

def merge_calendar_features(
    sales,
    calendar
):

    print("\nMerging calendar information...")

    sales["date"] = pd.to_datetime(
        sales["date"],
        errors="coerce"
    )

    calendar["date"] = pd.to_datetime(
        calendar["date"],
        errors="coerce"
    )

    calendar = calendar.drop_duplicates(
        subset=["date"]
    )

    sales = sales.merge(
        calendar,
        on="date",
        how="left",
        suffixes=("", "_calendar")
    )

    return sales


# ============================================================
# PROMOTION FEATURES
# ============================================================

def create_promotion_features(df):

    print("\nCreating promotion features...")

    df = df.copy()

    if "promo_flag" in df.columns:

        df["is_promotion"] = (
            df["promo_flag"]
            .fillna(0)
            .astype(int)
        )

    else:

        df["is_promotion"] = 0

    return df


# ============================================================
# ============================================================
# MISSING VALUE HANDLING
# ============================================================

# ============================================================
# MISSING VALUE HANDLING
# ============================================================

def handle_feature_missing_values(df):

    print("\nHandling feature missing values...")

    df = df.copy()

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    for col in numeric_columns:
        df[col] = df[col].fillna(0)

    return df

# =====================================================
# FEATURE VALIDATION
# ============================================================

def validate_features(df):

    print("\n" + "=" * 70)
    print("FEATURE VALIDATION")
    print("=" * 70)

    print(
        f"Rows       : {len(df):,}"
    )

    print(
        f"Columns    : {len(df.columns):,}"
    )

    print(
        f"Missing cells : {df.isna().sum().sum():,}"
    )

    print(
        f"Duplicate rows: {df.duplicated().sum():,}"
    )

    print("\nFeature columns:")

    for column in df.columns:

        print(
            f"  • {column}"
        )


# ============================================================
# SAVE FEATURES
# ============================================================

def save_features(df):

    output_path = (
        PROCESSED_DIR /
        "model_features.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print("\n" + "=" * 70)

    print(
        f"✓ Feature dataset saved:"
    )

    print(output_path)

    print(
        f"Rows    : {len(df):,}"
    )

    print(
        f"Columns : {len(df.columns):,}"
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "NORTHBAY FORESIGHT - FEATURE ENGINEERING"
    )

    print("=" * 70)

    sales, sku, inventory, calendar = load_data()

    sales = create_date_features(
        sales
    )

    sales = create_lag_features(
        sales
    )

    sales = create_rolling_features(
        sales
    )

    sales = create_demand_features(
        sales
    )

    sales = merge_sku_features(
        sales,
        sku
    )

    sales = create_profit_features(
        sales
    )

    sales = merge_inventory_features(
        sales,
        inventory
    )

    sales = create_inventory_features(
        sales
    )

    sales = merge_calendar_features(
        sales,
        calendar
    )

    sales = create_promotion_features(
        sales
    )

    sales = handle_feature_missing_values(
        sales
    )

    validate_features(
        sales
    )

    save_features(
        sales
    )


if __name__ == "__main__":
    main()