from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


# ============================================================
# EXPECTED SCHEMAS
# ============================================================

EXPECTED_COLUMNS = {

    "sales_daily": [
        "date",
        "store_id",
        "sku_id",
        "units_sold",
        "revenue",
        "avg_unit_price",
        "transaction_count",
        "promo_flag",
    ],

    "sku_master": [
        "sku_id",
        "sku_name",
        "category",
        "subcategory",
        "unit_price",
        "cost_price",
        "brand",
    ],

    "inventory_snapshots": [
        "sku_id",
        "store_id",
        "stock_on_hand",
        "reorder_point",
        "safety_stock",
        "last_restock_date",
    ],

    "calendar": [
        "date",
        "year",
        "month",
        "month_name",
        "week",
        "day",
        "day_of_week",
        "day_name",
        "quarter",
        "is_weekend",
        "season",
        "is_holiday",
        "promo_event",
    ],
}


# ============================================================
# COLUMN NAME CLEANING
# ============================================================

def clean_column_names(df):

    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("/", "_", regex=False)
    )

    return df


# ============================================================
# BASIC QUALITY CHECK
# ============================================================

def basic_quality_check(df, dataset_name):

    print("\n" + "=" * 70)
    print(f"QUALITY CHECK: {dataset_name}")
    print("=" * 70)

    print(f"Rows              : {len(df):,}")
    print(f"Columns           : {len(df.columns):,}")
    print(f"Duplicate rows    : {df.duplicated().sum():,}")
    print(f"Missing cells     : {df.isna().sum().sum():,}")

    missing = df.isna().sum()

    missing = missing[missing > 0]

    if len(missing) == 0:

        print("\n✓ No missing values")

    else:

        print("\nMissing values:")

        for column, count in missing.items():

            print(
                f"  {column:<30} {count:,}"
            )


# ============================================================
# SCHEMA VALIDATION
# ============================================================

def validate_required_columns(
    df,
    required_columns,
    dataset_name
):

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    unexpected = [
        column
        for column in df.columns
        if column not in required_columns
    ]

    print("\n" + "-" * 70)
    print(f"SCHEMA VALIDATION: {dataset_name}")
    print("-" * 70)

    if missing:

        print("❌ Missing columns:")

        for column in missing:
            print(f"   - {column}")

    else:

        print("✓ All expected columns present")

    if unexpected:

        print("\n⚠ Additional columns:")

        for column in unexpected:
            print(f"   + {column}")

    else:

        print("✓ No unexpected columns")

    return len(missing) == 0


# ============================================================
# SALES VALIDATION
# ============================================================

def validate_sales_daily(df):

    df = clean_column_names(df)

    valid = validate_required_columns(
        df,
        EXPECTED_COLUMNS["sales_daily"],
        "sales_daily"
    )

    if not valid:
        return False

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    if df["date"].isna().any():

        print("❌ Invalid sales dates")

        return False

    if (df["units_sold"] < 0).any():

        print("❌ Negative units_sold found")

        return False

    if (df["revenue"] < 0).any():

        print("❌ Negative revenue found")

        return False

    print("✓ Sales data validation passed")

    return True


# ============================================================
# SKU VALIDATION
# ============================================================

def validate_sku_master(df):

    df = clean_column_names(df)

    valid = validate_required_columns(
        df,
        EXPECTED_COLUMNS["sku_master"],
        "sku_master"
    )

    if not valid:
        return False

    if df["sku_id"].duplicated().any():

        print("❌ Duplicate sku_id found")

        return False

    print("✓ SKU master validation passed")

    return True


# ============================================================
# INVENTORY VALIDATION
# ============================================================

def validate_inventory_snapshots(df):

    df = clean_column_names(df)

    valid = validate_required_columns(
        df,
        EXPECTED_COLUMNS["inventory_snapshots"],
        "inventory_snapshots"
    )

    if not valid:
        return False

    print("✓ Inventory validation passed")

    return True


# ============================================================
# CALENDAR VALIDATION
# ============================================================

def validate_calendar(df):

    df = clean_column_names(df)

    valid = validate_required_columns(
        df,
        EXPECTED_COLUMNS["calendar"],
        "calendar"
    )

    if not valid:
        return False

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    invalid_dates = df["date"].isna().sum()

    if invalid_dates > 0:

        print(
            f"❌ Invalid calendar dates: "
            f"{invalid_dates}"
        )

        return False

    print("✓ Calendar validation passed")

    return True


# ============================================================
# GENERIC DATASET VALIDATION
# ============================================================

def validate_dataset(df, dataset_name):

    if dataset_name not in EXPECTED_COLUMNS:

        print(
            f"⚠ No schema defined for "
            f"{dataset_name}"
        )

        return True

    return validate_required_columns(
        df,
        EXPECTED_COLUMNS[dataset_name],
        dataset_name
    )


# ============================================================
# FULL DATASET CHECK
# ============================================================

def validate_file(path, dataset_name):

    print("\n" + "=" * 70)
    print(f"DATASET: {dataset_name}")
    print("=" * 70)

    print(f"File: {path}")

    if not path.exists():

        print("❌ File not found")

        return False

    df = pd.read_csv(path)

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    df = clean_column_names(df)

    schema_ok = validate_dataset(
        df,
        dataset_name
    )

    basic_quality_check(
        df,
        dataset_name
    )

    return schema_ok