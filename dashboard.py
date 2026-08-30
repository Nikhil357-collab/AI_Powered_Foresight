from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

import requests
# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = (
    BASE_DIR /
    "data" /
    "processed"
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NorthBay FORESIGHT",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .metric-card {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ddd;
        background-color: #ffffff;
    }

    .small-text {
        color: #666;
        font-size: 13px;
    }

    </style>
    """,
    unsafe_allow_html=True
)



# LOCAL DEVELOPMENT
API_URL = "http://127.0.0.1:5000"

# AFTER RENDER DEPLOYMENT, CHANGE TO:
# API_URL = "https://your-api-name.onrender.com"


# ============================================================
# TITLE
# ============================================================




# ============================================================
# API HELPER
# ============================================================

def api_get(endpoint, params=None):

    try:

        response = requests.get(
            f"{API_URL}{endpoint}",
            params=params,
            timeout=20
        )

        if response.status_code == 200:

            return response.json()

        else:

            st.error(
                f"API error {response.status_code}: "
                f"{response.text}"
            )

            return None

    except requests.exceptions.ConnectionError:

        st.error(
            "Cannot connect to the Flask API. "
            "Make sure app.py is running."
        )

        return None

    except requests.exceptions.Timeout:

        st.error(
            "API request timed out."
        )

        return None

    except Exception as e:

        st.error(
            f"Unexpected error: {e}"
        )

        return None

# ============================================================
# HELPERS
# ============================================================

def find_file(filename):

    path = PROCESSED_DIR / filename

    if not path.exists():
        return None

    return path


@st.cache_data
def load_csv(filename):

    path = find_file(filename)

    if path is None:
        return pd.DataFrame()

    try:

        df = pd.read_csv(path)

        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
        )

        return df

    except Exception as e:

        st.error(
            f"Unable to load {filename}: {e}"
        )

        return pd.DataFrame()


def safe_numeric(
    df,
    columns
):

    for column in columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df


# ============================================================
# LOAD DATA
# ============================================================

risk_df = load_csv(
    "inventory_risk_scores.csv"
)

predictions_df = load_csv(
    "xgboost_predictions.csv"
)

risk_summary = load_csv(
    "risk_summary.csv"
)

reorder_df = load_csv(
    "reorder_priority_list.csv"
)

markdown_df = load_csv(
    "markdown_priority_list.csv"
)

metrics_df = load_csv(
    "model_metrics.csv"
)


# ============================================================
# VALIDATION
# ============================================================

if risk_df.empty:

    st.error(
        "Risk data not found."
    )

    st.info(
        "Run risk.py first to generate inventory_risk_scores.csv."
    )

    st.stop()


# ============================================================
# CLEAN DATA
# ============================================================

risk_df = safe_numeric(
    risk_df,
    [
        "prediction",
        "forecast_daily_demand",
        "lead_time_days",
        "lead_time_demand",
        "stock_on_hand",
        "on_order_units",
        "reorder_point",
        "safety_stock",
        "days_of_supply",
        "stock_gap",
        "risk_score",
        "value_at_stake",
        "unit_price",
        "cost_price"
    ]
)


if "date" in risk_df.columns:

    risk_df["date"] = pd.to_datetime(
        risk_df["date"],
        errors="coerce"
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "📦 NORTHBAY FORESIGHT"
)

st.subheader(
    "AI-Powered Demand Forecasting & Inventory Planning"
)

st.caption(
    "Demand forecast • Stockout risk • Overstock risk • Reorder & markdown decisions"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "🔎 Filters"
)


# ------------------------------------------------------------
# CATEGORY FILTER
# ------------------------------------------------------------

if "category" in risk_df.columns:

    categories = sorted(
        risk_df["category"]
        .dropna()
        .astype(str)
        .unique()
    )

    selected_categories = st.sidebar.multiselect(
        "Category",
        categories,
        default=categories
    )

else:

    selected_categories = []


# ------------------------------------------------------------
# SKU FILTER
# ------------------------------------------------------------

if "sku_id" in risk_df.columns:

    sku_values = sorted(
        risk_df["sku_id"]
        .dropna()
        .astype(str)
        .unique()
    )

    selected_skus = st.sidebar.multiselect(
        "SKU",
        sku_values
    )

else:

    selected_skus = []


# ------------------------------------------------------------
# RISK FILTER
# ------------------------------------------------------------

if "risk_level" in risk_df.columns:

    risk_levels = [
        "STOCKOUT",
        "OVERSTOCK",
        "NORMAL"
    ]

    selected_risk = st.sidebar.multiselect(
        "Risk Level",
        risk_levels,
        default=risk_levels
    )

else:

    selected_risk = []


# ============================================================
# APPLY FILTERS
# ============================================================

filtered_df = risk_df.copy()


if selected_categories:

    if "category" in filtered_df.columns:

        filtered_df = filtered_df[
            filtered_df["category"]
            .astype(str)
            .isin(selected_categories)
        ]


if selected_skus:

    if "sku_id" in filtered_df.columns:

        filtered_df = filtered_df[
            filtered_df["sku_id"]
            .astype(str)
            .isin(selected_skus)
        ]


if selected_risk:

    if "risk_level" in filtered_df.columns:

        filtered_df = filtered_df[
            filtered_df["risk_level"]
            .isin(selected_risk)
        ]


# ============================================================
# EMPTY STATE
# ============================================================

if filtered_df.empty:

    st.warning(
        "No records match the selected filters."
    )

    st.info(
        "Try removing one or more filters."
    )

    st.stop()


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_items = len(filtered_df)


stockout_count = int(
    (
        filtered_df["risk_level"]
        == "STOCKOUT"
    ).sum()
)


overstock_count = int(
    (
        filtered_df["risk_level"]
        == "OVERSTOCK"
    ).sum()
)


normal_count = int(
    (
        filtered_df["risk_level"]
        == "NORMAL"
    ).sum()
)


if "value_at_stake" in filtered_df.columns:

    total_value = (
        filtered_df["value_at_stake"]
        .fillna(0)
        .sum()
    )

else:

    total_value = 0


stockout_value = (
    filtered_df.loc[
        filtered_df["risk_level"] == "STOCKOUT",
        "value_at_stake"
    ].fillna(0).sum()
    if "value_at_stake" in filtered_df.columns
    else 0
)


overstock_value = (
    filtered_df.loc[
        filtered_df["risk_level"] == "OVERSTOCK",
        "value_at_stake"
    ].fillna(0).sum()
    if "value_at_stake" in filtered_df.columns
    else 0
)


# ============================================================
# KPI ROW
# ============================================================

st.markdown("### 📊 Inventory Risk Overview")

col1, col2, col3, col4, col5 = st.columns(5)


with col1:

    st.metric(
        "Store-SKU Records",
        f"{total_items:,}"
    )


with col2:

    st.metric(
        "🔴 Stockout Risk",
        f"{stockout_count:,}"
    )


with col3:

    st.metric(
        "🟠 Overstock Risk",
        f"{overstock_count:,}"
    )


with col4:

    st.metric(
        "🟢 Normal",
        f"{normal_count:,}"
    )


with col5:

    st.metric(
        "₹ Value at Stake",
        f"₹{total_value:,.0f}"
    )


# ============================================================
# RISK VALUE
# ============================================================

st.markdown("### 💰 Financial Exposure")

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Stockout Value",
        f"₹{stockout_value:,.0f}"
    )


with col2:

    st.metric(
        "Overstock Value",
        f"₹{overstock_value:,.0f}"
    )


with col3:

    st.metric(
        "Total Exposure",
        f"₹{total_value:,.0f}"
    )


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📈 Forecast",
        "🚨 Risk",
        "📦 Reorder",
        "🏷️ Markdown",
        "🤖 Model"
    ]
)


# ============================================================
# TAB 1 - FORECAST
# ============================================================

with tab1:

    st.header(
        "Demand Forecast"
    )

    if predictions_df.empty:

        st.warning(
            "Forecast prediction file not found."
        )

    else:

        predictions_df = safe_numeric(
            predictions_df,
            [
                "prediction",
                "units_sold"
            ]
        )

        if "date" in predictions_df.columns:

            predictions_df["date"] = pd.to_datetime(
                predictions_df["date"],
                errors="coerce"
            )

        forecast_df = predictions_df.copy()


        # ----------------------------------------------------
        # SKU selector
        # ----------------------------------------------------

        if "sku_id" in forecast_df.columns:

            forecast_skus = sorted(
                forecast_df["sku_id"]
                .dropna()
                .astype(str)
                .unique()
            )

            forecast_sku = st.selectbox(
                "Select SKU",
                forecast_skus
            )

            sku_forecast = forecast_df[
                forecast_df["sku_id"]
                .astype(str)
                == forecast_sku
            ].copy()

        else:

            sku_forecast = forecast_df


        # ----------------------------------------------------
        # Aggregate
        # ----------------------------------------------------

        if "date" in sku_forecast.columns:

            daily_forecast = (
                sku_forecast
                .groupby("date", as_index=False)
                .agg(
                    actual=(
                        "units_sold",
                        "sum"
                    ),
                    forecast=(
                        "prediction",
                        "sum"
                    )
                )
            )

            plot_df = daily_forecast.melt(
                id_vars="date",
                value_vars=[
                    "actual",
                    "forecast"
                ],
                var_name="Type",
                value_name="Units"
            )

            fig = px.line(
                plot_df,
                x="date",
                y="Units",
                color="Type",
                markers=True,
                title=f"Forecast vs Actual — {forecast_sku}"
            )

            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Units"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# ============================================================
# TAB 2 - RISK
# ============================================================

with tab2:

    st.header(
        "Inventory Risk Dashboard"
    )

    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # RISK DISTRIBUTION
    # --------------------------------------------------------

    with col1:

        risk_counts = (
            filtered_df["risk_level"]
            .value_counts()
            .reset_index()
        )

        risk_counts.columns = [
            "risk_level",
            "count"
        ]

        fig = px.pie(
            risk_counts,
            names="risk_level",
            values="count",
            title="Risk Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # --------------------------------------------------------
    # VALUE BY RISK
    # --------------------------------------------------------

    with col2:

        if "value_at_stake" in filtered_df.columns:

            value_data = (
                filtered_df
                .groupby("risk_level")[
                    "value_at_stake"
                ]
                .sum()
                .reset_index()
            )

            fig = px.bar(
                value_data,
                x="risk_level",
                y="value_at_stake",
                title="Rupee Value at Stake",
                text_auto=".2s"
            )

            fig.update_layout(
                xaxis_title="Risk Level",
                yaxis_title="₹ Value"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


    # --------------------------------------------------------
    # RISK TABLE
    # --------------------------------------------------------

    st.subheader(
        "Prioritised Risk Records"
    )

    display_columns = [
        "store_id",
        "sku_id",
        "sku_name",
        "category",
        "risk_level",
        "risk_score",
        "stock_on_hand",
        "forecast_daily_demand",
        "days_of_supply",
        "lead_time_days",
        "value_at_stake",
        "recommended_action"
    ]

    display_columns = [
        c
        for c in display_columns
        if c in filtered_df.columns
    ]


    risk_table = (
        filtered_df[
            display_columns
        ]
        .sort_values(
            "risk_score",
            ascending=False
        )
        .head(100)
    )


    st.dataframe(
        risk_table,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# TAB 3 - REORDER
# ============================================================

with tab3:

    st.header(
        "🔴 Reorder Priority"
    )

    if reorder_df.empty:

        st.info(
            "No reorder priority file available."
        )

    else:

        reorder_df = safe_numeric(
            reorder_df,
            [
                "risk_score",
                "value_at_stake",
                "stock_gap",
                "lead_time_demand",
                "stock_on_hand"
            ]
        )

        st.write(
            "Products requiring the most urgent replenishment."
        )

        st.dataframe(
            reorder_df.head(100),
            use_container_width=True,
            hide_index=True
        )

        csv = reorder_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Reorder List",
            csv,
            "reorder_priority_list.csv",
            "text/csv"
        )


# ============================================================
# TAB 4 - MARKDOWN
# ============================================================

with tab4:

    st.header(
        "🟠 Overstock / Markdown Priority"
    )

    if markdown_df.empty:

        st.info(
            "No markdown priority file available."
        )

    else:

        markdown_df = safe_numeric(
            markdown_df,
            [
                "risk_score",
                "value_at_stake",
                "stock_on_hand",
                "forecast_daily_demand",
                "days_of_supply"
            ]
        )

        st.write(
            "Products carrying excessive inventory relative to expected demand."
        )

        st.dataframe(
            markdown_df.head(100),
            use_container_width=True,
            hide_index=True
        )

        csv = markdown_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Markdown List",
            csv,
            "markdown_priority_list.csv",
            "text/csv"
        )


# ============================================================
# # ============================================================
# TAB 5 - MODEL PERFORMANCE
# ============================================================

with tab5:

    st.header(
        "🤖 Demand Forecast Model"
    )

    st.write(
        "FORESIGHT uses XGBoost to forecast SKU-level demand. "
        "The model is evaluated using a time-based holdout so "
        "future observations are not used during training."
    )

    # ========================================================
    # MODEL RESULTS
    # ========================================================

    st.subheader(
        "XGBoost Test Performance"
    )

    # Actual results from train_forecast.py
    xgb_wape = 46.59
    baseline_wape = 59.62
    xgb_mae = 0.8854
    xgb_bias = 0.0022

    # Calculate improvements
    absolute_improvement = (
        baseline_wape - xgb_wape
    )

    relative_improvement = (
        absolute_improvement /
        baseline_wape
    ) * 100


    # ========================================================
    # KPI CARDS
    # ========================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "XGBoost WAPE",
            f"{xgb_wape:.2f}%"
        )

    with col2:

        st.metric(
            "Seasonal-Naive WAPE",
            f"{baseline_wape:.2f}%"
        )

    with col3:

        st.metric(
            "Relative Improvement",
            f"{relative_improvement:.2f}%"
        )

    with col4:

        st.metric(
            "Bias",
            f"{xgb_bias:.4f}"
        )


    # ========================================================
    # MODEL DECISION
    # ========================================================

    st.subheader(
        "Model Selection Decision"
    )

    if xgb_wape < baseline_wape:

        st.success(
            f"""
            **XGBoost selected**

            XGBoost achieved a WAPE of **{xgb_wape:.2f}%**
            compared with **{baseline_wape:.2f}%** for the
            seasonal-naive baseline.

            This is an improvement of **{absolute_improvement:.2f}
            percentage points**, or approximately
            **{relative_improvement:.2f}% relative improvement**.
            """
        )

    else:

        st.warning(
            """
            XGBoost does not outperform the seasonal-naive
            baseline on the current test period.
            """
        )


    # ========================================================
    # MODEL COMPARISON CHART
    # ========================================================

    st.subheader(
        "Forecast Accuracy Comparison"
    )

    comparison_df = pd.DataFrame(
        {
            "Model": [
                "Seasonal Naive",
                "XGBoost"
            ],
            "WAPE (%)": [
                baseline_wape,
                xgb_wape
            ]
        }
    )

    fig = px.bar(
        comparison_df,
        x="Model",
        y="WAPE (%)",
        text="WAPE (%)",
        title="WAPE Comparison — Lower is Better"
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    fig.update_layout(
        yaxis_title="WAPE (%)",
        xaxis_title="Model",
        yaxis_range=[
            0,
            max(baseline_wape, xgb_wape) * 1.25
        ]
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ========================================================
    # ADDITIONAL METRICS
    # ========================================================

    st.subheader(
        "Additional XGBoost Metrics"
    )

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:

        st.metric(
            "MAE",
            f"{xgb_mae:.4f}"
        )

    with metric_col2:

        st.metric(
            "Bias",
            f"{xgb_bias:.4f}"
        )


    # ========================================================
    # DATA SPLIT
    # ========================================================

    st.subheader(
        "Validation Design"
    )

    validation_data = pd.DataFrame(
        {
            "Dataset": [
                "Training",
                "Test"
            ],
            "Period": [
                "2022-01-01 → 2025-12-01",
                "2025-12-02 → 2025-12-31"
            ],
            "Rows": [
                "776,937",
                "22,350"
            ]
        }
    )

    st.dataframe(
        validation_data,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    st.subheader(
        "Model Information"
    )

    st.markdown(
        """
        **Selected algorithm:** XGBoost

        **Forecast target:** `units_sold`

        **Feature dataset:** 799,287 rows × 54 columns

        **Model features:** 41

        **Training strategy:** Time-based split

        **Training period:** 2022-01-01 to 2025-12-01

        **Test period:** 2025-12-02 to 2025-12-31

        **Test observations:** 22,350

        **Baseline:** 7-day seasonal-naive forecast

        **Primary metric:** WAPE
        """
    )


    # ========================================================
    # INTERPRETATION
    # ========================================================

    st.info(
        """
        **How to interpret this result**

        WAPE measures forecast error relative to the total demand
        volume. Lower WAPE indicates better forecasting accuracy.

        XGBoost reduces WAPE from **59.62% to 46.59%** on the
        held-out December 2025 test period.

        The model therefore provides a stronger forecasting signal
        than the seasonal-naive baseline for this evaluation period.
        """
    )


    # ========================================================
    # HONEST LIMITATION
    # ========================================================

    st.warning(
        """
        **Validation limitation**

        The 46.59% result comes from a single time-based holdout
        covering 2025-12-02 to 2025-12-31.

        For production confidence, rolling-origin backtesting should
        also be considered to verify that the improvement is stable
        across multiple forecast periods.
        """
    )
    # --------------------------------------------------------
    # CURRENT MODEL INFORMATION
    # --------------------------------------------------------



# ============================================================
# ASSUMPTIONS
# ============================================================

st.divider()

st.subheader(
    "⚠️ Planning Assumptions"
)

st.info(
    """
    **Current risk-scoring assumptions**

    • Inventory data does not contain a date column, so the latest
      available inventory snapshot is used.

    • `on_order_units` is not available in the source inventory data,
      therefore it is currently treated as 0.

    • Supplier lead time is not available in the source data,
      therefore a default 7-day lead time is used.

    • Risk is calculated at the store-SKU level.

    • Value at stake is calculated using the SKU unit price.

    These assumptions should be replaced with operational values
    when NorthBay provides them.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "NORTHBAY FORESIGHT | Demand Planning & Inventory Risk Intelligence"
)