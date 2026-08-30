import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NorthBay FORESIGHT",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_API_URL = "http://127.0.0.1:5000"

API_URL = os.getenv(
    "API_URL",
    DEFAULT_API_URL,
).rstrip("/")


# ============================================================
# CONSTANT MODEL RESULTS
# ============================================================

MODEL_NAME = "XGBoost"

XGB_WAPE = 46.59
BASELINE_WAPE = 59.62
MAE = 0.8854
BIAS = 0.0022

RELATIVE_IMPROVEMENT = (
    (BASELINE_WAPE - XGB_WAPE)
    / BASELINE_WAPE
) * 100

WAPE_REDUCTION = BASELINE_WAPE - XGB_WAPE


# ============================================================
# API HELPER
# ============================================================

def api_get(endpoint, params=None, timeout=30):
    """
    Safely call the Flask scoring API.
    """

    url = f"{API_URL}{endpoint}"

    try:
        response = requests.get(
            url,
            params=params,
            timeout=timeout,
        )

        if response.status_code == 200:
            return response.json()

        try:
            error_message = response.json().get(
                "message",
                response.text,
            )
        except Exception:
            error_message = response.text

        st.error(
            f"API error {response.status_code}: "
            f"{error_message}"
        )

        return None

    except requests.exceptions.ConnectionError:
        st.error(
            "Unable to connect to the Flask scoring API."
        )
        return None

    except requests.exceptions.Timeout:
        st.error(
            "The scoring API timed out."
        )
        return None

    except requests.exceptions.RequestException as exc:
        st.error(
            f"API request failed: {exc}"
        )
        return None

    except ValueError:
        st.error(
            "The API returned invalid JSON."
        )
        return None

    except Exception as exc:
        st.error(
            f"Unexpected error: {exc}"
        )
        return None


# ============================================================
# TITLE
# ============================================================

st.title("📦 NORTHBAY FORESIGHT")

st.subheader(
    "AI-Powered Demand Forecasting & Inventory Planning"
)

st.caption(
    "XGBoost Forecasting • Stockout Risk • Overstock Risk • "
    "Reorder • Markdown"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ System")

st.sidebar.write(
    f"**Scoring API:** `{API_URL}`"
)


# ============================================================
# API HEALTH CHECK
# ============================================================

health = api_get(
    "/health",
    timeout=10,
)

if health:

    st.sidebar.success(
        "🟢 Scoring API Online"
    )

else:

    st.sidebar.error(
        "🔴 Scoring API Offline"
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.header("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Selected Model",
        MODEL_NAME,
    )

with col2:

    st.metric(
        "XGBoost WAPE",
        f"{XGB_WAPE:.2f}%",
    )

with col3:

    st.metric(
        "Seasonal Naive WAPE",
        f"{BASELINE_WAPE:.2f}%",
    )

with col4:

    st.metric(
        "Relative Improvement",
        f"{RELATIVE_IMPROVEMENT:.2f}%",
    )


st.success(
    f"XGBoost reduced WAPE from "
    f"{BASELINE_WAPE:.2f}% to "
    f"{XGB_WAPE:.2f}%, giving a "
    f"{RELATIVE_IMPROVEMENT:.2f}% relative improvement "
    f"({WAPE_REDUCTION:.2f} percentage points)."
)


# ============================================================
# PLANNING OVERVIEW
# ============================================================

st.header("📦 Planning Overview")


prediction_rows = 0
risk_rows = 0

if health:

    prediction_rows = health.get(
        "prediction_rows",
        0,
    )

    risk_rows = health.get(
        "risk_rows",
        0,
    )


# ============================================================
# LOAD RISK OVERVIEW
# ============================================================

risk_response = api_get(
    "/risk",
    params={"limit": 5000},
)

risk_overview = pd.DataFrame()

if risk_response:

    records = risk_response.get(
        "risk",
        [],
    )

    if isinstance(records, list):

        risk_overview = pd.DataFrame(
            records
        )


# ============================================================
# RISK KPI CALCULATION
# ============================================================

stockout_count = 0
overstock_count = 0
total_value = 0.0


if not risk_overview.empty:

    if "risk_level" in risk_overview.columns:

        stockout_count = int(
            (
                risk_overview["risk_level"]
                .astype(str)
                .str.upper()
                == "STOCKOUT"
            ).sum()
        )

        overstock_count = int(
            (
                risk_overview["risk_level"]
                .astype(str)
                .str.upper()
                == "OVERSTOCK"
            ).sum()
        )

    if "value_at_stake" in risk_overview.columns:

        risk_overview["value_at_stake"] = pd.to_numeric(
            risk_overview["value_at_stake"],
            errors="coerce",
        ).fillna(0)

        total_value = float(
            risk_overview["value_at_stake"].sum()
        )


c1, c2, c3, c4 = st.columns(4)


with c1:

    st.metric(
        "Forecast Records",
        f"{prediction_rows:,}",
    )


with c2:

    st.metric(
        "Stockout Risk",
        f"{stockout_count:,}",
    )


with c3:

    st.metric(
        "Overstock Risk",
        f"{overstock_count:,}",
    )


with c4:

    st.metric(
        "Value at Stake",
        f"₹{total_value:,.0f}",
    )


# ============================================================
# TABS
# ============================================================

tab_forecast, tab_risk, tab_reorder, tab_markdown, tab_model = st.tabs(
    [
        "📈 Forecast",
        "🚨 Risk",
        "📦 Reorder",
        "🏷️ Markdown",
        "🤖 Model",
    ]
)


# ============================================================
# TAB 1 — FORECAST
# ============================================================

with tab_forecast:

    st.header("📈 Demand Forecast")

    sku_input = st.text_input(
        "Enter SKU ID",
        placeholder="Example: SKU00001",
    )

    if sku_input:

        sku_id = sku_input.strip().upper()

        if not sku_id:

            st.warning(
                "Please enter a valid SKU ID."
            )

        else:

            with st.spinner(
                f"Loading forecast for {sku_id}..."
            ):

                response = api_get(
                    f"/sku/{sku_id}",
                    timeout=30,
                )

            if response:

                forecast_records = response.get(
                    "forecast",
                    [],
                )

                if forecast_records:

                    forecast_df = pd.DataFrame(
                        forecast_records
                    )

                    if "date" in forecast_df.columns:

                        forecast_df["date"] = pd.to_datetime(
                            forecast_df["date"],
                            errors="coerce",
                        )

                    if "prediction" in forecast_df.columns:

                        forecast_df["prediction"] = pd.to_numeric(
                            forecast_df["prediction"],
                            errors="coerce",
                        ).fillna(0).clip(lower=0)

                    st.subheader(
                        f"XGBoost Forecast — {sku_id}"
                    )

                    if {
                        "date",
                        "prediction",
                    }.issubset(forecast_df.columns):

                        chart_df = (
                            forecast_df
                            .groupby(
                                "date",
                                as_index=False,
                            )["prediction"]
                            .sum()
                        )

                        fig = px.line(
                            chart_df,
                            x="date",
                            y="prediction",
                            markers=True,
                            title=(
                                f"Forecasted Demand — "
                                f"{sku_id}"
                            ),
                        )

                        fig.update_layout(
                            xaxis_title="Date",
                            yaxis_title="Forecast Units",
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                        )

                    st.dataframe(
                        forecast_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                else:

                    st.warning(
                        f"No forecast found for {sku_id}."
                    )


# ============================================================
# TAB 2 — RISK
# ============================================================

    with tab_risk:

     st.header("🚨 Inventory Risk")

    risk_filter = st.selectbox(
        "Risk Type",
        [
            "ALL",
            "STOCKOUT",
            "OVERSTOCK",
            "NORMAL",
        ],
    )

    params = {
        "limit": 1000,
    }

    if risk_filter != "ALL":

        params["risk_level"] = risk_filter

    with st.spinner("Loading risk data..."):

        risk_response = api_get(
            "/risk",
            params=params,
            timeout=30,
        )

    if risk_response:

        records = risk_response.get(
            "risk",
            [],
        )

        risk_df = pd.DataFrame(records)

        if risk_df.empty:

            st.info(
                "No records found for this risk filter."
            )

        else:

            if "risk_score" in risk_df.columns:

                risk_df["risk_score"] = pd.to_numeric(
                    risk_df["risk_score"],
                    errors="coerce",
                ).fillna(0)

            if "value_at_stake" in risk_df.columns:

                risk_df["value_at_stake"] = pd.to_numeric(
                    risk_df["value_at_stake"],
                    errors="coerce",
                ).fillna(0)

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "Risk Records",
                    f"{len(risk_df):,}",
                )

            with c2:

                highest_score = (
                    risk_df["risk_score"].max()
                    if "risk_score" in risk_df.columns
                    else 0
                )

                st.metric(
                    "Highest Risk Score",
                    f"{highest_score:.0f}",
                )

            with c3:

                value = (
                    risk_df["value_at_stake"].sum()
                    if "value_at_stake"
                    in risk_df.columns
                    else 0
                )

                st.metric(
                    "Value at Stake",
                    f"₹{value:,.0f}",
                )

            # ------------------------------------------------
            # RISK CHART
            # ------------------------------------------------

            if "risk_level" in risk_df.columns:

                counts = (
                    risk_df["risk_level"]
                    .value_counts()
                    .reset_index()
                )

                counts.columns = [
                    "risk_level",
                    "count",
                ]

                fig = px.pie(
                    counts,
                    names="risk_level",
                    values="count",
                    title="Risk Distribution",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            # ------------------------------------------------
            # RISK TABLE
            # ------------------------------------------------

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
                "recommended_action",
            ]

            display_columns = [
                column
                for column in display_columns
                if column in risk_df.columns
            ]

            table = risk_df[display_columns].copy()

            if "risk_score" in table.columns:

                table = table.sort_values(
                    "risk_score",
                    ascending=False,
                )

            st.dataframe(
                table.head(100),
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# TAB 3 — REORDER
# ============================================================

    with tab_reorder:

     st.header("📦 Reorder Priority")

    with st.spinner(
        "Loading reorder recommendations..."
    ):

        response = api_get(
            "/reorder",
            params={"limit": 100},
        )

    if response:

        records = response.get(
            "reorder",
            [],
        )

        reorder_df = pd.DataFrame(records)

        if reorder_df.empty:

            st.success(
                "No urgent reorder items."
            )

        else:

            st.info(
                "Prioritised products requiring replenishment."
            )

            st.dataframe(
                reorder_df,
                use_container_width=True,
                hide_index=True,
            )

            csv_data = reorder_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Download Reorder List",
                csv_data,
                "reorder_priority_list.csv",
                "text/csv",
            )


# ============================================================
# TAB 4 — MARKDOWN
# ============================================================

    with tab_markdown:

     st.header("🏷️ Markdown Priority")

    with st.spinner(
        "Loading markdown recommendations..."
    ):

        response = api_get(
            "/markdown",
            params={"limit": 100},
        )

    if response:

        records = response.get(
            "markdown",
            [],
        )

        markdown_df = pd.DataFrame(records)

        if markdown_df.empty:

            st.success(
                "No urgent markdown items."
            )

        else:

            st.info(
                "Prioritised products carrying excess inventory."
            )

            st.dataframe(
                markdown_df,
                use_container_width=True,
                hide_index=True,
            )

            csv_data = markdown_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Download Markdown List",
                csv_data,
                "markdown_priority_list.csv",
                "text/csv",
            )


# ============================================================
# TAB 5 — MODEL
# ============================================================

    with tab_model:

     st.header(
        "🤖 XGBoost Demand Forecast Model"
    )

    st.write(
        "FORESIGHT uses XGBoost to forecast SKU-level "
        "demand from historical demand and engineered "
        "time-series features."
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "XGBoost WAPE",
            f"{XGB_WAPE:.2f}%",
        )

    with c2:

        st.metric(
            "Seasonal Naive",
            f"{BASELINE_WAPE:.2f}%",
        )

    with c3:

        st.metric(
            "Relative Improvement",
            f"{RELATIVE_IMPROVEMENT:.2f}%",
        )

    with c4:

        st.metric(
            "MAE",
            f"{MAE:.4f}",
        )

    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    comparison_df = pd.DataFrame(
        {
            "Model": [
                "Seasonal Naive",
                "XGBoost",
            ],
            "WAPE": [
                BASELINE_WAPE,
                XGB_WAPE,
            ],
        }
    )

    fig = px.bar(
        comparison_df,
        x="Model",
        y="WAPE",
        text="WAPE",
        title="Forecast Accuracy Comparison",
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
    )

    fig.update_layout(
        yaxis_title="WAPE (%) — Lower is Better",
        xaxis_title="Model",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # MODEL DECISION
    # --------------------------------------------------------

    st.subheader(
        "Model Selection"
    )

    st.success(
        f"""
        **XGBoost selected**

        XGBoost achieved **{XGB_WAPE:.2f}% WAPE**
        compared with **{BASELINE_WAPE:.2f}%** for the
        seasonal-naive baseline.

        This is a **{WAPE_REDUCTION:.2f} percentage-point**
        reduction in WAPE and approximately
        **{RELATIVE_IMPROVEMENT:.2f}% relative improvement**.
        """
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    st.subheader(
        "Validation"
    )

    validation_df = pd.DataFrame(
        {
            "Dataset": [
                "Training",
                "Test",
            ],
            "Period": [
                "2022-01-01 → 2025-12-01",
                "2025-12-02 → 2025-12-31",
            ],
            "Rows": [
                776937,
                22350,
            ],
        }
    )

    st.dataframe(
        validation_df,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        """
        The reported XGBoost result comes from a
        time-based holdout. The test period is kept
        after the training period so future observations
        are not used to train the model.
        """
    )

    # --------------------------------------------------------
    # ADDITIONAL METRICS
    # --------------------------------------------------------

    st.subheader(
        "Additional Metrics"
    )

    m1, m2 = st.columns(2)

    with m1:

        st.metric(
            "MAE",
            f"{MAE:.4f}",
        )

    with m2:

        st.metric(
            "Bias",
            f"{BIAS:.4f}",
        )


# ============================================================
# METHODOLOGY
# ============================================================

st.divider()

st.header(
    "ℹ️ FORESIGHT Methodology"
)

with st.expander(
    "How FORESIGHT works"
):

    st.markdown(
        """
        ### 1. Demand Forecast

        XGBoost forecasts SKU-level demand using
        leakage-safe historical demand features.

        ### 2. Inventory Risk

        Forecast demand is compared with available
        inventory and supplier lead-time assumptions.

        ### 3. Stockout Detection

        Products where expected demand can exceed
        available inventory are prioritised for
        replenishment.

        ### 4. Overstock Detection

        Products carrying materially more inventory
        than expected demand are flagged for markdown
        consideration.

        ### 5. Decisioning

        The system produces:

        - Forecast
        - Risk level
        - Risk score
        - Value at stake
        - Reorder recommendation
        - Markdown recommendation
        """
    )


# ============================================================
# ASSUMPTIONS
# ============================================================

st.divider()

st.header(
    "⚠️ Planning Assumptions"
)

st.info(
    """
    • Inventory snapshots do not contain a date field,
      therefore the latest available store-SKU inventory
      record is used.

    • `on_order_units` is not available in the current
      inventory source and is treated as 0.

    • Supplier lead time is not available in the current
      source and a default 7-day lead time is used.

    • Risk scoring is performed at store-SKU level.

    • Value at stake is based on SKU unit price.

    • Production deployment should replace these
      assumptions with live operational inventory data.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "NORTHBAY FORESIGHT | Demand Planning & Inventory Risk Intelligence"
)
