import streamlit as st
import pandas as pd
import requests


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NorthBay FORESIGHT",
    page_icon="📦",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

# LOCAL DEVELOPMENT
API_URL = "http://127.0.0.1:5000"

# AFTER RENDER DEPLOYMENT, CHANGE TO:
# API_URL = "https://your-api-name.onrender.com"

response = requests.get(
    f"{API_URL}/forecast",
    timeout=30
)

if response.status_code == 200:
    forecast_data = response.json()
else:
    st.error("Forecast service is unavailable.")
# ============================================================

st.title("📦 NorthBay FORESIGHT")

st.caption(
    "AI-powered demand forecasting and inventory risk planning"
)


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
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Controls")

st.sidebar.write(
    f"API: `{API_URL}`"
)


# ============================================================
# API STATUS
# ============================================================

health = api_get("/health")


if health:

    st.sidebar.success("🟢 API Connected")

else:

    st.sidebar.error("🔴 API Offline")


# ============================================================
# MODEL INFORMATION
# ============================================================

st.subheader("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Model",
        "XGBoost"
    )

with col2:

    st.metric(
        "XGBoost WAPE",
        "46.59%"
    )

with col3:

    st.metric(
        "Baseline WAPE",
        "59.62%"
    )

with col4:

    improvement = (
        (59.62 - 46.59) /
        59.62
    ) * 100

    st.metric(
        "Improvement",
        f"{improvement:.2f}%"
    )


st.info(
    "XGBoost improves forecast WAPE from "
    "59.62% for the seasonal-naive baseline "
    "to 46.59%."
)


# ============================================================
# KPI SECTION
# ============================================================

st.subheader("📦 Inventory Overview")

if health:

    prediction_rows = health.get(
        "prediction_rows",
        0
    )

    risk_rows = health.get(
        "risk_rows",
        0
    )

else:

    prediction_rows = 0
    risk_rows = 0


col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Forecast Records",
        f"{prediction_rows:,}"
    )

with col2:

    st.metric(
        "Risk Records",
        f"{risk_rows:,}"
    )

with col3:

    st.metric(
        "Forecast Accuracy",
        "46.59% WAPE"
    )


# ============================================================
# SKU SEARCH
# ============================================================

st.subheader("🔎 SKU Analysis")

sku_id = st.text_input(
    "Enter SKU ID",
    placeholder="Example: SKU00001"
)
###########################
sku_id = st.selectbox(
    "Select SKU",
    sku_id.split(",") if sku_id else [],
    index=0 if sku_id else -1
)

response = requests.get(
    f"{API_URL}/sku/{sku_id}",
    timeout=30
)

if response.status_code == 200:
    data = response.json()
    st.json(data)
else:
    st.error("Unable to retrieve SKU information.")
####========================
if sku_id:

    data = api_get(
        f"/sku/{sku_id.strip()}"
    )

    if data:

        st.markdown(
            f"### SKU: `{sku_id}`"
        )

        # ----------------------------------------------------
        # FORECAST
        # ----------------------------------------------------

        forecast = data.get(
            "forecast",
            []
        )

        if forecast:

            forecast_df = pd.DataFrame(
                forecast
            )

            st.write("### 📈 Demand Forecast")

            if "date" in forecast_df.columns:

                forecast_df["date"] = pd.to_datetime(
                    forecast_df["date"],
                    errors="coerce"
                )

            st.dataframe(
                forecast_df,
                use_container_width=True,
                hide_index=True
            )
        


            if "prediction" in forecast_df.columns:

                chart_df = (
                    forecast_df[
                        ["date", "prediction"]
                    ]
                    .set_index("date")
                )

                st.line_chart(
                    chart_df
                )

        else:

            st.warning(
                "No forecast available for this SKU."
            )


        # ----------------------------------------------------
        # RISK
        # ----------------------------------------------------

        risk = data.get(
            "risk",
            []
        )

        if risk:

            risk_df = pd.DataFrame(
                risk
            )

            st.write("### ⚠️ Inventory Risk")

            st.dataframe(
                risk_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No risk record available for this SKU."
            )


# ============================================================
# RISK EXPLORER
# ============================================================

st.divider()

st.subheader("⚠️ Inventory Risk Explorer")

risk_filter = st.selectbox(
    "Risk Type",
    [
        "ALL",
        "STOCKOUT",
        "OVERSTOCK"
    ]
)

if st.button("Load Risk Data"):

    params = {}

    if risk_filter != "ALL":

        params["risk_level"] = risk_filter

    risk_response = api_get(
        "/risk",
        params=params
    )

    if risk_response:

        risk_records = risk_response.get(
            "risk",
            []
        )

        if risk_records:

            risk_df = pd.DataFrame(
                risk_records
            )

            st.success(
                f"{len(risk_df):,} risk records loaded."
            )

            # ------------------------------------------------
            # SUMMARY
            # ------------------------------------------------

            if "risk_level" in risk_df.columns:

                counts = (
                    risk_df["risk_level"]
                    .value_counts()
                )

                c1, c2, c3 = st.columns(3)

                with c1:

                    st.metric(
                        "Total",
                        f"{len(risk_df):,}"
                    )

                with c2:

                    st.metric(
                        "Stockout",
                        f"{counts.get('STOCKOUT', 0):,}"
                    )

                with c3:

                    st.metric(
                        "Overstock",
                        f"{counts.get('OVERSTOCK', 0):,}"
                    )

            # ------------------------------------------------
            # PRIORITY TABLE
            # ------------------------------------------------

            st.write(
                "### Prioritised Risk List"
            )

            priority_columns = [
                "sku_id",
                "sku_name",
                "category",
                "risk_level",
                "risk_score",
                "recommended_action",
                "value_at_stake"
            ]

            available = [
                col
                for col in priority_columns
                if col in risk_df.columns
            ]

            if available:

                display_df = risk_df[
                    available
                ].copy()

                if "value_at_stake" in display_df.columns:

                    display_df = display_df.sort_values(
                        "value_at_stake",
                        ascending=False
                    )

                st.dataframe(
                    display_df.head(100),
                    use_container_width=True,
                    hide_index=True
                )

        else:

            st.info(
                "No matching risk records."
            )


# ============================================================
# BUSINESS ACTIONS
# ============================================================

st.divider()

st.subheader("🎯 Recommended Business Actions")

col1, col2 = st.columns(2)

with col1:

    st.markdown(
        """
        ### 🔴 Stockout

        **Action: REORDER**

        Prioritize products where forecast demand
        during supplier lead time exceeds available
        inventory.

        - Protect sales
        - Reduce lost revenue
        - Prioritize high-value SKUs
        """
    )

with col2:

    st.markdown(
        """
        ### 🟠 Overstock

        **Action: MARKDOWN**

        Prioritize products where available inventory
        materially exceeds expected demand.

        - Reduce holding cost
        - Free warehouse capacity
        - Recover working capital
        """
    )


# ============================================================
# MODEL DETAILS
# ============================================================

with st.expander("ℹ️ Model & Methodology"):

    st.markdown(
        """
        **Forecasting model:** XGBoost

        **Baseline:** Seasonal Naive

        **Baseline WAPE:** 59.62%

        **XGBoost WAPE:** 46.59%

        **Relative improvement:** 21.86%

        **Test period:** 2 December 2025 – 31 December 2025

        **Test observations:** 22,350

        **Risk methodology:**

        - Forecast demand is compared with available inventory.
        - Supplier lead time is considered.
        - Stockout risk identifies insufficient inventory.
        - Overstock risk identifies excess inventory.
        - Each risk receives a transparent score.
        - Recommended actions are REORDER or MARKDOWN.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "NorthBay FORESIGHT | Demand Planning & Inventory Risk Intelligence"
)
