
import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# SHOPIFLOW - AI/ML BUSINESS INTELLIGENCE & RETAIL MANAGEMENT
# ============================================================

st.set_page_config(
    page_title="ShopiFlow | Retail Management",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Keep the original dataset and model filenames so your
# existing project files continue to work.

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "smartmart_data.csv"
)

ELECTRICITY_MODEL_FILE = os.path.join(
    BASE_DIR,
    "models",
    "electricity_model.pkl"
)

WATER_MODEL_FILE = os.path.join(
    BASE_DIR,
    "models",
    "water_model.pkl"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    if not os.path.exists(DATA_FILE):
        st.error(f"Dataset not found: {DATA_FILE}")
        st.stop()

    data = pd.read_csv(DATA_FILE)

    if data.empty:
        st.error("The dataset is empty.")
        st.stop()

    required_columns = [
        "date",
        "revenue",
        "net_profit",
        "customers",
        "total_resource_cost",
        "electricity_kwh",
        "water_m3",
        "fuel_liters",
        "waste_kg",
        "operating_profit",
        "electricity_cost",
        "water_cost",
        "fuel_cost",
        "packaging_cost",
        "cleaning_cost",
        "land_value",
        "building_value",
        "equipment_value",
        "inventory",
        "cash",
        "accounts_receivable",
        "total_assets",
        "bank_loan",
        "accounts_payable",
        "total_liabilities",
        "total_equity",
        "electricity_per_customer",
        "water_per_customer",
        "resource_cost_percentage"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        st.error(
            "The dataset is missing required columns: "
            + ", ".join(missing_columns)
        )
        st.stop()

    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce"
    )

    data = data.dropna(subset=["date"])

    data = data.sort_values("date").reset_index(drop=True)

    if data.empty:
        st.error("No valid dates were found in the dataset.")
        st.stop()

    return data


# ============================================================
# LOAD MACHINE LEARNING MODELS
# ============================================================

@st.cache_resource
def load_models():
    if not os.path.exists(ELECTRICITY_MODEL_FILE):
        st.error(
            f"Electricity model not found: "
            f"{ELECTRICITY_MODEL_FILE}"
        )
        st.stop()

    if not os.path.exists(WATER_MODEL_FILE):
        st.error(
            f"Water model not found: {WATER_MODEL_FILE}"
        )
        st.stop()

    electricity_model = joblib.load(
        ELECTRICITY_MODEL_FILE
    )

    water_model = joblib.load(
        WATER_MODEL_FILE
    )

    return electricity_model, water_model


# ============================================================
# INITIALIZE APPLICATION DATA
# ============================================================

df = load_data()

electricity_model, water_model = load_models()

latest = df.iloc[-1]


# ============================================================
# CURRENCY SETTINGS
# ============================================================

st.sidebar.title("🛒 ShopiFlow")

st.sidebar.caption(
    "AI-Powered Retail Intelligence"
)

currency = st.sidebar.selectbox(
    "💱 Display Currency",
    [
        "INR (₹)",
        "USD ($)",
        "EUR (€)"
    ]
)

# Illustrative conversion rates.
# Update these rates when required.
currency_rates = {
    "INR (₹)": 1.00,
    "USD ($)": 0.0117,
    "EUR (€)": 0.0100
}

currency_symbols = {
    "INR (₹)": "₹",
    "USD ($)": "$",
    "EUR (€)": "€"
}

rate = currency_rates[currency]
symbol = currency_symbols[currency]


def money(value):
    """Format financial values in the selected currency."""
    try:
        if pd.isna(value):
            return f"{symbol}0"

        return f"{symbol}{float(value) * rate:,.0f}"

    except (ValueError, TypeError):
        return f"{symbol}0"


# ============================================================
# PROFESSIONAL UI STYLE
# ============================================================

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: #F3F4F6;
    }

    [data-testid="stHeader"] {
        background-color: #F3F4F6;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3, h4 {
        color: #111827 !important;
        font-weight: 700 !important;
    }

    p, label {
        color: #1F2937 !important;
    }

    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #D1D5DB;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
    }

    [data-testid="stMetricLabel"] {
        color: #4B5563 !important;
    }

    [data-testid="stMetricValue"] {
        color: #111827 !important;
    }

    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
    }

    [data-testid="stSidebar"] p {
        color: #1F2937 !important;
    }

    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        min-height: 42px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PLOTLY CHART THEME
# ============================================================

def apply_chart_theme(fig):
    fig.update_layout(
        paper_bgcolor="#F3F4F6",
        plot_bgcolor="#E5E7EB",

        font=dict(
            family="Arial",
            color="#111827",
            size=13
        ),

        title=dict(
            font=dict(
                family="Arial",
                color="#111827",
                size=19
            ),
            x=0.02,
            xanchor="left"
        ),

        xaxis=dict(
            title_font=dict(
                color="#111827",
                size=14
            ),
            tickfont=dict(
                color="#111827",
                size=12
            ),
            gridcolor="#B8BEC8",
            linecolor="#6B7280",
            showline=True
        ),

        yaxis=dict(
            title_font=dict(
                color="#111827",
                size=14
            ),
            tickfont=dict(
                color="#111827",
                size=12
            ),
            gridcolor="#B8BEC8",
            linecolor="#6B7280",
            showline=True
        ),

        legend=dict(
            font=dict(
                color="#111827",
                size=12
            ),
            bgcolor="rgba(255,255,255,0.90)",
            bordercolor="#9CA3AF",
            borderwidth=1
        ),

        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor="#6B7280",
            font=dict(
                color="#111827",
                size=12
            )
        ),

        margin=dict(
            l=70,
            r=30,
            t=70,
            b=60
        )
    )

    return fig


# ============================================================
# REUSABLE CHART FUNCTION
# ============================================================

def show_line_chart(
    data,
    x,
    y,
    title,
    y_title,
    color=None,
    tickprefix=None
):
    fig = px.line(
        data,
        x=x,
        y=y,
        color=color,
        title=title,
        markers=True
    )

    fig.update_yaxes(
        title=y_title,
        tickprefix=tickprefix or ""
    )

    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=7)
    )

    st.plotly_chart(
        apply_chart_theme(fig),
        use_container_width=True
    )


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.divider()

st.sidebar.subheader("🧭 Navigation")

page = st.sidebar.radio(
    "Main Section",
    [
        "📊 Dashboard",
        "💰 Financial Analysis",
        "📑 Balance Sheet",
        "🤖 AI Predictions",
        "🌱 Sustainability",
        "⚡ Resource Consumption"
    ],
    label_visibility="collapsed"
)


# ============================================================
# RESOURCE SUB-NAVIGATION
# ============================================================

resource_category = "📊 Overview"

if page == "⚡ Resource Consumption":

    st.sidebar.divider()

    st.sidebar.subheader(
        "⚡ Resource Categories"
    )

    resource_category = st.sidebar.radio(
        "Select Resource",
        [
            "📊 Overview",
            "⚡ Electricity",
            "💧 Water",
            "⛽ Fuel",
            "♻️ Waste"
        ],
        label_visibility="collapsed"
    )


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title("🛒 ShopiFlow")

st.subheader(
    "AI/ML Business Intelligence & Retail Management System"
)

st.write(
    """
    An intelligent retail management platform for analyzing
    business performance, financial operations, resource
    consumption, machine-learning predictions, and sustainability.
    """
)

st.caption(
    f"Latest available dataset record: "
    f"{latest['date'].strftime('%d %B %Y')}"
)

st.divider()


# ============================================================
# DASHBOARD
# ============================================================

if page == "📊 Dashboard":

    st.header("📊 Business Dashboard")

    st.write(
        "Overview of ShopiFlow's latest business performance."
    )

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "💰 Revenue",
            money(latest["revenue"])
        )

    with col2:
        st.metric(
            "📈 Net Profit",
            money(latest["net_profit"])
        )

    with col3:
        st.metric(
            "👥 Customers",
            f"{latest['customers']:,.0f}"
        )

    with col4:
        st.metric(
            "⚡ Resource Cost",
            money(latest["total_resource_cost"])
        )

    st.divider()

    # --------------------------------------------------------
    # REVENUE TREND
    # --------------------------------------------------------

    st.subheader("📈 Revenue Trend")

    show_line_chart(
        df,
        "date",
        "revenue",
        "ShopiFlow Revenue Trend",
        "Revenue",
        tickprefix=symbol
    )

    # --------------------------------------------------------
    # NET PROFIT TREND
    # --------------------------------------------------------

    st.subheader("📈 Net Profit Trend")

    show_line_chart(
        df,
        "date",
        "net_profit",
        "ShopiFlow Net Profit Trend",
        "Net Profit",
        tickprefix=symbol
    )


# ============================================================
# RESOURCE CONSUMPTION
# ============================================================

elif page == "⚡ Resource Consumption":

    st.header("⚡ Resource Consumption")

    st.write(
        "Monitor electricity, water, fuel, and waste."
    )

    st.divider()

    resource_columns = {
        "⚡ Electricity": (
            "electricity_kwh",
            "Electricity",
            "kWh"
        ),
        "💧 Water": (
            "water_m3",
            "Water",
            "m³"
        ),
        "⛽ Fuel": (
            "fuel_liters",
            "Fuel",
            "L"
        ),
        "♻️ Waste": (
            "waste_kg",
            "Waste",
            "kg"
        )
    }

    # --------------------------------------------------------
    # RESOURCE OVERVIEW
    # --------------------------------------------------------

    if resource_category == "📊 Overview":

        st.subheader("📊 Resource Overview")

        col1, col2, col3, col4 = st.columns(4)

        overview_items = [
            ("⚡ Electricity", "electricity_kwh", "kWh"),
            ("💧 Water", "water_m3", "m³"),
            ("⛽ Fuel", "fuel_liters", "L"),
            ("♻️ Waste", "waste_kg", "kg")
        ]

        for col, (label, column, unit) in zip(
            [col1, col2, col3, col4],
            overview_items
        ):
            with col:
                st.metric(
                    label,
                    f"{latest[column]:,.0f} {unit}"
                )

        st.subheader(
            "📈 All Resource Consumption Trends"
        )

        resource_data = df[
            [
                "date",
                "electricity_kwh",
                "water_m3",
                "fuel_liters",
                "waste_kg"
            ]
        ].melt(
            id_vars="date",
            var_name="Resource",
            value_name="Consumption"
        )

        show_line_chart(
            resource_data,
            "date",
            "Consumption",
            "Resource Consumption Overview",
            "Consumption",
            color="Resource"
        )

    # --------------------------------------------------------
    # INDIVIDUAL RESOURCE PAGES
    # --------------------------------------------------------

    else:

        column, name, unit = resource_columns[
            resource_category
        ]

        st.subheader(
            f"{resource_category} Consumption"
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Latest Recorded",
                f"{latest[column]:,.0f} {unit}"
            )

        with col2:
            st.metric(
                "Historical Average",
                f"{df[column].mean():,.0f} {unit}"
            )

        with col3:
            st.metric(
                "Maximum Recorded",
                f"{df[column].max():,.0f} {unit}"
            )

        show_line_chart(
            df,
            "date",
            column,
            f"{name} Consumption Trend",
            f"{name} ({unit})"
        )


# ============================================================
# FINANCIAL ANALYSIS
# ============================================================

elif page == "💰 Financial Analysis":

    st.header("💰 Financial Analysis")

    # --------------------------------------------------------
    # FINANCIAL KPI CARDS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "💰 Revenue",
            money(latest["revenue"])
        )

    with col2:
        st.metric(
            "📊 Operating Profit",
            money(latest["operating_profit"])
        )

    with col3:
        st.metric(
            "📈 Net Profit",
            money(latest["net_profit"])
        )

    st.divider()

    # --------------------------------------------------------
    # REVENUE VS NET PROFIT
    # --------------------------------------------------------

    st.subheader("📊 Revenue vs Net Profit")

    financial_data = df[
        [
            "date",
            "revenue",
            "net_profit"
        ]
    ].melt(
        id_vars="date",
        var_name="Metric",
        value_name="Amount"
    )

    show_line_chart(
        financial_data,
        "date",
        "Amount",
        "Revenue vs Net Profit",
        "Amount",
        color="Metric",
        tickprefix=symbol
    )

    # --------------------------------------------------------
    # RESOURCE COST TREND
    # --------------------------------------------------------

    st.subheader("💸 Resource Cost Trend")

    show_line_chart(
        df,
        "date",
        "total_resource_cost",
        "Total Resource Cost",
        "Resource Cost",
        tickprefix=symbol
    )

    # --------------------------------------------------------
    # COST BREAKDOWN
    # --------------------------------------------------------

    st.subheader("📊 Current Cost Breakdown")

    cost_data = pd.DataFrame({
        "Cost Type": [
            "Electricity",
            "Water",
            "Fuel",
            "Packaging",
            "Cleaning"
        ],
        "Cost": [
            float(latest["electricity_cost"]) * rate,
            float(latest["water_cost"]) * rate,
            float(latest["fuel_cost"]) * rate,
            float(latest["packaging_cost"]) * rate,
            float(latest["cleaning_cost"]) * rate
        ]
    })

    fig = px.bar(
        cost_data,
        x="Cost Type",
        y="Cost",
        title="Resource Cost Breakdown",
        text="Cost"
    )

    fig.update_yaxes(
        title=f"Cost ({symbol})"
    )

    fig.update_traces(
        texttemplate=symbol + "%{text:,.0f}",
        textposition="outside"
    )

    st.plotly_chart(
        apply_chart_theme(fig),
        use_container_width=True
    )


# ============================================================
# BALANCE SHEET
# ============================================================

elif page == "📑 Balance Sheet":

    st.header("📑 Balance Sheet")

    # --------------------------------------------------------
    # ASSETS
    # --------------------------------------------------------

    st.subheader("🏢 Assets")

    assets = pd.DataFrame({
        "Asset": [
            "Land",
            "Building",
            "Equipment",
            "Inventory",
            "Cash",
            "Accounts Receivable"
        ],
        "Amount": [
            latest["land_value"],
            latest["building_value"],
            latest["equipment_value"],
            latest["inventory"],
            latest["cash"],
            latest["accounts_receivable"]
        ]
    })

    assets_display = assets.copy()

    assets_display["Amount"] = (
        assets_display["Amount"].apply(money)
    )

    st.dataframe(
        assets_display,
        use_container_width=True,
        hide_index=True
    )

    st.metric(
        "Total Assets",
        money(latest["total_assets"])
    )

    st.divider()

    # --------------------------------------------------------
    # LIABILITIES
    # --------------------------------------------------------

    st.subheader("🏦 Liabilities")

    liabilities = pd.DataFrame({
        "Liability": [
            "Bank Loan",
            "Accounts Payable"
        ],
        "Amount": [
            latest["bank_loan"],
            latest["accounts_payable"]
        ]
    })

    liabilities_display = liabilities.copy()

    liabilities_display["Amount"] = (
        liabilities_display["Amount"].apply(money)
    )

    st.dataframe(
        liabilities_display,
        use_container_width=True,
        hide_index=True
    )

    st.metric(
        "Total Liabilities",
        money(latest["total_liabilities"])
    )

    st.divider()

    # --------------------------------------------------------
    # EQUITY
    # --------------------------------------------------------

    st.subheader("📊 Equity")

    st.metric(
        "Total Equity",
        money(latest["total_equity"])
    )

    st.divider()

    # --------------------------------------------------------
    # ACCOUNTING EQUATION
    # --------------------------------------------------------

    st.subheader("🧮 Accounting Equation")

    st.write(
        "Assets = Liabilities + Equity"
    )

    left, right = st.columns(2)

    with left:
        st.metric(
            "Assets",
            money(latest["total_assets"])
        )

    with right:
        liabilities_equity = (
            float(latest["total_liabilities"])
            + float(latest["total_equity"])
        )

        st.metric(
            "Liabilities + Equity",
            money(liabilities_equity)
        )

    difference = (
        float(latest["total_assets"])
        - liabilities_equity
    )

    if abs(difference) < 1:

        st.success(
            "✓ The balance sheet is balanced."
        )

    else:

        st.warning(
            f"Difference: {money(difference)}"
        )


# ============================================================
# AI PREDICTIONS
# ============================================================

elif page == "🤖 AI Predictions":

    st.header(
        "🤖 ShopiFlow AI Resource Predictions"
    )

    st.write(
        """
        Enter the expected business conditions below.
        ShopiFlow's trained Random Forest models will estimate
        electricity and water consumption.
        """
    )

    st.info(
        "Predictions depend on the data and features used "
        "when training the existing models."
    )

    st.divider()

    # --------------------------------------------------------
    # INPUT SECTION
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        customers = st.number_input(
            "👥 Expected Customers",
            min_value=1000,
            max_value=100000,
            value=20000,
            step=1000
        )

        employees = st.number_input(
            "👷 Employees",
            min_value=1,
            max_value=500,
            value=50,
            step=1
        )

        operating_hours = st.number_input(
            "🕐 Operating Hours per Day",
            min_value=1.0,
            max_value=24.0,
            value=14.0,
            step=0.5
        )

        sales_area = st.number_input(
            "🏪 Sales Area (m²)",
            min_value=100.0,
            max_value=10000.0,
            value=2500.0,
            step=100.0
        )

    with col2:

        revenue = st.number_input(
            "💰 Expected Revenue (₹)",
            min_value=100000.0,
            max_value=100000000.0,
            value=15000000.0,
            step=100000.0
        )

        water = st.number_input(
            "💧 Expected Water Consumption (m³)",
            min_value=100.0,
            max_value=10000.0,
            value=1500.0,
            step=100.0
        )

        fuel = st.number_input(
            "⛽ Expected Fuel Consumption (L)",
            min_value=100.0,
            max_value=50000.0,
            value=4000.0,
            step=100.0
        )

        electricity = st.number_input(
            "⚡ Expected Electricity (kWh)",
            min_value=10000.0,
            max_value=1000000.0,
            value=100000.0,
            step=5000.0
        )

    st.divider()

    # --------------------------------------------------------
    # GENERATE PREDICTIONS
    # --------------------------------------------------------

    if st.button(
        "🔮 Generate ShopiFlow Predictions",
        use_container_width=True
    ):

        try:

            electricity_input = pd.DataFrame({
                "customers": [customers],
                "employees": [employees],
                "operating_hours": [operating_hours],
                "sales_area_m2": [sales_area],
                "revenue": [revenue],
                "water_m3": [water],
                "fuel_liters": [fuel]
            })

            water_input = pd.DataFrame({
                "customers": [customers],
                "employees": [employees],
                "operating_hours": [operating_hours],
                "sales_area_m2": [sales_area],
                "revenue": [revenue],
                "electricity_kwh": [electricity]
            })

            predicted_electricity = float(
                electricity_model.predict(
                    electricity_input
                )[0]
            )

            predicted_water = float(
                water_model.predict(
                    water_input
                )[0]
            )

            st.subheader("🔮 Prediction Results")

            result1, result2 = st.columns(2)

            with result1:
                st.metric(
                    "⚡ Predicted Electricity",
                    f"{predicted_electricity:,.0f} kWh"
                )

            with result2:
                st.metric(
                    "💧 Predicted Water",
                    f"{predicted_water:,.0f} m³"
                )

            st.success(
                "ShopiFlow predictions completed using the "
                "trained Random Forest machine-learning models."
            )

            prediction_results = pd.DataFrame({
                "Resource": [
                    "Electricity",
                    "Water"
                ],
                "Predicted Consumption": [
                    predicted_electricity,
                    predicted_water
                ],
                "Unit": [
                    "kWh",
                    "m³"
                ]
            })

            st.subheader("📋 Prediction Summary")

            st.dataframe(
                prediction_results,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                label="⬇️ Download Prediction Results",
                data=prediction_results.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="shopiflow_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )

        except Exception as e:

            st.error(
                "The prediction could not be completed. "
                "Check the model files and their expected "
                "input feature names."
            )

            with st.expander("Technical Error Details"):
                st.exception(e)


# ============================================================
# SUSTAINABILITY & SDG ANALYSIS
# ============================================================

elif page == "🌱 Sustainability":

    st.header(
        "🌱 ShopiFlow Sustainability & SDG Analysis"
    )

    # --------------------------------------------------------
    # EFFICIENCY METRICS
    # --------------------------------------------------------

    electricity_efficiency = float(
        latest["electricity_per_customer"]
    )

    water_efficiency = float(
        latest["water_per_customer"]
    )

    resource_cost = float(
        latest["resource_cost_percentage"]
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "⚡ Electricity / Customer",
            f"{electricity_efficiency:.2f} kWh"
        )

    with col2:
        st.metric(
            "💧 Water / Customer",
            f"{water_efficiency:.3f} m³"
        )

    with col3:
        st.metric(
            "💰 Resource Cost / Revenue",
            f"{resource_cost:.2f}%"
        )

    st.divider()

    # --------------------------------------------------------
    # SUSTAINABLE DEVELOPMENT GOALS
    # --------------------------------------------------------

    st.subheader(
        "🌍 Relevant Sustainable Development Goals"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.info(
            """
            **SDG 6 – Clean Water and Sanitation**

            Monitor water consumption and identify
            opportunities to improve water efficiency.
            """
        )

        st.info(
            """
            **SDG 7 – Affordable and Clean Energy**

            Monitor electricity use and identify
            opportunities to improve energy efficiency.
            """
        )

        st.info(
            """
            **SDG 9 – Industry, Innovation and Infrastructure**

            Apply AI/ML and data analytics to improve
            retail resource-management decisions.
            """
        )

    with col2:

        st.info(
            """
            **SDG 12 – Responsible Consumption and Production**

            Monitor packaging, material consumption,
            and waste generation.
            """
        )

        st.info(
            """
            **SDG 13 – Climate Action**

            Track electricity and fuel consumption
            to support resource-efficiency initiatives.
            """
        )

    # --------------------------------------------------------
    # SUSTAINABILITY CHART
    # --------------------------------------------------------

    st.subheader(
        "📈 Resource Consumption Trends"
    )

    sustainability_data = df[
        [
            "date",
            "electricity_kwh",
            "water_m3",
            "waste_kg"
        ]
    ].melt(
        id_vars="date",
        var_name="Resource",
        value_name="Consumption"
    )

    show_line_chart(
        sustainability_data,
        "date",
        "Consumption",
        "ShopiFlow Sustainability Trends",
        "Consumption",
        color="Resource"
    )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "ShopiFlow | AI/ML Retail Management Project"
)

st.sidebar.caption(
    "Business Intelligence • Analytics • Sustainability"
)

st.sidebar.caption(
    "Built with Python, Streamlit, Pandas, Plotly & Scikit-learn"
)
