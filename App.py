import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from io import StringIO

# ==============================================================================
# STREAMLIT CONFIGURATION & CUSTOM STYLE
# ==============================================================================
st.set_page_config(
    page_title="AI Adoption & Macroeconomic Outcomes",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom css styling for a clean, professional financial/economic dashboard
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1.25rem;
        border-radius: 0.5rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-title {
        font-size: 0.875rem;
        color: #64748B;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 0.25rem;
    }
    .metric-delta {
        font-size: 0.875rem;
        margin-top: 0.25rem;
        font-weight: 600;
    }
    .delta-positive { color: #10B981; }
    .delta-negative { color: #EF4444; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# DATA INJECTOR (SELF-CONTAINED DATA TO AVOID MISSING FILES AT RUNTIME)
# ==============================================================================
@st.cache_data
def load_data():
    # Inject the exact dataset from your Google Colab research
    csv_data = """Year,total_postings,ai_postings,ai_intensity,avg_monthly_earnings,output_per_worker,unemployment_rate
2019,907,4,0.00441,4233.085,137114.11,3.669
2020,898,5,0.00557,4502.189,142544.73,8.055
2021,1191,19,0.01595,4600.127,146363.48,5.349
2022,1356,382,0.28171,4844.618,145652.92,3.65
2023,1066,347,0.32552,5334.394,147857.38,3.638
2024,990,239,0.24141,5985.292,151532.02,4.022
2025,1122,258,0.22995,6272.925,154263.67,4.282"""
    
    df = pd.read_csv(StringIO(csv_data))
    
    # Pre-calculate lag features for dynamic analysis
    df['ai_intensity_lag1'] = df['ai_intensity'].shift(1)
    df['ai_intensity_lag2'] = df['ai_intensity'].shift(2)
    return df

@st.cache_data
def load_sector_data():
    sectors = {
        "Sector": [
            "Management of Companies and Enterprises",
            "Utilities",
            "Transportation and Warehousing",
            "Educational Services",
            "Health Care and Social Assistance",
            "Wholesale Trade",
            "Administrative and Support / Waste Mgmt",
            "Professional, Scientific, and Technical Services",
            "Finance and Insurance",
            "Information / Technology"
        ],
        "AI_Intensity_Pct": [100.0, 100.0, 36.4, 35.7, 31.2, 30.0, 29.8, 28.5, 24.6, 22.1],
        "Total_Postings": [120, 85, 320, 440, 580, 210, 350, 820, 610, 740]
    }
    return pd.DataFrame(sectors)

df_macro = load_data()
df_sectors = load_sector_data()

# ==============================================================================
# SIDEBAR CONTROLS & FILTERING OPTIONS
# ==============================================================================
st.sidebar.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=80)
st.sidebar.title("Dashboard Controls")
st.sidebar.markdown("Filter and adjust parameters for the econometric analysis.")

# 1. Year Filter
min_year = int(df_macro['Year'].min())
max_year = int(df_macro['Year'].max())
selected_years = st.sidebar.slider(
    "Select Year Range",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)

# Filter main dataframe based on selection
df_filtered = df_macro[(df_macro['Year'] >= selected_years[0]) & (df_macro['Year'] <= selected_years[1])]

# 2. Regression Variable Selector
st.sidebar.subheader("OLS Regression Settings")
dependent_var = st.sidebar.selectbox(
    "Dependent Variable (Y)",
    options=["avg_monthly_earnings", "output_per_worker", "unemployment_rate"],
    format_func=lambda x: {
        "avg_monthly_earnings": "Avg Monthly Earnings ($)",
        "output_per_worker": "Labor Productivity / Output per Worker ($)",
        "unemployment_rate": "Unemployment Rate (%)"
    }[x]
)

independent_vars = st.sidebar.multiselect(
    "Independent Variables (X)",
    options=["ai_intensity", "ai_intensity_lag1", "ai_intensity_lag2", "unemployment_rate", "output_per_worker"],
    default=["ai_intensity"],
    format_func=lambda x: {
        "ai_intensity": "AI Adoption Intensity",
        "ai_intensity_lag1": "AI Intensity (Lag 1 Year)",
        "ai_intensity_lag2": "AI Intensity (Lag 2 Years)",
        "unemployment_rate": "Unemployment Rate (%)",
        "output_per_worker": "Labor Productivity ($)"
    }[x]
)

# Ensure dependent variable is not in independent variables to avoid infinite recursion / trivial identity
independent_vars = [var for var in independent_vars if var != dependent_var]

# ==============================================================================
# MAIN PAGE LAYOUT
# ==============================================================================
st.markdown('<div class="main-header">AI Job Adoption & Economic Outcomes</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Analyzing the structural impact of Artificial Intelligence on wages, productivity, and unemployment (2019-2025)</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# KPI Row
# ------------------------------------------------------------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

# Latest values from the dataset
latest_row = df_macro.iloc[-1]
prev_row = df_macro.iloc[-2]

with kpi1:
    delta_ai = (latest_row['ai_intensity'] - prev_row['ai_intensity']) * 100
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">AI Adoption Intensity</div>
            <div class="metric-value">{latest_row['ai_intensity'] * 100:.2f}%</div>
            <div class="metric-delta delta-{'negative' if delta_ai < 0 else 'positive'}">
                {'↓' if delta_ai < 0 else '↑'} {abs(delta_ai):.2f}% vs. prev year
            </div>
        </div>
    """, unsafe_allow_html=True)

with kpi2:
    delta_wages = ((latest_row['avg_monthly_earnings'] - prev_row['avg_monthly_earnings']) / prev_row['avg_monthly_earnings']) * 100
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Avg Monthly Earnings</div>
            <div class="metric-value">${latest_row['avg_monthly_earnings']:,.2f}</div>
            <div class="metric-delta delta-{'negative' if delta_wages < 0 else 'positive'}">
                {'↓' if delta_wages < 0 else '↑'} {abs(delta_wages):.2f}% vs. prev year
            </div>
        </div>
    """, unsafe_allow_html=True)

with kpi3:
    delta_prod = ((latest_row['output_per_worker'] - prev_row['output_per_worker']) / prev_row['output_per_worker']) * 100
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Labor Productivity</div>
            <div class="metric-value">${latest_row['output_per_worker']:,.2f}</div>
            <div class="metric-delta delta-{'negative' if delta_prod < 0 else 'positive'}">
                {'↓' if delta_prod < 0 else '↑'} {abs(delta_prod):.2f}% vs. prev year
            </div>
        </div>
    """, unsafe_allow_html=True)

with kpi4:
    delta_unemp = latest_row['unemployment_rate'] - prev_row['unemployment_rate']
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Unemployment Rate</div>
            <div class="metric-value">{latest_row['unemployment_rate']:.2f}%</div>
            <div class="metric-delta delta-{'positive' if delta_unemp < 0 else 'negative'}">
                {'↓' if delta_unemp < 0 else '↑'} {abs(delta_unemp):.2f}% vs. prev year
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ------------------------------------------------------------------------------
# Tabs for Navigation
# ------------------------------------------------------------------------------
tab_macro, tab_sector, tab_regression = st.tabs([
    "📊 Macroeconomic Trends", 
    "🏢 Sector Analysis", 
    "📈 Econometric Regressions"
])

# ==============================================================================
# TAB 1: MACROECONOMIC TRENDS
# ==============================================================================
with tab_macro:
    st.subheader("Dual-Axis Labor Market Trends")
    st.write("Examine how changes in AI job posting intensity track together with economic outcome measures over time.")
    
    # Metric Selector for Dual-Axis Visualizer
    macro_metric = st.selectbox(
        "Choose Secondary Economic Metric to Compare with AI Intensity",
        options=["avg_monthly_earnings", "output_per_worker", "unemployment_rate"],
        format_func=lambda x: {
            "avg_monthly_earnings": "Avg Monthly Earnings (Wages)",
            "output_per_worker": "Labor Productivity (GDP per Worker)",
            "unemployment_rate": "Unemployment Rate (%)"
        }[x]
    )
    
    # Constructing a beautiful interactive Dual Axis Plotly Chart
    fig_dual = go.Figure()
    
    # Line 1: AI Intensity
    fig_dual.add_trace(go.Scatter(
        x=df_filtered['Year'],
        y=df_filtered['ai_intensity'] * 100,
        name="AI Intensity (%)",
        mode="lines+markers",
        line=dict(color="#1E3A8A", width=3),
        marker=dict(size=8, symbol="circle")
    ))
    
    # Determine metric style/labels
    metric_label = {
        "avg_monthly_earnings": "Avg Monthly Earnings ($)",
        "output_per_worker": "Labor Productivity ($)",
        "unemployment_rate": "Unemployment Rate (%)"
    }[macro_metric]
    
    metric_color = {
        "avg_monthly_earnings": "#F59E0B", # Gold
        "output_per_worker": "#10B981", # Green
        "unemployment_rate": "#EF4444" # Red
    }[macro_metric]
    
    # Line 2: The compared macro metric
    fig_dual.add_trace(go.Scatter(
        x=df_filtered['Year'],
        y=df_filtered[macro_metric],
        name=metric_label,
        mode="lines+markers",
        line=dict(color=metric_color, width=2.5, dash="dash"),
        marker=dict(size=8, symbol="square"),
        yaxis="y2"
    ))
    
    # Update axes styling
    fig_dual.update_layout(
        title=dict(
            text=f"AI Adoption Intensity vs. {metric_label} Over Time",
            font=dict(size=16, color="#0F172A")
        ),
        xaxis=dict(
            title="Year",
            tickmode="linear",
            tick0=min_year,
            dtick=1,
            gridcolor="#F1F5F9"
        ),
        yaxis=dict(
    title=dict(text="AI Adoption Intensity (%)", font=dict(color="#1E3A8A")),
    tickfont=dict(color="#1E3A8A"),
    gridcolor="#F1F5F9"
),
yaxis2=dict(
    title=dict(text=metric_label, font=dict(color=metric_color)),
    tickfont=dict(color=metric_color),
    overlaying="y",
    side="right"
),
        legend=dict(x=0.05, y=0.95, bgcolor="rgba(255,255,255,0.8)"),
        plot_bgcolor="white",
        hovermode="x unified",
        height=500,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    st.plotly_chart(fig_dual, use_container_width=True)
    
    # Pearson Correlations summary
    st.markdown("#### Pearson Correlation Analysis")
    col_c1, col_c2, col_c3 = st.columns(3)
    
    with col_c1:
        corr_wage = df_filtered['ai_intensity'].corr(df_filtered['avg_monthly_earnings'])
        st.metric("Correlation: AI vs Wages", f"{corr_wage:.4f}", help="Strong positive correlation signifies both rose together over the timeline.")
        
    with col_c2:
        corr_prod = df_filtered['ai_intensity'].corr(df_filtered['output_per_worker'])
        st.metric("Correlation: AI vs Productivity", f"{corr_prod:.4f}", help="Indicates the link between AI recruiting demand share (based on postings to recruit) and national labor productivity output.")
        
    with col_c3:
        corr_unemp = df_filtered['ai_intensity'].corr(df_filtered['unemployment_rate'])
        st.metric("Correlation: AI vs Unemployment", f"{corr_unemp:.4f}", help="Negative correlation indicates that as AI adoption increased, unemployment declined.")

# ==============================================================================
# TAB 2: SECTOR ANALYSIS
# ==============================================================================
with tab_sector:
    st.subheader("Industry Sector Adoption (2025/2026)")
    st.write("Understand which sectors demonstrate the highest share of AI-generated job postings to recruit.")
    
    col_s1, col_s2 = st.columns([3, 2])
    
    with col_s1:
        # Sector bar chart
        fig_sec = px.bar(
            df_sectors.sort_values("AI_Intensity_Pct", ascending=True),
            x="AI_Intensity_Pct",
            y="Sector",
            orientation="h",
            labels={"AI_Intensity_Pct": "AI Posting Intensity (%)", "Sector": "Industry Sector (NAICS)"},
            title="AI Posting Intensity by NAICS Sector",
            color="AI_Intensity_Pct",
            color_continuous_scale="Blues",
        )
        fig_sec.update_layout(
            plot_bgcolor="white",
            height=450,
            coloraxis_showscale=False,
            xaxis=dict(gridcolor="#F1F5F9"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            margin=dict(t=50, b=30, l=10, r=10)
        )
        st.plotly_chart(fig_sec, use_container_width=True)
        
    with col_s2:
        st.write("#### Detailed Sector Metrics")
        st.dataframe(
            df_sectors.sort_values("AI_Intensity_Pct", ascending=False),
            column_config={
                "Sector": "Industry",
                "AI_Intensity_Pct": st.column_config.ProgressColumn(
                    "AI Intensity (%)",
                    format="%.1f%%",
                    min_value=0.0,
                    max_value=100.0
                ),
                "Total_Postings": "Job Postings Sampled"
            },
            hide_index=True,
            use_container_width=True
        )

# ==============================================================================
# TAB 3: ECONOMETRIC REGRESSIONS
# ==============================================================================
with tab_regression:
    st.subheader("Interactive Econometric Model Estimator (OLS)")
    st.write("Leverage standard ordinary least squares (OLS) regressions via **statsmodels** to model macroeconomic relationships.")
    
    # 1. Check if independent variables are empty
    if not independent_vars:
        st.warning("⚠️ Please select at least one independent variable (X) in the sidebar to run the regression model.")
    else:
        # Drop rows with NaN if lag was selected
        df_model = df_filtered[[dependent_var] + independent_vars].dropna()
        
        if len(df_model) < len(independent_vars) + 2:
            st.error("❌ Not enough data points available in the selected year range to estimate the regression. Please extend the Year Range slider or reduce independent variables.")
        else:
            # Fit OLS
            Y_reg = df_model[dependent_var]
            X_reg = sm.add_constant(df_model[independent_vars])
            
            model = sm.OLS(Y_reg, X_reg).fit()
            
            # Print regression results
            st.markdown(f"### Regression Target: **{dependent_var.replace('_', ' ').title()}**")
            
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                st.metric("R-Squared (Explained Variance)", f"{model.rsquared:.4f}")
            with col_r2:
                st.metric("Adj. R-Squared", f"{model.rsquared_adj:.4f}")
            with col_r3:
                st.metric("F-Statistic", f"{model.fvalue:.4f}")
                
            # Render coefficients table nicely
            st.markdown("#### Estimated Coefficients and Statistical Significance")
            
            # Extract statsmodels details into a pandas DataFrame
            coef_df = pd.DataFrame({
                "Coefficient (Beta)": model.params,
                "Std Error": model.bse,
                "t-Statistic": model.tvalues,
                "p-Value": model.pvalues,
                "[0.025 Conf. Int.]": model.conf_int()[0],
                "[0.975 Conf. Int.]": model.conf_int()[1]
            })
            
            # Highlight p-values
            st.dataframe(
                coef_df.style.format({
                    "Coefficient (Beta)": "{:,.5f}",
                    "Std Error": "{:,.5f}",
                    "t-Statistic": "{:,.4f}",
                    "p-Value": "{:,.4f}",
                    "[0.025 Conf. Int.]": "{:,.5f}",
                    "[0.975 Conf. Int.]": "{:,.5f}"
                }).map(
                    lambda v: 'background-color: #DEF7EC; color: #03543F;' if v < 0.05 else '',
                    subset=["p-Value"]
                ),
                use_container_width=True
            )
            
            st.info("💡 **Tip**: Cells highlighted in green indicate statistical significance at the 95% confidence level (p-value < 0.05).")
            
            # Show regression formula
            formula_terms = [f"({val:,.3f} * {var})" for var, val in model.params.items() if var != 'const']
            formula_str = f"Estimated {dependent_var.replace('_', ' ').title()} = {model.params['const']:,.3f} " + "".join([f" + {term}" if val >= 0 else f" - {term.replace('-', '')}" for term, val in zip(formula_terms, [v for k, v in model.params.items() if k != 'const'])])
            st.code(formula_str, language="text")

# Footer
st.markdown("---")
st.markdown("💻 *Developed for Research. To run locally, save this file as `app.py` and execute `streamlit run app.py`.*")
