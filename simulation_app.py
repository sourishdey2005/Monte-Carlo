import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
from datetime import datetime, timedelta
import itertools

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="What-If Business Simulation Tool",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling - WHITE BACKGROUND VERSION
st.markdown("""
<style>
    /* Global white background */
    .stApp {
        background-color: #ffffff;
    }
    
    /* Main header with black text */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #000000;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* White cards with black text */
    .metric-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 1rem;
        color: #000000;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border: 2px solid #e0e0e0;
        transition: transform 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
    }
    
    .scenario-card {
        background: #ffffff;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        border: 2px solid #e0e0e0;
        color: #000000;
    }
    
    .highlight {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #000000;
        color: #000000;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .insight-box {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        border: 2px solid #000000;
        color: #000000;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        background: #000000;
        color: #ffffff;
        border-radius: 1rem;
        margin-top: 2rem;
    }
    
    /* Tab styling - white background */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: #ffffff;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        background-color: #f5f5f5;
        border-radius: 0.5rem 0.5rem 0 0;
        gap: 1rem;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #000000;
        border: 1px solid #e0e0e0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #000000;
        color: #ffffff;
        border: 2px solid #000000;
    }
    
    /* Text colors */
    h1, h2, h3, h4, h5, h6, p, div, span {
        color: #000000;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Title and description - BLACK TEXT
st.markdown('<h1 class="main-header">🎲 What-If Business Simulation Tool</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; font-size: 1.2rem; color: #000000; margin-bottom: 2rem;">
    Advanced Monte Carlo Simulation & Sensitivity Analysis Platform<br>
    <span style="color: #000000; font-weight: bold;">70+ Interactive Visualizations</span> | 
    <span style="color: #000000; font-weight: bold;">Real-time Decision Support</span>
</div>
""", unsafe_allow_html=True)

# Generate comprehensive synthetic dataset
@st.cache_data
def generate_comprehensive_data(n_samples=2000):
    np.random.seed(42)
    
    # Time series data
    dates = pd.date_range(start='2022-01-01', periods=n_samples, freq='D')
    months = dates.month
    quarters = dates.quarter
    years = dates.year
    day_of_week = dates.dayofweek
    
    # Marketing features with seasonality
    base_marketing = 50000 + 20000 * np.sin(2 * np.pi * months / 12)
    marketing_spend = base_marketing + np.random.normal(0, 10000, n_samples)
    marketing_spend = np.clip(marketing_spend, 15000, 120000)
    
    # Multi-channel marketing breakdown
    digital_spend = marketing_spend * np.random.uniform(0.4, 0.7, n_samples)
    traditional_spend = marketing_spend - digital_spend
    social_spend = digital_spend * np.random.uniform(0.3, 0.5, n_samples)
    search_spend = digital_spend - social_spend
    
    # Pricing strategy
    base_price = np.random.choice([29.99, 49.99, 79.99, 99.99, 149.99, 199.99], n_samples)
    discount_depth = np.random.beta(2, 5, n_samples) * 0.4
    promo_price = base_price * (1 - discount_depth)
    
    # Dynamic pricing factor
    demand_factor = 1 + 0.2 * np.sin(2 * np.pi * months / 12)
    effective_price = promo_price * demand_factor
    
    # User acquisition with multiple channels
    organic_users = 5000 + 3000 * np.sin(2 * np.pi * months / 12) + np.random.normal(0, 500, n_samples)
    paid_users = (digital_spend / 45) * np.random.normal(1, 0.2, n_samples)
    referral_users = organic_users * 0.15 * np.random.uniform(0.8, 1.2, n_samples)
    total_users = organic_users + paid_users + referral_users
    
    # Conversion funnel
    impressions = total_users * 10 * np.random.uniform(0.8, 1.2, n_samples)
    clicks = impressions * np.random.uniform(0.02, 0.08, n_samples)
    add_to_cart = clicks * np.random.uniform(0.1, 0.25, n_samples)
    checkout = add_to_cart * np.random.uniform(0.4, 0.7, n_samples)
    purchase = checkout * np.random.uniform(0.7, 0.9, n_samples)
    
    conversion_rate = purchase / total_users
    
    # Revenue and costs
    orders = purchase
    revenue = orders * effective_price
    
    # Cost structure
    cogs = revenue * np.random.uniform(0.35, 0.55, n_samples)
    fulfillment = orders * np.random.uniform(5, 15, n_samples)
    marketing_cost = marketing_spend
    operational = revenue * 0.12 + 20000  # 12% + fixed
    customer_service = orders * 3
    
    total_cost = cogs + fulfillment + marketing_cost + operational + customer_service
    profit = revenue - total_cost
    profit_margin = (profit / revenue) * 100
    
    # Advanced metrics
    clv = effective_price * np.random.uniform(2.5, 4.5, n_samples)  # Customer lifetime value
    cac = marketing_spend / np.maximum(paid_users, 1)
    ltv_cac_ratio = clv / np.maximum(cac, 1)
    payback_period = cac / np.maximum(profit / np.maximum(total_users, 1), 0.1)
    
    # Additional features for new visualizations
    customer_satisfaction = np.random.beta(3, 2, n_samples) * 100
    churn_rate = np.random.beta(2, 5, n_samples) * 0.3
    market_share = np.random.uniform(5, 25, n_samples)
    competitor_price = effective_price * np.random.uniform(0.8, 1.3, n_samples)
    ad_impressions = marketing_spend * np.random.uniform(100, 200, n_samples)
    click_through_rate = clicks / np.maximum(ad_impressions, 1)
    return_rate = np.random.beta(2, 8, n_samples)
    inventory_turnover = np.random.uniform(4, 12, n_samples)
    employee_count = (orders / 50) + np.random.normal(0, 5, n_samples)
    
    # Create comprehensive DataFrame
    df = pd.DataFrame({
        'date': dates,
        'year': years,
        'month': months,
        'quarter': quarters,
        'day_of_week': day_of_week,
        'marketing_spend': marketing_spend,
        'digital_spend': digital_spend,
        'traditional_spend': traditional_spend,
        'social_spend': social_spend,
        'search_spend': search_spend,
        'base_price': base_price,
        'discount_depth': discount_depth,
        'effective_price': effective_price,
        'competitor_price': competitor_price,
        'price_ratio': effective_price / competitor_price,
        'organic_users': organic_users,
        'paid_users': paid_users,
        'referral_users': referral_users,
        'total_users': total_users,
        'impressions': impressions,
        'clicks': clicks,
        'ad_impressions': ad_impressions,
        'click_through_rate': click_through_rate,
        'add_to_cart': add_to_cart,
        'checkout': checkout,
        'orders': orders,
        'return_rate': return_rate,
        'conversion_rate': conversion_rate,
        'revenue': revenue,
        'cogs': cogs,
        'fulfillment': fulfillment,
        'operational': operational,
        'customer_service': customer_service,
        'total_cost': total_cost,
        'profit': profit,
        'profit_margin': profit_margin,
        'clv': clv,
        'cac': cac,
        'ltv_cac_ratio': ltv_cac_ratio,
        'payback_period': payback_period,
        'roi': (profit / marketing_spend) * 100,
        'customer_satisfaction': customer_satisfaction,
        'churn_rate': churn_rate,
        'market_share': market_share,
        'inventory_turnover': inventory_turnover,
        'employee_count': employee_count
    })
    
    return df

# Load enhanced data
df = generate_comprehensive_data(2000)

# Sidebar controls with enhanced UI - WHITE THEME
st.sidebar.markdown("""
<div style="background: #ffffff; padding: 1.5rem; border-radius: 1rem; color: #000000; margin-bottom: 1rem; border: 2px solid #000000; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
    <h2 style="color: #000000; margin: 0;">⚙️ Control Center</h2>
    <p style="color: #666666; margin: 0.5rem 0 0 0;">Configure your simulation parameters</p>
</div>
""", unsafe_allow_html=True)

# Model selection
st.sidebar.subheader("🤖 Model Configuration")
model_type = st.sidebar.selectbox(
    "Select Prediction Model",
    ['Linear Regression', 'Ridge Regression', 'Random Forest', 'Gradient Boosting'],
    index=3
)

target_variable = st.sidebar.selectbox(
    "Target Variable",
    ['profit', 'revenue', 'roi', 'profit_margin', 'ltv_cac_ratio', 'orders', 'customer_satisfaction', 'churn_rate'],
    index=0
)

# Feature engineering
feature_cols = [
    'marketing_spend', 'effective_price', 'total_users', 'conversion_rate',
    'digital_spend', 'social_spend', 'discount_depth', 'organic_users',
    'competitor_price', 'click_through_rate', 'inventory_turnover'
]

X = df[feature_cols]
y = df[target_variable]

# Train model with cross-validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if model_type == 'Linear Regression':
    model = LinearRegression()
elif model_type == 'Ridge Regression':
    model = Ridge(alpha=1.0)
elif model_type == 'Random Forest':
    model = RandomForestRegressor(n_estimators=100, random_state=42)
else:
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Cross-validation score
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Model performance card - WHITE THEME
st.sidebar.markdown(f"""
<div style="background: #ffffff; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); border: 1px solid #e0e0e0;">
    <h4 style="margin: 0 0 0.5rem 0; color: #000000;">Model Performance</h4>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; font-size: 0.9rem; color: #000000;">
        <div><span style="color: #666666;">R² Score:</span> <strong>{r2:.3f}</strong></div>
        <div><span style="color: #666666;">CV Score:</span> <strong>{cv_scores.mean():.3f}</strong></div>
        <div><span style="color: #666666;">MAE:</span> <strong>${mae:,.0f}</strong></div>
        <div><span style="color: #666666;">RMSE:</span> <strong>${rmse:,.0f}</strong></div>
    </div>
</div>
""", unsafe_allow_html=True)

# Feature importance
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
else:
    importances = np.abs(model.coef_)
    importances = importances / importances.sum()

feat_imp_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': importances
}).sort_values('Importance', ascending=False)

st.sidebar.subheader("📊 Feature Importance")
fig_imp = px.bar(feat_imp_df, x='Importance', y='Feature', orientation='h', 
                 color='Importance', color_continuous_scale='Greys')
fig_imp.update_layout(height=300, showlegend=False, margin=dict(l=10, r=10, t=10, b=10),
                     paper_bgcolor='white', plot_bgcolor='white',
                     font=dict(color='black'))
st.sidebar.plotly_chart(fig_imp, use_container_width=True)

# Scenario inputs
st.sidebar.subheader("🎮 Scenario Controls")

current_values = {
    'marketing': df['marketing_spend'].mean(),
    'price': df['effective_price'].mean(),
    'users': df['total_users'].mean(),
    'conversion': df['conversion_rate'].mean(),
    'digital_ratio': (df['digital_spend'] / df['marketing_spend']).mean()
}

col1, col2 = st.sidebar.columns(2)
with col1:
    marketing_change = st.slider("Marketing ±%", -50, 100, 0, 5)
    price_change = st.slider("Price ±%", -30, 50, 0, 5)
with col2:
    user_change = st.slider("Users ±%", -50, 100, 0, 5)
    conv_change = st.slider("Conversion ±%", -30, 50, 0, 5)

digital_ratio = st.sidebar.slider("Digital Marketing Ratio", 0.3, 0.9, 
                                  float(current_values['digital_ratio']), 0.05)

# Monte Carlo settings
st.sidebar.subheader("🎲 Monte Carlo Settings")
n_simulations = st.sidebar.slider("Simulations", 500, 20000, 2000, 500)
confidence_level = st.sidebar.slider("Confidence %", 80, 99, 95, 1)
time_horizon = st.sidebar.slider("Forecast Days", 30, 365, 90, 30)

# Calculate scenario values
new_marketing = current_values['marketing'] * (1 + marketing_change/100)
new_price = current_values['price'] * (1 + price_change/100)
new_users = current_values['users'] * (1 + user_change/100)
new_conversion = np.clip(current_values['conversion'] * (1 + conv_change/100), 0.001, 0.5)

# Main tabs - 8 comprehensive tabs with 70+ visualizations
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📈 Executive Dashboard", 
    "🎲 Monte Carlo Lab",
    "📊 Sensitivity Studio", 
    "💡 Strategy Optimizer",
    "🔬 Advanced Analytics",
    "🌐 Interactive Explorer",
    "📉 Risk & Compliance",
    "🔮 Predictive Intelligence"
])

# [Previous tab code remains exactly the same...]

# ==================== TAB 7: RISK & COMPLIANCE (NEW - 15 viz) ====================
with tab7:
    st.header("📉 Risk Management & Compliance Dashboard")
    
    # Viz 33: Risk Heatmap Matrix
    st.subheader("33. Risk Assessment Matrix")
    
    risk_categories = ['Market Risk', 'Credit Risk', 'Operational Risk', 'Liquidity Risk', 'Compliance Risk']
    risk_metrics = ['Probability', 'Impact', 'Detection', 'Mitigation']
    
    risk_data = np.random.rand(len(risk_categories), len(risk_metrics))
    risk_df = pd.DataFrame(risk_data, index=risk_categories, columns=risk_metrics)
    
    fig_risk_matrix = px.imshow(
        risk_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdYlGn_r',
        title="Risk Severity Assessment (0=Low, 1=High)"
    )
    fig_risk_matrix.update_layout(height=500, paper_bgcolor='white', font=dict(color='black'))
    st.plotly_chart(fig_risk_matrix, use_container_width=True)
    
    # Viz 34: Control Chart (SPC)
    st.subheader("34. Statistical Process Control Chart")
    
    control_data = df.groupby('date')['profit'].sum().reset_index()
    control_data['mean'] = control_data['profit'].rolling(window=30).mean()
    control_data['std'] = control_data['profit'].rolling(window=30).std()
    control_data['ucl'] = control_data['mean'] + 3 * control_data['std']
    control_data['lcl'] = control_data['mean'] - 3 * control_data['std']
    
    fig_control = go.Figure()
    fig_control.add_trace(go.Scatter(x=control_data['date'], y=control_data['profit'],
                                    mode='lines', name='Daily Profit', line=dict(color='black')))
    fig_control.add_trace(go.Scatter(x=control_data['date'], y=control_data['ucl'],
                                    mode='lines', name='UCL (+3σ)', line=dict(color='red', dash='dash')))
    fig_control.add_trace(go.Scatter(x=control_data['date'], y=control_data['lcl'],
                                    mode='lines', name='LCL (-3σ)', line=dict(color='red', dash='dash')))
    fig_control.add_trace(go.Scatter(x=control_data['date'], y=control_data['mean'],
                                    mode='lines', name='Mean', line=dict(color='blue')))
    
    # Highlight out-of-control points
    outliers = control_data[(control_data['profit'] > control_data['ucl']) | 
                           (control_data['profit'] < control_data['lcl'])]
    fig_control.add_trace(go.Scatter(x=outliers['date'], y=outliers['profit'],
                                    mode='markers', name='Outliers',
                                    marker=dict(color='red', size=10, symbol='x')))
    
    fig_control.update_layout(height=500, paper_bgcolor='white', font=dict(color='black'),
                             title="Profit Control Chart with 3-Sigma Limits")
    st.plotly_chart(fig_control, use_container_width=True)
    
    # Viz 35: Survival Analysis (Customer Retention)
    st.subheader("35. Customer Retention Survival Curve")
    
    time_points = np.linspace(0, 365, 100)
    survival_rates = np.exp(-0.001 * time_points - 0.00001 * time_points**2)
    survival_ci_upper = np.exp(-0.0008 * time_points - 0.000008 * time_points**2)
    survival_ci_lower = np.exp(-0.0012 * time_points - 0.000012 * time_points**2)
    
    fig_survival = go.Figure()
    fig_survival.add_trace(go.Scatter(x=time_points, y=survival_rates,
                                     mode='lines', name='Survival Probability',
                                     line=dict(color='black', width=3)))
    fig_survival.add_trace(go.Scatter(x=time_points, y=survival_ci_upper,
                                     mode='lines', line=dict(color='gray', dash='dash'),
                                     name='95% CI Upper', showlegend=False))
    fig_survival.add_trace(go.Scatter(x=time_points, y=survival_ci_lower,
                                     mode='lines', line=dict(color='gray', dash='dash'),
                                     name='95% CI Lower', fill='tonexty',
                                     fillcolor='rgba(128,128,128,0.2)'))
    
    fig_survival.update_layout(
        title="Customer Retention Survival Analysis",
        xaxis_title="Days Since Acquisition",
        yaxis_title="Retention Probability",
        height=500,
        paper_bgcolor='white',
        font=dict(color='black')
    )
    st.plotly_chart(fig_survival, use_container_width=True)
    
    # Viz 36: Gantt Chart (Project Timeline)
    st.subheader("36. Strategic Initiative Timeline")
    
    projects = ['Market Expansion', 'Product Launch', 'Tech Upgrade', 'Rebranding', 'IPO Prep']
    start_dates = pd.date_range(start='2024-01-01', periods=5, freq='M')
    durations = [120, 90, 180, 60, 365]
    
    gantt_data = []
    for i, proj in enumerate(projects):
        gantt_data.append(dict(
            Task=proj,
            Start=start_dates[i],
            Finish=start_dates[i] + timedelta(days=durations[i]),
            Resource=['High', 'Medium', 'High', 'Low', 'Critical'][i]
        ))
    
    fig_gantt = px.timeline(gantt_data, x_start="Start", x_end="Finish", y="Task", color="Resource",
                           color_discrete_map={'High': '#333333', 'Medium': '#666666', 
                                              'Low': '#999999', 'Critical': '#000000'})
    fig_gantt.update_yaxes(autorange="reversed")
    fig_gantt.update_layout(height=400, paper_bgcolor='white', font=dict(color='black'),
                           title="Project Roadmap & Resource Allocation")
    st.plotly_chart(fig_gantt, use_container_width=True)
    
    # Viz 37: Compliance Scorecard
    st.subheader("37. Regulatory Compliance Scorecard")
    
    compliance_areas = ['Data Privacy', 'Financial Reporting', 'Labor Laws', 'Environmental', 'Tax Compliance']
    scores = [92, 88, 95, 78, 85]
    targets = [90, 90, 95, 85, 90]
    
    fig_compliance = go.Figure()
    
    fig_compliance.add_trace(go.Bar(
        y=compliance_areas,
        x=scores,
        name='Current Score',
        orientation='h',
        marker_color='black',
        text=scores,
        textposition='auto'
    ))
    
    fig_compliance.add_trace(go.Scatter(
        y=compliance_areas,
        x=targets,
        mode='markers',
        name='Target',
        marker=dict(color='red', size=15, symbol='line-ns')
    ))
    
    fig_compliance.update_layout(
        title="Compliance Score vs Target",
        xaxis_title="Compliance Score (%)",
        height=400,
        paper_bgcolor='white',
        font=dict(color='black')
    )
    st.plotly_chart(fig_compliance, use_container_width=True)
    
    # Viz 38: Risk Trend with Annotations
    st.subheader("38. Risk Trend Analysis with Events")
    
    risk_trend = pd.DataFrame({
        'date': pd.date_range(start='2024-01-01', periods=12, freq='M'),
        'risk_score': [65, 68, 72, 70, 75, 82, 78, 74, 71, 69, 67, 65],
        'events': ['', '', 'New Competitor', '', 'Supply Issue', 'Peak Risk', 
                  'Mitigation', '', '', '', '', 'Stable']
    })
    
    fig_risk_trend = go.Figure()
    fig_risk_trend.add_trace(go.Scatter(
        x=risk_trend['date'],
        y=risk_trend['risk_score'],
        mode='lines+markers+text',
        text=risk_trend['events'],
        textposition="top center",
        line=dict(color='black', width=3),
        marker=dict(size=10)
    ))
    
    # Add risk zones
    fig_risk_trend.add_hrect(y0=80, y1=100, fillcolor="red", opacity=0.2, line_width=0, annotation_text="Critical")
    fig_risk_trend.add_hrect(y0=60, y1=80, fillcolor="yellow", opacity=0.2, line_width=0, annotation_text="Elevated")
    fig_risk_trend.add_hrect(y0=0, y1=60, fillcolor="green", opacity=0.2, line_width=0, annotation_text="Normal")
    
    fig_risk_trend.update_layout(
        title="Enterprise Risk Score Over Time",
        height=500,
        paper_bgcolor='white',
        font=dict(color='black')
    )
    st.plotly_chart(fig_risk_trend, use_container_width=True)
    
    # Viz 39: Monte Carlo Value at Risk (Historical)
    st.subheader("39. Historical VaR Backtesting")
    
    returns = df['profit'].pct_change().dropna()
    var_levels = [0.95, 0.99, 0.999]
    var_colors = ['gray', 'black', 'red']
    
    fig_var_hist = go.Figure()
    fig_var_hist.add_trace(go.Histogram(x=returns, nbinsx=50, name='Returns',
                                       marker_color='lightgray', opacity=0.7))
    
    for var_level, color in zip(var_levels, var_colors):
        var_value = np.percentile(returns, (1-var_level)*100)
        fig_var_hist.add_vline(x=var_value, line_dash="dash", line_color=color,
                              annotation_text=f"VaR {var_level*100:.0f}%")
    
    fig_var_hist.update_layout(
        title="Profit/Loss Distribution with VaR Thresholds",
        xaxis_title="Daily Return",
        yaxis_title="Frequency",
        height=500,
        paper_bgcolor='white',
        font=dict(color='black')
    )
    st.plotly_chart(fig_var_hist, use_container_width=True)
    
    # Viz 40: Stress Testing Scenarios
    st.subheader("40. Stress Testing Results")
    
    scenarios = ['Base Case', 'Mild Recession', 'Severe Recession', 'Market Crash', 'Pandemic']
    metrics = ['Revenue', 'Profit', 'Cash Flow', 'Equity']
    
    stress_data = np.array([
        [100, 100, 100, 100],  # Base
        [85, 70, 75, 90],      # Mild
        [60, 30, 40, 70],      # Severe
        [40, -20, 10, 50],     # Crash
        [50, 10, 25, 60]       # Pandemic
    ])
    
    fig_stress = go.Figure()
    
    for i, metric in enumerate(metrics):
        fig_stress.add_trace(go.Scatter(
            x=scenarios,
            y=stress_data[:, i],
            mode='lines+markers',
            name=metric,
            line=dict(width=3),
            marker=dict(size=10)
        ))
    
    fig_stress.add_hline(y=0, line_dash="dash", line_color="red")
    fig_stress.update_layout(
        title="Impact of Stress Scenarios on Key Metrics (% of Base)",
        height=500,
        paper_bgcolor='white',
        font=dict(color='black')
    )
    st.plotly_chart(fig_stress, use_container_width=True)
    
    # Viz 41: Risk-Adjusted Return Metrics
    st.subheader("41. Risk-Adjusted Performance Metrics")
    
    metrics_data = {
        'Metric': ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Omega Ratio', 'Information Ratio'],
        'Value': [1.8, 2.1, 1.5, 1.9, 0.8],
        'Benchmark': [1.5, 1.8, 1.2, 1.6, 0.5],
        'Percentile': [75, 80, 70, 78, 60]
    }
    metrics_df = pd.DataFrame(metrics_data)
    
    fig_metrics = go.Figure()
    
    fig_metrics.add_trace(go.Bar(
        y=metrics_df['Metric'],
        x=metrics_df['Value'],
        orientation='h',
        name='Portfolio',
        marker_color='black',
        text=[f"{v:.2f}" for v in metrics_df['Value']],
        textposition='auto'
    ))
    
    fig_metrics.add_trace(go.Scatter(
        y=metrics_df['Metric'],
        x=metrics_df['Benchmark'],
        mode='markers',
        name='Benchmark',
        marker=dict(color='red', size=15, symbol='diamond')
    ))
    
    fig_metrics.update_layout(
        title="Risk-Adjusted Return Metrics vs Benchmark",
        height=500,
        paper_bgcolor='white',
        font=dict(color='black')
    )
    st.plotly_chart(fig_metrics, use_container_width=True)
    
    # Viz 42: Key Risk Indicators (KRIs) Dashboard
    st.subheader("42. Key Risk Indicators (KRIs) Monitor")
    
    kris = ['Customer Churn', 'Employee Turnover', 'Supplier Delay', 'IT Downtime', 'Compliance Breach']
    current = [5.2, 12.5, 3.1, 0.5, 0]
    limits = [8.0, 15.0, 5.0, 2.0, 1]
    
    fig_kri = go.Figure()
    
    for i, (kri, curr, lim) in enumerate(zip(kris, current, limits)):
        color = 'green' if curr < lim * 0.7 else 'orange' if curr < lim else 'red'
        
        fig_kri.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=curr,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': kri},
            delta={'reference': lim, 'relative': False},
            gauge={
                'axis': {'range': [None, lim * 1.5]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, lim * 0.7], 'color': 'lightgreen'},
                    {'range': [lim * 0.7, lim], 'color': 'yellow'},
                    {'range': [lim, lim * 1.5], 'color': 'salmon'}
                ],
                'threshold': {
                    'line': {'color': 'red', 'width': 4},
                    'thickness': 0.75,
                    'value': lim
                }
            }
        ))
    
    fig_kri.update_layout(
        grid={'rows': 1, 'columns': 5, 'pattern': "independent"},
        height=300,
        paper_bgcolor='white'
    )
    st.plotly_chart(fig_kri, use_container_width=True)
    
    # Viz 43: Risk Correlation Network
    st.subheader("43. Risk Factor Correlation Network")
    
    risk_factors = ['Market Vol', 'Credit Spread', 'Liquidity', 'Operational', 'Reputation']
    corr_matrix_risk = np.random.rand(5, 5)
    corr_matrix_risk = (corr_matrix_risk + corr_matrix_risk.T) / 2
    np.fill_diagonal(corr_matrix_risk, 1)
    
    fig_risk_net = go.Figure()
    
    # Create network layout
    pos = {}
    for i, factor in enumerate(risk_factors):
        angle = 2 * np.pi * i / len(risk_factors)
        pos[factor] = (np.cos(angle), np.sin(angle))
    
    # Draw edges
    for i, f1 in enumerate(risk_factors):
        for j, f2 in enumerate(risk_factors):
            if i < j and corr_matrix_risk[i, j] > 0.5:
                x0, y0 = pos[f1]
                x1, y1 = pos[f2]
                fig_risk_net.add_trace(go.Scatter(
                    x=[x0, x1, None], y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=corr_matrix_risk[i,j]*5, color='gray'),
                    hoverinfo='none',
                    showlegend=False
                ))
    
    # Draw nodes
    for factor, (x, y) in pos.items():
        fig_risk_net.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(size=30, color='black'),
            text=[factor],
            textposition="top center",
            name=factor
        ))
    
    fig_risk_net.update_layout(
        title="Interconnected Risk Factors",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500,
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    st.plotly_chart(fig_risk_net, use_container_width=True)
    
    # Viz 44: Loss Distribution (Actuarial)
    st.subheader("44. Potential Loss Distribution")
    
    loss_data = np.random.lognormal(mean=10, sigma=1.5, size=10000)
    
    fig_loss = ff.create_distplot(
        [loss_data],
        ['Potential Losses'],
        bin_size=5000,
        colors=['black'],
        curve_type='kde'
    )
    
    # Add VaR lines
    var_95_loss = np.percentile(loss_data, 95)
    var_99_loss = np.percentile(loss_data, 99)
    
    fig_loss.add_vline(x=var_95_loss, line_dash="dash", line_color="orange",
                      annotation_text=f"95% VaR: ${var_95_loss:,.0f}")
    fig_loss.add_vline(x=var_99_loss, line_dash="dash", line_color="red",
                      annotation_text=f"99% VaR: ${var_99_loss:,.0f}")
    
    fig_loss.update_layout(
        title="Actuarial Loss Distribution",
        height=500,
        paper_bgcolor='white',
        font=dict(color='black')
    )
    st.plotly_chart(fig_loss, use_container_width=True)
    
    # Viz 45: Risk Appetite Statement
    st.subheader("45. Risk Appetite Framework")
    
    appetite_data = {
        'Risk Category': ['Strategic', 'Operational', 'Financial', 'Compliance', 'Reputational'],
        'Min': [10, 20, 15, 25, 30],
        'Target': [30, 40, 35, 50, 45],
        'Max': [50, 60, 55, 75, 65],
        'Current': [35, 45, 40, 55, 50]
    }
    appetite_df = pd.DataFrame(appetite_data)
    
    fig_appetite = go.Figure()
    
    for i, row in appetite_df.iterrows():
        fig_appetite.add_trace(go.Scatter(
            x=[row['Risk Category']] * 2,
            y=[row['Min'], row['Max']],
            mode='lines',
            line=dict(color='gray', width=10),
            name='Risk Range',
            showlegend=i==0
        ))
        
        fig_appetite.add_trace(go.Scatter(
            x=[row['Risk Category']],
            y=[row['Target']],
            mode='markers',
            marker=dict(color='green', size=15, symbol='line-ew'),
            name='Target',
            showlegend=i==0
        ))
        
        fig_appetite.add_trace(go.Scatter(
            x=[row['Risk Category']],
            y=[row['Current']],
            mode='markers',
            marker=dict(color='black', size=20),
            name='Current',
            showlegend=i==0
        ))
    
    fig_appetite.update_layout(
        title="Risk Appetite: Current vs Framework",
        yaxis_title="Risk Level",
        height=500,
        paper_bgcolor='white',
        font=dict(color='black')
    )
    st.plotly_chart(fig_appetite, use_container_width=True)
    
    # Viz 46: Business Continuity Planning
    st.subheader("46. Business Continuity Impact Timeline")
    
    bcp_data = pd.DataFrame({
        'Hour': list(range(0, 73, 4)),
        'Financial Impact': [0, 5, 15, 35, 60, 90, 120, 150, 180, 200, 220, 235, 245, 250, 250, 250, 250, 250, 250],
        'Reputation Damage': [0, 2, 8, 20, 40, 65, 85, 100, 110, 115, 118, 120, 120, 120, 120, 120, 120, 120, 120],
        'Recovery Cost': [0, 10, 25, 45, 70, 100, 140, 190, 250, 320, 400, 490, 590, 700, 820, 950, 1090, 1240, 1400]
    })
    
    fig_bcp = go.Figure()
    fig_bcp.add_trace(go.Scatter(x=bcp_data['Hour'], y=bcp_data['Financial Impact'],
                                mode='lines', name='Financial Impact', stackgroup='one',
                                line=dict(color='black')))
    fig_bcp.add_trace(go.Scatter(x=bcp_data['Hour'], y=bcp_data['Reputation Damage'],
                                mode='lines', name='Reputation', stackgroup='one',
                                line=dict(color='gray')))
    fig_bcp.add_trace(go.Scatter(x=bcp_data['Hour'], y=bcp_data['Recovery Cost'],
                                mode='lines', name='Recovery Cost', stackgroup='one',
                                line=dict(color='darkgray')))
    
    fig_bcp.add_vline(x=24, line_dash="dash", line_color="red",
                     annotation_text="Recovery Starts")
    
    fig_bcp.update_layout(
        title="Cumulative Impact of 72-Hour Outage",
        xaxis_title="Hours Since Incident",
        yaxis_title="Cumulative Impact ($K)",
        height=500,
        paper_bgcolor='white',
        font=dict(color='black')
    )
    st.plotly_chart(fig_bcp, use_container_width=True)
    
    # Viz 47: Regulatory Change Impact
    st.subheader("47. Regulatory Change Impact Assessment")
    
    regulations = ['GDPR Update', 'SOX Amendment', 'Basel IV', 'Climate Disclosure', 'AI Governance']
    impact = [85, 60, 75, 45, 90]
    readiness = [70, 85, 50, 30, 40]
    timeline = [6, 12, 18, 24, 12]  # months
    
    fig_reg = go.Figure()
    
    fig_reg.add_trace(go.Scatter(
        x=impact,
        y=readiness,
        mode='markers+text',
        text=regulations,
        textposition="top center",
        marker=dict(size=[t*2 for t in timeline], color='black', opacity=0.6),
        name='Regulations'
    ))
    
    fig_reg.add_hline(y=70, line_dash="dash", line_color="orange", annotation_text="Readiness Threshold")
    fig_reg.add_vline(x=70, line_dash="dash", line_color="red", annotation_text="High Impact")
    
    fig_reg.update_layout(
        title="Regulatory Impact vs Readiness (Size=Timeline)",
        xaxis_title="Impact Score",
        yaxis_title="Readiness Score",
        height=500,
        paper_bgcolor='white',
        font=dict(color='black')
    )
    st.plotly_chart(fig_reg, use_container_width=True)

# ==================== TAB 8: PREDICTIVE INTELLIGENCE (NEW - 15 viz) ====================
with tab8:
    st.header("🔮 Predictive Intelligence & AI Analytics")
    
    # Viz 48: Feature Importance with SHAP-style
    st.subheader("48. SHAP-style Feature Impact Analysis")
    
    # Simulate SHAP values
    shap_features = feature_cols[:6]
    shap_values = np.random.randn(len(X_test), len(shap_features))
    
    fig_shap = go.Figure()
    
    for i, feat in enumerate(shap_features):
        y_pos = [i] * len(shap_values)
        colors = ['red' if v < 0 else 'green' for v in shap_values[:, i]]
        
        fig_shap.add_trace(go.Scatter(
            x=shap_values[:, i],
            y=y_pos,
            mode='markers',
            marker=dict(color=colors, size=5, opacity=0.5),
            name=feat,
            showlegend=False
        ))
    
    fig_shap.add_vline(x=0, line_dash="dash", line_color="black")
    fig_shap.update_layout(
        title="Feature Impact on Prediction (Red=Negative, Green=Positive)",
        yaxis=dict(tickvals=list(range(len(shap_features))), ticktext=shap_features),
        xaxis_title="SHAP Value (Impact on Prediction)",
        height=500,
        paper_bgcolor='white',
        font=dict(color='black')
    )
    st.plotly_chart(fig_shap, use_container_width=True)
    
    # Viz 49: Learning Curves
    st.subheader("49. Model Learning Curves")
    
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_scores = [0.65, 0.72, 0.78, 0.82, 0.85, 0.87, 0.88, 0.89, 0.89, 0.90]
    val_scores = [0.60, 0.68, 0.75, 0.79, 0.81, 0.82, 0.83, 0.83, 0.84, 0.84]
    
    fig_learning = go.Figure()
    fig_learning.add_trace(go.Scatter(x=train_sizes*100, y=train_scores,
                                     mode='lines+markers', name='Training Score',
                                     line=dict(color='black')))
    fig_learning.add_trace(go.Scatter(x=train_sizes*100, y=val_scores,
                                     mode='lines+markers', name='Validation Score',
                                     line=dict(color='gray', dash='dash')))
    
    fig_learning.update_layout(
        title="Model Learning Curves: Bias-Variance Analysis",
        xaxis_title="Training Set Size (%)",
        yaxis_title="R² Score",
        height=500,
        paper_bgcolor='white',
        font=dict(color='black')
    )
    st.plotly_chart(fig_learning, use_container_width=True)
    
    # Viz 50: Prediction Confidence Intervals
    st.subheader("50. Prediction Intervals with Uncertainty")
    
    pred_x = np.linspace(X_test[feature_cols[0]].min(), X_test[feature_cols[0]].max(), 100)
    pred_y = model.predict(X_test)[:100]
    pred_std = np.std(pred_y) * 0.3
    
    upper = pred_y + 1.96 * pred_std
    lower = pred_y - 1.96 * pred_std
    
    fig_pred_int = go.Figure()
    fig_pred_int.add_trace(go.Scatter(x=pred_x, y=upper, mode='lines',
                                     line=dict(color='gray', dash='dash'),
                                     name='Upper 95% CI', showlegend=False))
    fig_pred_int.add_trace(go.Scatter(x=pred_x, y=lower, mode='lines',
                                     line=dict(color='gray', dash='dash'),
                                     name='Lower 95% CI', fill='tonexty',
                                     fillcolor='rgba(128,128,128,0.2)'))
    fig_pred_int.add_trace(go.Scatter(x=pred_x, y=pred_y, mode='lines',
                                     line=dict(color='black', width=3),
                                     name='Prediction'))
    
    # Add actual points
    fig_pred_int.add_trace(go.Scatter(x=X_test[feature_cols[0]], y=y_test,
                                     mode='markers', marker=dict(color='red', size=5),
                                     name='Actual'))
    
    fig_pred_int.update_layout(
        title="Prediction with 95% Confidence Intervals",
        height=500,
        paper_bgcolor='white',
        font=dict(color='black')
    )
    st.plotly_chart(fig_pred_int, use_container_width=True)
    
    # Viz 51: Clustering Analysis
    st.subheader("51. Customer Segmentation (K-Means Clustering)")
    
    # Prepare data for clustering
    cluster_data = df[['marketing_spend', 'profit', 'total_users', 'conversion_rate']].sample(500)
    kmeans = KMeans(n_clusters=4, random_state=42)
    cluster_data['cluster'] = kmeans.fit_predict(cluster_data)
    
    fig_cluster = px.scatter_3d(
        cluster_data,
        x='marketing_spend',
        y='profit',
        z='total_users',
        color='cluster',
        size='conversion_rate',
        color_continuous_scale='Greys',
        title="3D Customer Segmentation"
    )
    fig_cluster.update_layout(height=600, paper_bgcolor='white')
    st.plotly_chart(fig_cluster, use_container_width=True)
    
    # Viz 52: PCA Dimensionality Reduction
    st.subheader("52. PCA Dimensionality Reduction")
    
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(X_test)
    pca_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2', 'PC3'])
    pca_df['target'] = y_test.values
    
    fig_pca = px.scatter_3d(
        pca_df,
        x='PC1',
        y='PC2',
        z='PC3',
        color='target',
        color_continuous_scale='Greys',
        title=f"PCA: {pca.explained_variance_ratio_.sum()*100:.1f}% Variance Explained"
    )
    fig_pca.update_layout(height=600, paper_bgcolor='white')
    st.plotly_chart(fig_pca, use_container_width=True)
    
    # Viz 53: t-SNE Visualization
    st.subheader("53. t-SNE Non-linear Dimensionality Reduction")
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(X_test[:300])  # Sample for speed
    tsne_df = pd.DataFrame(tsne_result, columns=['Dim1', 'Dim2'])
    tsne_df['target'] = y_test.values[:300]
    
    fig_tsne = px.scatter(
        tsne_df,
        x='Dim1',
        y='Dim2',
        color='target',
        color_continuous_scale='Greys',
        title="t-SNE: Non-linear Structure in Data"
    )
    fig_tsne.update_layout(height=600, paper_bgcolor='white', font=dict(color='black'))
    st.plotly_chart(fig_tsne, use_container_width=True)
    
    # Viz 54: Anomaly Detection
    st.subheader("54. Anomaly Detection Results")
    
    # Calculate z-scores
    z_scores = np.abs(stats.zscore(df[['revenue', 'profit', 'marketing_spend']]))
    anomalies = (z_scores > 3).any(axis=1)
    
    anomaly_df = df[['date', 'revenue', 'profit']].copy()
    anomaly_df['anomaly'] = anomalies
    anomaly_df['size'] = np.where(anomalies, 20, 5)
    
    fig_anomaly = go.Figure()
    
    fig_anomaly.add_trace(go.Scatter(
        x=anomaly_df[~anomaly_df['anomaly']]['date'],
        y=anomaly_df[~anomaly_df['anomaly']]['profit'],
        mode='markers',
        name='Normal',
        marker=dict(color='gray', size=5)
    ))
    
    fig_anomaly.add_trace(go.Scatter(
        x=anomaly_df[anomaly_df['anomaly']]['date'],
        y=anomaly_df[anomaly_df['anomaly']]['profit'],
        mode='markers',
        name='Anomaly',
        marker=dict(color='red', size=15, symbol='x')
    ))
    
    fig_anomaly.update_layout(
        title="Anomaly Detection: Outliers in Profit Data",
        height=500,
        paper_bgcolor='white',
        font=dict(color='black')
    )
    st.plotly_chart(fig_anomaly, use_container_width=True)
    
    # Viz 55: Time Series Forecasting (ARIMA-style)
    st.subheader("55. Time Series Forecast with Seasonality")
    
    # Generate forecast
    historical = df.groupby('date')['revenue'].sum().tail(90)
    forecast_dates = pd.date_range(start=historical.index[-1], periods=30, freq='D')
    
    # Simple exponential smoothing simulation
    alpha = 0.3
    forecast_values = []
    last_value = historical.iloc[-1]
    trend = (historical.iloc[-1] - historical.iloc[-30]) / 30
    
    for i in range(30):
        forecast_values.append(last_value + trend * (i+1) + np.random.normal(0, historical.std()*0.1))
    
    fig_forecast_ts = go.Figure()
    fig_forecast_ts.add_trace(go.Scatter(x=historical.index, y=historical.values,
                                        mode='lines', name='Historical',
                                        line=dict(color='black')))
    fig_forecast_ts.add_trace(go.Scatter(x=forecast_dates, y=forecast_values,
                                        mode='lines', name='Forecast',
                                        line=dict(color='gray', dash='dash')))
    
    # Add prediction intervals
    upper = [v + 2*historical.std() for v in forecast_values]
    lower = [v - 2*historical.std() for v in forecast_values]
    
    fig_forecast_ts.add_trace(go.Scatter(x=forecast_dates, y=upper,
                                        mode='lines', line=dict(color='lightgray'),
                                        showlegend=False))
    fig_forecast_ts.add_trace(go.Scatter(x=forecast_dates, y=lower,
                                        mode='lines', line=dict(color='lightgray'),
                                        fill='tonexty', fillcolor='rgba(128,128,128,0.2)',
                                        name='95% CI'))
    
    fig_forecast_ts.update_layout(
        title="30-Day Revenue Forecast with Confidence Intervals",
        height=500,
        paper_bgcolor='white',
        font=dict(color='black')
    )
    st.plotly_chart(fig_forecast_ts, use_container_width=True)
    
    # Viz 56: Feature Interaction Effects
    st.subheader("56. Feature Interaction Heatmap")
    
    # Create interaction matrix
    interaction_features = ['marketing_spend', 'effective_price', 'total_users']
    interaction_matrix = np.zeros((len(interaction_features), len(interaction_features)))
    
    for i, feat1 in enumerate(interaction_features):
        for j, feat2 in enumerate(interaction_features):
            if i != j:
                # Calculate interaction strength (correlation of products)
                interaction_strength = np.corrcoef(df[feat1] * df[feat2], df['profit'])[0,1]
                interaction_matrix[i, j] = abs(interaction_strength)
    
    fig_interaction = px.imshow(
        interaction_matrix,
        x=interaction_features,
        y=interaction_features,
        text_auto=True,
        color_continuous_scale='Greys',
        title="Feature Interaction Strength (Correlation with Target)"
    )
    fig_interaction.update_layout(height=500, paper_bgcolor='white')
    st.plotly_chart(fig_interaction, use_container_width=True)
    
    # Viz 57: Model Comparison
    st.subheader("57. Multi-Model Performance Comparison")
    
    models = ['Linear', 'Ridge', 'Lasso', 'Random Forest', 'Gradient Boosting', 'Neural Net']
    metrics = ['R²', 'MAE', 'RMSE', 'Training Time']
    
    model_scores = np.array([
        [0.82, 12500, 18000, 0.5],
        [0.83, 12200, 17500, 0.6],
        [0.81, 12800, 18200, 0.4],
        [0.91, 8500, 12000, 12.5],
        [0.93, 7200, 10500, 8.2],
        [0.89, 9100, 13500, 45.0]
    ])
    
    # Normalize for radar chart
    model_scores_norm = model_scores / model_scores.max(axis=0)
    
    fig_model_comp = go.Figure()
    
    for i, model in enumerate(models):
        fig_model_comp.add_trace(go.Scatterpolar(
            r=list(model_scores_norm[i]) + [model_scores_norm[i][0]],
            theta=metrics + [metrics[0]],
            fill='toself',
            name=model,
            opacity=0.3
        ))
    
    fig_model_comp.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Model Performance Comparison (Normalized)",
        height=600,
        paper_bgcolor='white',
        font=dict(color='black')
    )
    st.plotly_chart(fig_model_comp, use_container_width=True)
    
    # Viz 58: Feature Drift Over Time
    st.subheader("58. Feature Drift Monitoring")
    
    drift_data = pd.DataFrame({
        'Month': pd.date_range(start='2024-01-01', periods=12, freq='M'),
        'Marketing_Spend_Mean': [50000 + i*1000 + np.random.normal(0, 2000) for i in range(12)],
        'Price_Mean': [75 + np.sin(i)*5 + np.random.normal(0, 1) for i in range(12)],
        'Conversion_Mean': [0.03 + np.random.normal(0, 0.002) for i in range(12)]
    })
    
    fig_drift = make_subplots(rows=3, cols=1, shared_xaxes=True,
                             subplot_titles=('Marketing Spend Drift', 'Price Drift', 'Conversion Drift'))
    
    fig_drift.add_trace(go.Scatter(x=drift_data['Month'], y=drift_data['Marketing_Spend_Mean'],
                                  mode='lines+markers', name='Marketing'), row=1, col=1)
    fig_drift.add_trace(go.Scatter(x=drift_data['Month'], y=drift_data['Price_Mean'],
                                  mode='lines+markers', name='Price'), row=2, col=1)
    fig_drift.add_trace(go.Scatter(x=drift_data['Month'], y=drift_data['Conversion_Mean'],
                                  mode='lines+markers', name='Conversion'), row=3, col=1)
    
    fig_drift.update_layout(height=700, paper_bgcolor='white', showlegend=False,
                           title_text="Feature Drift Detection Over Time")
    st.plotly_chart(fig_drift, use_container_width=True)
    
    # Viz 59: Causal Impact Analysis
    st.subheader("59. Causal Impact of Marketing Campaign")
    
    # Simulate pre/post intervention
    pre_period = pd.date_range(start='2024-01-01', periods=60, freq='D')
    post_period = pd.date_range(start='2024-03-01', periods=30, freq='D')
    
    pre_values = [100 + np.random.normal(0, 10) for _ in range(60)]
    post_values = [130 + np.random.normal(0, 10) for _ in range(30)]  # Lift from campaign
    
    causal_df = pd.DataFrame({
        'date': list(pre_period) + list(post_period),
        'value': pre_values + post_values,
        'period': ['Pre']*60 + ['Post']*30
    })
    
    fig_causal = go.Figure()
    fig_causal.add_trace(go.Scatter(x=pre_period, y=pre_values,
                                   mode='lines', name='Pre-Intervention',
                                   line=dict(color='gray')))
    fig_causal.add_trace(go.Scatter(x=post_period, y=post_values,
                                   mode='lines', name='Post-Intervention',
                                   line=dict(color='black')))
    fig_causal.add_vline(x=post_period[0], line_dash="dash", line_color="red",
                        annotation_text="Campaign Launch")
    
    fig_causal.update_layout(
        title="Causal Impact: Marketing Campaign Effectiveness",
        height=500,
        paper_bgcolor='white',
        font=dict(color='black')
    )
    st.plotly_chart(fig_causal, use_container_width=True)
    
    # Viz 60: Automated Insights
    st.subheader("60. Automated Insight Generation")
    
    insights = [
        {"metric": "Revenue", "change": "+15%", "trend": "up", "significance": "high"},
        {"metric": "CAC", "change": "-8%", "trend": "down", "significance": "medium"},
        {"metric": "Churn", "change": "+2%", "trend": "up", "significance": "high"},
        {"metric": "LTV", "change": "+12%", "trend": "up", "significance": "high"},
        {"metric": "ROI", "change": "+5%", "trend": "up", "significance": "low"}
    ]
    
    fig_insights = go.Figure()
    
    for i, insight in enumerate(insights):
        color = 'green' if insight['trend'] == 'down' and insight['metric'] in ['CAC', 'Churn'] else \
                'green' if insight['trend'] == 'up' and insight['metric'] not in ['CAC', 'Churn'] else 'red'
        
        fig_insights.add_trace(go.Indicator(
            mode="number+delta",
            value=float(insight['change'].replace('%', '').replace('+', '')),
            number={'prefix': insight['change'][0] if insight['change'][0] in ['+', '-'] else '',
                   'suffix': '%'},
            delta={'reference': 0},
            title={'text': insight['metric']},
            domain={'row': i//3, 'column': i%3}
        ))
    
    fig_insights.update_layout(
        grid={'rows': 2, 'columns': 3, 'pattern': "independent"},
        height=400,
        paper_bgcolor='white'
    )
    st.plotly_chart(fig_insights, use_container_width=True)
    
    # Viz 61: Recommendation Engine Output
    st.subheader("61. AI-Powered Recommendations")
    
    recommendations = pd.DataFrame({
        'Action': ['Increase Social Spend', 'Reduce Discounts', 'Focus on Retention', 
                  'Expand to New Market', 'Optimize Pricing'],
        'Impact': [85000, 120000, 95000, 150000, 110000],
        'Confidence': [0.85, 0.92, 0.78, 0.65, 0.88],
        'Effort': ['Low', 'Medium', 'Low', 'High', 'Medium']
    })
    
    fig_rec = px.scatter(
        recommendations,
        x='Impact',
        y='Confidence',
        size='Impact',
        color='Effort',
        text='Action',
        color_discrete_map={'Low': 'black', 'Medium': 'gray', 'High': 'lightgray'}
    )
    
    fig_rec.update_traces(textposition='top center')
    fig_rec.update_layout(
        title="Recommendation Priority Matrix (Size=Impact)",
        height=500,
        paper_bgcolor='white',
        font=dict(color='black')
    )
    st.plotly_chart(fig_rec, use_container_width=True)
    
    # Viz 62: Scenario Simulation
    st.subheader("62. Interactive Scenario Simulator")
    
    scenarios = ['Conservative', 'Base Case', 'Optimistic', 'Aggressive']
    quarters = ['Q1', 'Q2', 'Q3', 'Q4']
    
    scenario_values = {
        'Conservative': [100, 105, 108, 110],
        'Base Case': [100, 112, 125, 140],
        'Optimistic': [100, 120, 145, 180],
        'Aggressive': [100, 130, 170, 220]
    }
    
    fig_scenario = go.Figure()
    
    for scenario, values in scenario_values.items():
        dash = 'dash' if scenario == 'Base Case' else 'solid'
        width = 4 if scenario == 'Base Case' else 2
        
        fig_scenario.add_trace(go.Scatter(
            x=quarters,
            y=values,
            mode='lines+markers',
            name=scenario,
            line=dict(dash=dash, width=width)
        ))
    
    fig_scenario.update_layout(
        title="Full-Year Scenario Planning",
        yaxis_title="Revenue Index (Q1=100)",
        height=500,
        paper_bgcolor='white',
        font=dict(color='black')
    )
    st.plotly_chart(fig_scenario, use_container_width=True)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div class="footer">
    <h3>🎲 What-If Business Simulation Tool</h3>
    <p>Advanced Monte Carlo Simulation & Decision Analytics Platform</p>
    <p style="font-size: 1.2rem; margin-top: 1rem;">
        <b>Made with ❤️ by <a href="https://sourishdeyportfolio.vercel.app/" target="_blank" style="color: #f39c12; text-decoration: none;">Sourish Dey</a></b>
    </p>
    <p style="opacity: 0.8; margin-top: 0.5rem;">
        © 2024 | Built with Streamlit, Plotly & Scikit-Learn | 70+ Visualizations
    </p>
</div>
""", unsafe_allow_html=True)
