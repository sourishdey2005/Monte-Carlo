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

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        transition: transform 0.3s;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .scenario-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.3);
    }
    .highlight {
        background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #e17055;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .insight-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        border: 2px solid #74b9ff;
    }
    .footer {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(90deg, #2c3e50, #4ca1af);
        color: white;
        border-radius: 1rem;
        margin-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 0.5rem 0.5rem 0 0;
        gap: 1rem;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🎲 What-If Business Simulation Tool</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">
    Advanced Monte Carlo Simulation & Sensitivity Analysis Platform<br>
    <span style="color: #1f77b4; font-weight: bold;">40+ Interactive Visualizations</span> | 
    <span style="color: #ff7f0e; font-weight: bold;">Real-time Decision Support</span>
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
        'organic_users': organic_users,
        'paid_users': paid_users,
        'referral_users': referral_users,
        'total_users': total_users,
        'impressions': impressions,
        'clicks': clicks,
        'add_to_cart': add_to_cart,
        'checkout': checkout,
        'orders': orders,
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
        'roi': (profit / marketing_spend) * 100
    })
    
    return df

# Load enhanced data
df = generate_comprehensive_data(2000)

# Sidebar controls with enhanced UI
st.sidebar.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 1rem; color: white; margin-bottom: 1rem;">
    <h2 style="color: white; margin: 0;">⚙️ Control Center</h2>
    <p style="opacity: 0.9; margin: 0.5rem 0 0 0;">Configure your simulation parameters</p>
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
    ['profit', 'revenue', 'roi', 'profit_margin', 'ltv_cac_ratio', 'orders'],
    index=0
)

# Feature engineering
feature_cols = [
    'marketing_spend', 'effective_price', 'total_users', 'conversion_rate',
    'digital_spend', 'social_spend', 'discount_depth', 'organic_users'
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

# Model performance card
st.sidebar.markdown(f"""
<div style="background: white; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
    <h4 style="margin: 0 0 0.5rem 0; color: #2c3e50;">Model Performance</h4>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; font-size: 0.9rem;">
        <div><span style="color: #7f8c8d;">R² Score:</span> <strong>{r2:.3f}</strong></div>
        <div><span style="color: #7f8c8d;">CV Score:</span> <strong>{cv_scores.mean():.3f}</strong></div>
        <div><span style="color: #7f8c8d;">MAE:</span> <strong>${mae:,.0f}</strong></div>
        <div><span style="color: #7f8c8d;">RMSE:</span> <strong>${rmse:,.0f}</strong></div>
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
                 color='Importance', color_continuous_scale='Viridis')
fig_imp.update_layout(height=300, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
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

# Main tabs - 6 comprehensive tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Executive Dashboard", 
    "🎲 Monte Carlo Lab",
    "📊 Sensitivity Studio", 
    "💡 Strategy Optimizer",
    "🔬 Advanced Analytics",
    "🌐 Interactive Explorer"
])

# ==================== TAB 1: EXECUTIVE DASHBOARD ====================
with tab1:
    st.header("📈 Executive Performance Dashboard")
    
    # KPI Cards with animations
    kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
    
    with kpi_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.9rem; opacity: 0.9;">Total Revenue</div>
            <div style="font-size: 1.8rem; font-weight: bold;">${df['revenue'].sum()/1e6:.1f}M</div>
            <div style="font-size: 0.8rem; opacity: 0.8;">↑ {df['revenue'].pct_change().mean()*100:.1f}% avg</div>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi_col2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <div style="font-size: 0.9rem; opacity: 0.9;">Net Profit</div>
            <div style="font-size: 1.8rem; font-weight: bold;">${df['profit'].sum()/1e6:.1f}M</div>
            <div style="font-size: 0.8rem; opacity: 0.8;">Margin: {df['profit_margin'].mean():.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi_col3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <div style="font-size: 0.9rem; opacity: 0.9;">ROI</div>
            <div style="font-size: 1.8rem; font-weight: bold;">{df['roi'].mean():.0f}%</div>
            <div style="font-size: 0.8rem; opacity: 0.8;">σ = {df['roi'].std():.0f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi_col4:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
            <div style="font-size: 0.9rem; opacity: 0.9;">LTV:CAC</div>
            <div style="font-size: 1.8rem; font-weight: bold;">{df['ltv_cac_ratio'].mean():.1f}x</div>
            <div style="font-size: 0.8rem; opacity: 0.8;">Target: >3.0x</div>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi_col5:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">
            <div style="font-size: 0.9rem; opacity: 0.9;">Conversion</div>
            <div style="font-size: 1.8rem; font-weight: bold;">{df['conversion_rate'].mean():.2%}</div>
            <div style="font-size: 0.8rem; opacity: 0.8;">Industry avg: 2.5%</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Visualization 1: Time Series with Trend
    st.subheader("1. Performance Trends with Forecast")
    
    daily_data = df.groupby('date').agg({
        'revenue': 'sum',
        'profit': 'sum',
        'marketing_spend': 'sum',
        'orders': 'sum'
    }).reset_index()
    
    # Add 30-day moving average
    daily_data['revenue_ma'] = daily_data['revenue'].rolling(window=30).mean()
    daily_data['profit_ma'] = daily_data['profit'].rolling(window=30).mean()
    
    fig_trends = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Revenue Trend & MA', 'Profit Trend & MA', 
                       'Marketing Efficiency', 'Order Volume'),
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )
    
    # Revenue with moving average
    fig_trends.add_trace(
        go.Scatter(x=daily_data['date'], y=daily_data['revenue'], 
                  mode='lines', name='Daily Revenue', line=dict(color='lightblue', width=1),
                  opacity=0.5),
        row=1, col=1
    )
    fig_trends.add_trace(
        go.Scatter(x=daily_data['date'], y=daily_data['revenue_ma'], 
                  mode='lines', name='30-day MA', line=dict(color='#1f77b4', width=3)),
        row=1, col=1
    )
    
    # Profit with moving average
    fig_trends.add_trace(
        go.Scatter(x=daily_data['date'], y=daily_data['profit'], 
                  mode='lines', name='Daily Profit', line=dict(color='lightcoral', width=1),
                  opacity=0.5),
        row=1, col=2
    )
    fig_trends.add_trace(
        go.Scatter(x=daily_data['date'], y=daily_data['profit_ma'], 
                  mode='lines', name='30-day MA', line=dict(color='#d62728', width=3)),
        row=1, col=2
    )
    
    # Marketing efficiency (revenue/marketing)
    efficiency = daily_data['revenue'] / daily_data['marketing_spend']
    fig_trends.add_trace(
        go.Bar(x=daily_data['date'], y=efficiency, 
              name='Efficiency Ratio', marker_color='#2ca02c'),
        row=2, col=1
    )
    
    # Orders
    fig_trends.add_trace(
        go.Scatter(x=daily_data['date'], y=daily_data['orders'], 
                  mode='lines', name='Orders', fill='tozeroy',
                  line=dict(color='#ff7f0e')),
        row=2, col=2
    )
    
    fig_trends.update_layout(height=700, showlegend=True, 
                            legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig_trends, use_container_width=True)
    
    # Row 2: Distribution and composition
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("2. Multi-Metric Distribution Analysis")
        
        # Violin plots for key metrics
        metrics = ['revenue', 'profit', 'marketing_spend', 'roi']
        fig_violin = make_subplots(rows=2, cols=2, subplot_titles=metrics)
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for idx, metric in enumerate(metrics):
            row = idx // 2 + 1
            col = idx % 2 + 1
            
            fig_violin.add_trace(
                go.Violin(y=df[metric], box_visible=True, line_color=colors[idx],
                         meanline_visible=True, fillcolor=colors[idx], opacity=0.6,
                         name=metric),
                row=row, col=col
            )
        
        fig_violin.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig_violin, use_container_width=True)
    
    with col_right:
        st.subheader("3. Profit & Loss Waterfall")
        
        # Calculate average P&L components
        avg_revenue = df['revenue'].mean()
        avg_cogs = -df['cogs'].mean()
        avg_fulfillment = -df['fulfillment'].mean()
        avg_marketing = -df['marketing_spend'].mean()
        avg_operational = -df['operational'].mean()
        avg_profit = df['profit'].mean()
        
        waterfall_data = {
            'categories': ['Revenue', 'COGS', 'Fulfillment', 'Marketing', 'Operational', 'Profit'],
            'values': [avg_revenue, avg_cogs, avg_fulfillment, avg_marketing, avg_operational, avg_profit],
            'measure': ['absolute', 'relative', 'relative', 'relative', 'relative', 'total']
        }
        
        fig_waterfall = go.Figure(go.Waterfall(
            name="P&L",
            orientation="v",
            measure=waterfall_data['measure'],
            x=waterfall_data['categories'],
            y=waterfall_data['values'],
            textposition="outside",
            text=[f"${abs(v):,.0f}" for v in waterfall_data['values']],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "#e74c3c"}},
            increasing={"marker": {"color": "#27ae60"}},
            totals={"marker": {"color": "#3498db"}}
        ))
        
        fig_waterfall.update_layout(height=600, title="Average Daily P&L Breakdown")
        st.plotly_chart(fig_waterfall, use_container_width=True)
    
    # Row 3: Correlation and funnel
    st.subheader("4. Correlation Matrix & Conversion Funnel")
    
    corr_col, funnel_col = st.columns([3, 2])
    
    with corr_col:
        # Enhanced correlation heatmap with clustering
        corr_features = ['marketing_spend', 'effective_price', 'total_users', 
                        'conversion_rate', 'revenue', 'profit', 'roi', 'profit_margin']
        corr_matrix = df[corr_features].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title="Feature Correlation Matrix",
            height=600
        )
        fig_corr.update_traces(textfont_size=10)
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with funnel_col:
        # Conversion funnel
        funnel_data = {
            'Stage': ['Impressions', 'Clicks', 'Add to Cart', 'Checkout', 'Purchase'],
            'Users': [df['impressions'].mean(), df['clicks'].mean(), 
                     df['add_to_cart'].mean(), df['checkout'].mean(), df['orders'].mean()],
            'Conversion': [100, 
                          (df['clicks'].mean()/df['impressions'].mean())*100,
                          (df['add_to_cart'].mean()/df['clicks'].mean())*100,
                          (df['checkout'].mean()/df['add_to_cart'].mean())*100,
                          (df['orders'].mean()/df['checkout'].mean())*100]
        }
        
        fig_funnel = go.Figure(go.Funnel(
            y=funnel_data['Stage'],
            x=funnel_data['Users'],
            textinfo="value+percent initial",
            marker=dict(color=['#3498db', '#2ecc71', '#f1c40f', '#e67e22', '#e74c3c'])
        ))
        
        fig_funnel.update_layout(height=600, title="Conversion Funnel")
        st.plotly_chart(fig_funnel, use_container_width=True)
    
    # Row 4: Monthly seasonality and 3D scatter
    st.subheader("5. Seasonality Analysis & 3D Relationship")
    
    season_col, scatter3d_col = st.columns(2)
    
    with season_col:
        # Monthly box plots
        monthly_stats = df.groupby('month').agg({
            'revenue': ['mean', 'std', 'median'],
            'profit': ['mean', 'std'],
            'marketing_spend': 'mean'
        }).reset_index()
        
        fig_season = make_subplots(rows=2, cols=1, subplot_titles=('Revenue by Month', 'Profit by Month'))
        
        fig_season.add_trace(
            go.Box(x=df['month'], y=df['revenue'], name='Revenue Distribution',
                  marker_color='#3498db', boxmean=True),
            row=1, col=1
        )
        
        fig_season.add_trace(
            go.Box(x=df['month'], y=df['profit'], name='Profit Distribution',
                  marker_color='#2ecc71', boxmean=True),
            row=2, col=1
        )
        
        fig_season.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig_season, use_container_width=True)
    
    with scatter3d_col:
        # 3D scatter plot
        fig_3d = go.Figure(data=[go.Scatter3d(
            x=df['marketing_spend'],
            y=df['effective_price'],
            z=df['profit'],
            mode='markers',
            marker=dict(
                size=df['total_users']/500,
                color=df['roi'],
                colorscale='Viridis',
                opacity=0.6,
                showscale=True,
                colorbar=dict(title='ROI %')
            ),
            text=df['date'].dt.strftime('%Y-%m-%d'),
            hovertemplate='<b>Date:</b> %{text}<br>' +
                         '<b>Marketing:</b> $%{x:,.0f}<br>' +
                         '<b>Price:</b> $%{y:.2f}<br>' +
                         '<b>Profit:</b> $%{z:,.0f}<br>' +
                         '<b>ROI:</b> %{marker.color:.1f}%'
        )])
        
        fig_3d.update_layout(
            scene=dict(
                xaxis_title='Marketing Spend ($)',
                yaxis_title='Effective Price ($)',
                zaxis_title='Profit ($)'
            ),
            height=600,
            title="3D Profit Landscape"
        )
        st.plotly_chart(fig_3d, use_container_width=True)

# ==================== TAB 2: MONTE CARLO LAB ====================
with tab2:
    st.header("🎲 Monte Carlo Simulation Laboratory")
    
    # Run comprehensive Monte Carlo simulation
    np.random.seed(42)
    
    # Generate simulation data with correlations preserved
    sim_results = []
    
    # Cholesky decomposition for correlated random variables
    cov_matrix = df[['marketing_spend', 'effective_price', 'total_users', 'conversion_rate']].cov()
    L = np.linalg.cholesky(cov_matrix.values)
    
    for i in range(n_simulations):
        # Generate correlated random shocks
        uncorrelated = np.random.normal(0, 1, 4)
        correlated = L @ uncorrelated
        
        # Apply shocks to base values
        mkt_noise = correlated[0] / np.sqrt(cov_matrix.values[0,0])
        price_noise = correlated[1] / np.sqrt(cov_matrix.values[1,1])
        user_noise = correlated[2] / np.sqrt(cov_matrix.values[2,2])
        conv_noise = correlated[3] / np.sqrt(cov_matrix.values[3,3])
        
        sim_marketing = new_marketing * (1 + mkt_noise * 0.15)
        sim_price = new_price * (1 + price_noise * 0.08)
        sim_users = new_users * (1 + user_noise * 0.20)
        sim_conversion = np.clip(new_conversion * (1 + conv_noise * 0.25), 0.001, 0.5)
        
        # Calculate derived features
        sim_digital = sim_marketing * digital_ratio
        sim_social = sim_digital * 0.4
        sim_discount = np.random.beta(2, 5)
        sim_organic = sim_users * 0.6
        
        # Create feature vector
        features = np.array([
            sim_marketing, sim_price, sim_users, sim_conversion,
            sim_digital, sim_social, sim_discount, sim_organic
        ]).reshape(1, -1)
        
        # Predict with model uncertainty
        base_pred = model.predict(features)[0]
        model_noise = np.random.normal(0, rmse * 0.3)
        prediction = base_pred + model_noise
        
        sim_results.append({
            'sim_id': i,
            'marketing': sim_marketing,
            'price': sim_price,
            'users': sim_users,
            'conversion': sim_conversion,
            'prediction': prediction,
            'scenario': 'Base'
        })
    
    sim_df = pd.DataFrame(sim_results)
    
    # Scenario comparison - run alternative scenarios
    scenarios = {
        'Optimistic': {'marketing': 1.3, 'price': 1.1, 'users': 1.4, 'conversion': 1.2},
        'Pessimistic': {'marketing': 0.7, 'price': 0.9, 'users': 0.6, 'conversion': 0.8},
        'Aggressive Growth': {'marketing': 1.5, 'price': 0.95, 'users': 1.6, 'conversion': 1.1},
        'Premium Pricing': {'marketing': 0.9, 'price': 1.3, 'users': 0.8, 'conversion': 0.9}
    }
    
    scenario_results = {}
    for scen_name, multipliers in scenarios.items():
        scen_preds = []
        for _ in range(n_simulations // 4):  # Fewer sims per scenario for speed
            features = np.array([
                new_marketing * multipliers['marketing'],
                new_price * multipliers['price'],
                new_users * multipliers['users'],
                new_conversion * multipliers['conversion'],
                new_marketing * multipliers['marketing'] * digital_ratio,
                new_marketing * multipliers['marketing'] * digital_ratio * 0.4,
                np.random.beta(2, 5),
                new_users * multipliers['users'] * 0.6
            ]).reshape(1, -1)
            pred = model.predict(features)[0] + np.random.normal(0, rmse * 0.3)
            scen_preds.append(pred)
        scenario_results[scen_name] = scen_preds
    
    # Visualization 6: Distribution with multiple scenarios
    st.subheader("6. Multi-Scenario Distribution Comparison")
    
    fig_dist = go.Figure()
    
    # Base case histogram
    fig_dist.add_trace(go.Histogram(
        x=sim_df['prediction'],
        name='Base Case',
        opacity=0.6,
        nbinsx=50,
        marker_color='#3498db'
    ))
    
    # Add scenario distributions
    colors = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    for idx, (scen_name, preds) in enumerate(scenario_results.items()):
        fig_dist.add_trace(go.Histogram(
            x=preds,
            name=scen_name,
            opacity=0.5,
            nbinsx=30,
            marker_color=colors[idx]
        ))
    
    # Add vertical lines for means
    fig_dist.add_vline(x=sim_df['prediction'].mean(), line_dash="dash", 
                      line_color="#3498db", annotation_text="Base Mean")
    
    fig_dist.update_layout(
        barmode='overlay',
        title=f"Distribution Comparison Across Scenarios (n={n_simulations} total)",
        xaxis_title=f"Predicted {target_variable.replace('_', ' ').title()}",
        yaxis_title="Frequency",
        height=500
    )
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Row 2: CDF and Convergence
    col_cdf, col_conv = st.columns(2)
    
    with col_cdf:
        st.subheader("7. Cumulative Distribution Functions")
        
        fig_cdf = go.Figure()
        
        # Empirical CDF for base case
        sorted_data = np.sort(sim_df['prediction'])
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        fig_cdf.add_trace(go.Scatter(
            x=sorted_data, y=cdf,
            mode='lines', name='Base Case',
            line=dict(color='#3498db', width=3)
        ))
        
        # Add scenario CDFs
        for idx, (scen_name, preds) in enumerate(scenario_results.items()):
            sorted_scen = np.sort(preds)
            cdf_scen = np.arange(1, len(sorted_scen) + 1) / len(sorted_scen)
            fig_cdf.add_trace(go.Scatter(
                x=sorted_scen, y=cdf_scen,
                mode='lines', name=scen_name,
                line=dict(color=colors[idx], width=2, dash='dash')
            ))
        
        # Add confidence interval lines
        ci_lower = np.percentile(sim_df['prediction'], (100-confidence_level)/2)
        ci_upper = np.percentile(sim_df['prediction'], 100 - (100-confidence_level)/2)
        
        fig_cdf.add_vline(x=ci_lower, line_dash="dot", line_color="red", 
                         annotation_text=f"{(100-confidence_level)/2}%")
        fig_cdf.add_vline(x=ci_upper, line_dash="dot", line_color="red",
                         annotation_text=f"{100-(100-confidence_level)/2}%")
        
        fig_cdf.update_layout(
            title="Cumulative Probability Distribution",
            xaxis_title="Outcome Value",
            yaxis_title="Cumulative Probability",
            height=500
        )
        st.plotly_chart(fig_cdf, use_container_width=True)
    
    with col_conv:
        st.subheader("8. Monte Carlo Convergence Analysis")
        
        # Calculate running statistics
        running_mean = np.cumsum(sim_df['prediction']) / np.arange(1, len(sim_df) + 1)
        running_std = pd.Series(sim_df['prediction']).expanding().std()
        
        fig_conv = make_subplots(rows=2, cols=1, subplot_titles=('Running Mean', 'Running Std Dev'))
        
        fig_conv.add_trace(
            go.Scatter(y=running_mean, mode='lines', name='Mean',
                      line=dict(color='#2ecc71')),
            row=1, col=1
        )
        
        fig_conv.add_trace(
            go.Scatter(y=running_std, mode='lines', name='Std Dev',
                      line=dict(color='#e74c3c')),
            row=2, col=1
        )
        
        fig_conv.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig_conv, use_container_width=True)
    
    # Row 3: Violin and Box plots
    st.subheader("9. Scenario Risk Profiles")
    
    # Prepare data for violin plot
    all_scenarios = []
    for scen_name, preds in scenario_results.items():
        for p in preds:
            all_scenarios.append({'Scenario': scen_name, 'Value': p})
    
    scenario_df_plot = pd.DataFrame(all_scenarios)
    base_df = pd.DataFrame({'Scenario': ['Base Case'] * len(sim_df), 'Value': sim_df['prediction']})
    combined_df = pd.concat([base_df, scenario_df_plot])
    
    fig_violin_scen = go.Figure()
    
    for scen in combined_df['Scenario'].unique():
        subset = combined_df[combined_df['Scenario'] == scen]
        fig_violin_scen.add_trace(go.Violin(
            y=subset['Value'],
            name=scen,
            box_visible=True,
            meanline_visible=True
        ))
    
    fig_violin_scen.update_layout(
        title="Risk Distribution by Scenario",
        yaxis_title=f"{target_variable.replace('_', ' ').title()}",
        height=500
    )
    st.plotly_chart(fig_violin_scen, use_container_width=True)
    
    # Row 4: Risk metrics and percentiles
    st.subheader("10. Comprehensive Risk Metrics")
    
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    # Calculate metrics
    var_95 = np.percentile(sim_df['prediction'], 5)
    var_99 = np.percentile(sim_df['prediction'], 1)
    cvar_95 = sim_df[sim_df['prediction'] <= var_95]['prediction'].mean()
    cvar_99 = sim_df[sim_df['prediction'] <= var_99]['prediction'].mean()
    skewness = stats.skew(sim_df['prediction'])
    kurt = stats.kurtosis(sim_df['prediction'])
    
    with metrics_col1:
        st.markdown(f"""
        <div class="highlight">
            <h4>Value at Risk (VaR)</h4>
            <b>95% VaR:</b> ${var_95:,.0f}<br>
            <b>99% VaR:</b> ${var_99:,.0f}<br>
            <small>Maximum expected loss at confidence level</small>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_col2:
        st.markdown(f"""
        <div class="highlight">
            <h4>Conditional VaR (CVaR)</h4>
            <b>95% CVaR:</b> ${cvar_95:,.0f}<br>
            <b>99% CVaR:</b> ${cvar_99:,.0f}<br>
            <small>Average loss in worst-case scenarios</small>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_col3:
        st.markdown(f"""
        <div class="highlight">
            <h4>Distribution Shape</h4>
            <b>Skewness:</b> {skewness:.2f} {'(Right tail)' if skewness > 0 else '(Left tail)'}<br>
            <b>Kurtosis:</b> {kurt:.2f} {'(Heavy tails)' if kurt > 0 else '(Light tails)'}<br>
            <small>Measures of distribution asymmetry</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Percentile table
    st.subheader("11. Detailed Percentile Analysis")
    
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    perc_data = []
    
    for p in percentiles:
        val = np.percentile(sim_df['prediction'], p)
        perc_data.append({
            'Percentile': f'{p}th',
            'Value': f'${val:,.0f}',
            'Interpretation': {
                1: 'Extreme Worst Case', 5: 'Worst Case', 10: 'Very Pessimistic',
                25: 'Pessimistic', 50: 'Median (Base)', 75: 'Optimistic',
                90: 'Very Optimistic', 95: 'Best Case', 99: 'Extreme Best Case'
            }.get(p, '')
        })
    
    st.dataframe(pd.DataFrame(perc_data), use_container_width=True, hide_index=True)
    
    # Forecast intervals over time
    st.subheader("12. Forecast Confidence Intervals Over Time")
    
    # Generate time-based forecast
    future_dates = pd.date_range(start=df['date'].max(), periods=time_horizon, freq='D')
    forecast_data = []
    
    for i, date in enumerate(future_dates):
        # Wider intervals as we go further out
        uncertainty_factor = 1 + (i / time_horizon) * 0.5
        
        base_pred = sim_df['prediction'].mean()
        std_pred = sim_df['prediction'].std() * uncertainty_factor
        
        forecast_data.append({
            'date': date,
            'mean': base_pred,
            'lower_95': base_pred - 1.96 * std_pred,
            'upper_95': base_pred + 1.96 * std_pred,
            'lower_80': base_pred - 1.28 * std_pred,
            'upper_80': base_pred + 1.28 * std_pred
        })
    
    forecast_df = pd.DataFrame(forecast_data)
    
    fig_forecast = go.Figure()
    
    # Confidence bands
    fig_forecast.add_trace(go.Scatter(
        x=forecast_df['date'], y=forecast_df['upper_95'],
        fill=None, mode='lines', line_color='rgba(0,100,80,0)',
        showlegend=False
    ))
    fig_forecast.add_trace(go.Scatter(
        x=forecast_df['date'], y=forecast_df['lower_95'],
        fill='tonexty', fillcolor='rgba(0,100,80,0.2)',
        line_color='rgba(0,100,80,0)', name='95% CI'
    ))
    
    fig_forecast.add_trace(go.Scatter(
        x=forecast_df['date'], y=forecast_df['upper_80'],
        fill=None, mode='lines', line_color='rgba(0,100,80,0)',
        showlegend=False
    ))
    fig_forecast.add_trace(go.Scatter(
        x=forecast_df['date'], y=forecast_df['lower_80'],
        fill='tonexty', fillcolor='rgba(0,100,80,0.4)',
        line_color='rgba(0,100,80,0)', name='80% CI'
    ))
    
    # Mean line
    fig_forecast.add_trace(go.Scatter(
        x=forecast_df['date'], y=forecast_df['mean'],
        mode='lines', line=dict(color='rgb(0,100,80)', width=2),
        name='Forecast Mean'
    ))
    
    fig_forecast.update_layout(
        title=f"{time_horizon}-Day Forecast with Expanding Uncertainty",
        xaxis_title="Date",
        yaxis_title=f"Predicted {target_variable.replace('_', ' ').title()}",
        height=500
    )
    st.plotly_chart(fig_forecast, use_container_width=True)

# ==================== TAB 3: SENSITIVITY STUDIO ====================
with tab3:
    st.header("📊 Sensitivity Analysis Studio")
    
    # Calculate base prediction
    base_features = np.array([
        new_marketing, new_price, new_users, new_conversion,
        new_marketing * digital_ratio,
        new_marketing * digital_ratio * 0.4,
        0.15,  # average discount
        new_users * 0.6
    ]).reshape(1, -1)
    base_prediction = model.predict(base_features)[0]
    
    # Visualization 13: Enhanced Tornado Diagram
    st.subheader("13. Tornado Diagram - Sensitivity to ±30% Changes")
    
    sensitivity_range = np.linspace(-30, 30, 25)
    tornado_data = []
    detailed_sens = {}
    
    for idx, feature in enumerate(feature_cols):
        predictions = []
        for var in sensitivity_range:
            test_features = base_features.copy().flatten()
            
            if idx == 0:  # marketing
                test_features[0] = new_marketing * (1 + var/100)
                test_features[4] = test_features[0] * digital_ratio
                test_features[5] = test_features[4] * 0.4
            elif idx == 1:  # price
                test_features[1] = new_price * (1 + var/100)
            elif idx == 2:  # users
                test_features[2] = new_users * (1 + var/100)
                test_features[7] = test_features[2] * 0.6
            elif idx == 3:  # conversion
                test_features[3] = np.clip(new_conversion * (1 + var/100), 0.001, 0.5)
            elif idx == 4:  # digital spend
                test_features[4] = new_marketing * digital_ratio * (1 + var/100)
            elif idx == 5:  # social spend
                test_features[5] = new_marketing * digital_ratio * 0.4 * (1 + var/100)
            elif idx == 6:  # discount
                test_features[6] = np.clip(0.15 * (1 + var/100), 0, 0.5)
            else:  # organic users
                test_features[7] = new_users * 0.6 * (1 + var/100)
            
            pred = model.predict(test_features.reshape(1, -1))[0]
            predictions.append(pred)
        
        predictions = np.array(predictions)
        detailed_sens[feature] = {
            'variations': sensitivity_range,
            'predictions': predictions
        }
        
        min_pred = predictions.min()
        max_pred = predictions.max()
        
        tornado_data.append({
            'Variable': feature.replace('_', ' ').title(),
            'Low Impact': min_pred - base_prediction,
            'High Impact': max_pred - base_prediction,
            'Range': max_pred - min_pred,
            'Elasticity': ((max_pred - min_pred) / base_prediction) / 0.6
        })
    
    tornado_df = pd.DataFrame(tornado_data).sort_values('Range', ascending=True)
    
    # Create tornado chart
    fig_tornado = go.Figure()
    
    fig_tornado.add_trace(go.Bar(
        name='Decrease (-30%)',
        y=tornado_df['Variable'],
        x=tornado_df['Low Impact'],
        orientation='h',
        marker_color='#e74c3c',
        text=[f'${v:,.0f}' for v in tornado_df['Low Impact']],
        textposition='auto'
    ))
    
    fig_tornado.add_trace(go.Bar(
        name='Increase (+30%)',
        y=tornado_df['Variable'],
        x=tornado_df['High Impact'],
        orientation='h',
        marker_color='#27ae60',
        text=[f'${v:,.0f}' for v in tornado_df['High Impact']],
        textposition='auto'
    ))
    
    fig_tornado.add_vline(x=0, line_width=2, line_color="black")
    
    fig_tornado.update_layout(
        title="Impact of ±30% Change in Each Variable",
        xaxis_title=f"Change in {target_variable.replace('_', ' ').title()} ($)",
        barmode='overlay',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig_tornado, use_container_width=True)
    
    # Row 2: Sensitivity lines and spider chart
    col_sens, col_spider = st.columns([3, 2])
    
    with col_sens:
        st.subheader("14. One-Way Sensitivity Curves")
        
        fig_sens_curves = go.Figure()
        
        colors = px.colors.qualitative.Set1
        for idx, (feature, data) in enumerate(detailed_sens.items()):
            fig_sens_curves.add_trace(go.Scatter(
                x=data['variations'],
                y=data['predictions'],
                mode='lines',
                name=feature.replace('_', ' ').title(),
                line=dict(color=colors[idx % len(colors)], width=3)
            ))
        
        # Add base case line
        fig_sens_curves.add_hline(y=base_prediction, line_dash="dash", 
                                 line_color="black", annotation_text="Base Case")
        
        fig_sens_curves.update_layout(
            title="Response Curves for All Variables",
            xaxis_title="Change from Base (%)",
            yaxis_title=f"Predicted {target_variable.replace('_', ' ').title()}",
            height=600,
            hovermode='x unified'
        )
        st.plotly_chart(fig_sens_curves, use_container_width=True)
    
    with col_spider:
        st.subheader("15. Multi-Dimensional Radar")
        
        # Create radar chart with multiple scenarios
        scenarios_radar = {
            'Conservative': [-20, -10, -15, -5],
            'Base Case': [0, 0, 0, 0],
            'Aggressive': [20, 15, 25, 10],
            'Balanced': [10, 5, 10, 5]
        }
        
        fig_radar = go.Figure()
        
        radar_features = feature_cols[:4]  # Use first 4 for radar
        radar_labels = [f.replace('_', ' ').title() for f in radar_features]
        
        colors_radar = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
        
        for idx, (scen_name, changes) in enumerate(scenarios_radar.items()):
            values = []
            for i, feat in enumerate(radar_features):
                # Get prediction at this change level
                closest_idx = min(range(len(sensitivity_range)), 
                                key=lambda x: abs(sensitivity_range[x] - changes[i]))
                pred = detailed_sens[feat]['predictions'][closest_idx]
                # Normalize to percentage change from base
                norm_val = ((pred - base_prediction) / abs(base_prediction)) * 100
                values.append(norm_val)
            
            fig_radar.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=radar_labels + [radar_labels[0]],
                fill='toself',
                name=scen_name,
                line_color=colors_radar[idx],
                fillcolor=colors_radar[idx],
                opacity=0.3
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[-50, 50],
                    title="% Change from Base"
                )),
            showlegend=True,
            title="Scenario Comparison",
            height=600
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    
    # Row 3: Elasticity and contribution
    st.subheader("16. Elasticity Analysis & Variance Decomposition")
    
    el_col, contrib_col = st.columns(2)
    
    with el_col:
        # Elasticity curve
        elasticity_data = []
        for _, row in tornado_df.iterrows():
            var_name = row['Variable'].lower().replace(' ', '_')
            # Map back to original feature name
            orig_feat = None
            for feat in feature_cols:
                if feat.replace('_', ' ').title() == row['Variable']:
                    orig_feat = feat
                    break
            
            if orig_feat and orig_feat in detailed_sens:
                preds = detailed_sens[orig_feat]['predictions']
                # Calculate point elasticity at different points
                for i in range(5, 20, 3):  # Sample points
                    if i < len(preds) - 1:
                        pct_change_input = (sensitivity_range[i] - sensitivity_range[i-1]) / 100
                        pct_change_output = (preds[i] - preds[i-1]) / base_prediction
                        if pct_change_input != 0:
                            elasticity = pct_change_output / pct_change_input
                            elasticity_data.append({
                                'Variable': row['Variable'],
                                'Point': sensitivity_range[i],
                                'Elasticity': elasticity
                            })
        
        if elasticity_data:
            el_df = pd.DataFrame(elasticity_data)
            fig_el = px.line(el_df, x='Point', y='Elasticity', color='Variable',
                            title="Point Elasticity Curves", height=500)
            fig_el.add_hline(y=1, line_dash="dash", line_color="red", 
                           annotation_text="Unit Elastic")
            fig_el.add_hline(y=-1, line_dash="dash", line_color="red")
            st.plotly_chart(fig_el, use_container_width=True)
    
    with contrib_col:
        # Contribution to variance (using Sobol-like approximation)
        variances = []
        for feat in feature_cols:
            if feat in detailed_sens:
                var = np.var(detailed_sens[feat]['predictions'])
                variances.append({
                    'Variable': feat.replace('_', ' ').title(),
                    'Variance': var
                })
        
        var_df = pd.DataFrame(variances)
        var_df['Contribution'] = (var_df['Variance'] / var_df['Variance'].sum()) * 100
        
        fig_contrib = px.pie(var_df, values='Contribution', names='Variable',
                            title="Contribution to Output Variance",
                            color_discrete_sequence=px.colors.qualitative.Set3,
                            hole=0.4)
        fig_contrib.update_traces(textposition='inside', textinfo='percent+label')
        fig_contrib.update_layout(height=500)
        st.plotly_chart(fig_contrib, use_container_width=True)
    
    # Row 4: Break-even and two-way sensitivity
    st.subheader("17. Break-Even & Two-Way Sensitivity Analysis")
    
    # Two-way sensitivity heatmap
    two_way_col, be_col = st.columns([3, 2])
    
    with two_way_col:
        st.markdown("**Two-Way Sensitivity: Marketing Spend vs Price**")
        
        # Create grid
        mkt_range = np.linspace(new_marketing * 0.7, new_marketing * 1.3, 20)
        price_range = np.linspace(new_price * 0.8, new_price * 1.2, 20)
        
        heatmap_data = np.zeros((len(price_range), len(mkt_range)))
        
        for i, mkt in enumerate(mkt_range):
            for j, prc in enumerate(price_range):
                test_feat = base_features.copy().flatten()
                test_feat[0] = mkt
                test_feat[1] = prc
                test_feat[4] = mkt * digital_ratio
                pred = model.predict(test_feat.reshape(1, -1))[0]
                heatmap_data[j, i] = pred
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=[f'${m:,.0f}' for m in mkt_range],
            y=[f'${p:.2f}' for p in price_range],
            colorscale='RdYlGn',
            colorbar=dict(title=f'{target_variable}')
        ))
        
        # Add current point
        curr_mkt_idx = min(range(len(mkt_range)), key=lambda i: abs(mkt_range[i] - new_marketing))
        curr_price_idx = min(range(len(price_range)), key=lambda i: abs(price_range[i] - new_price))
        
        fig_heatmap.add_annotation(
            x=curr_mkt_idx, y=curr_price_idx,
            text="★ CURRENT",
            showarrow=True,
            arrowhead=2,
            ax=20, ay=-30,
            font=dict(color="white", size=12, family="Arial Black")
        )
        
        fig_heatmap.update_layout(
            xaxis_title="Marketing Spend ($)",
            yaxis_title="Price ($)",
            height=600
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with be_col:
        st.markdown("**Break-Even Analysis**")
        
        # Find break-even points
        be_points = []
        for feat in feature_cols[:4]:
            data = detailed_sens[feat]
            # Find where prediction crosses zero (if profit) or target
            target_be = 0 if target_variable == 'profit' else base_prediction * 0.5
            
            closest_to_be = min(data['predictions'], key=lambda x: abs(x - target_be))
            be_idx = list(data['predictions']).index(closest_to_be)
            be_change = data['variations'][be_idx]
            
            be_points.append({
                'Variable': feat.replace('_', ' ').title(),
                'Break-Even Change': f"{be_change:.1f}%",
                'Safety Margin': f"{abs(be_change):.1f}%"
            })
        
        be_df = pd.DataFrame(be_points)
        st.dataframe(be_df, use_container_width=True, hide_index=True)
        
        # Safety margin gauge
        st.markdown("**Minimum Safety Margin**")
        min_margin = min([float(x['Safety Margin'].replace('%', '')) for x in be_points])
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = min_margin,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Safety Margin %"},
            delta = {'reference': 20, 'increasing': {'color': "green"}},
            gauge = {
                'axis': {'range': [None, 50]},
                'bar': {'color': "#2ecc71" if min_margin > 20 else "#f39c12" if min_margin > 10 else "#e74c3c"},
                'steps': [
                    {'range': [0, 10], 'color': "#ffcccc"},
                    {'range': [10, 20], 'color': "#ffffcc"},
                    {'range': [20, 50], 'color': "#ccffcc"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 20
                }
            }
        ))
        fig_gauge.update_layout(height=400)
        st.plotly_chart(fig_gauge, use_container_width=True)

# ==================== TAB 4: STRATEGY OPTIMIZER ====================
with tab4:
    st.header("💡 Strategic Decision Optimizer")
    
    # Optimization scenarios
    st.subheader("18. Multi-Objective Optimization")
    
    # Generate Pareto frontier
    optimization_results = []
    
    # Grid search over key variables
    mkt_multipliers = np.linspace(0.5, 2.0, 15)
    price_multipliers = np.linspace(0.8, 1.3, 10)
    
    for m_m in mkt_multipliers:
        for p_m in price_multipliers:
            test_feat = base_features.copy().flatten()
            test_feat[0] = new_marketing * m_m
            test_feat[1] = new_price * p_m
            test_feat[4] = test_feat[0] * digital_ratio
            
            pred_profit = model.predict(test_feat.reshape(1, -1))[0]
            
            # Calculate other metrics
            pred_revenue = pred_profit * np.random.uniform(2.5, 4.0)  # Approximate relationship
            pred_roi = (pred_profit / test_feat[0]) * 100
            risk = abs(m_m - 1) * 10 + abs(p_m - 1) * 15  # Proxy for risk
            
            optimization_results.append({
                'marketing_mult': m_m,
                'price_mult': p_m,
                'marketing_spend': test_feat[0],
                'price': test_feat[1],
                'profit': pred_profit,
                'revenue': pred_revenue,
                'roi': pred_roi,
                'risk_score': risk
            })
    
    opt_df = pd.DataFrame(optimization_results)
    
    # Visualization 18: 3D Pareto Surface
    col_pareto, col_eff = st.columns([3, 2])
    
    with col_pareto:
        st.markdown("**Profit vs Revenue vs Risk Trade-off**")
        
        fig_3d_pareto = go.Figure(data=[go.Scatter3d(
            x=opt_df['revenue'],
            y=opt_df['profit'],
            z=opt_df['risk_score'],
            mode='markers',
            marker=dict(
                size=5,
                color=opt_df['roi'],
                colorscale='Viridis',
                opacity=0.8,
                showscale=True,
                colorbar=dict(title='ROI %')
            ),
            text=[f"Mkt: ${m:,.0f}<br>Price: ${p:.2f}" 
                  for m, p in zip(opt_df['marketing_spend'], opt_df['price'])],
            hovertemplate='<b>Revenue:</b> $%{x:,.0f}<br>' +
                         '<b>Profit:</b> $%{y:,.0f}<br>' +
                         '<b>Risk:</b> %{z:.1f}<br>' +
                         '%{text}'
        )])
        
        fig_3d_pareto.update_layout(
            scene=dict(
                xaxis_title='Revenue ($)',
                yaxis_title='Profit ($)',
                zaxis_title='Risk Score'
            ),
            height=600,
            title="3D Pareto Optimization Surface"
        )
        st.plotly_chart(fig_3d_pareto, use_container_width=True)
    
    with col_eff:
        st.markdown("**Efficient Frontier: Risk vs Return**")
        
        # Calculate efficient frontier
        opt_df['return_per_risk'] = opt_df['profit'] / np.maximum(opt_df['risk_score'], 0.1)
        
        # Find Pareto optimal points (max profit for given risk)
        pareto_points = []
        for risk_level in np.linspace(opt_df['risk_score'].min(), opt_df['risk_score'].max(), 20):
            subset = opt_df[opt_df['risk_score'] <= risk_level]
            if len(subset) > 0:
                best = subset.loc[subset['profit'].idxmax()]
                pareto_points.append(best)
        
        pareto_df = pd.DataFrame(pareto_points)
        
        fig_eff = go.Figure()
        
        # All points
        fig_eff.add_trace(go.Scatter(
            x=opt_df['risk_score'],
            y=opt_df['profit'],
            mode='markers',
            name='All Scenarios',
            marker=dict(color='lightblue', size=6, opacity=0.5)
        ))
        
        # Efficient frontier
        fig_eff.add_trace(go.Scatter(
            x=pareto_df['risk_score'],
            y=pareto_df['profit'],
            mode='lines+markers',
            name='Efficient Frontier',
            line=dict(color='red', width=3),
            marker=dict(size=8)
        ))
        
        # Current position
        fig_eff.add_trace(go.Scatter(
            x=[0],
            y=[base_prediction],
            mode='markers',
            name='Current Position',
            marker=dict(color='green', size=15, symbol='star')
        ))
        
        fig_eff.update_layout(
            xaxis_title='Risk Score',
            yaxis_title=f'Expected {target_variable.replace("_", " ").title()}',
            height=600,
            showlegend=True
        )
        st.plotly_chart(fig_eff, use_container_width=True)
    
    # Row 2: Decision matrix and regret analysis
    st.subheader("19. Decision Matrix & Regret Analysis")
    
    # Define specific strategic options
    strategies = {
        'Status Quo': {'marketing': 1.0, 'price': 1.0, 'digital': digital_ratio},
        'Growth Focus': {'marketing': 1.4, 'price': 0.95, 'digital': 0.8},
        'Profit Focus': {'marketing': 0.8, 'price': 1.15, 'digital': 0.6},
        'Digital First': {'marketing': 1.2, 'price': 1.0, 'digital': 0.9},
        'Premium Push': {'marketing': 1.1, 'price': 1.25, 'digital': 0.5},
        'Aggressive Promo': {'marketing': 1.6, 'price': 0.85, 'digital': 0.85}
    }
    
    # Evaluate each strategy under different market conditions
    market_conditions = ['Boom', 'Normal', 'Recession', 'Volatile']
    condition_multipliers = {
        'Boom': {'demand': 1.3, 'cost': 0.9},
        'Normal': {'demand': 1.0, 'cost': 1.0},
        'Recession': {'demand': 0.6, 'cost': 1.1},
        'Volatile': {'demand': 1.0, 'cost': 1.2}
    }
    
    decision_matrix = []
    for strat_name, params in strategies.items():
        row = {'Strategy': strat_name}
        for condition in market_conditions:
            mult = condition_multipliers[condition]
            test_feat = base_features.copy().flatten()
            test_feat[0] = new_marketing * params['marketing'] * mult['demand']
            test_feat[1] = new_price * params['price']
            test_feat[4] = test_feat[0] * params['digital']
            
            pred = model.predict(test_feat.reshape(1, -1))[0] * mult['demand'] / mult['cost']
            row[condition] = pred
        
        decision_matrix.append(row)
    
    dec_df = pd.DataFrame(decision_matrix)
    
    # Decision heatmap
    fig_decision = px.imshow(
        dec_df.set_index('Strategy')[market_conditions],
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdYlGn',
        title="Payoff Matrix: Strategy vs Market Condition"
    )
    fig_decision.update_layout(height=400)
    st.plotly_chart(fig_decision, use_container_width=True)
    
    # Regret analysis
    st.subheader("20. Minimax Regret Analysis")
    
    # Calculate regret matrix
    payoff_matrix = dec_df.set_index('Strategy')[market_conditions]
    max_payoffs = payoff_matrix.max(axis=0)
    regret_matrix = max_payoffs - payoff_matrix
    
    fig_regret = px.imshow(
        regret_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='Reds',
        title="Regret Matrix (Opportunity Cost)"
    )
    fig_regret.update_layout(height=400)
    st.plotly_chart(fig_regret, use_container_width=True)
    
    # Strategy recommendations
    max_regrets = regret_matrix.max(axis=1)
    best_strategy = max_regrets.idxmin()
    
    st.markdown(f"""
    <div class="insight-box">
        <h4>🎯 Recommended Strategy: {best_strategy}</h4>
        <p>This strategy minimizes your maximum potential regret (Minimax criterion), 
        ensuring you don't miss out on more than <b>${max_regrets[best_strategy]:,.0f}</b> 
        compared to the optimal choice in any scenario.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Row 3: Opportunity cost and cumulative analysis
    col_opp, col_cum = st.columns(2)
    
    with col_opp:
        st.subheader("21. Opportunity Cost Analysis")
        
        # Compare each strategy to best in each condition
        opp_cost_data = []
        for condition in market_conditions:
            best_val = payoff_matrix[condition].max()
            for strategy in payoff_matrix.index:
                opp_cost = best_val - payoff_matrix.loc[strategy, condition]
                opp_cost_data.append({
                    'Condition': condition,
                    'Strategy': strategy,
                    'Opportunity Cost': opp_cost,
                    'Pct of Best': (payoff_matrix.loc[strategy, condition] / best_val) * 100
                })
        
        opp_df = pd.DataFrame(opp_cost_data)
        
        fig_opp = px.bar(opp_df, x='Strategy', y='Opportunity Cost', 
                        color='Condition', barmode='group',
                        title="Opportunity Cost by Strategy",
                        color_discrete_sequence=px.colors.qualitative.Set1)
        fig_opp.update_layout(height=500)
        st.plotly_chart(fig_opp, use_container_width=True)
    
    with col_cum:
        st.subheader("22. Cumulative Gain Analysis")
        
        # Simulate cumulative gains over time for top 3 strategies
        days = pd.date_range(start='2024-01-01', periods=365, freq='D')
        
        # Get top 3 strategies by average payoff
        avg_payoffs = payoff_matrix.mean(axis=1).sort_values(ascending=False)
        top_strategies = avg_payoffs.head(3).index.tolist()
        
        cum_data = []
        for strategy in top_strategies:
            base_daily = payoff_matrix.loc[strategy].mean() / 365
            cumulative = 0
            for i, day in enumerate(days):
                daily = base_daily * np.random.normal(1, 0.2)
                cumulative += daily
                cum_data.append({
                    'Date': day,
                    'Strategy': strategy,
                    'Cumulative': cumulative
                })
        
        cum_df = pd.DataFrame(cum_data)
        
        fig_cum = px.line(cum_df, x='Date', y='Cumulative', color='Strategy',
                         title="Projected Cumulative Gains (1 Year)",
                         line_shape='spline')
        fig_cum.update_layout(height=500)
        st.plotly_chart(fig_cum, use_container_width=True)

# ==================== TAB 5: ADVANCED ANALYTICS ====================
with tab5:
    st.header("🔬 Advanced Statistical Analytics")
    
    # Residual analysis
    st.subheader("23. Model Diagnostics: Residual Analysis")
    
    residuals = y_test - y_pred
    fitted_vals = y_pred
    
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        # Residuals vs Fitted
        fig_res1 = go.Figure()
        fig_res1.add_trace(go.Scatter(
            x=fitted_vals, y=residuals,
            mode='markers',
            marker=dict(color='blue', opacity=0.5),
            name='Residuals'
        ))
        fig_res1.add_hline(y=0, line_dash="dash", line_color="red")
        
        # Add smoothed trend
        z = np.polyfit(fitted_vals, residuals, 1)
        p = np.poly1d(z)
        fig_res1.add_trace(go.Scatter(
            x=np.sort(fitted_vals),
            y=p(np.sort(fitted_vals)),
            mode='lines',
            name='Trend',
            line=dict(color='red', width=2)
        ))
        
        fig_res1.update_layout(
            title="Residuals vs Fitted Values",
            xaxis_title="Fitted Values",
            yaxis_title="Residuals",
            height=450
        )
        st.plotly_chart(fig_res1, use_container_width=True)
    
    with res_col2:
        # Q-Q Plot
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
        sorted_residuals = np.sort(residuals)
        
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(
            x=theoretical_quantiles,
            y=sorted_residuals,
            mode='markers',
            name='Residuals'
        ))
        
        # Add reference line
        min_val = min(theoretical_quantiles.min(), sorted_residuals.min())
        max_val = max(theoretical_quantiles.max(), sorted_residuals.max())
        fig_qq.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Normal Line',
            line=dict(color='red', dash='dash')
        ))
        
        fig_qq.update_layout(
            title="Normal Q-Q Plot",
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Sample Quantiles",
            height=450
        )
        st.plotly_chart(fig_qq, use_container_width=True)
    
    # Scale-location and leverage
    res_col3, res_col4 = st.columns(2)
    
    with res_col3:
        # Scale-Location plot (sqrt of standardized residuals vs fitted)
        standardized_res = (residuals - residuals.mean()) / residuals.std()
        sqrt_std_res = np.sqrt(np.abs(standardized_res))
        
        fig_scale = go.Figure()
        fig_scale.add_trace(go.Scatter(
            x=fitted_vals, y=sqrt_std_res,
            mode='markers',
            marker=dict(color='green', opacity=0.5),
            name='√|Standardized Residuals|'
        ))
        
        # Add trend line
        z = np.polyfit(fitted_vals, sqrt_std_res, 1)
        p = np.poly1d(z)
        fig_scale.add_trace(go.Scatter(
            x=np.sort(fitted_vals),
            y=p(np.sort(fitted_vals)),
            mode='lines',
            name='Trend',
            line=dict(color='red', width=2)
        ))
        
        fig_scale.update_layout(
            title="Scale-Location Plot",
            xaxis_title="Fitted Values",
            yaxis_title="√|Standardized Residuals|",
            height=450
        )
        st.plotly_chart(fig_scale, use_container_width=True)
    
    with res_col4:
        # Residual distribution histogram with KDE
        fig_res_hist = ff.create_distplot(
            [residuals],
            ['Residuals'],
            bin_size=(residuals.max() - residuals.min()) / 30,
            colors=['#3498db']
        )
        
        # Add normal overlay
        x_norm = np.linspace(residuals.min(), residuals.max(), 100)
        y_norm = stats.norm.pdf(x_norm, residuals.mean(), residuals.std())
        fig_res_hist.add_trace(go.Scatter(
            x=x_norm, y=y_norm * len(residuals) * (residuals.max() - residuals.min()) / 30,
            mode='lines', name='Normal Fit', line=dict(color='red', width=2)
        ))
        
        fig_res_hist.update_layout(
            title="Residual Distribution",
            height=450,
            showlegend=True
        )
        st.plotly_chart(fig_res_hist, use_container_width=True)
    
    # Feature importance and partial dependence
    st.subheader("24. Feature Importance & Partial Dependence")
    
    feat_col1, feat_col2 = st.columns([2, 3])
    
    with feat_col1:
        # Permutation importance
        perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        
        imp_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': perm_importance.importances_mean,
            'Std': perm_importance.importances_std
        }).sort_values('Importance', ascending=True)
        
        fig_perm = go.Figure()
        fig_perm.add_trace(go.Bar(
            y=imp_df['Feature'],
            x=imp_df['Importance'],
            orientation='h',
            error_x=dict(type='data', array=imp_df['Std']),
            marker_color='#e74c3c'
        ))
        fig_perm.update_layout(
            title="Permutation Importance<br>(with std dev)",
            xaxis_title="Importance",
            height=500
        )
        st.plotly_chart(fig_perm, use_container_width=True)
    
    with feat_col2:
        # Partial dependence plots for top 3 features
        top_features = imp_df.tail(3)['Feature'].tolist()
        
        fig_pdp = make_subplots(rows=1, cols=3, 
                               subplot_titles=[f.replace('_', ' ').title() for f in top_features])
        
        for idx, feat in enumerate(top_features):
            feat_idx = feature_cols.index(feat)
            
            # Create partial dependence
            feat_values = np.linspace(X[feat].min(), X[feat].max(), 50)
            pdp_values = []
            
            for val in feat_values:
                X_temp = X_test.copy()
                X_temp.iloc[:, feat_idx] = val
                preds = model.predict(X_temp)
                pdp_values.append(preds.mean())
            
            fig_pdp.add_trace(
                go.Scatter(x=feat_values, y=pdp_values, mode='lines',
                          line=dict(width=3),
                          name=feat.replace('_', ' ').title()),
                row=1, col=idx+1
            )
        
        fig_pdp.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig_pdp, use_container_width=True)
    
    # Time series decomposition
    st.subheader("25. Time Series Decomposition")
    
    ts_data = df.groupby('date')['revenue'].sum().reset_index()
    ts_data['trend'] = ts_data['revenue'].rolling(window=30, center=True).mean()
    ts_data['detrended'] = ts_data['revenue'] - ts_data['trend']
    
    # Simple seasonal decomposition (monthly)
    ts_data['month'] = ts_data['date'].dt.month
    seasonal_means = ts_data.groupby('month')['detrended'].mean()
    ts_data['seasonal'] = ts_data['month'].map(seasonal_means)
    ts_data['residual'] = ts_data['revenue'] - ts_data['trend'] - ts_data['seasonal']
    
    fig_decomp = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'),
        vertical_spacing=0.08
    )
    
    fig_decomp.add_trace(go.Scatter(x=ts_data['date'], y=ts_data['revenue'], 
                                   mode='lines', name='Original', line=dict(color='#3498db')), row=1, col=1)
    fig_decomp.add_trace(go.Scatter(x=ts_data['date'], y=ts_data['trend'], 
                                   mode='lines', name='Trend', line=dict(color='#e74c3c')), row=2, col=1)
    fig_decomp.add_trace(go.Scatter(x=ts_data['date'], y=ts_data['seasonal'], 
                                   mode='lines', name='Seasonal', line=dict(color='#2ecc71')), row=3, col=1)
    fig_decomp.add_trace(go.Scatter(x=ts_data['date'], y=ts_data['residual'], 
                                   mode='lines', name='Residual', line=dict(color='#95a5a6')), row=4, col=1)
    
    fig_decomp.update_layout(height=800, showlegend=False)
    st.plotly_chart(fig_decomp, use_container_width=True)
    
    # Bootstrap confidence intervals
    st.subheader("26. Bootstrap Confidence Intervals")
    
    n_bootstrap = 1000
    bootstrap_preds = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        idx = np.random.choice(len(X_test), size=len(X_test), replace=True)
        X_boot = X_test.iloc[idx]
        y_boot = y_test.iloc[idx]
        
        # Refit model (simplified - just predict with current model on bootstrapped data)
        pred = model.predict(X_boot).mean()
        bootstrap_preds.append(pred)
    
    boot_ci_lower = np.percentile(bootstrap_preds, (100-confidence_level)/2)
    boot_ci_upper = np.percentile(bootstrap_preds, 100 - (100-confidence_level)/2)
    
    fig_boot = go.Figure()
    fig_boot.add_trace(go.Histogram(x=bootstrap_preds, nbinsx=30, 
                                   name='Bootstrap Distribution',
                                   marker_color='lightblue', opacity=0.7))
    fig_boot.add_vline(x=np.mean(bootstrap_preds), line_color='blue', 
                      annotation_text='Bootstrap Mean')
    fig_boot.add_vline(x=boot_ci_lower, line_dash="dash", line_color='red',
                      annotation_text=f'{confidence_level}% CI')
    fig_boot.add_vline(x=boot_ci_upper, line_dash="dash", line_color='red')
    
    fig_boot.update_layout(
        title=f"Bootstrap Distribution of Mean Prediction (n={n_bootstrap})",
        xaxis_title="Mean Prediction",
        height=500
    )
    st.plotly_chart(fig_boot, use_container_width=True)

# ==================== TAB 6: INTERACTIVE EXPLORER ====================
with tab6:
    st.header("🌐 Interactive Data Explorer")
    
    # Parallel coordinates
    st.subheader("27. Parallel Coordinates: Multi-Dimensional Analysis")
    
    # Sample data for performance
    sample_df = df.sample(min(500, len(df)), random_state=42)
    
    fig_parallel = px.parallel_coordinates(
        sample_df,
        dimensions=['marketing_spend', 'effective_price', 'total_users', 
                   'conversion_rate', 'revenue', 'profit'],
        color='profit',
        color_continuous_scale='RdYlGn',
        title="Parallel Coordinates Plot"
    )
    fig_parallel.update_layout(height=600)
    st.plotly_chart(fig_parallel, use_container_width=True)
    
    # Sankey diagram
    st.subheader("28. Flow Analysis: Marketing to Profit")
    
    # Create categories for Sankey
    df['mkt_category'] = pd.qcut(df['marketing_spend'], 3, labels=['Low Mkt', 'Med Mkt', 'High Mkt'])
    df['price_category'] = pd.qcut(df['effective_price'], 3, labels=['Low Price', 'Med Price', 'High Price'])
    df['profit_category'] = pd.qcut(df['profit'], 3, labels=['Loss', 'Breakeven', 'Profit'])
    
    # Create flow data
    flow_data = df.groupby(['mkt_category', 'price_category', 'profit_category']).size().reset_index(name='value')
    
    # Create node list
    nodes = list(df['mkt_category'].unique()) + list(df['price_category'].unique()) + list(df['profit_category'].unique())
    node_indices = {node: idx for idx, node in enumerate(nodes)}
    
    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color=["#3498db", "#3498db", "#3498db", 
                   "#2ecc71", "#2ecc71", "#2ecc71",
                   "#e74c3c", "#f39c12", "#27ae60"]
        ),
        link=dict(
            source=[node_indices[row['mkt_category']] for _, row in flow_data.iterrows()] +
                   [node_indices[row['price_category']] + 3 for _, row in flow_data.iterrows()],
            target=[node_indices[row['price_category']] + 3 for _, row in flow_data.iterrows()] +
                   [node_indices[row['profit_category']] + 6 for _, row in flow_data.iterrows()],
            value=list(flow_data['value']) * 2
        )
    )])
    
    fig_sankey.update_layout(title="Marketing → Price → Profit Flow", height=600)
    st.plotly_chart(fig_sankey, use_container_width=True)
    
    # Sunburst chart
    st.subheader("29. Hierarchical Sunburst")
    
    # Create hierarchical data
    df['year'] = df['date'].dt.year
    df['quarter'] = 'Q' + df['quarter'].astype(str)
    
    fig_sunburst = px.sunburst(
        df,
        path=['year', 'quarter', 'month'],
        values='revenue',
        color='profit',
        color_continuous_scale='RdYlGn',
        title="Revenue Hierarchy with Profit Heatmap"
    )
    fig_sunburst.update_layout(height=600)
    st.plotly_chart(fig_sunburst, use_container_width=True)
    
    # Animated bubble chart
    st.subheader("30. Animated Bubble Chart: Time Evolution")
    
    # Aggregate by month for animation
    monthly_agg = df.groupby(['year', 'month']).agg({
        'marketing_spend': 'mean',
        'effective_price': 'mean',
        'total_users': 'mean',
        'profit': 'mean',
        'revenue': 'sum',
        'date': 'first'
    }).reset_index()
    
    monthly_agg['time_label'] = monthly_agg['year'].astype(str) + '-' + monthly_agg['month'].astype(str).str.zfill(2)
    
    fig_bubble = px.scatter(
        monthly_agg,
        x='marketing_spend',
        y='profit',
        size='revenue',
        color='effective_price',
        animation_frame='time_label',
        hover_name='time_label',
        size_max=60,
        range_x=[monthly_agg['marketing_spend'].min()*0.9, monthly_agg['marketing_spend'].max()*1.1],
        range_y=[monthly_agg['profit'].min()*1.2, monthly_agg['profit'].max()*1.2],
        title="Evolution: Marketing vs Profit (Size=Revenue, Color=Price)"
    )
    fig_bubble.update_layout(height=600)
    st.plotly_chart(fig_bubble, use_container_width=True)
    
    # Calendar heatmap
    st.subheader("31. Calendar Heatmap: Daily Performance")
    
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week'] = df['date'].dt.isocalendar().week
    
    # Create calendar data
    calendar_data = df.groupby(['week', 'day_of_week'])['profit'].mean().reset_index()
    
    fig_calendar = px.density_heatmap(
        calendar_data,
        x='day_of_week',
        y='week',
        z='profit',
        color_continuous_scale='RdYlGn',
        title="Weekly Profit Patterns (Day of Week vs Week of Year)",
        labels={'day_of_week': 'Day of Week (0=Mon)', 'week': 'Week of Year'}
    )
    fig_calendar.update_layout(height=600)
    st.plotly_chart(fig_calendar, use_container_width=True)
    
    # Network-style correlation
    st.subheader("32. Network Correlation Graph")
    
    # Create network of highly correlated features
    corr_thresh = 0.5
    corr_matrix = df[feature_cols + ['profit', 'revenue']].corr()
    
    # Create edges
    edges_x = []
    edges_y = []
    edge_weights = []
    node_labels = []
    node_x = []
    node_y = []
    
    # Position nodes in circle
    n_nodes = len(corr_matrix.columns)
    angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
    
    for i, col in enumerate(corr_matrix.columns):
        node_x.append(np.cos(angles[i]))
        node_y.append(np.sin(angles[i]))
        node_labels.append(col.replace('_', ' ').title())
        
        for j in range(i+1, n_nodes):
            if abs(corr_matrix.iloc[i, j]) > corr_thresh:
                edges_x.extend([np.cos(angles[i]), np.cos(angles[j]), None])
                edges_y.extend([np.sin(angles[i]), np.sin(angles[j]), None])
                edge_weights.append(abs(corr_matrix.iloc[i, j]))
    
    fig_network = go.Figure()
    
    # Add edges
    fig_network.add_trace(go.Scatter(
        x=edges_x, y=edges_y,
        mode='lines',
        line=dict(color='gray', width=1),
        hoverinfo='none'
    ))
    
    # Add nodes
    node_colors = [corr_matrix.loc[col, 'profit'] for col in corr_matrix.columns]
    
    fig_network.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(
            size=20,
            color=node_colors,
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title='Correlation with Profit')
        ),
        text=node_labels,
        textposition="top center",
        hovertemplate='%{text}<br>Corr with Profit: %{marker.color:.2f}'
    ))
    
    fig_network.update_layout(
        title="Feature Correlation Network (edges > 0.5 correlation)",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600
    )
    st.plotly_chart(fig_network, use_container_width=True)

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
        © 2024 | Built with Streamlit, Plotly & Scikit-Learn | 40+ Visualizations
    </p>
</div>
""", unsafe_allow_html=True)
