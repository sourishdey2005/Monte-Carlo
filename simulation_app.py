import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
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
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .scenario-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .highlight {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 0.3rem;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🎲 What-If Business Simulation Tool</h1>', unsafe_allow_html=True)
st.markdown("""
This interactive tool helps you answer **"What happens if we change X?"** using Monte Carlo simulation 
and sensitivity analysis. Adjust marketing budget, pricing, and growth assumptions to see probabilistic outcomes.
""")

# Generate synthetic real-world dataset
@st.cache_data
def generate_business_data(n_samples=1000):
    """
    Generate a realistic e-commerce dataset with marketing, sales, and user metrics.
    """
    np.random.seed(42)
    
    # Base features
    months = np.random.choice(range(1, 13), n_samples)
    marketing_spend = np.random.normal(50000, 15000, n_samples)  # Monthly marketing budget
    marketing_spend = np.clip(marketing_spend, 10000, 150000)
    
    # Price per product (with some variation)
    base_price = np.random.choice([29.99, 49.99, 79.99, 99.99, 149.99], n_samples)
    discount_rate = np.random.beta(2, 5, n_samples) * 0.3  # 0-30% discount
    
    effective_price = base_price * (1 - discount_rate)
    
    # User growth (organic + paid)
    organic_growth = np.random.normal(0.05, 0.02, n_samples)  # 5% monthly organic
    paid_acquisition = (marketing_spend / 50) * np.random.normal(1, 0.3, n_samples)  # $50 CAC variation
    total_new_users = (organic_growth * 10000) + paid_acquisition
    
    # Seasonality effect
    seasonality = 1 + 0.3 * np.sin((months / 12) * 2 * np.pi) + np.random.normal(0, 0.1, n_samples)
    
    # Conversion rate affected by price (inverse relationship) and marketing quality
    base_conversion = 0.03  # 3% base conversion
    price_elasticity = -0.002 * (effective_price - 50)  # Price sensitivity
    marketing_quality = np.random.beta(3, 2, n_samples)  # Marketing efficiency
    
    conversion_rate = base_conversion + price_elasticity + (marketing_quality * 0.02)
    conversion_rate = np.clip(conversion_rate, 0.005, 0.15)
    
    # Calculate revenue
    orders = total_new_users * conversion_rate * seasonality
    revenue = orders * effective_price
    cost_of_goods = revenue * np.random.uniform(0.4, 0.6, n_samples)  # 40-60% COGS
    operational_cost = marketing_spend + (revenue * 0.15)  # Marketing + 15% ops
    
    profit = revenue - cost_of_goods - operational_cost
    
    # Create DataFrame
    df = pd.DataFrame({
        'month': months,
        'marketing_spend': marketing_spend,
        'base_price': base_price,
        'discount_rate': discount_rate,
        'effective_price': effective_price,
        'organic_growth_rate': organic_growth,
        'paid_acquisition': paid_acquisition,
        'total_new_users': total_new_users,
        'seasonality': seasonality,
        'conversion_rate': conversion_rate,
        'orders': orders,
        'revenue': revenue,
        'cost_of_goods': cost_of_goods,
        'operational_cost': operational_cost,
        'profit': profit,
        'roi': (profit / marketing_spend) * 100
    })
    
    return df

# Load data
df = generate_business_data(2000)

# Sidebar controls
st.sidebar.header("⚙️ Simulation Parameters")

# Variable selection for what-if analysis
st.sidebar.subheader("1. Baseline Model")
target_variable = st.sidebar.selectbox(
    "Target Variable to Predict",
    ['revenue', 'profit', 'roi', 'orders'],
    index=1
)

feature_cols = ['marketing_spend', 'effective_price', 'total_new_users', 'conversion_rate']
X = df[feature_cols]
y = df[target_variable]

# Train baseline model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Model performance metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

st.sidebar.markdown(f"""
<div class="metric-card">
    <strong>Model Performance</strong><br>
    R² Score: {r2:.3f}<br>
    MAE: ${mae:,.0f}
</div>
""", unsafe_allow_html=True)

# Feature importance (coefficients)
st.sidebar.subheader("Feature Impact (Coefficients)")
coef_df = pd.DataFrame({
    'Feature': feature_cols,
    'Coefficient': model.coef_,
    'Impact': ['High' if abs(c) > np.abs(model.coef_).mean() else 'Medium' for c in model.coef_]
})
st.sidebar.dataframe(coef_df, hide_index=True)

# What-if scenario inputs
st.sidebar.subheader("2. What-If Scenarios")

# Current means for reference
current_marketing = df['marketing_spend'].mean()
current_price = df['effective_price'].mean()
current_users = df['total_new_users'].mean()

# Scenario inputs
marketing_change = st.sidebar.slider(
    "Marketing Budget Change (%)", 
    -50, 100, 0, 5,
    help="Adjust marketing spend relative to baseline"
)

price_change = st.sidebar.slider(
    "Price Change (%)", 
    -30, 50, 0, 5,
    help="Adjust product pricing relative to baseline"
)

user_growth_change = st.sidebar.slider(
    "User Growth Change (%)", 
    -50, 100, 0, 5,
    help="Adjust user acquisition relative to baseline"
)

# Monte Carlo parameters
st.sidebar.subheader("3. Monte Carlo Settings")
n_simulations = st.sidebar.slider("Number of Simulations", 100, 10000, 1000, 100)
confidence_level = st.sidebar.slider("Confidence Level", 80, 99, 95, 1)

# Main content area
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Baseline Analysis", 
    "🎲 Monte Carlo Simulation", 
    "📈 Sensitivity Analysis",
    "💡 Decision Insights"
])

# Tab 1: Baseline Analysis
with tab1:
    st.header("Historical Data & Baseline Model")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg Revenue", f"${df['revenue'].mean():,.0f}", 
                 f"${df['revenue'].std():,.0f} σ")
    with col2:
        st.metric("Avg Profit", f"${df['profit'].mean():,.0f}", 
                 f"${df['profit'].std():,.0f} σ")
    with col3:
        st.metric("Avg ROI", f"{df['roi'].mean():.1f}%", 
                 f"{df['roi'].std():.1f}% σ")
    with col4:
        st.metric("Conversion Rate", f"{df['conversion_rate'].mean():.2%}", 
                 f"{df['conversion_rate'].std():.2%} σ")
    
    # Historical trends
    st.subheader("Historical Performance Trends")
    
    # Aggregate by month for trend
    monthly_stats = df.groupby('month').agg({
        'revenue': 'mean',
        'profit': 'mean',
        'marketing_spend': 'mean',
        'roi': 'mean'
    }).reset_index()
    
    fig_trends = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Revenue Trend', 'Profit Trend', 'Marketing Spend', 'ROI Trend'),
        vertical_spacing=0.15
    )
    
    fig_trends.add_trace(
        go.Scatter(x=monthly_stats['month'], y=monthly_stats['revenue'], 
                  mode='lines+markers', name='Revenue', line=dict(color='#2E86AB')),
        row=1, col=1
    )
    fig_trends.add_trace(
        go.Scatter(x=monthly_stats['month'], y=monthly_stats['profit'], 
                  mode='lines+markers', name='Profit', line=dict(color='#A23B72')),
        row=1, col=2
    )
    fig_trends.add_trace(
        go.Scatter(x=monthly_stats['month'], y=monthly_stats['marketing_spend'], 
                  mode='lines+markers', name='Marketing', line=dict(color='#F18F01')),
        row=2, col=1
    )
    fig_trends.add_trace(
        go.Scatter(x=monthly_stats['month'], y=monthly_stats['roi'], 
                  mode='lines+markers', name='ROI', line=dict(color='#C73E1D')),
        row=2, col=2
    )
    
    fig_trends.update_layout(height=600, showlegend=False, 
                            title_text="Monthly Performance Metrics")
    st.plotly_chart(fig_trends, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("Feature Correlations")
    corr_cols = ['marketing_spend', 'effective_price', 'conversion_rate', 
                 'total_new_users', 'revenue', 'profit', 'roi']
    corr_matrix = df[corr_cols].corr()
    
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Correlation Matrix of Key Business Metrics"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# Tab 2: Monte Carlo Simulation
with tab2:
    st.header("🎲 Monte Carlo Simulation Results")
    
    # Calculate new baseline with changes
    new_marketing = current_marketing * (1 + marketing_change/100)
    new_price = current_price * (1 + price_change/100)
    new_users = current_users * (1 + user_growth_change/100)
    
    # Keep conversion rate but add uncertainty
    base_conversion = df['conversion_rate'].mean()
    conversion_std = df['conversion_rate'].std()
    
    # Run Monte Carlo simulation
    np.random.seed(42)
    simulation_results = []
    
    for i in range(n_simulations):
        # Add randomness to inputs based on historical variance
        sim_marketing = np.random.normal(new_marketing, new_marketing * 0.1)
        sim_price = np.random.normal(new_price, new_price * 0.05)
        sim_users = np.random.normal(new_users, new_users * 0.15)
        sim_conversion = np.random.normal(base_conversion, conversion_std)
        sim_conversion = np.clip(sim_conversion, 0.001, 0.5)
        
        # Create feature vector (matching training features)
        features = np.array([sim_marketing, sim_price, sim_users, sim_conversion]).reshape(1, -1)
        
        # Predict with some added noise for model uncertainty
        base_pred = model.predict(features)[0]
        noise = np.random.normal(0, mae * 0.5)  # Add model uncertainty
        prediction = base_pred + noise
        
        simulation_results.append({
            'marketing': sim_marketing,
            'price': sim_price,
            'users': sim_users,
            'conversion': sim_conversion,
            'prediction': prediction
        })
    
    sim_df = pd.DataFrame(simulation_results)
    
    # Display scenario summary
    st.markdown(f"""
    <div class="scenario-card">
        <h3>Scenario Configuration</h3>
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;">
            <div>
                <strong>Marketing Budget</strong><br>
                ${new_marketing:,.0f} ({marketing_change:+.0f}%)
            </div>
            <div>
                <strong>Price Point</strong><br>
                ${new_price:.2f} ({price_change:+.0f}%)
            </div>
            <div>
                <strong>User Growth</strong><br>
                {new_users:,.0f} users ({user_growth_change:+.0f}%)
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Key statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Prediction", f"${sim_df['prediction'].mean():,.0f}")
    with col2:
        st.metric("Std Deviation", f"${sim_df['prediction'].std():,.0f}")
    with col3:
        ci_lower = np.percentile(sim_df['prediction'], (100-confidence_level)/2)
        ci_upper = np.percentile(sim_df['prediction'], 100 - (100-confidence_level)/2)
        st.metric(f"{confidence_level}% CI Lower", f"${ci_lower:,.0f}")
    with col4:
        st.metric(f"{confidence_level}% CI Upper", f"${ci_upper:,.0f}")
    
    # Distribution plot
    st.subheader("Outcome Distribution")
    
    fig_dist = go.Figure()
    
    # Histogram
    fig_dist.add_trace(go.Histogram(
        x=sim_df['prediction'],
        nbinsx=50,
        name='Simulations',
        opacity=0.7,
        marker_color='#1f77b4'
    ))
    
    # Add mean line
    mean_val = sim_df['prediction'].mean()
    fig_dist.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                      annotation_text=f"Mean: ${mean_val:,.0f}")
    
    # Add confidence interval
    fig_dist.add_vrect(
        x0=ci_lower, x1=ci_upper,
        fillcolor="green", opacity=0.2,
        layer="below", line_width=0,
        annotation_text=f"{confidence_level}% Confidence Interval"
    )
    
    fig_dist.update_layout(
        title=f"Distribution of {n_simulations} Monte Carlo Simulations",
        xaxis_title=f"Predicted {target_variable.replace('_', ' ').title()} ($)",
        yaxis_title="Frequency",
        showlegend=False,
        height=500
    )
    
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # Percentile table
    st.subheader("Detailed Percentile Analysis")
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    percentile_values = [np.percentile(sim_df['prediction'], p) for p in percentiles]
    
    percentile_df = pd.DataFrame({
        'Percentile': [f"{p}th" for p in percentiles],
        'Value': [f"${v:,.0f}" for v in percentile_values],
        'Interpretation': [
            'Worst Case (5%)', 'Very Pessimistic', 'Pessimistic', 
            'Median (Base Case)', 'Optimistic', 'Very Optimistic', 'Best Case (95%)'
        ]
    })
    
    st.dataframe(percentile_df, use_container_width=True, hide_index=True)
    
    # Risk metrics
    st.subheader("Risk Assessment")
    var_95 = np.percentile(sim_df['prediction'], 5)  # Value at Risk
    cvar_95 = sim_df[sim_df['prediction'] <= var_95]['prediction'].mean()  # Conditional VaR
    
    risk_col1, risk_col2 = st.columns(2)
    with risk_col1:
        st.markdown(f"""
        <div class="highlight">
            <strong>Value at Risk (95%)</strong><br>
            There is a 5% chance that {target_variable} will fall below ${var_95:,.0f}
        </div>
        """, unsafe_allow_html=True)
    with risk_col2:
        st.markdown(f"""
        <div class="highlight">
            <strong>Conditional VaR (95%)</strong><br>
            Average loss in worst 5% scenarios: ${cvar_95:,.0f}
        </div>
        """, unsafe_allow_html=True)

# Tab 3: Sensitivity Analysis
with tab3:
    st.header("📈 Sensitivity Analysis (Tornado Diagram)")
    
    # Run sensitivity analysis: vary one variable at a time
    sensitivity_results = []
    
    # Base case
    base_features = np.array([new_marketing, new_price, new_users, base_conversion]).reshape(1, -1)
    base_prediction = model.predict(base_features)[0]
    
    # Vary each feature by ±20%
    variation_range = np.linspace(-30, 30, 13)  # -30% to +30%
    
    for feature_idx, feature_name in enumerate(feature_cols):
        variations = []
        predictions = []
        
        for var_pct in variation_range:
            # Calculate new value
            if feature_idx == 0:  # marketing
                new_val = new_marketing * (1 + var_pct/100)
                input_vec = np.array([new_val, new_price, new_users, base_conversion])
            elif feature_idx == 1:  # price
                new_val = new_price * (1 + var_pct/100)
                input_vec = np.array([new_marketing, new_val, new_users, base_conversion])
            elif feature_idx == 2:  # users
                new_val = new_users * (1 + var_pct/100)
                input_vec = np.array([new_marketing, new_price, new_val, base_conversion])
            else:  # conversion
                new_val = base_conversion * (1 + var_pct/100)
                new_val = np.clip(new_val, 0.001, 0.5)
                input_vec = np.array([new_marketing, new_price, new_users, new_val])
            
            pred = model.predict(input_vec.reshape(1, -1))[0]
            variations.append(var_pct)
            predictions.append(pred)
        
        sensitivity_results.append({
            'feature': feature_name,
            'variations': variations,
            'predictions': predictions,
            'sensitivity': np.std(predictions)  # Measure of sensitivity
        })
    
    # Create tornado diagram data
    tornado_data = []
    for result in sensitivity_results:
        min_pred = min(result['predictions'])
        max_pred = max(result['predictions'])
        tornado_data.append({
            'Variable': result['feature'].replace('_', ' ').title(),
            'Low Impact': min_pred - base_prediction,
            'High Impact': max_pred - base_prediction,
            'Range': max_pred - min_pred,
            'Sensitivity Score': result['sensitivity']
        })
    
    tornado_df = pd.DataFrame(tornado_data)
    tornado_df = tornado_df.sort_values('Range', ascending=True)
    
    # Tornado chart
    fig_tornado = go.Figure()
    
    # Add bars for negative and positive impacts
    fig_tornado.add_trace(go.Bar(
        name='Decrease (Low)',
        y=tornado_df['Variable'],
        x=tornado_df['Low Impact'],
        orientation='h',
        marker_color='#E74C3C',
        hovertemplate='%{y}<br>Impact: $%{x:,.0f}<extra></extra>'
    ))
    
    fig_tornado.add_trace(go.Bar(
        name='Increase (High)',
        y=tornado_df['Variable'],
        x=tornado_df['High Impact'],
        orientation='h',
        marker_color='#27AE60',
        hovertemplate='%{y}<br>Impact: $%{x:,.0f}<extra></extra>'
    ))
    
    # Add center line at 0
    fig_tornado.add_vline(x=0, line_width=2, line_color="black")
    
    fig_tornado.update_layout(
        title="Tornado Diagram: Sensitivity to ±30% Changes",
        xaxis_title=f"Impact on {target_variable.replace('_', ' ').title()} ($)",
        yaxis_title="",
        barmode='overlay',
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_tornado, use_container_width=True)
    
    # Sensitivity ranking
    st.subheader("Variable Sensitivity Ranking")
    st.dataframe(
        tornado_df[['Variable', 'Range', 'Sensitivity Score']].sort_values('Range', ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            'Range': st.column_config.NumberColumn(format="$%.0f"),
            'Sensitivity Score': st.column_config.NumberColumn(format="$%.0f")
        }
    )
    
    # Spider/Radar chart for multi-dimensional view
    st.subheader("Multi-Dimensional Impact View")
    
    # Select specific variation points for radar chart
    radar_vars = ['marketing_spend', 'effective_price', 'total_new_users']
    radar_labels = [v.replace('_', ' ').title() for v in radar_vars]
    
    # Scenarios: Pessimistic, Base, Optimistic
    scenarios = {
        'Pessimistic (-20%)': -20,
        'Base Case (0%)': 0,
        'Optimistic (+20%)': 20
    }
    
    fig_radar = go.Figure()
    
    colors = ['#E74C3C', '#3498DB', '#27AE60']
    for idx, (scenario_name, var_pct) in enumerate(scenarios.items()):
        values = []
        for feature in radar_vars:
            # Find closest variation point
            closest_idx = min(range(len(variation_range)), 
                            key=lambda i: abs(variation_range[i] - var_pct))
            
            # Get feature index
            feat_idx = feature_cols.index(feature) if feature in feature_cols else -1
            if feat_idx >= 0:
                val = sensitivity_results[feat_idx]['predictions'][closest_idx]
                # Normalize to percentage change from base
                normalized = ((val - base_prediction) / base_prediction) * 100
                values.append(normalized)
            else:
                values.append(0)
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # Close the polygon
            theta=radar_labels + [radar_labels[0]],
            fill='toself',
            name=scenario_name,
            line_color=colors[idx],
            fillcolor=colors[idx],
            opacity=0.3
        ))
    
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                title="% Change from Base"
            )),
        showlegend=True,
        title="Scenario Comparison: % Impact on Target Variable",
        height=500
    )
    
    st.plotly_chart(fig_radar, use_container_width=True)

# Tab 4: Decision Insights
with tab4:
    st.header("💡 Strategic Decision Insights")
    
    # Optimization recommendations
    st.subheader("Optimization Opportunities")
    
    # Find optimal combinations using grid search simulation
    st.markdown("""
    Based on the sensitivity analysis and Monte Carlo results, here are the key insights:
    """)
    
    # Calculate elasticity-like metrics
    marketing_elasticity = (tornado_df[tornado_df['Variable'] == 'Marketing Spend']['Range'].values[0] / base_prediction) / 0.6
    price_elasticity = (tornado_df[tornado_df['Variable'] == 'Effective Price']['Range'].values[0] / base_prediction) / 0.6
    
    col1, col2 = st.columns(2)
    
    with col1:
        if marketing_elasticity > 1:
            st.success(f"""
            **📈 Marketing Leverage: HIGH ({marketing_elasticity:.2f})**
            
            Increasing marketing spend has strong positive returns. 
            Consider increasing budget by 10-20% for maximum impact.
            """)
        else:
            st.warning(f"""
            **📉 Marketing Leverage: LOW ({marketing_elasticity:.2f})**
            
            Current marketing efficiency is below threshold. 
            Optimize channels before increasing spend.
            """)
    
    with col2:
        if price_elasticity < -1:
            st.error(f"""
            **⚠️ Price Sensitivity: HIGH ({price_elasticity:.2f})**
            
            Demand is highly elastic. Price increases may significantly 
            reduce volume. Consider value-adds instead of raising prices.
            """)
        else:
            st.success(f"""
            **✅ Price Sensitivity: LOW ({price_elasticity:.2f})**
            
            You have pricing power. Consider strategic price increases 
            to improve margins without significant volume loss.
            """)
    
    # Scenario comparison table
    st.subheader("Scenario Comparison Matrix")
    
    scenarios_matrix = []
    for mkt_var in [-20, 0, 20]:
        for pr_var in [-10, 0, 10]:
            mkt_val = new_marketing * (1 + mkt_var/100)
            pr_val = new_price * (1 + pr_var/100)
            usr_val = new_users  # Keep constant for this analysis
            
            features = np.array([mkt_val, pr_val, usr_val, base_conversion]).reshape(1, -1)
            pred = model.predict(features)[0]
            
            scenarios_matrix.append({
                'Marketing Change': f"{mkt_var:+.0f}%",
                'Price Change': f"{pr_var:+.0f}%",
                'Predicted Value': f"${pred:,.0f}",
                'vs Base': f"{((pred - base_prediction)/base_prediction)*100:+.1f}%",
                'Risk Level': 'Low' if abs(mkt_var) < 10 and abs(pr_var) < 10 else 'Medium' if abs(mkt_var) < 20 else 'High'
            })
    
    scenario_df = pd.DataFrame(scenarios_matrix)
    st.dataframe(scenario_df, use_container_width=True, hide_index=True)
    
    # ROI Optimization curve
    st.subheader("Marketing Efficiency Curve")
    
    # Simulate different marketing spends
    marketing_range = np.linspace(new_marketing * 0.5, new_marketing * 2, 50)
    roi_curve = []
    
    for mkt in marketing_range:
        # Diminishing returns model
        base_pred = model.predict(np.array([mkt, new_price, new_users, base_conversion]).reshape(1, -1))[0]
        # Add diminishing returns factor
        efficiency_factor = 1 / (1 + 0.00001 * (mkt - current_marketing))
        adjusted_pred = base_pred * efficiency_factor
        
        roi_val = ((adjusted_pred - (mkt + (adjusted_pred * 0.55))) / mkt) * 100
        roi_curve.append({
            'Marketing Spend': mkt,
            'Predicted Revenue': adjusted_pred,
            'ROI': roi_val,
            'Profit': adjusted_pred - mkt - (adjusted_pred * 0.55)
        })
    
    roi_df = pd.DataFrame(roi_curve)
    
    fig_roi = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig_roi.add_trace(
        go.Scatter(x=roi_df['Marketing Spend'], y=roi_df['Predicted Revenue'],
                  name='Revenue', line=dict(color='#2E86AB')),
        secondary_y=False
    )
    
    fig_roi.add_trace(
        go.Scatter(x=roi_df['Marketing Spend'], y=roi_df['Profit'],
                  name='Profit', line=dict(color='#A23B72')),
        secondary_y=False
    )
    
    fig_roi.add_trace(
        go.Scatter(x=roi_df['Marketing Spend'], y=roi_df['ROI'],
                  name='ROI %', line=dict(color='#F18F01', dash='dash')),
        secondary_y=True
    )
    
    fig_roi.update_xaxes(title_text="Marketing Spend ($)")
    fig_roi.update_yaxes(title_text="Revenue / Profit ($)", secondary_y=False)
    fig_roi.update_yaxes(title_text="ROI (%)", secondary_y=True)
    fig_roi.update_layout(
        title="Marketing Spend Optimization: Revenue vs Profit vs ROI",
        height=500,
        hovermode='x unified'
    )
    
    # Add optimal point annotation
    optimal_idx = roi_df['Profit'].idxmax()
    optimal_spend = roi_df.loc[optimal_idx, 'Marketing Spend']
    optimal_profit = roi_df.loc[optimal_idx, 'Profit']
    
    fig_roi.add_annotation(
        x=optimal_spend, y=optimal_profit,
        text=f"Optimal Spend<br>${optimal_spend:,.0f}",
        showarrow=True,
        arrowhead=2,
        ax=40, ay=-40
    )
    
    st.plotly_chart(fig_roi, use_container_width=True)
    
    # Final recommendations
    st.subheader("🎯 Executive Summary")
    
    st.markdown(f"""
    <div class="scenario-card" style="background-color: #e8f4f8;">
        <h4>Key Findings</h4>
        <ul>
            <li><strong>Baseline {target_variable.replace('_', ' ').title()}:</strong> ${base_prediction:,.0f}</li>
            <li><strong>Expected Range ({confidence_level}% CI):</strong> ${ci_lower:,.0f} - ${ci_upper:,.0f}</li>
            <li><strong>Most Sensitive Variable:</strong> {tornado_df.iloc[-1]['Variable']}</li>
            <li><strong>Optimal Marketing Spend:</strong> ${optimal_spend:,.0f} (Current: ${new_marketing:,.0f})</li>
        </ul>
        
        <h4>Recommended Actions</h4>
        <ol>
            <li>Focus optimization efforts on <strong>{tornado_df.iloc[-1]['Variable']}</strong> for maximum impact</li>
            <li>Marketing budget should be {'increased' if optimal_spend > new_marketing else 'decreased'} to ~${optimal_spend:,.0f} for optimal profit</li>
            <li>Monitor <strong>{tornado_df.iloc[0]['Variable']}</strong> closely as it has lowest sensitivity</li>
            <li>Run Monte Carlo simulations monthly to update uncertainty ranges</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<small>
Built with Streamlit | Data: Synthetic E-commerce Dataset | Model: Linear Regression with Monte Carlo Simulation
</small>
""", unsafe_allow_html=True)
