import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

def test_models():
    # Generate some fake data to test model fitting
    X = pd.DataFrame(np.random.rand(100, 8), columns=[
        'marketing_spend', 'effective_price', 'total_users', 'conversion_rate',
        'digital_spend', 'social_spend', 'discount_depth', 'organic_users'
    ])
    y = np.random.rand(100)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            results[name] = "Success"
        except Exception as e:
            results[name] = f"Error: {e}"
            
    return results

if __name__ == "__main__":
    print(test_models())
