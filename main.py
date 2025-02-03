#!/usr/bin/env python3
"""
House Price Prediction and Valuation using the California Housing Dataset with Advanced Techniques

This project loads the California Housing dataset from scikit-learn, performs data exploration,
applies feature engineering (including polynomial features and scaling), and builds and compares multiple
multivariable regression models (Linear Regression, Ridge, and Lasso) using Grid Search with cross-validation.
It evaluates model performance using metrics, visualizes predictions vs. actual values, and performs residual analysis.

Requirements:
    - Python 3.x
    - pandas, numpy, scikit-learn, matplotlib, seaborn

Usage:
    python house_price_prediction_real_extended.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, make_scorer


def load_data():
    """
    Load the California Housing dataset and return a DataFrame with features and target.

    Returns:
        df (pd.DataFrame): DataFrame containing features and the target variable.
    """
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    df.rename(columns={"MedHouseVal": "HouseValue"}, inplace=True)
    # HouseValue is in units of 100,000 USD.
    return df


def explore_data(df):
    """
    Perform data exploration and visualization.

    Parameters:
        df (pd.DataFrame): The California Housing dataset.
    """
    print("First five rows of the dataset:")
    print(df.head(), "\n")

    print("Dataset description:")
    print(df.describe(), "\n")

    # Correlation matrix
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()

    # Pairplot of a subset of features and the target
    subset = df[['MedInc', 'HouseAge', 'AveRooms', 'HouseValue']]
    sns.pairplot(subset, diag_kind='kde')
    plt.show()


def build_model_pipeline(model, poly_degree=1):
    """
    Build a pipeline that performs polynomial feature expansion, scaling, and then regression.

    Parameters:
        model: The regression model instance (e.g., LinearRegression, Ridge, or Lasso)
        poly_degree (int): Degree of polynomial features to generate.

    Returns:
        pipeline: A scikit-learn Pipeline object.
    """
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=poly_degree, include_bias=False)),
        ('scaler', StandardScaler()),
        ('regressor', model)
    ])
    return pipeline


def grid_search_model(pipeline, param_grid, X_train, y_train, cv=5):
    """
    Perform grid search cross-validation on the pipeline.

    Parameters:
        pipeline: The scikit-learn Pipeline to tune.
        param_grid (dict): Dictionary with parameters names and lists of values to try.
        X_train (pd.DataFrame): Training feature set.
        y_train (pd.Series): Training target values.
        cv (int): Number of cross-validation folds.

    Returns:
        grid: The fitted GridSearchCV object.
    """
    mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
    grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring=mse_scorer, n_jobs=-1)
    grid.fit(X_train, y_train)
    print("Best parameters found:", grid.best_params_)
    print("Best CV MSE:", -grid.best_score_)
    return grid


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate the model on test data and produce evaluation metrics and residual plots.

    Parameters:
        model: The trained regression model.
        X_test (pd.DataFrame): Test feature set.
        y_test (pd.Series): True target values.
        model_name (str): A label for the model.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\nEvaluation for {model_name}:")
    print(f"Mean Squared Error (MSE): {mse:.3f}")
    print(f"R² Score: {r2:.3f}")

    # Plot Actual vs Predicted Values
    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual HouseValue (100k USD)")
    plt.ylabel("Predicted HouseValue (100k USD)")
    plt.title(f"Actual vs Predicted ({model_name})")
    plt.grid(True)
    plt.show()

    # Residual analysis
    residuals = y_test - y_pred
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, kde=True)
    plt.xlabel("Residuals")
    plt.title(f"Residuals Distribution ({model_name})")
    plt.grid(True)
    plt.show()

    return y_pred


def main():
    # Step 1: Load the data
    df = load_data()

    # Step 2: Explore the data
    explore_data(df)

    # Define features and target
    X = df.drop(columns=["HouseValue"])
    y = df["HouseValue"]

    # Step 3: Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Define model pipelines
    # We'll compare three models: Linear Regression, Ridge, and Lasso with polynomial features.
    # Set poly_degree to 2 to capture non-linear relationships.
    poly_degree = 2

    # Note: Updated Lasso model with increased max_iter and adjusted tolerance to reduce convergence warnings.
    lasso_model = Lasso(max_iter=50000, tol=1e-4)

    pipelines = {
        "Linear": build_model_pipeline(LinearRegression(), poly_degree=poly_degree),
        "Ridge": build_model_pipeline(Ridge(), poly_degree=poly_degree),
        "Lasso": build_model_pipeline(lasso_model, poly_degree=poly_degree)
    }

    # Define parameter grids for Ridge and Lasso
    param_grids = {
        "Ridge": {
            'regressor__alpha': [0.1, 1, 10, 100],
            'poly__degree': [1, 2, 3]
        },
        "Lasso": {
            'regressor__alpha': [0.001, 0.01, 0.1, 1],
            'poly__degree': [1, 2, 3]
        }
    }

    best_models = {}

    # Step 5: For Linear Regression, no hyperparameter tuning is needed:
    print("Training Linear Regression Pipeline...")
    pipelines["Linear"].fit(X_train, y_train)
    best_models["Linear"] = pipelines["Linear"]

    # Step 6: Perform Grid Search for Ridge and Lasso
    for model_name in ["Ridge", "Lasso"]:
        print(f"\nTuning {model_name} Regression Pipeline with Grid Search...")
        grid = grid_search_model(pipelines[model_name], param_grids[model_name], X_train, y_train, cv=5)
        best_models[model_name] = grid.best_estimator_

    # Step 7: Evaluate models on the test set
    for name, model in best_models.items():
        evaluate_model(model, X_test, y_test, model_name=name)

    # Step 8: Predict the price for a new house as an example of valuation
    # These feature values are chosen based on realistic ranges in the dataset.
    new_house = {
        'MedInc': 8.0,  # Median income (in tens of thousands)
        'HouseAge': 20,  # Median house age
        'AveRooms': 6.0,  # Average number of rooms
        'AveBedrms': 1.0,  # Average number of bedrooms
        'Population': 1000,  # Population in block group
        'AveOccup': 3.0,  # Average occupants per household
        'Latitude': 34.0,  # Latitude
        'Longitude': -118.0  # Longitude
    }

    # Use the best performing model for prediction (here, we select Ridge as an example)
    selected_model = best_models["Ridge"]
    predicted_value = selected_model.predict(pd.DataFrame([new_house]))[0]
    print("\nNew House Valuation:")
    print("Features:")
    for k, v in new_house.items():
        print(f"  {k}: {v}")
    print(f"\nPredicted House Value: ${predicted_value * 100000:,.2f}")  # Convert from 100k USD to USD


if __name__ == "__main__":
    main()
