# -------------------------------------------------------
# Program : Supervised ML Regression Models
# File    : regression_models.py
# Usage   : Called by main.py with cleaned X, y data
# -------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score)
import warnings
warnings.filterwarnings('ignore')

border      = "="*55
thin_border = "-"*55


# ======================================================
# HELPER : Train Test Split
# ======================================================

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"\n  Train Size : {X_train.shape[0]} samples")
    print(f"  Test Size  : {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test


# ======================================================
# HELPER : Display Metrics
# ======================================================

def display_metrics(model_name, y_test, y_pred):
    print(f"\n{border}")
    print(f"  RESULTS : {model_name}")
    print(f"{border}")

    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    print(f"  R2 Score : {r2   * 100:.2f}%")
    print(f"  RMSE     : {rmse:.4f}")
    print(f"  MAE      : {mae:.4f}")
    print(f"  MSE      : {mse:.4f}")

   

    # Residuals Plot
    residuals = y_test - y_pred
   
    return {
        'Model'   : model_name,
        'R2 Score': round(r2   * 100, 2),
        'RMSE'    : round(rmse, 4),
        'MAE'     : round(mae,  4),
        'MSE'     : round(mse,  4)
    }


# ======================================================
# MODEL 1 : Linear Regression
# ======================================================

def linear_regression_model(X, y):
    from sklearn.linear_model import LinearRegression

    print(f"\n{border}")
    print(f"  MODEL 1 : LINEAR REGRESSION")
    print(f"{border}")
    print("  Type    : Linear Model")
    print("  Best For: Linear relationships, Interpretable results")

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Coefficients
    coef_df = pd.DataFrame({
        'Feature'    : X.columns,
        'Coefficient': model.coef_
    }).sort_values('Coefficient', ascending=False)
    print(f"\n  Feature Coefficients:")
    print(coef_df.to_string(index=False))

    cv_scores = cross_val_score(
        model, X, y, cv=5, scoring='r2'
    )
    print(f"\n  Cross Val R2 : {cv_scores.mean()*100:.2f}% "
          f"(+/- {cv_scores.std()*100:.2f}%)")

    result = display_metrics("Linear Regression", y_test, y_pred)
    result['CV R2'] = round(cv_scores.mean() * 100, 2)
    return result


# ======================================================
# MODEL 2 : Ridge Regression
# ======================================================

def ridge_regression_model(X, y):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV

    print(f"\n{border}")
    print(f"  MODEL 2 : RIDGE REGRESSION")
    print(f"{border}")
    print("  Type    : Regularized Linear Model (L2)")
    print("  Best For: Multicollinearity, Prevents overfitting")

    X_train, X_test, y_train, y_test = split_data(X, y)

    # Find best alpha
    alphas     = [0.01, 0.1, 1.0, 10.0, 100.0]
    param_grid = {'alpha': alphas}
    grid       = GridSearchCV(Ridge(), param_grid, cv=5,
                              scoring='r2')
    grid.fit(X_train, y_train)
    best_alpha = grid.best_params_['alpha']
    print(f"\n  Best Alpha : {best_alpha}")

    model  = Ridge(alpha=best_alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cv_scores = cross_val_score(
        model, X, y, cv=5, scoring='r2'
    )
    print(f"  Cross Val R2 : {cv_scores.mean()*100:.2f}% "
          f"(+/- {cv_scores.std()*100:.2f}%)")

    result = display_metrics("Ridge Regression", y_test, y_pred)
    result['CV R2'] = round(cv_scores.mean() * 100, 2)
    return result


# ======================================================
# MODEL 3 : Lasso Regression
# ======================================================

def lasso_regression_model(X, y):
    from sklearn.linear_model import Lasso
    from sklearn.model_selection import GridSearchCV

    print(f"\n{border}")
    print(f"  MODEL 3 : LASSO REGRESSION")
    print(f"{border}")
    print("  Type    : Regularized Linear Model (L1)")
    print("  Best For: Feature selection, Sparse models")

    X_train, X_test, y_train, y_test = split_data(X, y)

    alphas     = [0.001, 0.01, 0.1, 1.0, 10.0]
    param_grid = {'alpha': alphas}
    grid       = GridSearchCV(Lasso(max_iter=10000),
                              param_grid, cv=5, scoring='r2')
    grid.fit(X_train, y_train)
    best_alpha = grid.best_params_['alpha']
    print(f"\n  Best Alpha : {best_alpha}")

    model  = Lasso(alpha=best_alpha, max_iter=10000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Features zeroed out by Lasso
    zero_features = (model.coef_ == 0).sum()
    print(f"  Features zeroed out : {zero_features} "
          f"(feature selection)")

    cv_scores = cross_val_score(
        model, X, y, cv=5, scoring='r2'
    )
    print(f"  Cross Val R2 : {cv_scores.mean()*100:.2f}% "
          f"(+/- {cv_scores.std()*100:.2f}%)")

    result = display_metrics("Lasso Regression", y_test, y_pred)
    result['CV R2'] = round(cv_scores.mean() * 100, 2)
    return result


# ======================================================
# MODEL 4 : Decision Tree Regressor
# ======================================================

def decision_tree_regressor(X, y):
    from sklearn.tree import DecisionTreeRegressor

    print(f"\n{border}")
    print(f"  MODEL 4 : DECISION TREE REGRESSOR")
    print(f"{border}")
    print("  Type    : Tree-based Model")
    print("  Best For: Non-linear data, Interpretable")

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = DecisionTreeRegressor(max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Feature importance
    importance = pd.Series(
        model.feature_importances_, index=X.columns
    ).sort_values(ascending=False)
    print(f"\n  Top 5 Important Features:")
    print(importance.head().to_string())

    cv_scores = cross_val_score(
        model, X, y, cv=5, scoring='r2'
    )
    print(f"\n  Cross Val R2 : {cv_scores.mean()*100:.2f}% "
          f"(+/- {cv_scores.std()*100:.2f}%)")

    result = display_metrics("Decision Tree Regressor",
                             y_test, y_pred)
    result['CV R2'] = round(cv_scores.mean() * 100, 2)
    return result


# ======================================================
# MODEL 5 : Random Forest Regressor
# ======================================================

def random_forest_regressor(X, y):
    from sklearn.ensemble import RandomForestRegressor

    print(f"\n{border}")
    print(f"  MODEL 5 : RANDOM FOREST REGRESSOR")
    print(f"{border}")
    print("  Type    : Ensemble — Bagging")
    print("  Best For: High accuracy, General purpose")

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = RandomForestRegressor(
        n_estimators=100, random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Feature importance plot
    importance   = pd.Series(
        model.feature_importances_, index=X.columns
    )
    top_features = importance.sort_values(ascending=False).head(10)

   
  

    cv_scores = cross_val_score(
        model, X, y, cv=5, scoring='r2'
    )
    print(f"\n  Cross Val R2 : {cv_scores.mean()*100:.2f}% "
          f"(+/- {cv_scores.std()*100:.2f}%)")

    result = display_metrics("Random Forest Regressor",
                             y_test, y_pred)
    result['CV R2'] = round(cv_scores.mean() * 100, 2)
    return result


# ======================================================
# MODEL 6 : Gradient Boosting Regressor
# ======================================================

def gradient_boosting_regressor(X, y):
    from sklearn.ensemble import GradientBoostingRegressor

    print(f"\n{border}")
    print(f"  MODEL 6 : GRADIENT BOOSTING REGRESSOR")
    print(f"{border}")
    print("  Type    : Ensemble — Boosting")
    print("  Best For: Tabular data, High accuracy")

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cv_scores = cross_val_score(
        model, X, y, cv=5, scoring='r2'
    )
    print(f"\n  Cross Val R2 : {cv_scores.mean()*100:.2f}% "
          f"(+/- {cv_scores.std()*100:.2f}%)")

    result = display_metrics("Gradient Boosting Regressor",
                             y_test, y_pred)
    result['CV R2'] = round(cv_scores.mean() * 100, 2)
    return result


# ======================================================
# MODEL 7 : XGBoost Regressor
# ======================================================

def xgboost_regressor(X, y):
    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("  XGBoost not installed. Run: pip3 install xgboost")
        return None

    print(f"\n{border}")
    print(f"  MODEL 7 : XGBOOST REGRESSOR")
    print(f"{border}")
    print("  Type    : Extreme Gradient Boosting")
    print("  Best For: Large datasets, Missing values ok")

    X_train, X_test, y_train, y_test = split_data(X, y, random_state=42 )

    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cv_scores = cross_val_score(
        model, X, y, cv=5, scoring='r2'
    )
    print(f"\n  Cross Val R2 : {cv_scores.mean()*100:.2f}% "
          f"(+/- {cv_scores.std()*100:.2f}%)")

    result = display_metrics("XGBoost Regressor",
                             y_test, y_pred)
    result['CV R2'] = round(cv_scores.mean() * 100, 2)
    return result


# ======================================================
# MODEL 8 : SVR — Support Vector Regressor
# ======================================================

def svr_model(X, y):
    from sklearn.svm import SVR

    print(f"\n{border}")
    print(f"  MODEL 8 : SUPPORT VECTOR REGRESSOR (SVR)")
    print(f"{border}")
    print("  Type    : Margin-based Regression")
    print("  Best For: Small-medium datasets, Non-linear")

    X_train, X_test, y_train, y_test = split_data(X, y , random_state=42 )

    model  = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cv_scores = cross_val_score(
        model, X, y, cv=5, scoring='r2'
    )
    print(f"\n  Cross Val R2 : {cv_scores.mean()*100:.2f}% "
          f"(+/- {cv_scores.std()*100:.2f}%)")

    result = display_metrics("SVR", y_test, y_pred)
    result['CV R2'] = round(cv_scores.mean() * 100, 2)
    return result


# ======================================================
# MODEL 9 : KNN Regressor
# ======================================================

def knn_regressor(X, y):
    from sklearn.neighbors import KNeighborsRegressor

    print(f"\n{border}")
    print(f"  MODEL 9 : KNN REGRESSOR")
    print(f"{border}")
    print("  Type    : Instance-based Regression")
    print("  Best For: Small datasets, Local patterns")

    X_train, X_test, y_train, y_test = split_data(X, y, random_state=42 )

    # Find best K
    print("\n  Finding best K...")
    k_scores = []
    k_range  = range(1, 21)

    for k in k_range:
        knn    = KNeighborsRegressor(n_neighbors=k)
        scores = cross_val_score(
            knn, X_train, y_train, cv=5, scoring='r2'
        )
        k_scores.append(scores.mean())

    best_k = k_range[k_scores.index(max(k_scores))]
    print(f"  Best K : {best_k}")

    

    model  = KNeighborsRegressor(n_neighbors=best_k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cv_scores = cross_val_score(
        model, X, y, cv=5, scoring='r2'
    )
    print(f"\n  Cross Val R2 : {cv_scores.mean()*100:.2f}% "
          f"(+/- {cv_scores.std()*100:.2f}%)")

    result = display_metrics("KNN Regressor", y_test, y_pred)
    result['CV R2'] = round(cv_scores.mean() * 100, 2)
    return result


# ======================================================
# MODEL 10 : Extra Trees Regressor
# ======================================================

def extra_trees_regressor(X, y):
    from sklearn.ensemble import ExtraTreesRegressor

    print(f"\n{border}")
    print(f"  MODEL 10 : EXTRA TREES REGRESSOR")
    print(f"{border}")
    print("  Type    : Ensemble — Extremely Randomized Trees")
    print("  Best For: Fast training, High variance datasets")

    X_train, X_test, y_train, y_test = split_data(X, y)

    model  = ExtraTreesRegressor(
        n_estimators=100, random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cv_scores = cross_val_score(
        model, X, y, cv=5, scoring='r2'
    )
    print(f"\n  Cross Val R2 : {cv_scores.mean()*100:.2f}% "
          f"(+/- {cv_scores.std()*100:.2f}%)")

    result = display_metrics("Extra Trees Regressor",
                             y_test, y_pred)
    result['CV R2'] = round(cv_scores.mean() * 100, 2)
    return result


# ======================================================
# RUN ALL REGRESSION MODELS — Called by main.py
# ======================================================

def run_all_regression_models(X, y):
    """
    Main function called by main.py
    Accepts cleaned X (features) and y (target)
    Runs all regression models and returns results
    """
    print(f"\n{border}")
    print(f"   RUNNING ALL REGRESSION MODELS")
    print(f"{border}")
    print(f"  Features : {X.shape[1]}")
    print(f"  Samples  : {X.shape[0]}")

    results = []

    models = [
        linear_regression_model,
        ridge_regression_model,
        lasso_regression_model,
        decision_tree_regressor,
        random_forest_regressor,
        gradient_boosting_regressor,
        xgboost_regressor,
        svr_model,
        knn_regressor,
        extra_trees_regressor
    ]

    for model_func in models:
        try:
            result = model_func(X, y)
            if result:
                results.append(result)
        except Exception as e:
            print(f"\n  ERROR in {model_func.__name__}: {e}")
            continue

    return results