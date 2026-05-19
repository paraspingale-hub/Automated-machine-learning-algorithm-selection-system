# -------------------------------------------------------
# Program : Supervised ML Classification Models
# File    : classification_models.py
# Usage   : Called by main.py with cleaned X, y data
# -------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             classification_report, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

border = "="*55


# ======================================================
# HELPER : Train Test Split
# ======================================================

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size,
        random_state=random_state, stratify=y
        if y.nunique() <= 20 else None
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

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"  Accuracy  : {acc  * 100:.2f}%")
    print(f"  Precision : {prec * 100:.2f}%")
    print(f"  Recall    : {rec  * 100:.2f}%")
    print(f"  F1 Score  : {f1   * 100:.2f}%")

    print(f"\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    

    return {
        'Model'    : model_name,
        'Accuracy' : round(acc  * 100, 2),
        'Precision': round(prec * 100, 2),
        'Recall'   : round(rec  * 100, 2),
        'F1 Score' : round(f1   * 100, 2)
    }


# ======================================================
# MODEL 1 : Logistic Regression
# ======================================================

def logistic_regression_model(X, y):
    from sklearn.linear_model import LogisticRegression

    print(f"\n{border}")
    print(f"  MODEL 1 : LOGISTIC REGRESSION")
    print(f"{border}")
    print("  Type    : Linear Classifier")
    print("  Best For: Binary / Multi-class, Linearly separable data")

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Cross validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"\n  Cross Val Accuracy : {cv_scores.mean()*100:.2f}% "
          f"(+/- {cv_scores.std()*100:.2f}%)")

    result = display_metrics("Logistic Regression", y_test, y_pred)
    result['CV Score'] = round(cv_scores.mean() * 100, 2)
    return result


# ======================================================
# MODEL 2 : Decision Tree Classifier
# ======================================================

def decision_tree_model(X, y):
    from sklearn.tree import DecisionTreeClassifier

    print(f"\n{border}")
    print(f"  MODEL 2 : DECISION TREE CLASSIFIER")
    print(f"{border}")
    print("  Type    : Tree-based Classifier")
    print("  Best For: Non-linear data, Interpretable results")

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = DecisionTreeClassifier(random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Feature importance
    importance = pd.Series(model.feature_importances_, index=X.columns)
    print(f"\n  Top 5 Important Features:")
    print(importance.sort_values(ascending=False).head())

    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"\n  Cross Val Accuracy : {cv_scores.mean()*100:.2f}% "
          f"(+/- {cv_scores.std()*100:.2f}%)")

    result = display_metrics("Decision Tree", y_test, y_pred)
    result['CV Score'] = round(cv_scores.mean() * 100, 2)
    return result


# ======================================================
# MODEL 3 : Random Forest Classifier
# ======================================================

def random_forest_model(X, y):
    from sklearn.ensemble import RandomForestClassifier

    print(f"\n{border}")
    print(f"  MODEL 3 : RANDOM FOREST CLASSIFIER")
    print(f"{border}")
    print("  Type    : Ensemble — Bagging")
    print("  Best For: High accuracy, Handles missing values well")

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Feature importance plot
    importance = pd.Series(model.feature_importances_, index=X.columns)
    top_features = importance.sort_values(ascending=False).head(10)

   

    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"\n  Cross Val Accuracy : {cv_scores.mean()*100:.2f}% "
          f"(+/- {cv_scores.std()*100:.2f}%)")

    result = display_metrics("Random Forest", y_test, y_pred)
    result['CV Score'] = round(cv_scores.mean() * 100, 2)
    return result


# ======================================================
# MODEL 4 : Support Vector Machine (SVM)
# ======================================================

def svm_model(X, y):
    from sklearn.svm import SVC

    print(f"\n{border}")
    print(f"  MODEL 4 : SUPPORT VECTOR MACHINE (SVM)")
    print(f"{border}")
    print("  Type    : Margin-based Classifier")
    print("  Best For: High dimensional data, Small-medium datasets")

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = SVC(kernel='rbf', random_state=42, probability=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"\n  Cross Val Accuracy : {cv_scores.mean()*100:.2f}% "
          f"(+/- {cv_scores.std()*100:.2f}%)")

    result = display_metrics("SVM", y_test, y_pred)
    result['CV Score'] = round(cv_scores.mean() * 100, 2)
    return result


# ======================================================
# MODEL 5 : K-Nearest Neighbors (KNN)
# ======================================================

def knn_model(X, y):
    from sklearn.neighbors import KNeighborsClassifier

    print(f"\n{border}")
    print(f"  MODEL 5 : K-NEAREST NEIGHBORS (KNN)")
    print(f"{border}")
    print("  Type    : Instance-based Classifier")
    print("  Best For: Small datasets, Non-linear boundaries")

    X_train, X_test, y_train, y_test = split_data(X, y)

    # Find best K
    print("\n  Finding best K value...")
    k_scores = []
    k_range  = range(1, 21)

    for k in k_range:
        knn    = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
        k_scores.append(scores.mean())

    best_k = k_range[k_scores.index(max(k_scores))]
    print(f"  Best K : {best_k}")

    

    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"\n  Cross Val Accuracy : {cv_scores.mean()*100:.2f}% "
          f"(+/- {cv_scores.std()*100:.2f}%)")

    result = display_metrics("KNN", y_test, y_pred)
    result['CV Score'] = round(cv_scores.mean() * 100, 2)
    return result


# ======================================================
# MODEL 6 : Naive Bayes
# ======================================================

def naive_bayes_model(X, y):
    from sklearn.naive_bayes import GaussianNB

    print(f"\n{border}")
    print(f"  MODEL 6 : NAIVE BAYES (Gaussian)")
    print(f"{border}")
    print("  Type    : Probabilistic Classifier")
    print("  Best For: Text classification, Fast training, Small data")

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"\n  Cross Val Accuracy : {cv_scores.mean()*100:.2f}% "
          f"(+/- {cv_scores.std()*100:.2f}%)")

    result = display_metrics("Naive Bayes", y_test, y_pred)
    result['CV Score'] = round(cv_scores.mean() * 100, 2)
    return result


# ======================================================
# MODEL 7 : Gradient Boosting Classifier
# ======================================================

def gradient_boosting_model(X, y):
    from sklearn.ensemble import GradientBoostingClassifier

    print(f"\n{border}")
    print(f"  MODEL 7 : GRADIENT BOOSTING CLASSIFIER")
    print(f"{border}")
    print("  Type    : Ensemble — Boosting")
    print("  Best For: Tabular data, High accuracy competitions")

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"\n  Cross Val Accuracy : {cv_scores.mean()*100:.2f}% "
          f"(+/- {cv_scores.std()*100:.2f}%)")

    result = display_metrics("Gradient Boosting", y_test, y_pred)
    result['CV Score'] = round(cv_scores.mean() * 100, 2)
    return result


# ======================================================
# MODEL 8 : XGBoost Classifier
# ======================================================

def xgboost_model(X, y):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("  XGBoost not installed. Run: pip3 install xgboost")
        return None

    print(f"\n{border}")
    print(f"  MODEL 8 : XGBOOST CLASSIFIER")
    print(f"{border}")
    print("  Type    : Ensemble — Extreme Gradient Boosting")
    print("  Best For: Large datasets, Handles nulls natively")

    X_train, X_test, y_train, y_test = split_data(X, y)

    # Encode y if needed
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)

    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        eval_metric='mlogloss',
        verbosity=0
    )
    model.fit(X_train, y_train_enc)
    y_pred_enc = model.predict(X_test)
    y_pred     = le.inverse_transform(y_pred_enc)

    cv_scores = cross_val_score(model, X, le.transform(y), cv=5, scoring='accuracy')
    print(f"\n  Cross Val Accuracy : {cv_scores.mean()*100:.2f}% "
          f"(+/- {cv_scores.std()*100:.2f}%)")

    result = display_metrics("XGBoost", y_test, y_pred)
    result['CV Score'] = round(cv_scores.mean() * 100, 2)
    return result


# ======================================================
# MODEL 9 : AdaBoost Classifier
# ======================================================

def adaboost_model(X, y):
    from sklearn.ensemble import AdaBoostClassifier

    print(f"\n{border}")
    print(f"  MODEL 9 : ADABOOST CLASSIFIER")
    print(f"{border}")
    print("  Type    : Ensemble — Adaptive Boosting")
    print("  Best For: Binary classification, Weak learner combination")

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = AdaBoostClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"\n  Cross Val Accuracy : {cv_scores.mean()*100:.2f}% "
          f"(+/- {cv_scores.std()*100:.2f}%)")

    result = display_metrics("AdaBoost", y_test, y_pred)
    result['CV Score'] = round(cv_scores.mean() * 100, 2)
    return result


# ======================================================
# MODEL 10 : Extra Trees Classifier
# ======================================================

def extra_trees_model(X, y):
    from sklearn.ensemble import ExtraTreesClassifier

    print(f"\n{border}")
    print(f"  MODEL 10 : EXTRA TREES CLASSIFIER")
    print(f"{border}")
    print("  Type    : Ensemble — Extremely Randomized Trees")
    print("  Best For: Fast training, High variance datasets")

    X_train, X_test, y_train, y_test = split_data(X, y)

    model = ExtraTreesClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"\n  Cross Val Accuracy : {cv_scores.mean()*100:.2f}% "
          f"(+/- {cv_scores.std()*100:.2f}%)")

    result = display_metrics("Extra Trees", y_test, y_pred)
    result['CV Score'] = round(cv_scores.mean() * 100, 2)
    return result


# ======================================================
# RUN ALL MODELS — Called by main.py
# ======================================================

def run_all_classification_models(X, y, selected_models=None):
    print(f"\n{border}\n   RUNNING CLASSIFICATION MODELS\n{border}")
    
    # Map frontend string names to the python functions
    model_map = {
        "Logistic Regression": logistic_regression_model,
        "Decision Tree": decision_tree_model,
        "Random Forest": random_forest_model,
        "SVM": svm_model,
        "KNN": knn_model,
        "Naive Bayes": naive_bayes_model,
        "Gradient Boosting": gradient_boosting_model,
        "XGBoost": xgboost_model,
        "AdaBoost": adaboost_model,
        "Extra Trees": extra_trees_model
    }

    # Filter models based on user selection
    models_to_run = []
    if selected_models:
        models_to_run = [model_map[name] for name in selected_models if name in model_map]
    else:
        models_to_run = list(model_map.values())

    results = []
    for model_func in models_to_run:
        try:
            result = model_func(X, y)
            if result: results.append(result)
        except Exception as e:
            print(f"  ERROR in {model_func.__name__}: {e}")
            continue

    return results

    