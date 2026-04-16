# -------------------------------------------------------
# Program : Smart ML Pipeline — Main Entry Point
# File    : main.py
# Pipeline:
#   1. Load Dataset
#   2. Detect ML Type (Supervised / Unsupervised)
#   3. Confirm with User
#   4. Perform EDA
#   5. Clean Dataset
#   6. Run Models
#   7. Display Final Comparison
# -------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
import sys
import os
import warnings


border     = "="*55
thin_border = "-"*55


# ======================================================
# STEP 1 : LOAD DATASET
# ======================================================

def load_dataset():
    print(f"\n{border}")
    print(f"         STEP 1 : LOAD DATASET")
    print(f"{border}")

    if len(sys.argv) > 1:
        path = sys.argv[1].strip()
    else:
        path = input("\n  Enter path to your dataset: ").strip()

    if not os.path.exists(path):
        print(f"\n  ERROR: File not found at '{path}'")
        return None

    try:
        if path.endswith(".csv"):
            df = pd.read_csv(path)
        elif path.endswith(".xlsx") or path.endswith(".xls"):
            df = pd.read_excel(path)
        elif path.endswith(".json"):
            df = pd.read_json(path)
        else:
            print("  ERROR: Unsupported file type! Use .csv / .xlsx / .json")
            return None
    except Exception as e:
        print(f"  ERROR loading file: {e}")
        return None

    print(f"\n  File loaded successfully!")
    print(f"  Path    : {path}")
    print(f"  Rows    : {df.shape[0]}")
    print(f"  Columns : {df.shape[1]}")
    print(f"\n  Columns : {df.columns.tolist()}")
    print(f"\n  Preview :")
    print(df.head())

    return df


# ======================================================
# STEP 2 : DETECT ML TYPE
# ======================================================

def detect_ml_type(df):
    print(f"\n{border}")
    print(f"    STEP 2 : DETECTING ML TYPE")
    print(f"{border}")

    score      = 0
    reasons    = []
    target_col = None

    # Check 1: Common target column names
    common_names = [
        'target', 'label', 'output', 'class', 'y',
        'result', 'outcome', 'prediction', 'response',
        'category', 'diagnosis', 'survived', 'price',
        'salary', 'grade', 'status', 'species'
    ]
    for col in df.columns:
        if col.lower() in common_names:
            score      += 3
            target_col  = col
            reasons.append(
                f"  [+3] Column '{col}' matches a known target name"
            )
            break

    # Check 2: Last column has low unique values
    last_col     = df.columns[-1]
    unique_ratio = df[last_col].nunique() / len(df)
    if unique_ratio < 0.05 or df[last_col].nunique() <= 20:
        score      += 2
        target_col  = target_col or last_col
        reasons.append(
            f"  [+2] Last column '{last_col}' has "
            f"{df[last_col].nunique()} unique values — likely target"
        )

    # Check 3: Binary column exists
    for col in df.columns:
        vals = df[col].dropna().unique()
        if len(vals) == 2:
            score      += 2
            target_col  = target_col or col
            reasons.append(
                f"  [+2] Column '{col}' is binary {vals} — "
                f"classification signal"
            )
            break

    # Check 4: Low cardinality column
    for col in df.columns:
        if 2 < df[col].nunique() <= 10:
            score      += 1
            target_col  = target_col or col
            reasons.append(
                f"  [+1] Column '{col}' has {df[col].nunique()} "
                f"unique values — possible class labels"
            )
            break

    # Check 5: All numeric — unsupervised signal
    all_numeric = all(df[col].dtype != 'object' for col in df.columns)
    if all_numeric and score == 0:
        score -= 2
        reasons.append(
            "  [-2] All columns numeric, no obvious target"
        )

    # Check 6: No low cardinality — unsupervised signal
    low_card = [c for c in df.columns if df[c].nunique() <= 20]
    if not low_card:
        score -= 1
        reasons.append(
            "  [-1] No low-cardinality column found"
        )

    print(f"\n  Detection Findings:")
    for r in reasons:
        print(r)

    print(f"\n  Final Score : {score}")

    ml_type = "SUPERVISED" if score >= 2 else "UNSUPERVISED"

    print(f"\n  Detected ML Type : {ml_type}")
    if ml_type == "SUPERVISED" and target_col:
        print(f"  Suggested Target : '{target_col}'")

    return ml_type, target_col


# ======================================================
# STEP 3 : CONFIRM WITH USER
# ======================================================

def confirm_with_user(df, ml_type, target_col):
    print(f"\n{border}")
    print(f"       STEP 3 : USER CONFIRMATION")
    print(f"{border}")

    print(f"\n  Detected Type    : {ml_type}")
    if target_col:
        print(f"  Suggested Target : '{target_col}'")
    print(f"  All Columns      : {df.columns.tolist()}")

    print(f"\n  Options:")
    print(f"  1. Yes — Confirmed, proceed")
    print(f"  2. No  — Change to SUPERVISED (I will pick target)")
    print(f"  3. No  — Change to UNSUPERVISED (no target)")

    choice = input("\n  Enter choice (1 / 2 / 3) : ").strip()

    if choice == "1":
        if ml_type == "SUPERVISED":
            confirm = input(
                f"\n  Target is '{target_col}'. "
                f"Press Enter to confirm or type new name: "
            ).strip().strip("'\"")
            if confirm != "":
                target_col = confirm

            # Validate
            while target_col not in df.columns:
                print(f"\n  '{target_col}' not found!")
                print(f"  Columns: {df.columns.tolist()}")
                target_col = input(
                    "  Enter valid column name: "
                ).strip().strip("'\"")

        print(f"\n  Confirmed as : {ml_type}")

    elif choice == "2":
        ml_type    = "SUPERVISED"
        print(f"\n  Columns: {df.columns.tolist()}")
        target_col = input(
            "  Enter target column name: "
        ).strip().strip("'\"")

        while target_col not in df.columns:
            print(f"\n  '{target_col}' not found!")
            print(f"  Columns: {df.columns.tolist()}")
            target_col = input(
                "  Enter valid column name: "
            ).strip().strip("'\"")

        print(f"\n  Set to SUPERVISED, target = '{target_col}'")

    elif choice == "3":
        ml_type    = "UNSUPERVISED"
        target_col = None
        print(f"\n  Set to UNSUPERVISED")

    else:
        print("  Invalid choice — keeping original detection")

    return ml_type, target_col


# ======================================================
# STEP 4 : EDA
# ======================================================

def run_eda(df, ml_type, target_col):
    print(f"\n{border}")
    print(f"    STEP 4 : EXPLORATORY DATA ANALYSIS")
    print(f"{border}")

    # Basic Info
    print(f"\n{thin_border}")
    print(f"  Basic Information")
    print(f"{thin_border}")
    print(f"  Rows                 : {df.shape[0]}")
    print(f"  Columns              : {df.shape[1]}")
    print(f"  Duplicate Rows       : {df.duplicated().sum()}")
    print(f"  Total Missing Values : {df.isnull().sum().sum()}")
    print(f"  Memory Usage         : "
          f"{df.memory_usage().sum() / 1024:.2f} KB")
    print(f"\n  Data Types:\n{df.dtypes}")

    # Missing Values
    print(f"\n{thin_border}")
    print(f"  Missing Value Report")
    print(f"{thin_border}")
    missing     = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    report      = pd.DataFrame({
        'Count'  : missing,
        'Percent': missing_pct
    }).sort_values('Percent', ascending=False)
    report = report[report['Count'] > 0]

    if report.empty:
        print("  No missing values found!")
    else:
        print(report)
        for col, pct in missing_pct.items():
            if pct > 50:
                print(f"  CRITICAL : '{col}' {pct:.1f}% → Drop column")
            elif pct > 20:
                print(f"  HIGH     : '{col}' {pct:.1f}% → Impute")
            elif pct > 0:
                print(f"  LOW      : '{col}' {pct:.1f}% → Easy fix")



    # Statistical Summary
    print(f"\n{thin_border}")
    print(f"  Statistical Summary")
    print(f"{thin_border}")
    numeric_df = df.select_dtypes(include=['number'])
    if not numeric_df.empty:
        print(numeric_df.describe().round(4))

    # Target Analysis (Supervised only)
    problem_type = None
    if ml_type == "SUPERVISED" and target_col:
        print(f"\n{thin_border}")
        print(f"  Target Column Analysis : '{target_col}'")
        print(f"{thin_border}")

        if (df[target_col].dtype == 'object' or
                df[target_col].nunique() <= 20):
            problem_type = "CLASSIFICATION"
        else:
            problem_type = "REGRESSION"

        print(f"  Problem Type  : {problem_type}")
        print(f"  Unique Values : {df[target_col].nunique()}")
        print(f"  Missing       : {df[target_col].isnull().sum()}")

        if problem_type == "CLASSIFICATION":
            counts  = df[target_col].value_counts()
            percent = df[target_col].value_counts(normalize=True) * 100
            print(f"\n  Class Distribution:")
            for cls in counts.index:
                print(f"    '{cls}' : {counts[cls]} "
                      f"({percent[cls]:.1f}%)")

            ratio = counts.max() / counts.min()
            if ratio > 10:
                print(f"\n  WARNING: Severe imbalance! {ratio:.1f}:1")
            elif ratio > 3:
                print(f"\n  CAUTION: Moderate imbalance {ratio:.1f}:1")
            else:
                print(f"\n  Classes balanced {ratio:.1f}:1")

          

        else:
            print(f"  Min    : {df[target_col].min():.4f}")
            print(f"  Max    : {df[target_col].max():.4f}")
            print(f"  Mean   : {df[target_col].mean():.4f}")
            print(f"  Std    : {df[target_col].std():.4f}")

            

    # Feature Distributions
    print(f"\n{thin_border}")
    print(f"  Feature Distributions")
    print(f"{thin_border}")
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    if target_col and target_col in num_cols:
        num_cols.remove(target_col)

    if num_cols:
        rows      = (len(num_cols) + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(15, rows * 4))
        axes      = axes.flatten()
        for i, col in enumerate(num_cols):
            sns.histplot(df[col], ax=axes[i], kde=True,
                         color='steelblue')
            axes[i].set_title(col, fontsize=10)
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
       
        # Skewness
        print(f"\n  Skewness Report:")
        for col in num_cols:
            skew = df[col].skew()
            tag  = ("Highly skewed" if abs(skew) > 1
                    else "Moderate" if abs(skew) > 0.5
                    else "Normal")
            print(f"    '{col}' : {skew:.2f} — {tag}")

    # Correlation Heatmap
    print(f"\n{thin_border}")
    print(f"  Correlation Analysis")
    print(f"{thin_border}")
    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr()
        

    return problem_type


# ======================================================
# STEP 5 : CLEAN DATASET
# ======================================================

def clean_dataset(df, target_col=None):
    print(f"\n{border}")
    print(f"       STEP 5 : CLEANING DATASET")
    print(f"{border}")

    df_clean = df.copy()

    # Remove duplicates
    before   = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    print(f"\n  Duplicates removed  : {before - len(df_clean)}")

    # Drop columns with >50% missing
    for col in df_clean.columns:
        pct = df_clean[col].isnull().sum() / len(df_clean) * 100
        if pct > 50:
            df_clean.drop(columns=[col], inplace=True)
            print(f"  Dropped '{col}' ({pct:.1f}% missing)")

    # Handle missing values
    for col in df_clean.columns:
        missing = df_clean[col].isnull().sum()
        if missing == 0:
            continue
        if col == target_col:
            df_clean.dropna(subset=[col], inplace=True)
            print(f"  Dropped {missing} rows — missing target '{col}'")
        elif df_clean[col].dtype == 'object':
            mode = df_clean[col].mode()[0]
            df_clean[col].fillna(mode, inplace=True)
            print(f"  '{col}' — filled {missing} nulls with mode")
        else:
            median = df_clean[col].median()
            df_clean[col].fillna(median, inplace=True)
            print(f"  '{col}' — filled {missing} nulls with median")

    # Fix data types
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object' and col != target_col:
            try:
                df_clean[col] = pd.to_numeric(df_clean[col])
                print(f"  '{col}' converted to numeric")
            except:
                pass

    # Encode categoricals
    le = LabelEncoder()
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = le.fit_transform(
                df_clean[col].astype(str)
            )
            print(f"  '{col}' label encoded")

    # Remove outliers (features only, not target)
    num_cols = df_clean.select_dtypes(
        include=['number']
    ).columns.tolist()
    if target_col and target_col in num_cols:
        num_cols.remove(target_col)

    before = len(df_clean)
    for col in num_cols:
        Q1  = df_clean[col].quantile(0.25)
        Q3  = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        df_clean = df_clean[
            (df_clean[col] >= Q1 - 1.5 * IQR) &
            (df_clean[col] <= Q3 + 1.5 * IQR)
        ]
    print(f"  Outlier rows removed : {before - len(df_clean)}")

    # Split X and y
    if target_col:
        X = df_clean.drop(columns=[target_col])
        y = df_clean[target_col].reset_index(drop=True)
    else:
        X = df_clean.copy()
        y = None

    # Scale features
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X        = pd.DataFrame(X_scaled, columns=X.columns)

    print(f"\n  Cleaning complete!")
    print(f"  Final X shape : {X.shape}")
    if y is not None:
        print(f"  Final y shape : {y.shape}")

    return X, y


# ======================================================
# STEP 6A : STRATIFIED K-FOLD CROSS VALIDATION SETUP
# ======================================================

def create_stratified_folds(X, y=None, n_splits=5):
    print(f"\n{border}")
    print(f"  STEP 6A : STRATIFIED K-FOLD SETUP")
    print(f"{border}")

    from sklearn.model_selection import (StratifiedKFold, KFold)

    print(f"\n  Total Samples   : {len(X)}")
    print(f"  Total Features  : {X.shape[1]}")
    print(f"  Number of Folds : {n_splits}")

    if y is not None:
        # Check if classification or regression
        if y.nunique() <= 20:
            # Classification — use StratifiedKFold
            # preserves class % in each fold
            kf = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=42
            )
            splits    = list(kf.split(X, y))
            fold_type = "Stratified K-Fold"

            # Show class distribution per fold
            print(f"\n  Fold Type : {fold_type}")
            print(f"  Class distribution preserved across folds:")
            for fold_idx, (train_idx, val_idx) in enumerate(splits):
                y_fold = y.iloc[val_idx]
                dist   = y_fold.value_counts(normalize=True) * 100
                print(f"\n    Fold {fold_idx+1} "
                      f"({len(val_idx)} samples):")
                for cls, pct in dist.items():
                    print(f"      Class '{cls}' : {pct:.1f}%")

        else:
            # Regression — use regular KFold
            kf = KFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=42
            )
            splits    = list(kf.split(X, y))
            fold_type = "K-Fold"

            print(f"\n  Fold Type : {fold_type}")
            print(f"  (Regression — standard KFold used)")
            for fold_idx, (train_idx, val_idx) in enumerate(splits):
                print(f"\n    Fold {fold_idx+1} : "
                      f"Train={len(train_idx)}, "
                      f"Val={len(val_idx)}")

    else:
        # Unsupervised — regular KFold
        kf = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42
        )
        splits    = list(kf.split(X))
        fold_type = "K-Fold (Unsupervised)"

        print(f"\n  Fold Type : {fold_type}")
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            print(f"    Fold {fold_idx+1} : "
                  f"Train={len(train_idx)}, "
                  f"Val={len(val_idx)}")

    print(f"\n  Each fold size   : "
          f"~{len(X) // n_splits} samples")
    print(f"  Train per fold   : "
          f"~{len(X) - len(X)//n_splits} samples")
    print(f"  Val per fold     : "
          f"~{len(X) // n_splits} samples")

    return splits, kf, fold_type


# ======================================================
# STEP 6B : EVALUATE ALL MODELS USING K-FOLD
# ======================================================

def evaluate_models_kfold(X, y, splits, ml_type,
                           problem_type):
    print(f"\n{border}")
    print(f"  STEP 6B : EVALUATING MODELS USING K-FOLD")
    print(f"{border}")

    import numpy as np
    from sklearn.metrics import (accuracy_score, f1_score,
                                 r2_score, mean_squared_error)

    # Define all models based on problem type
    if ml_type == "SUPERVISED":
        if problem_type == "CLASSIFICATION":
            from sklearn.linear_model  import LogisticRegression
            from sklearn.tree          import DecisionTreeClassifier
            from sklearn.ensemble      import (
                RandomForestClassifier,
                GradientBoostingClassifier,
                AdaBoostClassifier,
                ExtraTreesClassifier
            )
            from sklearn.svm           import SVC
            from sklearn.neighbors     import KNeighborsClassifier
            from sklearn.naive_bayes   import GaussianNB

            models = {
                'Logistic Regression' : LogisticRegression(
                    max_iter=1000, random_state=42),
                'Decision Tree'       : DecisionTreeClassifier(
                    max_depth=10, random_state=42),
                'Random Forest'       : RandomForestClassifier(
                    n_estimators=100, random_state=42),
                'SVM'                 : SVC(
                    kernel='rbf', random_state=42),
                'KNN'                 : KNeighborsClassifier(
                    n_neighbors=5),
                'Naive Bayes'         : GaussianNB(),
                'Gradient Boosting'   : GradientBoostingClassifier(
                    n_estimators=100, random_state=42),
                'AdaBoost'            : AdaBoostClassifier(
                    n_estimators=100, random_state=42),
                'Extra Trees'         : ExtraTreesClassifier(
                    n_estimators=100, random_state=42),
            }

            try:
                from xgboost import XGBClassifier
                models['XGBoost'] = XGBClassifier(
                    n_estimators=100, random_state=42,
                    eval_metric='mlogloss', verbosity=0
                )
            except ImportError:
                print("  XGBoost not found — skipping")

            metric_name = "Accuracy"

        else:  # REGRESSION
            from sklearn.linear_model  import (LinearRegression,
                                               Ridge, Lasso)
            from sklearn.tree          import DecisionTreeRegressor
            from sklearn.ensemble      import (
                RandomForestRegressor,
                GradientBoostingRegressor,
                ExtraTreesRegressor
            )
            from sklearn.svm           import SVR
            from sklearn.neighbors     import KNeighborsRegressor

            models = {
                'Linear Regression'          : LinearRegression(),
                'Ridge Regression'           : Ridge(alpha=1.0),
                'Lasso Regression'           : Lasso(
                    alpha=0.01, max_iter=10000),
                'Decision Tree Regressor'    : DecisionTreeRegressor(
                    max_depth=10, random_state=42),
                'Random Forest Regressor'    : RandomForestRegressor(
                    n_estimators=100, random_state=42),
                'Gradient Boosting Regressor': GradientBoostingRegressor(
                    n_estimators=100, random_state=42),
                'SVR'                        : SVR(kernel='rbf'),
                'KNN Regressor'              : KNeighborsRegressor(
                    n_neighbors=5),
                'Extra Trees Regressor'      : ExtraTreesRegressor(
                    n_estimators=100, random_state=42),
            }

            try:
                from xgboost import XGBRegressor
                models['XGBoost Regressor'] = XGBRegressor(
                    n_estimators=100, random_state=42,
                    verbosity=0
                )
            except ImportError:
                print("  XGBoost not found — skipping")

            metric_name = "R2 Score"

    # Store results per model
    all_results = {}

    X_arr = X.values
    y_arr = y.values if y is not None else None

    for model_name, model in models.items():
        print(f"\n  Testing : {model_name}")
        print(f"  {thin_border}")

        fold_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            X_train = X_arr[train_idx]
            X_val   = X_arr[val_idx]
            y_train = y_arr[train_idx]
            y_val   = y_arr[val_idx]

            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)

                if problem_type == "CLASSIFICATION":
                    score = accuracy_score(y_val, y_pred) * 100
                else:
                    score = r2_score(y_val, y_pred) * 100

                fold_scores.append(score)
                print(f"    Fold {fold_idx+1} : "
                      f"{metric_name} = {score:.2f}%")

            except Exception as e:
                print(f"    Fold {fold_idx+1} : ERROR — {e}")
                fold_scores.append(0.0)

        mean_score = np.mean(fold_scores)
        std_score  = np.std(fold_scores)

        print(f"\n    Mean {metric_name} : {mean_score:.2f}%")
        print(f"    Std Dev          : {std_score:.2f}%")

        all_results[model_name] = {
            'Model'            : model_name,
            metric_name        : round(mean_score, 2),
            'Std Dev'          : round(std_score,  2),
            'Fold Scores'      : fold_scores,
            'Best Fold Score'  : round(max(fold_scores), 2),
            'Worst Fold Score' : round(min(fold_scores), 2)
        }

    return all_results, metric_name

# ======================================================
# STEP 6C : VISUALIZE K-FOLD RESULTS
# ======================================================

def visualize_kfold_results(all_results, metric_name):
    print(f"\n{border}")
    print(f"  STEP 6C : K-FOLD RESULTS VISUALIZATION")
    print(f"{border}")

    import numpy as np

    models      = list(all_results.keys())
    mean_scores = [all_results[m][metric_name] for m in models]
    std_scores  = [all_results[m]['Std Dev']   for m in models]

    # Sort by mean score
    sorted_idx  = np.argsort(mean_scores)[::-1]
    models      = [models[i]      for i in sorted_idx]
    mean_scores = [mean_scores[i] for i in sorted_idx]
    std_scores  = [std_scores[i]  for i in sorted_idx]

    # Print summary table
    print(f"\n  {'Model':<35} {'Mean':>8} {'Std':>8} "
          f"{'Best':>8} {'Worst':>8}")
    print(f"  {thin_border}")
    for m in models:
        r = all_results[m]
        print(f"  {m:<35} "
              f"{r[metric_name]:>7.2f}% "
              f"{r['Std Dev']:>7.2f}% "
              f"{r['Best Fold Score']:>7.2f}% "
              f"{r['Worst Fold Score']:>7.2f}%")

    # Bar chart with error bars
    colors = ['green'] + ['steelblue'] * (len(models) - 1)

    
    n_folds = len(
        list(all_results.values())[0]['Fold Scores']
    )
    fold_labels = [f"Fold {i+1}" for i in range(n_folds)]

    for model_name in models[:5]:  # Top 5 models
        scores = all_results[model_name]['Fold Scores']
       

    return models[0]  # Return best model name

# ======================================================
# STEP 6D : TRAIN BEST MODEL ON FULL DATASET
# ======================================================

def train_best_model_full(best_model_name, X, y,
                           ml_type, problem_type):
    print(f"\n{border}")
    print(f"  STEP 6D : TRAINING BEST MODEL ON FULL DATA")
    print(f"{border}")
    print(f"\n  Model   : {best_model_name}")
    print(f"  X shape : {X.shape}")
    if y is not None:
        print(f"  y shape : {y.shape}")

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, f1_score,
                                 precision_score, recall_score,
                                 r2_score, mean_squared_error,
                                 mean_absolute_error,
                                 classification_report,
                                 confusion_matrix)
    import numpy as np

    # -----------------------------------------------
    # Build the correct model object
    # -----------------------------------------------
    model = None

    if ml_type == "SUPERVISED":

        if problem_type == "CLASSIFICATION":
            from sklearn.linear_model import LogisticRegression
            from sklearn.tree         import DecisionTreeClassifier
            from sklearn.ensemble     import (
                RandomForestClassifier,
                GradientBoostingClassifier,
                AdaBoostClassifier,
                ExtraTreesClassifier
            )
            from sklearn.svm          import SVC
            from sklearn.neighbors    import KNeighborsClassifier
            from sklearn.naive_bayes  import GaussianNB

            model_map = {
                'Logistic Regression' : LogisticRegression(
                    max_iter=1000, random_state=42),
                'Decision Tree'       : DecisionTreeClassifier(
                    max_depth=10, random_state=42),
                'Random Forest'       : RandomForestClassifier(
                    n_estimators=100, random_state=42),
                'SVM'                 : SVC(
                    kernel='rbf', random_state=42),
                'KNN'                 : KNeighborsClassifier(
                    n_neighbors=5),
                'Naive Bayes'         : GaussianNB(),
                'Gradient Boosting'   : GradientBoostingClassifier(
                    n_estimators=100, random_state=42),
                'AdaBoost'            : AdaBoostClassifier(
                    n_estimators=100, random_state=42),
                'Extra Trees'         : ExtraTreesClassifier(
                    n_estimators=100, random_state=42),
            }

            try:
                from xgboost import XGBClassifier
                model_map['XGBoost'] = XGBClassifier(
                    n_estimators=100, random_state=42,
                    eval_metric='mlogloss', verbosity=0
                )
            except ImportError:
                pass

        elif problem_type == "REGRESSION":
            from sklearn.linear_model import (LinearRegression,
                                              Ridge, Lasso)
            from sklearn.tree         import DecisionTreeRegressor
            from sklearn.ensemble     import (
                RandomForestRegressor,
                GradientBoostingRegressor,
                ExtraTreesRegressor
            )
            from sklearn.svm          import SVR
            from sklearn.neighbors    import KNeighborsRegressor

            model_map = {
                'Linear Regression'          : LinearRegression(),
                'Ridge Regression'           : Ridge(alpha=1.0),
                'Lasso Regression'           : Lasso(
                    alpha=0.01, max_iter=10000),
                'Decision Tree Regressor'    : DecisionTreeRegressor(
                    max_depth=10, random_state=42),
                'Random Forest Regressor'    : RandomForestRegressor(
                    n_estimators=100, random_state=42),
                'Gradient Boosting Regressor': GradientBoostingRegressor(
                    n_estimators=100, random_state=42),
                'SVR'                        : SVR(kernel='rbf'),
                'KNN Regressor'              : KNeighborsRegressor(
                    n_neighbors=5),
                'Extra Trees Regressor'      : ExtraTreesRegressor(
                    n_estimators=100, random_state=42),
            }

            try:
                from xgboost import XGBRegressor
                model_map['XGBoost Regressor'] = XGBRegressor(
                    n_estimators=100, random_state=42,
                    verbosity=0
                )
            except ImportError:
                pass

    else:
        # Unsupervised
        from sklearn.cluster import (KMeans, DBSCAN,
                                     AgglomerativeClustering)
        from sklearn.mixture import GaussianMixture

        model_map = {
            'K-Means'      : KMeans(n_clusters=3,
                                    random_state=42),
            'DBSCAN'       : DBSCAN(eps=0.5, min_samples=5),
            'Hierarchical' : AgglomerativeClustering(
                n_clusters=3),
            'GMM'          : GaussianMixture(
                n_components=3, random_state=42),
        }

    # -----------------------------------------------
    # Check model exists in map
    # -----------------------------------------------
    if best_model_name not in model_map:
        print(f"\n  ERROR: '{best_model_name}' not in model map!")
        print(f"  Available: {list(model_map.keys())}")
        return None

    model = model_map[best_model_name]

    # -----------------------------------------------
    # Train and Evaluate on Full Data
    # -----------------------------------------------
    print(f"\n  Splitting full data (80% train / 20% test)...")

    if ml_type == "SUPERVISED":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"  Train : {X_train.shape[0]} samples")
        print(f"  Test  : {X_test.shape[0]} samples")
        print(f"\n  Training {best_model_name} on full data...")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # -----------------------------------------------
        # Classification Metrics
        # -----------------------------------------------
        if problem_type == "CLASSIFICATION":
            acc  = accuracy_score(y_test, y_pred) * 100
            prec = precision_score(y_test, y_pred,
                                   average='weighted',
                                   zero_division=0) * 100
            rec  = recall_score(y_test, y_pred,
                                average='weighted',
                                zero_division=0) * 100
            f1   = f1_score(y_test, y_pred,
                            average='weighted',
                            zero_division=0) * 100

            print(f"\n  Full Data Results:")
            print(f"  Accuracy  : {acc:.2f}%")
            print(f"  Precision : {prec:.2f}%")
            print(f"  Recall    : {rec:.2f}%")
            print(f"  F1 Score  : {f1:.2f}%")

            print(f"\n  Classification Report:")
            print(classification_report(y_test, y_pred,
                                        zero_division=0))

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
           

            return {
                'Model'    : best_model_name,
                'Accuracy' : round(acc,  2),
                'Precision': round(prec, 2),
                'Recall'   : round(rec,  2),
                'F1 Score' : round(f1,   2)
            }

        # -----------------------------------------------
        # Regression Metrics
        # -----------------------------------------------
        elif problem_type == "REGRESSION":
            r2   = r2_score(y_test, y_pred) * 100
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae  = mean_absolute_error(y_test, y_pred)
            mse  = mean_squared_error(y_test, y_pred)

            print(f"\n  Full Data Results:")
            print(f"  R2 Score : {r2:.2f}%")
            print(f"  RMSE     : {rmse:.4f}")
            print(f"  MAE      : {mae:.4f}")
            print(f"  MSE      : {mse:.4f}")

            # Actual vs Predicted
            

            return {
                'Model'   : best_model_name,
                'R2 Score': round(r2,   2),
                'RMSE'    : round(rmse, 4),
                'MAE'     : round(mae,  4),
                'MSE'     : round(mse,  4)
            }

    # -----------------------------------------------
    # Unsupervised
    # -----------------------------------------------
    else:
        from sklearn.metrics import (silhouette_score,
                                     davies_bouldin_score)

        print(f"\n  Fitting {best_model_name} on full data...")
        X_arr  = X.values
        labels = model.fit_predict(X_arr)

        n_clusters   = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise      = list(labels).count(-1)

        print(f"\n  Clusters Found : {n_clusters}")
        print(f"  Noise Points   : {n_noise}")

        result = {
            'Model'     : best_model_name,
            'Clusters'  : n_clusters,
            'Silhouette': None,
            'DB Score'  : None
        }

        if n_clusters > 1:
            sil = silhouette_score(X_arr, labels)
            db  = davies_bouldin_score(X_arr, labels)
            print(f"  Silhouette     : {sil:.4f}")
            print(f"  Davies-Bouldin : {db:.4f}")
            result['Silhouette'] = round(sil, 4)
            result['DB Score']   = round(db,  4)

        # Plot clusters
        from sklearn.decomposition import PCA
        pca  = PCA(n_components=2)
        X_2d = pca.fit_transform(X_arr)
        exp  = pca.explained_variance_ratio_.sum() * 100

        

        return result


# ======================================================
# STEP 6 : RUN MODELS (K-FOLD → BEST → FULL TRAIN)
# ======================================================

def run_models(X, y, ml_type, problem_type):
    print(f"\n{border}")
    print(f"     STEP 6 : RUN MODELS WITH K-FOLD CV")
    print(f"{border}")

    n_splits = 5
    print(f"\n  Strategy : Stratified {n_splits}-Fold "
          f"Cross Validation")
    print(f"  Each model tested on every fold")
    print(f"  Best model then trained on full dataset")

    # Step 6A: Create folds
    splits, kf, fold_type = create_stratified_folds(
        X, y, n_splits=n_splits
    )

    # Step 6B: Evaluate all models across folds
    all_results, metric_name = evaluate_models_kfold(
        X, y, splits, ml_type, problem_type
    )

    # Step 6C: Visualize fold results
    best_model_name = visualize_kfold_results(
        all_results, metric_name
    )

    # Convert to list for report
    sample_results = list(all_results.values())

    # Remove fold scores from report (too verbose)
    for r in sample_results:
        r.pop('Fold Scores', None)

    print(f"\n  BEST MODEL : {best_model_name}")
    print(f"  Now training on FULL dataset...")

    # Step 6D: Train best model on full data
    full_result = train_best_model_full(
        best_model_name, X, y, ml_type, problem_type
    )

    return sample_results, full_result, best_model_name



# ======================================================
# STEP 7 : FINAL COMPARISON REPORT
# ======================================================

def display_final_report(sample_results, full_result,
                          best_model_name, ml_type,
                          problem_type):
    print(f"\n{border}")
    print(f"     STEP 7 : FINAL COMPARISON REPORT")
    print(f"{border}")

    if not sample_results:
        print("  No results to display.")
        return

    metric = ("Accuracy"   if problem_type == "CLASSIFICATION"
              else "R2 Score" if problem_type == "REGRESSION"
              else "Silhouette")

    df_sample = pd.DataFrame(sample_results)

    # Sample results table
    print(f"\n  --- ALL MODELS ON 10% SAMPLE ---")
    if metric in df_sample.columns:
        print(df_sample[['Model', metric]].to_string(index=False))

    # Full data result
    if full_result:
        print(f"\n  --- BEST MODEL ON FULL DATASET ---")
        print(f"  Model     : {best_model_name}")
        if metric in full_result:
            sample_score = df_sample.loc[
                df_sample['Model'] == best_model_name,
                metric
            ].values

            full_score   = full_result[metric]

            print(f"  Sample {metric:10} : "
                  f"{sample_score[0] if len(sample_score) > 0 else 'N/A'}")
            print(f"  Full   {metric:10} : {full_score}")

            if len(sample_score) > 0:
                improvement = full_score - sample_score[0]
                tag = "improved" if improvement > 0 else "dropped"
                print(f"  Change            : "
                      f"{improvement:+.2f} ({tag} on full data)")

    # Final bar chart — sample vs full
    if full_result and metric in df_sample.columns:
        df_plot = df_sample[['Model', metric]].copy()
        df_plot = df_plot.sort_values(metric, ascending=False)

        colors = ['green' if m == best_model_name
                  else 'steelblue'
                  for m in df_plot['Model']]

       
        # Mark full data score
       

    print(f"\n{border}")
    print(f"  WINNER : {best_model_name}")
    if full_result and metric in full_result:
        print(f"  FINAL {metric} ON FULL DATA : "
              f"{full_result[metric]}")
    print(f"{border}")
    print(f"\n  PIPELINE COMPLETE!")
    print(f"{border}\n")


# ======================================================
# MAIN
# ======================================================

def main():
    print(f"\n{border}")
    print(f"     SMART ML PIPELINE — START")
    print(f"{border}")

    # Step 1: Load
    df = load_dataset()
    if df is None:
        print("  Exiting — no dataset loaded.")
        return

    # Step 2: Detect ML type
    ml_type, target_col = detect_ml_type(df)

    # Step 3: Confirm with user
    ml_type, target_col = confirm_with_user(df, ml_type, target_col)

    # Step 4: EDA
    problem_type = run_eda(df, ml_type, target_col)

    # Step 5: Clean
    X, y = clean_dataset(df, target_col)

    # Step 6: Sample → Compare → Best on Full
    sample_results, full_result, best_model_name = run_models(
        X, y, ml_type, problem_type
    )

    # Step 7: Final report
    display_final_report(
        sample_results, full_result,
        best_model_name, ml_type, problem_type
    )


if __name__ == "__main__":
    main()