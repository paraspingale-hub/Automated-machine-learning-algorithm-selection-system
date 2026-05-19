# -------------------------------------------------------
# Program : Smart ML Pipeline — Web API Entry Point
# File    : app.py (Replaces MainModel.py)
# -------------------------------------------------------

from flask import Flask, request, jsonify, render_template
import pandas as pd
import os
import warnings
from werkzeug.utils import secure_filename
from sklearn.preprocessing import LabelEncoder

import regression_models
import unsupervised_models
import classification_models 

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def clean_dataset(df, target_col=None):
    df_clean = df.copy().drop_duplicates()

    # Drop columns with >50% missing data
    for col in df_clean.columns:
        if (df_clean[col].isnull().sum() / len(df_clean) * 100) > 50:
            df_clean.drop(columns=[col], inplace=True)

    # Impute remaining missing data
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() == 0: continue
        if col == target_col:
            df_clean.dropna(subset=[col], inplace=True)
        elif df_clean[col].dtype == 'object':
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        else:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)

    # Encode categoricals
    le = LabelEncoder()
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            if col != target_col:
                try:
                    df_clean[col] = pd.to_numeric(df_clean[col])
                    continue
                except: pass
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))

    # Remove outliers for numerical columns
    num_cols = df_clean.select_dtypes(include=['number']).columns.tolist()
    if target_col and target_col in num_cols:
        num_cols.remove(target_col)

    for col in num_cols:
        Q1, Q3 = df_clean[col].quantile(0.25), df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        df_clean = df_clean[(df_clean[col] >= Q1 - 1.5 * IQR) & (df_clean[col] <= Q3 + 1.5 * IQR)]

    if target_col:
        X = df_clean.drop(columns=[target_col])
        y = df_clean[target_col].reset_index(drop=True)
    else:
        X = df_clean.copy()
        y = None

    return X, y

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_pipeline', methods=['POST'])
def run_pipeline():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    ml_type = request.form.get('ml_type')
    problem_type = request.form.get('problem_type')
    target_col = request.form.get('target_col', '').strip()

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(filepath)
        elif filename.endswith('.json'):
            df = pd.read_json(filepath)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400

        if ml_type == 'SUPERVISED' and target_col not in df.columns:
            return jsonify({'error': f"Target column '{target_col}' not found."}), 400

        X, y = clean_dataset(df, target_col if ml_type == 'SUPERVISED' else None)
        
        results = []
        if ml_type == "SUPERVISED":
            if problem_type == "REGRESSION":
                results = regression_models.run_all_regression_models(X, y)
                metric = "R2 Score"
            elif problem_type == "CLASSIFICATION":
                results = classification_models.run_all_classification_models(X, y)
                metric = "Accuracy"
        elif ml_type == "UNSUPERVISED":
            results = unsupervised_models.run_all_unsupervised_models(X)
            metric = "Silhouette"
            # Filter out models that don't produce a silhouette score
            results = [r for r in results if r.get('Silhouette') is not None]

        # Sort results for the frontend leaderboard
        results = sorted(results, key=lambda x: x.get(metric, 0), reverse=True)

        return jsonify({
            'status': 'success',
            'data': results,
            'metric': metric,
            'samples': X.shape[0],
            'features': X.shape[1]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    app.run(debug=True, port=5000)