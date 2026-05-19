# -------------------------------------------------------
# Program : Unsupervised ML Models
# File    : unsupervised_models.py
# Usage   : Called by main.py with cleaned X data
# -------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

border = "="*55


# ======================================================
# HELPER : Display Cluster Summary
# ======================================================

def display_cluster_summary(model_name, X, labels):
    print(f"\n{border}")
    print(f"  RESULTS : {model_name}")
    print(f"{border}")

    n_clusters  = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise     = list(labels).count(-1)

    print(f"  Total Clusters Found : {n_clusters}")

    if n_noise > 0:
        print(f"  Noise Points         : {n_noise}")

    # Cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\n  Cluster Size Distribution:")
    for u, c in zip(unique, counts):
        label = "Noise" if u == -1 else f"Cluster {u}"
        print(f"    {label} : {c} samples ({c/len(labels)*100:.1f}%)")

    # Silhouette Score (only if more than 1 cluster)
    result = {
        'Model'     : model_name,
        'Clusters'  : n_clusters,
        'Silhouette': None,
        'DB Score'  : None
    }

    if n_clusters > 1:
        try:
            sil = silhouette_score(X, labels)
            db  = davies_bouldin_score(X, labels)
            print(f"\n  Silhouette Score     : {sil:.4f}  (higher is better, max=1)")
            print(f"  Davies-Bouldin Score : {db:.4f}   (lower is better, min=0)")
            result['Silhouette'] = round(sil, 4)
            result['DB Score']   = round(db,  4)
        except Exception as e:
            print(f"  Could not compute scores: {e}")
    else:
        print("  Only 1 cluster found — scores not applicable")

    return result


# ======================================================
# HELPER : Plot Clusters (2D using PCA)
# ======================================================

def plot_clusters(X, labels, title):
    from sklearn.decomposition import PCA

    # Reduce to 2D for visualization
    pca        = PCA(n_components=2)
    X_2d       = pca.fit_transform(X)
    explained  = pca.explained_variance_ratio_.sum() * 100



# ======================================================
# MODEL 1 : K-Means Clustering
# ======================================================

def kmeans_model(X):
    from sklearn.cluster import KMeans

    print(f"\n{border}")
    print(f"  MODEL 1 : K-MEANS CLUSTERING")
    print(f"{border}")
    print("  Type    : Centroid-based Clustering")
    print("  Best For: Spherical clusters, Known number of clusters")

    # -----------------------------------------------
    # Find best K using Elbow Method
    # -----------------------------------------------
    print("\n  Finding best K using Elbow Method...")

    inertia   = []
    sil_scores = []
    k_range   = range(2, 11)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertia.append(km.inertia_)
        sil_scores.append(silhouette_score(X, km.labels_))

    # Plot Elbow
    
    # Pick best K from silhouette
    best_k = k_range[sil_scores.index(max(sil_scores))]
    print(f"  Best K (Silhouette) : {best_k}")

    # Train final model
    model  = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = model.fit_predict(X)

    # Plot clusters
    plot_clusters(X, labels, f"K-Means Clustering (K={best_k})")

    result = display_cluster_summary("K-Means", X, labels)
    result['Best K'] = best_k
    return result


# ======================================================
# MODEL 2 : DBSCAN
# ======================================================

def dbscan_model(X):
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors

    print(f"\n{border}")
    print(f"  MODEL 2 : DBSCAN")
    print(f"{border}")
    print("  Type    : Density-based Clustering")
    print("  Best For: Arbitrary shaped clusters, Noise detection")

    # -----------------------------------------------
    # Find best eps using K-Distance Graph
    # -----------------------------------------------
    print("\n  Finding best eps using K-Distance Graph...")

    neighbors = NearestNeighbors(n_neighbors=5)
    neighbors.fit(X)
    distances, _ = neighbors.kneighbors(X)
    distances     = np.sort(distances[:, 4], axis=0)


    # Auto estimate eps from elbow point
    eps_value = float(np.percentile(distances, 90))
    print(f"  Auto estimated eps : {eps_value:.4f}")

    model  = DBSCAN(eps=eps_value, min_samples=5)
    labels = model.fit_predict(X)

    plot_clusters(X, labels, "DBSCAN Clustering")

    result = display_cluster_summary("DBSCAN", X, labels)
    result['eps'] = round(eps_value, 4)
    return result


# ======================================================
# MODEL 3 : Agglomerative Hierarchical Clustering
# ======================================================

def hierarchical_model(X):
    from sklearn.cluster import AgglomerativeClustering
    from scipy.cluster.hierarchy import dendrogram, linkage

    print(f"\n{border}")
    print(f"  MODEL 3 : AGGLOMERATIVE HIERARCHICAL CLUSTERING")
    print(f"{border}")
    print("  Type    : Hierarchical Clustering")
    print("  Best For: Unknown number of clusters, Tree structure")

    # -----------------------------------------------
    # Dendrogram (use sample for large datasets)
    # -----------------------------------------------
    print("\n  Plotting Dendrogram...")

    sample_size = min(200, len(X))
    X_sample    = X[:sample_size]

    linked = linkage(X_sample, method='ward')

   
    # Find best n_clusters using silhouette
    print("\n  Finding best number of clusters...")
    sil_scores = []
    k_range    = range(2, 9)

    for k in k_range:
        model  = AgglomerativeClustering(n_clusters=k)
        labels = model.fit_predict(X)
        sil_scores.append(silhouette_score(X, labels))

    best_k = k_range[sil_scores.index(max(sil_scores))]
    print(f"  Best Clusters : {best_k}")

    # Final model
    model  = AgglomerativeClustering(n_clusters=best_k)
    labels = model.fit_predict(X)

    plot_clusters(X, labels, f"Hierarchical Clustering (k={best_k})")

    result = display_cluster_summary("Hierarchical", X, labels)
    result['Best K'] = best_k
    return result


# ======================================================
# MODEL 4 : Gaussian Mixture Model (GMM)
# ======================================================

def gmm_model(X):
    from sklearn.mixture import GaussianMixture

    print(f"\n{border}")
    print(f"  MODEL 4 : GAUSSIAN MIXTURE MODEL (GMM)")
    print(f"{border}")
    print("  Type    : Probabilistic Clustering")
    print("  Best For: Soft cluster assignments, Elliptical clusters")

    # Find best n_components using BIC
    print("\n  Finding best components using BIC score...")

    bic_scores = []
    aic_scores = []
    k_range    = range(2, 11)

    for k in k_range:
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(X)
        bic_scores.append(gmm.bic(X))
        aic_scores.append(gmm.aic(X))

    # Plot BIC and AIC
   
    best_k = k_range[bic_scores.index(min(bic_scores))]
    print(f"  Best Components (BIC) : {best_k}")

    model  = GaussianMixture(n_components=best_k, random_state=42)
    model.fit(X)
    labels = model.predict(X)

    # Probabilities
    probs      = model.predict_proba(X)
    confidence = probs.max(axis=1).mean()
    print(f"\n  Average Cluster Confidence : {confidence*100:.2f}%")

    plot_clusters(X, labels, f"GMM Clustering (k={best_k})")

    result = display_cluster_summary("GMM", X, labels)
    result['Best K']     = best_k
    result['Confidence'] = round(confidence * 100, 2)
    return result


# ======================================================
# MODEL 5 : PCA — Dimensionality Reduction
# ======================================================

def pca_analysis(X):
    from sklearn.decomposition import PCA

    print(f"\n{border}")
    print(f"  MODEL 5 : PCA — DIMENSIONALITY REDUCTION")
    print(f"{border}")
    print("  Type    : Linear Dimensionality Reduction")
    print("  Best For: Feature reduction, Noise removal, Visualization")

    # Full PCA
    pca            = PCA(random_state=42)
    pca.fit(X)
    explained_var  = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)

    # Plot explained variance
    

    # Find components for 95% variance
    n_95 = np.argmax(cumulative_var >= 0.95) + 1
    n_99 = np.argmax(cumulative_var >= 0.99) + 1

    print(f"\n  Original Features       : {X.shape[1]}")
    print(f"  Components for 95% var  : {n_95}")
    print(f"  Components for 99% var  : {n_99}")

    # Apply PCA with 95% variance
    pca_95  = PCA(n_components=n_95, random_state=42)
    X_pca   = pca_95.fit_transform(X)

    print(f"\n  Reduced Shape : {X_pca.shape}")
    print(f"  Variance kept : {pca_95.explained_variance_ratio_.sum()*100:.2f}%")

    result = {
        'Model'              : 'PCA',
        'Original Features'  : X.shape[1],
        'Components (95%)'   : n_95,
        'Components (99%)'   : n_99,
        'Variance Kept'      : round(pca_95.explained_variance_ratio_.sum()*100, 2)
    }
    return result, X_pca


# ======================================================
# MODEL 6 : t-SNE — Visualization
# ======================================================

def tsne_analysis(X):
    from sklearn.manifold import TSNE

    print(f"\n{border}")
    print(f"  MODEL 6 : t-SNE VISUALIZATION")
    print(f"{border}")
    print("  Type    : Non-linear Dimensionality Reduction")
    print("  Best For: Visualizing high-dimensional clusters in 2D/3D")

    # Use sample for large datasets
    sample_size = min(2000, len(X))
    X_sample    = X[:sample_size]

    print(f"\n  Running t-SNE on {sample_size} samples...")
    print("  (This may take a moment...)")

    tsne   = TSNE(n_components=2, random_state=42,
                  perplexity=30, n_iter=1000)
    X_tsne = tsne.fit_transform(X_sample)


    print(f"\n  t-SNE complete!")
    print(f"  Natural clusters visible in the plot above")

    result = {
        'Model'       : 't-SNE',
        'Samples Used': sample_size,
        'Output Dims' : 2
    }
    return result


# ======================================================
# MODEL 7 : Isolation Forest — Anomaly Detection
# ======================================================

def isolation_forest_model(X):
    from sklearn.ensemble import IsolationForest

    print(f"\n{border}")
    print(f"  MODEL 7 : ISOLATION FOREST — ANOMALY DETECTION")
    print(f"{border}")
    print("  Type    : Anomaly / Outlier Detection")
    print("  Best For: Finding rare unusual data points")

    model       = IsolationForest(contamination=0.05,
                                  random_state=42)
    labels      = model.fit_predict(X)

    # -1 = anomaly, 1 = normal
    n_anomalies = (labels == -1).sum()
    n_normal    = (labels ==  1).sum()
    anomaly_pct = n_anomalies / len(labels) * 100

    print(f"\n  Total Samples  : {len(labels)}")
    print(f"  Normal Points  : {n_normal}  ({100-anomaly_pct:.1f}%)")
    print(f"  Anomaly Points : {n_anomalies} ({anomaly_pct:.1f}%)")

    # Visualize anomalies
    from sklearn.decomposition import PCA
    pca  = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    colors = ['red' if l == -1 else 'steelblue' for l in labels]



    result = {
        'Model'         : 'Isolation Forest',
        'Normal Points' : n_normal,
        'Anomalies'     : n_anomalies,
        'Anomaly %'     : round(anomaly_pct, 2)
    }
    return result


# ======================================================
# RUN ALL UNSUPERVISED MODELS — Called by main.py
# ======================================================

def run_all_unsupervised_models(X, selected_models=None):
    print(f"\n{border}\n  RUNNING UNSUPERVISED MODELS\n{border}")
    scaler = StandardScaler()
    X_array = scaler.fit_transform(X)

    model_map = {
        "K-Means": kmeans_model,
        "DBSCAN": dbscan_model,
        "Hierarchical": hierarchical_model,
        "GMM": gmm_model,
        "PCA": pca_analysis,
        "t-SNE": tsne_analysis,
        "Isolation Forest": isolation_forest_model
    }

    models_to_run = []
    if selected_models:
        models_to_run = [model_map[name] for name in selected_models if name in model_map]
    else:
        models_to_run = list(model_map.values())

    all_results = []
    for model_func in models_to_run:
        try:
            # PCA and t-SNE return different structures, handle carefully
            if model_func.__name__ == 'pca_analysis':
                res, _ = model_func(X_array)
                all_results.append(res)
            else:
                res = model_func(X_array)
                if res: all_results.append(res)
        except Exception as e:
            print(f"  ERROR in {model_func.__name__}: {e}")
            continue

    return all_results