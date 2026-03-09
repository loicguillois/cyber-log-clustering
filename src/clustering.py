"""
Clustering Module for Host Behavior Analysis

This module implements multiple clustering algorithms to group hosts by their
network behavior patterns. Each algorithm has different strengths:

ALGORITHM COMPARISON FOR CYBERSECURITY:
=======================================

1. KMEANS
   - Pros: Fast, scalable, deterministic cluster count
   - Cons: Assumes spherical clusters, sensitive to outliers
   - Best for: Initial exploration, when you have prior knowledge of cluster count
   - Cybersecurity use: Quick segmentation of known behavior types

2. DBSCAN
   - Pros: Finds arbitrary-shaped clusters, identifies outliers as noise
   - Cons: Struggles with varying density, sensitive to parameters
   - Best for: When cluster shapes are irregular
   - Cybersecurity use: Isolating anomalous hosts that don't fit any pattern

3. HDBSCAN
   - Pros: Handles varying densities, robust parameter selection
   - Cons: Slower on large datasets
   - Best for: Production deployment, automatic cluster discovery
   - Cybersecurity use: Discovering unknown attack patterns, robust anomaly detection

Dimensionality Reduction for Visualization:
- PCA: Fast, preserves variance, linear transformations
- UMAP: Better preserves local structure, reveals clusters visually
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, Optional, Dict, List
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports with fallbacks
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    logger.warning("HDBSCAN not available. Install with: pip install hdbscan")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logger.warning("UMAP not available. Install with: pip install umap-learn")


class ClusteringPipeline:
    """
    Comprehensive clustering pipeline for host behavior analysis.

    Provides multiple clustering methods and automatic parameter tuning
    optimized for cybersecurity use cases.
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize clustering pipeline.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.cluster_labels = {}
        self.reduced_data = {}
        self.cluster_stats = {}

    def reduce_dimensions_pca(
        self,
        X: np.ndarray,
        n_components: int = 2,
        variance_threshold: float = 0.95
    ) -> Tuple[np.ndarray, PCA]:
        """
        Reduce dimensionality using PCA.

        PCA projects data onto principal components that capture maximum variance.
        Useful for:
        - Visualization (2D/3D)
        - Noise reduction before clustering
        - Understanding which features drive variance

        Args:
            X: Scaled feature matrix
            n_components: Target dimensions (2 for visualization)
            variance_threshold: Alternative - keep components until this variance

        Returns:
            Tuple of (reduced data, fitted PCA model)
        """
        logger.info(f"Applying PCA (n_components={n_components})...")

        pca = PCA(n_components=n_components, random_state=self.random_state)
        X_pca = pca.fit_transform(X)

        explained_var = sum(pca.explained_variance_ratio_)
        logger.info(f"  Explained variance: {explained_var:.2%}")

        self.reduced_data['pca'] = X_pca
        return X_pca, pca

    def reduce_dimensions_umap(
        self,
        X: np.ndarray,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1
    ) -> np.ndarray:
        """
        Reduce dimensionality using UMAP.

        UMAP (Uniform Manifold Approximation and Projection) is excellent for:
        - Preserving local cluster structure
        - Revealing hidden patterns in high-dimensional data
        - Better visualization than PCA for complex data

        UMAP parameters explained:
        - n_neighbors: Larger = more global structure, Smaller = more local detail
        - min_dist: How tightly UMAP packs points (smaller = tighter clusters)

        Args:
            X: Scaled feature matrix
            n_components: Target dimensions
            n_neighbors: Size of local neighborhood
            min_dist: Minimum distance between points

        Returns:
            Reduced data matrix
        """
        if not UMAP_AVAILABLE:
            logger.warning("UMAP not available, falling back to PCA")
            X_reduced, _ = self.reduce_dimensions_pca(X, n_components)
            return X_reduced

        logger.info(f"Applying UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")

        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric='euclidean',
            random_state=self.random_state
        )
        X_umap = reducer.fit_transform(X)

        self.reduced_data['umap'] = X_umap
        logger.info(f"  UMAP reduction complete: {X_umap.shape}")

        return X_umap

    def estimate_kmeans_k(
        self,
        X: np.ndarray,
        k_range: Tuple[int, int] = (2, 15)
    ) -> Dict[str, List]:
        """
        Estimate optimal K for KMeans using multiple metrics.

        Methods used:
        1. Elbow Method (Inertia): Look for "elbow" in inertia curve
        2. Silhouette Score: Measures cluster separation (higher = better)
        3. Calinski-Harabasz: Ratio of between/within cluster variance

        Args:
            X: Feature matrix
            k_range: Range of K values to try

        Returns:
            Dictionary with metrics for each K
        """
        logger.info(f"Estimating optimal K for KMeans (range: {k_range})...")

        k_values = list(range(k_range[0], k_range[1] + 1))
        inertias = []
        silhouettes = []
        calinski = []

        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)

            inertias.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(X, labels))
            calinski.append(calinski_harabasz_score(X, labels))

        # Find optimal K based on silhouette score
        best_k = k_values[np.argmax(silhouettes)]
        logger.info(f"  Best K by silhouette score: {best_k}")

        return {
            'k_values': k_values,
            'inertias': inertias,
            'silhouettes': silhouettes,
            'calinski': calinski,
            'best_k': best_k
        }

    def fit_kmeans(self, X: np.ndarray, n_clusters: int = 5) -> np.ndarray:
        """
        Fit KMeans clustering.

        KMeans partitions hosts into K groups minimizing within-cluster variance.
        Good for:
        - Quick initial segmentation
        - When you expect a specific number of behavior types

        For cybersecurity, common cluster interpretations:
        - Normal workstation traffic (most hosts)
        - Server behavior (high incoming connections)
        - Scanning activity (high destination diversity)
        - Data exfiltration (high outbound bytes)
        - C2 beaconing (regular connection patterns)

        Args:
            X: Feature matrix
            n_clusters: Number of clusters

        Returns:
            Cluster labels
        """
        logger.info(f"Fitting KMeans (n_clusters={n_clusters})...")

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )
        labels = kmeans.fit_predict(X)

        self.models['kmeans'] = kmeans
        self.cluster_labels['kmeans'] = labels

        # Compute metrics
        silhouette = silhouette_score(X, labels)
        logger.info(f"  Silhouette score: {silhouette:.4f}")

        return labels

    def estimate_dbscan_eps(self, X: np.ndarray, k: int = 5) -> float:
        """
        Estimate DBSCAN epsilon using k-nearest neighbors.

        The "elbow" in the k-distance graph indicates a good eps value.
        Points beyond the elbow have significantly larger k-distances,
        suggesting they are outliers.

        Args:
            X: Feature matrix
            k: Number of neighbors to consider

        Returns:
            Estimated epsilon value
        """
        logger.info(f"Estimating DBSCAN epsilon using {k}-NN distances...")

        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X)
        distances, _ = nn.kneighbors(X)

        # Sort distances to k-th neighbor
        k_distances = np.sort(distances[:, -1])

        # Find elbow using gradient change
        gradient = np.gradient(k_distances)
        elbow_idx = np.argmax(gradient)

        eps = k_distances[elbow_idx]
        logger.info(f"  Estimated eps: {eps:.4f}")

        return eps

    def fit_dbscan(
        self,
        X: np.ndarray,
        eps: Optional[float] = None,
        min_samples: int = 5
    ) -> np.ndarray:
        """
        Fit DBSCAN clustering.

        DBSCAN groups points that are closely packed together, marking points
        in low-density regions as outliers (label = -1).

        For cybersecurity:
        - Outliers are often the most interesting (anomalous hosts)
        - Does not force anomalies into clusters like KMeans
        - Good for finding hosts that don't fit normal patterns

        Args:
            X: Feature matrix
            eps: Maximum distance between neighbors (auto-estimated if None)
            min_samples: Minimum points to form dense region

        Returns:
            Cluster labels (-1 = outlier)
        """
        if eps is None:
            eps = self.estimate_dbscan_eps(X)

        logger.info(f"Fitting DBSCAN (eps={eps:.4f}, min_samples={min_samples})...")

        dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = dbscan.fit_predict(X)

        self.models['dbscan'] = dbscan
        self.cluster_labels['dbscan'] = labels

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_outliers = np.sum(labels == -1)
        logger.info(f"  Clusters found: {n_clusters}, Outliers: {n_outliers}")

        if n_clusters > 1:
            # Only compute silhouette for non-outlier points
            mask = labels != -1
            if mask.sum() > 1:
                silhouette = silhouette_score(X[mask], labels[mask])
                logger.info(f"  Silhouette score (excl. outliers): {silhouette:.4f}")

        return labels

    def fit_hdbscan(
        self,
        X: np.ndarray,
        min_cluster_size: int = 10,
        min_samples: Optional[int] = None
    ) -> np.ndarray:
        """
        Fit HDBSCAN clustering.

        HDBSCAN (Hierarchical DBSCAN) is an improved version that:
        - Handles clusters of varying densities
        - Does not require eps parameter
        - Provides cluster membership probabilities

        For cybersecurity, HDBSCAN is often the best choice:
        - Automatically identifies the number of behavior clusters
        - Robust to parameter selection
        - Provides confidence scores for cluster membership

        Args:
            X: Feature matrix
            min_cluster_size: Minimum cluster size
            min_samples: Core points threshold (defaults to min_cluster_size)

        Returns:
            Cluster labels (-1 = outlier)
        """
        if not HDBSCAN_AVAILABLE:
            logger.warning("HDBSCAN not available, falling back to DBSCAN")
            return self.fit_dbscan(X)

        if min_samples is None:
            min_samples = min_cluster_size

        logger.info(f"Fitting HDBSCAN (min_cluster_size={min_cluster_size})...")

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom',  # Excess of Mass
            prediction_data=True
        )
        labels = clusterer.fit_predict(X)

        self.models['hdbscan'] = clusterer
        self.cluster_labels['hdbscan'] = labels

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_outliers = np.sum(labels == -1)
        logger.info(f"  Clusters found: {n_clusters}, Outliers: {n_outliers}")

        # HDBSCAN provides membership probabilities
        if hasattr(clusterer, 'probabilities_'):
            avg_prob = np.mean(clusterer.probabilities_)
            logger.info(f"  Average membership probability: {avg_prob:.4f}")

        return labels

    def compute_cluster_statistics(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str],
        host_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute detailed statistics for each cluster.

        Essential for interpreting what each cluster represents.
        SOC analysts need to understand cluster characteristics to
        prioritize investigation.

        Args:
            X: Feature matrix
            labels: Cluster labels
            feature_names: Names of features
            host_df: Original host features DataFrame

        Returns:
            DataFrame with cluster statistics
        """
        logger.info("Computing cluster statistics...")

        stats_list = []

        for cluster_id in sorted(set(labels)):
            mask = labels == cluster_id
            cluster_size = mask.sum()

            stats = {
                'cluster': cluster_id,
                'size': cluster_size,
                'pct_of_total': cluster_size / len(labels) * 100
            }

            # Feature statistics
            for i, feat_name in enumerate(feature_names):
                if i < X.shape[1]:
                    stats[f'{feat_name}_mean'] = np.mean(X[mask, i])
                    stats[f'{feat_name}_std'] = np.std(X[mask, i])

            # Add attack ratio if available
            if 'attack_ratio' in host_df.columns:
                stats['attack_ratio'] = host_df.loc[mask, 'attack_ratio'].mean()

            stats_list.append(stats)

        stats_df = pd.DataFrame(stats_list)
        self.cluster_stats = stats_df

        return stats_df

    def interpret_clusters(
        self,
        stats_df: pd.DataFrame,
        feature_names: List[str]
    ) -> Dict[int, str]:
        """
        Automatically interpret cluster behaviors based on feature patterns.

        Uses domain knowledge to map feature patterns to cybersecurity behaviors.

        Interpretation rules:
        - High dst_entropy + high n_unique_dsts = Scanning activity
        - High bytes_ratio + high total_bytes_sent = Data exfiltration
        - High n_connections + low dst_entropy = Server behavior
        - High rst_ratio = Possible scan or firewall blocks
        - Normal low activity = Typical workstation

        Args:
            stats_df: Cluster statistics DataFrame
            feature_names: List of feature names

        Returns:
            Dictionary mapping cluster ID to interpretation
        """
        interpretations = {}

        for _, row in stats_df.iterrows():
            cluster_id = int(row['cluster'])

            if cluster_id == -1:
                interpretations[-1] = "Outliers / Anomalous Behavior"
                continue

            # Initialize interpretation
            indicators = []

            # Check for scanning behavior
            if 'dst_entropy_mean' in row and row.get('dst_entropy_mean', 0) > 1:
                if 'n_unique_dsts_mean' in row and row.get('n_unique_dsts_mean', 0) > 10:
                    indicators.append("Scanning/Reconnaissance")

            # Check for data exfiltration
            if 'bytes_ratio_mean' in row and row.get('bytes_ratio_mean', 0) > 2:
                if 'total_bytes_sent_mean' in row and row.get('total_bytes_sent_mean', 0) > 100000:
                    indicators.append("Potential Data Exfiltration")

            # Check for server behavior
            if 'n_connections_mean' in row and row.get('n_connections_mean', 0) > 100:
                if 'dst_entropy_mean' in row and row.get('dst_entropy_mean', 0) < 1:
                    indicators.append("Server Behavior")

            # Check for connection anomalies
            if 'rst_ratio_mean' in row and row.get('rst_ratio_mean', 0) > 0.3:
                indicators.append("High Reset Ratio (Scans/Blocks)")

            # Check for high packet loss
            if 'loss_ratio_mean' in row and row.get('loss_ratio_mean', 0) > 0.1:
                indicators.append("High Packet Loss")

            # Check for attack ratio
            if 'attack_ratio' in row and row.get('attack_ratio', 0) > 0.5:
                indicators.append("High Attack Traffic")

            # Default interpretation
            if not indicators:
                if row.get('n_connections_mean', 0) < 20:
                    indicators.append("Low-Activity Workstation")
                else:
                    indicators.append("Normal Traffic Pattern")

            interpretations[cluster_id] = " | ".join(indicators)

        return interpretations

    def compare_clustering_methods(
        self,
        X: np.ndarray,
        kmeans_k: int = 5,
        dbscan_eps: Optional[float] = None,
        hdbscan_min_size: int = 10
    ) -> pd.DataFrame:
        """
        Compare all clustering methods and return metrics.

        Helps select the best method for the specific dataset.

        Args:
            X: Feature matrix
            kmeans_k: Number of clusters for KMeans
            dbscan_eps: Epsilon for DBSCAN
            hdbscan_min_size: Min cluster size for HDBSCAN

        Returns:
            Comparison DataFrame with metrics
        """
        logger.info("Comparing clustering methods...")

        results = []

        # KMeans
        kmeans_labels = self.fit_kmeans(X, n_clusters=kmeans_k)
        results.append({
            'method': 'KMeans',
            'n_clusters': kmeans_k,
            'n_outliers': 0,
            'silhouette': silhouette_score(X, kmeans_labels),
            'calinski_harabasz': calinski_harabasz_score(X, kmeans_labels),
            'davies_bouldin': davies_bouldin_score(X, kmeans_labels)
        })

        # DBSCAN
        dbscan_labels = self.fit_dbscan(X, eps=dbscan_eps)
        n_clusters_db = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
        mask_db = dbscan_labels != -1
        if n_clusters_db > 1 and mask_db.sum() > n_clusters_db:
            results.append({
                'method': 'DBSCAN',
                'n_clusters': n_clusters_db,
                'n_outliers': np.sum(dbscan_labels == -1),
                'silhouette': silhouette_score(X[mask_db], dbscan_labels[mask_db]),
                'calinski_harabasz': calinski_harabasz_score(X[mask_db], dbscan_labels[mask_db]),
                'davies_bouldin': davies_bouldin_score(X[mask_db], dbscan_labels[mask_db])
            })
        else:
            results.append({
                'method': 'DBSCAN',
                'n_clusters': n_clusters_db,
                'n_outliers': np.sum(dbscan_labels == -1),
                'silhouette': np.nan,
                'calinski_harabasz': np.nan,
                'davies_bouldin': np.nan
            })

        # HDBSCAN
        if HDBSCAN_AVAILABLE:
            hdbscan_labels = self.fit_hdbscan(X, min_cluster_size=hdbscan_min_size)
            n_clusters_hdb = len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)
            mask_hdb = hdbscan_labels != -1
            if n_clusters_hdb > 1 and mask_hdb.sum() > n_clusters_hdb:
                results.append({
                    'method': 'HDBSCAN',
                    'n_clusters': n_clusters_hdb,
                    'n_outliers': np.sum(hdbscan_labels == -1),
                    'silhouette': silhouette_score(X[mask_hdb], hdbscan_labels[mask_hdb]),
                    'calinski_harabasz': calinski_harabasz_score(X[mask_hdb], hdbscan_labels[mask_hdb]),
                    'davies_bouldin': davies_bouldin_score(X[mask_hdb], hdbscan_labels[mask_hdb])
                })
            else:
                results.append({
                    'method': 'HDBSCAN',
                    'n_clusters': n_clusters_hdb,
                    'n_outliers': np.sum(hdbscan_labels == -1),
                    'silhouette': np.nan,
                    'calinski_harabasz': np.nan,
                    'davies_bouldin': np.nan
                })

        return pd.DataFrame(results)


if __name__ == "__main__":
    # Demo with synthetic data
    from sklearn.datasets import make_blobs

    # Generate test data
    X, _ = make_blobs(n_samples=500, centers=4, n_features=10, random_state=42)

    pipeline = ClusteringPipeline()

    # Reduce dimensions
    X_umap = pipeline.reduce_dimensions_umap(X)
    print(f"UMAP shape: {X_umap.shape}")

    # Estimate optimal K
    k_metrics = pipeline.estimate_kmeans_k(X)
    print(f"Best K: {k_metrics['best_k']}")

    # Compare methods
    comparison = pipeline.compare_clustering_methods(X, kmeans_k=4)
    print("\n=== Clustering Comparison ===")
    print(comparison)
