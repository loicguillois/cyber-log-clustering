"""
Visualization Module for Cybersecurity Clustering Analysis

This module provides comprehensive visualizations for:
- Cluster exploration and interpretation
- Anomaly detection results
- Feature importance and distributions
- Dimensionality reduction projections

Visualizations are designed for SOC analysts and security researchers
who need to understand host behavior patterns at a glance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Tuple
import logging
import warnings
import os

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for professional plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class Visualizer:
    """
    Comprehensive visualization suite for cybersecurity clustering analysis.

    Generates publication-quality plots that help analysts understand:
    - How hosts cluster by behavior
    - Which features drive cluster separation
    - Where anomalies appear in the data
    """

    def __init__(
        self,
        output_dir: str = "outputs",
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 150
    ):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save plots
            figsize: Default figure size
            dpi: Resolution for saved figures
        """
        self.output_dir = output_dir
        self.figsize = figsize
        self.dpi = dpi

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def _save_figure(self, fig: plt.Figure, name: str) -> str:
        """Save figure to output directory."""
        path = os.path.join(self.output_dir, f"{name}.png")
        fig.savefig(path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved: {path}")
        return path

    def plot_cluster_scatter(
        self,
        X_2d: np.ndarray,
        labels: np.ndarray,
        title: str = "Host Behavior Clusters",
        method: str = "UMAP",
        interpretations: Optional[Dict[int, str]] = None,
        save: bool = True
    ) -> plt.Figure:
        """
        Create scatter plot of clustered hosts in 2D space.

        This is the primary visualization for understanding cluster structure.
        Each point is a host, colored by cluster membership.

        Args:
            X_2d: 2D coordinates from dimensionality reduction
            labels: Cluster labels
            title: Plot title
            method: Reduction method name (for axis labels)
            interpretations: Optional dict mapping cluster to description
            save: Whether to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Define colors - outliers in red
        unique_labels = sorted(set(labels))
        n_clusters = len(unique_labels) - (1 if -1 in labels else 0)

        # Color palette
        colors = sns.color_palette("husl", n_clusters)
        color_map = {}

        cluster_idx = 0
        for label in unique_labels:
            if label == -1:
                color_map[label] = 'red'  # Outliers in red
            else:
                color_map[label] = colors[cluster_idx]
                cluster_idx += 1

        # Plot each cluster
        for label in unique_labels:
            mask = labels == label
            if label == -1:
                cluster_name = "Outliers"
                marker = 'x'
                alpha = 0.8
                size = 50
            else:
                if interpretations and label in interpretations:
                    cluster_name = f"Cluster {label}: {interpretations[label]}"
                else:
                    cluster_name = f"Cluster {label}"
                marker = 'o'
                alpha = 0.6
                size = 30

            ax.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                c=[color_map[label]],
                label=f"{cluster_name} (n={mask.sum()})",
                marker=marker,
                alpha=alpha,
                s=size,
                edgecolors='white',
                linewidth=0.5
            )

        ax.set_xlabel(f"{method} Dimension 1", fontsize=12)
        ax.set_ylabel(f"{method} Dimension 2", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

        plt.tight_layout()

        if save:
            self._save_figure(fig, "cluster_scatter")

        return fig

    def plot_cluster_distribution(
        self,
        labels: np.ndarray,
        interpretations: Optional[Dict[int, str]] = None,
        save: bool = True
    ) -> plt.Figure:
        """
        Create bar chart showing cluster size distribution.

        Important for understanding:
        - Class imbalance
        - Which behaviors are most common
        - Relative size of anomaly groups

        Args:
            labels: Cluster labels
            interpretations: Optional cluster descriptions
            save: Whether to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        unique_labels, counts = np.unique(labels, return_counts=True)

        # Create labels
        bar_labels = []
        for label in unique_labels:
            if label == -1:
                bar_labels.append("Outliers")
            elif interpretations and label in interpretations:
                bar_labels.append(f"C{label}: {interpretations[label][:20]}...")
            else:
                bar_labels.append(f"Cluster {label}")

        # Colors
        colors = ['red' if l == -1 else sns.color_palette("husl", len(unique_labels))[i]
                  for i, l in enumerate(unique_labels)]

        bars = ax.bar(range(len(unique_labels)), counts, color=colors, edgecolor='black')

        ax.set_xticks(range(len(unique_labels)))
        ax.set_xticklabels(bar_labels, rotation=45, ha='right')
        ax.set_ylabel("Number of Hosts", fontsize=12)
        ax.set_title("Cluster Size Distribution", fontsize=14, fontweight='bold')

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            ax.annotate(
                f'{count:,}',
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5),
                textcoords='offset points',
                ha='center',
                fontsize=10
            )

        plt.tight_layout()

        if save:
            self._save_figure(fig, "cluster_distribution")

        return fig

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 15,
        save: bool = True
    ) -> plt.Figure:
        """
        Create bar chart of feature importance (by variance).

        Shows which features contribute most to cluster separation.
        Important for:
        - Understanding what behaviors differentiate hosts
        - Feature selection
        - Model interpretation

        Args:
            importance_df: DataFrame with feature and variance columns
            top_n: Number of top features to show
            save: Whether to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        top_features = importance_df.head(top_n).copy()
        top_features = top_features.sort_values('variance')

        colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(top_features)))

        ax.barh(
            range(len(top_features)),
            top_features['variance'],
            color=colors,
            edgecolor='black'
        )

        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'], fontsize=10)
        ax.set_xlabel("Variance (Feature Importance)", fontsize=12)
        ax.set_title("Top Features by Variance", fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save:
            self._save_figure(fig, "feature_importance")

        return fig

    def plot_anomaly_scores(
        self,
        scores: np.ndarray,
        threshold: Optional[float] = None,
        save: bool = True
    ) -> plt.Figure:
        """
        Create histogram of anomaly scores.

        Shows distribution of anomaly scores and threshold placement.
        Helps in:
        - Understanding how anomalous the population is
        - Setting investigation thresholds
        - Validating detection parameters

        Args:
            scores: Anomaly scores (0 to 1)
            threshold: Optional threshold line
            save: Whether to save figure

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        ax1 = axes[0]
        ax1.hist(scores, bins=50, color='steelblue', edgecolor='black', alpha=0.7)

        if threshold:
            ax1.axvline(
                threshold, color='red', linestyle='--',
                linewidth=2, label=f'Threshold: {threshold:.3f}'
            )
            ax1.legend()

        ax1.set_xlabel("Anomaly Score", fontsize=12)
        ax1.set_ylabel("Number of Hosts", fontsize=12)
        ax1.set_title("Anomaly Score Distribution", fontsize=14, fontweight='bold')

        # Box plot
        ax2 = axes[1]
        box = ax2.boxplot(scores, vert=True, patch_artist=True)
        box['boxes'][0].set_facecolor('steelblue')

        ax2.set_ylabel("Anomaly Score", fontsize=12)
        ax2.set_title("Anomaly Score Statistics", fontsize=14, fontweight='bold')

        # Add statistics text
        stats_text = (
            f"Mean: {np.mean(scores):.3f}\n"
            f"Median: {np.median(scores):.3f}\n"
            f"Std: {np.std(scores):.3f}\n"
            f"Max: {np.max(scores):.3f}"
        )
        ax2.text(1.3, np.median(scores), stats_text, fontsize=10,
                 verticalalignment='center')

        plt.tight_layout()

        if save:
            self._save_figure(fig, "anomaly_scores")

        return fig

    def plot_anomaly_scatter(
        self,
        X_2d: np.ndarray,
        scores: np.ndarray,
        threshold: Optional[float] = None,
        method: str = "UMAP",
        save: bool = True
    ) -> plt.Figure:
        """
        Create scatter plot colored by anomaly score.

        Visualizes where anomalies appear in the behavior space.
        Red/orange points are more anomalous.

        Args:
            X_2d: 2D coordinates
            scores: Anomaly scores
            threshold: Optional threshold for binary coloring
            method: Dimensionality reduction method name
            save: Whether to save figure

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)

        # Sort by score so anomalies are plotted on top
        order = np.argsort(scores)

        scatter = ax.scatter(
            X_2d[order, 0],
            X_2d[order, 1],
            c=scores[order],
            cmap='RdYlBu_r',
            alpha=0.7,
            s=30,
            edgecolors='white',
            linewidth=0.3
        )

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Anomaly Score', fontsize=12)

        # Mark threshold if provided
        if threshold:
            anomaly_mask = scores > threshold
            ax.scatter(
                X_2d[anomaly_mask, 0],
                X_2d[anomaly_mask, 1],
                facecolors='none',
                edgecolors='red',
                s=100,
                linewidth=2,
                label=f'Anomalies (n={anomaly_mask.sum()})'
            )
            ax.legend()

        ax.set_xlabel(f"{method} Dimension 1", fontsize=12)
        ax.set_ylabel(f"{method} Dimension 2", fontsize=12)
        ax.set_title("Anomaly Detection Results", fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save:
            self._save_figure(fig, "anomaly_scatter")

        return fig

    def plot_cluster_feature_heatmap(
        self,
        stats_df: pd.DataFrame,
        feature_names: List[str],
        top_n_features: int = 15,
        save: bool = True
    ) -> plt.Figure:
        """
        Create heatmap showing feature means across clusters.

        Helps interpret what makes each cluster unique.
        Red = high values, Blue = low values (relative to other clusters).

        Args:
            stats_df: Cluster statistics DataFrame
            feature_names: List of feature names
            top_n_features: Number of features to show
            save: Whether to save figure

        Returns:
            Matplotlib figure
        """
        # Extract mean columns
        mean_cols = [f'{f}_mean' for f in feature_names if f'{f}_mean' in stats_df.columns]
        mean_cols = mean_cols[:top_n_features]

        if not mean_cols:
            logger.warning("No mean columns found in stats_df")
            return None

        # Create matrix
        heatmap_data = stats_df[['cluster'] + mean_cols].set_index('cluster')
        heatmap_data.columns = [c.replace('_mean', '') for c in heatmap_data.columns]

        # Normalize across clusters for better visualization
        heatmap_normalized = (heatmap_data - heatmap_data.mean()) / heatmap_data.std()

        fig, ax = plt.subplots(figsize=(14, 8))

        sns.heatmap(
            heatmap_normalized.T,
            cmap='RdBu_r',
            center=0,
            annot=False,
            fmt='.2f',
            ax=ax,
            cbar_kws={'label': 'Normalized Value (z-score)'}
        )

        ax.set_xlabel("Cluster", fontsize=12)
        ax.set_ylabel("Feature", fontsize=12)
        ax.set_title("Cluster Feature Profiles (Normalized)", fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save:
            self._save_figure(fig, "cluster_heatmap")

        return fig

    def plot_elbow_analysis(
        self,
        k_metrics: Dict[str, List],
        save: bool = True
    ) -> plt.Figure:
        """
        Create elbow plot for KMeans K selection.

        Shows three metrics:
        - Inertia (elbow method)
        - Silhouette score
        - Calinski-Harabasz index

        Args:
            k_metrics: Dictionary from estimate_kmeans_k
            save: Whether to save figure

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        k_values = k_metrics['k_values']

        # Inertia (elbow)
        axes[0].plot(k_values, k_metrics['inertias'], 'bo-', linewidth=2, markersize=8)
        axes[0].set_xlabel("Number of Clusters (K)")
        axes[0].set_ylabel("Inertia")
        axes[0].set_title("Elbow Method")

        # Silhouette score
        axes[1].plot(k_values, k_metrics['silhouettes'], 'go-', linewidth=2, markersize=8)
        best_k = k_metrics.get('best_k', k_values[np.argmax(k_metrics['silhouettes'])])
        axes[1].axvline(best_k, color='red', linestyle='--', label=f'Best K={best_k}')
        axes[1].set_xlabel("Number of Clusters (K)")
        axes[1].set_ylabel("Silhouette Score")
        axes[1].set_title("Silhouette Analysis")
        axes[1].legend()

        # Calinski-Harabasz
        axes[2].plot(k_values, k_metrics['calinski'], 'ro-', linewidth=2, markersize=8)
        axes[2].set_xlabel("Number of Clusters (K)")
        axes[2].set_ylabel("Calinski-Harabasz Index")
        axes[2].set_title("Calinski-Harabasz Index")

        plt.tight_layout()

        if save:
            self._save_figure(fig, "elbow_analysis")

        return fig

    def plot_method_comparison(
        self,
        comparison_df: pd.DataFrame,
        save: bool = True
    ) -> plt.Figure:
        """
        Create comparison chart of clustering methods.

        Shows metrics for KMeans, DBSCAN, HDBSCAN side by side.

        Args:
            comparison_df: DataFrame from compare_clustering_methods
            save: Whether to save figure

        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        methods = comparison_df['method'].tolist()
        x = range(len(methods))

        # Number of clusters
        axes[0, 0].bar(x, comparison_df['n_clusters'], color='steelblue', edgecolor='black')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(methods)
        axes[0, 0].set_ylabel("Number of Clusters")
        axes[0, 0].set_title("Clusters Found")

        # Number of outliers
        axes[0, 1].bar(x, comparison_df['n_outliers'], color='indianred', edgecolor='black')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(methods)
        axes[0, 1].set_ylabel("Number of Outliers")
        axes[0, 1].set_title("Outliers Detected")

        # Silhouette score
        silhouettes = comparison_df['silhouette'].fillna(0)
        axes[1, 0].bar(x, silhouettes, color='seagreen', edgecolor='black')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(methods)
        axes[1, 0].set_ylabel("Silhouette Score")
        axes[1, 0].set_title("Silhouette Score (higher = better)")

        # Davies-Bouldin (lower is better)
        db_scores = comparison_df['davies_bouldin'].fillna(0)
        axes[1, 1].bar(x, db_scores, color='darkorange', edgecolor='black')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(methods)
        axes[1, 1].set_ylabel("Davies-Bouldin Index")
        axes[1, 1].set_title("Davies-Bouldin Index (lower = better)")

        plt.tight_layout()

        if save:
            self._save_figure(fig, "method_comparison")

        return fig

    def plot_attack_by_cluster(
        self,
        host_df: pd.DataFrame,
        labels: np.ndarray,
        save: bool = True
    ) -> plt.Figure:
        """
        Create stacked bar chart showing attack distribution per cluster.

        Validates clustering by showing if attacks concentrate in specific clusters.

        Args:
            host_df: Host features DataFrame with attack_ratio column
            labels: Cluster labels
            save: Whether to save figure

        Returns:
            Matplotlib figure
        """
        if 'attack_ratio' not in host_df.columns:
            logger.warning("attack_ratio column not found")
            return None

        fig, ax = plt.subplots(figsize=(10, 6))

        df = host_df.copy()
        df['cluster'] = labels

        # Calculate attack statistics per cluster
        cluster_stats = df.groupby('cluster').agg({
            'attack_ratio': 'mean'
        }).reset_index()

        cluster_stats = cluster_stats.sort_values('cluster')

        # Create labels
        bar_labels = [f"Cluster {c}" if c != -1 else "Outliers"
                      for c in cluster_stats['cluster']]

        colors = ['red' if c == -1 else sns.color_palette("husl", len(cluster_stats))[i]
                  for i, c in enumerate(cluster_stats['cluster'])]

        bars = ax.bar(
            range(len(cluster_stats)),
            cluster_stats['attack_ratio'] * 100,
            color=colors,
            edgecolor='black'
        )

        ax.set_xticks(range(len(cluster_stats)))
        ax.set_xticklabels(bar_labels, rotation=45, ha='right')
        ax.set_ylabel("Attack Traffic Percentage (%)", fontsize=12)
        ax.set_title("Attack Rate by Cluster", fontsize=14, fontweight='bold')
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
        ax.legend()

        # Add value labels
        for bar, ratio in zip(bars, cluster_stats['attack_ratio']):
            ax.annotate(
                f'{ratio*100:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5),
                textcoords='offset points',
                ha='center',
                fontsize=10
            )

        plt.tight_layout()

        if save:
            self._save_figure(fig, "attack_by_cluster")

        return fig

    def create_full_report(
        self,
        X_2d: np.ndarray,
        labels: np.ndarray,
        anomaly_scores: np.ndarray,
        host_df: pd.DataFrame,
        importance_df: pd.DataFrame,
        stats_df: pd.DataFrame,
        k_metrics: Optional[Dict] = None,
        comparison_df: Optional[pd.DataFrame] = None,
        interpretations: Optional[Dict[int, str]] = None,
        feature_names: Optional[List[str]] = None
    ) -> List[str]:
        """
        Generate all visualizations and save to output directory.

        Creates a complete visual report for the clustering analysis.

        Returns:
            List of saved file paths
        """
        logger.info("Generating full visualization report...")
        saved_files = []

        # Cluster scatter
        fig = self.plot_cluster_scatter(X_2d, labels, interpretations=interpretations)
        saved_files.append(os.path.join(self.output_dir, "cluster_scatter.png"))
        plt.close(fig)

        # Cluster distribution
        fig = self.plot_cluster_distribution(labels, interpretations=interpretations)
        saved_files.append(os.path.join(self.output_dir, "cluster_distribution.png"))
        plt.close(fig)

        # Feature importance
        fig = self.plot_feature_importance(importance_df)
        saved_files.append(os.path.join(self.output_dir, "feature_importance.png"))
        plt.close(fig)

        # Anomaly scores
        threshold = np.percentile(anomaly_scores, 95)
        fig = self.plot_anomaly_scores(anomaly_scores, threshold=threshold)
        saved_files.append(os.path.join(self.output_dir, "anomaly_scores.png"))
        plt.close(fig)

        # Anomaly scatter
        fig = self.plot_anomaly_scatter(X_2d, anomaly_scores, threshold=threshold)
        saved_files.append(os.path.join(self.output_dir, "anomaly_scatter.png"))
        plt.close(fig)

        # Cluster heatmap
        if feature_names:
            fig = self.plot_cluster_feature_heatmap(stats_df, feature_names)
            if fig:
                saved_files.append(os.path.join(self.output_dir, "cluster_heatmap.png"))
                plt.close(fig)

        # Elbow analysis
        if k_metrics:
            fig = self.plot_elbow_analysis(k_metrics)
            saved_files.append(os.path.join(self.output_dir, "elbow_analysis.png"))
            plt.close(fig)

        # Method comparison
        if comparison_df is not None:
            fig = self.plot_method_comparison(comparison_df)
            saved_files.append(os.path.join(self.output_dir, "method_comparison.png"))
            plt.close(fig)

        # Attack by cluster
        fig = self.plot_attack_by_cluster(host_df, labels)
        if fig:
            saved_files.append(os.path.join(self.output_dir, "attack_by_cluster.png"))
            plt.close(fig)

        logger.info(f"Generated {len(saved_files)} visualizations")
        return saved_files


if __name__ == "__main__":
    # Demo with synthetic data
    from sklearn.datasets import make_blobs

    # Generate test data
    X, y = make_blobs(n_samples=500, centers=4, n_features=10, random_state=42)

    # Mock 2D projection
    X_2d = X[:, :2]

    # Mock labels
    labels = y

    # Mock anomaly scores
    anomaly_scores = np.random.random(500)

    # Create visualizer
    viz = Visualizer(output_dir="outputs")

    # Test plots
    viz.plot_cluster_scatter(X_2d, labels)
    viz.plot_cluster_distribution(labels)
    viz.plot_anomaly_scores(anomaly_scores)

    print("Demo visualizations saved to outputs/")
