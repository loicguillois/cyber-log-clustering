#!/usr/bin/env python3
"""
Cyber Log Clustering - Main Pipeline

This script orchestrates the complete ML pipeline for clustering host behaviors
from cybersecurity logs. It's designed for SOC (Security Operations Center)
threat detection workflows.

Pipeline Steps:
1. Load UNSW-NB15 network traffic data
2. Engineer behavioral features per source IP
3. Preprocess and normalize features
4. Reduce dimensionality for visualization
5. Apply multiple clustering algorithms
6. Detect anomalies using ensemble methods
7. Generate visualizations and reports

Usage:
    python main.py                          # Run with default settings
    python main.py --sample 0.1             # Use 10% sample for faster testing
    python main.py --output results         # Custom output directory
    python main.py --clusters 5             # Specify number of clusters

Author: Cyber Log Clustering Project
"""

import argparse
import logging
import os
import sys
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from load_data import DataLoader
from feature_engineering import FeatureEngineer
from clustering import ClusteringPipeline
from anomaly_detection import AnomalyDetector
from visualization import Visualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Cyber Log Clustering - Host Behavior Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                        # Full pipeline with defaults
  python main.py --sample 0.1           # Quick test with 10% data
  python main.py --clusters 6           # Use 6 clusters for KMeans
  python main.py --contamination 0.1    # Expect 10% anomalies
        """
    )

    parser.add_argument(
        '--data-path',
        type=str,
        default='/Volumes/Data_IA/UNSW_NB15',
        help='Path to UNSW-NB15 dataset directory'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='outputs',
        help='Output directory for results and plots'
    )

    parser.add_argument(
        '--sample',
        type=float,
        default=None,
        help='Fraction of data to sample (0.0 to 1.0) for faster testing'
    )

    parser.add_argument(
        '--min-connections',
        type=int,
        default=5,
        help='Minimum connections per host to include'
    )

    parser.add_argument(
        '--clusters',
        type=int,
        default=None,
        help='Number of clusters for KMeans (auto-estimated if not specified)'
    )

    parser.add_argument(
        '--contamination',
        type=float,
        default=0.05,
        help='Expected proportion of anomalies (0.01 to 0.5)'
    )

    parser.add_argument(
        '--use-raw',
        action='store_true',
        help='Use raw data files instead of train/test sets'
    )

    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip plot generation'
    )

    return parser.parse_args()


def run_pipeline(args):
    """
    Execute the complete clustering pipeline.

    Returns:
        Dictionary with all results
    """
    start_time = datetime.now()
    results = {}

    logger.info("=" * 60)
    logger.info("CYBER LOG CLUSTERING PIPELINE")
    logger.info("=" * 60)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # =========================================================================
    # STEP 1: DATA LOADING
    # =========================================================================
    logger.info("\n[STEP 1/7] Loading UNSW-NB15 Dataset...")

    loader = DataLoader(data_path=args.data_path)

    try:
        raw_df = loader.load_combined_dataset(
            use_raw=args.use_raw,
            sample_frac=args.sample
        )
    except FileNotFoundError as e:
        logger.error(f"Dataset not found: {e}")
        logger.error(f"Please ensure UNSW-NB15 data is at: {args.data_path}")
        sys.exit(1)

    summary = loader.get_data_summary()
    logger.info(f"  Loaded {summary['total_records']:,} connection records")
    logger.info(f"  Memory usage: {summary['memory_usage_mb']:.2f} MB")

    if 'attack_distribution' in summary:
        dist = summary['attack_distribution']
        logger.info(f"  Normal traffic: {dist['normal']:,} | Attack traffic: {dist['attack']:,}")

    results['data_summary'] = summary

    # =========================================================================
    # STEP 2: FEATURE ENGINEERING
    # =========================================================================
    logger.info("\n[STEP 2/7] Engineering Host Behavior Features...")

    engineer = FeatureEngineer()
    host_df = engineer.aggregate_by_source_ip(
        raw_df,
        min_connections=args.min_connections
    )

    logger.info(f"  Created {len(host_df)} host profiles")
    logger.info(f"  Features per host: {len(host_df.columns)}")

    results['n_hosts'] = len(host_df)

    # =========================================================================
    # STEP 3: PREPROCESSING
    # =========================================================================
    logger.info("\n[STEP 3/7] Preprocessing Features...")

    host_df_processed, X_scaled = engineer.preprocess_features(host_df)
    feature_names = engineer.feature_columns

    logger.info(f"  Feature matrix shape: {X_scaled.shape}")
    logger.info(f"  Features: {feature_names[:5]}...")

    # Feature importance
    importance_df = engineer.get_feature_importance(X_scaled)
    logger.info(f"  Top 3 features by variance: {importance_df.head(3)['feature'].tolist()}")

    results['feature_names'] = feature_names
    results['importance'] = importance_df

    # =========================================================================
    # STEP 4: DIMENSIONALITY REDUCTION
    # =========================================================================
    logger.info("\n[STEP 4/7] Reducing Dimensionality...")

    clustering = ClusteringPipeline(random_state=42)

    # Try UMAP first, fall back to PCA
    try:
        X_2d = clustering.reduce_dimensions_umap(X_scaled, n_components=2)
        reduction_method = "UMAP"
    except Exception as e:
        logger.warning(f"UMAP failed ({e}), using PCA...")
        X_2d, _ = clustering.reduce_dimensions_pca(X_scaled, n_components=2)
        reduction_method = "PCA"

    logger.info(f"  Reduced to 2D using {reduction_method}")

    results['reduction_method'] = reduction_method
    results['X_2d'] = X_2d

    # =========================================================================
    # STEP 5: CLUSTERING
    # =========================================================================
    logger.info("\n[STEP 5/7] Clustering Host Behaviors...")

    # Estimate optimal K if not specified
    if args.clusters is None:
        logger.info("  Estimating optimal number of clusters...")
        k_metrics = clustering.estimate_kmeans_k(X_scaled, k_range=(2, 12))
        n_clusters = k_metrics['best_k']
        logger.info(f"  Optimal K estimated: {n_clusters}")
        results['k_metrics'] = k_metrics
    else:
        n_clusters = args.clusters
        k_metrics = None
        results['k_metrics'] = None

    # Compare clustering methods
    logger.info("  Running clustering algorithms...")
    comparison_df = clustering.compare_clustering_methods(
        X_scaled,
        kmeans_k=n_clusters,
        hdbscan_min_size=max(10, len(host_df) // 50)
    )

    logger.info("\n  Clustering Results:")
    for _, row in comparison_df.iterrows():
        sil = row['silhouette']
        sil_str = f"{sil:.4f}" if pd.notna(sil) else "N/A"
        logger.info(f"    {row['method']}: {row['n_clusters']} clusters, "
                    f"{row['n_outliers']} outliers, silhouette={sil_str}")

    results['comparison'] = comparison_df

    # Use HDBSCAN labels if available, otherwise KMeans
    if 'hdbscan' in clustering.cluster_labels:
        best_labels = clustering.cluster_labels['hdbscan']
        best_method = 'HDBSCAN'
    else:
        best_labels = clustering.cluster_labels['kmeans']
        best_method = 'KMeans'

    logger.info(f"  Using {best_method} labels for analysis")

    # Compute cluster statistics
    cluster_stats = clustering.compute_cluster_statistics(
        X_scaled, best_labels, feature_names, host_df_processed
    )

    # Interpret clusters
    interpretations = clustering.interpret_clusters(cluster_stats, feature_names)
    logger.info("\n  Cluster Interpretations:")
    for cluster_id, interpretation in interpretations.items():
        logger.info(f"    Cluster {cluster_id}: {interpretation}")

    results['labels'] = best_labels
    results['cluster_stats'] = cluster_stats
    results['interpretations'] = interpretations

    # =========================================================================
    # STEP 6: ANOMALY DETECTION
    # =========================================================================
    logger.info("\n[STEP 6/7] Detecting Anomalies...")

    detector = AnomalyDetector(contamination=args.contamination)

    # Run multiple detectors
    anomaly_results = detector.detect_all(
        X_scaled,
        methods=['isolation_forest', 'lof']
    )

    logger.info("\n  Anomaly Detection Results:")
    for method, data in anomaly_results.items():
        pct = data['n_anomalies'] / len(X_scaled) * 100
        logger.info(f"    {method}: {data['n_anomalies']} anomalies ({pct:.1f}%)")

    # Get ensemble scores
    ensemble_scores = anomaly_results['ensemble']['scores']

    # Get top anomalies
    top_anomalies = detector.get_top_anomalies(
        host_df_processed,
        ensemble_scores,
        top_n=20
    )

    logger.info("\n  Top 5 Most Anomalous Hosts:")
    for i, (_, row) in enumerate(top_anomalies.head(5).iterrows()):
        logger.info(f"    {i+1}. Host {row['host_ip']}: score={row['anomaly_score']:.4f}")

    # Compute anomaly statistics
    anomaly_stats = detector.compute_anomaly_statistics(
        X_scaled,
        anomaly_results['ensemble']['predictions'],
        feature_names
    )

    results['anomaly_scores'] = ensemble_scores
    results['top_anomalies'] = top_anomalies
    results['anomaly_stats'] = anomaly_stats

    # =========================================================================
    # STEP 7: VISUALIZATION
    # =========================================================================
    if not args.no_plots:
        logger.info("\n[STEP 7/7] Generating Visualizations...")

        viz = Visualizer(output_dir=args.output)

        saved_files = viz.create_full_report(
            X_2d=X_2d,
            labels=best_labels,
            anomaly_scores=ensemble_scores,
            host_df=host_df_processed,
            importance_df=importance_df,
            stats_df=cluster_stats,
            k_metrics=k_metrics,
            comparison_df=comparison_df,
            interpretations=interpretations,
            feature_names=feature_names
        )

        logger.info(f"  Saved {len(saved_files)} visualizations to {args.output}/")
        results['plots'] = saved_files
    else:
        logger.info("\n[STEP 7/7] Skipping visualizations (--no-plots)")
        results['plots'] = []

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("SAVING RESULTS")
    logger.info("=" * 60)

    # Save host features with cluster labels and anomaly scores
    output_df = host_df_processed.copy()
    output_df['cluster'] = best_labels
    output_df['anomaly_score'] = ensemble_scores
    output_df['is_anomaly'] = anomaly_results['ensemble']['predictions'] == -1

    output_path = os.path.join(args.output, "host_analysis_results.csv")
    output_df.to_csv(output_path, index=False)
    logger.info(f"  Saved host analysis to: {output_path}")

    # Save top anomalies
    anomalies_path = os.path.join(args.output, "top_anomalies.csv")
    top_anomalies.to_csv(anomalies_path, index=False)
    logger.info(f"  Saved top anomalies to: {anomalies_path}")

    # Save cluster statistics
    stats_path = os.path.join(args.output, "cluster_statistics.csv")
    cluster_stats.to_csv(stats_path, index=False)
    logger.info(f"  Saved cluster stats to: {stats_path}")

    # Save interpretations
    interp_path = os.path.join(args.output, "cluster_interpretations.json")
    with open(interp_path, 'w') as f:
        json.dump({str(k): v for k, v in interpretations.items()}, f, indent=2)
    logger.info(f"  Saved interpretations to: {interp_path}")

    # Summary
    elapsed = datetime.now() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Total hosts analyzed: {len(host_df):,}")
    logger.info(f"  Clusters found: {len(set(best_labels)) - (1 if -1 in best_labels else 0)}")
    logger.info(f"  Anomalies detected: {(best_labels == -1).sum() + anomaly_results['ensemble']['n_anomalies']}")
    logger.info(f"  Elapsed time: {elapsed}")
    logger.info(f"  Results saved to: {args.output}/")

    return results


def main():
    """Main entry point."""
    args = parse_args()

    try:
        results = run_pipeline(args)
        return 0
    except KeyboardInterrupt:
        logger.info("\nPipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"\nPipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
