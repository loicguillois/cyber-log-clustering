"""
Anomaly Detection Module for Cybersecurity Host Analysis

This module implements anomaly detection algorithms to identify hosts with
suspicious or unusual behavior patterns. Anomaly detection complements
clustering by explicitly finding outliers.

ANOMALY DETECTION IN CYBERSECURITY:
===================================

Anomaly detection is critical for SOC operations because:
1. Zero-day attacks have no signature - must detect behavioral anomalies
2. Insider threats often show subtle behavioral deviations
3. APTs (Advanced Persistent Threats) try to blend in but have detectable patterns

ALGORITHMS IMPLEMENTED:
=======================

1. ISOLATION FOREST
   - How it works: Randomly partitions data; anomalies are easier to isolate
   - Pros: Fast, handles high dimensions well, works with any data distribution
   - Cons: Contamination parameter must be estimated
   - Best for: General anomaly detection, when anomaly rate is unknown

2. LOCAL OUTLIER FACTOR (LOF)
   - How it works: Measures local density deviation from neighbors
   - Pros: Detects local anomalies, handles varying densities
   - Cons: Slower for large datasets, sensitive to k parameter
   - Best for: Detecting contextual anomalies (unusual within their neighborhood)

3. ONE-CLASS SVM
   - How it works: Learns a decision boundary around normal data
   - Pros: Works well with clear "normal" region
   - Cons: Slow for large datasets, kernel selection matters
   - Best for: When you have clean normal data for training

ANOMALY SCORING:
================
- Scores are normalized to [0, 1] range
- Higher scores = more anomalous
- Thresholds can be tuned based on SOC capacity (how many alerts can be investigated)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional, Dict, List
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnomalyDetector:
    """
    Multi-method anomaly detector for identifying suspicious hosts.

    Provides multiple detection algorithms and ensemble methods
    to improve detection reliability in cybersecurity applications.
    """

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        """
        Initialize anomaly detector.

        Args:
            contamination: Expected proportion of anomalies (0.0 to 0.5)
                          In cybersecurity, typically 1-10% of traffic is malicious
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.models = {}
        self.scores = {}
        self.predictions = {}

    def fit_isolation_forest(
        self,
        X: np.ndarray,
        n_estimators: int = 100,
        max_samples: str = 'auto',
        contamination: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using Isolation Forest.

        Isolation Forest isolates anomalies by randomly selecting features
        and split values. Anomalies are "few and different" - they require
        fewer splits to isolate.

        For cybersecurity:
        - Effective at finding hosts with unusual behavior combinations
        - Works well even when normal behavior is complex
        - Scales well to large host populations

        Args:
            X: Feature matrix
            n_estimators: Number of isolation trees
            max_samples: Samples for each tree ('auto' = min(256, n_samples))
            contamination: Override default contamination rate

        Returns:
            Tuple of (predictions [-1=anomaly, 1=normal], anomaly scores)
        """
        if contamination is None:
            contamination = self.contamination

        logger.info(f"Fitting Isolation Forest (contamination={contamination:.2%})...")

        iforest = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=self.random_state,
            n_jobs=-1
        )

        predictions = iforest.fit_predict(X)

        # Get anomaly scores (more negative = more anomalous)
        raw_scores = iforest.decision_function(X)

        # Normalize scores to [0, 1] where 1 = most anomalous
        scores = self._normalize_scores(-raw_scores)

        self.models['isolation_forest'] = iforest
        self.predictions['isolation_forest'] = predictions
        self.scores['isolation_forest'] = scores

        n_anomalies = np.sum(predictions == -1)
        logger.info(f"  Anomalies detected: {n_anomalies} ({n_anomalies/len(X):.2%})")

        return predictions, scores

    def fit_local_outlier_factor(
        self,
        X: np.ndarray,
        n_neighbors: int = 20,
        contamination: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using Local Outlier Factor (LOF).

        LOF measures the local deviation of a data point's density compared
        to its neighbors. Points with substantially lower density than their
        neighbors are considered outliers.

        For cybersecurity:
        - Good at finding hosts that are unusual "for their type"
        - Detects contextual anomalies (e.g., a server behaving like a client)
        - Handles clusters of different densities

        The k-neighbors parameter:
        - Small k: More sensitive to local anomalies
        - Large k: More robust but may miss local patterns

        Args:
            X: Feature matrix
            n_neighbors: Number of neighbors for local density estimation
            contamination: Override default contamination rate

        Returns:
            Tuple of (predictions, anomaly scores)
        """
        if contamination is None:
            contamination = self.contamination

        logger.info(f"Fitting Local Outlier Factor (n_neighbors={n_neighbors})...")

        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=False,  # Use for outlier detection (not novelty)
            n_jobs=-1
        )

        predictions = lof.fit_predict(X)

        # LOF scores: negative outlier factor (more negative = more anomalous)
        raw_scores = lof.negative_outlier_factor_

        # Normalize scores to [0, 1] where 1 = most anomalous
        scores = self._normalize_scores(-raw_scores)

        self.models['lof'] = lof
        self.predictions['lof'] = predictions
        self.scores['lof'] = scores

        n_anomalies = np.sum(predictions == -1)
        logger.info(f"  Anomalies detected: {n_anomalies} ({n_anomalies/len(X):.2%})")

        return predictions, scores

    def fit_one_class_svm(
        self,
        X: np.ndarray,
        kernel: str = 'rbf',
        nu: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using One-Class SVM.

        One-Class SVM learns a decision boundary that separates the data
        from the origin in a high-dimensional feature space. Points outside
        the boundary are anomalies.

        For cybersecurity:
        - Works well when you have a clean training set of normal behavior
        - Can capture complex non-linear boundaries
        - Less suitable for high-dimensional sparse data

        Note: Slower than other methods for large datasets.

        Args:
            X: Feature matrix
            kernel: Kernel type ('rbf', 'linear', 'poly')
            nu: Upper bound on fraction of anomalies (0, 1]

        Returns:
            Tuple of (predictions, anomaly scores)
        """
        if nu is None:
            nu = self.contamination

        logger.info(f"Fitting One-Class SVM (kernel={kernel}, nu={nu:.2f})...")

        # Sample if dataset is too large (OCSVM is slow)
        if len(X) > 10000:
            logger.info(f"  Sampling to 10000 points for efficiency...")
            indices = np.random.choice(len(X), 10000, replace=False)
            X_train = X[indices]
        else:
            X_train = X

        ocsvm = OneClassSVM(
            kernel=kernel,
            nu=nu,
            gamma='scale'
        )

        ocsvm.fit(X_train)
        predictions = ocsvm.predict(X)

        # Get decision function scores
        raw_scores = ocsvm.decision_function(X)

        # Normalize scores to [0, 1] where 1 = most anomalous
        scores = self._normalize_scores(-raw_scores)

        self.models['ocsvm'] = ocsvm
        self.predictions['ocsvm'] = predictions
        self.scores['ocsvm'] = scores

        n_anomalies = np.sum(predictions == -1)
        logger.info(f"  Anomalies detected: {n_anomalies} ({n_anomalies/len(X):.2%})")

        return predictions, scores

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize anomaly scores to [0, 1] range.

        Normalization allows:
        - Comparison across different algorithms
        - Intuitive interpretation (higher = more suspicious)
        - Threshold setting based on percentiles

        Args:
            scores: Raw anomaly scores

        Returns:
            Normalized scores in [0, 1]
        """
        scaler = MinMaxScaler()
        return scaler.fit_transform(scores.reshape(-1, 1)).flatten()

    def ensemble_scores(
        self,
        methods: List[str] = ['isolation_forest', 'lof'],
        weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Combine anomaly scores from multiple methods.

        Ensemble methods improve detection by:
        - Reducing false positives (consensus across methods)
        - Capturing different types of anomalies
        - Providing more robust rankings

        Args:
            methods: List of methods to combine
            weights: Optional weights for each method

        Returns:
            Combined anomaly scores
        """
        logger.info(f"Creating ensemble from: {methods}")

        if weights is None:
            weights = [1.0 / len(methods)] * len(methods)

        ensemble_score = np.zeros(len(self.scores[methods[0]]))

        for method, weight in zip(methods, weights):
            if method in self.scores:
                ensemble_score += weight * self.scores[method]
            else:
                logger.warning(f"Method {method} not found in scores")

        # Normalize final ensemble score
        ensemble_score = self._normalize_scores(ensemble_score)
        self.scores['ensemble'] = ensemble_score

        return ensemble_score

    def get_top_anomalies(
        self,
        host_df: pd.DataFrame,
        scores: np.ndarray,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Get the top N most anomalous hosts with their features.

        This is what SOC analysts need - a prioritized list of
        hosts to investigate, with context about why they're flagged.

        Args:
            host_df: Original host features DataFrame
            scores: Anomaly scores
            top_n: Number of top anomalies to return

        Returns:
            DataFrame with top anomalies and their features
        """
        result = host_df.copy()
        result['anomaly_score'] = scores
        result['anomaly_rank'] = np.argsort(np.argsort(-scores)) + 1

        top_anomalies = result.nlargest(top_n, 'anomaly_score')

        logger.info(f"Top {top_n} anomalies by score:")
        for i, row in top_anomalies.head(5).iterrows():
            logger.info(f"  Rank {int(row['anomaly_rank'])}: Score={row['anomaly_score']:.4f}")

        return top_anomalies

    def compute_anomaly_statistics(
        self,
        X: np.ndarray,
        predictions: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """
        Compare feature distributions between normal and anomalous hosts.

        Understanding HOW anomalies differ from normal hosts is crucial
        for:
        - Validating that detections are meaningful
        - Understanding attack patterns
        - Tuning detection parameters

        Args:
            X: Feature matrix
            predictions: Binary predictions (-1=anomaly, 1=normal)
            feature_names: Names of features

        Returns:
            Dictionary with comparison statistics
        """
        mask_anomaly = predictions == -1
        mask_normal = predictions == 1

        X_anomaly = X[mask_anomaly]
        X_normal = X[mask_normal]

        comparison = []

        for i, feat_name in enumerate(feature_names):
            if i < X.shape[1]:
                comparison.append({
                    'feature': feat_name,
                    'normal_mean': np.mean(X_normal[:, i]) if len(X_normal) > 0 else 0,
                    'normal_std': np.std(X_normal[:, i]) if len(X_normal) > 0 else 0,
                    'anomaly_mean': np.mean(X_anomaly[:, i]) if len(X_anomaly) > 0 else 0,
                    'anomaly_std': np.std(X_anomaly[:, i]) if len(X_anomaly) > 0 else 0,
                })

        comparison_df = pd.DataFrame(comparison)

        # Calculate difference ratio
        comparison_df['diff_ratio'] = (
            comparison_df['anomaly_mean'] - comparison_df['normal_mean']
        ).abs() / (comparison_df['normal_std'] + 1e-10)

        comparison_df = comparison_df.sort_values('diff_ratio', ascending=False)

        return {
            'feature_comparison': comparison_df,
            'n_anomalies': int(mask_anomaly.sum()),
            'n_normal': int(mask_normal.sum()),
            'anomaly_rate': float(mask_anomaly.sum() / len(predictions))
        }

    def detect_all(
        self,
        X: np.ndarray,
        methods: List[str] = ['isolation_forest', 'lof']
    ) -> Dict[str, np.ndarray]:
        """
        Run all specified anomaly detection methods.

        Convenience method to run multiple detectors and return
        all results for comparison.

        Args:
            X: Feature matrix
            methods: List of methods to run

        Returns:
            Dictionary with predictions and scores for each method
        """
        results = {}

        for method in methods:
            if method == 'isolation_forest':
                preds, scores = self.fit_isolation_forest(X)
            elif method == 'lof':
                preds, scores = self.fit_local_outlier_factor(X)
            elif method == 'ocsvm':
                preds, scores = self.fit_one_class_svm(X)
            else:
                logger.warning(f"Unknown method: {method}")
                continue

            results[method] = {
                'predictions': preds,
                'scores': scores,
                'n_anomalies': np.sum(preds == -1)
            }

        # Create ensemble if multiple methods
        if len(methods) > 1:
            ensemble_scores = self.ensemble_scores(methods)
            threshold = np.percentile(ensemble_scores, 100 * (1 - self.contamination))
            ensemble_preds = np.where(ensemble_scores > threshold, -1, 1)

            results['ensemble'] = {
                'predictions': ensemble_preds,
                'scores': ensemble_scores,
                'n_anomalies': np.sum(ensemble_preds == -1)
            }

        return results

    def explain_anomaly(
        self,
        host_features: pd.Series,
        feature_names: List[str],
        normal_stats: pd.DataFrame
    ) -> List[str]:
        """
        Generate human-readable explanation for why a host is anomalous.

        Critical for SOC analysts who need to understand and act on alerts.

        Args:
            host_features: Features of the anomalous host
            feature_names: Names of features
            normal_stats: Statistics of normal hosts

        Returns:
            List of explanation strings
        """
        explanations = []

        for feat_name in feature_names:
            if feat_name not in host_features:
                continue

            value = host_features[feat_name]
            normal_row = normal_stats[normal_stats['feature'] == feat_name]

            if len(normal_row) == 0:
                continue

            normal_mean = normal_row['normal_mean'].values[0]
            normal_std = normal_row['normal_std'].values[0]

            if normal_std > 0:
                z_score = (value - normal_mean) / normal_std

                if abs(z_score) > 3:
                    direction = "higher" if z_score > 0 else "lower"
                    explanations.append(
                        f"{feat_name}: {value:.2f} is {abs(z_score):.1f}x std {direction} than normal"
                    )

        return explanations


if __name__ == "__main__":
    # Demo with synthetic data
    from sklearn.datasets import make_blobs

    # Generate test data with some outliers
    X_normal, _ = make_blobs(n_samples=1000, centers=3, n_features=10, random_state=42)
    X_outliers = np.random.uniform(-10, 10, (50, 10))
    X = np.vstack([X_normal, X_outliers])

    detector = AnomalyDetector(contamination=0.05)

    # Run all methods
    results = detector.detect_all(X, methods=['isolation_forest', 'lof'])

    print("\n=== Anomaly Detection Results ===")
    for method, data in results.items():
        print(f"{method}: {data['n_anomalies']} anomalies detected")

    # Get top anomalies
    feature_names = [f"feature_{i}" for i in range(10)]
    host_df = pd.DataFrame(X, columns=feature_names)

    top = detector.get_top_anomalies(host_df, results['ensemble']['scores'], top_n=10)
    print(f"\nTop 10 anomaly scores: {top['anomaly_score'].values[:5]}")
