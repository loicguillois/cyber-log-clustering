"""
Feature Engineering Module for Host Behavior Analysis

This module transforms connection-level network data into host-level behavioral
features suitable for unsupervised clustering. The goal is to characterize
each source IP's behavior profile to detect compromised or malicious hosts.

CYBERSECURITY RATIONALE FOR FEATURES:
=====================================

1. CONNECTION VOLUME FEATURES
   - n_connections: High connection counts may indicate scanning or C2 beaconing
   - n_unique_dsts: Many destinations suggest reconnaissance or lateral movement
   - n_unique_ports: Port variety indicates scanning or service enumeration

2. TRAFFIC VOLUME FEATURES
   - total_bytes_sent/received: Data exfiltration shows high outbound bytes
   - bytes_ratio: Normal clients have ratio < 1, servers > 1
   - Asymmetric ratios may indicate data theft or DDoS amplification

3. TEMPORAL FEATURES
   - Connection duration stats: Long-lived connections may be tunnels or backdoors
   - Short bursts of connections: May indicate brute force or scanning

4. PROTOCOL DISTRIBUTION
   - Protocol entropy: Diverse protocols may indicate enumeration
   - Unusual protocol usage: DNS over non-standard ports, ICMP tunneling

5. DESTINATION ENTROPY
   - Measures randomness of destination IPs
   - High entropy: Scanning or worm propagation
   - Low entropy: Normal client behavior (few regular servers)

6. ATTACK INDICATORS
   - Connection states: RST floods, incomplete handshakes
   - TTL anomalies: May indicate spoofing or tunneling
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List, Optional
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Transforms connection-level data to host-level behavioral features.

    The core idea: Instead of analyzing individual connections, we aggregate
    all connections from each source IP to build a "behavioral profile".
    This profile reveals patterns that indicate normal vs suspicious activity.
    """

    def __init__(self):
        """Initialize the feature engineer with default settings."""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns: List[str] = []
        self.host_features: Optional[pd.DataFrame] = None

    def _calculate_entropy(self, series: pd.Series) -> float:
        """
        Calculate Shannon entropy of a categorical series.

        Entropy measures the randomness/diversity of values.
        - High entropy (near log2(n)): Many equally distributed values
        - Low entropy (near 0): Few dominant values

        In cybersecurity:
        - Scanner hosts have HIGH destination IP entropy
        - Normal hosts have LOW entropy (connect to few servers)

        Args:
            series: Categorical series to analyze

        Returns:
            Shannon entropy value (bits)
        """
        if len(series) == 0:
            return 0.0

        value_counts = series.value_counts(normalize=True)
        entropy = stats.entropy(value_counts, base=2)
        return entropy

    def _safe_div(self, a: float, b: float, default: float = 0.0) -> float:
        """Safe division handling zero denominators."""
        return a / b if b != 0 else default

    def aggregate_by_source_ip(
        self,
        df: pd.DataFrame,
        min_connections: int = 5,
        src_col: str = 'srcip'
    ) -> pd.DataFrame:
        """
        Aggregate connection data by source IP to create host profiles.

        This is the core feature engineering step. We transform millions of
        individual network flows into host-level behavioral summaries.

        The min_connections threshold filters out hosts with too few
        connections to establish a meaningful behavioral pattern.

        Args:
            df: Connection-level DataFrame
            min_connections: Minimum connections required per host
            src_col: Column name for source IP

        Returns:
            DataFrame with one row per host and behavioral features
        """
        logger.info(f"Aggregating features by source IP (min_connections={min_connections})...")

        # Determine available columns (train/test vs raw format)
        has_srcip = 'srcip' in df.columns
        has_dstip = 'dstip' in df.columns

        # If using train/test set format (no srcip), we'll use connection features directly
        if not has_srcip:
            logger.warning("No 'srcip' column found - creating synthetic host IDs")
            # Create bins based on traffic patterns for demo purposes
            df = df.copy()
            # Use a hash of features to create pseudo-host IDs
            df['srcip'] = (df.index // 100).astype(str).str.zfill(6)

        # Pre-filter valid data
        df_clean = df.dropna(subset=[src_col]).copy()

        # Group by source IP and compute aggregated features
        host_groups = df_clean.groupby(src_col)

        logger.info(f"Processing {len(host_groups)} unique source IPs...")

        # Initialize feature dictionary
        features_list = []

        for host_ip, group in host_groups:
            if len(group) < min_connections:
                continue

            features = {'host_ip': host_ip}

            # === CONNECTION VOLUME FEATURES ===
            # n_connections: Total number of connections from this host
            # High values may indicate: scanning, worm propagation, or server behavior
            features['n_connections'] = len(group)

            # n_unique_dsts: Number of unique destination IPs contacted
            # High values indicate: reconnaissance, lateral movement, or P2P
            if 'dstip' in group.columns:
                features['n_unique_dsts'] = group['dstip'].nunique()
            elif 'ct_dst_src_ltm' in group.columns:
                features['n_unique_dsts'] = group['ct_dst_src_ltm'].mean()

            # n_unique_dst_ports: Number of unique destination ports
            # High values indicate: port scanning or service enumeration
            if 'dsport' in group.columns:
                features['n_unique_dst_ports'] = group['dsport'].nunique()
            else:
                features['n_unique_dst_ports'] = 0

            # === TRAFFIC VOLUME FEATURES ===
            # sbytes/dbytes: Source/Destination bytes
            # Data exfiltration: high sbytes, low dbytes
            # DDoS target: low sbytes, high dbytes

            if 'sbytes' in group.columns:
                features['total_bytes_sent'] = group['sbytes'].sum()
                features['avg_bytes_sent'] = group['sbytes'].mean()
                features['std_bytes_sent'] = group['sbytes'].std()
            else:
                features['total_bytes_sent'] = 0
                features['avg_bytes_sent'] = 0
                features['std_bytes_sent'] = 0

            if 'dbytes' in group.columns:
                features['total_bytes_received'] = group['dbytes'].sum()
                features['avg_bytes_received'] = group['dbytes'].mean()
            else:
                features['total_bytes_received'] = 0
                features['avg_bytes_received'] = 0

            # bytes_ratio: sent / received
            # Normal clients: ratio < 1 (download more than upload)
            # Data exfiltration: ratio >> 1
            features['bytes_ratio'] = self._safe_div(
                features['total_bytes_sent'],
                features['total_bytes_received'],
                default=1.0
            )

            # === PACKET FEATURES ===
            if 'spkts' in group.columns:
                features['total_packets_sent'] = group['spkts'].sum()
                features['avg_packets_sent'] = group['spkts'].mean()
            else:
                features['total_packets_sent'] = 0
                features['avg_packets_sent'] = 0

            if 'dpkts' in group.columns:
                features['total_packets_received'] = group['dpkts'].sum()
            else:
                features['total_packets_received'] = 0

            features['packets_ratio'] = self._safe_div(
                features['total_packets_sent'],
                features['total_packets_received'],
                default=1.0
            )

            # === TEMPORAL FEATURES ===
            # dur: Connection duration
            # Long durations: Persistent connections (backdoors, tunnels)
            # Very short: SYN scans, connection refused

            if 'dur' in group.columns:
                features['avg_duration'] = group['dur'].mean()
                features['std_duration'] = group['dur'].std()
                features['max_duration'] = group['dur'].max()
                features['min_duration'] = group['dur'].min()
            else:
                features['avg_duration'] = 0
                features['std_duration'] = 0
                features['max_duration'] = 0
                features['min_duration'] = 0

            # === PROTOCOL DISTRIBUTION ===
            # Different protocols have different attack signatures
            # DNS amplification, ICMP tunneling, etc.

            if 'proto' in group.columns:
                proto_counts = group['proto'].value_counts(normalize=True)
                features['proto_entropy'] = self._calculate_entropy(group['proto'])
                features['tcp_ratio'] = proto_counts.get('tcp', 0)
                features['udp_ratio'] = proto_counts.get('udp', 0)
                features['icmp_ratio'] = proto_counts.get('icmp', 0)
            else:
                features['proto_entropy'] = 0
                features['tcp_ratio'] = 0
                features['udp_ratio'] = 0
                features['icmp_ratio'] = 0

            # === DESTINATION ENTROPY ===
            # Measures how "spread out" the host's destinations are
            # High entropy: Scanning many different targets
            # Low entropy: Normal behavior (connecting to few servers)

            if 'dstip' in group.columns:
                features['dst_entropy'] = self._calculate_entropy(group['dstip'])
            elif 'ct_dst_ltm' in group.columns:
                # Use proxy metric from aggregate features
                features['dst_entropy'] = np.log2(group['ct_dst_ltm'].mean() + 1)
            else:
                features['dst_entropy'] = 0

            if 'dsport' in group.columns:
                features['port_entropy'] = self._calculate_entropy(group['dsport'])
            else:
                features['port_entropy'] = 0

            # === SERVICE DISTRIBUTION ===
            if 'service' in group.columns:
                features['n_unique_services'] = group['service'].nunique()
                features['service_entropy'] = self._calculate_entropy(group['service'])
            else:
                features['n_unique_services'] = 0
                features['service_entropy'] = 0

            # === CONNECTION STATE FEATURES ===
            # RST: Connection reset (possible scan detection)
            # FIN: Normal closure
            # CON: Established connection

            if 'state' in group.columns:
                state_counts = group['state'].value_counts(normalize=True)
                features['state_entropy'] = self._calculate_entropy(group['state'])
                features['rst_ratio'] = state_counts.get('RST', 0) + state_counts.get('RSTOS0', 0)
                features['fin_ratio'] = state_counts.get('FIN', 0)
                features['established_ratio'] = state_counts.get('CON', 0) + state_counts.get('ECO', 0)
            else:
                features['state_entropy'] = 0
                features['rst_ratio'] = 0
                features['fin_ratio'] = 0
                features['established_ratio'] = 0

            # === TTL FEATURES ===
            # TTL anomalies may indicate IP spoofing or tunneling

            if 'sttl' in group.columns:
                features['avg_sttl'] = group['sttl'].mean()
                features['std_sttl'] = group['sttl'].std()
            else:
                features['avg_sttl'] = 0
                features['std_sttl'] = 0

            if 'dttl' in group.columns:
                features['avg_dttl'] = group['dttl'].mean()
            else:
                features['avg_dttl'] = 0

            # === LOAD FEATURES ===
            # sload/dload: bits per second
            # High load: possible DDoS or heavy data transfer

            if 'sload' in group.columns:
                features['avg_sload'] = group['sload'].mean()
                features['max_sload'] = group['sload'].max()
            else:
                features['avg_sload'] = 0
                features['max_sload'] = 0

            if 'dload' in group.columns:
                features['avg_dload'] = group['dload'].mean()
            else:
                features['avg_dload'] = 0

            # === LOSS FEATURES ===
            # Packet loss may indicate: congestion, DDoS, or poor network

            if 'sloss' in group.columns:
                features['total_sloss'] = group['sloss'].sum()
                features['loss_ratio'] = self._safe_div(
                    group['sloss'].sum(),
                    features['total_packets_sent'],
                    default=0.0
                )
            else:
                features['total_sloss'] = 0
                features['loss_ratio'] = 0

            # === JITTER FEATURES ===
            # High jitter: VoIP quality issues or tunnel overhead

            if 'sjit' in group.columns:
                features['avg_sjit'] = group['sjit'].mean()
            else:
                features['avg_sjit'] = 0

            # === DERIVED ATTACK INDICATORS ===

            # Connections per unique destination (scan detection)
            features['conns_per_dst'] = self._safe_div(
                features['n_connections'],
                features.get('n_unique_dsts', 1),
                default=features['n_connections']
            )

            # Average bytes per connection (small = probing, large = data transfer)
            total_bytes = features['total_bytes_sent'] + features['total_bytes_received']
            features['bytes_per_conn'] = self._safe_div(
                total_bytes,
                features['n_connections'],
                default=0.0
            )

            # === ATTACK LABEL (for validation only - not used in unsupervised) ===
            if 'label' in group.columns:
                features['attack_ratio'] = group['label'].mean()
            else:
                features['attack_ratio'] = 0

            if 'attack_cat' in group.columns:
                # Most common attack category for this host
                features['primary_attack_cat'] = group['attack_cat'].mode().iloc[0] if len(group['attack_cat'].mode()) > 0 else 'Normal'
            else:
                features['primary_attack_cat'] = 'Unknown'

            features_list.append(features)

        # Create DataFrame from features
        host_df = pd.DataFrame(features_list)
        logger.info(f"Created {len(host_df)} host profiles with {len(host_df.columns)} features")

        self.host_features = host_df
        return host_df

    def preprocess_features(
        self,
        df: pd.DataFrame,
        exclude_cols: List[str] = ['host_ip', 'primary_attack_cat', 'attack_ratio']
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Preprocess features for clustering: handle missing values, normalize.

        Standardization is critical for clustering algorithms:
        - KMeans uses Euclidean distance, sensitive to scale
        - DBSCAN uses distance-based density, needs normalized features

        Args:
            df: Host features DataFrame
            exclude_cols: Columns to exclude from scaling (IDs, labels)

        Returns:
            Tuple of (processed DataFrame, scaled feature matrix)
        """
        logger.info("Preprocessing features...")

        # Identify numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

        # Store feature names
        self.feature_columns = numeric_cols

        # Handle missing values - use median for robustness against outliers
        df_processed = df.copy()
        for col in numeric_cols:
            if df_processed[col].isnull().any():
                median_val = df_processed[col].median()
                df_processed[col].fillna(median_val, inplace=True)
                logger.info(f"  Filled {col} NaN with median: {median_val:.4f}")

        # Handle infinite values
        for col in numeric_cols:
            df_processed[col] = df_processed[col].replace([np.inf, -np.inf], np.nan)
            df_processed[col].fillna(df_processed[col].median(), inplace=True)

        # Extract feature matrix
        X = df_processed[numeric_cols].values

        # Standardize features (zero mean, unit variance)
        X_scaled = self.scaler.fit_transform(X)

        logger.info(f"Scaled {X_scaled.shape[0]} hosts with {X_scaled.shape[1]} features")

        return df_processed, X_scaled

    def get_feature_importance(self, X_scaled: np.ndarray) -> pd.DataFrame:
        """
        Calculate feature importance based on variance.

        High-variance features contribute more to cluster separation.
        This helps identify which behaviors differentiate hosts.

        Args:
            X_scaled: Scaled feature matrix

        Returns:
            DataFrame with features ranked by importance
        """
        variances = np.var(X_scaled, axis=0)

        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'variance': variances,
            'importance_rank': np.argsort(variances)[::-1] + 1
        }).sort_values('variance', ascending=False)

        return importance_df

    def get_feature_descriptions(self) -> dict:
        """
        Return descriptions of features for interpretability.

        Understanding what each feature represents is critical
        for SOC analysts interpreting cluster results.
        """
        return {
            'n_connections': 'Total connections from this host',
            'n_unique_dsts': 'Number of unique destination IPs contacted',
            'n_unique_dst_ports': 'Number of unique destination ports used',
            'total_bytes_sent': 'Total bytes sent by this host',
            'avg_bytes_sent': 'Average bytes per connection (sent)',
            'bytes_ratio': 'Ratio of bytes sent/received (>1 = sender)',
            'packets_ratio': 'Ratio of packets sent/received',
            'avg_duration': 'Average connection duration (seconds)',
            'proto_entropy': 'Protocol diversity (high = many protocols)',
            'dst_entropy': 'Destination IP diversity (high = scanning behavior)',
            'port_entropy': 'Destination port diversity',
            'service_entropy': 'Service diversity',
            'state_entropy': 'Connection state diversity',
            'rst_ratio': 'Ratio of RST (reset) connections',
            'tcp_ratio': 'Fraction of TCP connections',
            'udp_ratio': 'Fraction of UDP connections',
            'avg_sttl': 'Average source TTL value',
            'conns_per_dst': 'Connections per unique destination',
            'bytes_per_conn': 'Average bytes per connection',
        }


if __name__ == "__main__":
    # Demo usage
    from load_data import DataLoader

    loader = DataLoader()
    df = loader.load_combined_dataset(sample_frac=0.1)

    engineer = FeatureEngineer()
    host_features = engineer.aggregate_by_source_ip(df, min_connections=3)

    print("\n=== Host Feature Sample ===")
    print(host_features.head())

    df_processed, X_scaled = engineer.preprocess_features(host_features)

    print(f"\n=== Feature Matrix Shape ===")
    print(f"Hosts: {X_scaled.shape[0]}, Features: {X_scaled.shape[1]}")

    importance = engineer.get_feature_importance(X_scaled)
    print("\n=== Top 10 Features by Variance ===")
    print(importance.head(10))
