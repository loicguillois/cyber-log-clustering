"""
Data Ingestion Module for UNSW-NB15 Dataset

This module handles loading and combining the UNSW-NB15 intrusion detection dataset.
The dataset contains network traffic features extracted from real and simulated attacks,
making it ideal for training cybersecurity anomaly detection models.

Key capabilities:
- Load multiple CSV files efficiently using chunking for large datasets
- Handle both raw data files (no headers) and preprocessed train/test sets
- Provide data validation and quality checks
"""

import os
import glob
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Column names for raw UNSW-NB15 CSV files (files 1-4 don't have headers)
RAW_COLUMNS = [
    'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur',
    'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service',
    'sload', 'dload', 'spkts', 'dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb',
    'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'sjit', 'djit',
    'stime', 'ltime', 'sintpkt', 'dintpkt', 'tcprtt', 'synack', 'ackdat',
    'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login',
    'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm',
    'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'label'
]


class DataLoader:
    """
    Handles loading and preprocessing of UNSW-NB15 network traffic data.

    The UNSW-NB15 dataset is designed for evaluating network intrusion detection
    systems. It contains 49 features including:
    - Flow features (duration, bytes, packets)
    - Basic features (protocol, service, state)
    - Content features (HTTP methods, FTP commands)
    - Time features (jitter, inter-packet time)
    - Additional generated features for ML

    Attack categories in the dataset:
    - Fuzzers: Attempt to crash programs by feeding random data
    - Analysis: Port scans, spam, HTML file penetrations
    - Backdoors: Bypasses normal authentication
    - DoS: Denial of Service attacks
    - Exploits: Exploitation of vulnerabilities
    - Generic: Works against all block-ciphers
    - Reconnaissance: Gathers information for attacks
    - Shellcode: Small code used as payload
    - Worms: Self-replicating malware
    """

    def __init__(self, data_path: str = "/Volumes/Data_IA/UNSW_NB15"):
        """
        Initialize the data loader.

        Args:
            data_path: Path to the UNSW-NB15 dataset directory
        """
        self.data_path = data_path
        self.raw_data: Optional[pd.DataFrame] = None
        self.attack_categories = [
            'Normal', 'Fuzzers', 'Analysis', 'Backdoor', 'DoS',
            'Exploits', 'Generic', 'Reconnaissance', 'Shellcode', 'Worms'
        ]

    def load_raw_files(self, sample_frac: Optional[float] = None) -> pd.DataFrame:
        """
        Load raw CSV files (UNSW-NB15_1.csv to UNSW-NB15_4.csv).

        These files contain ~2.5 million records of network traffic.
        For memory efficiency, we use chunked reading.

        Args:
            sample_frac: Optional fraction of data to sample (0.0 to 1.0)

        Returns:
            Combined DataFrame with all raw data
        """
        raw_files = sorted(glob.glob(os.path.join(self.data_path, "UNSW-NB15_[1-4].csv")))

        if not raw_files:
            raise FileNotFoundError(f"No raw data files found in {self.data_path}")

        logger.info(f"Found {len(raw_files)} raw data files")

        dfs = []
        for file_path in raw_files:
            logger.info(f"Loading {os.path.basename(file_path)}...")

            # Read in chunks for memory efficiency
            chunks = []
            for chunk in pd.read_csv(
                file_path,
                names=RAW_COLUMNS,
                header=None,
                chunksize=100000,
                low_memory=False
            ):
                if sample_frac and sample_frac < 1.0:
                    chunk = chunk.sample(frac=sample_frac, random_state=42)
                chunks.append(chunk)

            df = pd.concat(chunks, ignore_index=True)
            dfs.append(df)
            logger.info(f"  Loaded {len(df):,} records")

        self.raw_data = pd.concat(dfs, ignore_index=True)
        logger.info(f"Total records loaded: {len(self.raw_data):,}")

        return self.raw_data

    def load_train_test_sets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load the preprocessed training and testing sets.

        These sets have headers and are cleaner than raw files.
        Training set: ~175K records, Testing set: ~82K records

        Returns:
            Tuple of (training_df, testing_df)
        """
        train_path = os.path.join(self.data_path, "UNSW_NB15_training-set.csv")
        test_path = os.path.join(self.data_path, "UNSW_NB15_testing-set.csv")

        logger.info("Loading training set...")
        train_df = pd.read_csv(train_path, low_memory=False)
        logger.info(f"  Training records: {len(train_df):,}")

        logger.info("Loading testing set...")
        test_df = pd.read_csv(test_path, low_memory=False)
        logger.info(f"  Testing records: {len(test_df):,}")

        return train_df, test_df

    def load_combined_dataset(
        self,
        use_raw: bool = False,
        sample_frac: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Load combined dataset (train + test or raw files).

        For clustering host behaviors, we need sufficient data per source IP.
        Using combined train/test gives ~257K records which is a good balance
        between computational cost and statistical significance.

        Args:
            use_raw: If True, load raw files; otherwise use train/test sets
            sample_frac: Optional fraction to sample (for testing/development)

        Returns:
            Combined DataFrame ready for feature engineering
        """
        if use_raw:
            return self.load_raw_files(sample_frac)

        train_df, test_df = self.load_train_test_sets()
        combined = pd.concat([train_df, test_df], ignore_index=True)

        if sample_frac and sample_frac < 1.0:
            combined = combined.sample(frac=sample_frac, random_state=42)
            logger.info(f"Sampled to {len(combined):,} records")

        self.raw_data = combined
        return combined

    def get_data_summary(self, df: Optional[pd.DataFrame] = None) -> dict:
        """
        Generate summary statistics for the loaded data.

        Useful for understanding the dataset distribution before clustering.
        In cybersecurity, understanding the baseline is critical for
        identifying anomalies.

        Args:
            df: DataFrame to summarize (uses loaded data if None)

        Returns:
            Dictionary with summary statistics
        """
        if df is None:
            df = self.raw_data

        if df is None:
            raise ValueError("No data loaded. Call load_* method first.")

        summary = {
            'total_records': len(df),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
        }

        # Attack distribution (if label column exists)
        if 'label' in df.columns:
            summary['attack_distribution'] = {
                'normal': int((df['label'] == 0).sum()),
                'attack': int((df['label'] == 1).sum())
            }

        if 'attack_cat' in df.columns:
            summary['attack_categories'] = df['attack_cat'].value_counts().to_dict()

        # Protocol distribution (important for understanding traffic types)
        if 'proto' in df.columns:
            summary['protocol_distribution'] = df['proto'].value_counts().head(10).to_dict()

        # Service distribution
        if 'service' in df.columns:
            summary['service_distribution'] = df['service'].value_counts().head(10).to_dict()

        return summary

    def validate_data(self, df: Optional[pd.DataFrame] = None) -> List[str]:
        """
        Validate data quality and return list of issues found.

        Data quality is critical for cybersecurity ML - garbage in, garbage out.
        Common issues include:
        - Invalid IP addresses
        - Impossible byte counts
        - Missing critical fields

        Args:
            df: DataFrame to validate (uses loaded data if None)

        Returns:
            List of validation warnings/issues
        """
        if df is None:
            df = self.raw_data

        if df is None:
            raise ValueError("No data loaded. Call load_* method first.")

        issues = []

        # Check for missing values in critical columns
        critical_cols = ['srcip', 'dstip', 'proto'] if 'srcip' in df.columns else []
        for col in critical_cols:
            if col in df.columns:
                missing = df[col].isnull().sum()
                if missing > 0:
                    issues.append(f"Missing values in {col}: {missing}")

        # Check for negative values in byte/packet counts
        numeric_cols = ['sbytes', 'dbytes', 'spkts', 'dpkts']
        for col in numeric_cols:
            if col in df.columns:
                negative = (df[col] < 0).sum()
                if negative > 0:
                    issues.append(f"Negative values in {col}: {negative}")

        # Check for impossible duration values
        if 'dur' in df.columns:
            negative_dur = (df['dur'] < 0).sum()
            if negative_dur > 0:
                issues.append(f"Negative duration values: {negative_dur}")

        if not issues:
            logger.info("Data validation passed - no issues found")
        else:
            for issue in issues:
                logger.warning(f"Data validation: {issue}")

        return issues


if __name__ == "__main__":
    # Demo usage
    loader = DataLoader()

    # Load combined train/test data
    df = loader.load_combined_dataset(use_raw=False)

    # Print summary
    summary = loader.get_data_summary()
    print("\n=== Data Summary ===")
    print(f"Total records: {summary['total_records']:,}")
    print(f"Memory usage: {summary['memory_usage_mb']:.2f} MB")
    print(f"\nAttack distribution: {summary.get('attack_distribution', 'N/A')}")
    print(f"\nTop protocols: {summary.get('protocol_distribution', 'N/A')}")

    # Validate data
    issues = loader.validate_data()
    print(f"\nValidation issues: {len(issues)}")
