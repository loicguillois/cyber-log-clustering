"""
Cyber Log Clustering - Streamlit Dashboard

Interactive dashboard for exploring host behavior clusters and anomalies
from cybersecurity log analysis.

Run with: streamlit run app.py

Features:
- Load and process UNSW-NB15 data
- Interactive cluster visualization
- Anomaly exploration
- Host detail view
- Export capabilities
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from load_data import DataLoader
from feature_engineering import FeatureEngineer
from clustering import ClusteringPipeline
from anomaly_detection import AnomalyDetector

# Page configuration
st.set_page_config(
    page_title="Cyber Log Clustering",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .cluster-interpretation {
        background-color: #e8f4f8;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .anomaly-alert {
        background-color: #ffe6e6;
        border-left: 4px solid #ff4444;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_and_process_data(data_path: str, sample_frac: float, min_connections: int):
    """Load and process data with caching."""
    # Load data
    loader = DataLoader(data_path=data_path)
    df = loader.load_combined_dataset(use_raw=False, sample_frac=sample_frac)

    # Feature engineering
    engineer = FeatureEngineer()
    host_df = engineer.aggregate_by_source_ip(df, min_connections=min_connections)
    host_df_processed, X_scaled = engineer.preprocess_features(host_df)

    return host_df_processed, X_scaled, engineer.feature_columns


@st.cache_data
def run_clustering(X_scaled, n_clusters: int, method: str):
    """Run clustering with caching."""
    clustering = ClusteringPipeline(random_state=42)

    # Dimensionality reduction
    try:
        X_2d = clustering.reduce_dimensions_umap(X_scaled, n_components=2)
        reduction_method = "UMAP"
    except Exception:
        X_2d, _ = clustering.reduce_dimensions_pca(X_scaled, n_components=2)
        reduction_method = "PCA"

    # Clustering
    if method == "KMeans":
        labels = clustering.fit_kmeans(X_scaled, n_clusters=n_clusters)
    elif method == "DBSCAN":
        labels = clustering.fit_dbscan(X_scaled)
    else:  # HDBSCAN
        labels = clustering.fit_hdbscan(X_scaled, min_cluster_size=max(10, len(X_scaled) // 50))

    return X_2d, labels, reduction_method


@st.cache_data
def run_anomaly_detection(X_scaled, contamination: float):
    """Run anomaly detection with caching."""
    detector = AnomalyDetector(contamination=contamination)
    results = detector.detect_all(X_scaled, methods=['isolation_forest', 'lof'])
    return results['ensemble']['scores'], results['ensemble']['predictions']


def main():
    """Main dashboard application."""

    # Header
    st.markdown('<p class="main-header">🔒 Cyber Log Clustering Dashboard</p>', unsafe_allow_html=True)
    st.markdown("""
    <p style="text-align: center; color: #666;">
    Interactive exploration of host behavior clusters and anomalies from UNSW-NB15 cybersecurity logs
    </p>
    """, unsafe_allow_html=True)

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    data_path = st.sidebar.text_input(
        "Data Path",
        value="/Volumes/Data_IA/UNSW_NB15",
        help="Path to UNSW-NB15 dataset"
    )

    sample_frac = st.sidebar.slider(
        "Sample Fraction",
        min_value=0.05,
        max_value=1.0,
        value=0.2,
        step=0.05,
        help="Fraction of data to use (lower = faster)"
    )

    min_connections = st.sidebar.slider(
        "Min Connections per Host",
        min_value=1,
        max_value=50,
        value=5,
        help="Minimum connections to include a host"
    )

    clustering_method = st.sidebar.selectbox(
        "Clustering Method",
        ["HDBSCAN", "KMeans", "DBSCAN"],
        help="Algorithm for clustering hosts"
    )

    if clustering_method == "KMeans":
        n_clusters = st.sidebar.slider(
            "Number of Clusters",
            min_value=2,
            max_value=15,
            value=5
        )
    else:
        n_clusters = 5  # Not used for DBSCAN/HDBSCAN

    contamination = st.sidebar.slider(
        "Anomaly Contamination",
        min_value=0.01,
        max_value=0.2,
        value=0.05,
        step=0.01,
        help="Expected proportion of anomalies"
    )

    # Load data button
    if st.sidebar.button("🚀 Run Analysis", type="primary"):
        st.session_state['run_analysis'] = True

    # Check if we should run analysis
    if not st.session_state.get('run_analysis', False):
        st.info("👈 Configure parameters and click 'Run Analysis' to start")
        return

    # Run analysis
    with st.spinner("Loading and processing data..."):
        try:
            host_df, X_scaled, feature_names = load_and_process_data(
                data_path, sample_frac, min_connections
            )
        except FileNotFoundError:
            st.error(f"Dataset not found at: {data_path}")
            st.info("Please ensure the UNSW-NB15 dataset is downloaded and the path is correct.")
            return

    with st.spinner("Running clustering..."):
        X_2d, labels, reduction_method = run_clustering(
            X_scaled, n_clusters, clustering_method
        )

    with st.spinner("Detecting anomalies..."):
        anomaly_scores, anomaly_predictions = run_anomaly_detection(
            X_scaled, contamination
        )

    # Add results to dataframe
    host_df = host_df.copy()
    host_df['cluster'] = labels
    host_df['anomaly_score'] = anomaly_scores
    host_df['is_anomaly'] = anomaly_predictions == -1

    # Calculate metrics
    n_hosts = len(host_df)
    n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers = (labels == -1).sum()
    n_anomalies = host_df['is_anomaly'].sum()

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Hosts", f"{n_hosts:,}")
    with col2:
        st.metric("Clusters Found", n_clusters_found)
    with col3:
        st.metric("Cluster Outliers", n_outliers)
    with col4:
        st.metric("Anomalies Detected", n_anomalies)

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Cluster Overview",
        "🔴 Anomaly Analysis",
        "📋 Host Details",
        "📈 Feature Analysis"
    ])

    # Tab 1: Cluster Overview
    with tab1:
        st.subheader("Cluster Visualization")

        col1, col2 = st.columns([2, 1])

        with col1:
            # Interactive scatter plot
            fig = px.scatter(
                x=X_2d[:, 0],
                y=X_2d[:, 1],
                color=labels.astype(str),
                title=f"Host Clusters ({clustering_method} + {reduction_method})",
                labels={'x': f'{reduction_method} Dim 1', 'y': f'{reduction_method} Dim 2', 'color': 'Cluster'},
                color_discrete_sequence=px.colors.qualitative.Set1,
                hover_data={
                    'Connections': host_df['n_connections'].values,
                    'Anomaly Score': np.round(anomaly_scores, 3)
                }
            )
            fig.update_traces(marker=dict(size=8, opacity=0.7))
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Cluster distribution
            cluster_counts = pd.Series(labels).value_counts().sort_index()
            fig2 = px.bar(
                x=cluster_counts.index.astype(str),
                y=cluster_counts.values,
                title="Cluster Sizes",
                labels={'x': 'Cluster', 'y': 'Count'},
                color=cluster_counts.index.astype(str),
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            fig2.update_layout(showlegend=False, height=250)
            st.plotly_chart(fig2, use_container_width=True)

            # Attack ratio by cluster
            if 'attack_ratio' in host_df.columns:
                attack_by_cluster = host_df.groupby('cluster')['attack_ratio'].mean() * 100
                fig3 = px.bar(
                    x=attack_by_cluster.index.astype(str),
                    y=attack_by_cluster.values,
                    title="Attack Rate by Cluster (%)",
                    labels={'x': 'Cluster', 'y': 'Attack %'},
                    color=attack_by_cluster.values,
                    color_continuous_scale='RdYlBu_r'
                )
                fig3.update_layout(height=250)
                st.plotly_chart(fig3, use_container_width=True)

        # Cluster statistics
        st.subheader("Cluster Statistics")

        stats_cols = ['cluster', 'n_connections', 'n_unique_dsts', 'bytes_ratio',
                      'avg_duration', 'dst_entropy', 'attack_ratio']
        stats_cols = [c for c in stats_cols if c in host_df.columns]

        cluster_stats = host_df.groupby('cluster')[stats_cols[1:]].mean().round(3)
        cluster_stats['count'] = host_df.groupby('cluster').size()
        st.dataframe(cluster_stats, use_container_width=True)

    # Tab 2: Anomaly Analysis
    with tab2:
        st.subheader("Anomaly Detection Results")

        col1, col2 = st.columns([2, 1])

        with col1:
            # Anomaly scatter plot
            fig = px.scatter(
                x=X_2d[:, 0],
                y=X_2d[:, 1],
                color=anomaly_scores,
                title="Anomaly Scores",
                labels={'x': f'{reduction_method} Dim 1', 'y': f'{reduction_method} Dim 2', 'color': 'Score'},
                color_continuous_scale='RdYlBu_r',
                hover_data={
                    'Host': host_df['host_ip'].values,
                    'Score': np.round(anomaly_scores, 3),
                    'Cluster': labels
                }
            )
            # Highlight anomalies
            anomaly_mask = anomaly_predictions == -1
            fig.add_trace(go.Scatter(
                x=X_2d[anomaly_mask, 0],
                y=X_2d[anomaly_mask, 1],
                mode='markers',
                marker=dict(size=15, symbol='circle-open', color='red', line=dict(width=2)),
                name='Anomalies',
                hoverinfo='skip'
            ))
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Score distribution
            fig2 = px.histogram(
                x=anomaly_scores,
                nbins=50,
                title="Anomaly Score Distribution",
                labels={'x': 'Anomaly Score', 'y': 'Count'}
            )
            threshold = np.percentile(anomaly_scores, 100 * (1 - contamination))
            fig2.add_vline(x=threshold, line_dash="dash", line_color="red",
                           annotation_text=f"Threshold: {threshold:.3f}")
            fig2.update_layout(height=250)
            st.plotly_chart(fig2, use_container_width=True)

            # Anomaly count by cluster
            anomaly_by_cluster = host_df[host_df['is_anomaly']].groupby('cluster').size()
            if len(anomaly_by_cluster) > 0:
                fig3 = px.bar(
                    x=anomaly_by_cluster.index.astype(str),
                    y=anomaly_by_cluster.values,
                    title="Anomalies per Cluster",
                    labels={'x': 'Cluster', 'y': 'Anomaly Count'},
                    color=anomaly_by_cluster.values,
                    color_continuous_scale='Reds'
                )
                fig3.update_layout(height=250)
                st.plotly_chart(fig3, use_container_width=True)

        # Top anomalies table
        st.subheader("Top Anomalous Hosts")

        top_n = st.slider("Number of top anomalies to show", 5, 50, 20)
        top_anomalies = host_df.nlargest(top_n, 'anomaly_score')

        display_cols = ['host_ip', 'anomaly_score', 'cluster', 'n_connections',
                        'n_unique_dsts', 'bytes_ratio', 'dst_entropy']
        display_cols = [c for c in display_cols if c in top_anomalies.columns]

        st.dataframe(
            top_anomalies[display_cols].style.background_gradient(
                subset=['anomaly_score'], cmap='Reds'
            ),
            use_container_width=True
        )

    # Tab 3: Host Details
    with tab3:
        st.subheader("Host Detail View")

        # Host search
        host_list = host_df['host_ip'].tolist()
        selected_host = st.selectbox("Select Host", host_list)

        if selected_host:
            host_data = host_df[host_df['host_ip'] == selected_host].iloc[0]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Cluster", int(host_data['cluster']))
            with col2:
                st.metric("Anomaly Score", f"{host_data['anomaly_score']:.4f}")
            with col3:
                if host_data['is_anomaly']:
                    st.error("⚠️ ANOMALY DETECTED")
                else:
                    st.success("✅ Normal")

            # Host features
            st.subheader("Host Features")

            feature_data = []
            for col in host_data.index:
                if col not in ['host_ip', 'cluster', 'anomaly_score', 'is_anomaly', 'primary_attack_cat']:
                    feature_data.append({
                        'Feature': col,
                        'Value': round(host_data[col], 4) if isinstance(host_data[col], float) else host_data[col]
                    })

            st.dataframe(pd.DataFrame(feature_data), use_container_width=True)

            # Compare with cluster
            st.subheader("Comparison with Cluster")

            cluster_id = host_data['cluster']
            cluster_hosts = host_df[host_df['cluster'] == cluster_id]

            comparison_features = ['n_connections', 'n_unique_dsts', 'bytes_ratio',
                                   'avg_duration', 'dst_entropy', 'anomaly_score']
            comparison_features = [f for f in comparison_features if f in host_df.columns]

            comparison_data = []
            for feat in comparison_features:
                comparison_data.append({
                    'Feature': feat,
                    'This Host': round(host_data[feat], 4),
                    'Cluster Mean': round(cluster_hosts[feat].mean(), 4),
                    'Cluster Std': round(cluster_hosts[feat].std(), 4),
                    'Z-Score': round((host_data[feat] - cluster_hosts[feat].mean()) / (cluster_hosts[feat].std() + 1e-10), 2)
                })

            st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

    # Tab 4: Feature Analysis
    with tab4:
        st.subheader("Feature Analysis")

        # Feature correlation
        numeric_features = [f for f in feature_names if f in host_df.columns][:15]
        corr_matrix = host_df[numeric_features].corr()

        fig = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation"),
            x=numeric_features,
            y=numeric_features,
            color_continuous_scale='RdBu_r',
            title="Feature Correlation Matrix"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Feature distributions by cluster
        st.subheader("Feature Distribution by Cluster")

        selected_feature = st.selectbox("Select Feature", numeric_features)

        fig = px.box(
            host_df,
            x='cluster',
            y=selected_feature,
            color='cluster',
            title=f"{selected_feature} by Cluster"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Feature vs anomaly score
        st.subheader("Feature vs Anomaly Score")

        fig = px.scatter(
            host_df,
            x=selected_feature,
            y='anomaly_score',
            color='cluster',
            title=f"{selected_feature} vs Anomaly Score",
            opacity=0.6
        )
        st.plotly_chart(fig, use_container_width=True)

    # Export options
    st.sidebar.header("📥 Export")

    if st.sidebar.button("Download Results CSV"):
        csv = host_df.to_csv(index=False)
        st.sidebar.download_button(
            label="📄 Download CSV",
            data=csv,
            file_name="host_analysis_results.csv",
            mime="text/csv"
        )

    # Footer
    st.markdown("---")
    st.markdown("""
    <p style="text-align: center; color: #666; font-size: 0.9rem;">
    🔒 Cyber Log Clustering | UNSW-NB15 Dataset Analysis |
    Built for SOC Threat Detection
    </p>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
