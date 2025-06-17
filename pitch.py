import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import statsmodels.formula.api as smf
from tqdm import tqdm  # for progress bars
from sklearn.linear_model import LogisticRegression

import utils
from datetime import datetime
from scipy import sparse
import patsy
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


def cluster_pitches_kmeans(df, n_clusters, numerical_columns, categorical_columns=[],
                           random_state=42, max_iter=300):
    # Drop rows with missing values in the required columns
    df = df.dropna(subset=numerical_columns + categorical_columns).copy()

    # Standardize numerical features
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[numerical_columns])

    # Always use one-hot encoding for categorical variables
    if categorical_columns:
        encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        X_cat = encoder.fit_transform(df[categorical_columns])
        cat_columns = encoder.get_feature_names_out(categorical_columns)
        X = np.hstack((X_num, X_cat))
        column_names = numerical_columns + list(cat_columns)
    else:
        X = X_num
        column_names = numerical_columns

    # Convert the processed array into a DataFrame for visualization
    X_df = pd.DataFrame(X, columns=column_names, index=df.index)
    # utils.pdf(X_df.head(5))

    # Initialize and fit the KMeans model
    kmeans = KMeans(n_clusters=n_clusters,
                    random_state=random_state,
                    max_iter=max_iter,
                    n_init=10)
    labels = kmeans.fit_predict(X)

    # Append cluster labels to the DataFrame
    df_with_clusters = df.copy()
    df_with_clusters.loc[df.index, 'kmeans_cluster'] = labels

    return df_with_clusters, labels, X


def tune_kmeans(df, numerical_columns, categorical_columns, cluster_range=range(5, 16), random_state=42):
    best_score = -1
    best_n_clusters = None
    best_labels = None
    best_X = None
    best_result_df = None

    for n_clusters in tqdm(cluster_range, desc="n_clusters"):
        result_df, labels, X = cluster_pitches_kmeans(
            df,
            n_clusters=n_clusters,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
            random_state=random_state
        )
        # Compute the silhouette score for the clustering
        score = silhouette_score(X, labels)
        print(f"n_clusters: {n_clusters} -> Silhouette Score: {score:.3f}")
        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters
            best_labels = labels
            best_X = X
            best_result_df = result_df

    return best_result_df, best_labels, best_score, best_n_clusters, best_X


if __name__ == "__main__":
    # Load the data
    today = datetime.today()
    df = utils.grab_data('data/statcast', today.year - 3, today.year)
    utils.pdf(df.head(5))
    print(f"Data is {len(df)} rows deep.")

    # Get all unique pitch types (e.g., CH, FF, SL, etc.)
    unique_pitch_types = df['pitch_type'].unique()

    # Define columns to use for clustering
    numerical_columns = ["release_spin_rate", "effective_speed", "pfx_x", "pfx_z"]
    categorical_columns = ["p_throws"]

    all_clustered = []  # List to hold clustering results for each pitch type

    print("\nClustering each pitch type separately...\n")
    for ptype in tqdm(unique_pitch_types, desc="Pitch Types"):
        # Filter the DataFrame for the current pitch type
        df_ptype = df[df['pitch_type'] == ptype].copy()
        if df_ptype.empty:
            continue

        # Cluster the current pitch type with 5 clusters
        clustered_df, labels, X_df = cluster_pitches_kmeans(
            df_ptype,
            n_clusters=5,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns
        )
        # Rename the cluster labels to include the pitch type as a suffix (e.g., "CH_0")
        clustered_df['kmeans_cluster'] = clustered_df['kmeans_cluster'].apply(lambda x: f"{ptype}_{x}")
        all_clustered.append(clustered_df)

    # Concatenate the results from all pitch types
    pitches_df = pd.concat(all_clustered)

    # Optionally, continue with further analysis (e.g., grouping and saving results)
    peep = pitches_df.groupby(['kmeans_cluster', 'p_throws', 'pitch_name']).agg({
        'pitcher': 'count',
        'effective_speed': 'mean',
        'release_spin_rate': 'mean',
        'pfx_x': 'mean',
        'pfx_z': 'mean',
        'description': 'count'
    })
    peep['pitch_type'] = [ix[2] for ix in peep.index]
    peep['description_pct'] = (
        peep.groupby('pitch_type')['description']
        .transform(lambda x: (x / x.sum()) * 100)
    )
    peep['description_pct'] = peep['description_pct'].round(2)
    # peep['description'] = peep['description'] / peep['description'].sum()
    utils.pdf(peep)

    utils.make_dir('data/clust')
    pitches_df.to_parquet('data/clust/pitch_clusters.parquet')

    # and the value is the count of pitches thrown in that cluster.
    pitcher_cluster_counts = pitches_df.groupby(['player_name', 'kmeans_cluster']).size().unstack(fill_value=0)
    # utils.pdf(pitcher_cluster_counts.tail(20))

    # Convert counts to proportions for each pitcher (i.e. relative frequency of each pitch type)
    pitcher_cluster_props = pitcher_cluster_counts.div(pitcher_cluster_counts.sum(axis=1), axis=0)
    # utils.pdf(pitcher_cluster_props.tail(20))

    pitcher_info = pitches_df.groupby('player_name').agg({
        'p_throws': lambda x: x.mode()[0],
        'release_pos_x': 'mean',
        # 'release_pos_z': 'mean'
        # 'effective_speed': 'mean'
    })

    pitcher_features = pitcher_cluster_props.merge(pitcher_info, left_index=True, right_index=True)

    # print("Pitcher features before encoding:")
    # print(pitcher_features.head())

    # One-hot encode the pitcher's handedness (p_throws)
    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    p_throws_encoded = encoder.fit_transform(pitcher_features[['p_throws']])
    p_throws_df = pd.DataFrame(p_throws_encoded,
                               index=pitcher_features.index,
                               columns=encoder.get_feature_names_out(['p_throws']))

    # Combine the numerical features (pitch cluster proportions and average release_pos_x)
    # with the one-hot encoded handedness.
    pitcher_features_final = pd.concat([pitcher_features.drop(columns=['p_throws']), p_throws_df], axis=1)

    # print("\nFinal pitcher feature matrix (before scaling):")
    # print(pitcher_features_final.head())

    # Scale the features so that each feature has zero mean and unit variance
    scaler = StandardScaler()
    pitcher_features_scaled = scaler.fit_transform(pitcher_features_final)

    # -----------------------------
    # 3. Cluster the Pitchers with KMeans
    # -----------------------------

    # Try clustering with a range of cluster numbers, and pick the best based on silhouette score.
    best_score = -1
    best_n_clusters = None
    best_labels = None

    for n_clusters in range(10, 21):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pitcher_features_scaled)
        score = silhouette_score(pitcher_features_scaled, labels)
        print(f"n_clusters: {n_clusters}, silhouette score: {score:.3f}")
        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters
            best_labels = labels

    print(f"\nOptimal pitcher clustering: {best_n_clusters} clusters, silhouette score = {best_score:.3f}")

    # Append the cluster labels to the pitcher features DataFrame
    pitcher_features_final['pitcher_cluster'] = best_labels

    # Optionally, inspect the resulting pitcher clusters by looking at the average feature values in each cluster
    # cluster_summary = pitcher_features_final.groupby('pitcher_cluster').mean()
    # print("\nCluster summary (mean features by cluster):")
    # utils.pdf(cluster_summary)

    # Now you can save or further analyze the pitcher-level clustering.
    utils.pdf(pitcher_features_final.head(10))
    pitcher_features_final.to_parquet('data/clust/pitcher_clusters.parquet')
