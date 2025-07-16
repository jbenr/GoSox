import os
import math
import pickle
import numpy as np
import pandas as pd
from datetime import date, timedelta
import matplotlib.pyplot as plt
from tqdm import tqdm
import utils
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def convert_statcast_name(name):
    if not isinstance(name, str) or "," not in name:
        return name
    last, first = name.split(",", 1)
    return f"{first.strip()[0]}. {last.strip()}"


def get_weight(days_ago, midpoint=60.0, steepness=0.05):
    """Logistic-style decay weight."""
    return 1.0 / (1.0 + math.exp(steepness * (days_ago - midpoint)))


def plot_weight_decay(max_days=300):
    days = np.arange(0, max_days + 1)
    weights = [get_weight(d) for d in days]
    plt.figure(figsize=(8, 5))
    plt.plot(days, weights, marker='o', markersize=3)
    plt.title("Logistic Weight Decay (Example)")
    plt.xlabel("Days ago")
    plt.ylabel("Weight")
    plt.grid(True)
    plt.show()


def weighted_stats_for_df(df, max_lookback_days=300, sum_cols=None, ratio_pairs=None):
    if sum_cols is None:
        sum_cols = []
    if ratio_pairs is None:
        ratio_pairs = []

    df = df.sort_values("game_date").copy()
    results = []

    for idx, row in df.iterrows():
        current_date = row["game_date"]
        cutoff_date = current_date - timedelta(days=max_lookback_days)

        window = df.loc[(df["game_date"] < current_date) & (df["game_date"] >= cutoff_date)]
        games_in_window = window["game_date"].nunique() # Track number of games in window

        if window.empty:
            for col in sum_cols:
                row[f"weighted_{col}"] = np.nan
            for (_, _, ratio_name) in ratio_pairs:
                row[f"weighted_{ratio_name}"] = np.nan
        else:
            sum_w = 0.0
            accum = {col: 0.0 for col in sum_cols}
            ratio_accum = {
                ratio_name: {"num": 0.0, "den": 0.0}
                for (_, _, ratio_name) in ratio_pairs
            }

            for _, prior in window.iterrows():
                days_ago = (current_date - prior["game_date"]).days
                w = get_weight(days_ago)
                sum_w += w

                for col in sum_cols:
                    val = prior.get(col, np.nan)
                    if pd.notna(val):
                        accum[col] += w * val

                for (num_col, den_col, ratio_name) in ratio_pairs:
                    num_val = prior.get(num_col, np.nan)
                    den_val = prior.get(den_col, np.nan)
                    if pd.notna(num_val):
                        ratio_accum[ratio_name]["num"] += w * num_val
                    if pd.notna(den_val):
                        ratio_accum[ratio_name]["den"] += w * den_val

            if sum_w > 0:
                for col in sum_cols:
                    row[f"weighted_{col}"] = accum[col] / sum_w
                for (num_col, den_col, ratio_name) in ratio_pairs:
                    num_val = ratio_accum[ratio_name]["num"]
                    den_val = ratio_accum[ratio_name]["den"]
                    row[f"weighted_{ratio_name}"] = (num_val / den_val if den_val != 0 else np.nan)
            else:
                for col in sum_cols:
                    row[f"weighted_{col}"] = np.nan
                for (num_col, den_col, ratio_name) in ratio_pairs:
                    row[f"weighted_{ratio_name}"] = np.nan

        row["games_in_window"] = games_in_window # Assign the credibility metric
        results.append(row)

    return pd.DataFrame(results)


def pitcher_and_offense_crunch(start_year=2022,end_year=2024, only_starters=True,
                               statcast_dir="data/statcast", max_lookback_days=300):
    ### Data Pull ###
    yrs_lookback = max_lookback_days//365
    years = range(start_year - yrs_lookback, end_year + 1)
    all_stats = []
    for yr in years:
        fpath = f"{statcast_dir}/statcast_{yr}.parquet"
        if os.path.isfile(fpath):
            all_stats.append(pd.read_parquet(fpath))
    if not all_stats:
        print("No statcast data found.")
        return pd.DataFrame()

    stats = pd.concat(all_stats, ignore_index=True)
    stats["game_date"] = pd.to_datetime(stats["game_date"]).dt.date

    ###############################
    ### 1) Calculate some stats ###
    ###############################
    # Mark at-bat groupings
    stats["is_hit"] = stats["events"].isin(["single", "double", "triple", "home_run", "inside_the_park_hr"])
    stats["is_k"] = stats["events"].isin(["strikeout"])

    # Balls strikes
    stats['strike'] = stats['description'].isin([
        'swinging_strike', 'foul', 'called_strike', 'foul_tip', 'swinging_strike_blocked',
        "foul_bunt", "missed_bunt", "bunt_foul_tip"
    ]).astype(int)
    stats['ball'] = (stats['description'] == 'ball').astype(int)
    stats['real_pitch'] = stats['ball']+stats['strike']

    # Mark pitch-based groupings
    stats["is_swing"] = stats["description"].isin([
        "swinging_strike", "swinging_strike_blocked", "foul",
        "foul_tip", "hit_into_play",
        "foul_bunt", "missed_bunt", "bunt_foul_tip"
    ])
    stats["is_whiff"] = stats["description"].isin(["swinging_strike", "swinging_strike_blocked", "missed_bunt"])
    stats["is_strike_looking"] = stats["description"].isin(["called_strike"])
    utils.pdf(stats.tail(2))

    #############################
    # 2) PITCHER-level aggregator
    #############################
    # Summarize at bats per game
    grouped_events = (
        stats.dropna(subset=["events"])
        .groupby(["player_name", "game_date", "p_throws", "inning_topbot", "game_pk", "home_team", "away_team"], dropna=False)
        .agg({"is_hit":"sum","is_k":"sum","events":"count","launch_speed":"mean"}).reset_index()
    )
    grouped_events['hit_per_at_bat'] = grouped_events['is_hit']/grouped_events['events']
    grouped_events['k_per_at_bat'] = grouped_events['is_k']/grouped_events['events']

    # Summarize pitches
    grouped_pitches = (
        stats.groupby(["player_name", "game_date", "p_throws", "inning_topbot", "game_pk", "home_team", "away_team"], dropna=False)
        .agg(
            sum_real_pitch=("real_pitch", "sum"),
            sum_swing=("is_swing", "sum"),
            sum_whiff=("is_whiff", "sum"),
            sum_looking=("is_strike_looking","sum"),
            sum_strike=("strike","sum"),
            sum_ball=("ball","sum")
        ).reset_index())

    # Merge
    merged_pitcher = pd.merge(
        grouped_events,grouped_pitches,
        on=["player_name", "game_date", "p_throws", "inning_topbot", "game_pk", "home_team", "away_team"],
        how="outer"
    )
    merged_pitcher.rename(columns={"player_name": "pitcher"}, inplace=True)
    merged_pitcher["pitcher"] = merged_pitcher["pitcher"].apply(convert_statcast_name)
    merged_pitcher.sort_values(["pitcher", "game_date"], inplace=True)

    if only_starters:
        print("Filtering to only starting pitchers...")

        # Define a unique game+team+topbot identifier
        stats["pitcher"] = stats["player_name"].apply(convert_statcast_name)
        stats.sort_values(["game_pk", "inning_topbot", "inning", "at_bat_number", "pitch_number"], inplace=True)

        first_pitchers = (
            stats.groupby(["game_pk", "inning_topbot"], dropna=False)
            .first()
            .reset_index()[["game_pk", "inning_topbot", "pitcher"]]
        )

        merged_pitcher = pd.merge(
            merged_pitcher,
            first_pitchers,
            on=["game_pk", "inning_topbot", "pitcher"],
            how="inner"
        )

    # Weighted stats for pitchers
    pitchers_dfs = []
    grouped_p = merged_pitcher.groupby("pitcher", group_keys=False)
    print("Calculating weighted stats for pitchers...")
    for pitcher_name, subdf in tqdm(grouped_p, total=len(grouped_p)):
        subdf_res = weighted_stats_for_df(
            subdf, max_lookback_days=max_lookback_days,
            sum_cols=["is_k", "sum_real_pitch", "is_hit", "events", "launch_speed"],
            ratio_pairs=[
                ("is_hit","events","hit_pct"),
                ("is_k","events","strikout_pct"),
                ("sum_whiff","sum_swing","whiff_pct"),
                ("sum_looking","sum_real_pitch","strike_looking_pct"),
                ("sum_strike", "sum_ball", "strike_ball_ratio")
            ]
        )
        pitchers_dfs.append(subdf_res)
    roll_df_pitchers = pd.concat(pitchers_dfs, ignore_index=True)

    ############################################################################
    # 3) TEAM OFFENSE aggregator
    ############################################################################
    # Make a batting_team column:
    # if inning_topbot == 'Top' => away_team is batting
    # if inning_topbot == 'Bot' => home_team is batting
    stats_offense = stats.dropna(subset=["inning_topbot"]).copy()
    stats_offense["batting_team"] = np.where(
        stats_offense["inning_topbot"] == "Top",
        stats_offense["away_team"],
        stats_offense["home_team"]
    )

    # Summarize events
    grouped_ev_off = (
        stats_offense.groupby(["batting_team", "game_date", "game_pk"], dropna=False)
        .agg({"is_hit":"sum","is_k":"sum","events": "count","launch_speed":"mean"}).reset_index()
    )
    grouped_ev_off['hit_per_at_bat'] = grouped_ev_off['is_hit']/grouped_ev_off['events']
    grouped_ev_off['k_per_at_bat'] = grouped_ev_off['is_k']/grouped_ev_off['events']

    # Summarize pitches for offense
    grouped_pit_off = (
        stats_offense.groupby(["batting_team", "game_date", "game_pk"], dropna=False)
        .agg(
            sum_real_pitch=("real_pitch", "sum"),
            sum_swing=("is_swing", "sum"),
            sum_whiff=("is_whiff", "sum"),
            sum_looking=("is_strike_looking","sum"),
            sum_strike=("strike","sum"),
            sum_ball=("ball","sum")
        ).reset_index()
    )
    grouped_pit_off["contact_pct"] = 1.0 - (grouped_pit_off["sum_whiff"] / grouped_pit_off["sum_swing"])
    grouped_pit_off.loc[grouped_pit_off["sum_swing"] == 0, "contact_pct"] = np.nan
    grouped_pit_off["strike_looking_pct"] = (grouped_pit_off["sum_looking"] / grouped_pit_off["sum_real_pitch"])

    # Merge for offense
    merged_offense = pd.merge(
        grouped_ev_off,
        grouped_pit_off,
        on=["batting_team", "game_date", "game_pk"],
        how="outer"
    )
    merged_offense.sort_values(["batting_team", "game_date"], inplace=True)

    # Weighted stats for offense
    offense_dfs = []
    grouped_o = merged_offense.groupby("batting_team", group_keys=False)
    print("Calculating weighted stats for team offense...")
    for team_name, subdf in tqdm(grouped_o, total=len(grouped_o)):
        subdf_res = weighted_stats_for_df(
            subdf, max_lookback_days=max_lookback_days,
            sum_cols=["is_k", "sum_real_pitch", "is_hit", "events", "launch_speed"],
            ratio_pairs=[
                ("is_hit","events","hit_pct"),
                ("is_k","events","strikout_pct"),
                ("sum_whiff","sum_swing","whiff_pct"),
                ("sum_looking","sum_real_pitch","strike_looking_pct"),
                ("sum_strike","sum_ball","strike_ball_ratio")
            ]
        )
            # _weighted_stats_for_df(subdf, group_label=team_name, max_lookback_days=max_lookback_days))
        offense_dfs.append(subdf_res)
    roll_df_offense = pd.concat(offense_dfs, ignore_index=True)

    return roll_df_pitchers, roll_df_offense


def comparatively_speaking(
    start_year=2022, end_year=2024, lookback=300,
    pitcher_positive_stats=None,
    batter_positive_stats=None
):
    if pitcher_positive_stats is None:
        pitcher_positive_stats = ['weighted_strikout_pct']
    if batter_positive_stats is None:
        batter_positive_stats = ['weighted_is_hit']

    p, b = pitcher_and_offense_crunch(
        start_year=start_year,
        end_year=end_year,
        statcast_dir="data/statcast",
        max_lookback_days=lookback
    )
    utils.pdf(p.tail(3))
    utils.pdf(b.tail(3))

    top = pd.merge(
        p[p.inning_topbot == 'Top'], b, how='left',
        left_on=['game_date', 'game_pk', 'away_team'],
        right_on=['game_date', 'game_pk', 'batting_team']
    )
    bot = pd.merge(
        p[p.inning_topbot == 'Bot'], b, how='left',
        left_on=['game_date', 'game_pk', 'home_team'],
        right_on=['game_date', 'game_pk', 'batting_team']
    )
    m = pd.concat([top, bot]).sort_values(by=['game_date', 'home_team']).drop_duplicates().reset_index(drop=True)
    feat = m[['game_pk', 'game_date', 'pitcher', 'home_team', 'away_team']].copy()
    feat["credibility_pitcher"] = m["games_in_window_x"]
    feat["credibility_batter"] = m["games_in_window_y"]

    def normalize(val, vmin, vmax, invert=False):
        if pd.isna(val) or vmax == vmin:
            return np.nan
        norm = (val - vmin) / (vmax - vmin)
        return 1 - norm if invert else norm

    def build_matchup(stat, pitcher_invert=False, batter_invert=False):
        stat_x = f"{stat}_x"
        stat_y = f"{stat}_y"

        if stat_x not in m.columns or stat_y not in m.columns:
            print(f"Warning: {stat_x} or {stat_y} missing. Skipping.")
            return

        x_min, x_max = m[stat_x].min(), m[stat_x].max()
        y_min, y_max = m[stat_y].min(), m[stat_y].max()

        norm_x = m[stat_x].apply(lambda v: normalize(v, x_min, x_max, pitcher_invert))
        norm_y = m[stat_y].apply(lambda v: normalize(v, y_min, y_max, batter_invert))

        feat[f"matchup_{stat}"] = norm_x - norm_y

    for stat in pitcher_positive_stats: build_matchup(stat, pitcher_invert=False, batter_invert=True)
    for stat in batter_positive_stats: build_matchup(stat, pitcher_invert=True, batter_invert=False)

    print("feat")
    utils.pdf(feat.tail(3))

    return feat


def clusterson(df_window, numerical_columns, categorical_columns,
               pitch_clusters_per_type=5, pitcher_clusters=16,
               model_dir="data/models"):
    df_window = df_window.dropna(subset=numerical_columns + categorical_columns).copy()
    clustered_frames = []
    pitch_models = {}

    os.makedirs(model_dir, exist_ok=True)

    # --- Step 1: Pitch Clustering (per pitch type) ---
    for ptype in df_window['pitch_type'].unique():
        subset = df_window[df_window['pitch_type'] == ptype].copy()
        if len(subset) < pitch_clusters_per_type:
            continue

        # Numerical + categorical processing
        scaler = StandardScaler()
        X_num = scaler.fit_transform(subset[numerical_columns])

        encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        X_cat = encoder.fit_transform(subset[categorical_columns])

        # Construct DataFrame with labeled columns
        col_names = (
            [f"{col}_num" for col in numerical_columns] +
            encoder.get_feature_names_out(categorical_columns).tolist()
        )
        X_df = pd.DataFrame(np.hstack([X_num, X_cat]), columns=col_names, index=subset.index)

        # KMeans clustering
        model = MiniBatchKMeans(n_clusters=pitch_clusters_per_type, n_init='auto', random_state=42)
        model.fit(X_df)
        labels = model.predict(X_df)
        subset['pitch_cluster'] = [f"{ptype}_{label}" for label in labels]

        # Save models and encoders
        with open(os.path.join(model_dir, f"{ptype}_kmeans.pkl"), "wb") as f:
            pickle.dump(model, f)
        with open(os.path.join(model_dir, f"{ptype}_encoder.pkl"), "wb") as f:
            pickle.dump(encoder, f)
        with open(os.path.join(model_dir, f"{ptype}_features.pkl"), "wb") as f:
            pickle.dump(col_names, f)

        pitch_models[ptype] = model
        clustered_frames.append(subset)

    if not clustered_frames:
        return {}, None, pd.DataFrame()

    clustered_pitches = pd.concat(clustered_frames, ignore_index=True)

    # --- Step 2: Pitcher Clustering ---
    cluster_counts = clustered_pitches.groupby(['player_name', 'pitch_cluster']).size().unstack(fill_value=0)
    cluster_props = cluster_counts.div(cluster_counts.sum(axis=1), axis=0)

    pitcher_info = clustered_pitches.groupby('player_name').agg({
        'p_throws': lambda x: x.mode()[0],
        'release_pos_x': 'mean'
    })

    pitcher_features = cluster_props.merge(pitcher_info, left_index=True, right_index=True)

    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    p_throws_encoded = encoder.fit_transform(pitcher_features[['p_throws']])
    p_throws_df = pd.DataFrame(p_throws_encoded, index=pitcher_features.index,
                               columns=encoder.get_feature_names_out(['p_throws']))

    pitcher_features_final = pd.concat([pitcher_features.drop(columns=['p_throws']), p_throws_df], axis=1)

    # Final features (drop non-numerical, e.g. release_pos_x if needed)
    features_for_clustering = pitcher_features_final.drop(columns=['release_pos_x'], errors='ignore')

    # Save the exact order of pitcher clustering columns
    with open(os.path.join(model_dir, "pitcher_feature_columns.pkl"), "wb") as f:
        pickle.dump(list(features_for_clustering.columns), f)

    # Cluster pitchers
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_for_clustering)

    pitcher_kmeans = KMeans(n_clusters=pitcher_clusters, n_init=10, random_state=42)
    pitcher_kmeans.fit(X_scaled)
    pitcher_features_final["pitcher_cluster"] = pitcher_kmeans.labels_

    with open(os.path.join(model_dir, "pitcher_kmeans.pkl"), "wb") as f:
        pickle.dump(pitcher_kmeans, f)

    return pitch_models, pitcher_kmeans, pitcher_features_final


def assign_pitcher_clusters(df_today, pitch_models, pitcher_model,
                             numerical_columns, categorical_columns,
                             model_dir="data/models"):
    df_today = df_today.dropna(subset=numerical_columns + categorical_columns).copy()
    clustered_frames = []

    for ptype in df_today['pitch_type'].unique():
        subset = df_today[df_today['pitch_type'] == ptype].copy()
        if ptype not in pitch_models or subset.empty:
            continue

        model = pitch_models[ptype]

        # Load encoder & expected feature list
        with open(os.path.join(model_dir, f"{ptype}_encoder.pkl"), "rb") as f:
            encoder = pickle.load(f)
        with open(os.path.join(model_dir, f"{ptype}_features.pkl"), "rb") as f:
            expected_cols = pickle.load(f)

        scaler = StandardScaler()
        X_num = scaler.fit_transform(subset[numerical_columns])
        X_cat = encoder.transform(subset[categorical_columns])

        # Combine and reindex to match model expectations
        X = np.hstack((X_num, X_cat))
        col_names = (
            [f"{col}_num" for col in numerical_columns] +
            encoder.get_feature_names_out(categorical_columns).tolist()
        )
        X_df = pd.DataFrame(X, columns=col_names, index=subset.index)
        X_df = X_df.reindex(columns=expected_cols, fill_value=0)

        labels = model.predict(X_df)
        subset["pitch_cluster"] = [f"{ptype}_{label}" for label in labels]
        clustered_frames.append(subset)

    if not clustered_frames:
        return pd.DataFrame()

    clustered_pitches = pd.concat(clustered_frames, ignore_index=True)

    # --- Aggregate for pitcher clustering ---
    cluster_counts = clustered_pitches.groupby(['player_name', 'pitch_cluster']).size().unstack(fill_value=0)
    cluster_props = cluster_counts.div(cluster_counts.sum(axis=1), axis=0)

    pitcher_info = clustered_pitches.groupby('player_name').agg({
        'p_throws': lambda x: x.mode()[0],
        'release_pos_x': 'mean'
    })

    pitcher_features = cluster_props.merge(pitcher_info, left_index=True, right_index=True)

    # One-hot encode handedness
    with open(os.path.join(model_dir, "pitcher_feature_columns.pkl"), "rb") as f:
        expected_pitcher_features = pickle.load(f)

    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    p_throws_encoded = encoder.fit_transform(pitcher_features[['p_throws']])
    p_throws_df = pd.DataFrame(p_throws_encoded, index=pitcher_features.index,
                               columns=encoder.get_feature_names_out(['p_throws']))

    pitcher_features_final = pd.concat([pitcher_features.drop(columns=['p_throws']), p_throws_df], axis=1)

    # Drop unused, reorder to match training columns
    features_for_clustering = pitcher_features_final.drop(columns=['release_pos_x'], errors='ignore')
    features_for_clustering = features_for_clustering.reindex(columns=expected_pitcher_features, fill_value=0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_for_clustering)

    pitcher_features_final["pitcher_cluster"] = pitcher_model.predict(X_scaled)

    return pitcher_features_final.reset_index()


def run_cluster(numerical_columns, categorical_columns, lookback_days=300,
                start_year=None, end_year=None,
                pitch_clusters_per_type=5, pitcher_clusters=10):

    lookback_yrs = 300//365
    df = utils.grab_data('data/statcast', start_year - lookback_yrs, end_year)

    df["game_date"] = pd.to_datetime(df["game_date"])
    all_dates = sorted(df["game_date"].unique())

    if start_year is None:
        start_year = df["game_date"].dt.year.min()
    if end_year is None:
        end_year = df["game_date"].dt.year.max()

    # Use ALL valid calendar dates to avoid gaps from days with no games
    min_date = df["game_date"].min() + timedelta(days=lookback_days)
    max_date = df["game_date"].max()
    valid_feature_dates = pd.date_range(min_date, max_date, freq="D")
    valid_feature_dates = [d for d in valid_feature_dates if start_year <= d.year <= end_year and 3 <= d.month <= 10]

    results = []

    for current_date in tqdm(valid_feature_dates):
        window_start = current_date - timedelta(days=lookback_days)
        window_df = df[(df["game_date"] > window_start) & (df["game_date"] <= current_date)]
        df_today = df[df["game_date"] == current_date]

        if len(window_df) < 100 or df_today.empty:
            continue

        # Step 1: Fit models from trailing window
        pitch_models, pitcher_model, _ = clusterson(
            window_df,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
            pitch_clusters_per_type=pitch_clusters_per_type,
            pitcher_clusters=pitcher_clusters
        )

        # Step 2: Assign clusters to today's pitchers using fitted models
        pitcher_features = assign_pitcher_clusters(
            df_today,
            pitch_models,
            pitcher_model,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns
        )

        if pitcher_features.empty:
            continue

        pitcher_features["feature_date"] = current_date
        results.append(pitcher_features)

    feature_panel = pd.concat(results, ignore_index=True)
    return feature_panel


def prep_test_train(start_year=None, end_year=None,
                    lookback_days=300,
                    pitcher_clusters=10):

    feat = comparatively_speaking(
        start_year=start_year, end_year=end_year, lookback=lookback_days,
        pitcher_positive_stats=[
            'weighted_is_k', 'weighted_whiff_pct', 'weighted_strike_looking_pct', 'weighted_strike_ball_ratio'
        ],
        batter_positive_stats=[
            'weighted_is_hit', 'weighted_launch_speed',
        ]
    ).dropna()

    numerical_columns = ["release_spin_rate", "effective_speed", "pfx_x", "pfx_z", "arm_angle"]
    categorical_columns = ["p_throws"]
    feature_panel = run_cluster(
        numerical_columns, categorical_columns, lookback_days=lookback_days,
        start_year=start_year, end_year=end_year, pitcher_clusters=pitcher_clusters
    )
    print("feature panel")
    utils.pdf(feature_panel.tail(3))

    # feat = pd.read_parquet('data/calc/feat.parquet')
    # feature_panel = pd.read_parquet('data/test/feature_panel.parquet')

    feature_panel['pitcher'] = feature_panel['player_name'].apply(lambda x: f"{x.split(', ')[1][0]}. {x.split(', ')[0]}")
    feature_panel['game_date'] = feature_panel['feature_date'].dt.date
    feat = pd.merge(feat, feature_panel[['game_date', 'pitcher', 'pitcher_cluster']], on=['game_date', 'pitcher'], how='left')
    print('feat here')
    utils.pdf(feat.tail(3))

    ### get target col, strikeouts
    years = range(start_year - 1, end_year + 1)
    all_stats = []
    for yr in years:
        fpath = f"data/statcast/statcast_{yr}.parquet"
        if os.path.isfile(fpath):
            all_stats.append(pd.read_parquet(fpath))
    if not all_stats:
        print("No statcast data found.")
        return pd.DataFrame()

    stats = pd.concat(all_stats, ignore_index=True)
    stats["game_date"] = pd.to_datetime(stats["game_date"]).dt.date

    # Mark at-bat groupings
    stats["is_k"] = stats["events"].isin(["strikeout"])
    grouped_events = (
        stats.dropna(subset=["events"])  # only rows w/ actual event
        .groupby(["player_name", "game_date", "p_throws", "inning_topbot", "game_pk", "home_team", "away_team"], dropna=False)
        .agg({"is_k":"sum","events":"count"}).reset_index()
    )
    grouped_events['pitcher'] = grouped_events['player_name'].apply(
        lambda x: f"{x.split(', ')[1][0]}. {x.split(', ')[0]}")
    k_s = (grouped_events[['game_date','game_pk','home_team','away_team','pitcher','is_k']]
           .sort_values(by=['game_date','home_team'])
           .drop_duplicates())

    ### extra stats
    cols = ['game_date','game_pk','game_type','inning_topbot','player_name','home_team','away_team','pitcher_days_since_prev_game']

    top = stats[stats.inning_topbot == 'Top'][cols].drop_duplicates()
    top['pitch_team'] = top['home_team']
    top['bat_team'] = top['away_team']

    bot = stats[stats.inning_topbot == 'Bot'][cols].drop_duplicates()
    bot['pitch_team'] = bot['away_team']
    bot['bat_team'] = bot['home_team']

    m = pd.concat([top, bot]).sort_values(by=['game_date', 'home_team']).drop_duplicates().reset_index(drop=True)
    m['pitcher'] = m['player_name'].apply(lambda x: f"{x.split(', ')[1][0]}. {x.split(', ')[0]}")
    m['pitch_is_home'] = m['pitch_team'] == m['home_team']
    m.drop(columns=['player_name','inning_topbot'], inplace=True)

    feat = pd.merge(feat,m,on=['game_date','game_pk','home_team','away_team','pitcher'],how='left')
    feat = pd.merge(feat,k_s,on=['game_date','game_pk','home_team','away_team','pitcher'],how='left').dropna()
    utils.pdf(feat.tail(3))

    # ### sched, need sched
    # sched = []
    # for i in range(feat['game_date'].min().year, feat['game_date'].max().year+1):
    #     sched += [pd.read_csv(f'data/sched/{i}schedule.csv')]
    # sched = pd.concat(sched)
    # utils.pdf(sched.tail(10))

    # Set your target and feature columns
    target_col = 'is_k'

    categorical_cols = [
        'pitch_is_home','pitcher_cluster'
    ]
    feat[categorical_cols] = feat[categorical_cols].fillna('Unknown')
    feat = pd.get_dummies(feat, columns=categorical_cols, drop_first=True)

    drop_cols = [
        'game_pk','game_date', 'home_team','away_team', 'pitcher', 'player_name', 'pitch_team',
        'bat_team', 'game_type'
    ]
    feature_cols = [col for col in feat.columns if col not in drop_cols + [target_col]]

    # Step 3: Sort by date
    feat = feat.sort_values("game_date")
    for col in feat.columns:
        if pd.api.types.is_integer_dtype(feat[col]):
            feat[col] = feat[col].fillna(0).astype(np.int64)

    # Step 4: Time-based split
    split_frac = 0.8
    split_idx = int(len(feat) * split_frac)

    train_df = feat.iloc[:split_idx]
    test_df = feat.iloc[split_idx:]

    # Step 5: Build final arrays
    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[target_col].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df[target_col].values.astype(np.float32)

    return X_train, y_train, X_test, y_test, train_df, test_df


if __name__ == "__main__":
    # Example usage
    # plot_weight_decay()  # see the logistic shape

    start_year, end_year = 2022, 2025
    lookback = 300

    prep_test_train(
        start_year=start_year, end_year=end_year,
        lookback_days=lookback, pitcher_clusters=16
    )
