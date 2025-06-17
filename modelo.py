import pandas as pd
import numpy as np

import data_crunchski
import utils


def modelo():
    data_crunchski.prep_test_train()
    return None


# b = pd.read_parquet('data/calc/df_offense.parquet')
# p = pd.read_parquet('data/calc/df_pitchers.parquet').sort_values(by='game_date').dropna()
# p_c = pd.read_parquet('data/clust/pitcher_clusters.parquet')
# p_c_ = pd.read_parquet('data/clust/pitch_clusters.parquet')
#
# top = pd.merge(
#     p[p.inning_topbot=='Top'], b, how='left',
#     left_on=['game_date','game_pk','away_team'],
#     right_on=['game_date','game_pk','batting_team']
# )
# bot = pd.merge(
#     p[p.inning_topbot=='Bot'], b, how='left',
#     left_on=['game_date','game_pk','home_team'],
#     right_on=['game_date','game_pk','batting_team']
# )
#
# m = pd.concat([top,bot]).sort_values(by=['game_date','home_team']).dropna().reset_index(drop=True)
#
#
# m['pred_so'] = 0.9*(m['weighted_is_k_y']*(m['weighted_events_x']/m['weighted_events_y'])) + 0.1*(m['weighted_is_k_x'])
# m["pred_ou"] = np.where(
#         m["pred_so"] > m['consensus_ou'],
#         "over","under"
# )
# m['win?'] = np.where(
#     m['pred_ou'] == m['result'],1,0
# )
# m['diff'] = abs(m['pred_so']-m['consensus_ou'])
# utils.pdf(m.tail(10))
# m = m[[
#     'game_date','pitcher','home_team','away_team','is_k_x','consensus_ou','pred_so',
#     'consensus_over_odds','consensus_under_odds','result','pred_ou','diff','win?'
# ]]
# utils.pdf(m.tail(10))
#
# print(m['win?'].sum()/len(m))
# print(m[m['diff'].abs() > 1]['win?'].sum()/len(m[m['diff'].abs() > 1]))
#
# min_val = m["diff"].min()
# max_val = m["diff"].max()
# bin_width = 0.1
# bin_edges = np.arange(min_val, max_val + bin_width, bin_width)
#
# m["diff_bin"] = pd.cut(m["diff"], bins=bin_edges, right=True)
#
# grouped = (
#     m.groupby("diff_bin")
#     .agg(
#         win_rate=("win?", "mean"),  # average of "win?" => proportion of 1s
#         count=("win?", "size")      # how many rows in that bin
#     )
#     .reset_index()
# )
#
# utils.pdf(grouped)
#
# print("P")
# utils.pdf(p.tail(10))
#
# print("B")
# utils.pdf(b.tail(10))
#
# print("P_C")
# utils.pdf(p_c.tail(10))
#
# print("P_C_")
# utils.pdf(p_c_.tail(10))
# utils.pdf(p_c_.groupby(['kmeans_cluster','pitch_name']).agg({
#     'effective_speed': 'mean',
#     'release_spin_rate': 'mean',
#     'pfx_x': 'mean',
#     'pfx_z': 'mean',
#     'description': 'count',
# }))