import pybaseball as pyb
import utils

# df = pyb.statcast('2024-09-01','2024-10-06')
# g = df.groupby(['pitcher','player_name','p_throws','pitch_name']).agg({
#     'pitch_type':'count',
#     'effective_speed':'mean',
#     'release_spin_rate':'mean',
#     'release_pos_y':'mean',
#     'arm_angle':'mean',
#     'spin_axis':'mean',
#     'api_break_z_with_gravity':'mean',
#     'api_break_x_arm':'mean',
# }).reset_index()

# helpers.pdf(df.tail(10))

log = pyb.team_game_logs(2024, 'BOS')
utils.pdf(log.tail(10))

df = pyb.team_ids(2019,'ALL')
utils.pdf(df)

st = pyb.schedule_and_record(2019, 'BOS')
utils.pdf(st.tail(10))