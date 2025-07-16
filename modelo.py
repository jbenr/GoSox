import pandas as pd
import numpy as np
from datetime import datetime

import data_crunchski
import utils

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, TensorDataset

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# class RNNPredictor(nn.Module):
#     def __init__(self, input_size, hidden_size=16, num_layers=1):
#         super(RNNPredictor, self).__init__()
#         self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, 1)
#
#     def forward(self, x):
#         out, _ = self.rnn(x)
#         out = self.fc(out[:, -1, :])  # Take last time step
#         return out
#
#
# def modelo_torch(X_train, y_train, X_test, y_test):
#     print(f"Training model on {len(X_train)} samples with {X_train.shape[1]} features...")
#
#     X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
#     y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
#     X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
#     y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)
#
#     train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
#     train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
#
#     input_size = X_train.shape[-1]
#     model = RNNPredictor(input_size)
#
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
#
#     for epoch in range(100):
#         model.train()
#         running_loss = 0.0
#         for batch_X, batch_y in train_loader:
#             batch_X = batch_X.unsqueeze(1)  # RNN requires (batch, seq_len, features)
#             output = model(batch_X)
#             loss = criterion(output, batch_y)
#
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             running_loss += loss.item()
#
#         avg_loss = running_loss / len(train_loader)
#         print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")
#
#     # Evaluate on test set
#     model.eval()
#     with torch.no_grad():
#         X_test_tensor = X_test_tensor.unsqueeze(1)
#         preds = model(X_test_tensor).squeeze().numpy()
#         y_true = y_test_tensor.squeeze().numpy()
#
#     return model, preds, y_true


def modelo_keras(X_train, y_train, X_test, y_test, verbose=True):
    def create_model(input_dim):
        model = Sequential()
        model.add(Dropout(0.1, input_shape=(input_dim,)))
        model.add(Dense(input_dim, activation='elu'))
        model.add(Dense((input_dim + 1) // 2, activation='elu'))
        model.add(Dense((input_dim + 1) // 3, activation='elu'))
        model.add(Dense(1, activation='linear'))
        return model

    def sign_penalty(y_true, y_pred):
        penalty = 1.3
        loss = tf.where(tf.less(y_true * y_pred, 0),
                        penalty * tf.square(y_true - y_pred),
                        tf.square(y_true - y_pred))
        return tf.reduce_mean(loss, axis=-1)

    keras.losses.sign_penalty = sign_penalty

    tf.keras.backend.clear_session()
    model = create_model(X_train.shape[1])
    model.compile(optimizer='adam', loss=sign_penalty)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, mode='auto')

    model.fit(X_train, y_train, epochs=100, callbacks=[reduce_lr], verbose=0 if not verbose else 1)

    train_preds = model.predict(X_train).flatten()
    test_preds = model.predict(X_test).flatten()

    r2 = r2_score(y_train, train_preds)
    mae = mean_absolute_error(y_train, train_preds)
    mse = mean_squared_error(y_train, train_preds)

    if verbose:
        print(f'\nR2: {r2:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}')

    return model, test_preds



def main():
    start_year, end_year = 2022, 2024
    lookback = 300
    X_train, y_train, X_test, y_test, train_df, test_df = data_crunchski.prep_test_train(
        start_year=start_year, end_year=end_year,
        lookback_days=lookback
    )
    X_train.to_parquet('data/train_test/X_train.parquet')
    y_train.to_parquet('data/train_test/y_train.parquet')
    X_test.to_parquet('data/train_test/X_test.parquet')
    y_test.to_parquet('data/train_test/y_test.parquet')
    train_df.to_parquet('data/train_test/train_df.parquet')
    test_df.to_parquet('data/train_test/test_df.parquet')

    # model, preds, y_true = modelo_torch(X_train, y_train, X_test, y_test)
    model, preds = modelo_keras(X_train, y_train, X_test, y_test)

    test_df['pred'] = preds.round(2)
    display_cols = ['game_date','pitcher','pitch_team','home_team','away_team','pitch_is_home_True','is_k','pred']

    odds = []
    for val in test_df['game_date']:
        temp = pd.read_parquet(f'data/strikeout_odds/{val.year}/strikeout_odds_{val.strftime("%m_%d_%Y")}.parquet').drop_duplicates()
        temp['game_date'] = val
        temp = temp.drop_duplicates()
        odds += [temp]
    odds = pd.concat(odds)
    odds.drop_duplicates(inplace=True)
    odds['consensus_ou'] = pd.to_numeric(odds['consensus_ou'])
    display_cols += ['consensus_ou','consensus_over_odds','consensus_under_odds']

    test_df = pd.merge(test_df, odds, how='left', on=['game_date','pitcher','home_team','away_team'])[display_cols]

    test_df['diff'] = test_df['pred'] - test_df['consensus_ou']
    threshold = 1.5
    conditions = [
        test_df['diff'] > threshold,
        test_df['diff'] < -threshold
    ]
    choices = ['over', 'under']
    test_df['bet'] = np.select(conditions, choices, default=None)

    conditions = [
        test_df['is_k'] > test_df['consensus_ou'],
        test_df['is_k'] < test_df['consensus_ou']
    ]
    choices = ['over', 'under']
    test_df['result'] = np.select(conditions, choices, default='push')

    test_df['win?'] = (test_df['bet'] == test_df['result']).astype(int)
    test_df.dropna(inplace=True, subset=['bet'])
    utils.pdf(test_df)
    print(test_df['win?'].sum()/len(test_df['win?']), len(test_df['win?']))


if __name__ == "__main__":
    main()