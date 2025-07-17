import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from datetime import timedelta

import data_crunchski
import utils

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# from tensorflow import keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


def modelo_torch(X_train, y_train, X_test, y_test=None):
    print(f"Training model on {len(X_train)} samples with {X_train.shape[1]} features...")

    # 1. Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 2. Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # 3. Train/Val split (only if there's enough data to do so)
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    if len(dataset) >= 2:
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        use_validation = True
    else:
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        use_validation = False

    # 4. Model & training setup
    input_size = X_train.shape[1]
    model = MLP(input_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # 5. Training loop with optional early stopping
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(100):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            output = model(batch_X)
            loss = criterion(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Optional validation
        if use_validation:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    val_output = model(batch_X)
                    val_loss += criterion(val_output, batch_y).item()
            val_loss /= len(val_loader)

            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break
        else:
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}")

        scheduler.step()

    # 6. Inference
    model.eval()
    with torch.no_grad():
        preds = model(X_test_tensor).squeeze().numpy()
        if y_test is not None:
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)
            y_true = y_test_tensor.squeeze().numpy()
        else:
            y_true = None

    return model, preds, y_true


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


def collate(test_df, preds, var, thresh=0):
    test_df['pred'] = preds.round(2)
    test_df['var'] = var.round(4)
    display_cols = ['game_date', 'pitcher', 'pitch_team', 'home_team', 'away_team', 'pitch_is_home_True', 'is_k', 'pred','var']

    odds = []
    for val in test_df['game_date']:
        temp = pd.read_parquet(
            f'data/strikeout_odds/{val.year}/strikeout_odds_{val.strftime("%m_%d_%Y")}.parquet').drop_duplicates()
        temp['game_date'] = val
        temp = temp.drop_duplicates()
        odds += [temp]
    odds = pd.concat(odds)
    odds.drop_duplicates(inplace=True)
    odds['consensus_ou'] = pd.to_numeric(odds['consensus_ou'])
    display_cols += ['consensus_ou', 'consensus_over_odds', 'consensus_under_odds']

    test_df = pd.merge(test_df, odds, how='left', on=['game_date', 'pitcher', 'home_team', 'away_team'])[display_cols]

    test_df['diff'] = test_df['pred'] - test_df['consensus_ou']
    test_df['abs_diff'] = test_df['diff'].abs()

    threshold = (thresh, 10)
    conditions = [
        (test_df['diff'] > threshold[0]) & (test_df['diff'] <= threshold[1]),
        (test_df['diff'] < -threshold[0]) & (test_df['diff'] >= -threshold[1])
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

    test_df['plus_odds_bet'] = (
            ((test_df['bet'] == 'over') & test_df['consensus_over_odds'].astype(str).str.contains('\+')) |
            ((test_df['bet'] == 'under') & test_df['consensus_under_odds'].astype(str).str.contains('\+'))
    )

    test_df.dropna(inplace=True, subset=['bet'])
    test_df_dog = test_df[test_df['plus_odds_bet'] == True]
    test_df_fav = test_df[test_df['plus_odds_bet'] == False]

    utils.pdf(test_df)
    print("Overall:")
    print(round(test_df['win?'].sum() / len(test_df['win?']), 4), len(test_df['win?']))

    test_df_dog = test_df[test_df['plus_odds_bet'] == True]
    test_df_fav = test_df[test_df['plus_odds_bet'] == False]
    print("\nDog:")
    print(round(test_df_dog['win?'].sum() / len(test_df_dog['win?']), 4), len(test_df_dog['win?']))
    print("\nFaves:")
    print(round(test_df_fav['win?'].sum() / len(test_df_fav['win?']), 4), len(test_df_fav['win?']))

    # bins
    max_cutoff = 5
    bucket_size = 0.1
    bins = np.arange(0, max_cutoff + bucket_size, bucket_size)
    test_df['mispricing_bucket'] = pd.cut(test_df['abs_diff'], bins=bins, right=False)
    stats = test_df.groupby(['plus_odds_bet', 'mispricing_bucket'], observed=True)['win?'].agg(
        ['mean', 'count']).rename(columns={'mean': 'win_rate'}).reset_index()
    utils.pdf(stats)

    return test_df


def back_test(start_year=2022, end_year=2025, lookback=150, n_runs=10):
    save_path = Path(f'data/bt/backtest_{start_year}_{end_year}_{lookback}')
    save_path.mkdir(parents=True, exist_ok=True)
    daily_save_path = save_path / "preds"
    daily_save_path.mkdir(parents=True, exist_ok=True)

    # Only use files from directories 2022–2025
    odds_dirs = [Path(f"data/strikeout_odds/{y}") for y in range(start_year, end_year + 1)]
    all_odds_files = sorted(f for d in odds_dirs if d.exists() for f in d.glob("strikeout_odds_*.parquet"))

    print(f"Found {len(all_odds_files)} odds files for years {start_year}-{end_year}")

    for file_path in all_odds_files:
        try:
            date_str = "_".join(file_path.stem.split("_")[-3:])
            target_date = datetime.strptime(date_str, "%m_%d_%Y")
            print(target_date)

            output_path = daily_save_path / f"preds_{target_date.strftime('%Y-%m-%d')}.parquet"
            if output_path.exists():
                print(f"✔ Skipping {target_date.date()} (already processed)")
                continue

            start_year_bt = (target_date - timedelta(days=lookback + 1)).year
            end_year_bt = target_date.year
            X_train, y_train, X_test, _, train_df, test_df = data_crunchski.prep_test_train(
                start_year=start_year_bt, end_year=end_year_bt,
                lookback_days=lookback,
                live_mode=True, end_date=target_date
            )

            # Run model n times, store all predictions
            all_preds = []
            for i in range(n_runs):
                model, preds, _ = modelo_torch(X_train, y_train, X_test, None)
                all_preds.append(preds)

            all_preds = np.array(all_preds)  # shape: (n_runs, num_samples)
            mean_preds = all_preds.mean(axis=0)
            var_preds = all_preds.var(axis=0)

            # Add to test_df
            test_df['pred'] = mean_preds
            test_df['var'] = var_preds

            test_df = collate(test_df, mean_preds, var_preds)

            # Save
            test_df.to_parquet(output_path)
            print(f"✅ Saved predictions for {target_date.date()}")

        except Exception as e:
            print(f"❌ Error on {target_date.date()}: {e}")
            continue

    # Final collation
    all_pred_files = sorted(daily_save_path.glob("preds_*.parquet"))
    if all_pred_files:
        full_df = pd.concat([pd.read_parquet(f) for f in all_pred_files])
        full_df.to_parquet(save_path / "backtest_all.parquet")
        print(f"\n✅ Backtest complete! Final file saved to {save_path}/backtest_all.parquet")
    else:
        print("\n❌ No valid prediction files found to collate.")


def main():
    start_year, end_year = 2023, 2025
    lookback = 150
    # X_train, y_train, X_test, y_test, train_df, test_df = data_crunchski.prep_test_train(
    #     start_year=start_year, end_year=end_year,
    #     lookback_days=lookback
    # )
    # np.savez(
    #     "data/train_test/train_test.npz",
    #     X_train=X_train,
    #     y_train=y_train,
    #     X_test=X_test,
    #     y_test=y_test
    # )
    # train_df.to_parquet('data/train_test/train_df.parquet')
    # test_df.to_parquet('data/train_test/test_df.parquet')

    data = np.load("data/train_test/train_test.npz")
    X_train = data['X_train']
    y_train = data['y_train']
    X_test = data['X_test']
    y_test = data['y_test']

    train_df = pd.read_parquet("data/train_test/train_df.parquet")
    test_df = pd.read_parquet("data/train_test/test_df.parquet")

    n_runs = 3
    all_preds = []
    for i in range(n_runs):
        model, preds, _ = modelo_torch(X_train, y_train, X_test, y_test)
        all_preds.append(preds)

    all_preds = np.array(all_preds)  # shape: (n_runs, num_samples)
    mean_preds = all_preds.mean(axis=0)
    var_preds = all_preds.var(axis=0)

    # Add to test_df
    test_df['mean_pred'] = mean_preds
    test_df['var_pred'] = var_preds
    utils.pdf(test_df.tail(5))

    # model, preds, y_true = modelo_torch(X_train, y_train, X_test, y_test)
    # model, preds = modelo_keras(X_train, y_train, X_test, y_test)

    test_df = collate(test_df, mean_preds, var_preds)



if __name__ == "__main__":
    main()