from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.data.mondodb_dataloader import load_stock_data, load_sp500_data


WINDOW_SIZE = 30 
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3

FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "log_return",
    "macd",
    "rsi_14",
    "sp500_value",
    "sp500_log_return",
    "vix",
    "beta",
    "correlation_sp_vix"
]

TARGET_COLUMN = "target_relative_return"

def prepare_dataframe(stock_df: pd.DataFrame, sp500_df: pd.DataFrame) -> pd.DataFrame:
    stock_df = stock_df.copy()
    sp500_df = sp500_df.copy()

    stock_df["date"] = pd.to_datetime(stock_df["date"])
    sp500_df["date"] = pd.to_datetime(sp500_df["date"])

    sp500_df = sp500_df.sort_values("date")
    sp500_df["sp500_log_return"] = np.log(sp500_df["close"] / sp500_df["close"].shift(1))

    sp500_features = sp500_df[
        ["date", "close", "vix", "rsi_14", "beta", "correlation_sp_vix", "sp500_log_return"]
    ].rename(columns={"close": "sp500_close"})

    df = stock_df.merge(sp500_features, on="date", how="inner")

    if "sp500_value" not in df.columns:
        df["sp500_value"] = df["sp500_close"]

    df["relative_return"] = df["log_return"] - df["sp500_log_return"]

    df = df.sort_values(["ticker", "date"])
    df[TARGET_COLUMN] = df.groupby("ticker")["relative_return"].shift(-1)

    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])

    return df

def chronological_split(df: pd.DataFrame):
    dates = sorted(df["date"].unique())

    train_end = int(len(dates) * 0.7)
    val_end = int(len(dates) * 0.85)

    train_dates = dates[:train_end]
    val_dates = dates[train_end:val_end]
    test_dates = dates[val_end:]

    train_df = df[df["date"].isin(train_dates)].copy()
    val_df = df[df["date"].isin(val_dates)].copy()
    test_df = df[df["date"].isin(test_dates)].copy()

    return train_df, val_df, test_df

def scale_features(train_df, val_df, test_df):

    scaler = StandardScaler()

    train_df[FEATURE_COLUMNS] = scaler.fit_transform(train_df[FEATURE_COLUMNS])
    val_df[FEATURE_COLUMNS] = scaler.transform(val_df[FEATURE_COLUMNS])
    test_df[FEATURE_COLUMNS] = scaler.transform(test_df[FEATURE_COLUMNS])

    return train_df, val_df, test_df, scaler

class RelativeReturnDataset(Dataset):
    def __init__(self, df: pd.DataFrame, window_size: int = 30):
        self.X = []
        self.y = []

        for ticker, group in df.groupby("ticker"):
            group = group.sort_values("date").reset_index(drop=True)

            features = group[FEATURE_COLUMNS].values.astype(np.float32)
            targets = group[TARGET_COLUMN].values.astype(np.float32)

            for i in range(len(group) - window_size):
                self.X.append(features[i : i + window_size])
                self.y.append(targets[i + window_size - 1])

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class LSTMRelativeReturnModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        last_hidden = hidden[-1]
        return self.fc(last_hidden)
    

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            preds = model(X).cpu().numpy()

            y_pred.extend(preds.flatten())
            y_true.extend(y.numpy().flatten())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    directional_accuracy = accuracy_score(
        y_true > 0,
        y_pred > 0,
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "directional_accuracy": directional_accuracy
    }

def run_lstm_relative():
    stock_df = load_stock_data()
    sp500_df = load_sp500_data()

    df = prepare_dataframe(stock_df, sp500_df)

    train_df, val_df, test_df = chronological_split(df)
    train_df, val_df, test_df, scaler = scale_features(train_df, val_df, test_df)

    train_dataset = RelativeReturnDataset(train_df, window_size=WINDOW_SIZE)
    val_dataset = RelativeReturnDataset(val_df, window_size=WINDOW_SIZE)
    test_dataset = RelativeReturnDataset(test_df, window_size=WINDOW_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMRelativeReturnModel(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        val_metrics = evaluate(model, val_loader, device)

        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Validation: {val_metrics}")

        print(f"Final Test Evaluation:")
        test_metrics = evaluate(model, test_loader, device)
        print(test_metrics)

        Path("models/lstm/saved_models").mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), "models/lstm/saved_models/lstm_relative_return.pt")

        return model, scaler, test_metrics
    

if __name__ == "__main__":
    run_lstm_relative()