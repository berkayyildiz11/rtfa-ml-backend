from pathlib import Path
import copy
import json
import random

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.data.mongodb_loader import load_stock_data, load_sp500_data

SEED = 777

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


WINDOW_SIZE = 126
BATCH_SIZE = 16
EPOCHS = 50
LR = 5e-4
EARLY_STOPPING_PATIENCE = 8

NEUTRAL_THRESHOLD = 0.03
NUM_CLASSES = 3

TARGET_RETURN_COLUMN = "target_relative_return"
TARGET_CLASS_COLUMN = "target_class"

FEATURE_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "log_return", "macd", "rsi_14",
    "sp500_value", "sp500_log_return", "sp500_rsi_14",
    "vix", "beta", "correlation_sp_vix",
]

HORIZON_CONFIGS = {
    "1d": 1,
    "1w": 5,
    "1m": 21,
    "3m": 63,
    "6m": 126,
    "1y": 252,
}


def prepare_dataframe(
    stock_df: pd.DataFrame,
    sp500_df: pd.DataFrame,
    forecast_horizon: int,
) -> pd.DataFrame:
    stock_df = stock_df.copy()
    sp500_df = sp500_df.copy()

    stock_df["date"] = pd.to_datetime(stock_df["date"])
    sp500_df["date"] = pd.to_datetime(sp500_df["date"])

    stock_df = stock_df.sort_values(["ticker", "date"])
    sp500_df = sp500_df.sort_values("date")

    sp500_df["sp500_log_return"] = np.log(sp500_df["close"] / sp500_df["close"].shift(1))

    sp500_features = sp500_df[
        ["date", "close", "vix", "rsi_14", "beta", "correlation_sp_vix", "sp500_log_return"]
    ].rename(columns={"close": "sp500_close", "rsi_14": "sp500_rsi_14"})

    df = stock_df.merge(sp500_features, on="date", how="inner")

    if "sp500_value" not in df.columns:
        df["sp500_value"] = df["sp500_close"]

    df["relative_return"] = df["log_return"] - df["sp500_log_return"]

    df[TARGET_RETURN_COLUMN] = (
        df.groupby("ticker")["relative_return"]
        .transform(
            lambda x: x.shift(-1)
            .rolling(window=forecast_horizon)
            .sum()
            .shift(-(forecast_horizon - 1))
        )
    )

    df[TARGET_CLASS_COLUMN] = 1
    df.loc[df[TARGET_RETURN_COLUMN] > NEUTRAL_THRESHOLD, TARGET_CLASS_COLUMN] = 2
    df.loc[df[TARGET_RETURN_COLUMN] < -NEUTRAL_THRESHOLD, TARGET_CLASS_COLUMN] = 0

    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_RETURN_COLUMN, TARGET_CLASS_COLUMN])
    df[TARGET_CLASS_COLUMN] = df[TARGET_CLASS_COLUMN].astype(int)

    return df


def chronological_split(df: pd.DataFrame):
    dates = sorted(df["date"].unique())

    train_end = int(len(dates) * 0.7)
    val_end = int(len(dates) * 0.85)

    train_dates = dates[:train_end]
    val_dates = dates[train_end:val_end]
    test_dates = dates[val_end:]

    return (
        df[df["date"].isin(train_dates)].copy(),
        df[df["date"].isin(val_dates)].copy(),
        df[df["date"].isin(test_dates)].copy(),
    )


def scale_features(train_df, val_df, test_df):
    scaler = StandardScaler()

    train_df[FEATURE_COLUMNS] = scaler.fit_transform(train_df[FEATURE_COLUMNS])
    val_df[FEATURE_COLUMNS] = scaler.transform(val_df[FEATURE_COLUMNS])
    test_df[FEATURE_COLUMNS] = scaler.transform(test_df[FEATURE_COLUMNS])

    return train_df, val_df, test_df, scaler


class RelativeReturnClassificationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, allowed_dates, window_size: int = 30):
        self.X = []
        self.y = []

        allowed_dates = set(pd.to_datetime(allowed_dates))

        for _, group in df.groupby("ticker"):
            group = group.sort_values("date").reset_index(drop=True)

            features = group[FEATURE_COLUMNS].values.astype(np.float32)
            targets = group[TARGET_CLASS_COLUMN].values.astype(np.int64)
            dates = pd.to_datetime(group["date"]).values

            for i in range(len(group) - window_size):
                target_idx = i + window_size - 1
                target_date = pd.to_datetime(dates[target_idx])

                if target_date in allowed_dates:
                    self.X.append(features[i:i + window_size])
                    self.y.append(targets[target_idx])

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_split_dates(df: pd.DataFrame):
    dates = sorted(df["date"].unique())

    train_end = int(len(dates) * 0.7)
    val_end = int(len(dates) * 0.85)

    train_dates = dates[:train_end]
    val_dates = dates[train_end:val_end]
    test_dates = dates[val_end:]

    return train_dates, val_dates, test_dates

"""
class RelativeReturnClassificationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, window_size: int = 30):
        self.X = []
        self.y = []

        for _, group in df.groupby("ticker"):
            group = group.sort_values("date").reset_index(drop=True)

            features = group[FEATURE_COLUMNS].values.astype(np.float32)
            targets = group[TARGET_CLASS_COLUMN].values.astype(np.int64)

            for i in range(len(group) - window_size):
                self.X.append(features[i:i + window_size])
                self.y.append(targets[i + window_size - 1])

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y = torch.tensor(np.array(self.y), dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
"""

class LSTMRelativeReturnClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.4):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, NUM_CLASSES),
        )

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)

            logits = model(X)
            loss = criterion(logits, y)
            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item()
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    return {
        "loss": total_loss / len(loader),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=["worse", "neutral", "better"],
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]),
    }


def print_class_distribution_from_dataset(dataset, name):
    y = dataset.y.numpy()
    unique, counts = np.unique(y, return_counts=True)

    print(f"\n{name} dataset class distribution:")
    print(dict(zip(unique, counts)))

    total = len(y)
    print({int(k): float(v / total) for k, v in zip(unique, counts)})


def print_majority_baseline(dataset, name):
    y = dataset.y.numpy()
    majority_class = np.bincount(y).argmax()
    baseline_accuracy = (y == majority_class).mean()

    labels = {0: "worse", 1: "neutral", 2: "better"}

    print(f"\n{name} majority baseline:")
    print(f"Majority class: {majority_class} ({labels[majority_class]})")
    print(f"Baseline accuracy: {baseline_accuracy:.4f}")


def save_metrics(metrics, save_path):
    serializable_metrics = {
        "loss": metrics["loss"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "classification_report": metrics["classification_report"],
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
    }

    with open(save_path, "w") as f:
        json.dump(serializable_metrics, f, indent=4)


def run_lstm_relative(
    horizon_name="1y",
    forecast_horizon=252,
    stock_csv_path=None,
    sp500_csv_path=None,
):
    if stock_csv_path and sp500_csv_path:
        stock_df = pd.read_csv(stock_csv_path)
        sp500_df = pd.read_csv(sp500_csv_path)
    else:
        stock_df = load_stock_data()
        sp500_df = load_sp500_data()

    df = prepare_dataframe(stock_df, sp500_df, forecast_horizon)
    
    """train_df, val_df, test_df = chronological_split(df)
    train_df, val_df, test_df, scaler = scale_features(train_df, val_df, test_df)

    train_dataset = RelativeReturnClassificationDataset(train_df, WINDOW_SIZE)
    val_dataset = RelativeReturnClassificationDataset(val_df, WINDOW_SIZE)
    test_dataset = RelativeReturnClassificationDataset(test_df, WINDOW_SIZE)"""

    train_dates, val_dates, test_dates = get_split_dates(df)

    train_df = df[df["date"].isin(train_dates)].copy()

    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLUMNS])

    df[FEATURE_COLUMNS] = scaler.transform(df[FEATURE_COLUMNS])

    train_dataset = RelativeReturnClassificationDataset(df, train_dates, WINDOW_SIZE)
    val_dataset = RelativeReturnClassificationDataset(df, val_dates, WINDOW_SIZE)
    test_dataset = RelativeReturnClassificationDataset(df, test_dates, WINDOW_SIZE)

    print_class_distribution_from_dataset(train_dataset, "Train")
    print_class_distribution_from_dataset(val_dataset, "Validation")
    print_class_distribution_from_dataset(test_dataset, "Test")

    print_majority_baseline(train_dataset, "Train")
    print_majority_baseline(val_dataset, "Validation")
    print_majority_baseline(test_dataset, "Test")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"\nUsing device: {device}")

    model = LSTMRelativeReturnClassifier(
        input_size=len(FEATURE_COLUMNS),
        hidden_size=64,
        num_layers=1,
        dropout=0.4,
    ).to(device)

    class_weights = torch.tensor([1.0, 1.2, 1.0], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    best_val_macro_f1 = -1
    best_model_state = None
    best_val_metrics = None
    epochs_without_improvement = 0

    save_dir = Path("saved_models/lstm")
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_metrics["macro_f1"])

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Validation Loss: {val_metrics['loss']:.6f}")
        print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"Validation Macro F1: {val_metrics['macro_f1']:.4f}")
        print(f"Validation Weighted F1: {val_metrics['weighted_f1']:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.8f}")

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_model_state = copy.deepcopy(model.state_dict())
            best_val_metrics = val_metrics
            epochs_without_improvement = 0
            torch.save(best_model_state, save_dir / f"best_lstm_{horizon_name}.pt")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("\nEarly stopping triggered.")
            break

    model.load_state_dict(best_model_state)

    print("\nBest Validation Results:")
    print(f"Accuracy: {best_val_metrics['accuracy']:.4f}")
    print(f"Macro F1: {best_val_metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {best_val_metrics['weighted_f1']:.4f}")
    print(best_val_metrics["classification_report"])
    print("Confusion Matrix:")
    print(best_val_metrics["confusion_matrix"])

    print("\nFinal Test Evaluation:")
    test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"Test Weighted F1: {test_metrics['weighted_f1']:.4f}")
    print(test_metrics["classification_report"])
    print("Confusion Matrix:")
    print(test_metrics["confusion_matrix"])

    torch.save(model.state_dict(), save_dir / f"final_lstm_{horizon_name}.pt")
    save_metrics(test_metrics, save_dir / f"lstm_{horizon_name}_test_metrics.json")

    return model, scaler, test_metrics


if __name__ == "__main__":
    run_lstm_relative(
        horizon_name="1y",
        forecast_horizon=HORIZON_CONFIGS["1y"],
    )