import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from pymongo import MongoClient
from sklearn.preprocessing import StandardScaler

from models.lstm import AlgorithmicMomentumLSTM

import os
from dotenv import load_dotenv

load_dotenv()

uri = os.getenv("MONGODB_URI")

print("Connecting to MongoDB...")
client = MongoClient(uri)
collection = client["stock_tracking_db"]["historical_prices"]

TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "NFLX", "INTC",
    "CSCO", "ADBE", "QCOM", "PYPL", "AMD", "TMUS", "COST", "BKNG", "AMGN", "SBUX"
]

features = ["close", "volume", "log_return", "macd", "rsi_14"]
scaler = StandardScaler()

def create_multistep_sequences(data, lookback=60, forecast=5, threshold=0.005):
    xs, ys = [], []
    for i in range(len(data) - lookback - forecast):
        xs.append(data[i : i + lookback])

        current_price = data[i + lookback - 1, 0]
        future_price = data[i + lookback + forecast - 1, 0]
        actual_return = (future_price / current_price) - 1

        if actual_return > threshold:
            label = 0 # up
        elif actual_return < -threshold:
            label = 1 # down
        else:
            label = 2 # neutral

        ys.append(label)
    return np.array(xs), np.array(ys)

all_train_X, all_train_y = [], []
all_test_X, all_test_y = [], []

print("Extracting and formatting data per stock...")
for ticker in TICKERS:
    cursor = collection.find({"ticker": ticker})
    df = pd.DataFrame(list(cursor))

    if df.empty:
        print(f"No data found for {ticker}, skipping.")
        continue

    df = df.drop(columns=["_id"]).sort_values("date").reset_index(drop=True)
    df = df.dropna()

    scale_data = scaler.fit_transform(df[features].values)

    X, y = create_multistep_sequences(scale_data)

    split_idx = int(0.8 * len(X))
    all_train_X.append(X[:split_idx])
    all_train_y.append(y[:split_idx])

    all_test_X.append(X[split_idx:])
    all_test_y.append(y[split_idx:])

X_train_global = np.vstack(all_train_X)
y_train_global = np.concatenate(all_train_y).ravel()
X_test_global = np.vstack(all_test_X)
y_test_global = np.concatenate(all_test_y).ravel()

# ---------------------------------------------------------------------
print("\n--- Label Distribution Audit ---")
unique, counts = np.unique(y_train_global, return_counts=True)
dist = dict(zip(unique, counts))
total = sum(counts)
print(f"Total Sequences: {total}")
print(f"Up (0):      {dist.get(0,0)} ({100*dist.get(0,0)/total:.1f}%)")
print(f"Down (1):    {dist.get(1,0)} ({100*dist.get(1,0)/total:.1f}%)")
print(f"Neutral (2): {dist.get(2,0)} ({100*dist.get(2,0)/total:.1f}%)")
# ---------------------------------------------------------------------

print(f"--- Data Preparation Completed ---")
print(f"Total training sequences: {len(X_train_global)}")
print(f"Total test sequences: {len(X_test_global)}")

X_tensor = torch.tensor(X_train_global, dtype=torch.float32)
y_tensor = torch.tensor(y_train_global, dtype=torch.float32)

train_dataset = TensorDataset(torch.tensor(X_train_global, dtype=torch.float32), torch.tensor(y_train_global, dtype=torch.long))
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(torch.tensor(X_test_global, dtype=torch.float32), torch.tensor(y_test_global, dtype=torch.long))
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("Initializing LSTM Engine...")
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model = AlgorithmicMomentumLSTM().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-4)

print(f"Starting Training Phase on {device}...")
EPOCHS = 30

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0

    for batch_X, batch_y in train_dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device).long()

        optimizer.zero_grad()
        logits = model(batch_X)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    model.eval()
    total_test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_X, batch_y in test_dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device).long()

            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            total_test_loss += loss.item()

            probabilities = torch.softmax(logits, dim=1)
            _, predicted = torch.max(probabilities, dim=1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    avg_train_loss = total_train_loss / len(train_dataloader)
    avg_test_loss = total_test_loss / len(test_dataloader)
    accuracy = (correct / total) * 100

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch + 1}/{EPOCHS}]")
        print(f"Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")
        print(f"Test Accuracy: {accuracy:.2f}%\n")
        print("-" * 30)

torch.save(model.state_dict(), "lstm_weights.pth")
print("\nTraining Completed! Neural weights saved as 'lstm_weights.pth'.")