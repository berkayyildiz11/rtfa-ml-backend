from pathlib import Path
import pandas as pd

from src.data.mongodb_loader import load_stock_data, load_sp500_data


def export_training_data():
    output_dir = Path("data/training")
    output_dir.mkdir(parents=True, exist_ok=True)

    stock_df = load_stock_data()
    sp500_df = load_sp500_data()

    stock_df.to_csv(output_dir / "stock_data.csv", index=False)
    sp500_df.to_csv(output_dir / "sp500_data.csv", index=False)

    print("Export completed:")
    print(output_dir / "stock_data.csv")
    print(output_dir / "sp500_data.csv")


if __name__ == "__main__":
    export_training_data()