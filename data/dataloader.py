import torch
from torch.utils.data import Dataset, DataLoader
import nasdaqdatalink
import pandas as pd

# nasdaqdatalink.ApiConfig.api_key ='5YAuUoytxGpvWSLKmEjA'
# end_date = '2024-11-7'
# start_date = '2020-11-7'

# data = nasdaqdatalink.get_table(
#     'QDL/BITFINEX',
#     code='ETHUSD',
#     date={'gte': start_date, 'lte': end_date}
# )

# print(len(data))

class KalshiDataset(Dataset):
    def __init__(self, csv_file="data/kalshi.csv"):
        self.data = pd.read_csv(csv_file)
        feature_columns = [
            "current_price", "price_change_1h", "price_change_6h", "price_change_24h",
            "rolling_mean_6h", "rolling_std_6h", "hour", "day_of_week", "is_weekend",
            "is_market_hours", "time_till_resolution"
        ]
        self.features = self.data[feature_columns].values
        self.labels = self.data["label"].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y

class CryptoCurrencyDataset(Dataset):
    def __init__(self, token='ETHUSD', start_date='2016-11-7', end_date='2024-11-7', lookback=3):
        self.data = nasdaqdatalink.get_table(
                            'QDL/BITFINEX',
                            code=token,
                            date={'gte': start_date, 'lte': end_date})
        self.lookback = lookback

    def __len__(self):
        return len(self.data) - self.lookback

    def __getitem__(self, idx):
        x = torch.tensor(self.data['last'].values[idx:idx + self.lookback], dtype=torch.float32)
        x = (x - x.mean()) / x.std()
        future_price = self.data['last'].values[idx + self.lookback]
        curr_price = self.data['last'].values[idx + self.lookback - 1]
        y = torch.tensor(1 if future_price > curr_price else 0, dtype=torch.float32)
        return x, y