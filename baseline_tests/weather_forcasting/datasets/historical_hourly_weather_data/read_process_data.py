import pandas as pd
import torch


def load_and_process_dataset(file_path, city_name="Toronto", seq_length=30):
    data = pd.read_csv(file_path)

    if "datetime" not in data.columns:
        raise ValueError("The dataset must have a 'datetime' column.")
    data["datetime"] = pd.to_datetime(data["datetime"])
    data = data.sort_values("datetime") # Make sure data are sorted by date

    if city_name not in data.columns:
        raise ValueError(f"The dataset does not have a column named '{city_name}'.")
    data = data[["datetime", city_name]]

    data = data.dropna() # Drop rows with missing values

    data[city_name] = (data[city_name] - data[city_name].mean()) / data[city_name].std()

    return torch.tensor(data[city_name].values, dtype=torch.float32)