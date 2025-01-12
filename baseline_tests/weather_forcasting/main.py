from dataset_handler import WeatherDataset
from torch.utils.data import DataLoader
from datasets.historical_hourly_weather_data.read_process_data import load_and_process_dataset
from LNN_pipeline import create_model, train_model, evaluate_model


if __name__ == '__main__':
    file_path = "datasets/historical_hourly_weather_data/temperature.csv"
    city_name = "Toronto"
    seq_length = 30
    batch_size = 32
    num_epochs = 5

    data = load_and_process_dataset(file_path, city_name=city_name)

    # Chronologically split the data
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]

    train_dataset = WeatherDataset(train_data, seq_length)
    test_dataset = WeatherDataset(test_data, seq_length)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  # No shuffling for time-series
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = create_model(input_size=1, hidden_size=20, output_size=1)
    train_model(model, train_dataloader, num_epochs=num_epochs)
    evaluate_model(model, test_dataloader)