from dataset_handler import WeatherDataset
from torch.utils.data import DataLoader
from LNN_pipeline import create_model, train_model, evaluate_model
import numpy as np


if __name__ == '__main__':
    np.random.seed(42)
    data = np.random.rand(1000, 11)
    seq_length = 30

    dataset = WeatherDataset(data, seq_length)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = create_model()
    train_model(model, dataloader)
    evaluate_model(model, dataloader)
