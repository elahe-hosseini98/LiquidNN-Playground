from dataset_handler import WeatherDataset
from LNN import LiquidNeuralNetwork
import torch
from torch.utils.data import DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim


if __name__ == '__main__':
    np.random.seed(42)
    data = np.random.rand(1000, 11)
    seq_length = 30

    dataset = WeatherDataset(data, seq_length)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    input_size = 10
    hidden_size = 20
    output_size = 1

    model = LiquidNeuralNetwork(input_size, hidden_size, output_size)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}")

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            y_pred = model(x_batch)
            loss = criterion(y_pred.squeeze(), y_batch)
            test_loss += loss.item()

    print(f"Test Loss: {test_loss / len(dataloader)}")


