from LNN import LiquidNeuralNetwork
import torch
import torch.nn as nn
import torch.optim as optim


def create_model(input_size=10, hidden_size=20, output_size=1):
    model = LiquidNeuralNetwork(input_size, hidden_size, output_size)
    return model


def train_model(model, dataloader, learning_rate=0.001, num_epochs=20):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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

    return model


def evaluate_model(model, dataloader):
    criterion = nn.MSELoss()
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            y_pred = model(x_batch)
            loss = criterion(y_pred.squeeze(), y_batch)
            test_loss += loss.item()

    avg_loss = test_loss / len(dataloader)
    print(f"Test Loss: {avg_loss}")
    return avg_loss