from LNN import LiquidNeuralNetwork
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def create_model(input_size, hidden_size, output_size):
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
    all_preds = []
    all_true = []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            y_pred = model(x_batch)
            loss = criterion(y_pred.squeeze(), y_batch)
            test_loss += loss.item()

            # Store predictions and true values
            all_preds.extend(y_pred.squeeze().tolist())
            all_true.extend(y_batch.tolist())

    avg_loss = test_loss / len(dataloader)
    print(f"Test Loss: {avg_loss}")

    plt.figure(figsize=(20, 15))
    plt.plot(all_true, label="True Values", alpha=0.7)
    plt.plot(all_preds, label="Predictions", alpha=0.7)
    plt.legend(fontsize=20)
    plt.title("Liquid NN model's predictions vs true values", fontsize=24)
    plt.xlabel("Sample Index", fontsize=20)
    plt.ylabel("Normalized Value", fontsize=20)
    plt.grid(True)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()

    return avg_loss