from transformer_baseline_model import TimeSeriesTransformer
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torch


def create_baseline_model(input_size, d_model, n_heads, num_layers, output_size, dropout=0.1):
    model = TimeSeriesTransformer(
        input_size=input_size,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        output_size=output_size,
        dropout=dropout,
    )
    return model


def train_baseline_model(model, dataloader, learning_rate=0.001, num_epochs=20):
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


def evaluate_baseline_model(model, dataloader):
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

            all_preds.extend(y_pred.squeeze().tolist())
            all_true.extend(y_batch.tolist())

    avg_loss = test_loss / len(dataloader)
    print(f"Test Loss: {avg_loss}")

    plt.figure(figsize=(12, 6))
    plt.plot(all_true, label="True Values", alpha=0.7)
    plt.plot(all_preds, label="Predictions", alpha=0.7)
    plt.legend()
    plt.title("Transformer-based model's predictions vs true values")
    plt.xlabel("Sample Index")
    plt.ylabel("Normalized Value")
    plt.grid(True)
    plt.show()

    return avg_loss