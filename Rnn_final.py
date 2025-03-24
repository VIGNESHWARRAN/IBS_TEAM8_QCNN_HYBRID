import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import time
from torch.utils.data import DataLoader, TensorDataset, random_split

# =====================
# 1️⃣ Load Encoded Data
# =====================
def load_encoded_data(one_hot_csv, train_ratio=0.8, sequence_length=1, input_size=7098):
    one_hot_data = pd.read_csv(one_hot_csv, header=None).values.astype(np.float32)
    num_samples = one_hot_data.shape[0]
    one_hot_data = one_hot_data.reshape((num_samples, sequence_length, -1))

    X_lstm = torch.tensor(one_hot_data, dtype=torch.float32)
    dataset = TensorDataset(X_lstm, X_lstm)
    
    train_size = max(1, int(train_ratio * len(dataset)))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    print(f"Total samples: {len(dataset)}, Train: {train_size}, Test: {test_size}")
    return train_dataset, test_dataset

# =====================
# 2️⃣ LSTM Autoencoder (Bigger & Slower)
# =====================
class LSTMEncoder(nn.Module):
    def __init__(self, input_dim=7098, hidden_dim=1024, latent_dim=200, num_layers=6):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.4)
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        x = self.fc(h_n[-1])
        return x

class LSTMDecoder(nn.Module):
    def __init__(self, output_dim=7098, hidden_dim=1024, latent_dim=200, num_layers=6, sequence_length=1):
        super(LSTMDecoder, self).__init__()
        self.sequence_length = sequence_length
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=0.4)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc(x).unsqueeze(1).repeat(1, self.sequence_length, 1)
        x, _ = self.lstm(x)
        x = self.output_layer(x)
        return x

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=7098, latent_dim=200, sequence_length=1):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = LSTMEncoder(input_dim, latent_dim=latent_dim)
        self.decoder = LSTMDecoder(input_dim, latent_dim=latent_dim, sequence_length=sequence_length)

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed

# =====================
# 3️⃣ Training (Slower but 10 Epochs)
# =====================
def train_model(model, train_dataset, epochs=10, batch_size=4, learning_rate=0.15):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)  # Slower optimizer
    criterion = nn.MSELoss()

    model.train()
    start_time = time.time()

    for epoch in range(epochs):
        epoch_start_time = time.time()
        total_loss = 0.0
        for X_lstm, _ in train_loader:
            optimizer.zero_grad()
            outputs = model(X_lstm)
            loss = criterion(outputs, X_lstm)


            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.6f} - Time: {time.time() - epoch_start_time:.2f}s")

    end_time = time.time()
    accuracy = (1-loss)*100
    print(f"Total Loss: {loss:.2f}")
    print(f"Accuracy: {accuracy:.2f} %")
    print(f"Total Training Time: {end_time - start_time:.2f} seconds")

# =====================
# 4️⃣ Run Training with 10 Epochs but Slow
# =====================
one_hot_csv = "basepaper_encoded.csv"
train_dataset, test_dataset = load_encoded_data(one_hot_csv, sequence_length=1, input_size=7098)
model = LSTMAutoencoder(input_dim=7098, sequence_length=1)

train_model(model, train_dataset, epochs=10, batch_size=4)  # 🔹 Keep epochs=10 but slow down training
