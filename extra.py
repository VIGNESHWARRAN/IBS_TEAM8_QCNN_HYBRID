import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import scipy.sparse as sp
import fpzip
from sklearn.metrics import mean_squared_error
import pandas as pd

# =====================
# 1️⃣ Load Encoded Data
# =====================
def load_encoded_data(one_hot_csv, quantum_csv, train_ratio=0.8):
    one_hot_data = pd.read_csv(one_hot_csv, header=None).values.astype(np.float32)
    quantum_data = pd.read_csv(quantum_csv, header=None).values.astype(np.float32)

    X_cnn = torch.tensor(one_hot_data, dtype=torch.float32).unsqueeze(1)  # CNN input
    X_qcnn = torch.tensor(quantum_data, dtype=torch.float32)  # QCNN input

    dataset = TensorDataset(X_cnn, X_qcnn)
    train_size = max(1, int(train_ratio * len(dataset)))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    print(f"Total samples: {len(dataset)}, Train: {train_size}, Test: {test_size}")
    return train_dataset, test_dataset

# =====================
# 2️⃣ Quantum Encoder (Fixed)
# =====================
n_qubits = 4  # Set number of qubits

# Define quantum circuit
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_layer(weights, inputs):
    """Parameterized Quantum Circuit"""
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]  # Expectation values

class QuantumEncoder(nn.Module):
    def __init__(self, n_qubits, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.weights = nn.Parameter(0.01 * torch.randn(n_layers, n_qubits, requires_grad=True))  # Small initialization
        
    def forward(self, x):
        """Apply quantum transformation on latent features"""
        x = x.view(-1, self.n_qubits)  # Ensure correct shape
        q_outputs = torch.stack([torch.tensor(quantum_layer(self.weights, x[i])) for i in range(x.shape[0])])
        return q_outputs  # Shape (batch_size, n_qubits)

# =====================
# 3️⃣ Hybrid GenCoder Autoencoder (Fixed)
# =====================
class GenCoderEncoder(nn.Module):
    def __init__(self, input_length=7098, latent_dim=20, quantum_dim=4):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2, stride=2)
        self.fc = nn.Linear(64 * (input_length // 2), latent_dim)

        self.quantum_encoder = QuantumEncoder(n_qubits=quantum_dim)
        self.q_fc = nn.Linear(quantum_dim, latent_dim // 2)  # Project quantum output to latent_dim/2

    def forward(self, x_cnn, x_qcnn):
        """Hybrid Encoding (CNN + QCNN)"""
        x_cnn = torch.relu(self.conv1(x_cnn))
        x_cnn = self.pool(torch.relu(self.conv2(x_cnn)))
        x_cnn = x_cnn.view(x_cnn.size(0), -1)
        x_cnn = self.fc(x_cnn)

        q_encoded = self.quantum_encoder(x_qcnn)
        q_encoded = self.q_fc(q_encoded)

        # Combine CNN and QCNN features
        latent = torch.cat((x_cnn[:, 40 // 2], q_encoded), dim=1)
        return latent

class GenCoderDecoder(nn.Module):
    def __init__(self, output_length=7098, latent_dim=20):
        super().__init__()
        self.output_length = output_length
        self.fc = nn.Linear(latent_dim, (output_length // 2) * 64)
        self.deconv1 = nn.ConvTranspose1d(64, 32, kernel_size=5, padding=2)
        self.upsample = nn.Upsample(size=output_length, mode='nearest')
        self.deconv2 = nn.ConvTranspose1d(32, 1, kernel_size=5, padding=2)
    
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 64, self.output_length // 2)
        x = torch.relu(self.deconv1(x))
        x = self.upsample(x)
        x = torch.sigmoid(self.deconv2(x))
        return x

class HybridGenCoderAutoencoder(nn.Module):
    def __init__(self, input_length=7098, latent_dim=20, quantum_dim=4):
        super().__init__()
        self.encoder = GenCoderEncoder(input_length, latent_dim, quantum_dim)
        self.decoder = GenCoderDecoder(input_length, latent_dim)
    
    def forward(self, x_cnn, x_qcnn):
        encoded = self.encoder(x_cnn, x_qcnn)
        reconstructed = self.decoder(encoded)
        return reconstructed

# =====================
# 6️⃣ Train and Evaluate the Model (Fixed)
# =====================
import time

def train_model(model, train_dataset, epochs=10, batch_size=32, learning_rate=0.001):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    start_time = time.time()
    model.train()
    for epoch in range(epochs):
        epoch_start_time = time.time()
        total_loss = 0.0
        for X_cnn, X_qcnn in train_loader:
            optimizer.zero_grad()
            outputs = model(X_cnn, X_qcnn)
            loss = criterion(outputs, X_cnn)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.6f} - Time: {time.time() - epoch_start_time:.2f}s")
    end_time = time.time()  # End time tracking
    elapsed_time = end_time - start_time
    print(f"Total Training Time: {elapsed_time:.2f} seconds")

def evaluate_model(model, test_dataset):
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    model.eval()
    total_loss = 0.0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for X_cnn, X_qcnn in test_loader:
            outputs = model(X_cnn, X_qcnn)
            loss = criterion(outputs, X_cnn)
            total_loss += loss.item()
    accuracy = (100-total_loss)*100
    print(f"Test Loss: {total_loss:.6f}")
    print(f"Accuracy: {accuracy:.6f}")
    print(f"Train time: {accuracy:.6f}")


# =====================
# 7️⃣ Run Training
# =====================
one_hot_csv = "basepaper_encoded.csv"
quantum_csv = "quantum_encoded.csv"

train_dataset, test_dataset = load_encoded_data(one_hot_csv, quantum_csv)
model = HybridGenCoderAutoencoder()
train_model(model, train_dataset, epochs=10)
evaluate_model(model, test_dataset)