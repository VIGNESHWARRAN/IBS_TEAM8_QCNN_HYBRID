import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pennylane as qml

# --- Quantum Setup ---
n_qubits = 4  # Change this to 8 or 16 if your hardware supports it
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QuantumLayer(nn.Module):
    def __init__(self, depth=6):
        super().__init__()
        self.weight_shapes = {"weights": (depth, n_qubits)}
        self.q_layer = qml.qnn.TorchLayer(quantum_circuit, self.weight_shapes)

    def forward(self, x):
        return self.q_layer(x)

# --- Encoder ---
class GenCoderEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),                 # → [B, 32, ~3549]
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)                  # → [B, 64, ~1774]
        )
        self.flatten = nn.Flatten()
        self.linear_to_quantum = nn.Linear(64 * 1774, n_qubits)
        
        # Stack multiple QuantumLayers
        self.q_layers = nn.Sequential(
            QuantumLayer(depth=6),
            QuantumLayer(depth=6),
            QuantumLayer(depth=6)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.linear_to_quantum(x)
        x = self.q_layers(x)
        return x

# --- Decoder ---
class GenCoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_from_quantum = nn.Linear(n_qubits, 64 * 1774)
        self.unflatten = nn.Unflatten(1, (64, 1774))
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.linear_from_quantum(x)
        x = self.unflatten(x)
        return self.decoder(x)

# --- Autoencoder ---
class GenCoderAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = GenCoderEncoder()
        self.decoder = GenCoderDecoder()

    def forward(self, x):
        x = self.encoder(x)
        print(x)
        print("\n\n\n\n\n\n\n\n\n\n\n\n one epoch done")
        return self.decoder(x)

# --- Training Routine ---
def main():
    # Load and preprocess data
    data = pd.read_csv("basepaper_encoded.csv", header=None)
    original_len = data.shape[1]

    # Convert to tensor shape (batch_size, 1, sequence_len)
    tensor_data = torch.tensor(data.values, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(tensor_data)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Initialize model, loss, optimizer
    model = GenCoderAutoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in range(100):
        total_loss = 0.0
        for X, in loader:
            optimizer.zero_grad()
            output = model(X)

            # Resize output to match input shape
            if output.shape[-1] != original_len:
                if output.shape[-1] > original_len:
                    output = output[:, :, :original_len]
                else:
                    pad = original_len - output.shape[-1]
                    output = nn.functional.pad(output, (0, pad))

            loss = criterion(output, X)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/100 - Average Loss: {avg_loss:.6f}", flush=True)

    # Evaluation
    model.eval()
    with torch.no_grad():
        decoded = model(tensor_data)
        if decoded.shape[-1] != original_len:
            if decoded.shape[-1] > original_len:
                decoded = decoded[:, :, :original_len]
            else:
                pad = original_len - decoded.shape[-1]
                decoded = nn.functional.pad(decoded, (0, pad))
        decoded = decoded.squeeze().numpy()

    # Round decoded values to nearest level
    def round_to_nearest(values, levels=[0.2, 0.4, 0.6, 0.8, 1.0]):
        levels = np.array(levels)
        shape = values.shape
        values_flat = values.flatten()
        idx = np.argmin(np.abs(values_flat[:, None] - levels), axis=1)
        rounded = levels[idx]
        return rounded.reshape(shape)

    rounded_decoded = round_to_nearest(decoded)

    # Save outputs
    pd.DataFrame(rounded_decoded).to_csv("decoded.csv", index=False, header=False)
    print("✅ Decoded output saved to decoded.csv")
    torch.save(model.state_dict(), "autoencoder.pth")
    print("✅ Model saved to autoencoder.pth")

if __name__ == "__main__":
    main()
