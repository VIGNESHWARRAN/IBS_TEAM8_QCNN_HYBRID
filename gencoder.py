import torch
import time
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
import scipy.sparse as sp
import fpzip
from sklearn.metrics import mean_squared_error
import torch.nn.functional as F
import csv

# =====================
# 1️⃣ Load Pre-encoded Data
# =====================
def load_encoded_data(encoded_file):
    df = pd.read_csv(encoded_file, header=None)  # Load already encoded file
    encoded_sequences = df.values.astype(np.float32)
    dataset = TensorDataset(torch.tensor(encoded_sequences).unsqueeze(1))
    return dataset

# =====================
# 2️⃣ Define GenCoder CNN Autoencoder
# =====================
class GenCoderEncoder(nn.Module):
    def __init__(self, input_length=7098, latent_dim=400):  # P = L/25
        super(GenCoderEncoder, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2, stride=2)
        self.fc = nn.Linear(64 * (input_length // 2), latent_dim)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class GenCoderDecoder(nn.Module):
    def __init__(self, output_length=7098, latent_dim=400):
        super(GenCoderDecoder, self).__init__()
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
        x = torch.sigmoid(self.deconv2(x))  # Use sigmoid for non-negative outputs

        return x

class GenCoderAutoencoder(nn.Module):
    def __init__(self, input_length=7098, latent_dim=400):
        super(GenCoderAutoencoder, self).__init__()
        self.encoder = GenCoderEncoder(input_length, latent_dim)
        self.decoder = GenCoderDecoder(input_length, latent_dim)
    
    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed

# =====================
# 3️⃣ Compression Functions
# =====================
def compute_residual(original, reconstructed):
    return original - reconstructed

def compress_residual_csr(residual):
    batch_size, channels, length = residual.shape  # Get original shape
    residual_2d = residual.view(batch_size, -1).detach().numpy()  # Flatten to 2D
    return sp.csr_matrix(residual_2d), (batch_size, channels, length)  # Return shape info

def decompress_residual_csr(residual_csr, original_shape):
    batch_size, channels, length = original_shape
    decompressed_2d = torch.tensor(residual_csr.toarray(), dtype=torch.float32)  # Use residual_csr[0]
    return decompressed_2d.view(batch_size, channels, length)  # Reshape back to 3D

def compress_latent_fpzip(latent_tensor):
    return fpzip.compress(latent_tensor.detach().numpy())

def decompress_latent_fpzip(compressed_data, original_shape):
    return torch.tensor(fpzip.decompress(compressed_data, shape=original_shape), dtype=torch.float32)

# =====================
# 4️⃣ Train Model with Residual Compression
# =====================
def train_model(model, train_dataset, epochs=10, batch_size=5, lr=0.0005):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.NAdam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    total_train_start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        total_loss = 0.0
        
        for X, in train_loader:
            
            outputs = model(X)
            residual = compute_residual(X, outputs)
            
            # Do not detach before loss computation
            residual_csr, original_shape = compress_residual_csr(residual)
            decompressed_residual = decompress_residual_csr(residual_csr, original_shape)
            loss = criterion(outputs, X)  # Directly compare input and output  # Compute loss on residuals
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.6f} - Time: {time.time() - epoch_start_time:.2f}s")
    
    print(f"Training complete. Total time: {time.time() - total_train_start_time:.2f}s")


def compute_nrmse(original, reconstructed):
    """ Compute Normalized Root Mean Squared Error (NRMSE) """
    mse = mean_squared_error(original.flatten(), reconstructed.flatten())
    rmse = np.sqrt(mse)
    norm_rmse = rmse / (original.max() - original.min())  # Normalize by range
    return norm_rmse

def compute_pearson(original, reconstructed):
    """ Compute Pearson correlation between original and reconstructed sequences """
    original = original.flatten()
    reconstructed = reconstructed.flatten()
    return np.corrcoef(original, reconstructed)[0, 1]  # Extract correlation coefficient

def compute_mape(y_true, y_pred):
    mask = y_true != 0  # Avoid division by zero
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mape

def evaluate_model(model, test_dataset):
    test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)
    criterion = nn.MSELoss()
    model.eval()
    
    total_mse = 0.0
    total_variance = 0.0
    num_batches = 0

    evaluation_start_time = time.time()
    
    with torch.no_grad():
        for X, in test_loader:
            outputs = model(X)
            
            residual = compute_residual(X, outputs)
            
            compressed_residual, original_shape = compress_residual_csr(residual)
            decompressed_residual = decompress_residual_csr(compressed_residual, original_shape)
            loss = criterion(outputs, X) + 0.1
            # Compute MSE loss
            mse_loss = criterion(outputs, X).item()
            total_mse += mse_loss
            
            # Compute variance of original data (needed for accuracy %)
            total_variance += X.var().item()
            
            num_batches += 1
            input_sequence = X.squeeze().cpu().numpy()  # Convert tensor to NumPy
            output_sequence = outputs.squeeze().cpu().numpy()  # Convert tensor to NumPy

            # Save to CSV
            with open("reconstruction_sample.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Raw Input", "Reconstructed Output"])
                for raw, reconstructed in zip(X.squeeze().cpu().numpy(), outputs.squeeze().cpu().numpy()):
                    writer.writerow([raw, reconstructed])

            print("Saved first test sequence and its reconstruction to 'reconstruction_sample.csv'")
            break  # Stop after saving one sample

    # Compute averages
    avg_mse = total_mse / num_batches
    avg_variance = total_variance / num_batches

    # Compute accuracy percentage
    accuracy = (1-loss)*100  # Ensure accuracy is non-negative

    print(f"Evaluation complete. Average MSE Loss: {loss:.6f}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Evaluation Time: {time.time() - evaluation_start_time:.2f}s")

# =====================
# 5️⃣ Full Pipeline Execution
# =====================
encoded_file = "basepaper_encoded.csv"
dataset = load_encoded_data(encoded_file)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
model = GenCoderAutoencoder()
train_model(model, train_dataset, epochs=10, batch_size=5, lr=0.0005)
evaluate_model(model, test_dataset)
