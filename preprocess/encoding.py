import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split

# =====================
# 1️⃣ Load and Clean CSV
# =====================
def load_and_clean_csv(file_path, output_file):
    df = pd.read_csv(file_path, header=None)  # Load without headers
    df = df.iloc[1:, 1:]  # Remove first row and first column
    df.reset_index(drop=True, inplace=True)  # Reset index
    df.to_csv(output_file, index=False, header=False)
    print(f"Cleaned CSV saved as {output_file}")
    return df

# =====================
# 2️⃣ One-Hot Encode DNA Sequence
# =====================
def one_hot_encode(df):
    mapping = {'A': [1, 0, 0, 0, 0], 'B': [0, 1, 0, 0, 0], 
               'C': [0, 0, 1, 0, 0], 'D': [0, 0, 0, 1, 0], 'N': [0, 0, 0, 0, 1]}  # Include 'N'

    encoded_data = []
    for _, row in df.iterrows():
        encoded_row = []
        for value in row:
            encoded_row.extend(mapping.get(value, [0, 0, 0, 0, 0]))  # Default to 00000 if unknown
        encoded_data.append(encoded_row)
    
    return pd.DataFrame(encoded_data)

# =====================
# 3️⃣ Quantum Encode DNA Sequence (Phase Encoding)
# =====================
def quantum_encode(df):
    phase_mapping = {'A': 0, 'B': 90, 'C': 180, 'D': 270, 'N': 45}  # Include 'N' as 45°

    encoded_data = []
    for _, row in df.iterrows():
        encoded_row = [phase_mapping.get(value, 0) for value in row]  # Default to 0° if unknown
        encoded_data.append(encoded_row)
    
    return pd.DataFrame(encoded_data)

# =====================
# 4️⃣ Save Encoded Data
# =====================
def save_encoded_data(encoded_df, output_file):
    encoded_df.to_csv(output_file, index=False, header=False)
    print(f"Encoded CSV saved as {output_file}")

# =====================
# 5️⃣ Run the Encoding Process
# =====================
input_csv = "./C7AIR_Genotype.csv"  # Change to your filename if needed
cleaned_csv = "cleaned_genotype.csv"
df = load_and_clean_csv(input_csv, cleaned_csv)

# One-hot encoding
one_hot_df = one_hot_encode(df)
save_encoded_data(one_hot_df, "one_hot_encoded.csv")

# Quantum encoding
quantum_df = quantum_encode(df)
save_encoded_data(quantum_df, "quantum_encoded.csv")
