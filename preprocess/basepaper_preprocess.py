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

    encoding_map = {'A': 1, 'T': 2, 'G': 3, 'C': 4, 'N': 5}  # Proper integer encoding
    encoded_data = df.applymap(lambda x: encoding_map.get(x, 0))  # Default to 0 if unknown
    encoded_data = encoded_data.astype(np.float32) / 5.0  # Normalize values to [0,1]
    return encoded_data

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
cleaned_csv = "basepaper_cleaned_genotype.csv"
df = load_and_clean_csv(input_csv, cleaned_csv)

# One-hot encoding
one_hot_df = one_hot_encode(df)
save_encoded_data(one_hot_df, "basepaper_encoded.csv")

