import numpy as np
import csv

def round_to_nearest(value, choices):
    return min(choices, key=lambda x: abs(x - value))

def process_csv(input_file, output_file):
    choices = [0.2, 0.4, 0.6, 0.8, 1.0]
    
    with open(input_file, 'r') as infile:
        reader = csv.reader(infile)
        data = [[round_to_nearest(float(value), choices) for value in row] for row in reader]
    
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(data)

if __name__ == "__main__":
    input_csv = "testsequences.csv"   # Change this to your input CSV file name
    output_csv = "gencoder_model_processed_inputs.csv"  # Change this to your desired output file name
    process_csv(input_csv, output_csv)
    print(f"Processed CSV saved as {output_csv}")
