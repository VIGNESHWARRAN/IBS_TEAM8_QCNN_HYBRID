import csv
from difflib import SequenceMatcher

# Function to read DNA sequences as lists from CSV
def read_dna_matrix(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        sequences = ["".join(row) for row in reader]  # Merge nucleotides into a single string
    return sequences

# Function to compute similarity percentage
def compute_similarity(seq1, seq2):
    return SequenceMatcher(None, seq1, seq2).ratio() * 100  # Convert to percentage

# Function to compare two CSV files row by row
def compare_dna_csv(file1, file2, output_file="similarity_results.csv"):
    sequences1 = read_dna_matrix(file1)
    sequences2 = read_dna_matrix(file2)

    if len(sequences1) != len(sequences2):
        print("Warning: Files contain a different number of sequences!")

    similarities = []
    total_similarity = 0
    for i, (s1, s2) in enumerate(zip(sequences1, sequences2), start=1):
        similarity = compute_similarity(s1, s2)
        total_similarity += similarity
        similarities.append([s1, s2, f"{similarity:.2f}%"])
        print(f"Row {i}: Similarity = {similarity:.2f}%")  # Print similarity for each row

    # Compute and print average similarity
    avg_similarity = total_similarity / len(similarities) if similarities else 0
    print(f"\nAverage Similarity: {avg_similarity:.2f}%")

    # Save results
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Sequence 1", "Sequence 2", "Similarity"])
        writer.writerows(similarities)

    print(f"Similarity results saved in {output_file}")

# Example usage
#compare_dna_csv("gencoder_model_processed_outputs.csv", "gencoder_model_processed_inputs.csv")
compare_dna_csv("gencoder_model_processed_inputs.csv", "gencoder_model_processed_inputs.csv")
