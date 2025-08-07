import pandas as pd
import argparse
import os

def set_uid(edges_file):
    """
    Reads a space-separated edge list from a .txt file,
    formats it into two columns ('uid1', 'uid2'), and saves it as a .csv file.
    """
    # Check if the source file exists
    if not os.path.exists(edges_file):
        print(f"Warning: Source file not found, skipping: {edges_file}")
        return

    print(f"Processing file: {edges_file}")
    
    # Read the text file
    with open(edges_file, 'r') as file:
        lines = file.readlines()

    # Split the lines into two columns
    column1 = []
    column2 = []
    for line in lines:
        values = line.split()
        if len(values) >= 2:
            column1.append(values[0])
            column2.append(values[1])

    # Create a DataFrame
    df = pd.DataFrame({'uid1': column1, 'uid2': column2})

    # Define the output path and save to CSV
    output_file = edges_file.replace(".txt", ".csv")
    df.to_csv(output_file, index=False)
    print(f"Successfully created: {output_file}")


if __name__ == "__main__":
    # --- Define the available datasets and their paths ---
    dataset_paths = {
        "credit": "dataset/credit/credit_edges.txt",
        "german": "dataset/german/german_edges.txt",
        "nba": "dataset/nba/nba_relationship.txt",
        "bail": "dataset/bail/bail_edges.txt",
        "pokec-n": "dataset/pokec/region_job_2_relationship.txt",
        "pokec-z": "dataset/pokec/region_job_relationship.txt"
    }
    
    # --- Set up command-line argument parsing ---
    parser = argparse.ArgumentParser(
        description="Convert space-separated edge list .txt files to .csv format."
    )
    parser.add_argument(
        '--dataset', 
        nargs='*', 
        default=list(dataset_paths.keys()),
        help=f"Name of the dataset(s) to process. "
             f"Choose from: {', '.join(dataset_paths.keys())}. "
             f"If not provided, all datasets will be processed."
    )
    
    args = parser.parse_args()
    datasets_to_process = args.dataset

    # --- Process the selected datasets ---
    for name in datasets_to_process:
        if name in dataset_paths:
            set_uid(dataset_paths[name])
        else:
            print(f"Error: Dataset '{name}' not recognized. Skipping.")
    
    print("\nProcessing complete.")