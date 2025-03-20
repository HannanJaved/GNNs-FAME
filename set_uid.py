import pandas as pd

def set_uid(edges_file):
    # Read the text file
    with open(edges_file, 'r') as file:
        lines = file.readlines()

    # Split the lines into two columns
    column1 = []
    column2 = []
    for line in lines:
        values = line.split()
        column1.append(values[0])
        column2.append(values[1])

    # Create dataframes for each column
    df = pd.DataFrame({'uid1': column1, 'uid2': column2})

    edges_file = edges_file.replace(".txt", ".csv")
    df.to_csv(edges_file, index=False)

if __name__ == "__main__":
    set_uid("dataset/credit/credit_edges.txt")
    set_uid("dataset/german/german_edges.txt")