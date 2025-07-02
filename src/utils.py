import pandas as pd

def ingest_csv_to_dataframe(filepath):
    """
    Ingests data from a CSV file into a pandas DataFrame.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        pandas.DataFrame or None: A DataFrame containing the CSV data if successful,
                                 otherwise None if an error occurs.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully ingested data from: {filepath}")
        return df
    except FileNotFoundError:
        print(f"Error: The file at '{filepath}' was not found.")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: The file at '{filepath}' is empty.")
        return None
    except pd.errors.ParserError:
        print(f"Error: Could not parse the CSV file at '{filepath}'. Check its format.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while reading '{filepath}': {e}")
        return None