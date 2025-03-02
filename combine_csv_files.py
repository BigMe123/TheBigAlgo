# combine_csv_files.py
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def read_csv_file(file, header_keywords):
    """
    Reads a CSV file and skips the header row if it contains any of the header_keywords.
    """
    with open(file, 'r') as f:
        first_line = f.readline()
    first_line_values = first_line.strip().split(',')
    skiprows = 1 if any(val.strip().lower() in header_keywords for val in first_line_values) else 0
    df = pd.read_csv(file, header=None, skiprows=skiprows)
    return df

def combine_csv_files(root_dir, output_file, columns=None):
    """
    Recursively finds and combines all CSV files under the given root directory
    into one DataFrame. Optionally, assigns the provided column names.
    """
    csv_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(dirpath, file))
    
    if not csv_files:
        raise Exception("No CSV files found in the directory: " + root_dir)
    
    header_keywords = set(['symbol', 'date', 'open', 'high', 'low', 'close', 'volume'])
    df_list = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(read_csv_file, file, header_keywords): file for file in csv_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Reading CSV files"):
            try:
                df_list.append(future.result())
            except Exception as e:
                print(f"Error processing file {futures[future]}: {e}")
    
    combined_df = pd.concat(df_list, ignore_index=True)
    if columns:
        if len(columns) != combined_df.shape[1]:
            raise ValueError(f"Length mismatch: Expected axis has {combined_df.shape[1]} elements, new values have {len(columns)} elements")
        combined_df.columns = columns
    combined_df.to_csv(output_file, index=False)
    print(f"Combined {len(csv_files)} files into {output_file}")
    return combined_df
