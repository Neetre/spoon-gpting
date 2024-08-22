import pandas as pd
from tqdm import tqdm


def read_parquet_ya(filename: str, n_lines: int) -> list:
    """
    Read parquet file and return the text column

    Args:
        filename (str): name of the parquet file
        n_lines (int): number of lines to read

    Returns:
        list: list of documents
    """
    try:
        parquet_data = pd.read_parquet(filename, engine="pyarrow")
    except Exception as e:
        print(f"An error occoured while reading the parquet file {filename}: {e}")

    title = parquet_data["title"]
    texts = parquet_data["text"]
    documents = []
    print(f"Tot rows: {n_lines}")
    c = 0
    for text in tqdm(texts):
        if n_lines != 0:
            if c == n_lines:
                print(c)
                documents.append(text)
                return documents
            documents.append(text)
        else:
            documents.append(text)
        
        if c % 10000:
            print(f"Row {c} done")
    
    return documents


file_paths = ["a.parquet", "b.parquet", "c.parquet", "d.parquet"]
read_parquet_ya()