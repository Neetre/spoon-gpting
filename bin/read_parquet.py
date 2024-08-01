import pandas as pd
from tqdm import tqdm

def read_parquet_ya(file_paths: list, len_limit = 10000):
    for path in tqdm(file_paths, desc="Reading parquet files"):
        try:
            parquet_data = pd.read_parquet(path, engine="pyarrow")
        except Exception as e:
            print(e)

        titles = parquet_data["title"]
        texts = parquet_data["text"]
        data = []
        c = 0
        for title, text in zip(titles, texts):
            if c == len_limit:
                return '\n'.join(data)
            else:
                data.append(title)
                data.append(text)
                c += 1


file_paths = ["a.parquet", "b.parquet", "c.parquet", "d.parquet"]
read_parquet_ya()