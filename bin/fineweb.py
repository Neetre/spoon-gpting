"""
FineWeb-Edu dataset
Run simply as:
$ python3 fineweb.py
Will save shards to the local directory "data/edu_fineweb10B".
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset  # pip install datasets
from tqdm import tqdm  # pip install tqdm


# ----------------------------------------------
local_dir = "../data/edu_fineweb10B"
remote_name = "sample_10BT"
shard_size = int(1e8) # 100M tokens per shardm total of 100 shards

# create the cache the kicak durectiry if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# download the dataset
fw = load_dataset("HuggingfaceFW/fineweb-edu", name=remote_name, split="train")

# init the tokenizer
enc = tiktoken.TikTokenizer("gpt2")
eot = enc._special_tokens['<|endoftext|>']  # end of text token
def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    # writes a numpy aray of uint16 tokens to a binary file
    with open(filename, "wb") as f:
        f.write(tokens_np.tobytes())