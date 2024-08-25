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
from datasets import load_dataset
from tqdm import tqdm

class Fineweb:
    def __init__(self):
        self.local_dir = "../data/edu_fineweb10B"
        self.remote_name = "sample_10BT"
        self.shard_size = int(1e8) # 100M tokens per shardm total of 100 shards

        # create the cache the kicak durectiry if it doesn't exist yet
        self.DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), self.local_dir)
        os.makedirs(self.DATA_CACHE_DIR, exist_ok=True)

        # download the dataset
        self.fw = load_dataset("HuggingfaceFW/fineweb-edu", name=self.remote_name, split="train")

        # init the tokenizer
        self.enc = tiktoken.TikTokenizer("gpt2")
        self.eot = self.enc._special_tokens['<|endoftext|>']
        
        self.nprocs = max(1, mp.cpu_count()//2)

    def tokenize(self, doc):
        tokens = [self.eot]
        tokens.extend(self.enc.encode_ordinary(doc["text"]))
        tokens_np = np.array(tokens)
        assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large"
        tokens_np_uint16 = tokens_np.astype(np.uint16)
        return tokens_np_uint16

    def write_datafile(self, filename, tokens_np):
        with open(filename, "wb") as f:
            f.write(tokens_np.tobytes())

    def generate_shards(self):
        with mp.Pool(self.nprocs) as pool:
            shard_index = 0
            all_tokens_np = np.empty((self.shard_size,), dtype=np.uint16)
            token_count = 0
            progress_bar = None
            for tokens in pool.imap(self.tokenize, self.fw, chunksize=16):
                if token_count + len(tokens) < self.shard_size:
                    all_tokens_np[token_count:token_count+len(tokens)] = tokens
                    token_count += len(tokens)
                    if progress_bar is None:
                        progress_bar = tqdm(total=self.shard_size, unit="tokens", desc=f"Shard {shard_index}")
                    progress_bar.update(len(tokens))
                else:
                    split = "val" if shard_index == 0 else "train"
                    filename = os.path.join(self.DATA_CACHE_DIR, f"edu_fineweb_{split}_{shard_index:06d}")
                    remainder = self.shard_size - token_count
                    progress_bar.update(remainder)
                    all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                    self.write_datafile(filename, all_tokens_np)
                    shard_index += 1
                    progress_bar = None
                    all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                    token_count = len(tokens) - remainder

            if token_count != 0:
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(self.DATA_CACHE_DIR, f"edu_fineweb_{split}_{shard_index:06d}")
                self.write_datafile(filename, all_tokens_np[:token_count])
