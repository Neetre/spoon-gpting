'''

https://github.com/openai/tiktoken
https://en.wikipedia.org/wiki/Byte_pair_encoding

Neetre 2024
'''

import argparse
import regex as re
import json
import datasets


def get_stats(ids: list, counts=None):
    """
    Get the frequency of each pair of ids in a list of ids.

    Args:
        ids (list): List of ids.
        counts (dict, optional): Dictionary of counts. Defaults to None.

    Returns:
        dict: Dictionary of counts.
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: list, pair, idx: int):
    """
    Merge a pair of ids in a list of ids.

    Args:
        ids (list): List of ids.
        pair (): Pair of ids to merge.
        idx (int): Index to replace the pair with.

    Returns:
        list: New list of ids.
    """

    newids = []  # new list of ids
    i = 0
    while i < len(ids):
        #  if not at the very last position AND the pair matches, replace it
        if i < len(ids) -1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


class BaseTokenizer:

    def __init__(self) -> None:
        self.merges = {}
        self.pattern = ""
        self.special_tokens = {}
        self.vocab = self._build_vocab()

    def train(self, text, vocab_zise):
        pass

    def encode(self, text):
        pass

    def decode(self, ids):
        pass

    def _build_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")

        return vocab


GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class BytePairTokenizer(BaseTokenizer):
    def __init__(self) -> None:
        super().__init__()
        self.pattern = GPT2_SPLIT_PATTERN
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def train(self, text: str, vocab_size: int):
        assert vocab_size >= 256, "Vocab size must be 256"
        num_merges = vocab_size - 256

        text_chunks = re.findall(self.compiled_pattern, text)

        ids = [list(tx.encode("utf-8")) for tx in text_chunks]

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}

        for i in range(num_merges):
            stats = {}

            for chunk_ids in ids:
                get_stats(chunk_ids, stats)

            top_pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = [merge(chunk_ids, top_pair, idx) for chunk_ids in ids]
            merges[top_pair] = idx
            vocab[idx] = vocab[top_pair[0]] + vocab[top_pair[1]]
            print(f"Merge {i+1}/{num_merges}: {top_pair} --> {idx}  | {vocab[idx]} had {stats[top_pair]} occurencies!!")

        self.merges = merges
        self.vocab = vocab

    def register_special_tokens(self, special_tokens: dict):
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids: list):
        part_bytes = []

        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"Invalid token {idx}")

        tokens = b"".join(part_bytes)
        text = tokens.decode("utf-8", errors="replace")
        return text

    def _encode_chunk(self, text_bytes: bytes):
        ids = list(text_bytes)

        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text: str):
        text_chunks = re.findall(self.compiled_pattern, text)

        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)

        return ids

    def encode(self, text: str, allowed_special="none_raise"):
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"Invalid value for allowed_special: {allowed_special}")

        if not special:
            return self.encode_ordinary(text)

        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)

        ids = []

        for part in special_chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))

        return ids

    def save_merges(self):
        with open("../data/merges.json", "w") as file:
            db ={}
            mer = {str(k): v for k, v in self.merges.items()}
            vocab = {str(k): v.decode("utf-8", errors="replace") for k, v in self.vocab.items()}
            db["merges"] = mer
            db["vocab"] = vocab
            json.dump(db, file, indent=4)

    def load_merges(self):
        with open("../data/merges.json", "r") as file:
            db = json.load(file)
            merges = {eval(k): v for k, v in db["merges"].items()}
            vocab = {eval(k): v.encode("utf-8") for k, v in db["vocab"].items()}
            self.merges = merges
            self.vocab = vocab

    def view_tokenized_text(self, ids: list):
        for idx in ids:
            print(f"{self.vocab[idx].decode('utf-8', errors='replace')}: {self.vocab[idx]}")


def get_data(file_path="../data/input.txt"):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text
 

def main():
    tokenizer = BytePairTokenizer()
    # text = get_data()

    text = get_data()
    tokenizer.train(text, 406)
    tokenizer.save_merges()

    print("Merges: ", tokenizer.merges)
    print("Vocab: ", tokenizer.vocab)

    special_tokens = {
        '<|endoftext|>': 100257,
        '<|fim_prefix|>' : 100258,
        '<|fim_middle|>' : 100259,
        '<|fim_suffix|>' : 100260,
        '<|endofprompt|>' : 100276
    }  # gpt2 special tokens

    tokenizer.register_special_tokens(special_tokens)