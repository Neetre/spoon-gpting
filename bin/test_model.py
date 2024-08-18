import os
import json
import requests
import tiktoken
from tqdm import tqdm
import torch
from torch.nn import functional as F
from transformers import GPT2LMHeadModel


DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")

class HellaSwagDataset:
    def __init__(self, split):
        self.split = split
        self.enc = tiktoken.get_encoding("gpt2")
        self.data = []
        self.load_data()
        self.hellaswags = {
            "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
            "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
            "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
        }
    
    def download_file(self, url: str, fname: str, chunk_size=1024):
        """
        Downloads a file from a URL to a local file.

        Args:
            url (str): the URL to download from
            fname (str): the filename to save the file as
            chunk_size (int, optional): the size of the chunks to download. Defaults to 1024.
        """
        resp = requests.get(url, stream=True)
        total = int(resp.headers.get("content-length", 0))
        with open(fname, "wb") as file, tqdm(
            desc=fname,
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in resp.iter_content(chunk_size=chunk_size):
                size = file.write(data)
                bar.update(size)

    def download(self, split):
        """Downloads HellaSwag DATA_CACHE_DIR"""
        os.makedirs(DATA_CACHE_DIR, exist_ok=True)
        data_url = self.hellaswags[split]
        data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
        if not os.path.exists(data_filename):
            print(f"Downloading {data_url} to {data_filename}...")
            self.download_file(data_url, data_filename)


    def render_example(self, example):
        """
        This function takes an example from the HellaSwag dataset and renders it into a format that can be used by the model.

        Args:
            example (dict): A dictionary containing the example data

        Returns:
            data (dict): A dictionary containing the data for the example
            tokens (torch.Tensor): A tensor of token IDs for the example
            mask (torch.Tensor): A tensor of mask values for the example
            label (int): The label for the example
        """

        ctx = example["ctx"]
        label = example["label"]
        endings = example["endings"]

        data = {
            "label": label,
            "ctx_tokens": None,
            "ending_tokens": [],
        }

        ctx_tokens = self.enc.encode(ctx)
        data["ctx_tokens"] = ctx_tokens
        tok_rows = []
        mask_rows = []
        for end in endings:
            end_tokens = self.enc.encode(" " + end)
            tok_rows.append(ctx_tokens + end_tokens)
            mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))
            data["ending_tokens"].append(end_tokens)

        max_len = max(len(row) for row in tok_rows)
        tokens = torch.zeros((4, max_len), dtype=torch.long)
        mask = torch.zeros((4, max_len), dtype=torch.long)
        for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
            tokens[i, :len(tok_row)] = torch.tensor(tok_row)
            mask[i, :len(mask_row)] = torch.tensor(mask_row)

        return data, tokens, mask, label

    def iterate_examples(self, split):
        """
        This function iterates over the examples in the HellaSwag dataset.

        Args:
            split (str): The split of the dataset to iterate over

        Yields:
            example (dict): A dictionary containing the example data
        """
        self.download(split)
        with open(os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") as f:
            for line in f:
                example = json.loads(line)
                yield example

    @torch.no_grad()
    def evaluate(self, model_type, device):
        """
        This function evaluates a model on the HellaSwag dataset.

        Args:
            model_type (str): The model type to use
            device (str): The device to use
        """

        torch.set_float32_matmul_precision('high') # use tf32
        model = GPT2LMHeadModel.from_pretrained(model_type)
        model.to(device)
        # model = torch.compile(model) # optionally torch compile the model

        num_correct_norm = 0
        num_correct = 0
        num_total = 0
        for example in self.iterate_examples(self.split):
            data, tokens, mask, label = self.render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)

            logits = model(tokens).logits
            shift_logits = (logits[..., :-1, :]).contiguous()
            shift_tokens = (tokens[..., 1:]).contiguous()
            flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_shift_tokens = shift_tokens.view(-1)
            shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
            shift_losses = shift_losses.view(tokens.size(0), -1)
            shift_mask = (mask[..., 1:]).contiguous()
            masked_shift_losses = shift_losses * shift_mask
            sum_loss = masked_shift_losses.sum(dim=1)
            avg_loss = sum_loss / shift_mask.sum(dim=1)
            pred = sum_loss.argmin().item()
            pred_norm = avg_loss.argmin().item()

            num_total += 1
            num_correct += int(pred == label)
            num_correct_norm += int(pred_norm == label)
            print(f"{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")

            if num_total < 10:
                print("---")
                print(f"Context:\n {example['ctx']}")
                print(f"Endings:")
                for i, end in enumerate(example["endings"]):
                    print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
                print(f"predicted: {pred_norm}, actual: {label}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", type=str, default="gpt2", help="the model type to use")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="the device to use")
    args = parser.parse_args()
    hella = HellaSwagDataset("val")  # or "train" or "test"
    hella.evaluate(args.model_type, args.device)
