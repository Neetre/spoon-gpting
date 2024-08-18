import os
import math
import time
import torch
import torch.nn.functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from model import DataLoaderLite, GPT, GPTConfig
import tiktoken


class DistributedGPTTrainer:
    def __init__(self, compile_flag=False):
        self.compile_flag = compile_flag
        self.device = self._setup_device()
        self.ddp, self.ddp_rank, self.ddp_local_rank, self.ddp_world_size, self.master_process = self._setup_ddp()
        self.total_batch_size = 524288  # 2**19, 0.5M tokens in total
        self.B = 4
        self.T = 1024
        self.grad_accum_steps = self._calculate_grad_accum_steps()
        self.train_loader = DataLoaderLite(B=self.B, T=self.T, process_rank=self.ddp_rank, num_processes=self.ddp_world_size)
        self.model = self._setup_model()
        self.optimizer = self._setup_optimizer()
        self.max_lr = 6e-4
        self.min_lr = self.max_lr * 0.1
        self.warmup_steps = 10
        self.max_steps = 50

    def _setup_device(self):
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print("Using device:", device)
        return device

    def _setup_ddp(self):
        ddp = int(os.environ.get('RANK', -1)) != -1
        if ddp:
            assert torch.cuda.is_available(), "DDP requires CUDA"
            init_process_group(backend='nccl')
            ddp_rank = int(os.environ['RANK'])
            ddp_local_rank = int(os.environ['LOCAL_RANK'])
            ddp_world_size = int(os.environ['WORLD_SIZE'])
            device = f"cuda:{ddp_local_rank}"
            torch.cuda.set_device(device)
            master_process = ddp_rank == 0
        else:
            ddp_rank = 0
            ddp_local_rank = 0
            ddp_world_size = 1
            master_process = True
            device = self.device
        return ddp, ddp_rank, ddp_local_rank, ddp_world_size, master_process

    def _calculate_grad_accum_steps(self):
        assert self.total_batch_size % (self.B * self.T * self.ddp_world_size) == 0, \
            "make sure the total batch size is divisible by B * T"
        grad_accum_steps = self.total_batch_size // (self.B * self.T * self.ddp_world_size)
        if self.master_process:
            print(f"total desired batch size: {self.total_batch_size}")
            print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
        return grad_accum_steps

    def _setup_model(self):
        torch.manual_seed(1337)
        torch.cuda.manual_seed(1337) if self.device == 'cuda' else None
        torch.set_float32_matmul_precision('high')

        model = GPT(GPTConfig(vocab_size=50304))
        model = model.to(self.device)
        if self.compile_flag:
            model = torch.compile(model)
        
        if self.ddp:
            model = DDP(model, device_ids=[self.ddp_local_rank])
        return model

    def _setup_optimizer(self):
        raw_model = self.model.module if self.ddp else self.model
        optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=self.max_lr, device=self.device)
        return optimizer

    def get_lr(self, it):
        if it < self.warmup_steps:
            return self.max_lr * (it + 1) / self.warmup_steps
        elif it > self.max_steps:
            return self.min_lr

        decay_ratio = (it - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)

    def train(self):
        """
        Train the model for a fixed number of steps.
        """
        for step in range(self.max_steps):
            t0 = time.time()
            self.optimizer.zero_grad()
            loss_accum = 0.0
            for micro_step in range(self.grad_accum_steps):
                x, y = self.train_loader.next_batch()
                x, y = x.to(self.device), y.to(self.device)
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                    logits, loss = self.model(x, y)
                loss = loss / self.grad_accum_steps
                loss_accum += loss.detach()
                if self.ddp:
                    self.model.require_backward_grad_sync = (micro_step == self.grad_accum_steps - 1)
                loss.backward()
            if self.ddp:
                dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            lr = self.get_lr(step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.optimizer.step()
            torch.cuda.synchronize() if self.device == 'cuda' else None
            t1 = time.time()
            dt = t1 - t0
            tokens_processed = self.train_loader.B * self.train_loader.T * self.grad_accum_steps * self.ddp_world_size
            tokens_per_second = tokens_processed / dt
            if self.master_process:
                print(f"step {step} | loss {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f} | tokens/s: {tokens_per_second:.2f}")

    def generate(self, tokens, num_return_sequences=5, max_length=30):
        """
        Generate sequences from the model

        Args:
            tokens (torch.Tensor): the text
            num_return_sequences (int, optional): _description_. Defaults to 5.
            max_length (int, optional): _description_. Defaults to 30.
        """
        enc = tiktoken.get_encoding('gpt2')
        self.model.eval()
        x = tokens.to(self.device)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42) if self.device == 'cuda' else None

        while x.size(1) < max_length:
            with torch.no_grad():
                logits = self.model(x)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, num_samples=1)
                xcol = torch.gather(topk_indices, -1, ix)
                x = torch.cat((x, xcol), dim=1)

        for i in range(num_return_sequences):
            output_tokens = x[i, :max_length].tolist()
            decoded = enc.decode(output_tokens)
            print(">", decoded)

    def cleanup(self):
        if self.ddp:
            destroy_process_group()


# Example usage:
# trainer = DistributedGPTTrainer(compile_flag=False)
# trainer.train()
# trainer.generate(tokens)
# trainer.cleanup()
