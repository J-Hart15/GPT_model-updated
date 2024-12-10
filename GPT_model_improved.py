#Improved GPT model with check pointing, gratien trimming and some other changes

import torch
import torch.nn as nn
from torch.nn import functional as F
import sentencepiece as spm
import os
# Gradient Clipping
from torch.nn.utils import clip_grad_norm_

import numpy as np
from numpy.linalg import norm

from collections import Counter


# SentencePiece tokenizer
sp = spm.SentencePieceProcessor(model_file=r"m.model")

batch_size = 64
block_size = 256
max_iters = 500
eval_interval = 100
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
eval_iters = 200
n_embd = 100
n_head = 4
n_layer = 4
dropout = 0.2

use_subword_tokens = False  # Flag to switch between tokenization modes

torch.manual_seed(1337)

with open('Training_data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

if use_subword_tokens:
    # Subword tokenization using SentencePiece
    data = torch.tensor(sp.encode(text, out_type=int), dtype=torch.long)
    vocab_size = sp.vocab_size()  # Use SentencePiece vocabulary size
    decode = lambda l: sp.decode(l)
else:
    # Character-level tokenization

    #Text Filtering
    text = text.lower()
    char_counts = Counter(text)
    # Filter characters that appear at least 100 times
    valid_chars = {char for char, count in char_counts.items() if count >= 100}
    # Create a new string with only valid characters
    text = ''.join(char for char in text if char in valid_chars)


    chars = sorted(list(set(text)))
    print(chars)
    vocab_size = len(chars)
    string_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_string = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [string_to_int[c] for c in s]
    decode = lambda l: ''.join([int_to_string[i] for i in l])
    
    data = torch.tensor(encode(text), dtype=torch.long)


# Train and test splits
train_data = torch.cat((data[:int(.5 * len(data))], data[int(.6 * len(data)):]))
val_data = data[int(.5*len(data)):int(.6*len(data))]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i+1:i + block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)

    return x,y

def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Head class
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        key = self.key(x)
        query = self.query(x)

        wei = query @ key.transpose(-2,-1) * key.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

# MultiHeadAttention class
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

# FeedForward class
class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

# Block class
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# GPT_LanguageModel class
class GPT_LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx



model = GPT_LanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# Function to save checkpoint
def save_checkpoint(iteration, model, optimizer, losses, filename="checkpoint.pth"):
    checkpoint = {
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'losses': losses
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at iteration {iteration}")

# Function to load checkpoint
def load_checkpoint(model, optimizer, filename="checkpoint.pth"):
    checkpoint = torch.load(filename, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_iter = checkpoint['iteration']
    losses = checkpoint['losses']
    print(f"Checkpoint loaded, resuming from iteration {start_iter}")
    return start_iter, losses

# Main training loop
max_grad_norm = 1.0  # Set the max gradient norm for clipping
checkpoint_path = "checkpoint.pth"

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Load checkpoint if exists
start_iter = 0
if os.path.exists(checkpoint_path):
    start_iter, losses = load_checkpoint(model, optimizer, checkpoint_path)
else:
    losses = None


for iter in range(start_iter, max_iters):
    optimizer.zero_grad(set_to_none=True)

    # Sample a batch of data
    xb, yb = get_batch('train')

    # Use autocast for mixed precision
    with torch.amp.autocast(device_type='cuda'):
        logits, loss = model(xb, yb)

    # Backpropagation
    loss.backward()

    # Clip gradients to avoid exploding gradients
    clip_grad_norm_(model.parameters(), max_grad_norm)

    # Optimizer step
    optimizer.step()

    # Every once in a while, evaluate the loss and save checkpoint
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        save_checkpoint(iter, model, optimizer, losses, filename=checkpoint_path)

# Generate text
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=1000)[0].tolist()))
