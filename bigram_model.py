#Bigram model
#https://github.com/karpathy/ng-video-lecture/blob/master/bigram.py
# ^ used as the bases for this model

import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
from numpy.linalg import norm

from collections import Counter

#Sentence peice is sub word tokenizer that is commonly used
import sentencepiece as spm
sp = spm.SentencePieceProcessor(model_file=r"m.model")



batch_size = 32 # How many sequences are being processed in parallel
block_size = 8 # maximum conetxt length for predictions
max_iters = 5000
eval_interval = 1000
learning_rate = .001
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print (device)
eval_iters = 200

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
    

'''
print (chars)
test = torch.tensor(encode("hello world"))
print("hello world is encoded into")
print(test)
print (torch.tensor(decode(test)))
'''

#train and test splits
n = int(.9*len(data)) # splits training and valadation data
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

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size,vocab_size)

    def forward(self, idx, targets = None):
        logits = self.token_embedding_table(idx) #[B, T, C] batch, time, channels -> [B, T, vocab size]

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters-1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=1000)[0].tolist()))

'''embedding_table = m.token_embedding_table.weight.detach().cpu().numpy()  # Detach embeddings and move to CPU
print("Embedding table after training:")
print(embedding_table)

# Find the indices for 'q' and 'u' in the tokenized vocabulary
if use_subword_tokens:
    index_q = sp.encode('q', out_type=int)
    index_u = sp.encode('u', out_type=int)
else:
    string_to_int = {ch: i for i, ch in enumerate(sorted(list(set(text))))}
    index_q = string_to_int['q']
    index_u = string_to_int['u']

# Retrieve the embeddings for 'q' and 'u'
embedding_q = embedding_table[index_q]
embedding_u = embedding_table[index_u]

# Compute cosine similarity
cosine_similarity = np.dot(embedding_q, embedding_u) / (norm(embedding_q) * norm(embedding_u))

# Print the similarity
print(f"Cosine similarity between 'q' and 'u': {cosine_similarity:.4f}")'''