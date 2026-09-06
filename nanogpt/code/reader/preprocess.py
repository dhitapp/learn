import os
import torch
from pathlib import Path

BATCH_SIZE = int(os.getenv('BATCH_SIZE', '4'))
CONTEXT_WINDOW_SIZE = int(os.getenv('CONTEXT_WINDOW_SIZE', '32'))

def open_file(file_path: str | Path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    f.close()
    return text

def encode(stoi: dict, sentence: str):
    encoded = []
    for c in sentence:
        encoded += [stoi[c]]
    return encoded

def decode(itos: dict, idxs: list):
    decoded = []
    for i in idxs:
        decoded += [itos[i]]
    return ''.join(decoded)

def pad_remain(data: list):
        max_len = max(len(seq) for seq in data)
        padded_data = [seq + [0] * (max_len - len(seq)) for seq in data]
        return padded_data

def get_batch(data, batch_size: int = BATCH_SIZE, context_window_size: int = CONTEXT_WINDOW_SIZE):
    X = []
    Y = []

    for corpus in data:
        ix = torch.randint(
            len(corpus) - context_window_size,
            (batch_size,)
        )

        x = torch.stack([
            torch.tensor(corpus[i:i + context_window_size])
            for i in ix.tolist()
        ])

        y = torch.stack([
            torch.tensor(corpus[i + 1:i + context_window_size + 1])
            for i in ix.tolist()
        ])

        X.append(x)
        Y.append(y)
    X = torch.stack(X)
    Y = torch.stack(Y)
    X = X.reshape(-1, CONTEXT_WINDOW_SIZE)
    Y = Y.reshape(-1, CONTEXT_WINDOW_SIZE)
    return X, Y



