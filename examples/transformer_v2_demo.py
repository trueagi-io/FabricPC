import os
import jax
import torch
from torch.utils.data import DataLoader, Dataset
from fabricpc.graph import create_pc_graph
from fabricpc.training.train import train_pcn
from fabricpc.nodes.transformer import create_deep_transformer


# ----------------------------------------------------------------------
# LOAD LOCAL DATA
# ----------------------------------------------------------------------
def load_data(path="data/tiny_shakespeare.txt"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    else:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}
    data = [char_to_ix[ch] for ch in text]

    return data, vocab_size, char_to_ix, ix_to_char


def split_data(data, train_frac=0.8, val_frac=0.1):
    N = len(data)
    n_train = int(N * train_frac)
    n_val = int(N * val_frac)

    train_data = data[:n_train]
    val_data = data[n_train : n_train + n_val]
    test_data = data[n_train + n_val :]

    return train_data, val_data, test_data


class TextDataset(Dataset):
    def __init__(self, data, seq_len, vocab_size):
        self.data = data
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, i):
        x = torch.tensor(self.data[i : i + self.seq_len], dtype=torch.float32)
        y = torch.tensor(self.data[i + 1 : i + self.seq_len + 1], dtype=torch.long)
        y_one_hot = torch.nn.functional.one_hot(y, num_classes=self.vocab_size).float()
        return x, y_one_hot


# ----------------------------------------------------------------------
# INITIALIZE DATA
# ----------------------------------------------------------------------
seq_len = 32
batch_size = 32
data, vocab_size, char_to_ix, ix_to_char = load_data()
train_data, val_data, test_data = split_data(data)

train_dataset = TextDataset(train_data, seq_len, vocab_size)
val_dataset = TextDataset(val_data, seq_len, vocab_size)
test_dataset = TextDataset(test_data, seq_len, vocab_size)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# ----------------------------------------------------------------------
# MODEL ARCHITECTURE
# ----------------------------------------------------------------------
config = create_deep_transformer(
    depth=1,
    embed_dim=64,
    num_heads=4,
    mlp_dim=128,
    seq_len=seq_len,
    vocab_size=vocab_size,
)

# ----------------------------------------------------------------------
# CREATE PC GRAPH & TRAIN
# ----------------------------------------------------------------------
master_rng_key = jax.random.PRNGKey(42)
graph_key, train_key, eval_key = jax.random.split(master_rng_key, 3)
params, structure = create_pc_graph(config, graph_key)

train_config = {
    "num_epochs": 5,
    "infer_steps": 20,
    "eta_infer": 0.01,
    "optimizer": {"type": "adam", "lr": 1e-4},
}

print(f"Vocab Size: {vocab_size} | Training on local tiny_shakespeare.txt...")
trained_params, _, _ = train_pcn(
    params, structure, train_loader, train_config, train_key, verbose=True
)
metrics = evaluate_pcn(
    trained_params, structure, test_loader, train_config, eval_key, verbose=True
)

print(f"Test Accuracy: {metrics['accuracy'] * 100:.2f}%")
print(f"Test Loss: {metrics['loss']:.4f}")
