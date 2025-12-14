import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

DATA_DIR = "data/ml-1m"
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

def load_ratings():
    path = os.path.join(DATA_DIR, "ratings.dat")
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user", "item", "rating", "timestamp"],
    )
    return df

def remap_ids(df):
    unique_users = df["user"].unique()
    unique_items = df["item"].unique()

    user2idx = {u: i for i, u in enumerate(unique_users)}
    item2idx = {m: i for i, m in enumerate(unique_items)}

    df["u_idx"] = df["user"].map(user2idx)
    df["i_idx"] = df["item"].map(item2idx)

    return df, len(unique_users), len(unique_items)

def build_dataset(df, U, I):
    reward = (df["rating"] >= 4).astype(np.float32).values

    states = df["u_idx"].values.astype(np.int64)
    actions = df["i_idx"].values.astype(np.int64)
    next_states = states.copy()
    done = np.ones_like(reward, dtype=np.float32)

    data = {
        "states": torch.tensor(states),
        "actions": torch.tensor(actions),
        "rewards": torch.tensor(reward),
        "next_states": torch.tensor(next_states),
        "dones": torch.tensor(done),
        "num_users": U,
        "num_items": I,
    }
    return data

def split_and_save(data):
    N = data["states"].shape[0]
    idx = np.arange(N)

    train_idx, temp_idx = train_test_split(idx, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    def sub(d, id):
        return {k: d[k][id] if isinstance(d[k], torch.Tensor) else d[k]
                for k in d}

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save({"data": sub(data, train_idx), "num_users": data["num_users"], "num_items": data["num_items"]},
               os.path.join(OUT_DIR, "train.pt"))
    torch.save({"data": sub(data, val_idx), "num_users": data["num_users"], "num_items": data["num_items"]},
               os.path.join(OUT_DIR, "val.pt"))
    torch.save({"data": sub(data, test_idx), "num_users": data["num_users"], "num_items": data["num_items"]},
               os.path.join(OUT_DIR, "test.pt"))
    print("Saved splits in data/processed")

if __name__ == "__main__":
    df = load_ratings()
    df, U, I = remap_ids(df)
    data = build_dataset(df, U, I)
    split_and_save(data)
    print("num_users:", U, "num_items:", I)
