# edu_qgc_repro_full.py
# -*- coding: utf-8 -*-
"""
EDU-QGC Node Classification with strict reproducibility and dataset saving.

Run:
  python edu_qgc_repro_full.py

Requirements:
  pip install torch pennylane pennylane-lightning[gpu] networkx matplotlib
(If you don't have CUDA or lightning.gpu, the code falls back to default.qubit CPU simulator.)
"""

import os
# MUST set CUBLAS_WORKSPACE_CONFIG before importing torch to enable deterministic cuBLAS
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
import math
import random
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np

# Now import torch after setting env var
import torch
import torch.nn.functional as F
from torch import nn

import networkx as nx
import pennylane as qml
import matplotlib.pyplot as plt

# ---------------------------
# Reproducibility helper
# ---------------------------
def set_global_determinism(seed: int):
    """
    Set seeds and flags to make experiments deterministic where feasible.
    Must set CUBLAS_WORKSPACE_CONFIG BEFORE importing torch (done at top).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Enable deterministic algorithms (will raise if impossible)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:
        # Give clear advice and re-raise
        print("Failed to enable deterministic algorithms:", e)
        print("Make sure CUBLAS_WORKSPACE_CONFIG is set before Python starts.")
        print("If running in Jupyter, restart the kernel after setting the env var.")
        raise

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------------------------
# Utilities
# ---------------------------
def confusion_matrix(y_true, y_pred, num_classes=2):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm

def macro_accuracy(y_true, y_pred, num_classes=2):
    cm = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    per_class = []
    for c in range(num_classes):
        support = cm[c, :].sum()
        correct = cm[c, c]
        acc_c = correct / support if support > 0 else 0.0
        per_class.append(acc_c)
    return float(np.mean(per_class))

def print_confusion(cm):
    num_classes = cm.shape[0]
    print("Confusion matrix (rows=true, cols=pred):")
    for i in range(num_classes):
        row = " ".join(f"{cm[i, j]:4d}" for j in range(num_classes))
        print(f"class {i}: {row}")

def save_dataset(dataset, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(dataset, f)
    print(f"[dataset saved] {path.resolve()} (num_graphs={len(dataset)})")

def load_dataset(path: Path):
    with open(path, "rb") as f:
        ds = pickle.load(f)
    print(f"[dataset loaded] {path.resolve()} (num_graphs={len(ds)})")
    return ds

# ---------------------------
# Dataset generation
# ---------------------------
def make_synthetic_graph(n_min=10, n_max=30, p_range=(0.08, 0.2), q_range=(0.01, 0.08),
                         feat_noise=0.2, weak_signal_scale=0.4, seed=None, shuffle_labels=False):
    rng = np.random.default_rng(seed)
    N = int(rng.integers(n_min, n_max + 1))
    n0 = N // 2

    p = float(rng.uniform(*p_range))
    q = float(rng.uniform(*q_range))
    if p < q:
        p, q = q, p

    G = nx.Graph()
    G.add_nodes_from(range(N))
    comm = np.zeros(N, dtype=int)
    comm[n0:] = 1

    for i in range(N):
        for j in range(i + 1, N):
            prob = p if comm[i] == comm[j] else q
            if rng.random() < prob:
                G.add_edge(i, j)

    edges = np.array(list(G.edges()), dtype=np.int64).T if G.number_of_edges() > 0 else np.zeros((2,0), dtype=np.int64)
    edges_rev = edges[::-1]
    edge_index = np.concatenate([edges, edges_rev], axis=1) if edges.shape[1] > 0 else edges

    y = comm.astype(np.int64)
    if shuffle_labels:
        rng.shuffle(y)

    degs = np.array([G.degree(i) for i in range(N)], dtype=np.float32)
    deg_norm = (degs / max(1, N - 1)).astype(np.float32)
    clustering = np.array(list(nx.clustering(G).values()), dtype=np.float32)
    weak_signal = np.where(y == 1, +weak_signal_scale, -weak_signal_scale).astype(np.float32)

    noise = rng.normal(0.0, feat_noise, size=(N, 3)).astype(np.float32)
    X = np.stack([deg_norm, clustering, weak_signal], axis=1) + noise

    return dict(edge_index=edge_index.astype(np.int64), X=X.astype(np.float32), y=y.astype(np.int64))

def train_val_test_split_graphs(num_graphs, splits=(0.6, 0.2, 0.2), seed=0):
    idx = np.arange(num_graphs)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_train = int(splits[0] * num_graphs)
    n_val = int(splits[1] * num_graphs)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx

# ---------------------------
# EDU-QGC model
# ---------------------------
class EDUQGCNodeClassifier(nn.Module):
    def __init__(self, n_nodes, in_feats=3, T=2, seed=0, use_gpu_qnode=True, use_feat_skip=True):
        super().__init__()
        self.n_nodes = n_nodes
        self.T = T
        self.use_feat_skip = use_feat_skip

        # Parameter packing
        self.enc_W = nn.Parameter(torch.randn(T, 2, in_feats) * 0.08)  # [T,2,F]
        self.enc_b = nn.Parameter(torch.randn(T, 2) * 0.02)           # [T,2]

        self.edge_phase  = nn.Parameter(torch.randn(T) * 0.08)
        self.pre_theta   = nn.Parameter(torch.randn(T) * 0.08)
        self.pre_psi     = nn.Parameter(torch.randn(T) * 0.08)
        self.post_theta  = nn.Parameter(torch.randn(T) * 0.08)
        self.post_psi    = nn.Parameter(torch.randn(T) * 0.08)

        readin_dim = 1 + in_feats if use_feat_skip else 1
        self.readout = nn.Linear(readin_dim, 2)

        use_cuda = torch.cuda.is_available()
        qdev_name = "lightning.gpu" if (use_gpu_qnode and use_cuda) else "default.qubit"
        self.dev = qml.device(qdev_name, wires=n_nodes, shots=None)

        @qml.qnode(self.dev, interface="torch", diff_method="best")
        def circuit(edge_index, X, enc_W, enc_b,
                    edge_phase, pre_theta, pre_psi, post_theta, post_psi):
            # Loop layers
            for t in range(self.T):
                enc_out = X @ enc_W[t].T + enc_b[t]  # [N,2]
                alphas = enc_out[:, 0]; betas = enc_out[:, 1]
                for i in range(self.n_nodes):
                    qml.RX(alphas[i], wires=i)
                    qml.RY(betas[i], wires=i)

                for i in range(self.n_nodes):
                    qml.RZ(pre_psi[t], wires=i)
                    qml.RX(pre_theta[t], wires=i)

                E = edge_index.shape[1]
                for e in range(E):
                    u = int(edge_index[0, e].item()); v = int(edge_index[1, e].item())
                    if u != v:
                        qml.ControlledPhaseShift(edge_phase[t], wires=[u, v])

                for i in range(self.n_nodes):
                    qml.RZ(post_psi[t], wires=i)
                    qml.RX(post_theta[t], wires=i)

            return [qml.expval(qml.Z(i)) for i in range(self.n_nodes)]

        self._circuit = circuit

    def forward(self, edge_index_torch, x_torch):
        model_device = next(self.parameters()).device
        edge_index = edge_index_torch.to(model_device)
        X = x_torch.to(model_device).float()

        # Call QNode (PennyLane handles device movement if needed)
        layer_out = self._circuit(
            edge_index, X,
            self.enc_W, self.enc_b,
            self.edge_phase,
            self.pre_theta, self.pre_psi,
            self.post_theta, self.post_psi
        )

        expvals = torch.stack(layer_out, dim=0).float().to(model_device)
        if expvals.dim() == 1:
            expvals = expvals.unsqueeze(1)
        elif expvals.dim() == 2 and expvals.shape[1] == 1:
            pass
        else:
            expvals = expvals.squeeze(-1).unsqueeze(1)

        if self.use_feat_skip:
            readin = torch.cat([expvals, X], dim=1)
        else:
            readin = expvals

        logits = self.readout(readin)
        return logits

# ---------------------------
# Train / Eval
# ---------------------------
def train_epoch(model, graphs, optimizer, device="cpu"):
    model.train()
    total_loss = 0.0; total_nodes = 0
    for g in graphs:
        edge_index = torch.from_numpy(g["edge_index"]).long().to(device)
        X = torch.from_numpy(g["X"]).float().to(device)
        y = torch.from_numpy(g["y"]).long().to(device)

        logits = model(edge_index, X)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.shape[0]
        total_nodes += X.shape[0]
    return total_loss / max(1, total_nodes)

@torch.no_grad()
def evaluate(model, graphs, device="cpu"):
    model.eval()
    all_true, all_pred = [], []
    total_loss, total_nodes = 0.0, 0
    for g in graphs:
        edge_index = torch.from_numpy(g["edge_index"]).long().to(device)
        X = torch.from_numpy(g["X"]).float().to(device)
        y = torch.from_numpy(g["y"]).long().to(device)

        logits = model(edge_index, X)
        loss = F.cross_entropy(logits, y)

        pred = logits.argmax(dim=1).cpu().numpy()
        all_pred.extend(pred)
        all_true.extend(list(y.cpu().numpy()))

        total_loss += loss.item() * X.shape[0]
        total_nodes += X.shape[0]

    avg_loss = total_loss / max(1, total_nodes)
    all_true = np.array(all_true); all_pred = np.array(all_pred)
    mac_acc = macro_accuracy(all_true, all_pred, num_classes=2)
    cm = confusion_matrix(all_true, all_pred, num_classes=2)
    return avg_loss, mac_acc, cm, all_true, all_pred

# ---------------------------
# Experiment runner (full)
# ---------------------------
def run_experiment(
    num_graphs=45,
    splits=(0.6, 0.2, 0.2),
    T=2,
    epochs=50,
    lr=0.03,
    seed=42,
    device="cuda",
    use_gpu_qnode=True,
    save_dataset_path: str = "./data/eduqgc_dataset_seed{seed}.pkl",
    plot_curves: bool = True
):
    # Reproducibility
    set_global_determinism(seed)
    print("[repro] seed:", seed)
    print("[repro] deterministic algos enabled:", torch.are_deterministic_algorithms_enabled())
    print("[repro] cudnn.deterministic:", torch.backends.cudnn.deterministic)
    print("[repro] cudnn.benchmark:", torch.backends.cudnn.benchmark)
    print("[repro] CUBLAS_WORKSPACE_CONFIG:", os.environ.get("CUBLAS_WORKSPACE_CONFIG"))

    # Data
    N_fixed = 20
    ds_path = Path(save_dataset_path.format(seed=seed))
    if ds_path.exists():
        dataset = load_dataset(ds_path)
    else:
        dataset = []
        for g in range(num_graphs):
            graph = make_synthetic_graph(n_min=N_fixed, n_max=N_fixed, seed=seed + 2000 + g, shuffle_labels=False)
            dataset.append(graph)
        save_dataset(dataset, ds_path)

    train_idx, val_idx, test_idx = train_val_test_split_graphs(num_graphs, splits=splits, seed=seed)
    train_graphs = [dataset[i] for i in train_idx]
    val_graphs = [dataset[i] for i in val_idx]
    test_graphs = [dataset[i] for i in test_idx]

    # Device selection
    device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
    device = torch.device(device)
    model = EDUQGCNodeClassifier(n_nodes=N_fixed, in_feats=3, T=T, seed=seed,
                                use_gpu_qnode=use_gpu_qnode, use_feat_skip=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val = (-1.0, None)
    for ep in range(1, epochs + 1):
        tr_loss = train_epoch(model, train_graphs, opt, device=device)
        val_loss, val_mac, val_cm, _, _ = evaluate(model, val_graphs, device=device)

        print(f"Epoch {ep:03d} | train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | val_macro_acc={val_mac:.3f}")

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_mac)

        if val_mac > best_val[0]:
            best_val = (val_mac, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

    # Load best
    if best_val[1] is not None:
        model.load_state_dict(best_val[1])

    # Final eval
    tr_loss, tr_mac, tr_cm, tr_y, tr_pred = evaluate(model, train_graphs, device=device)
    va_loss, va_mac, va_cm, va_y, va_pred = evaluate(model, val_graphs, device=device)
    te_loss, te_mac, te_cm, te_y, te_pred = evaluate(model, test_graphs, device=device)

    print("\nResults:")
    print(f"- Train: macro-acc={tr_mac:.3f}, loss={tr_loss:.4f}")
    print_confusion(tr_cm)
    print(f"- Val:   macro-acc={va_mac:.3f}, loss={va_loss:.4f}")
    print_confusion(va_cm)
    print(f"- Test:  macro-acc={te_mac:.3f}, loss={te_loss:.4f}")
    print_confusion(te_cm)

    # Save model + plots
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_path = out_dir / f"eduqgc_model_seed{seed}_{ts}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"[saved model] {model_path.resolve()}")

    plt_path = out_dir / f"training_curves_seed{seed}_{ts}.png"
    if plot_curves:
        fig, ax = plt.subplots(1, 2, figsize=(12,4))
        ax[0].plot(history["train_loss"], label="train loss")
        ax[0].plot(history["val_loss"], label="val loss")
        ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend(); ax[0].set_title("Loss")

        ax[1].plot(history["val_acc"], label="val macro-acc")
        ax[1].set_xlabel("epoch"); ax[1].set_ylabel("macro-accuracy"); ax[1].legend(); ax[1].set_title("Validation Macro-Accuracy")

        plt.tight_layout()
        plt.savefig(plt_path, dpi=200)
        print(f"[saved plot] {plt_path.resolve()}")
        plt.show()

    return {
        "model": model,
        "splits": (train_idx, val_idx, test_idx),
        "history": history,
        "metrics": {
            "train": dict(loss=tr_loss, macro_acc=tr_mac, cm=tr_cm, y=tr_y, pred=tr_pred),
            "val":   dict(loss=va_loss, macro_acc=va_mac, cm=va_cm, y=va_y, pred=va_pred),
            "test":  dict(loss=te_loss, macro_acc=te_mac, cm=te_cm, y=te_y, pred=te_pred),
        },
        "dataset_path": str(ds_path.resolve()),
        "model_path": str(model_path.resolve()),
        "plot_path": str(plt_path.resolve()),
    }

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_GPU_QNODE = True
    NUM_GRAPHS = 45

    results = run_experiment(
        num_graphs=NUM_GRAPHS,
        T=2,
        epochs=50,
        lr=0.03,
        seed=SEED,
        device=DEVICE,
        use_gpu_qnode=USE_GPU_QNODE,
        save_dataset_path="./data/eduqgc_dataset_seed{seed}.pkl",
        plot_curves=True
    )
    print("Done. Results summary keys:", list(results.keys()))
