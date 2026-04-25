import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


from dlsca.aes_utils import SBOX
from dlsca.metrics import ge_curve
from dlsca.model import CNN1D
from dlsca.metrics import guessing_entropy
from dlsca.metrics import ge_curve_avg

def make_synth_dataset(N=20000, L=800, key_byte=0x2B, byte_idx=0, leak_pos=320, snr=0.35, jitter=20, seed=1):
    rng = np.random.default_rng(seed)
    plaintexts = rng.integers(0, 256, size=(N, 16), dtype=np.uint8)
    labels = SBOX[np.bitwise_xor(plaintexts[:, byte_idx], key_byte)].astype(np.int64)

    traces = rng.normal(0, 1.0, size=(N, L)).astype(np.float32)

    x = np.arange(L)
    base = np.exp(-0.5 * ((x - leak_pos) / 8.0) ** 2).astype(np.float32)

    # amplitude liée au label mais plus faible
    amp = (labels.astype(np.float32) - 127.5) / 128.0  # plus petit qu’avant

    # jitter: décalage aléatoire de la fuite
    shifts = rng.integers(-jitter, jitter + 1, size=N)
    for i in range(N):
        g = np.roll(base, shifts[i])
        traces[i] += (snr * amp[i]) * g

    # z-score
    traces = (traces - traces.mean(axis=0, keepdims=True)) / (traces.std(axis=0, keepdims=True) + 1e-6)
    return traces, plaintexts, labels

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    key_byte = 0x2B
    byte_idx = 0

    X, P, y = make_synth_dataset(N=30000, L=800, key_byte=key_byte, byte_idx=byte_idx, snr=0.35, jitter=20)
    n_train = 24000
    Xtr, Ptr, ytr = X[:n_train], P[:n_train], y[:n_train]
    Xte, Pte, yte = X[n_train:], P[n_train:], y[n_train:]

    train_dl = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)), batch_size=256, shuffle=True)
    test_dl  = DataLoader(TensorDataset(torch.tensor(Xte), torch.tensor(yte)), batch_size=512, shuffle=False)

    model = CNN1D(input_len=X.shape[1], n_classes=256).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(1, 6):
        model.train()
        pbar = tqdm(train_dl, desc=f"epoch {epoch}")
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss.item()))

        model.eval()
        correct, total = 0, 0
        all_probs = []
        with torch.no_grad():
            for xb, yb in test_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_probs.append(probs)
                pred = logits.argmax(1)
                correct += int((pred == yb).sum().item())
                total += int(yb.numel())

        acc = correct / total
        probs = np.concatenate(all_probs, axis=0)

        ge = guessing_entropy(
            probs=probs[:2000],
            plaintexts=Pte[:2000],
            true_key=key_byte,
            byte_idx=byte_idx,
            sbox=SBOX
        )

        curve = ge_curve_avg(probs=probs, plaintexts=Pte, true_key=key_byte, byte_idx=byte_idx, sbox=SBOX, n_runs=20, seed=epoch)
        print(f"Epoch {epoch}: test_acc={acc:.3f} | GE avg curve: {curve}")

    torch.save(model.state_dict(), "cnn1d_baseline.pt")
    print("Saved -> cnn1d_baseline.pt")

if __name__ == "__main__":
    main()
