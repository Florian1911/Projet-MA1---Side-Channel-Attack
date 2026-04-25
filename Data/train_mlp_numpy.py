import argparse
import json
from pathlib import Path

import numpy as np


def center_and_detrend(traces: np.ndarray) -> np.ndarray:
    t = traces.astype(np.float64, copy=False)
    t = t - t.mean(axis=1, keepdims=True)
    x = np.linspace(-1.0, 1.0, t.shape[1], dtype=np.float64)
    x2 = float(np.dot(x, x))
    slopes = (t @ x) / x2
    t = t - np.outer(slopes, x)
    return t.astype(np.float32)


def select_poi_anova(x: np.ndarray, y: np.ndarray, poi: int, n_classes: int) -> np.ndarray:
    n, tlen = x.shape
    overall_mean = x.mean(axis=0)
    between_num = np.zeros(tlen, dtype=np.float64)
    within_num = np.zeros(tlen, dtype=np.float64)
    k_eff = 0
    for c in range(n_classes):
        idx = (y == c)
        nc = int(idx.sum())
        if nc < 2:
            continue
        k_eff += 1
        xc = x[idx]
        mc = xc.mean(axis=0)
        vc = xc.var(axis=0, ddof=1)
        between_num += nc * (mc - overall_mean) ** 2
        within_num += (nc - 1) * vc
    if k_eff < 2:
        return np.arange(min(poi, tlen), dtype=np.int32)
    between = between_num / (k_eff - 1)
    within = within_num / max(1, (n - k_eff))
    score = between / (within + 1e-12)
    idx = np.argsort(score)[::-1][:poi]
    return np.sort(idx.astype(np.int32))


def softmax(z: np.ndarray) -> np.ndarray:
    s = z - z.max(axis=1, keepdims=True)
    e = np.exp(s)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


def accuracy_topk(logits: np.ndarray, y: np.ndarray, k: int = 1) -> float:
    top = np.argpartition(logits, -k, axis=1)[:, -k:]
    ok = np.any(top == y[:, None], axis=1)
    return float(ok.mean())


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="dataset_20k.npz")
    ap.add_argument("--traces-key", default="traces")
    ap.add_argument("--out", default="model_mlp_numpy.npz")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--l2", type=float, default=1e-5)
    ap.add_argument("--poi", type=int, default=1024)
    ap.add_argument("--h1", type=int, default=512)
    ap.add_argument("--h2", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--test-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--patience", type=int, default=20)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    d = np.load(args.npz)
    traces = d[args.traces_key].astype(np.float32)
    labels = d["labels"].astype(np.int64)
    key = d["key"] if "key" in d.files else None

    n, tlen = traces.shape
    n_classes = int(labels.max()) + 1
    print(f"dataset: n={n}, tlen={tlen}, classes={n_classes}")

    x = center_and_detrend(traces)

    perm = rng.permutation(n)
    x = x[perm]
    y = labels[perm]

    n_test = int(n * args.test_ratio)
    n_val = int(n * args.val_ratio)
    n_train = n - n_val - n_test
    x_train, y_train = x[:n_train], y[:n_train]
    x_val, y_val = x[n_train:n_train + n_val], y[n_train:n_train + n_val]
    x_test, y_test = x[n_train + n_val:], y[n_train + n_val:]

    poi_idx = select_poi_anova(x_train, y_train, args.poi, n_classes)
    x_train = x_train[:, poi_idx]
    x_val = x_val[:, poi_idx]
    x_test = x_test[:, poi_idx]

    mu = x_train.mean(axis=0, keepdims=True)
    sigma = x_train.std(axis=0, keepdims=True) + 1e-6
    x_train = (x_train - mu) / sigma
    x_val = (x_val - mu) / sigma
    x_test = (x_test - mu) / sigma

    d_in = x_train.shape[1]
    h1 = args.h1
    h2 = args.h2
    c = n_classes

    # He init for ReLU
    w1 = (rng.standard_normal((d_in, h1)) * np.sqrt(2.0 / d_in)).astype(np.float32)
    b1 = np.zeros((h1,), dtype=np.float32)
    w2 = (rng.standard_normal((h1, h2)) * np.sqrt(2.0 / h1)).astype(np.float32)
    b2 = np.zeros((h2,), dtype=np.float32)
    w3 = (rng.standard_normal((h2, c)) * np.sqrt(2.0 / h2)).astype(np.float32)
    b3 = np.zeros((c,), dtype=np.float32)

    params = [w1, b1, w2, b2, w3, b3]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step = 0

    best_val = -1.0
    best = None
    no_improve = 0

    batch = args.batch_size
    nb = (n_train + batch - 1) // batch

    keep_prob = 1.0 - args.dropout
    use_dropout = args.dropout > 0.0

    for epoch in range(1, args.epochs + 1):
        p = rng.permutation(n_train)
        x_train = x_train[p]
        y_train = y_train[p]

        loss_sum = 0.0
        for i in range(nb):
            xb = x_train[i * batch:(i + 1) * batch]
            yb = y_train[i * batch:(i + 1) * batch]
            mbs = xb.shape[0]
            if mbs == 0:
                continue

            z1 = xb @ w1 + b1
            a1 = relu(z1)
            if use_dropout:
                m1 = (rng.random(a1.shape) < keep_prob).astype(np.float32) / keep_prob
                a1 = a1 * m1
            else:
                m1 = None

            z2 = a1 @ w2 + b2
            a2 = relu(z2)
            if use_dropout:
                m2 = (rng.random(a2.shape) < keep_prob).astype(np.float32) / keep_prob
                a2 = a2 * m2
            else:
                m2 = None

            z3 = a2 @ w3 + b3
            p3 = softmax(z3)

            ce = -np.log(p3[np.arange(mbs), yb] + 1e-12).mean()
            reg = args.l2 * float((w1 * w1).sum() + (w2 * w2).sum() + (w3 * w3).sum())
            loss = ce + reg
            loss_sum += loss

            dz3 = p3
            dz3[np.arange(mbs), yb] -= 1.0
            dz3 /= mbs

            gw3 = a2.T @ dz3 + (2.0 * args.l2) * w3
            gb3 = dz3.sum(axis=0)

            da2 = dz3 @ w3.T
            if m2 is not None:
                da2 = da2 * m2
            dz2 = da2 * (z2 > 0)
            gw2 = a1.T @ dz2 + (2.0 * args.l2) * w2
            gb2 = dz2.sum(axis=0)

            da1 = dz2 @ w2.T
            if m1 is not None:
                da1 = da1 * m1
            dz1 = da1 * (z1 > 0)
            gw1 = xb.T @ dz1 + (2.0 * args.l2) * w1
            gb1 = dz1.sum(axis=0)

            grads = [gw1, gb1, gw2, gb2, gw3, gb3]

            step += 1
            for j in range(len(params)):
                m[j] = beta1 * m[j] + (1 - beta1) * grads[j]
                v[j] = beta2 * v[j] + (1 - beta2) * (grads[j] * grads[j])
                mh = m[j] / (1 - beta1 ** step)
                vh = v[j] / (1 - beta2 ** step)
                params[j] -= args.lr * mh / (np.sqrt(vh) + eps)

        # Validation (no dropout)
        vz1 = x_val @ w1 + b1
        va1 = relu(vz1)
        vz2 = va1 @ w2 + b2
        va2 = relu(vz2)
        vlogits = va2 @ w3 + b3
        val_acc1 = accuracy_topk(vlogits, y_val, k=1)
        val_acc5 = accuracy_topk(vlogits, y_val, k=5)
        print(
            f"epoch {epoch:03d} | loss={loss_sum/nb:.4f} | val@1={val_acc1:.4f} | val@5={val_acc5:.4f}",
            flush=True,
        )

        if val_acc1 > best_val:
            best_val = val_acc1
            best = [p.copy() for p in params]
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"early stopping at epoch {epoch} (patience={args.patience})")
                break

    if best is not None:
        w1, b1, w2, b2, w3, b3 = best

    tz1 = x_test @ w1 + b1
    ta1 = relu(tz1)
    tz2 = ta1 @ w2 + b2
    ta2 = relu(tz2)
    tlogits = ta2 @ w3 + b3
    test_acc1 = accuracy_topk(tlogits, y_test, k=1)
    test_acc5 = accuracy_topk(tlogits, y_test, k=5)
    print(f"test@1={test_acc1:.4f} | test@5={test_acc5:.4f}")

    np.savez(
        args.out,
        w1=w1.astype(np.float32), b1=b1.astype(np.float32),
        w2=w2.astype(np.float32), b2=b2.astype(np.float32),
        w3=w3.astype(np.float32), b3=b3.astype(np.float32),
        mu=mu.astype(np.float32), sigma=sigma.astype(np.float32),
        poi_idx=poi_idx.astype(np.int32),
        key=key if key is not None else np.array([], dtype=np.uint8),
    )
    print(f"saved model: {args.out}")

    metrics_path = Path(args.out).with_suffix(".json")
    with metrics_path.open("w") as f:
        json.dump(
            {
                "dataset": args.npz,
                "n_total": int(n),
                "n_train": int(n_train),
                "n_val": int(n_val),
                "n_test": int(n_test),
                "n_classes": int(n_classes),
                "poi": int(args.poi),
                "h1": int(args.h1),
                "h2": int(args.h2),
                "dropout": float(args.dropout),
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "l2": float(args.l2),
                "patience": int(args.patience),
                "best_val_acc1": float(best_val),
                "test_acc1": float(test_acc1),
                "test_acc5": float(test_acc5),
            },
            f,
            indent=2,
        )
    print(f"saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
