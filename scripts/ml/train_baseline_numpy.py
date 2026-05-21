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


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)


def accuracy_topk(logits: np.ndarray, y: np.ndarray, k: int = 1) -> float:
    top = np.argpartition(logits, -k, axis=1)[:, -k:]
    ok = np.any(top == y[:, None], axis=1)
    return float(ok.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="dataset_aes_sca_no_uart_aligned.npz")
    ap.add_argument("--out", default="model_softmax_baseline.npz")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--poi", type=int, default=256)
    ap.add_argument("--selector", choices=["std", "anova"], default="anova")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--test-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    d = np.load(args.npz)
    traces = d["traces"].astype(np.float32)
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

    if args.selector == "anova":
        poi_idx = select_poi_anova(x_train, y_train, args.poi, n_classes)
    else:
        std = x_train.std(axis=0)
        poi_idx = np.sort(np.argsort(std)[::-1][:args.poi].astype(np.int32))
    x_train = x_train[:, poi_idx]
    x_val = x_val[:, poi_idx]
    x_test = x_test[:, poi_idx]

    mu = x_train.mean(axis=0, keepdims=True)
    sigma = x_train.std(axis=0, keepdims=True) + 1e-6
    x_train = (x_train - mu) / sigma
    x_val = (x_val - mu) / sigma
    x_test = (x_test - mu) / sigma

    d_in = x_train.shape[1]
    w = (0.01 * rng.standard_normal((d_in, n_classes))).astype(np.float32)
    b = np.zeros((n_classes,), dtype=np.float32)

    mw = np.zeros_like(w)
    vw = np.zeros_like(w)
    mb = np.zeros_like(b)
    vb = np.zeros_like(b)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step = 0

    best_val = -1.0
    best = None
    batch = args.batch_size
    nb = (n_train + batch - 1) // batch

    for epoch in range(1, args.epochs + 1):
        p = rng.permutation(n_train)
        x_train = x_train[p]
        y_train = y_train[p]

        loss_sum = 0.0
        for i in range(nb):
            xb = x_train[i * batch:(i + 1) * batch]
            yb = y_train[i * batch:(i + 1) * batch]
            m = xb.shape[0]
            if m == 0:
                continue

            logits = xb @ w + b
            prob = softmax(logits)
            ce = -np.log(prob[np.arange(m), yb] + 1e-12).mean()
            l2 = args.l2 * float((w * w).sum())
            loss = ce + l2
            loss_sum += loss

            dlog = prob
            dlog[np.arange(m), yb] -= 1.0
            dlog /= m
            gw = xb.T @ dlog + (2.0 * args.l2) * w
            gb = dlog.sum(axis=0)

            step += 1
            mw = beta1 * mw + (1 - beta1) * gw
            vw = beta2 * vw + (1 - beta2) * (gw * gw)
            mb = beta1 * mb + (1 - beta1) * gb
            vb = beta2 * vb + (1 - beta2) * (gb * gb)

            mwh = mw / (1 - beta1 ** step)
            vwh = vw / (1 - beta2 ** step)
            mbh = mb / (1 - beta1 ** step)
            vbh = vb / (1 - beta2 ** step)

            w -= args.lr * mwh / (np.sqrt(vwh) + eps)
            b -= args.lr * mbh / (np.sqrt(vbh) + eps)

        val_logits = x_val @ w + b
        val_acc1 = accuracy_topk(val_logits, y_val, k=1)
        val_acc5 = accuracy_topk(val_logits, y_val, k=5)
        print(
            f"epoch {epoch:03d} | loss={loss_sum/nb:.4f} | val@1={val_acc1:.4f} | val@5={val_acc5:.4f}",
            flush=True,
        )

        if val_acc1 > best_val:
            best_val = val_acc1
            best = (w.copy(), b.copy())

    if best is not None:
        w, b = best

    test_logits = x_test @ w + b
    test_acc1 = accuracy_topk(test_logits, y_test, k=1)
    test_acc5 = accuracy_topk(test_logits, y_test, k=5)
    print(f"test@1={test_acc1:.4f} | test@5={test_acc5:.4f}")

    np.savez(
        args.out,
        w=w.astype(np.float32),
        b=b.astype(np.float32),
        mu=mu.astype(np.float32),
        sigma=sigma.astype(np.float32),
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
                "selector": args.selector,
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "l2": float(args.l2),
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
