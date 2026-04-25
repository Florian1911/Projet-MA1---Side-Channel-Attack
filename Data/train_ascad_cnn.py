import argparse
import json
from pathlib import Path

import numpy as np

try:
    import tensorflow as tf
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "TensorFlow is required for ASCAD-CNN.\n"
        "Install with: python -m pip install tensorflow-cpu\n"
        f"Import error: {exc}"
    )

AES_SBOX = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
], dtype=np.uint8)


def parse_int(v: str) -> int:
    return int(v, 0)


def center_and_detrend(traces: np.ndarray) -> np.ndarray:
    t = traces.astype(np.float64, copy=False)
    t = t - t.mean(axis=1, keepdims=True)
    x = np.linspace(-1.0, 1.0, t.shape[1], dtype=np.float64)
    x2 = float(np.dot(x, x))
    slopes = (t @ x) / x2
    t = t - np.outer(slopes, x)
    return t.astype(np.float32)


def load_labels(d: np.lib.npyio.NpzFile, byte_idx: int, key_guess: int) -> np.ndarray:
    if "labels" in d.files:
        return d["labels"].astype(np.int64)
    if "plaintexts" not in d.files:
        raise ValueError("Dataset must contain either labels or plaintexts.")
    plains = d["plaintexts"].astype(np.uint8)
    labels = AES_SBOX[np.bitwise_xor(plains[:, byte_idx], key_guess)].astype(np.int64)
    return labels


def load_traces_and_labels(path: str, byte_idx: int, key_guess: int) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(path)
    traces = d["traces"].astype(np.float32)
    labels = load_labels(d, byte_idx, key_guess)
    return traces, labels


def crop_window(traces: np.ndarray, center: int, window: int) -> np.ndarray:
    half = window // 2
    w0 = max(0, center - half)
    w1 = min(traces.shape[1], center + half)
    return traces[:, w0:w1]


def build_ascad_cnn(input_len: int, n_classes: int, lr: float) -> tf.keras.Model:
    x_in = tf.keras.Input(shape=(input_len, 1), name="trace")
    x = tf.keras.layers.Conv1D(32, 11, padding="same", activation="relu")(x_in)
    x = tf.keras.layers.AveragePooling1D(pool_size=2)(x)
    x = tf.keras.layers.Conv1D(64, 11, padding="same", activation="relu")(x)
    x = tf.keras.layers.AveragePooling1D(pool_size=2)(x)
    x = tf.keras.layers.Conv1D(128, 11, padding="same", activation="relu")(x)
    x = tf.keras.layers.AveragePooling1D(pool_size=2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(200, activation="relu")(x)
    x = tf.keras.layers.Dense(200, activation="relu")(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)
    model = tf.keras.Model(x_in, out, name="ascad_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)],
    )
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description="ASCAD-like CNN training for SCA traces")
    ap.add_argument("--train-npz", required=True, help="Profiling dataset .npz")
    ap.add_argument("--test-npz", default="", help="Optional external test dataset .npz")
    ap.add_argument("--byte", type=int, default=0)
    ap.add_argument("--key", type=parse_int, default=0x2B, help="Used only if labels are absent")
    ap.add_argument("--window", type=int, default=700)
    ap.add_argument("--center", type=int, default=-1, help="window center; -1 keeps full trace")
    ap.add_argument("--preproc", choices=["none", "center_detrend"], default="center_detrend")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="ascad_cnn_model.keras")
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(args.seed)

    x, y = load_traces_and_labels(args.train_npz, args.byte, args.key)
    if args.center >= 0:
        x = crop_window(x, args.center, args.window)
    if args.preproc == "center_detrend":
        x = center_and_detrend(x)

    n = x.shape[0]
    n_val = int(n * args.val_ratio)
    n_train = n - n_val
    idx = np.random.default_rng(args.seed).permutation(n)
    x = x[idx]
    y = y[idx]
    x_train, y_train = x[:n_train], y[:n_train]
    x_val, y_val = x[n_train:], y[n_train:]

    mu = x_train.mean(axis=0, keepdims=True)
    sigma = x_train.std(axis=0, keepdims=True) + 1e-6
    x_train = (x_train - mu) / sigma
    x_val = (x_val - mu) / sigma

    x_train = x_train[..., None]
    x_val = x_val[..., None]
    n_classes = int(max(y.max(), y_train.max(), y_val.max())) + 1
    model = build_ascad_cnn(x_train.shape[1], n_classes, args.lr)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_sparse_categorical_accuracy",
            mode="max",
            patience=args.patience,
            restore_best_weights=True,
        )
    ]

    hist = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
        callbacks=callbacks,
    )

    eval_data = {}
    v_loss, v_acc1, v_acc5 = model.evaluate(x_val, y_val, verbose=0)
    eval_data["val"] = {"loss": float(v_loss), "acc1": float(v_acc1), "acc5": float(v_acc5)}

    if args.test_npz:
        xt, yt = load_traces_and_labels(args.test_npz, args.byte, args.key)
        if args.center >= 0:
            xt = crop_window(xt, args.center, args.window)
        if args.preproc == "center_detrend":
            xt = center_and_detrend(xt)
        xt = ((xt - mu) / sigma)[..., None]
        t_loss, t_acc1, t_acc5 = model.evaluate(xt, yt, verbose=0)
        eval_data["test"] = {"loss": float(t_loss), "acc1": float(t_acc1), "acc5": float(t_acc5)}

    model.save(args.out)
    np.savez(Path(args.out).with_suffix(".npz"), mu=mu.astype(np.float32), sigma=sigma.astype(np.float32))

    out_json = Path(args.out).with_suffix(".json")
    with out_json.open("w") as f:
        json.dump(
            {
                "train_dataset": args.train_npz,
                "test_dataset": args.test_npz,
                "byte": int(args.byte),
                "key": int(args.key),
                "preproc": args.preproc,
                "window": int(args.window),
                "center": int(args.center),
                "n_train": int(n_train),
                "n_val": int(n_val),
                "n_classes": int(n_classes),
                "epochs_requested": int(args.epochs),
                "epochs_run": int(len(hist.history["loss"])),
                "best_val_acc1": float(max(hist.history["val_sparse_categorical_accuracy"])),
                "metrics": eval_data,
            },
            f,
            indent=2,
        )
    print(f"saved model: {args.out}")
    print(f"saved stats: {Path(args.out).with_suffix('.npz')}")
    print(f"saved metrics: {out_json}")


if __name__ == "__main__":
    main()
