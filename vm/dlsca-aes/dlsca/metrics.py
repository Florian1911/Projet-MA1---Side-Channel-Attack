import numpy as np

def rank_from_logprobs(logp: np.ndarray, true_key: int) -> int:
    order = np.argsort(-logp)
    return int(np.where(order == true_key)[0][0])

def guessing_entropy(probs: np.ndarray, plaintexts: np.ndarray, true_key: int, byte_idx: int, sbox: np.ndarray) -> float:
    N = probs.shape[0]
    logp = np.zeros(256, dtype=np.float64)

    ptb = plaintexts[:, byte_idx]
    idx = np.arange(N)
    for k in range(256):
        labels = sbox[np.bitwise_xor(ptb, k)]
        logp[k] = np.sum(np.log(np.clip(probs[idx, labels], 1e-36, 1.0)))

    return float(rank_from_logprobs(logp, true_key))

def ge_curve(probs: np.ndarray, plaintexts: np.ndarray, true_key: int, byte_idx: int, sbox: np.ndarray, steps=None):
    if steps is None:
        steps = [50, 100, 200, 500, 1000, 2000]
    out = []
    for n in steps:
        n = min(n, probs.shape[0])
        out.append((n, guessing_entropy(probs[:n], plaintexts[:n], true_key, byte_idx, sbox)))
    return out

def ge_curve_avg(probs: np.ndarray, plaintexts: np.ndarray, true_key: int, byte_idx: int, sbox: np.ndarray,
                 steps=None, n_runs: int = 20, seed: int = 0):
    if steps is None:
        steps = [50, 100, 200, 500, 1000, 2000]
    rng = np.random.default_rng(seed)
    N = probs.shape[0]
    steps = [min(s, N) for s in steps]

    acc = {s: [] for s in steps}
    for _ in range(n_runs):
        perm = rng.permutation(N)
        p2 = probs[perm]
        pt2 = plaintexts[perm]
        for s in steps:
            acc[s].append(guessing_entropy(p2[:s], pt2[:s], true_key, byte_idx, sbox))

    return [(s, float(np.mean(acc[s]))) for s in steps]
