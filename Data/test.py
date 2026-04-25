import numpy as np
import matplotlib.pyplot as plt
import argparse


def main():
    parser = argparse.ArgumentParser(description="Plot SCA traces from NPZ dataset")
    parser.add_argument("file", help="Path to .npz file")  # <-- changer ici
    parser.add_argument("--n", type=int, default=50, help="Number of traces to display")
    args = parser.parse_args()

    data = np.load(args.file)

    traces = data["traces"]
    print(f"[INFO] Loaded {traces.shape[0]} traces, length = {traces.shape[1]} samples")

    n = min(args.n, traces.shape[0])

    # =========================
    # Plot individual traces
    # =========================
    plt.figure()
    for i in range(n):
        plt.plot(traces[i], alpha=0.3)
    plt.title(f"{n} traces")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude (mV)")
    plt.grid()
    plt.savefig(outdir / "individual_traces.png")  # Sauvegarde
    plt.close()

    # =========================
    # Plot mean trace
    # =========================
    mean_trace = np.mean(traces, axis=0)
    plt.figure()
    plt.plot(mean_trace)
    plt.title("Mean trace")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude (mV)")
    plt.grid()
    plt.savefig(outdir / "mean_trace.png")
    plt.close()

    # =========================
    # Plot standard deviation
    # =========================
    std_trace = np.std(traces, axis=0)
    plt.figure()
    plt.plot(std_trace)
    plt.title("Standard deviation (activity)")
    plt.xlabel("Sample")
    plt.ylabel("Std (mV)")
    plt.grid()
    plt.savefig(outdir / "std_trace.png")
    plt.close()

    plt.show()


if __name__ == "__main__":
    main()
