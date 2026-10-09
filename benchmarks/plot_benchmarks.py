"""Plot the cluster measurements stored next to this script (no MPI needed)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).parent
FIGURES = HERE.parent / "figures"


def matmul():
    df = pd.read_csv(HERE / "matmul_ring_kyle.csv")
    best = df.groupby(["algorithm", "nodes"]).gflops.max().unstack(0)
    nodes = best.index.values

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    ax = axes[0]
    ax.plot(nodes, best.loc[1, "p2p"] * nodes, "k:", label="ideal (linear)")
    ax.plot(nodes, best.p2p, "o-", label="ring, Sendrecv_replace")
    ax.plot(nodes, best.bcast, "s-", label="ring, Bcast")
    ax.set(xscale="log", yscale="log", xlabel="nodes (16 cores each)", ylabel="Gflops (best layout)",
           title="n = 16384, double precision")
    ax.set_xticks(nodes, [str(n) for n in nodes])
    ax.grid(True, which="both", ls=":", alpha=0.6)
    ax.legend()

    ax = axes[1]
    p2p = df[df.algorithm == "p2p"].pivot(index="nodes", columns="layout", values="gflops")
    for layout in p2p.columns:
        ax.plot(p2p.index, p2p[layout], "o-", label=layout)
    ax.set(xscale="log", xlabel="nodes", ylabel="Gflops", title="Sendrecv ring: MPI processes vs OpenBLAS threads")
    ax.set_xticks(nodes, [str(n) for n in nodes])
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend(fontsize=8)
    fig.suptitle("Distributed matrix product on the Kyle cluster", fontweight="bold")
    fig.savefig(FIGURES / "matmul_scaling.png", dpi=150)

    speedup = best / best.loc[1]
    print("best Gflops and speedup over one node:")
    print(pd.concat({"gflops": best, "speedup": speedup.round(2)}, axis=1))


def cg():
    df = pd.read_csv(HERE / "cg_weak_scaling.csv")
    efficiency = df.time_s.iloc[0] / df.time_s
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(df.processes, df.time_s, "o-", label="measured")
    axes[0].axhline(df.time_s.iloc[0], color="k", ls=":", label="ideal (constant)")
    axes[0].set(xlabel="MPI processes", ylabel="time [s]", title="1000 rows per process")
    axes[1].plot(df.processes, 100 * efficiency, "o-")
    axes[1].set(xlabel="MPI processes", ylabel="weak-scaling efficiency [%]", ylim=(0, 105),
                title="efficiency = T(4) / T(p)")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(df.processes, [str(p) for p in df.processes])
        ax.grid(True, ls=":", alpha=0.6)
    axes[0].legend()
    fig.suptitle("Conjugate gradient, Allgather version, one node (oversubscribed)", fontweight="bold")
    fig.savefig(FIGURES / "cg_weak_scaling.png", dpi=150)
    print("\nCG weak scaling efficiency:", (100 * efficiency).round(0).tolist())


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    matmul()
    cg()
