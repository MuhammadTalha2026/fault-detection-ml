# Task 2: exploratory plots from the raw (unstandardised) data
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from prepare import RESULTS, load_clean

def main():
    df, _, _ = load_clean()

    # 2.1 first 2000 rows: three currents with the fault rows shaded
    d = df.iloc[:2000]
    rows = np.arange(len(d))
    is_fault = (d["fault"] == 1).to_numpy()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(rows, d["Ia"], label="Ia")
    ax.plot(rows, d["Ib"], label="Ib")
    ax.plot(rows, d["Ic"], label="Ic")
    ax.fill_between(rows, 0, 1, where=is_fault, transform=ax.get_xaxis_transform(), color="red", alpha=0.2, label="fault = 1")
    ax.set_xlabel("Row number (time sample)")
    ax.set_ylabel("Current (A)")
    ax.set_title("Line currents, first 2000 rows")
    ax.legend()
    fig.savefig(RESULTS / "fig_currents_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2.2 scatter of Ia against Ib, coloured by fault
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(df.loc[df["fault"] == 0, "Ia"], df.loc[df["fault"] == 0, "Ib"], s=5, alpha=0.5, label="no fault (0)")
    ax.scatter(df.loc[df["fault"] == 1, "Ia"], df.loc[df["fault"] == 1, "Ib"], s=5, alpha=0.5, color="red", label="fault (1)")
    ax.set_xlabel("Ia (A)")
    ax.set_ylabel("Ib (A)")
    ax.set_title("Ia vs Ib by class")
    ax.legend()
    fig.savefig(RESULTS / "fig_scatter_ia_ib.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2.3 histogram of Ia for fault and no-fault rows on the same axes
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df.loc[df["fault"] == 0, "Ia"], bins=60, alpha=0.6, label="no fault (0)")
    ax.hist(df.loc[df["fault"] == 1, "Ia"], bins=60, alpha=0.6, color="red", label="fault (1)")
    ax.set_xlabel("Ia (A)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Ia")
    ax.legend()
    fig.savefig(RESULTS / "fig_hist_ia.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Saved 3 figures in", RESULTS)

if __name__ == "__main__":
    main()
