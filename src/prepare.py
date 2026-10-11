# Task 1: cleaning, split and standardisation
from pathlib import Path
import numpy as np
import pandas as pd
from load_data import load_raw

# project root folder (the folder that contains src, data and results)
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# the six measurement columns
FEATURES = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]

def load_clean(verbose=False):
    # load the raw csv from the data folder
    df = load_raw(str(ROOT / "data"))
    raw_shape = df.shape
    if verbose:
        print(df.shape)
        print(df.columns.tolist())
        print(df.head(5).to_string())

    # find the columns that are completely empty and drop them
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    df = df.drop(columns=empty_cols)

    # rename the label column to fault
    df = df.rename(columns={"Output (S)": "fault"})

    if verbose:
        print("Removed empty columns:", empty_cols)
        print(df[FEATURES].dtypes)
        print(df.isna().sum())
    return df, raw_shape, empty_cols

def split_indices(df):
    # split with the exact code from the assignment
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(df))
    n_train, n_val = int(0.6 * len(df)), int(0.2 * len(df))
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx

def add_intercept(X):
    # add the column of ones (x0 = 1) in front
    return np.c_[np.ones(len(X)), X]

def make_features_R(X_raw, mu, sigma):
    # Set R: standardise the six raw features with training statistics, then add the intercept
    return add_intercept((X_raw - mu) / sigma)

def make_features_Q(X_raw, muQ, sigmaQ):
    # Set Q: squares from the raw values first, then standardise all 12 columns, then add the intercept
    return add_intercept((np.c_[X_raw, X_raw ** 2] - muQ) / sigmaQ)

def prepare(verbose=False):
    df, raw_shape, empty_cols = load_clean(verbose)
    train_idx, val_idx, test_idx = split_indices(df)

    # raw feature arrays and labels for each split
    X = df[FEATURES].to_numpy(dtype=float)
    y = df["fault"].to_numpy()
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    # mean and standard deviation from the training rows only
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)

    # apply the training statistics to all three splits
    return {
        "df": df, "raw_shape": raw_shape, "empty_cols": empty_cols,
        "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "mu": mu, "sigma": sigma,
        "X_train_std": (X_train - mu) / sigma,
        "X_val_std": (X_val - mu) / sigma,
        "X_test_std": (X_test - mu) / sigma,
    }

def main():
    d = prepare(verbose=True)
    df = d["df"]

    # class counts
    n_fault, n_nofault = int((df["fault"] == 1).sum()), int((df["fault"] == 0).sum())
    print("fault=1:", n_fault, " fault=0:", n_nofault)

    # rows and fault fraction in each split
    for name in ["train", "val", "test"]:
        y_split = d["y_" + name]
        print(name, len(y_split), y_split.mean())

    # check: training means should be about 0 and stds about 1
    print(d["X_train_std"].mean(axis=0).round(6))
    print(d["X_train_std"].std(axis=0).round(6))

    # Table 1
    table1 = pd.DataFrame({
        "Item": ["Raw rows / raw columns", "Columns removed", "Missing values after cleaning",
                 "Fault rows / no-fault rows", "Train / validation / test rows",
                 "Fault fraction in train / validation / test"],
        "Value": [f"{d['raw_shape'][0]} / {d['raw_shape'][1]}", ", ".join(d["empty_cols"]),
                  int(df.isna().sum().sum()), f"{n_fault} / {n_nofault}",
                  f"{len(d['y_train'])} / {len(d['y_val'])} / {len(d['y_test'])}",
                  f"{d['y_train'].mean():.4f} / {d['y_val'].mean():.4f} / {d['y_test'].mean():.4f}"]})
    print(table1.to_string(index=False))
    table1.to_csv(RESULTS / "table1_data_preparation.csv", index=False)

if __name__ == "__main__":
    main()
