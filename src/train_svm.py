# Task 4: linear and RBF support vector machines on Set R (6 standardised raw features)
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from prepare import RESULTS, prepare
from metrics import accuracy

C_VALUES = [0.1, 1, 10, 100]
GAMMAS = [0.1, 1]

def main():
    d = prepare()
    X_train, X_val = d["X_train_std"], d["X_val_std"]
    y_train, y_val = d["y_train"], d["y_val"]

    rows, models = [], {}

    # 4.1 linear SVM for every C
    for C in C_VALUES:
        model = SVC(kernel="linear", C=C).fit(X_train, y_train)
        acc = accuracy(y_val, model.predict(X_val))
        rows.append({"kernel": "linear", "C": C, "gamma": "-", "val_acc": acc, "support_vectors": int(model.n_support_.sum())})
        models[("linear", C, "-")] = model
        print("linear | C =", C, "| val acc =", round(acc, 4), "| support vectors =", int(model.n_support_.sum()))

    # 4.2 RBF SVM for every combination of C and gamma (8 models)
    for C in C_VALUES:
        for g in GAMMAS:
            model = SVC(kernel="rbf", C=C, gamma=g).fit(X_train, y_train)
            acc = accuracy(y_val, model.predict(X_val))
            rows.append({"kernel": "rbf", "C": C, "gamma": g, "val_acc": acc, "support_vectors": int(model.n_support_.sum())})
            models[("rbf", C, g)] = model
            print("rbf | C =", C, "| gamma =", g, "| val acc =", round(acc, 4), "| support vectors =", int(model.n_support_.sum()))

    # 4.3 Table 3
    table3 = pd.DataFrame(rows)
    print(table3.to_string(index=False))
    table3.to_csv(RESULTS / "table3_svm_validation.csv", index=False)

    # 4.4 keep the highest validation accuracy for each kernel (first one wins a tie) and save with joblib
    best_lin = max([r for r in rows if r["kernel"] == "linear"], key=lambda r: r["val_acc"])
    best_rbf = max([r for r in rows if r["kernel"] == "rbf"], key=lambda r: r["val_acc"])
    print("Best linear: C =", best_lin["C"])
    print("Best RBF: C =", best_rbf["C"], ", gamma =", best_rbf["gamma"])
    joblib.dump(models[("linear", best_lin["C"], "-")], RESULTS / "best_linear_svm.joblib")
    joblib.dump(models[("rbf", best_rbf["C"], best_rbf["gamma"])], RESULTS / "best_rbf_svm.joblib")

    # figure: validation accuracy against C (log scale) for linear and both RBF gammas
    fig, ax = plt.subplots(figsize=(7, 4))
    lin = [r for r in rows if r["kernel"] == "linear"]
    ax.plot([r["C"] for r in lin], [r["val_acc"] for r in lin], marker="o", label="linear")
    for g in GAMMAS:
        sub = [r for r in rows if r["kernel"] == "rbf" and r["gamma"] == g]
        ax.plot([r["C"] for r in sub], [r["val_acc"] for r in sub], marker="o", label=f"rbf, gamma = {g}")
    ax.set_xscale("log")
    ax.set_xlabel("C (log scale)")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Validation accuracy vs C")
    ax.legend()
    fig.savefig(RESULTS / "fig_svm_val_acc_vs_C.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
