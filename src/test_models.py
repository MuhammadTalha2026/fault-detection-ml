# Task 5: evaluate the four final models on the test rows, once
import joblib
import numpy as np
import pandas as pd
from prepare import RESULTS, prepare, make_features_R, make_features_Q
from logistic import predict
from metrics import accuracy, confusion_matrix

def main():
    d = prepare()
    y_test = d["y_test"]

    # load the saved parameters and standardisation statistics
    theta_R, theta_Q = np.load(RESULTS / "theta_R.npy"), np.load(RESULTS / "theta_Q.npy")
    mu, sigma = np.load(RESULTS / "mu.npy"), np.load(RESULTS / "sigma.npy")
    muQ, sigmaQ = np.load(RESULTS / "muQ.npy"), np.load(RESULTS / "sigmaQ.npy")
    lin = joblib.load(RESULTS / "best_linear_svm.joblib")
    rbf = joblib.load(RESULTS / "best_rbf_svm.joblib")

    # test features built with the training statistics
    XR_test = make_features_R(d["X_test"], mu, sigma)
    XQ_test = make_features_Q(d["X_test"], muQ, sigmaQ)
    X_test_std = (d["X_test"] - mu) / sigma

    # predictions of the four final models
    preds = {
        "Logistic regression, Set R": predict(XR_test, theta_R),
        "Logistic regression, Set Q": predict(XQ_test, theta_Q),
        f"Linear SVM (C={lin.C:g})": lin.predict(X_test_std),
        f"RBF SVM (C={rbf.C:g}, gamma={rbf.gamma:g})": rbf.predict(X_test_std),
    }

    # accuracy, confusion matrix and missed faults for each model
    rows = []
    for name, y_pred in preds.items():
        cm = confusion_matrix(y_test, y_pred)
        assert cm.sum() == len(y_test), "confusion matrix must add up to the number of test rows"
        print(name, "| accuracy =", round(accuracy(y_test, y_pred), 4), "| entries sum =", cm.sum(), "(test rows =", len(y_test), ")")
        print(cm)
        rows.append({"Model": name, "Accuracy": accuracy(y_test, y_pred), "TN": cm[0, 0],
                     "False alarms": cm[0, 1], "Missed faults": cm[1, 0], "TP": cm[1, 1]})

    # Table 4
    table4 = pd.DataFrame(rows)
    print(table4.to_string(index=False))
    table4.to_csv(RESULTS / "table4_test_results.csv", index=False)

if __name__ == "__main__":
    main()
