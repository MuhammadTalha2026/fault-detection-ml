# Task 3: choose alpha, train logistic regression on Set R and Set Q, save theta
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from prepare import RESULTS, prepare, make_features_R, make_features_Q
from logistic import cost, fit_logistic, predict
from metrics import accuracy

ALPHAS = [0.01, 0.1, 1, 10]
N_ITERS = 2000

def choose_alpha(X, y, set_name):
    # try every alpha, plot J vs iteration, and keep the largest alpha whose J decreases at every iteration
    fig, ax = plt.subplots(figsize=(7, 4))
    best = None
    for a in ALPHAS:
        _, J = fit_logistic(X, y, a, N_ITERS)
        always_down = bool(np.all(np.diff(J) <= 0))
        print(set_name, "alpha =", a, "| final J =", round(J[-1], 4), "| decreases every iteration:", always_down)
        ax.plot(J, label=f"alpha = {a}")
        if always_down:
            best = a
    ax.set_xlabel("Iteration")
    ax.set_ylabel("J(theta)")
    ax.set_title(f"Cost vs iteration, {set_name}")
    ax.legend()
    fig.savefig(RESULTS / f"fig_cost_{set_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return best

def main():
    d = prepare()
    y_train, y_val = d["y_train"], d["y_val"]

    # Set R features (6 standardised raw features plus intercept)
    XR_train = make_features_R(d["X_train"], d["mu"], d["sigma"])
    XR_val = make_features_R(d["X_val"], d["mu"], d["sigma"])

    # Set Q features (raw plus squares, standardised with training statistics, plus intercept)
    XQraw_train = np.c_[d["X_train"], d["X_train"] ** 2]
    muQ, sigmaQ = XQraw_train.mean(axis=0), XQraw_train.std(axis=0)
    XQ_train = make_features_Q(d["X_train"], muQ, sigmaQ)
    XQ_val = make_features_Q(d["X_val"], muQ, sigmaQ)
    print("Shapes:", XR_train.shape, XQ_train.shape)

    # check: with theta = 0 the cost must be log 2 = 0.693
    print("Cost at theta = 0:", cost(XR_train, y_train, np.zeros(XR_train.shape[1])))

    # choose alpha for each feature set
    alpha_R = choose_alpha(XR_train, y_train, "Set_R")
    alpha_Q = choose_alpha(XQ_train, y_train, "Set_Q")
    print("Chosen alpha: Set R =", alpha_R, ", Set Q =", alpha_Q)

    # train the final models with the chosen alpha and 2000 iterations
    theta_R, J_R = fit_logistic(XR_train, y_train, alpha_R, N_ITERS)
    theta_Q, J_Q = fit_logistic(XQ_train, y_train, alpha_Q, N_ITERS)

    # training and validation accuracy
    accR_train, accR_val = accuracy(y_train, predict(XR_train, theta_R)), accuracy(y_val, predict(XR_val, theta_R))
    accQ_train, accQ_val = accuracy(y_train, predict(XQ_train, theta_Q)), accuracy(y_val, predict(XQ_val, theta_Q))
    print("Set R | train acc:", accR_train, "| val acc:", accR_val, "| final J:", J_R[-1])
    print("Set Q | train acc:", accQ_train, "| val acc:", accQ_val, "| final J:", J_Q[-1])
    print("theta R:", np.round(theta_R, 4))
    print("theta Q:", np.round(theta_Q, 4))

    # Table 2
    table2 = pd.DataFrame({"Feature set": ["Set R", "Set Q"], "Chosen alpha": [alpha_R, alpha_Q],
                           "Final J(theta)": [J_R[-1], J_Q[-1]], "Validation accuracy": [accR_val, accQ_val]})
    print(table2.to_string(index=False))
    table2.to_csv(RESULTS / "table2_logistic_regression.csv", index=False)

    # Table 5: theta of Set Q in the required order
    names = ["theta0 (intercept)", "Ia", "Ib", "Ic", "Va", "Vb", "Vc", "Ia^2", "Ib^2", "Ic^2", "Va^2", "Vb^2", "Vc^2"]
    table5 = pd.DataFrame({"Weight": names, "Value": theta_Q})
    print(table5.to_string(index=False))
    table5.to_csv(RESULTS / "table5_theta_Q.csv", index=False)

    # save theta and the standardisation statistics (needed by test_models.py and the front end)
    np.save(RESULTS / "theta_R.npy", theta_R)
    np.save(RESULTS / "theta_Q.npy", theta_Q)
    np.save(RESULTS / "muQ.npy", muQ)
    np.save(RESULTS / "sigmaQ.npy", sigmaQ)
    np.save(RESULTS / "mu.npy", d["mu"])
    np.save(RESULTS / "sigma.npy", d["sigma"])

if __name__ == "__main__":
    main()
