import numpy as np
import joblib
import gradio as gr
from pathlib import Path

# results folder sits next to the app folder
RES = Path(__file__).resolve().parent.parent / "results"

# load saved Set Q logistic regression parameters and its standardisation statistics
theta_Q = np.load(RES / "theta_Q.npy")
muQ = np.load(RES / "muQ.npy")
sigmaQ = np.load(RES / "sigmaQ.npy")

# load saved RBF SVM and the Set R standardisation statistics
rbf = joblib.load(RES / "best_rbf_svm.joblib")
mu = np.load(RES / "mu.npy")
sigma = np.load(RES / "sigma.npy")

# sigmoid function for the logistic regression probability
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

# take the six raw values and return both predictions
def predict_fault(ia, ib, ic, va, vb, vc):
    x = np.array([[ia, ib, ic, va, vb, vc]], dtype=float)

    # Set Q: add squares, standardise with training stats, add intercept, then apply theta
    xq = np.c_[x, x ** 2]
    xq = (xq - muQ) / sigmaQ
    xq = np.c_[np.ones(1), xq]
    p = float(sigmoid(xq @ theta_Q)[0])
    lr_text = ("FAULT" if p >= 0.5 else "NO FAULT") + f"  (probability of fault = {p:.3f})"

    # RBF SVM: standardise with training stats, then predict
    svm_pred = int(rbf.predict((x - mu) / sigma)[0])
    svm_text = "FAULT" if svm_pred == 1 else "NO FAULT"

    return lr_text, svm_text

# build the interface with six inputs and two outputs side by side
demo = gr.Interface(
    fn=predict_fault,
    inputs=[gr.Number(label="Ia (A)"), gr.Number(label="Ib (A)"), gr.Number(label="Ic (A)"),
            gr.Number(label="Va (pu)"), gr.Number(label="Vb (pu)"), gr.Number(label="Vc (pu)")],
    outputs=[gr.Textbox(label="Logistic regression (Set Q)"), gr.Textbox(label="RBF SVM")],
    examples=[[-170.472196, 9.219613, 161.252583, 0.054490, -0.659921, 0.605431]],
    title="Three-Phase Transmission Line Fault Detector",
    description="Enter one instant of the line currents and voltages to see both models' predictions.",
)

if __name__ == "__main__":
    demo.launch()