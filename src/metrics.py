# Task 5.1: metrics written by hand
import numpy as np

def accuracy(y_true, y_pred):
    # fraction of predictions that match the true labels
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true, y_pred):
    # returns [[TN, false alarm], [missed fault, TP]]
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return np.array([[tn, fp], [fn, tp]])
