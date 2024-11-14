import numpy as np

def calculate_accuracy(y_true, y_pred):
    y_pred = np.round(y_pred)
    accuracy = np.mean(y_true == y_pred)
    return accuracy