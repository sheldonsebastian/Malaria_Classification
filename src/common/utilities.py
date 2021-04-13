# %% --------------------
import numpy as np
from sklearn.metrics import f1_score, cohen_kappa_score
from sklearn.model_selection import PredefinedSplit


# %% --------------------
def evaluation_metric(y_true, y_pred):
    f1_val = f1_score(y_true, y_pred, average="macro")
    cohen_val = cohen_kappa_score(y_true, y_pred)

    return np.mean([f1_val, cohen_val])


# %% --------------------
def get_train_val_ps(X_train, y_train, X_val, y_val):
    """
    Get the:
    feature matrix and target vector in the combined training and validation data
    target vector in the combined training and validation data
    PredefinedSplit

    Parameters
    ----------
    X_train : the feature matrix in the training data
    y_train : the target vector in the training data
    X_val : the feature matrix in the validation data
    y_val : the target vector in the validation data

    Return
    ----------
    The feature matrix in the combined training and validation data
    The target vector in the combined training and validation data
    PredefinedSplit
    """

    # Combine the feature matrix in the training and validation data
    X_train_val = np.vstack((X_train, X_val))

    # Combine the target vector in the training and validation data
    y_train_val = np.vstack((y_train.reshape(-1, 1), y_val.reshape(-1, 1))).reshape(-1)

    # Get the indices of training and validation data
    train_val_idxs = np.append(np.full(X_train.shape[0], -1), np.full(X_val.shape[0], 0))

    # The PredefinedSplit
    ps = PredefinedSplit(train_val_idxs)

    return X_train_val, y_train_val, ps
