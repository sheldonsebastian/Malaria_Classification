# %% --------------------
import numpy as np
from sklearn.model_selection import PredefinedSplit

# %% --------------------
X = np.array([0, 1, 2, 3, 4])
y = np.array([0, 0, 1, 1, 1])

# %% --------------------
# For example, when using a validation set, set the test_fold to 0 for all samples that are part
# of the validation set, and to -1 for all other samples.
# test_fold = [2, 2, 1, 1, -1]
test_fold = np.append(np.full(4, -1), np.full(1, 0))

# %% --------------------
ps = PredefinedSplit(test_fold)

# %% --------------------
print(ps.get_n_splits())

# %% --------------------
print(ps)

# %% --------------------
for train_index, test_index in ps.split():
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# %% --------------------
