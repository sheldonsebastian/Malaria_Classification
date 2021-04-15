# https://github.com/optuna/optuna/blob/master/examples/keras/keras_simple.py
# https://www.analyticsvidhya.com/blog/2020/11/hyperparameter-tuning-using-optuna/
# %% --------------------
import sys

# local
# BASE_DIR = "D:/GWU/3 Fall 2020/6203 Machine Learning 2/Exam 1/Repo/Malaria_Classification"
# cerberus
BASE_DIR = "/home/ssebastian94/malaria_classification"

# add home directory to python path
sys.path.append(BASE_DIR)

# %% --------------------
import pickle
import tensorflow as tf
from src.model_trainers.model_dispatcher import model_v3, create_model
from optuna.integration import TFKerasPruningCallback
import optuna
from optuna.trial import TrialState
import numpy as np
import pandas as pd
from sklearn.utils import class_weight
from src.common.image_generator import CustomImageGen
import albumentations as A

# %% --------------------DIRECTORIES
DATA_DIR = f"{BASE_DIR}/input"
IMAGE_SAVE_DIR = f"{BASE_DIR}/saved_images"

# %% --------------------
# read all data
train = pd.read_csv(DATA_DIR + "/train.csv")
valid = pd.read_csv(DATA_DIR + "/valid.csv")
holdout = pd.read_csv(DATA_DIR + "/holdout.csv")

# %% --------------------
# 'red blood cell':0,  'ring':1, 'schizont':2, 'trophozoite':3
# initialize class_weights in loss to handle class imbalance using training data
class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(train["target"]),
                                                  train["target"])

# %% --------------------
# create keras train pipeline to augment data and resize image
# https://albumentations.ai/docs/api_reference/augmentations/
train_gen = CustomImageGen(train, batch_size=len(train), base_dir=DATA_DIR + "/train",
                           augmentations=[
                               A.RandomBrightnessContrast(),
                               A.Flip()
                           ])

# create keras validation pipeline to resize image
val_gen = CustomImageGen(valid, batch_size=len(valid), base_dir=DATA_DIR + "/train")

# create keras holdout pipeline to resize image
holdout_gen = CustomImageGen(holdout, batch_size=len(holdout), base_dir=DATA_DIR + "/train")

# %% --------------------
# get X_train, y_train
X_train, y_train = next(iter(train_gen))

# get X_val, y_val
X_val, y_val = next(iter(val_gen))

# get X_holdout, y_holdout
X_holdout, y_holdout = next(iter(holdout_gen))

# %% --------------------
print(np.unique(y_train, return_counts=True))


# %% --------------------
# define objective function to maximize
# https://github.com/optuna/optuna/blob/master/examples/tfkeras/tfkeras_integration.py
def objective(trial):
    # Clear clutter from previous TensorFlow graphs.
    tf.keras.backend.clear_session()

    monitor = "val_loss"

    # Create tf.keras model instance.
    model = model_v3(trial)

    # Create callbacks for early stopping and pruning.
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=300),
        TFKerasPruningCallback(trial, monitor),
    ]

    batch_size = trial.suggest_int("batch_size", 32, 1024)

    # Train model.
    history = model.fit(
        X_train,
        y_train,
        epochs=5000,
        batch_size=batch_size,
        verbose=1,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weights
    )

    return history.history[monitor][-1]


# %% --------------------
study = optuna.create_study(
    direction="minimize", pruner=optuna.pruners.MedianPruner(n_startup_trials=100)
)

study.optimize(objective, n_trials=300, timeout=1000)

# %% --------------------
# pickle study
with open(f"{BASE_DIR}/src/model_trainers/hyperopt_metrics/study", 'wb') as file:
    pickle.dump(study, file)

# %% --------------------
pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# %% --------------------
# recreate the best model and save it
best_found = study.best_trial

# %% --------------------
best_model = create_model(best_found.params)

# %% --------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=300),
]

# %% --------------------
best_model.fit(
    X_train,
    y_train,
    epochs=5000,
    batch_size=best_found.params["batch_size"],
    verbose=1,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    class_weight=class_weights
)

# %% --------------------
tf.keras.models.save_model(best_model,
                           f"{BASE_DIR}/src/model_trainers/saved_models/optuna.hdf5")
