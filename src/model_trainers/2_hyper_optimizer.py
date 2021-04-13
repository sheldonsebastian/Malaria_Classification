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
import os
import pickle
from imblearn.over_sampling import RandomOverSampler
import albumentations as A
import numpy as np
import pandas as pd
from sklearn.utils import class_weight
from src.common.image_generator import CustomImageGen
import tensorflow as tf
from src.model_trainers.model_dispatcher import model_v3, create_model
from optuna.integration import TFKerasPruningCallback
import optuna
from optuna.trial import TrialState
from src.common.utilities import evaluation_metric

# %% --------------------DIRECTORIES
DATA_DIR = f"{BASE_DIR}/input"
IMAGE_SAVE_DIR = f"{BASE_DIR}/saved_images"

# %% --------------------
fold = 0

# find fold when using cluster computer
if "SLURM_ARRAY_TASK_ID" in os.environ.keys():
    fold = int(os.environ["SLURM_ARRAY_TASK_ID"])

print(f"Fold used:{fold}")

# %% --------------------
# read all data
df = pd.read_csv(DATA_DIR + "/train.csv")

# %% --------------------
# get train and validation data based on training fold
train = df[df["fold"] != fold]
val = df[df["fold"] == fold]

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
                               A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.1,
                                             p=0.5),
                               A.Flip(p=0.5),
                               A.ShiftScaleRotate(shift_limit=0.0625,
                                                  scale_limit=0.1,
                                                  rotate_limit=30,
                                                  p=0.3),
                               A.GaussianBlur(p=0.3)
                           ])

# create keras validation pipeline to augment data and resize image
val_gen = CustomImageGen(val, batch_size=len(val), base_dir=DATA_DIR + "/train")

# %% --------------------
# get X_train, y_train
X_train, y_train = next(iter(train_gen))

# get X_val, y_val
X_val, y_val = next(iter(val_gen))

# %% --------------------
print(np.unique(y_train, return_counts=True))

# %% --------------------
# (X_train == X_train.reshape(X_train.shape[0], 50*50*3).reshape((X_train.shape[0],50,50,3))).all()
X_train = X_train.reshape(X_train.shape[0], 50 * 50 * 3)

# %% --------------------
# https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html#imblearn.over_sampling.RandomOverSampler
oversampler = RandomOverSampler(random_state=42)
X_train, y_train = oversampler.fit_resample(X_train, y_train)

# %% --------------------
print(np.unique(y_train, return_counts=True))

# %% --------------------
X_train = X_train.reshape(X_train.shape[0], 50, 50, 3)


# %% --------------------
# define objective function to maximize
# https://github.com/optuna/optuna/blob/master/examples/tfkeras/tfkeras_integration.py
def objective(trial):
    # Clear clutter from previous TensorFlow graphs.
    tf.keras.backend.clear_session()

    # Create tf.keras model instance.
    model = model_v3(trial)

    # Create callbacks for early stopping and pruning.
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=300),
        TFKerasPruningCallback(trial, "val_loss"),
    ]

    batch_size = trial.suggest_int("batch_size", 32, 512)

    # Train model.
    model.fit(
        X_train,
        y_train,
        epochs=1000,
        batch_size=batch_size,
        verbose=1,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weights
    )

    # compute evaluation metric on validation dataset
    preds = model.predict(X_val).argmax(axis=1)

    return evaluation_metric(y_val, preds)


# %% --------------------
study = optuna.create_study(
    direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=2)
)

study.optimize(objective, n_trials=200, timeout=1000)

# %% --------------------
# pickle study
with open(f"{BASE_DIR}/src/model_trainers/hyperopt_metrics/study_{fold}", 'wb') as file:
    pickle.dump(study, file)


# %% --------------------
def show_result(study):
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
show_result(study)

# %% --------------------
# recreate the best model and save it
best_found = study.best_trial

n_hidden = best_found.params["n_hidden"]
n_neurons = best_found.params["n_neurons"]
learning_rate = best_found.params["learning_rate"]
activation_function = best_found.params["activation_function"]
dropout_prob = best_found.params["dropout_prob"]

# %% --------------------
best_model = create_model(n_hidden, activation_function, n_neurons, dropout_prob, learning_rate)

# %% --------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=200),
]

# %% --------------------
best_model.fit(
    X_train,
    y_train,
    epochs=1000,
    batch_size=best_found.params["batch_size"],
    verbose=1,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    class_weight=class_weights
)

# %% --------------------
tf.keras.models.save_model(best_model,
                           f"{BASE_DIR}/src/model_trainers/saved_models/optuna_{fold}.hdf5")
