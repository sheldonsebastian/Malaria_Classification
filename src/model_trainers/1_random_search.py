# %% --------------------
import sys

# local
# BASE_DIR = "D:/GWU/3 Fall 2020/6203 Machine Learning 2/Exam 1/Repo/Malaria_Classification"
# cerberus
BASE_DIR = "/home/ssebastian94/malaria_classification"

# add home directory to python path
sys.path.append(BASE_DIR)

# %% --------------------

from scipy.stats import reciprocal
from src.model_trainers.model_dispatcher import model_v2
from sklearn.metrics import make_scorer
from tensorflow.keras.models import save_model
import tensorflow
from sklearn.model_selection import RandomizedSearchCV
from src.common.utilities import evaluation_metric, get_train_val_ps
from time import time
import numpy as np
import pandas as pd
from sklearn.utils import class_weight
from src.common.image_generator import CustomImageGen
import albumentations as A
import os

# %% --------------------DIRECTORIES
DATA_DIR = f"{BASE_DIR}/input"

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
# create predefined splits
X_train_val, y_train_val, ps = get_train_val_ps(X_train, y_train, X_val, y_val)

# %% --------------------
model = tensorflow.keras.wrappers.scikit_learn.KerasClassifier(model_v2)

# %% --------------------
param_distribs = {
    "n_hidden": [1, 2, 3, 4, 5],
    "n_neurons": np.arange(1, 2048).tolist(),
    "learning_rate": reciprocal(3e-4, 3e-2).rvs(1000).tolist(),
    "activation_function": ["relu", "selu", "elu"],
    "dropout_prob": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "batch_size": [32, 64, 128, 256, 512, 1024],
    "epochs": [300, 600, 1000]
}
# Randomized Search
# https://stackoverflow.com/questions/48127550/early-stopping-with-keras-and-sklearn-gridsearchcv-cross-validation
rnd_search_cv = RandomizedSearchCV(model, param_distribs, verbose=1,
                                   scoring=make_scorer(evaluation_metric),
                                   n_iter=25,
                                   cv=ps,
                                   return_train_score=True)

# %% --------------------TRAINING
print("Training Started")
start_time = time()

rnd_search_cv.fit(X_train_val, y_train_val,
                  verbose=1,
                  class_weight=class_weights)

end_time = time() - start_time

# %% --------------------
print(f"Training Ended in:: {end_time} seconds")

# %% --------------------
# Sort cv_results in ascending order of 'rank_test_score' and 'std_test_score'
cv_results = pd.DataFrame.from_dict(rnd_search_cv.cv_results_).sort_values(
    by=['rank_test_score', 'std_test_score'])

# Get the important columns in cv_results
important_columns = ['rank_test_score',
                     'mean_test_score',
                     'std_test_score',
                     'mean_train_score',
                     'std_train_score',
                     'mean_fit_time',
                     'std_fit_time',
                     'mean_score_time',
                     'std_score_time']

# Move the important columns ahead
cv_results = cv_results[
    important_columns + sorted(list(set(cv_results.columns) - set(important_columns)))]

# Write cv_results file
cv_results.to_csv(
    path_or_buf=f"{BASE_DIR}/src/model_trainers/randomized_metrics/randomized_search_logs.csv",
    index=False)

# %% --------------------
print(rnd_search_cv.best_params_)

# %% --------------------
print(rnd_search_cv.best_score_)

# %% --------------------
# get best model
best_model = rnd_search_cv.best_estimator_.model

# create save model directory
os.makedirs(f'{BASE_DIR}/src/model_trainers/saved_models/', exist_ok=True)

# save best model
save_model(best_model, f"{BASE_DIR}/src/model_trainers/saved_models/random_search.hdf5")
