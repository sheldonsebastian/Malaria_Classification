# %% --------------------
import sys

# local
# BASE_DIR = "D:/GWU/3 Fall 2020/6203 Machine Learning 2/Exam 1/Repo/Malaria_Classification"
# cerberus
BASE_DIR = "/home/ssebastian94/malaria_classification"

# add home directory to python path
sys.path.append(BASE_DIR)

# %% --------------------

from time import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from src.common.image_generator import CustomImageGen
from src.model_trainers.model_dispatcher import model_v1
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
model = model_v1()
print(model.summary())

# %% --------------------CALLBACKS
# early stopping
es = EarlyStopping(monitor='val_loss', mode="min", verbose=1, patience=200)

# model checkpoint
mc = ModelCheckpoint(f'{BASE_DIR}/src/model_trainers/saved_models/manual.hdf5',
                     monitor="val_loss", mode="min",
                     save_best_only=True, verbose=1)

# %% --------------------TRAINING
print("Training Started")
start_time = time()

# training
history = model.fit(X_train, y_train,
                    batch_size=1024,
                    epochs=5000,
                    verbose=1,
                    callbacks=[es, mc],
                    validation_data=(X_val, y_val),
                    class_weight=class_weights)

end_time = time() - start_time

# %% --------------------
print(f"Training Ended in:: {end_time} seconds")

# %% --------------------
# save the history as image
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.plot(history.history["loss"], label="Training Loss")
plt.legend()
plt.title(f"Manual ANN")
plt.savefig(f"{BASE_DIR}/saved_images/manual_loss")
plt.close()

# %% --------------------
# load best model
best_model = load_model(f'{BASE_DIR}/src/model_trainers/saved_models/manual.hdf5')
