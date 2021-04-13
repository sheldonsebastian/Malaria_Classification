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

from time import time
from imblearn.over_sampling import RandomOverSampler
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from src.common.image_generator import CustomImageGen
from src.model_trainers.model_dispatcher import model_v1
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, cohen_kappa_score
import seaborn as sns

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
model = model_v1()
print(model.summary())

# %% --------------------CALLBACKS
# early stopping
es = EarlyStopping(monitor='val_loss', mode="min", verbose=1, patience=100)

# model checkpoint
mc = ModelCheckpoint(f'{BASE_DIR}/src/model_trainers/saved_models/manual_{fold}.hdf5',
                     monitor="val_loss", mode="min",
                     save_best_only=True, verbose=1)

# %% --------------------TRAINING
print("Training Started")
start_time = time()

# training
history = model.fit(X_train, y_train,
                    batch_size=32,
                    epochs=1000,
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
plt.title(f"Manual ANN Fold: {fold}")
plt.savefig(f"{BASE_DIR}/saved_images/manual_loss_{fold}")
plt.close()

# %% --------------------
# load best model
best_model = load_model(f'{BASE_DIR}/src/model_trainers/saved_models/manual_{fold}.hdf5')

# %% --------------------
# Validation Data Evaluation Metrics
print(f"Validation {fold}::")

y_pred = best_model.predict(X_val).argmax(axis=1)
gt = y_val

matrix = confusion_matrix(gt, y_pred)
sns.heatmap(matrix, annot=True, cmap=sns.cm.rocket_r, fmt='g')
plt.title(f"Validation {fold}")
plt.savefig(f"{BASE_DIR}/saved_images/manual_confusion_matrix{fold}")
plt.close()
print(f"Final accuracy on validations set for fold {fold}:", accuracy_score(gt, y_pred))

cohen_score = cohen_kappa_score(gt, y_pred)
f1_score_value = f1_score(gt, y_pred, average="macro")

print(f"Cohen Kappa {fold}: ", cohen_score)
print(f"F1 score: {fold}", f1_score_value)

print(f"Final Mean Score {fold}: ", np.mean([cohen_score, f1_score_value]))
