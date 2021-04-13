# %% --------------------
import sys

# local
BASE_DIR = "D:/GWU/3 Fall 2020/6203 Machine Learning 2/Exam 1/Repo/Malaria_Classification"
# cerberus
# BASE_DIR = "/home/ssebastian94/malaria_classification"

# add home directory to python path
sys.path.append(BASE_DIR)

# %% --------------------
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, cohen_kappa_score, f1_score, accuracy_score
from tensorflow import keras
from src.common.image_generator import CustomImageGen

# %% --------------------DIRECTORIES
DATA_DIR = f"{BASE_DIR}/input"
MODEL_SAVE_DIR = f"{BASE_DIR}/src/model_trainers/saved_models"
IMAGE_SAVE_DIR = f"{BASE_DIR}/saved_images"

# %% --------------------
# approach = "manual"
# approach = "random_search"
approach = "optuna"

# %% --------------------
# add models from different folds into a list
estimators = [keras.models.load_model(f'{MODEL_SAVE_DIR}/{approach}_{fold}.hdf5') for fold in
              [0]]
# estimators = [keras.models.load_model(f'{BASE_DIR}/src/model_trainers/saved_models/mlp_ssebastian94.hdf5', compile=False)]

# %% --------------------
# read the holdout data
holdout = pd.read_csv(f"{DATA_DIR}/holdout.csv")

# %% --------------------
holdout_gen = CustomImageGen(holdout, batch_size=len(holdout), base_dir=DATA_DIR + "/train")

# %% --------------------
X_holdout, y_holdout = next(iter(holdout_gen))

# %% --------------------
# predict for all folds
all_fold_predictions = [fold_model.predict(X_holdout) for fold_model in estimators]

# take average for all folds
ensembled_outputs = np.mean(all_fold_predictions, axis=0)

# %% --------------------Evaluate ensembled models based on
print("Holdout Evaluation::")
# Holdout Data Evaluation Metrics
y_pred = ensembled_outputs.argmax(axis=1)
gt = y_holdout

matrix = confusion_matrix(gt, y_pred)
sns.heatmap(matrix, annot=True, cmap=sns.cm.rocket_r, fmt='g')
plt.title("Holdout Confusion Matrix")
plt.show()

print("Final accuracy on holdout set:", accuracy_score(gt, y_pred))

cohen_score = cohen_kappa_score(gt, y_pred)
f1_score_value = f1_score(gt, y_pred, average="macro")

print("Cohen Kappa: ", cohen_score)
print("F1 score: ", f1_score_value)

print("Final Mean Score: ", np.mean([cohen_score, f1_score_value]))
