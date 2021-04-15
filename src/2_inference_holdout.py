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
# read the validation data
valid = pd.read_csv(f"{DATA_DIR}/valid.csv")

# read the holdout data
holdout = pd.read_csv(f"{DATA_DIR}/holdout.csv")

# %% --------------------
valid_gen = CustomImageGen(valid, batch_size=len(holdout), base_dir=DATA_DIR + "/train")
holdout_gen = CustomImageGen(holdout, batch_size=len(holdout), base_dir=DATA_DIR + "/train")

# %% --------------------
X_val, y_val = next(iter(valid_gen))
X_holdout, y_holdout = next(iter(holdout_gen))

# %% --------------------
for approach in ["manual", "random_search", "optuna"]:
    print("-" * 20 + approach + "-" * 20)

    best_model = keras.models.load_model(f'{MODEL_SAVE_DIR}/{approach}.hdf5')

    # Validation Data Evaluation Metrics
    print(f"Validation::")

    y_pred = best_model.predict(X_val).argmax(axis=1)
    gt = y_val

    matrix = confusion_matrix(gt, y_pred)
    sns.heatmap(matrix, annot=True, cmap=sns.cm.rocket_r, fmt='g')
    plt.title(f"Validation")
    plt.savefig(f"{BASE_DIR}/saved_images/{approach}_confusion_matrix_validation")
    plt.close()
    print(f"Final accuracy on validation:", accuracy_score(gt, y_pred))

    cohen_score = cohen_kappa_score(gt, y_pred)
    f1_score_value = f1_score(gt, y_pred, average="macro")

    print(f"Cohen Kappa: ", cohen_score)
    print(f"F1 score: ", f1_score_value)

    print(f"Final Mean Score: ", np.mean([cohen_score, f1_score_value]))

    # Holdout Data Evaluation Metrics
    print(f"Holdout::")

    y_pred = best_model.predict(X_holdout).argmax(axis=1)
    gt = y_holdout

    matrix = confusion_matrix(gt, y_pred)
    sns.heatmap(matrix, annot=True, cmap=sns.cm.rocket_r, fmt='g')
    plt.title(f"Holdout")
    plt.savefig(f"{BASE_DIR}/saved_images/{approach}_confusion_matrix_holdout")
    plt.close()
    print(f"Final accuracy on validation:", accuracy_score(gt, y_pred))

    cohen_score = cohen_kappa_score(gt, y_pred)
    f1_score_value = f1_score(gt, y_pred, average="macro")

    print(f"Cohen Kappa: ", cohen_score)
    print(f"F1 score: ", f1_score_value)

    print(f"Final Mean Score: ", np.mean([cohen_score, f1_score_value]))
    print("\n")
