# %% --------------------
import sys

# local
BASE_DIR = "D:/GWU/3 Fall 2020/6203 Machine Learning 2/Exam 1/Repo/Malaria_Classification"
# cerberus
# BASE_DIR = "/home/ssebastian94/malaria_classification"

# add home directory to python path
sys.path.append(BASE_DIR)

# %% --------------------
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# %% --------------------PREPROCESSING
# file name, target,
input_data = []
for f in os.listdir(f"{BASE_DIR}/input/train"):
    if f[-3:] == "txt":
        with open(f"{BASE_DIR}/input/train/{f}") as s:
            target = s.read()
        input_data.append({"image_id": f[:-4], "target": target})

# convert list to dataframe
df = pd.DataFrame(input_data, columns=["image_id", "target"])

# %% --------------------
le = LabelEncoder()
df["target"] = le.fit_transform(df["target"])

# %% --------------------
print(le.classes_)  # 'red blood cell'=0, 'ring'=1, 'schizont'=2, 'trophozoite'=3

# %% --------------------
# 90-10 split
train, holdout = train_test_split(df, test_size=0.1, stratify=df["target"], random_state=42)

# %% --------------------
print("Train target distribution percentage:")
print(train["target"].value_counts() / len(train) * 100)
print()
print("Holdout target distribution percentage:")
print(holdout["target"].value_counts() / len(holdout) * 100)

# %% --------------------
holdout.to_csv(f"{BASE_DIR}/input/holdout.csv", index=False)

# %% --------------------
# 80-20 split
train, valid = train_test_split(train, test_size=0.2, stratify=train["target"], random_state=42)

# %% --------------------
print("Train target distribution percentage:")
print(train["target"].value_counts() / len(train) * 100)
print()
print("Validation target distribution percentage:")
print(valid["target"].value_counts() / len(valid) * 100)

# %% --------------------
train.to_csv(f"{BASE_DIR}/input/train.csv", index=False)
valid.to_csv(f"{BASE_DIR}/input/valid.csv", index=False)
