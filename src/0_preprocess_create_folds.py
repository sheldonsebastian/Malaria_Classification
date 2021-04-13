# %% --------------------
import os

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# %% --------------------PREPROCESSING
# file name, target,
input_data = []
for f in os.listdir("../input/train"):
    if f[-3:] == "txt":
        with open(f"../input/train/{f}") as s:
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
holdout.to_csv("../input/holdout.csv", index=False)

# %% --------------------CREATE 5 FOLDS
# create new column
train["fold"] = -1

# shuffle the dataframe
train = train.sample(frac=1, random_state=42).reset_index(drop=True)

# instantiate stratified kfolds
kf = StratifiedKFold(n_splits=5)

# add the fold to train column
for fold, (_, val_index) in enumerate(kf.split(X=train, y=train["target"])):
    train.loc[val_index, 'fold'] = fold

# %% --------------------
print(train["fold"].value_counts())

# %% --------------------
train.to_csv("../input/train.csv", index=False)
