# %% --------------------
import albumentations as A
import cv2
import numpy as np
import tensorflow as tf


# %% --------------------
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class CustomImageGen(tf.keras.utils.Sequence):

    def __init__(self, df, batch_size, base_dir, width=100, height=100, augmentations=[],
                 shuffle=True):
        self.df = df.copy()
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.n = len(self.df)
        self.augmentations = augmentations
        self.base_dir = base_dir
        self.shuffle = shuffle
        self.on_epoch_end()

    def __get_data(self, batches):
        # read the image as array based on image_id
        # https://github.com/afshinea/keras-data-generator/blob/master/my_classes.py

        # X.shape = batch_size x (height x width x channels)
        X_batch = []

        # y.shape = batch_size x targets(one hot encoded)
        y_batch = []

        # augmentations
        if len(self.augmentations) != 0:
            # add resize operation to the transformation pipeline
            self.augmentations.append(
                A.Resize(width=self.width, height=self.height, always_apply=True))
            transforms = A.Compose(self.augmentations)
        else:
            transforms = A.Compose(
                [A.Resize(width=self.width, height=self.height, always_apply=True)])

        for img_id, target in zip(batches["image_id"], batches["target"]):
            # read the image
            img_arr = cv2.imread(self.base_dir + "/" + img_id + ".png")

            # perform augmentation and resizing
            img_arr_transformed = transforms(image=img_arr)

            # normalize the image and append to batched data
            img_arr_normalized = img_arr_transformed["image"] / 255

            X_batch.append(img_arr_normalized)
            y_batch.append(target)

        return np.array(X_batch), np.array(y_batch)

    def __getitem__(self, index):
        batches = self.df[index * self.batch_size: (index + 1) * self.batch_size]
        X, y = self.__get_data(batches)
        return X, y

    def __len__(self):
        # return the number of iterations required to process all the data per epoch
        return self.n // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            self.df.sample(frac=1).reset_index(drop=True)
