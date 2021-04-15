# %% --------------------
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam


# %% --------------------
# model v2: Manually finding optimum model
def model_v1():
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(100, 100, 3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(512, activation="relu"))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(512, activation="relu"))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(512, activation="relu"))
    model.add(keras.layers.Dropout(0.4))
    # adding softmax layer to convert scores to probabilities for multiclass
    model.add(keras.layers.Dense(4, activation="softmax"))

    model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(0.00001),
                  metrics=['sparse_categorical_accuracy'])

    return model


# %% --------------------
# model v2: Find best Hyper params using randomized search
def model_v2(n_hidden=1, n_neurons=50, learning_rate=1e-3, activation_function="relu",
             dropout_prob=0.4):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(100, 100, 3)))

    for layer in range(n_hidden):
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(n_neurons, activation=activation_function))
        model.add(keras.layers.Dropout(dropout_prob))

    # adding softmax layer to convert scores to probabilities for multiclass
    model.add(keras.layers.Dense(4, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(learning_rate),
                  metrics=['sparse_categorical_accuracy'])

    return model


# %% --------------------
# model v3: Find best Hyper params using optuna
# https://github.com/optuna/optuna/blob/master/examples/tfkeras/tfkeras_integration.py
def model_v3(trial):
    # params
    n_hidden = trial.suggest_int("n_hidden", 1, 5)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-6, 1e-3)
    activation_function = trial.suggest_categorical("activation_function", ["relu", "selu", "elu"])
    dropout_prob = trial.suggest_categorical("dropout_prob", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(100, 100, 3)))

    for layer in range(n_hidden):
        model.add(keras.layers.BatchNormalization())
        n_neurons = trial.suggest_int(f"n_neurons_{layer}", 1, 2048)
        model.add(keras.layers.Dense(n_neurons, activation=activation_function))
        model.add(keras.layers.Dropout(dropout_prob))

    # adding softmax layer to convert scores to probabilities for multiclass
    model.add(keras.layers.Dense(4, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(learning_rate),
                  metrics=['sparse_categorical_accuracy'])

    return model


# %% --------------------
def create_model(best_found_params):
    n_hidden = best_found_params["n_hidden"]
    learning_rate = best_found_params["learning_rate"]
    activation_function = best_found_params["activation_function"]
    dropout_prob = best_found_params["dropout_prob"]

    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=(100, 100, 3)))

    for layer in range(n_hidden):
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(best_found_params[f"n_neurons_{layer}"],
                                     activation=activation_function))
        model.add(keras.layers.Dropout(dropout_prob))

    # adding softmax layer to convert scores to probabilities for multiclass
    model.add(keras.layers.Dense(4, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(learning_rate),
                  metrics=['sparse_categorical_accuracy'])

    return model
