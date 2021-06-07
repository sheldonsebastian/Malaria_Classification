## README

### Final Report:

https://sheldonsebastian.github.io/Red-Blood-Cell-Classification/

### Directory Structure:

|Path|Description|
|------------|-----------|
|/POCs| Files containing the proof of concepts/experimentation |
|/docs| Files related to report website |
|/input| Training image data  and labels split into train, validation and holdout set|
|/src/common| Common utility functions used by all scripts  |
|/src/model_trainers| Files containing code for training the model using: <br> 1. manual hyperparameter tuning<br> 2. random search hyperparameter tuning <br>3. Optuna (AutoML) hyperparameter tuning  |
|/src/0_preprocess.py | Code to preprocess the image files and split into train-validation-holdout splits |
|/src/1_eda.py | Exploratory Data Analysis Jupyter Notebook |
|/src/2_inference.py | Using the trained models make inference on validation and holdout set |
|requirements.txt | List of all the packages used for this project |


### Steps to replicate project:

1. Download repository
2. To install all required python packages use: <b>conda create --name rbc_classification --file requirements.txt</b>
3. Update BASE_DIR in src/0_manual.py, src/1_random_search.py, src/2_hyper_optimizer.py to current directory on your machine
4. Run src/0_manual.py, src/1_random_search.py, src/2_hyper_optimizer.py to train models and save them in saved_models directory
5. Run src/2_inference_holdout.py to perform inference on holdout(unseen) data.
