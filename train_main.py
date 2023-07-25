import pandas as pd
import numpy as np
import pickle
from predictive_models.predictive_models_functions import test_adboc_model
from test_evaluate_functions.test_evaluate_functions import calculate_accuracies, test_model, variable_importance, return_variable_importance
from training_functions.training_functions import calculate_binary_variables, split, split_x_y, train_ada_boost, split_easy_and_sudden_errors
import argparse
import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings("ignore", category=DataConversionWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--di", "--raw_data_folders_directory", help="Provide the local directory on your machine of the folders raw_data and meteo_raw_data. Example: /home/leibniz/Desktop/IHMAN")
parser.add_argument("--de", "--device", help="Indicate if you want to train the models for lights or for eboxes")
parser.add_argument("--st", "--store", help="Set to True if you want to store the models that you have trained", default=False)
parser.add_argument("--rew", "--rewrite", help="Set to True if you want to rewrite the existing models in the folder predictive_models", default=False)
parser.add_argument("--mo", "--model", help="In case that you choose lights you have to specify if you want to train the model using the reading columns or not. In case that you choose to train without the readings two models will be trained to be able to do predictions with the adboc predictor. In the other case we just train one model")

args = parser.parse_args()

data_dir = args.di
device = args.de
store = args.st
rewrite = args.rew
model_type = args.mo

if store == "True":
    store = True
if store == "False":
    store = False
if rewrite == "True":
    rewrite = True
if rewrite == "False":
    rewrite = False

if device not in ["lights", "eboxes"]:
    raise Exception("Error: Device is a mandatory argument and has to be either 'lights' or 'eboxes'. Write '--de eboxes' or '--de lights'")
if (device == "lights") & (model_type is None):
    raise Exception("If you choose to train a light breakdown predictor it is mandatory to specify the model type with the command '--mo adboc' for training without the readings or '--mo default' for training with the readings")

if device == "lights":
    # read the data from the output of the preprocessing_main.py
    print("Reading data...")
    dff = pd.read_csv(f"{data_dir}/preprocessing_results/out_lights.csv")
    print("Data reading done!")

    # In the case of lights we have two possibilities. Train two models for using the adboc predictor which uses the readings  or
    # training one model using the readings. Which way we go is specified in the model_type variable passed by the user

    if model_type == "default":
        # Define the dictionary of hyperparameters. Note that this are the hyperparameters that have worked better with the datasets
        # of Canyelles, Illora and Mejorada. For retraining you might want to do a grid search of parameters for your municipality to
        # get the best model so feel free to modify this code:
        hparams = {
            "max_depth_tree": 1,
            "n_estimators": 30,
            "lr": 0.5,
            "prob_threshold": 0.47
        }
        # Remove the rows with nan readings:
        df = dff.loc[~dff["ActivePeak_current_week"].isna()]

        # Drop some usless columns for the model:
        drop_cols = [
            col for col in df.columns if
                (col in ["lat", "lon"]) | 
                (col == "Unnamed: 0") |
                (col.startswith("week")) |
                (col == "type") |
                (col == "ebox_id") |
                (col == "location") |
                (col == "id")
        ]
        df = df.drop(drop_cols, axis=1)

        # Interpolate some left missing values:
        print("Interpolating remaining nan values...")
        df = df.fillna(df.mean(numeric_only=True))

        df = calculate_binary_variables(df)
        # Preprocessing and split:
        df["current_week"] = pd.to_datetime(df["current_week"])
        print("Split information:")
        train, validation, test = split(df, n_weeks=77)

        cols_to_train = df.drop(["current_week", "hours_next_four_weeks", "error_next_four_weeks", "hours_week+1", "hours_week+2", "hours_week+3", "hours_week+4"], axis=1).columns
        x_train, y_train = split_x_y(train, cols_to_train)
        x_validation, y_validation = split_x_y(validation, cols_to_train)
        x_test, y_test = split_x_y(test, cols_to_train)

        print("Training model...")

        ada_model = train_ada_boost(
            max_depth_tree = hparams["max_depth_tree"],
            n_estimators = hparams["n_estimators"],
            lr = hparams["lr"],
            x = x_train,
            y = y_train
        )

        print("Validation results:")
        print(
            test_model(
                model = ada_model,
                x = x_validation, 
                y = y_validation,
                prob_threshold = hparams["prob_threshold"]
            )
        )

        print("Test results:")
        print(
            test_model(
                model = ada_model,
                x = x_test, 
                y = y_test,
                prob_threshold = hparams["prob_threshold"]
            )
        )

        print("Variable importance of the trained model:")

        feature_importances = return_variable_importance(ada_model, cols_to_train)
        print(feature_importances.loc[feature_importances["importance"] != 0])

        # Store the model if the user wants to:
        if store:
            if rewrite:
                with open("predictive_models/ada_model_readings.pk1", "wb") as file:
                    pickle.dump(ada_model, file)

                # Save the probability thresholds:
                with open("predictive_models/ada_prob_readings.pk1", "wb") as file:
                    pickle.dump({"prob_ada_model": hparams["prob_threshold"]}, file)
                
                print("Model stored successfully!")
            else:
                # If we do not want to rewrite the last models we will just add the prefix "new" at the end

                with open("predictive_models/ada_model_readings_new.pk1", "wb") as file:
                    pickle.dump(ada_model, file)

                # Save the probability thresholds:
                with open("predictive_models/ada_prob_readings_new.pk1", "wb") as file:
                    pickle.dump({"prob_ada_model": hparams["prob_threshold"]}, file)

                print("Model stored successfully!")

    if model_type == "adboc":
        # Define the dictionary of hyperparameters. Note that this are the hyperparameters that have worked better with the datasets
        # of Canyelles, Illora and Mejorada. For retraining you might want to do a grid search of parameters for your municipality to
        # get the best model so feel free to modify this code:
        hparams = {
            "max_depth_tree": 1,
            "n_estimators": 30,
            "lr": 0.5,
            "prob_threshold": 0.45
        }
        # hparams for the sudden model:
        hparams_sudden = {
            "max_depth_tree": 3,
            "n_estimators": 10,
            "lr": 0.7,
            "prob_threshold": 0.48
        }

        # We will drop all the columns readings and lon and lat:
        drop_cols = [
                        col for col in dff.columns if 
                        (col.startswith("power")) | (col.startswith("Active")) | (col.startswith("Reactive") | 
                        (col in ["lat", "lon"])) | 
                        (col == "Unnamed: 0") |
                        (col.startswith("week")) |
                        (col == "type") |
                        (col == "ebox_id") |
                        (col == "location") |
                        (col == "id")
                    ]
        df = dff.drop(drop_cols, axis=1)
        # Interpolate with the mean in case it is necessary:
        print("Interpolating remaining nan values...")
        df = df.fillna(df.mean(numeric_only=True))
        df = calculate_binary_variables(df)
        # Preprocessing and split:
        df["current_week"] = pd.to_datetime(df["current_week"])
        print("Split information whole dataset:")
        train, validation, test = split(df, n_weeks=100)

        cols_to_train = df.drop(["current_week", "hours_next_four_weeks", "error_next_four_weeks", "hours_week+1", "hours_week+2", "hours_week+3", "hours_week+4"], axis=1).columns
        x_train, y_train = split_x_y(train, cols_to_train)
        x_validation, y_validation = split_x_y(validation, cols_to_train)
        x_test, y_test = split_x_y(test, cols_to_train)

        print("Training ada_model for the whole dataset...")

        ada_model = train_ada_boost(
            max_depth_tree = hparams["max_depth_tree"],
            n_estimators = hparams["n_estimators"],
            lr = hparams["lr"],
            x  = x_train,
            y = y_train
        )

        print("Validation results ada_model:")
        print(
            test_model(
                model = ada_model,
                x = x_validation, 
                y = y_validation,
                prob_threshold = hparams["prob_threshold"]
            )
        )

        print("Test results ada_model:")
        print(
            test_model(
                model = ada_model,
                x = x_test, 
                y = y_test,
                prob_threshold = hparams["prob_threshold"]
            )
        )

        print("Variable importance of the ada_model trained with the whole dataset:")

        feature_importances = return_variable_importance(ada_model, cols_to_train)
        print(feature_importances.loc[feature_importances["importance"] != 0])

        # Train another model with the sudden errors:
        test_sudden, test_easy = split_easy_and_sudden_errors(test)

        x_test_sudden, y_test_sudden = split_x_y(test_sudden, cols_to_train)
        x_test_easy, y_test_easy = split_x_y(test_easy, cols_to_train)

        train_sudden, train_easy = split_easy_and_sudden_errors(train)
        validation_sudden, validation_easy = split_easy_and_sudden_errors(validation)

        # The dataset train_sudden has just rows with the column "error_next_four_weeks" = "Yes" so we will have to add
        # some "No" rows in order to the model to train. To do this we will simply add a random sample of length len(train_sudden)
        # of "No" rows to the dataset train_sudden.
        train_sudden = pd.concat(
            [
                train.loc[train["error_next_four_weeks"] == "No"].sample(len(train_sudden)),
                train_sudden
            ],
            sort=True
        ).sample(frac=1.0, random_state=42)

        validation_sudden = pd.concat(
            [
                validation.loc[validation["error_next_four_weeks"] == "No"].sample(len(validation_sudden)),
                validation_sudden
            ],
            sort=True
        ).sample(frac=1.0, random_state=42)

        x_train_sudden, y_train_sudden = split_x_y(train_sudden, cols_to_train)
        x_validation_sudden, y_validation_sudden = split_x_y(validation_sudden, cols_to_train)

        print("Training ada_sudden_model for the sudden errors dataset:")

        ada_sudden_model = train_ada_boost(
            max_depth_tree = hparams_sudden["max_depth_tree"],
            n_estimators = hparams_sudden["n_estimators"],
            lr = hparams_sudden["lr"],
            x  = x_train_sudden,
            y = y_train_sudden
        )
        
        print("Validation results ada_sudden_model on sudden errors dataset:")

        print(
            test_model(
                model = ada_sudden_model, 
                x = x_validation_sudden, 
                y = y_validation_sudden,
                prob_threshold = hparams_sudden["prob_threshold"]
            )
        )

        print("Test results ada_sudden_model on sudden errors dataset:")

        print(
            test_model(
                model = ada_sudden_model, 
                x = x_test_sudden, 
                y = y_test_sudden,
                prob_threshold = hparams_sudden["prob_threshold"]
            )
        )

        print("Test results ada_sudden_model on whole data:")
        print(
            test_model(
                model = ada_sudden_model,
                x = x_test, 
                y = y_test,
                prob_threshold = 0.48
            )  
        )

        print("Variable importance of the ada_sudden_model trained with the sudden errors:")

        feature_importances_sudden = return_variable_importance(ada_sudden_model, cols_to_train)
        print(feature_importances_sudden.loc[feature_importances_sudden["importance"] != 0])

        print("Calculating predictions for the test of the whole dataset... (This may take a while)")
        acc, _ = test_adboc_model(
            x_test,
            y_test,
            ada_model,
            ada_sudden_model,
            hparams["prob_threshold"],
            hparams_sudden["prob_threshold"]
        )
        print("Test results adboc predictor on whole data:")
        print(acc)

        # Store the model if the user wants to:
        if store:
            if rewrite:
                with open("predictive_models/ada_model.pk1", "wb") as file:
                    pickle.dump(ada_model, file)

                with open("predictive_models/ada_sudden_model.pk1", "wb") as file:
                    pickle.dump(ada_sudden_model, file)

                # Save the probability thresholds:
                with open("predictive_models/ada_prob.pk1", "wb") as file:
                    pickle.dump(
                        {"prob_ada_model": 0.45, "prob_sudden_model": 0.48},
                        file
                    )
                
                print("Models stored successfully!")

            else:
                # If we do not want to rewrite the last models we will just add the prefix "new" at the end

                with open("predictive_models/ada_model_new.pk1", "wb") as file:
                    pickle.dump(ada_model, file)

                with open("predictive_models/ada_sudden_model_new.pk1", "wb") as file:
                    pickle.dump(ada_sudden_model, file)

                # Save the probability thresholds:
                with open("predictive_models/ada_prob_new.pk1", "wb") as file:
                    pickle.dump(
                        {"prob_ada_model": 0.45, "prob_sudden_model": 0.48},
                        file
                    )
                
                print("Models stored successfully")

if device == "eboxes":
    # read the data from the output of the preprocessing_main.py
    print("Reading data...")
    dff = pd.read_csv("/home/leibniz/Desktop/IHMAN/preprocessing_results/out_eboxes.csv")
    print("Data reading done!")

    # In the case of eboxes we have seen that using the adboc predictor does not give us good results. So in the eboxes case
    # we have one option to train a model

    hparams = {
        "max_depth_tree": 1,
        "n_estimators": 10,
        "lr": 0.5,
        "prob_threshold": 0.48
    }

    # We will drop all the columns readings and lon and lat:
    drop_cols = [
                    col for col in dff.columns if 
                    (col.startswith("power")) | (col.startswith("Active")) | (col.startswith("Reactive") | 
                    (col in ["lat", "lon"])) | 
                    (col == "Unnamed: 0") |
                    (col.startswith("week")) |
                    (col == "type") |
                    (col == "ebox_id") |
                    (col == "location") |
                    (col == "id")
                ]
    df = dff.drop(drop_cols, axis=1)

    # Interpolate the remaining nana values:
    print("Interpolating remaining nan values...")
    df = df.fillna(df.mean(numeric_only=True))

    print("Split information:")
    df = calculate_binary_variables(df)
    # Preprocessing and split:
    df["current_week"] = pd.to_datetime(df["current_week"])
    train, validation, test = split(df, n_weeks=80)

    cols_to_train = df.drop(["current_week", "hours_next_four_weeks", "error_next_four_weeks", "hours_week+1", "hours_week+2", "hours_week+3", "hours_week+4"], axis=1).columns
    x_train, y_train = split_x_y(train, cols_to_train)
    x_validation, y_validation = split_x_y(validation, cols_to_train)
    x_test, y_test = split_x_y(test, cols_to_train)

    print("Training model...")
    ada_model = train_ada_boost(
        max_depth_tree = hparams["max_depth_tree"],
        n_estimators = hparams["n_estimators"],
        lr = hparams["lr"],
        x  = x_train,
        y = y_train
    )
    
    print("Validation results:")
    print(test_model(
        model = ada_model,
        x = x_validation, 
        y = y_validation,
        prob_threshold = hparams["prob_threshold"]
    ))

    print("Test results:")
    print(test_model(
        model = ada_model,
        x = x_test, 
        y = y_test,
        prob_threshold = hparams["prob_threshold"]
    ))

    print("Variable importance of the trained model:")

    feature_importances = return_variable_importance(ada_model, cols_to_train)
    print(feature_importances.loc[feature_importances["importance"] != 0])

    # Store the model if the user wants to:

    if store:
        if rewrite:
            with open("predictive_models/ada_model_eboxes.pk1", "wb") as file:
                pickle.dump(ada_model, file)

            # Save the probability thresholds:
            with open("predictive_models/ada_prob_eboxes.pk1", "wb") as file:
                pickle.dump({"prob_ada_model": hparams["prob_threshold"]}, file)
            
            print("Model stored successfully!")
        else:
            # If we do not want to rewrite the last models we will just add the prefix "new" at the end

            with open("predictive_models/ada_model_eboxes_new.pk1", "wb") as file:
                pickle.dump(ada_model, file)

            # Save the probability thresholds:
            with open("predictive_models/ada_prob_eboxes_new.pk1", "wb") as file:
                pickle.dump({"prob_ada_model": hparams["prob_threshold"]}, file)

            print("Model stored successfully!")









        








