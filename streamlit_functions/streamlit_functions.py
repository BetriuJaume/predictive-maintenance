import streamlit as st
import pandas as pd
import pickle
from predictive_models.predictive_models_functions import test_adboc_model
from test_evaluate_functions.test_evaluate_functions import calculate_accuracies, test_model, variable_importance, return_variable_importance
from training_functions.training_functions import calculate_binary_variables, split, split_x_y, train_ada_boost, split_easy_and_sudden_errors

def run_lights_default_streamlit(data_dir: str, store: bool, rewrite: bool) -> None:
    
    with st.form(key="lights_default_form"):
        st.subheader("Select the hyperparameters")
        st.info("The default model for lights consists of an Ada Boost predictor trained with all the columns with exception of the readings")

        # Define the dict where we will store all the  hparams:
        hparams = {}

        col1, col2 = st.columns(2)

        with col1:
            hparams["n_estimators"] = st.slider(label="Number of estimators", min_value=1, max_value=100, value=30, step=1, help="Sets the number of decision trees to be used in the boosting model")
            hparams["max_depth_tree"] = st.slider(label="Tree maximum depth", min_value=1, max_value=20, value=1, step=1, help="Sets the maximum depth of the trees. Deep trees can lead to overfitting")

        with col2:
            hparams["lr"] = st.slider(label="Learning rate", min_value=0., max_value=1., value=0.5, step=0.01, help="Very small values can lead to the model no learning with every new tree")
            hparams["prob_threshold"] = st.slider(label="Probability threshold", min_value=0., max_value=1., value=0.47, step=0.01, help="Used to decide how to classify the predictions into the binary categories")

        submit_light_default_clicked = st.form_submit_button(label="Submit")

    if submit_light_default_clicked:

        with st.spinner(text="Preparing data..."):

            dff = pd.read_csv(f"{data_dir}/preprocessing_results/out_lights.csv")
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
            df = df.fillna(df.mean(numeric_only=True))

            df = calculate_binary_variables(df)
            # Preprocessing and split:
            df["current_week"] = pd.to_datetime(df["current_week"])

            train, validation, test = split(df, n_weeks=77)

            cols_to_train = df.drop(["current_week", "hours_next_four_weeks", "error_next_four_weeks", "hours_week+1", "hours_week+2", "hours_week+3", "hours_week+4"], axis=1).columns
            x_train, y_train = split_x_y(train, cols_to_train)
            x_validation, y_validation = split_x_y(validation, cols_to_train)
            x_test, y_test = split_x_y(test, cols_to_train)

        with st.spinner(text="Training model..."):

            ada_model = train_ada_boost(
                max_depth_tree = hparams["max_depth_tree"],
                n_estimators = hparams["n_estimators"],
                lr = hparams["lr"],
                x = x_train,
                y = y_train
            )

            validation_results = test_model(
                model = ada_model,
                x = x_validation, 
                y = y_validation,
                prob_threshold = hparams["prob_threshold"]
            )
            
            test_results = test_model(
                model = ada_model,
                x = x_test, 
                y = y_test,
                prob_threshold = hparams["prob_threshold"]
            )

            # Modify the values in the validations results so we can use them to build a df
            for key in validation_results:
                validation_results[key] = [validation_results[key]]
            for key in test_results:
                test_results[key] = [test_results[key]]

            col3, col4 = st.columns(2)
            with col3:
                st.subheader("Validation results:")
                st.write(pd.DataFrame(validation_results))
            with col4:
                st.subheader("Test results:")
                st.write(pd.DataFrame(test_results))

            # Calculate the importance of each one of the columns and display it:
            feature_importances = return_variable_importance(ada_model, cols_to_train)
            feature_importance_display = feature_importances.loc[feature_importances["importance"] != 0]

            st.subheader("Feature importances:")
            st.write(feature_importance_display)

            if store:
                if rewrite:
                    with open("predictive_models/ada_model_readings.pk1", "wb") as file:
                        pickle.dump(ada_model, file)

                    # Save the probability thresholds:
                    with open("predictive_models/ada_prob_readings.pk1", "wb") as file:
                        pickle.dump({"prob_ada_model": hparams["prob_threshold"]}, file)
                    
                    st.success("Models and probability thresholds successfully stored in the folder predictive_models under the names: ada_model_readings and ada_prob_readings")
                else:
                    # If we do not want to rewrite the last models we will just add the prefix "new" at the end

                    with open("predictive_models/ada_model_readings_new.pk1", "wb") as file:
                        pickle.dump(ada_model, file)

                    # Save the probability thresholds:
                    with open("predictive_models/ada_prob_readings_new.pk1", "wb") as file:
                        pickle.dump({"prob_ada_model": hparams["prob_threshold"]}, file)
                    
                    st.success("Models and probability thresholds successfully stored in the folder predictive_models under the names: ada_model_readings_new and ada_prob_readings_new")
        
def run_lights_adboc_streamlit(data_dir: str, store: bool, rewrite: bool) -> None:

    with st.form(key="lights_adboc_form"):
        st.subheader("Select the hyperparameters")
        st.info("The adboc model for lights consists of two Ada Boost predictors combined trained with all the columns including the electric readings")

        # Define the dict where we will store all the  hparams:
        hparams = {}
        hparams_sudden = {}

        st.subheader("Default model:")

        col1, col2 = st.columns(2)
        with col1:
            hparams["n_estimators"] = st.slider(label="Number of estimators", min_value=1, max_value=100, value=30, step=1, help="Sets the number of decision trees to be used in the boosting model")
            hparams["max_depth_tree"] = st.slider(label="Tree maximum depth", min_value=1, max_value=20, value=1, step=1, help="Sets the maximum depth of the trees. Deep trees can lead to overfitting")

        with col2:
            hparams["lr"] = st.slider(label="Learning rate", min_value=0., max_value=1., value=0.5, step=0.01, help="Very small values can lead to the model no learning with every new tree")
            hparams["prob_threshold"] = st.slider(label="Probability threshold", min_value=0., max_value=1., value=0.45, step=0.01, help="Used to decide how to classify the predictions into the binary categories")
        
        st.subheader("Sudden model:")

        col3, col4 = st.columns(2)
        with col3:
            hparams_sudden["n_estimators"] = st.slider(label="Number of estimators", min_value=1, max_value=100, value=10, step=1, help="Sets the number of decision trees to be used in the boosting model")
            hparams_sudden["max_depth_tree"] = st.slider(label="Tree maximum depth", min_value=1, max_value=20, value=3, step=1, help="Sets the maximum depth of the trees. Deep trees can lead to overfitting")

        with col4:
            hparams_sudden["lr"] = st.slider(label="Learning rate", min_value=0., max_value=1., value=0.7, step=0.01, help="Very small values can lead to the model no learning with every new tree")
            hparams_sudden["prob_threshold"] = st.slider(label="Probability threshold", min_value=0., max_value=1., value=0.48, step=0.01, help="Used to decide how to classify the predictions into the binary categories")

        submit_light_default_clicked = st.form_submit_button(label="Submit")

    if submit_light_default_clicked:

        with st.spinner("Preparing data..."):
            # Read the data
            dff = pd.read_csv(f"{data_dir}/preprocessing_results/out_lights.csv")

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

        with st.spinner("Training models..."):

            ada_model = train_ada_boost(
                max_depth_tree = hparams["max_depth_tree"],
                n_estimators = hparams["n_estimators"],
                lr = hparams["lr"],
                x  = x_train,
                y = y_train
            )

            validation_results = test_model(
                model = ada_model,
                x = x_validation, 
                y = y_validation,
                prob_threshold = hparams["prob_threshold"]
            )

            test_results = test_model(
                model = ada_model,
                x = x_test, 
                y = y_test,
                prob_threshold = hparams["prob_threshold"]
            )

            # Modify the values in the results so we can use them to build a df:
            for key in validation_results:
                validation_results[key] = [validation_results[key]]
            for key in test_results:
                test_results[key] = [test_results[key]]

            st.header("Validation and test results model trained with the whole train data:")
            col3, col4 = st.columns(2)
            
            with col3:
                st.subheader("Validation")
                st.write(pd.DataFrame(validation_results))
            with col4:
                st.subheader("Test")
                st.write(pd.DataFrame(test_results))

            feature_importances = return_variable_importance(ada_model, cols_to_train)
            features_importances_display = feature_importances.loc[feature_importances["importance"] != 0]

            st.subheader("Feature importances:")
            st.write(features_importances_display)

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

            ada_sudden_model = train_ada_boost(
                max_depth_tree = hparams_sudden["max_depth_tree"],
                n_estimators = hparams_sudden["n_estimators"],
                lr = hparams_sudden["lr"],
                x  = x_train_sudden,
                y = y_train_sudden
            )

            validation_results_sudden = test_model(
                model = ada_sudden_model, 
                x = x_validation_sudden, 
                y = y_validation_sudden,
                prob_threshold = hparams_sudden["prob_threshold"]
            )    

            test_results_sudden = test_model(
                model = ada_sudden_model, 
                x = x_test_sudden, 
                y = y_test_sudden,
                prob_threshold = hparams_sudden["prob_threshold"]
            )

            test_results_whole_dataset = test_model(
                model = ada_sudden_model,
                x = x_test, 
                y = y_test,
                prob_threshold = 0.48
            )
            
            # Modify the values in the results so we can use them to build a df:
            for key in validation_results_sudden:
                validation_results_sudden[key] = [validation_results_sudden[key]]
            for key in test_results_sudden:
                test_results_sudden[key] = [test_results_sudden[key]]
            for key in test_results_whole_dataset:
                test_results_whole_dataset[key] = [test_results_whole_dataset[key]]

            st.header("Validation and test results model trained with the subset 'sudden errors':")
            col5, col6 = st.columns(2)            
            with col5:
                st.subheader("Validation")
                st.write(pd.DataFrame(validation_results_sudden))
            with col6:
                st.subheader("Test")
                st.write(pd.DataFrame(test_results_sudden))
            
            st.subheader("Test results model trained with the subset 'sudden errors' for the whole test data:")
            st.write(pd.DataFrame(test_results_whole_dataset))

            feature_importances_sudden = return_variable_importance(ada_sudden_model, cols_to_train)
            features_importances_display = feature_importances_sudden.loc[feature_importances_sudden["importance"] != 0]

            st.subheader("Feature importances model trained with the subset 'sudden errors'")
            st.write(features_importances_display)

            st.subheader("Final test results using the Ada Boost Combined Predictor (adboc)")
            st.info("This combines the model trained with the whole data and the model trained with the subset to guarantee a better dettection of sudden errors without losing generality")
        
        with st.spinner("Creating predictions..."):
            acc, _ = test_adboc_model(
                x_test,
                y_test,
                ada_model,
                ada_sudden_model,
                hparams["prob_threshold"],
                hparams_sudden["prob_threshold"]
            )

            # Modify the dict to convert it to a DataFrame:
            for key in acc:
                acc[key] = [acc[key]]

            st.write(pd.DataFrame(acc))

            if store:
                if rewrite:
                    with open("predictive_models/ada_model.pk1", "wb") as file:
                        pickle.dump(ada_model, file)

                    with open("predictive_models/ada_sudden_model.pk1", "wb") as file:
                        pickle.dump(ada_sudden_model, file)

                    # Save the probability thresholds:
                    with open("predictive_models/ada_prob.pk1", "wb") as file:
                        pickle.dump(
                            {"prob_ada_model": hparams["prob_threshold"], "prob_sudden_model": hparams_sudden["prob_threshold"]},
                            file
                        )

                    st.success("Models and probability thresholds successfully stored in the folder predictive_models under the names: ada_model, ada_sudden_model and ada_prob")

                else:
                    # If we do not want to rewrite the last models we will just add the prefix "new" at the end

                    with open("predictive_models/ada_model_new.pk1", "wb") as file:
                        pickle.dump(ada_model, file)

                    with open("predictive_models/ada_sudden_model_new.pk1", "wb") as file:
                        pickle.dump(ada_sudden_model, file)

                    # Save the probability thresholds:
                    with open("predictive_models/ada_prob_new.pk1", "wb") as file:
                        pickle.dump(
                            {"prob_ada_model": hparams["prob_threshold"], "prob_sudden_model": hparams_sudden["prob_threshold"]},
                            file
                        )
                    
                    st.success("Models and probability thresholds successfully stored in the folder predictive_models under the names: ada_model_new, ada_sudden_model_new and ada_prob_new")
                    
def run_eboxes_streamlit(data_dir: str, store: bool, rewrite: bool):

    with st.form(key="eboxes_form"):
        st.subheader("Select the hyperparameters")
        st.info("The model for eboxes consists of an Ada Boost predictor trained with all the columns with exception of the readings")

        # Define the dict where we will store all the  hparams:
        hparams = {}

        col1, col2 = st.columns(2)

        with col1:
            hparams["n_estimators"] = st.slider(label="Number of estimators", min_value=1, max_value=100, value=10, step=1, help="Sets the number of decision trees to be used in the boosting model")
            hparams["max_depth_tree"] = st.slider(label="Tree maximum depth", min_value=1, max_value=20, value=1, step=1, help="Sets the maximum depth of the trees. Deep trees can lead to overfitting")

        with col2:
            hparams["lr"] = st.slider(label="Learning rate", min_value=0., max_value=1., value=0.5, step=0.01, help="Very small values can lead to the model no learning with every new tree")
            hparams["prob_threshold"] = st.slider(label="Probability threshold", min_value=0., max_value=1., value=0.48, step=0.01, help="Used to decide how to classify the predictions into the binary categories")

        submit_eboxes_clicked = st.form_submit_button(label="Submit")
    
    if submit_eboxes_clicked:

        with st.spinner("Preparing data..."):
            
            dff = pd.read_csv(f"{data_dir}/preprocessing_results/out_eboxes.csv")

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
            df = df.fillna(df.mean(numeric_only=True))

            # Split information:
            df = calculate_binary_variables(df)
            # Preprocessing and split:
        df["current_week"] = pd.to_datetime(df["current_week"])
        train, validation, test = split(df, n_weeks=80)

        cols_to_train = df.drop(["current_week", "hours_next_four_weeks", "error_next_four_weeks", "hours_week+1", "hours_week+2", "hours_week+3", "hours_week+4"], axis=1).columns
        x_train, y_train = split_x_y(train, cols_to_train)
        x_validation, y_validation = split_x_y(validation, cols_to_train)
        x_test, y_test = split_x_y(test, cols_to_train)

        with st.spinner("Training model..."):

            ada_model = train_ada_boost(
                max_depth_tree = hparams["max_depth_tree"],
                n_estimators = hparams["n_estimators"],
                lr = hparams["lr"],
                x  = x_train,
                y = y_train
            )
    
            validation_results = test_model(
                model = ada_model,
                x = x_validation, 
                y = y_validation,
                prob_threshold = hparams["prob_threshold"]
            )

            test_results = test_model(
                model = ada_model,
                x = x_test, 
                y = y_test,
                prob_threshold = hparams["prob_threshold"]
            )

            # Modify the values in the validations results so we can use them to build a df
            for key in validation_results:
                validation_results[key] = [validation_results[key]]
            for key in test_results:
                test_results[key] = [test_results[key]]

            col3, col4 = st.columns(2)
            with col3:
                st.subheader("Validation results:")
                st.write(pd.DataFrame(validation_results))
            with col4:
                st.subheader("Test results:")
                st.write(pd.DataFrame(test_results))

            # Calculate the importance of each one of the columns and display it:
            feature_importances = return_variable_importance(ada_model, cols_to_train)
            feature_importance_display = feature_importances.loc[feature_importances["importance"] != 0]

            st.subheader("Feature importances:")
            st.write(feature_importance_display)

            if store:
                if rewrite:
                    with open("predictive_models/ada_model_eboxes.pk1", "wb") as file:
                        pickle.dump(ada_model, file)

                    # Save the probability thresholds:
                    with open("predictive_models/ada_prob_eboxes.pk1", "wb") as file:
                        pickle.dump({"prob_ada_model": hparams["prob_threshold"]}, file)
                    
                    st.success("Models and probability thresholds successfully stored in the folder predictive_models under the names: ada_model_eboxes and ada_prob_eboxes")
                else:
                    # If we do not want to rewrite the last models we will just add the prefix "new" at the end

                    with open("predictive_models/ada_model_eboxes_new.pk1", "wb") as file:
                        pickle.dump(ada_model, file)

                    # Save the probability thresholds:
                    with open("predictive_models/ada_prob_eboxes_new.pk1", "wb") as file:
                        pickle.dump({"prob_ada_model": hparams["prob_threshold"]}, file)
                    
                    st.success("Models and probability thresholds successfully stored in the folder predictive_models under the names: ada_model_eboxes_new and ada_prob_eboxes_new")

