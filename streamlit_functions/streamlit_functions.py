import streamlit as st
import pandas as pd
import pickle
from predictive_models.predictive_models_functions import test_adboc_model
from test_evaluate_functions.test_evaluate_functions import calculate_accuracies, test_model, variable_importance, return_variable_importance
from training_functions.training_functions import calculate_binary_variables, split, split_x_y, train_ada_boost, split_easy_and_sudden_errors


def run_lights_default_streamlit(dff: pd.DataFrame) -> None:
    
    with st.form(key="lights_default_form"):
        st.subheader("Select the hyperparameters")
        st.info("The default model for lights consists of an Ada Boost predictor trained with all the columns with exception of the readings")

        # Define the dict wherew we will store all the  hparams:
        hparams = {}

        col1, col2 = st.columns(2)

        with col1:
            hparams["n_estimators"] = st.slider(label="Number of estimators", min_value=1, max_value=100, value=30, step=1, help="Sets the number of decision trees to be used in the boosting model")
            hparams["max_depth_tree"] = st.slider(label="Tree maximum depth", min_value=1, max_value=20, value=1, step=1, help="Sets the maximum depth of the trees. Deep trees can lead to overfitting")

        with col2:
            hparams["lr"] = st.slider(label="Learning rate", min_value=0., max_value=1., value=0.5, step=0.01, help="Very small values can lead to the model no learning with every new tree")
            hparams["prob_threshold"] = st.slider(label="Probability threshold", min_value=0., max_value=1., value=0.47, step=0.01, help="Used to decide how to classify the predictions into the binary categories")

        submit_light_default_clicked = st.form_submit_button(label="Submit")
    
    st.write(submit_light_default_clicked)

    if submit_light_default_clicked:

        with st.spinner(text="Preparing data..."):
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

            # Calculate the impoertance of each one of the columns and display it:
            feature_importances = return_variable_importance(ada_model, cols_to_train)
            feature_importance_display = feature_importances.loc[feature_importances["importance"] != 0]

            st.subheader("Feature importances:")
            st.write(feature_importance_display)

        with col4:
            store = st.button(label="Store model")
            rewrite = st.checkbox(label="Rewrite existing models")
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
        
def run_lights_adboc_streamlit():
    pass

def run_eboxes_streamlit():
    pass