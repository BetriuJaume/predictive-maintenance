import pandas as pd
import numpy as np
import pickle
from predictive_models.predictive_models_functions import test_adboc_model
from test_evaluate_functions.test_evaluate_functions import calculate_accuracies, test_model, variable_importance, return_variable_importance
from training_functions.training_functions import calculate_binary_variables, split, split_x_y, train_ada_boost, split_easy_and_sudden_errors
from streamlit_functions.streamlit_functions import run_lights_default_streamlit, run_lights_adboc_streamlit, run_eboxes_streamlit
import warnings
from sklearn.exceptions import DataConversionWarning
import streamlit as st

warnings.filterwarnings("ignore", category=DataConversionWarning)
st.set_page_config(layout="wide")

# Inizialize the variable at the begining of the session so the code does not crash
if "submit_button_clicked" not in st.session_state:
    st.session_state["submit_button_clicked"] = False

st.title("Welcome to the predictive maintenance training aplication!")

with st.form(key="input_form"):

    st.subheader("Basic configuration")

    col1, col2 = st.columns(2)

    with col1:
        data_dir = st.text_input(label="Enter the directory where you have your data", value="/home/leibniz/Desktop/IHMAN")
        device = st.selectbox("Set the device", ["lights", "eboxes"])

    with col2:
        model_type = st.selectbox("Select model type (Only for lights)", ["default", "adboc"])
        store = st.checkbox(label="Store models", help="Mark if you want the model to be stored once the model is trained. Leave unchecked while doing tests")
        rewrite = st.checkbox(label="Rewrite models", help="Mark if you want to rewrite the models already stored. Leave unckecked while doing tests")

    if (rewrite) & (not store):
        st.error("Must store the model in order to rewrite it")

    if (device == "eboxes") & (model_type == "adboc"):
        st.error("adboc predictor can only be used for light predictive maintenance")

    submit_button_clicked = st.form_submit_button(label="Submit")

# Store the value of the submit_button_clicked to the session state. This prevents the Basic configuration form to disapear when we submit ther next form for 
# training the models
if submit_button_clicked:
    st.session_state["submit_button_clicked"] = submit_button_clicked

# Once the submit button of the basic configuration is clicked:
if st.session_state["submit_button_clicked"]:

    if device == "lights":
        
        if model_type == "default":
            run_lights_default_streamlit(data_dir, store, rewrite)

        if model_type == "adboc":
            run_lights_adboc_streamlit(data_dir, store, rewrite)

    if device == "eboxes":
        run_eboxes_streamlit(data_dir, store, rewrite)
