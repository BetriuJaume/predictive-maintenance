import pickle
import pandas as pd
import argparse
from predictive_models.predictive_models_functions import adboc_predict
from preprocessing_functions.functions_preprocessing_alarms import first_eboxes_preprocess, first_lights_preprocess, big_preprocess_eboxes, big_preprocess_lights
from preprocessing_functions.functions_preprocessing_readings import pre_power, pre_power_peak
from preprocessing_functions.functions_preprocessing_meteo import first_meteo_preprocess, meteo_groupby
from preprocessing_functions.functions_joins import join_light_alarms_readings_meteo, join_eboxes_alarms_readings_meteo

pd.options.mode.chained_assignment = None

parser = argparse.ArgumentParser()
parser.add_argument("--di", "--raw_data_folders_directory", help="Provide the local directory on your machine of the folders raw_data and meteo_raw_data. Example: /home/leibniz/Desktop/IHMAN")
# About this dates it will be better to dettect the current day where we are (sunday or any other day from the week) and get the data bia sql from the database. For now passing the dates as an argument
# is how we will get the date ranges but it is a temporal fix
parser.add_argument("--da", "--min_max_dates", help="Provide the min and max dates of the data you are doing predictions", nargs="+", default=[])
parser.add_argument("--mo", "--model", help="Indicate the model you want to use for predictions. 'adboc' for a model that does not use the readings or 'default' for a model that does use the readings", default="default")

args = parser.parse_args()

# Raise an error in case that the user has not specified the date range:
if len(args.da) == 0:
    raise Exception("Error: You have to specify the date range of your data")

data_dir = args.di
predicting_min = args.da[0]
predicting_max = args.da[1]
model_type = args.mo

print(predicting_min)
print(predicting_max)

# read data:
eboxes_alarms = pd.read_csv(f"{data_dir}/data_to_predict/eboxes_alarms.csv")
light_alarms = pd.read_csv(f"{data_dir}/data_to_predict/lights_alarms.csv")
nodes = pd.read_csv(f"{data_dir}/data_to_predict/nodes.csv")
powerActive = pd.read_csv(f"{data_dir}/data_to_predict/powerActive.csv")
powerReactive = pd.read_csv(f"{data_dir}/data_to_predict/powerReactive.csv")
powerActivePeak = pd.read_csv(f"{data_dir}/data_to_predict/powerActivePeak.csv")
powerReactivePeak = pd.read_csv(f"{data_dir}/data_to_predict/powerActivePeak.csv")

# meteo here:
meteo = pd.read_csv(f"{data_dir}/data_to_predict/new_meteo_mejorada.csv")

# Almost the same preprocessing as we do on preparing the data for training with some minor changes.
# For example in the alarms preprocessing functions we have to specify the argument for_predicting

powerReactive, powerActive = pre_power(powerReactive, "Reactive"), pre_power(powerActive, "Active")
powerReactivePeak, powerActivePeak = pre_power_peak(powerReactivePeak, "Reactive"), pre_power_peak(powerActivePeak, "Active")

meteo = first_meteo_preprocess(
    meteo = meteo,
    filter_dates = False
)

meteo = meteo_groupby(meteo)

light_alarms = first_lights_preprocess(light_alarms, for_predicting=True)

light_alarms = big_preprocess_lights(
    light_alarms = light_alarms,
    for_predicting = True,
    predicting_min_date = predicting_min,
    predicting_max_date = predicting_max
)

light_alarms = pd.merge(light_alarms, nodes, on="id", how="left")

light_alarms = join_light_alarms_readings_meteo(
    light_errors = light_alarms,
    eboxes_powerReactivePeak = powerReactivePeak,
    eboxes_powerReactive = powerReactive,
    eboxes_powerActive = powerActive,
    eboxes_powerActivePeak = powerActivePeak,
    meteo = meteo
)

# Depending on model_type we do predictions with one model or the other:
model_type = "adboc"

# In this case we don't use the ada boost combined predictor and the model will take into accout the readings. This
# model has an overall better accuracy than the adboc but fails to detect the sudden errors
if model_type == "default":
    with open("predictive_models/ada_model_readings.pk1", "rb") as file:
        ada_model = pickle.load(file)
    with open("predictive_models/ada_prob_readings.pk1", "rb") as file:
        prob_ada_model = pickle.load(file)["prob_ada_model"]
    
    df = light_alarms.copy()

    # Drop some usless columns for the model:
    drop_cols = [
        col for col in df.columns if
            (col in ["lat", "lon"]) | 
            (col == "Unnamed: 0") |
            (col.startswith("week")) |
            (col == "current_week") |
            (col == "type") |
            (col == "ebox_id") |
            (col == "location")
    ]
    df = df.drop(drop_cols, axis=1)

    # Interpolate some left missing values:
    df = df.fillna(df.mean(numeric_only=True))

    predictions = ada_model.predict_proba(df.drop("id", axis=1))[:, 1]

    predictions_out = pd.DataFrame(
        {
            "id": df["id"],
            "pred": predictions
        }
    )

    print("Predictions:")
    print("Probability threshold recommended for this model: " + str(prob_ada_model))
    print(predictions_out)

# In this case we will use the adboc model. This model has a worse overall accuracy than the default but has a better chance 
# at dettecting sudden errors. The model does not use the readings in this case.
if model_type == "adboc":
    with open("predictive_models/ada_model.pk1", "rb") as file:
        ada_model = pickle.load(file)
    with open("predictive_models/ada_sudden_model.pk1", "rb") as file:
        ada_sudden_model = pickle.load(file)
    with open("predictive_models/ada_prob.pk1", "rb") as file:
        probs_dict = pickle.load(file)
        prob_ada_model = probs_dict["prob_ada_model"]
        prob_sudden_model = probs_dict["prob_sudden_model"]
    
    df = light_alarms.copy()
    drop_cols = [
                col for col in df.columns if 
                (col.startswith("power")) | (col.startswith("Active")) | (col.startswith("Reactive") | 
                (col in ["lat", "lon"])) | 
                (col == "Unnamed: 0") |
                (col.startswith("week")) |
                (col == "current_week") |
                (col == "type") |
                (col == "ebox_id") |
                (col == "location")
            ]
    df = df.drop(drop_cols, axis=1)
    df = df.fillna(df.mean(numeric_only=True))

    predictions = adboc_predict(
        df = df,
        ada_model = ada_model,
        ada_sudden_model = ada_sudden_model,
        prob_threshold_ada_model = prob_ada_model,
        prob_threshold_ada_sudden_model = prob_sudden_model,
    )

    predictions_out = pd.DataFrame(
        {
            "id": predictions.keys(),
            "pred": predictions.values()
        }
    )

    print("Predictions:")
    print("Probability threshold recommended for the model ada_model: " + str(prob_ada_model))
    print("Probability threshold recommended for the model ada_sudden_model: " + str(prob_sudden_model))
    print(predictions_out)