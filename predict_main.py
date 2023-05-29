import pickle
import pandas as pd
import argparse
from predictive_models.predictive_models_functions import make_predictions
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

print("Inputed date range:")
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

# Untill this point we have just considered the lights that have had an error in the last 4 weeks. 
# We should consider too all the other alarms that have no errors in the last 4 weeks and put them
# in a dataframe format identical to the light-alarms dataframe. This way we will be able to do predictionns
# for all the lights in the city.

# Let's get the nodes that have not suffered any alarms in the last weeks:
light_nodes = nodes.loc[nodes["type"] == "light"]
non_error_lights_nodes = light_nodes.loc[~light_nodes["id"].isin(light_alarms["id"])]

# Now we have to build a dataframe with the same structure as light_alarms for this non_error_lights:
non_error_lights = pd.DataFrame(
    {
        "id": non_error_lights_nodes["id"],
        "week-4": light_alarms["week-4"][0],
        "hours_week-4": 0,
        "week-3": light_alarms["week-3"][0],
        "hours_week-3": 0,
        "week-2": light_alarms["week-2"][0],
        "hours_week-2": 0,
        "week-1": light_alarms["week-1"][0],
        "hours_week-1": 0,
        "current_week": light_alarms["current_week"][0],
        "hours_current_week": 0,
        "ebox_id": non_error_lights_nodes["ebox_id"],
        "lat": non_error_lights_nodes["lat"],
        "lon": non_error_lights_nodes["lon"]
    }
)
# Create the dataframe ready for the model:

non_error_lights = join_light_alarms_readings_meteo(
    light_errors = non_error_lights,
    eboxes_powerReactivePeak = powerReactivePeak,
    eboxes_powerReactive = powerReactive,
    eboxes_powerActive = powerActive,
    eboxes_powerActivePeak = powerActivePeak,
    meteo = meteo
)

print("Predictions for luminarire with errors in the last weeks:")
make_predictions(light_alarms, model_type)
print("Predictions for luminarire with no errors in the last weeks:")
make_predictions(non_error_lights, model_type)