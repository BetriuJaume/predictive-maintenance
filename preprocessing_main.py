import pandas as pd
import argparse
from preprocessing_functions.functions_preprocessing_alarms import first_eboxes_preprocess, first_lights_preprocess, big_preprocess_eboxes, big_preprocess_lights
from preprocessing_functions.functions_preprocessing_readings import get_indexes_split, pre_readings_batches
from preprocessing_functions.functions_preprocessing_meteo import first_meteo_preprocess, meteo_groupby
from preprocessing_functions.functions_joins import join_light_alarms_readings_meteo, join_eboxes_alarms_readings_meteo, final_municipality_join
from preprocessing_functions.utils_preprocessing import return_meteo_filter_dates

# Supress the warning SettingWithCopyWarning from pandas:
pd.options.mode.chained_assignment = None

parser = argparse.ArgumentParser()
parser.add_argument("--di", "--raw_data_folders_directory", help="Provide the local directory on your machine of the folders raw_data and meteo_raw_data. Example: /home/leibniz/Desktop/IHMAN")
parser.add_argument("--mu", "--list_municipalities", help="List of the municipalities the user wants to preprocess", nargs='+', default=[])
parser.add_argument("--ws", "--week_split", help="Number of weeks to divide the batches of the readings dataset", default=20)

args = parser.parse_args()
data_dir = args.di
municipalities_list = args.mu
n_weeks = args.ws

# Lists for storing the preprocessed dataframes:
lights_dfs = []
eboxes_dfs = []

for municipality in municipalities_list:

    print(municipality + " in progress...")
    # Read the data from the folders raw_data and meteo_raw_data:
    meteo = pd.read_csv(f"{data_dir}/meteo_raw_data/new_meteo_{municipality}.csv")
    print("meteo read!")
    print(meteo["Date"].min())
    print(meteo["Date"].max())
    # The column info_value has mixed types and returns an error when reading it in the case of Illora. Since we don't need this column
    # we simply drop it
    lights_alarms = pd.read_csv(f"{data_dir}/raw_data/{municipality}_lights_alarms.csv", usecols=lambda column: column != "info_value")
    print("light_alarms read!")
    eboxes_alarms = pd.read_csv(f"{data_dir}/raw_data/{municipality}_eboxes_alarms.csv")
    print("eboxes_alarms read!")
    nodes = pd.read_csv(f"{data_dir}/raw_data/{municipality}_nodes.csv")
    print("nodes read!")
    # We will read the readings csv later in the code. First we need the dates so we can do
    # a temporal split of the data later:
    splitting_indexes = get_indexes_split(municipality, data_dir, n_weeks)
    print("split indexes read and preprocessed!")

    # Prepocess meteorological data:

    # First we call the function return_meteo_filter_dates to get date_min and date_max
    # for the first meteo preprocess:
    # meteo_date_min, meteo_date_max = return_meteo_filter_dates(municipality)
    meteo = first_meteo_preprocess(
        meteo = meteo,
        filter_dates = False,
        # date_min = meteo_date_min,
        # date_max = meteo_date_max
    )
    print("Meteo first preprocessed len: " + str(len(meteo)))

    meteo = meteo_groupby(meteo)
    print("Meteo done!")
    print("Meteo data preprocessed len: " + str(len(meteo)))

    # Readings preprocess:
    powerReactive, powerActive, powerReactivePeak, powerActivePeak = pre_readings_batches(splitting_indexes, municipality, data_dir)
    print("Readings done!")
    print("Preprocessed readings len: ")
    print("    powerReactive: " + str(len(powerReactive)))
    print("    powerActive: " + str(len(powerActive)))
    print("    powerActivePeak: " + str(len(powerActivePeak)))
    print("    powerReactivePeak: " + str(len(powerReactivePeak)))

    print("Alarms inputs:")
    print("     eboxes: " + str(len(eboxes_alarms)))
    print("     lights: " + str(len(lights_alarms)))
    
    # Alarms preprocess
    lights_alarms = first_lights_preprocess(lights_alarms)
    eboxes_alarms = first_eboxes_preprocess(eboxes_alarms)
    print("First preprocessed alarms len:")
    print("     eboxes: " + str(len(eboxes_alarms)))
    print("     lights: " + str(len(lights_alarms)))

    lights_alarms = big_preprocess_lights(lights_alarms)
    eboxes_alarms = big_preprocess_eboxes(eboxes_alarms)
    print("Alarms done!")
    print("Preprocessed alarms len: ")
    print("    eboxes: " + str(len(eboxes_alarms)))
    print("    lights: " + str(len(lights_alarms)))

    # Join alarms, and nodes
    lights_alarms = pd.merge(lights_alarms, nodes, on="id", how="left")
    print("Light alarms and nodes len: " + str(len(lights_alarms)))

    # Join readings, alarms and meteo:
    # You have to run the function join_eboxes_readings_meteo before runing join_light_alarms_readings_meteo
    # because this last function modifies the datasets of the readings. In case you run the lights join
    # first the code is going to return a Key error.
    eboxes_alarms = join_eboxes_alarms_readings_meteo(
        eboxes_alarms = eboxes_alarms,
        eboxes_powerReactivePeak = powerReactivePeak,
        eboxes_powerReactive = powerReactive,
        eboxes_powerActive = powerActive,
        eboxes_powerActivePeak = powerActivePeak,
        meteo = meteo
    )

    lights_alarms = join_light_alarms_readings_meteo(
        light_errors = lights_alarms,
        eboxes_powerReactivePeak = powerReactivePeak,
        eboxes_powerReactive = powerReactive,
        eboxes_powerActive = powerActive,
        eboxes_powerActivePeak = powerActivePeak,
        meteo = meteo
    )
    print("Joins done!")
    print("Eboxes alarms joined: " + str(len(eboxes_alarms)))
    print("Light alarms joined: " + str(len(lights_alarms)))

    eboxes_dfs.append(eboxes_alarms)
    lights_dfs.append(lights_alarms)

out_lights = final_municipality_join(municipalities_list, lights_dfs)
out_eboxes = final_municipality_join(municipalities_list, eboxes_dfs)

print("Out lights len:")
print(len(out_lights))
print("Out eboxes len:")
print(len(out_eboxes))

out_lights.to_csv(f"{data_dir}/preprocessing_results/out_lights.csv")
out_eboxes.to_csv(f"{data_dir}/preprocessing_results/out_eboxes.csv")