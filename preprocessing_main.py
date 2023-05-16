import pandas as pd
import argparse
from functions_preprocessing_alarms import first_eboxes_preprocess, first_lights_preprocess, big_preprocess_eboxes, big_preprocess_lights
from functions_preprocessing_readings import split_readings, pre_power, pre_power_peak
from functions_preprocessing_meteo import first_meteo_preprocess, meteo_groupby
from functions_joins import join_light_alarms_readings_meteo, join_eboxes_alarms_readings_meteo, final_municipality_join

# Supress the warning SettingWithCopyWarning from pandas:
pd.options.mode.chained_assignment = None

parser = argparse.ArgumentParser()

parser.add_argument("--di", "--raw_data_folders_directory", help="Provide the local directory on your machine of the folders raw_data and meteo_raw_data. Example: /home/leibniz/Desktop/IHMAN")
parser.add_argument("--mu", "--list_municipalities", help="List of the municipalities the user wants to preprocess", nargs='+', default=[])

args = parser.parse_args()
data_dir = args.di
municipalities_list = args.mu

# Lists for storing the preprocessed dataframes:
lights_dfs = []
eboxes_dfs = []

for municipality in municipalities_list:

    print(municipality + " in progress...")
    # Read the data from the folders raw_data and meteo_raw_data:
    meteo = pd.read_csv(f"{data_dir}/meteo_raw_data/meteo_{municipality}.csv")
    print("meteo read!")
    # The column info_value has mixed types and returns an error when reading it in the case of Illora. Since we don't need this column
    # we simply drop it
    lights_alarms = pd.read_csv(f"{data_dir}/raw_data/{municipality}_lights_alarms.csv", usecols=lambda column: column != "info_value")
    print("light_alarms read!")
    eboxes_alarms = pd.read_csv(f"{data_dir}/raw_data/{municipality}_eboxes_alarms.csv")
    print("eboxes_alarms read!")
    nodes = pd.read_csv(f"{data_dir}/raw_data/{municipality}_nodes.csv")
    print("nodes read!")
    readings = pd.read_csv(f"{data_dir}/raw_data/{municipality}_readings.csv")
    print("readings read!")


    # Prepocess meteorological data:
    meteo = first_meteo_preprocess(
        meteo = meteo,
        filter_dates = True,
        date_min = "2018-12-31",
        date_max = "2100-01-01"
    )

    meteo = meteo_groupby(meteo)
    print("Meteo done!")

    # Readings preprocess:
    powerReactive, powerActive, powerReactivePeak, powerActivePeak = split_readings(readings)
    powerReactive, powerActive = pre_power(powerReactive, "Reactive"), pre_power(powerActive, "Active")
    powerReactivePeak, powerActivePeak = pre_power_peak(powerReactivePeak, "Reactive"), pre_power_peak(powerActivePeak, "Active")
    print("Readings done!")

    # Alarms preprocess
    lights_alarms = first_lights_preprocess(lights_alarms)
    eboxes_alarms = first_eboxes_preprocess(eboxes_alarms)

    lights_alarms = big_preprocess_lights(lights_alarms)
    eboxes_alarms = big_preprocess_eboxes(eboxes_alarms)
    print("Alarms done!")

    # Join alarms, and nodes
    lights_alarms = pd.merge(lights_alarms, nodes, on="id", how="left")

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

    eboxes_dfs.append(eboxes_alarms)
    lights_dfs.append(lights_alarms)

out_lights = final_municipality_join(municipalities_list, lights_dfs)
out_eboxes = final_municipality_join(municipalities_list, eboxes_dfs)






