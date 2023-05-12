import pandas as pd
from typing import List

def change_format(date: str) -> str:
    """
    Get the date from a datime object converted into a string
    """
    return date[:10]

def join_light_alarms_readings_meteo(
        light_errors: pd.DataFrame, 
        eboxes_powerReactivePeak: pd.DataFrame,
        eboxes_powerReactive: pd.DataFrame,
        eboxes_powerActive: pd.DataFrame,
        eboxes_powerActivePeak: pd.DataFrame,
        meteo: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Joins the preprocessed data from light_alarms, readings, and meteo data by weeks
    """

    # First we have to get the date from the column "measure_week" to do the join with light_errors
    #eboxes_powerReactivePeak["dated"] = eboxes_powerReactivePeak["measure_week"].apply(lambda x: datetime.datetime.strptime(str(x), "%Y-%m-%dT%H:%M:%S.%fZ"))

    # Change the names of the columns so we can do the merge:
    eboxes_powerReactivePeak.rename(columns = {"id": "ebox_id"}, inplace=True)
    eboxes_powerReactive.rename(columns = {"id": "ebox_id"}, inplace=True)
    eboxes_powerActivePeak.rename(columns = {"id": "ebox_id"}, inplace=True)
    eboxes_powerActive.rename(columns = {"id": "ebox_id"}, inplace=True)

    meteo['dated'] = meteo['dated'].apply(change_format)

    # joins for the dataset eboxes_powerReactivePeak:
    for week in ["week-4", "week-3", "week-2", "week-1", "current_week"]:

        light_errors = pd.merge(
            light_errors, 
            eboxes_powerReactivePeak.rename(columns={"dated": week}), 
            on=["ebox_id", week],
            how="left"
        )
        
        light_errors.rename(columns={"ReactivePeak": "ReactivePeak_" + week}, inplace=True)

    # joins for the dataset eboxes_powerReactive:
    for week in ["week-4", "week-3", "week-2", "week-1", "current_week"]:

        light_errors = pd.merge(
            light_errors, 
            eboxes_powerReactive.rename(columns={"dated": week}), 
            on=["ebox_id", week],
            how="left"
        )
        
        light_errors.rename(
            columns={
                "powerReactive_sum": "powerReactive_sum_" + week,
                "powerReactive_p1": "powerReactive_p1_" + week,
                "powerReactive_p2": "powerReactive_p2_" + week,
                "powerReactive_p3": "powerReactive_p3_" + week
                    },
            inplace=True)

    # joins for the dataset eboxes_powerActivePeak:
    for week in ["week-4", "week-3", "week-2", "week-1", "current_week"]:

        light_errors = pd.merge(
            light_errors, 
            eboxes_powerActivePeak.rename(columns={"dated": week}), 
            on=["ebox_id", week],
            how="left"
        )
        
        light_errors.rename(columns={"ActivePeak": "ActivePeak_" + week}, inplace=True)

    # joins for the dataset eboxes_powerActive:
    for week in ["week-4", "week-3", "week-2", "week-1", "current_week"]:

        light_errors = pd.merge(
            light_errors, 
            eboxes_powerActive.rename(columns={"dated": week}), 
            on=["ebox_id", week],
            how="left"
        )
        
        light_errors.rename(
            columns={
                "powerActive_sum": "powerActive_sum_" + week,
                "powerActive_p1": "powerActive_p1_" + week,
                "powerActive_p2": "powerActive_p2_" + week,
                "powerActive_p3": "powerActive_p3_" + week
                    },
            inplace=True)

        # joins for the dataset meteo_canyelles:
        for week in ["week-4", "week-3", "week-2", "week-1", "current_week"]:

            light_errors = pd.merge(
                light_errors, 
                meteo.rename(columns={"dated": week}), 
                on=[week],
                how="left"
            )
            
            light_errors.rename(
                columns={
                    "Dew_max_max": "Dew_max_max_" + week,
                    "Hum_min_min": "Hum_min_min_" + week,
                    "Wind_max_max": "Wind_max_max_" + week,
                    "Temp_max_max": "Temp_max_max_" + week,
                    "Temp_min_min": "Temp_min_min_" + week,
                    "Pres_min_min": "Pres_min_min_" + week,
                    "Wind_avg_mean": "Wind_avg_avg_" + week,
                    "Wind_avg_std": "Wind_avg_std_" + week,
                    "Hum_avg_mean": "Hum_avg_avg_" + week,
                    "Dew_avg_mean": "Dew_avg_avg_" + week,
                    "Dew_avg_std": "Dew_avg_std_" + week,
                    "Hum_max_max": "Hum_max_max_" + week,
                    "Temp_avg_mean": "Temp_avg_avg_" + week,
                    "Temp_avg_std": "Temp_avg_std_" + week,
                    "Pres_max_max": "Pres_max_max_" + week,
                    "Pres_avg_mean": "Pres_avg_avg_" + week,
                    "Pres_avg_std": "Pres_avg_std_" + week,
                    "Precipitation_mean": "Precipitation_avg_" + week,
                    "Precipitation_sum": "Precipitation_sum_" + week,
                    "Dew_min_min": "Dew_min_min_" + week,
                    "Wind_min_min": "Wind_min_min_" + week,
                    "Hum_avg_std": "Hum_avg_std_" + week
                        },
                inplace=True)
            
def join_eboxes_alarms_readings_meteo(
        eboxes_alarms: pd.DataFrame, 
        eboxes_powerReactivePeak: pd.DataFrame,
        eboxes_powerReactive: pd.DataFrame,
        eboxes_powerActive: pd.DataFrame,
        eboxes_powerActivePeak: pd.DataFrame,
        meteo: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Joins the preprocessed data from light_alarms, readings, and meteo data by weeks
    """

    #Change meteo date format
    meteo['dated'] = meteo['dated'].apply(change_format)

    # joins for the dataset eboxes_powerReactivePeak:
    for week in ["week-4", "week-3", "week-2", "week-1", "current_week"]:

        eboxes_alarms = pd.merge(
            eboxes_alarms, 
            eboxes_powerReactivePeak.rename(columns={"dated": week}), 
            on=["id", week],
            how="left"
        )
        
        eboxes_alarms.rename(columns={"ReactivePeak": "ReactivePeak_" + week}, inplace=True)

    # joins for the dataset eboxes_powerReactive:
    for week in ["week-4", "week-3", "week-2", "week-1", "current_week"]:

        eboxes_alarms = pd.merge(
            eboxes_alarms, 
            eboxes_powerReactive.rename(columns={"dated": week}), 
            on=["id", week],
            how="left"
        )
        
        eboxes_alarms.rename(
            columns={
                "powerReactive_sum": "powerReactive_sum_" + week,
                "powerReactive_p1": "powerReactive_p1_" + week,
                "powerReactive_p2": "powerReactive_p2_" + week,
                "powerReactive_p3": "powerReactive_p3_" + week
                    },
            inplace=True)

    # joins for the dataset eboxes_powerActivePeak:
    for week in ["week-4", "week-3", "week-2", "week-1", "current_week"]:

        eboxes_alarms = pd.merge(
            eboxes_alarms, 
            eboxes_powerActivePeak.rename(columns={"dated": week}), 
            on=["id", week],
            how="left"
        )
        
        eboxes_alarms.rename(columns={"ActivePeak": "ActivePeak_" + week}, inplace=True)

    # joins for the dataset eboxes_powerActive:
    for week in ["week-4", "week-3", "week-2", "week-1", "current_week"]:

        eboxes_alarms = pd.merge(
            eboxes_alarms, 
            eboxes_powerActive.rename(columns={"dated": week}), 
            on=["id", week],
            how="left"
        )
        
        eboxes_alarms.rename(
            columns={
                "powerActive_sum": "powerActive_sum_" + week,
                "powerActive_p1": "powerActive_p1_" + week,
                "powerActive_p2": "powerActive_p2_" + week,
                "powerActive_p3": "powerActive_p3_" + week
                    },
            inplace=True)
        

    # joins for the dataset meteo_canyelles:
    for week in ["week-4", "week-3", "week-2", "week-1", "current_week"]:

        eboxes_alarms = pd.merge(
            eboxes_alarms, 
            meteo.rename(columns={"dated": week}), 
            on=[week],
            how="left"
        )
        
        eboxes_alarms.rename(
            columns={
                "Dew_max_max": "Dew_max_max_" + week,
                "Hum_min_min": "Hum_min_min_" + week,
                "Wind_max_max": "Wind_max_max_" + week,
                "Temp_max_max": "Temp_max_max_" + week,
                "Temp_min_min": "Temp_min_min_" + week,
                "Pres_min_min": "Pres_min_min_" + week,
                "Wind_avg_mean": "Wind_avg_avg_" + week,
                "Wind_avg_std": "Wind_avg_std_" + week,
                "Hum_avg_mean": "Hum_avg_avg_" + week,
                "Dew_avg_mean": "Dew_avg_avg_" + week,
                "Dew_avg_std": "Dew_avg_std_" + week,
                "Hum_max_max": "Hum_max_max_" + week,
                "Temp_avg_mean": "Temp_avg_avg_" + week,
                "Temp_avg_std": "Temp_avg_std_" + week,
                "Pres_max_max": "Pres_max_max_" + week,
                "Pres_avg_mean": "Pres_avg_avg_" + week,
                "Pres_avg_std": "Pres_avg_std_" + week,
                "Precipitation_mean": "Precipitation_avg_" + week,
                "Precipitation_sum": "Precipitation_sum_" + week,
                "Dew_min_min": "Dew_min_min_" + week,
                "Wind_min_min": "Wind_min_min_" + week,
                "Hum_avg_std": "Hum_avg_std_" + week
                    },
            inplace=True)
        
def final_municipality_join(municipality_names: List[str], municipality_dataframes: List[pd.DataFrame]):
    """
    Joins the data from all the diferent municipalities to train the models. You can use this function
    for eboxes and for lights.
    """

    # First we create a column on each one of the dataframes that contains information about the municipality
    for df, location in zip(municipality_dataframes, municipality_names):
        df["location"] = location

    # Concat all the data in one dataframe:
    complete_df = pd.concat(municipality_names)

    return complete_df