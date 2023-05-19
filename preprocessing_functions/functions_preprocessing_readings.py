import pandas as pd
from preprocessing_functions.utils_preprocessing import return_monday
import numpy as np
from typing import List

def split_readings(readings: pd.DataFrame) -> tuple[pd.DataFrame]:
    """
    Split the readings dataset in four: powerReactive, powerActive, powerReactivePeak, powerActivePeak
    """
    
    powerReactive = readings.loc[readings["reading"] == "powerReactive"]
    powerActive = readings.loc[readings["reading"] == "powerActive"]
    powerReactivePeak = readings.loc[readings["reading"] == "powerReactivePeak"]
    powerActivePeak = readings.loc[readings["reading"] == "powerActivePeak"]

    return (powerReactive, powerActive, powerReactivePeak, powerActivePeak)


def pre_power(power: pd.DataFrame, type_reading: str) -> pd.DataFrame:
    """
    Preprocessing of the datasets powerReactive and powerActive. Takes as 
    arguments the readings and the type: Active or Reactive
    """
    # to date:
    power["dated"] = pd.to_datetime(power["dated"])

    # Eliminate useless columns:
    power.drop(["componentid", "reading"], axis=1, inplace=True)

    # separate the dataframe by components and drop the column component beacuse it does not give info anymore:
    sum_df = power.loc[power["component"] == "sum"].drop("component", axis=1)
    p1_df = power.loc[power["component"] == "p1"].drop("component", axis=1)
    p2_df = power.loc[power["component"] == "p2"].drop("component", axis=1)
    p3_df = power.loc[power["component"] == "p3"].drop("component", axis=1)

    if type_reading == "Active":
        # Change the name of the column "value" to "value_" + "measure" so when we do the merge later
        sum_df.rename(columns = {"value": "powerActive_sum"}, inplace=True)
        p1_df.rename(columns = {"value": "powerActive_p1"}, inplace=True)
        p2_df.rename(columns = {"value": "powerActive_p2"}, inplace=True)
        p3_df.rename(columns = {"value": "powerActive_p3"}, inplace=True)
       
    if type_reading == "Reactive":
        # Change the name of the column "value" to "value_" + "measure" so when we do the merge later
        sum_df.rename(columns = {"value": "powerReactive_sum"}, inplace=True)
        p1_df.rename(columns = {"value": "powerReactive_p1"}, inplace=True)
        p2_df.rename(columns = {"value": "powerReactive_p2"}, inplace=True)
        p3_df.rename(columns = {"value": "powerReactive_p3"}, inplace=True)

    # Group for and drop the column
    sum_df = sum_df.groupby(["id", pd.Grouper(key="dated", freq="W")]).mean(numeric_only=True).reset_index()
    p1_df = p1_df.groupby(["id", pd.Grouper(key="dated", freq="W")]).mean(numeric_only=True).reset_index()
    p2_df = p2_df.groupby(["id", pd.Grouper(key="dated", freq="W")]).mean(numeric_only=True).reset_index()
    p3_df = p3_df.groupby(["id", pd.Grouper(key="dated", freq="W")]).mean(numeric_only=True).reset_index()

    # Dates to Monday and to string so we have no errors when doing joins:
    sum_df["dated"] = sum_df["dated"].apply(lambda x: return_monday(x))
    sum_df["dated"] = sum_df["dated"].apply(lambda x: str(x))

    p1_df["dated"] = p1_df["dated"].apply(lambda x: return_monday(x))
    p1_df["dated"] = p1_df["dated"].apply(lambda x: str(x))

    p2_df["dated"] = p2_df["dated"].apply(lambda x: return_monday(x))
    p2_df["dated"] = p2_df["dated"].apply(lambda x: str(x))

    p3_df["dated"] = p3_df["dated"].apply(lambda x: return_monday(x))
    p3_df["dated"] = p3_df["dated"].apply(lambda x: str(x))

    # add all the mesures in one dataframe
    df = pd.merge(sum_df, p1_df, on=["dated", "id"])
    df = pd.merge(df, p2_df, on=["dated", "id"])
    df = pd.merge(df, p3_df, on=["dated", "id"])

    return df

def pre_power_peak(power_peak: pd.DataFrame, type_reading: str) -> pd.DataFrame:
    """
    Preprocessing of the datasets powerReactivePeak and powerActivePeak. Takes as 
    arguments the readings and the type: Active or Reactive
    """

    power_peak["dated"] = pd.to_datetime(power_peak["dated"])

    # Eliminate useless columns:
    power_peak.drop(["componentid", "reading"], axis=1, inplace=True)

    if type_reading == "Active":
        # Change the name of the column "value" to "ActivePeak"
        power_peak.rename(columns = {"value": "ActivePeak"}, inplace=True)
    
    if type_reading == "Reactive":
        # Change the name of the column "value" to "ReactivePeak"
        power_peak.rename(columns = {"value": "ReactivePeak"}, inplace=True)
    
    # Group:
    power_peak = power_peak.groupby(["id", pd.Grouper(key="dated", freq="W")]).mean(numeric_only=True).reset_index()

    # Dates to Monday and to string so we have no errors when doing joins:
    power_peak["dated"] = power_peak["dated"].apply(lambda x: return_monday(x))
    power_peak["dated"] = power_peak["dated"].apply(lambda x: str(x))

    return power_peak

def get_indexes_split(municipality: str, data_dir: str, n_weeks: str) -> List:
    """
    Reads the "dated" column of csv file of the readings of the municipality and returns a list
    of indexes that represent intervals of n_weeks to later do the split of the readings csv 
    using this indexes. In case of getting a memory error, it might be a good idea to make num_weeks
    smaller so the batches become smaller.
    """

    # Read the dates:
    readings_dates = pd.read_csv(f"{data_dir}/raw_data/{municipality}_readings_sorted.csv", usecols=["dated"]).reindex()

    # Transform the number of weeks to string so we can pass it to the freq parameter of the pd.date_range
    n_weeks_str = str(n_weeks) + "W"
    # Generate the dates for the slpit:
    readings_dates["dated"] = pd.to_datetime(readings_dates["dated"])
    start_date = return_monday(pd.to_datetime(readings_dates["dated"].min()))
    end_date = return_monday(pd.to_datetime(readings_dates["dated"].max()) + pd.Timedelta(days=7))
    range_dates = pd.date_range(start=start_date, end=end_date, freq=n_weeks_str)

    # Convert to timestamps and add the last date of the dataframe:
    range_dates = [pd.Timestamp(return_monday(elem)) for elem in range_dates]
    range_dates.append(pd.Timestamp(end_date))

    # In this list we will store the indexes that will delimitate the batches:
    spliting_indexes = np.array([])
    for i in range(len(range_dates) -1):
        
        temp_df = readings_dates.loc[(range_dates[i] <= readings_dates["dated"]) & (readings_dates["dated"] < range_dates[i+1])]
        if len(temp_df) != 0:
            spliting_indexes = np.append(spliting_indexes, temp_df.index[-1])

    # Add the index zero:
    spliting_indexes = np.insert(spliting_indexes, 0, 0)

    return list(spliting_indexes)

def pre_readings_batches(spliting_indexes_list: list, municipality: str, data_dir: str) -> tuple[pd.DataFrame]:
    """
    Takes the output of the function get_indexes_split and iterates trough the csv file. For each batch of
    data we perform the operations split_readings, pre_power and pre_power_split.
    """
    # Concatenated batches will be added to this dataframes:
    powerReactive = pd.DataFrame()
    powerActive = pd.DataFrame()
    powerReactivePeak = pd.DataFrame()
    powerActivePeak = pd.DataFrame()

    # Name of the columns of the readings df:
    readings_column_names = ["id", "dated", "value", "reading", "component", "componentid"]

    # The readings preprocess must be done in batches because the datasets are too massive for some machines. In case that you 
    # get a Memory error it is a good idea to take make larger the arg range_split_weeks_readings when running the code:
    for i in range(len(spliting_indexes_list)-1):

        # For the first batch of date the function read_csv is able to read the names of the columns from the
        # first row of the csv file
        if i == 0:
            readings = pd.read_csv(
                f"{data_dir}/raw_data/{municipality}_readings_sorted.csv", 
                skiprows = int(spliting_indexes_list[i]), 
                nrows = int(spliting_indexes_list[i+1]-spliting_indexes_list[i]+1)
            )
        # For the other batches the function read_csv does not have acces to the column names so we have to
        # specify them
        else:
            readings = pd.read_csv(
                f"{data_dir}/raw_data/{municipality}_readings_sorted.csv",
                skiprows = int(spliting_indexes_list[i] + 2), 
                nrows = int(spliting_indexes_list[i+1]-spliting_indexes_list[i]),
                names = readings_column_names
            )

        powerReactive_, powerActive_, powerReactivePeak_, powerActivePeak_ = split_readings(readings)
        powerReactive_, powerActive_ = pre_power(powerReactive_, "Reactive"), pre_power(powerActive_, "Active")
        powerReactivePeak_, powerActivePeak_ = pre_power_peak(powerReactivePeak_, "Reactive"), pre_power_peak(powerActivePeak_, "Active")

        powerReactive = pd.concat([powerReactive, powerReactive_])
        powerActive = pd.concat([powerActive, powerActive_])
        powerReactivePeak = pd.concat([powerReactivePeak, powerReactivePeak_])
        powerActivePeak = pd.concat([powerActivePeak, powerActivePeak_])

    return (powerReactive, powerActive, powerReactivePeak, powerActivePeak)