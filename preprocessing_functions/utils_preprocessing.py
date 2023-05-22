import datetime

def return_monday(my_datetime: datetime) -> datetime:
    """
    Returns the date of the monday of the week this function will appear many times in the code
    """
    return my_datetime.date() - datetime.timedelta(days=my_datetime.weekday())

def return_meteo_filter_dates(municipality: str) -> tuple[str]:
    """
    The function returns the date_min and date_max needed for the function function_preprocessing_meteo.first_meteo_preprocess for each one of the
    municipalities. If you want to run your code with another municipality not registered in the following dictionary there are two options: Add the
    data to the dictionary in the correct format or don't do anything and the function will return a default value.
    """

    municip_dates = {
        "mejorada": ("2014-01-01", "2100-01-01"),
        "illora": ("2014-01-01", "2100-01-01"),
        "canyelles": ("2014-01-01", "2100-01-01")
    }

    if municipality in municip_dates.keys():
        return municip_dates[municipality]
    else:
        # Default case:
        return ("2015-06-08", "2100-01-01")

