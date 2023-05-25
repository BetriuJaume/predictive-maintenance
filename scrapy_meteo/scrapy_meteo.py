from bs4 import BeautifulSoup as BS
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from functools import reduce
import pandas as pd
import time
import argparse
import ast

def render_page(url: str) -> str:

    """
    This function uses the Selenium library to 
    open a web page in a Chrome browser and 
    then returns the HTML source code of the page as a string.

    Argument: url, which is a string that represents the URL of the web page to be loaded
    Return: HTML source code of the page as a string.
    """

    #Create an instance of ChromeOptions, which is a Selenium class that is used to configure options for the Chrome browser.
    options = webdriver.ChromeOptions()
    #Add an option to the ChromeOptions instance to exclude console logs from the Chrome browser. This is useful for reducing unnecessary log noise.
    options.add_experimental_option('excludeSwitches', ['enable-logging'])
    #Create an instance of WebDriver for the Chrome browser using ChromeDriverManager to handle the download and installation of the Chrome driver.
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    #Loads the provided URL in the Chrome browser.
    driver.get(url)
    #Wait for 3 seconds for the page to ensure that all elements of the page are loaded before interacting with them.
    time.sleep(3)
    #Extract the HTML source code of the web page 
    r = driver.page_source
    #Close the Chrome browser
    driver.quit()
    return r


def scrapy_meteo(page: str, years_months: dict) -> pd.DataFrame:
        """
        This function is a web scraper that extracts historical weather data from a website (wunderground.com)

        Arguments: "page" and "years_months". 
                   "page" is a string that represents the base URL of the website where the weather data is located. Ex: 'https://www.wunderground.com/history/monthly/es/canyelles/ICANYE10/date/'
                   "years_months" is a dictionary that contains the years and months for which the data needs to be extracted. Ex: {2019:[8,9,10,11,12], 2020:[1,2,3,4]}
        Return: Csv file with the weather data scraped
        """

        #Create an empty pandas DataFrame to save the output
        output = pd.DataFrame()

        #Loop over the years and months specified in the "years_months" argument
        for y in years_months:
            for m in years_months[y]:
                
                #Construct the URL for each year and month combination to extract the monthly weather data
                url = str(str(page) + str(y) + "-" + str(m))

                #Open a web page in a Chrome browser and return HTML data
                r = render_page(url)

                #Use BeautifulSoup to parse the HTML code of the page.
                soup = BS(r, "html.parser")
                container = soup.find('lib-city-history-observation')
                check = container.find('tbody')

                data = []

                #Search for a specific HTML tag in the parsed HTML code that contains the weather data. If the tag is found, 
                #extract the data and stores it in a list called "data".
                for c in check.find_all('tr', class_='ng-star-inserted'):
                    for i in c.find_all('td', class_='ng-star-inserted'):
                        trial = i.text
                        trial = trial.strip('  ')
                        data.append(trial)

                #Check the length of the "data" list to determine the number of days and number of variables that where collected
                #Then, we save the different meteorological data in different variables (Temperature, Precipitation, Humidity, etc.)
                if round(len(data) / 17 - 1) == 31:
                    Temperature = pd.DataFrame([data[32:128][x:x + 3] for x in range(0, len(data[32:128]), 3)][1:],
                                            columns=['Temp_max', 'Temp_avg', 'Temp_min'])
                    Dew_Point = pd.DataFrame([data[128:224][x:x + 3] for x in range(0, len(data[128:224]), 3)][1:],
                                            columns=['Dew_max', 'Dew_avg', 'Dew_min'])
                    Humidity = pd.DataFrame([data[224:320][x:x + 3] for x in range(0, len(data[224:320]), 3)][1:],
                                            columns=['Hum_max', 'Hum_avg', 'Hum_min'])
                    Wind = pd.DataFrame([data[320:416][x:x + 3] for x in range(0, len(data[320:416]), 3)][1:],
                                        columns=['Wind_max', 'Wind_avg', 'Wind_min'])
                    Pressure = pd.DataFrame([data[416:512][x:x + 3] for x in range(0, len(data[416:512]), 3)][1:],
                                            columns=['Pres_max', 'Pres_avg', 'Pres_min'])
                    Date = pd.DataFrame(data[:32][1:], columns=data[:1])
                    Precipitation = pd.DataFrame(data[512:][1:], columns=['Precipitation'])
                    print(str(str(y) + "-" + str(m) + ' finished!'))
                elif round(len(data) / 17 - 1) == 28:
                    Temperature = pd.DataFrame([data[29:116][x:x + 3] for x in range(0, len(data[29:116]), 3)][1:],
                                            columns=['Temp_max', 'Temp_avg', 'Temp_min'])
                    Dew_Point = pd.DataFrame([data[116:203][x:x + 3] for x in range(0, len(data[116:203]), 3)][1:],
                                            columns=['Dew_max', 'Dew_avg', 'Dew_min'])
                    Humidity = pd.DataFrame([data[203:290][x:x + 3] for x in range(0, len(data[203:290]), 3)][1:],
                                            columns=['Hum_max', 'Hum_avg', 'Hum_min'])
                    Wind = pd.DataFrame([data[290:377][x:x + 3] for x in range(0, len(data[290:377]), 3)][1:],
                                        columns=['Wind_max', 'Wind_avg', 'Wind_min'])
                    Pressure = pd.DataFrame([data[377:464][x:x + 3] for x in range(0, len(data[377:463]), 3)][1:],
                                            columns=['Pres_max', 'Pres_avg', 'Pres_min'])
                    Date = pd.DataFrame(data[:29][1:], columns=data[:1])
                    Precipitation = pd.DataFrame(data[464:][1:], columns=['Precipitation'])
                    print(str(str(y) + "-" + str(m) + ' finished!'))
                elif round(len(data) / 17 - 1) == 29:
                    Temperature = pd.DataFrame([data[30:120][x:x + 3] for x in range(0, len(data[30:120]), 3)][1:],
                                            columns=['Temp_max', 'Temp_avg', 'Temp_min'])
                    Dew_Point = pd.DataFrame([data[120:210][x:x + 3] for x in range(0, len(data[120:210]), 3)][1:],
                                            columns=['Dew_max', 'Dew_avg', 'Dew_min'])
                    Humidity = pd.DataFrame([data[210:300][x:x + 3] for x in range(0, len(data[210:300]), 3)][1:],
                                            columns=['Hum_max', 'Hum_avg', 'Hum_min'])
                    Wind = pd.DataFrame([data[300:390][x:x + 3] for x in range(0, len(data[300:390]), 3)][1:],
                                        columns=['Wind_max', 'Wind_avg', 'Wind_min'])
                    Pressure = pd.DataFrame([data[390:480][x:x + 3] for x in range(0, len(data[390:480]), 3)][1:],
                                            columns=['Pres_max', 'Pres_avg', 'Pres_min'])
                    Date = pd.DataFrame(data[:30][1:], columns=data[:1])
                    Precipitation = pd.DataFrame(data[480:][1:], columns=['Precipitation'])
                    print(str(str(y) + "-" + str(m) + ' finished!'))
                elif round(len(data) / 17 - 1) == 30:
                    Temperature = pd.DataFrame([data[31:124][x:x + 3] for x in range(0, len(data[31:124]), 3)][1:],
                                            columns=['Temp_max', 'Temp_avg', 'Temp_min'])
                    Dew_Point = pd.DataFrame([data[124:217][x:x + 3] for x in range(0, len(data[124:217]), 3)][1:],
                                            columns=['Dew_max', 'Dew_avg', 'Dew_min'])
                    Humidity = pd.DataFrame([data[217:310][x:x + 3] for x in range(0, len(data[217:310]), 3)][1:],
                                            columns=['Hum_max', 'Hum_avg', 'Hum_min'])
                    Wind = pd.DataFrame([data[310:403][x:x + 3] for x in range(0, len(data[310:403]), 3)][1:],
                                        columns=['Wind_max', 'Wind_avg', 'Wind_min'])
                    Pressure = pd.DataFrame([data[403:496][x:x + 3] for x in range(0, len(data[403:496]), 3)][1:],
                                            columns=['Pres_max', 'Pres_avg', 'Pres_min'])
                    Date = pd.DataFrame(data[:31][1:], columns=data[:1])
                    Precipitation = pd.DataFrame(data[496:][1:], columns=['Precipitation'])
                    print(str(str(y) + "-" + str(m) + ' finished!'))
                else:
                    print('Data not in normal length')

                
                #Save the different meteorological data in a compact list 
                dfs = [Date, Temperature, Dew_Point, Humidity, Wind, Pressure, Precipitation]

                #Add exctracted data to a pandas DataFrame
                df_final = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True), dfs)
                #Adds the year and month to the date column of DataFrame
                #**Initially, this column only has the day of the data. 
                df_final['Date'] = str(y) + "-" + str(m) + "-" + df_final.iloc[:, :1].astype(str)

                #Append this Month-Year weather data to the output DataFrame
                output = pd.concat([output, df_final])

                #Wait 10 second to extract the next Year-Month weather data to get rid of being blocked from making requests on the web
                time.sleep(10)

        print('Scraper done!')

        #Add column names to the output
        output = output[['Temp_max', 'Temp_avg', 'Temp_min', 'Dew_max', 'Dew_avg', 'Dew_min', 'Hum_max',
                         'Hum_avg', 'Hum_min', 'Wind_max', 'Wind_avg', 'Wind_min', 'Pres_max',
                         'Pres_avg', 'Pres_min', 'Precipitation', 'Date']]

        return output

def return_years_months(min_date: str, max_date: str) -> dict:
    """
    Generates a dictionary ready to be passed as the argument years_months to the function scrapy_meteo. An example of the output is:
    {2015:[1,2,3,4,5,6,7,8,9,10,11,12], 2016:[1,2,3,4,5,6,7,8,9,10,11,12], 2017:[1,2,3,4,5,6,7,8,9,10,11,12], 2018:[1,2,3,4,5,6,7,8,9,10,11,12], 2019:[1,2,3,4,5,6,7,8,9,10,11,12], 2020:[1,2,3,4,5,6,7,8,9,10,11,12], 2021:[1,2,3,4,5,6,7,8,9,10,11,12], 2022: [1,2,3,4,5,6,7,8,9,10,11,12], 2023:[1,2,3,4,5,6,7,8,9,10,11,12]}
    As can see the functon returns the years and the months that we want to scrap for each one of the municipalities.
    Note that the function returns the dictionary with all the months of the years that you pass instead of cutting at a certain month

    Important!
    The format of the dates should always be yyyy/mm/dd
    """

    # Get the years and months of the date range:
    min_year = int(min_date[0:4])
    max_year = int(max_date[0:4])
    min_month = int(min_date[5:7])
    max_month = int(max_date[5:7])

    # Until max_year:
    years_in_between = list(range(min_year, max_year+1))

    dict_r = dict()
    for yr in years_in_between:
        if min_year == max_year:
            dict_r[yr] = list(range(min_month, max_month +1))
        elif yr == min_year:
            #  If we are looking at the first year we just need the month untill the last month (month 12)
            dict_r[yr] = list(range(min_month, 12 + 1))
        elif yr == max_year:
            # If we are looking at the last year we will just need the months untill the max_month
            dict_r[yr] = list(range(1, max_month + 1))
        else:
            # get all the months in the years in between:
            dict_r[yr] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    return dict_r

def run_scrapy(list_municipalities: list, list_links_wunderground: list, dict_min_max_dates: dict[tuple], data_dir: str) -> None:

    for page, munip in zip(list_links_wunderground, list_municipalities):

        # Get dates and the dict of years: 
        date_min, date_max = dict_min_max_dates[munip]
        years_months = return_years_months(date_min, date_max)
        
        # run the scrapy code:
        df_output = scrapy_meteo(page, years_months)
        df_output.to_csv(f"{data_dir}/new_meteo_{munip}.csv")

    # Links:
    # https://www.wunderground.com/history/monthly/es/canyelles/ICANYE10/date/
    # https://www.wunderground.com/history/monthly/es/íllora/LEMG/date/
    # https://www.wunderground.com/history/monthly/es/mejorada-del-campo/IMEJOR1/date/

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--di", "--meteo_data_folders_directory", help="Provide the local directory on your machine where you want to store the scraped data. Example: /home/leibniz/Desktop/IHMAN/meteo_raw_data")
    parser.add_argument("--mu", "--list_municipalities", help="List of the municipalities the user wants to scrap", nargs='+', default=["illora", "mejorada", "canyelles"])
    parser.add_argument(
        "--dr", 
        "--date_ranges_dict", 
        help = "Pass in dictionary form the range of dates for each one of the municipalities. See default value for example", 
        default="{'illora': ('2015-01-01', '2023-04-01'), 'mejorada': ('2014-01-01', '2023-04-01'), 'canyelles': ('2015-01-01', '2023-04-01')}"
    )
    parser.add_argument(
        "--li",
        "--scrapy_link",
        help = "List of data directories",
        nargs = '+', 
        default = [
            "https://www.wunderground.com/history/monthly/es/íllora/LEMG/date/", 
            "https://www.wunderground.com/history/monthly/es/mejorada-del-campo/IMEJOR1/date/",
            "https://www.wunderground.com/history/monthly/es/canyelles/ICANYE10/date/"
        ]
    )

    args = parser.parse_args()
    municipalities_list = args.mu
    dict_min_max_dates = ast.literal_eval(args.dr)
    links = args.li
    data_dir = args.di

    # Canyelles is not working properly:
    run_scrapy(
        list_municipalities = municipalities_list,
        list_links_wunderground = links,
        dict_min_max_dates = dict_min_max_dates,
        data_dir = data_dir
    )