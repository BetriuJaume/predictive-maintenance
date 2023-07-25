# Documentation predictive maintenance
This is the repository for the implementation of the predictive maintenance project for INNERGY. 

## Description of the project:
The main objective of the project is the creation of a machine learning model able to calculate accurate predictions of future breakdowns for both public luminaires (from now on lights) and electric panels (eboxes). The repo contains the whole pipeline, from raw data preprocessing to training the models and making predictions. The code has excellent comments at each step, ensuring a clear understanding of each module.

The model will take as input information about the breakdown history from the last 5 weeks of a light or an ebox together with context information such as electric readings data and weather conditions. The output is the probability of breakdown in the next 4 weeks.

The code has four main functionalities that can be combined depending on the necessities of the user:
* **Meteo web scraping:** Gather the meteorological information from a certain municipality over a specified time period using web scraping.
* **Data preprocessing:** Pipeline to do the necessary data preprocessing of the raw breakdown data, the raw readings data and the scrapped meteorological data.
* **Model training:** Training of the machine learning models using the prepocessed data.
* **Doing predictions:** Gather data from the last 5 weeks, preprocess it and make predictions using the trained models.

Let's explain step-by-step how each one of the modules works:

## Meteo web scrapping
Code in the file scrapy_meteo.py of the folder scrapy_meteo. As we have already mentioned, this module gathers the raw meteorological context data from the website .
To execute the scraper, navigate to the folder scrapy_meteo in the repo and run the following in the terminal:

```
python weather_scrapy.py --di "/local/directory/folder" --mu municipalities_list --dr date_ranges --li list_of_links
```
Let's take a look at the arguments:
* ```--di``` is the local directory of the folder you want to store the data extracted using the scraper.
* ```--mu``` is a list of the names of the municipalities you want to get the data from.
* ```--dr``` is a dictionary where the municipality names serve as keys, and the corresponding values are tuples containing the desired date ranges for data extraction. See the example below for clarification.
* ```--li``` is a list of the links for doing the scraping.

For example, if you want to gather meteorogical information from the Spanish municipalities of Illora, Mejorada and Canyelles you should run this command in the terminal:

```python weather_scrapy.py --di "/home/leibniz/Desktop/IHMAN/meteo_raw_data" --mu illora mejorada canyelles --dr "{'illora': ('2015-01-01', '2023-04-01'), 'mejorada': ('2014-01-01', '2023-04-01'), 'canyelles': ('2015-01-01', '2023-04-01')}" --li "https://www.wunderground.com/history/monthly/es/íllora/LEMG/date/" "https://www.wunderground.com/history/monthly/es/mejorada-del-campo/IMEJOR1/date/" "https://www.wunderground.com/history/monthly/es/canyelles/ICANYE10/date/"```

The models have been trained with data from these municipalities and in these exact date ranges, so if your objective is to gather the information for replicating the training, some of the arguments have implemented default values to make the syntax more clear, so running the following will gather the same data as the previous command:

```python scrapy_meteo.py --di "/home/leibniz/Desktop/IHMAN/meteo_raw_data"```

It is common to get errors when running this code because of connection issues ("AttributeError: 'NoneType' object has no attribute 'find_all'"). We recommend trying to run the code again if you get an error. If the error persists after 3 or 4 tries, it will be better to try to get the information in separate dataframes by running one municipality at a time. If the code keeps crashing after multiple tries, we recommend connecting to a more stable wifi connection.

## Data preprocessing
The code in the file preprocessing_main.py executes the preprocessing of all the data and prepares it for the training phase. For the preprocessing to run you must have three subfolders in the same folder somewhere on your local machine. The first one is the folder **raw_data** where you must have the following files:
* municipality_eboxes_alarms.csv For example, canyelles_eboxes_alarms.csv
* municipality_lights_alarms.csv For example, canyelles_lights_alarms.csv
* municipality_nodes.csv This file contains the information about the eboxes associated with each of the lights. For example, canyelles_nodes.csv
* municipality_readings_sorted.csv This file contains the readings powerActive, powerReactive, powerActivePeak and powerReactivePeak **sorted** by date. For example, canyelles_readings_sorted.csv

The second folder is **meteo_raw_data** which must contain the file:
* new_meteo_municipality.csv This file contains the meteo data that is the output of the scraper. For example, new_meteo_canyelles.csv

Note that you must have each of these files for each of the municipalities that you want to preprocess.

The third folder is **preprocessing_results**. Once you execute the module, the code will look for this folder to drop the preprocessed dataframes.

To execute the data preprocess, run the following command in the terminal:
```
python preprocessing_main.py --di "/local/directory/folder" --mu municipalities_list --ws n_weeks
```
Let's take a look at the arguments:
* ```--di``` is the local directory of the folder where you must have the subfolders raw_data and meteo_raw_data.
* ```--mu``` is a list of the names of the municipalities you want to include in the data preprocessing.
* ```--ws``` is the number of weeks of data that we include in each one of the batches when preprocessing the dataframes of readings. If you get a memory error, consider making this argument smaller. It defaults to 40.

For example, in the case that you want to preprocess data from the Spanish municipalities of Illora, Mejorada and Canyelles with a data split of 40 you should run in the terminal:

```
python preprocessing_main.py --di "/home/leibniz/Desktop/IHMAN" --mu illora mejorada canyelles --ws 40
```

## Model training
For training the model, you have two options. The first one is to use the step-by-step code in the jupyter notebook model_train.ipynb where there is a clear explanation of each step of the process. If you choose this approach, you will be able to execute grid searches and tune the hyperparameters of the model to better fit your training data. The other option is to train the model with a set of hyperparameters optimized to fit the training dataset composed of data from Illora, Mejorada and Canyelles. If you choose the last option, run the following command in the terminal:

```
python train_main.py --di "/local/directory/folder" --de device --st store --rew rewrite --mo model
```
Let's take a look at the arguments:
* ```--di``` is the local directory where you must have the folder preprocessing_results with the output of the preprocessing module.
* ```--de``` is the device for which you want to train a prediction model.
* ```--st``` is a boolean that indicates whether or not you want to store the trained model in the folder predictive_models in the repo.
* ```--rew``` is a boolean that indicates whether or not you want to rewrite the default models that come with the repository.
* ```--mo``` is the type of model you want to train in the case that you have set the argument ```--de``` to light. It is either "default" for a model that uses the electrical readings but is trained with less data or "adboc" for training the Ada Boost Combined Predictor, which uses much more data but does not use the readings.

**Example:** For training a light breakdown prediction model storing the models without rewriting the models, I would run on my machine:

```
python train_main.py --di "/home/leibniz/Desktop/IHMAN" --de "lights" --st True --rew False --mo "adboc"
```
in case I want to train it for eboxes:
```
python train_main.py --di "/home/leibniz/Desktop/IHMAN" --de "eboxes" --st True --rew False
```
### Model selection for lights:
As we have already mentioned, in case that you choose to train a model for lights, we have two options. We can use either the "default" or the "adboc" argument for ```--mo``` to choose between the two. Let's explore the differences:
* The **"default"** model is trained with the whole dataset and has an impressive overall accuracy for the test data and uses the columns from the electrical readings, such as the powerReactivePeak. The lowlights of this model are that it is trained with less data than the "adboc" model (around 120k rows for test and 120k more for validation) and that it has poor performance on detecting sudden errors. These are the errors that appear out of nowhere without having more than 2 weeks of past errors (For example, a light that suffered 2 hours of breakdown in week-4 and hasn't suffered any error in the following weeks until week+4 where it suffers a 6 hour breakdown would be considered a sudden error).
* The **"adboc"** predictor is a combination of two models. The first one is trained with the whole data exactly like the "default" model with it's strengths and it's flaws. The second one is the model that we decided to call the "sudden model". This one is trained exclusively to be able to predict these sudden errors with more accuracy. The predictor uses the "default" model for lights that have been suffering errors for more than 2 weeks and the sudden model for the other cases. This way, we are able to better detect better this sudden errors. This models are trained dropping the reading columns, so we were able to train with much more data (around 300k rows). The flaw of this model is that to be able to detect more sudden errors, we have sacrificed overall accuracy.

## Making predictions:
This module is designed mainly to be implemented in the backend so the user can use the models to generate predictions each week. As we have already mentioned, the trained models use data from the current week and the 4 past weeks, so if the user wants to do predictions you will need to have somewhere in your local a folder called data_to_predict with the following files:
* eboxes_alarms.csv Record of the eboxes alarms for the last 4 weeks and the current week.
* lights_alarms.csv Record of the light alarms for the last 4 weeks and the current week.
* nodes.csv Information of the eboxes associated with each one of the lights.
* powerActive.csv Power active readings for the last 4 weeks and the current week.
* powerReactive.csv Power reactive readings for the last 4 weeks and the current week.
* powerActivePeak.csv Power active peak readings for the last 4 weeks and the current week.
* powerActivePeak.csv Power active peak readings for the last 4 weeks and the current week.
* new_meteo.csv Meteo context data for the last 4 weeks and the current week. This data can be obtained by running the data web scraping module and specifying the desired dates and the municipality.

All this data must be gathered since the Monday of the first week you want to consider util the Sunday of the last week. To obtain the predictions, you must run the following command in the terminal:
```
python predict_main.py --di "/local/directory/folder" --da min_date max_date --mo model
```

For example, in the case that I have gathered data from Monday 2023-04-17 untill Sunday 2023-05-21 and I'm interested in doing predictions with the "default" model, I would run:

```
python predict_main.py --di "/home/leibniz/Desktop/IHMAN" --da "2023-04-17" "2023-05-21"
```

Or in case I'm interested in having a higher detection accuracy of sudden errors, we could use the Ada Boost combined predictor:

```
python predict_main.py --di "/home/leibniz/Desktop/IHMAN" --da "2023-04-17" "2023-05-21" --mo "adboc"
```

**Note:** You have to specify the date ranges for the predictions. Otherwise, the code will crash and you will get an error. Note too that the dates must be passed with the format yyyy/mm/dd.

This code can be easily modified to get the data from SQL queries made to a database instead of scanning csv files for a backend implementation of the models.