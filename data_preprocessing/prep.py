import numpy as np
import pandas as pd
import xarray as xr
import json
import glob

## Space data
with open('data_retrieval/data-space-fire.json', 'r') as file:
    data = json.load(file)



## Weather data
'''weather_file_names = glob.glob("data_retrieval/*.grib")
weather_subsets = [xr.open_dataset(fp) for fp in weather_file_names]
weather_data = xr.concat(weather_subsets, dim='time')
weather_data.to_netcdf('data_preprocessing/weather_data.nc')'''

weather_data = xr.open_dataset('data_preprocessing/weather_data.nc')

# change to pandas df
weather_data = weather_data.to_dataframe().reset_index()


## Fire predictor
fire_predictor = pd.read_csv('data_retrieval/fire_predictor.csv')
print(fire_predictor)


## Merge
fire_predictor['start_time'] = pd.to_datetime(fire_predictor['start_time'], format='mixed').dt.tz_localize(None)
weather_data['time'] = pd.to_datetime(weather_data['time'], format='mixed').dt.tz_localize(None)

merged = weather_data.merge(fire_predictor, how='inner', left_on=['time', 'latitude', 'longitude'], right_on=['start_time', 'lat', 'lon'])

merged.drop(['Unnamed: 0', 'id', 'start_time', 'end_time', 'lon', 'lat'], axis=1, inplace=True)
merged.to_csv('training_data.csv')