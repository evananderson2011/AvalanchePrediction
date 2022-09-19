# eanderson42
# dec 2nd
# Avvy Prediction Project Data Exploration

import pandas as pd
import numpy as np

#Avvy data cleaning:
clean_avv_data = False
write_avvy = False

if clean_avv_data:
    avvy_path = 'Data/CAIC_avalanches_2011_10_to_2021_10.csv'
    avvy_data = pd.read_csv(avvy_path)
    avvy_data.columns = [c.replace(' ', '_') for c in avvy_data.columns]
    # print(avvy_data.head())

    sizes = ['R3', 'R4', 'R5']
    size_bool = avvy_data.Rsize.isin(sizes)

    av_large = avvy_data[size_bool]

    # print(av_large.BC_Zone.value_counts())
    #
    # Northern San Juan        741 - Idarado
    # Front Range              453 - Loveland Basin
    # Southern San Juan        391 - Red Mountain Pass
    # Gunnison                 385 - Schofield Pass
    # Vail & Summit County     356 - Vail Mountain
    # Aspen                    333 - Independence Pass


    bc_zones = ['Northern San Juan', 'Front Range', 'Southern San Juan', 'Gunnison', 'Vail & Summit County', 'Aspen']
    zone_bool = av_large.BC_Zone.isin(bc_zones)

    av_clean = av_large[zone_bool]

    # print(av_clean.columns)
    # These cols will be less relevant, really just need to know if one happened.
    rel_cols = ['Date', 'BC_Zone', 'Elev', 'Asp', 'Type', 'Trigger', 'Rsize', 'Dsize',
     'Avg_Slope_Angle', 'Start_Zone_Elev', 'Start_Zone_Elev_units', 'Sliding_Sfc',
     'Weak_Layer', 'Weak_Layer_Type', 'Avg_Width', 'Max_Width', 'Width_units']

    av_clean = av_clean[rel_cols]
    av_clean['av_bol'] = 1

    if write_avvy and clean_avv_data:
        avvy_path_out = 'Data/CAIC_clean.csv'
        av_clean.to_csv(avvy_path_out)

# Creating a simple avvy set with date, location, and boolean

simple_avv_data = False
write_simple_avvy = False

if simple_avv_data:
    av_clean_path = 'Data/CAIC_clean.csv'
    av_simple = pd.read_csv(av_clean_path)

    av_simple = av_simple[['Date', 'BC_Zone', 'av_bol']]
    av_simple = av_simple.drop_duplicates()

    bc_to_station = pd.DataFrame(
        {
            'BC_Zone': ["Northern San Juan", "Front Range", "Southern San Juan",
                        "Gunnison", "Vail & Summit County", "Aspen"],
            'Station': ["Idarado", "Loveland Basin", "Red Mountain Pass",
                        "Schofield Pass", "Vail Mountain", "Independence Pass"]

        }
    )
    frames = [av_simple, bc_to_station]
    av_simple = pd.merge(left=av_simple, right=bc_to_station, how='inner', left_on='BC_Zone', right_on='BC_Zone')
    # print(av_simple.head(20))

    if write_simple_avvy and write_simple_avvy:
        avvy_path_out = 'Data/CAIC_simple.csv'
        av_simple.drop
        av_simple.to_csv(avvy_path_out, index=False)


clean_weather_data = False
write_w = False

# Weather File

# Data items provided in this file:
#
# Element Name                        Value Type  Function Type  Function Duration             Base Data  Measurement Units   Sensor Depth  Element Code  Description
# Air Temperature Average             Value       None           Day                           N/A        Degrees fahrenheit  N/A           TAVG          Average air temperature - sub-hourly sampling frequency
# Air Temperature Maximum             Value       None           Day                           N/A        Degrees fahrenheit  N/A           TMAX          Maximum air temperature - sub-hourly sampling frequency
# Air Temperature Minimum             Value       None           Day                           N/A        Degrees fahrenheit  N/A           TMIN          Minimum air temperature - sub-hourly sampling frequency
# Precipitation Accumulation          Value       None           Instantaneous - Start of Day  N/A        Inches              N/A           PREC          Water year accumulated precipitation
# Precipitation Increment             Value       None           Day                           N/A        Inches              N/A           PRCP          Total precipitation
# Precipitation Increment - Snow-adj  Value       None           Day                           N/A        Inches              N/A           PRCPSA        Snow adjusted total preciptation
# Precipitation Month-to-date         Value       None           Instantaneous - Start of Day  N/A        Inches              N/A           PRCPMTD       Month-to-date precipitation
# Snow Depth                          Value       None           Instantaneous - Start of Day  N/A        Inches              N/A           SNWD          Total snow depth
# Snow Density                        Value       None           Instantaneous - Start of Day  N/A        Percent             N/A           SNDN          The mass of snow per unit volume, expressed as a percent of water content
# Snow Rain Ratio                     Value       None           Day                           N/A        Unitless            N/A           SNRR          Daily or sum of daily snow water equivalent increases as pct of total precip for the same period
# Wind Direction Average              Value       None           Day                           N/A        Degrees             N/A           WDIRV         Average wind direction
# Wind Speed Average                  Value       None           Day                           N/A        Miles/hour          N/A           WSPDV         Average wind speed
# Wind Speed Maximum                  Value       None           Day                           N/A        Miles/hour          N/A           WSPDX         Maximum wind speed


if clean_weather_data:
    w_path = 'Data/weather_data.csv'
    w_data = pd.read_csv(w_path)

    w_data.columns = [c.replace(' ', '_') for c in w_data.columns]

    new_columns = ['Date', 'Station_Id', 'Station_Name', 'Air_Temp_Avg_degF',
     'Air_Temp_Max_degF', 'Air_Temp_Min_degF', 'Precip_Accum_in_',
     'Precip_Increment_in', 'Precip_Increment_Snow_in', 'Precip_MTD_in',
     'Snow_Depth_in', 'Snow_Density_pct', 'Snow_Rain_Ratio',
     'Wind_Dir_Avg_degree', 'Wind_Speed_Avg_mph', 'Wind_Speed_Max_mph']

    w_data.columns = new_columns

    denstiy_bool = w_data['Snow_Density_pct'].notnull()
    w_clean = w_data[denstiy_bool]

    drop_wind_cols = ['Date', 'Station_Id', 'Station_Name', 'Air_Temp_Avg_degF',
     'Air_Temp_Max_degF', 'Air_Temp_Min_degF', 'Precip_Accum_in_',
     'Precip_Increment_in', 'Precip_Increment_Snow_in', 'Precip_MTD_in',
     'Snow_Depth_in', 'Snow_Density_pct', 'Snow_Rain_Ratio']
    w_clean = w_clean[drop_wind_cols]

    # print(w_data.head())

    if write_w and clean_weather_data:
        w_path_out = 'Data/weather_clean.csv'
        w_clean.to_csv(w_path_out, index=False)


# Manipulate weather data to add first snow, etc.
prep_weather_data = False
write_agg_weather = False

if prep_weather_data:
    w_path = 'Data/weather_clean.csv'
    w_data = pd.read_csv(w_path)

    w_data['Date'] = w_data['Date'].astype('datetime64[ns]')
    w_data = w_data.set_index('Date')
    w_data = w_data.sort_index()
    # print(w_data.index)


    # first snow - how do I quanitfy this in a mchine readable way?

    #Priors:

    """
    ['Date', 'Station_Id', 'Station_Name', 'Air_Temp_Avg_degF',
     'Air_Temp_Max_degF', 'Air_Temp_Min_degF', 'Precip_Accum_in_',
     'Precip_Increment_in', 'Precip_Increment_Snow_in', 'Precip_MTD_in',
     'Snow_Depth_in', 'Snow_Density_pct', 'Snow_Rain_Ratio']
    """
    def priors(days, data, column, agg='sum'):
        days = str(days) + 'd'
        if agg == 'sum':
            priors = data[column].rolling(days, min_periods=1).sum()
        if agg == 'avg':
            priors = data[column].rolling(days, min_periods=1).mean()
        if agg == 'min':
            priors = data[column].rolling(days, min_periods=1).min()
        if agg == 'max':
            priors = data[column].rolling(days, min_periods=1).max()

        return priors


    stations = ["Idarado", "Loveland Basin", "Red Mountain Pass",
                        "Schofield Pass", "Vail Mountain", "Independence Pass"]

    w_agg_out_data = 24

    for station in stations:
        station_bool = w_data['Station_Name'] == station
        station_data = w_data.loc[station_bool]

        # print(station)
        # print(station_data.head())

        station_data = station_data.sort_index()

        station_data['percip_prior_2days'] = priors(days=2, data=station_data, column='Precip_Accum_in_', agg='sum')
        station_data['percip_prior_4days'] = priors(days=4, data=station_data, column='Precip_Accum_in_', agg='sum')
        station_data['percip_prior_7days'] = priors(days=7, data=station_data, column='Precip_Accum_in_', agg='sum')
        station_data['percip_prior_14days'] = priors(days=14, data=station_data, column='Precip_Accum_in_', agg='sum')

        station_data['temp_min_prior_2days'] = priors(days=2, data=station_data, column='Air_Temp_Min_degF', agg='min')
        station_data['temp_min_prior_4days'] = priors(days=4, data=station_data, column='Air_Temp_Min_degF', agg='min')
        station_data['temp_min_prior_7days'] = priors(days=7, data=station_data, column='Air_Temp_Min_degF', agg='min')
        station_data['temp_min_prior_14days'] = priors(days=14, data=station_data, column='Air_Temp_Min_degF', agg='min')

        station_data['temp_max_prior_2days'] = priors(days=2, data=station_data, column='Air_Temp_Max_degF', agg='max')
        station_data['temp_max_prior_4days'] = priors(days=4, data=station_data, column='Air_Temp_Max_degF', agg='max')
        station_data['temp_max_prior_7days'] = priors(days=7, data=station_data, column='Air_Temp_Max_degF', agg='max')
        station_data['temp_max_prior_14days'] = priors(days=14, data=station_data, column='Air_Temp_Max_degF', agg='max')

        station_data['snow_density_avg_prior_2days'] = priors(days=2, data=station_data, column='Snow_Density_pct', agg='avg')
        station_data['snow_density_avg_prior_4days'] = priors(days=4, data=station_data, column='Snow_Density_pct', agg='avg')
        station_data['snow_density_avg_prior_7days'] = priors(days=7, data=station_data, column='Snow_Density_pct', agg='avg')
        station_data['snow_density_avg_prior_14days'] = priors(days=14, data=station_data, column='Snow_Density_pct', agg='avg')

        # print(station_data.columns)

        if type(w_agg_out_data) != type(pd.DataFrame()):
            w_agg_out_data = station_data
        else:
            w_agg_out_data = w_agg_out_data.append(station_data)

    if prep_weather_data and write_agg_weather:
        w_path_out = 'Data/weather_clean_aggd.csv'
        w_agg_out_data.to_csv(w_path_out)


# Combine avy and weather data


combine_data = False

if combine_data:
    weather_path = 'Data/weather_clean_aggd.csv'
    av_path = 'Data/CAIC_simple.csv'

    path_out_full = 'Data/av_data.csv'

    weather = pd.read_csv(weather_path)
    weather['Date'] = weather['Date'].astype('datetime64[ns]')

    av = pd.read_csv(av_path)
    av['Date'] = av['Date'].astype('datetime64[ns]')

    full_data = pd.merge(left=weather, right=av, how='outer', left_on=['Date', 'Station_Name'], right_on=['Date', 'Station'])

    columns = ['Date', 'Station_Id', 'Station_Name', 'Air_Temp_Avg_degF',
       'Air_Temp_Max_degF', 'Air_Temp_Min_degF', 'Precip_Accum_in_',
       'Precip_Increment_in', 'Precip_Increment_Snow_in', 'Precip_MTD_in',
       'Snow_Depth_in', 'Snow_Density_pct', 'Snow_Rain_Ratio',
       'percip_prior_2days', 'percip_prior_4days', 'percip_prior_7days',
       'percip_prior_14days', 'temp_min_prior_2days', 'temp_min_prior_4days',
       'temp_min_prior_7days', 'temp_min_prior_14days', 'temp_max_prior_2days',
       'temp_max_prior_4days', 'temp_max_prior_7days', 'temp_max_prior_14days',
       'snow_density_avg_prior_2days', 'snow_density_avg_prior_4days',
       'snow_density_avg_prior_7days', 'snow_density_avg_prior_14days',
       'BC_Zone', 'av_bol', 'Station']

    rel_cols_full = ['Date', 'Air_Temp_Avg_degF',
       'Air_Temp_Max_degF', 'Air_Temp_Min_degF', 'Precip_Accum_in_',
       'Precip_Increment_in', 'Precip_Increment_Snow_in', 'Precip_MTD_in',
       'Snow_Depth_in', 'Snow_Density_pct', 'Snow_Rain_Ratio',
       'percip_prior_2days', 'percip_prior_4days', 'percip_prior_7days',
       'percip_prior_14days', 'temp_min_prior_2days', 'temp_min_prior_4days',
       'temp_min_prior_7days', 'temp_min_prior_14days', 'temp_max_prior_2days',
       'temp_max_prior_4days', 'temp_max_prior_7days', 'temp_max_prior_14days',
       'snow_density_avg_prior_2days', 'snow_density_avg_prior_4days',
       'snow_density_avg_prior_7days', 'snow_density_avg_prior_14days',
       'av_bol']

    full_data = full_data[rel_cols_full]

    full_data['av_bol'] = full_data['av_bol'].fillna(0)

    full_data['Snow_Rain_Ratio'] = full_data['Snow_Rain_Ratio'].fillna(0)

    full_data.to_csv(path_out_full, index=False)








