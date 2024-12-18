import pyfes
import pandas as pd
import os
import pytz
from datetime import datetime, timedelta
from coastsat import SDS_slope


def load_config():
    print("Loading FES2022 config file...")
    config_filepath = os.pardir
    config = os.path.join(config_filepath, 'fes2022_clipped.yaml')
    handlers = pyfes.load_config(config)
    print("Config file loaded")
    ocean_tide = handlers['tide']
    load_tide = handlers['radial']

    return ocean_tide, load_tide


def write_to_csv(filename, dates_ts, tides_ts):
    tide_data = {'dates':dates_ts, 'tide':tides_ts}
    df = pd.DataFrame(tide_data)
    df.to_csv(filename, index=False)
    print(f"Tide data saved to {filename}")


def get_tide_data_at_point(ocean_tide, load_tide, date_range, centroid):
    if centroid[0] < 0: centroid[0] += 360
    timestep = 900
    dates_ts, tides_ts = SDS_slope.compute_tide(centroid, date_range, timestep, ocean_tide, load_tide)

    return dates_ts, tides_ts


if __name__ == '__main__':
    sites = {'Saanich_Peninsula': [-123.39250498953294, 48.629795636410165], 
             'Metlakatla': [-130.50867183985804, 54.353815728711], 
             'Masset_Tow_Hill': [-132.07556332102195, 54.056721827283184], 
             'Pullen_North_Head': [-134.29185240637477, 69.73942468610863], 
             'Tuktoyaktuk': [-133.05458108323464, 69.47195488187853], 
             'Parksville': [-124.30507856120093, 49.35013759879706],
             'Rose_Spit': [-131.5817801910054, 54.23965681093256]}
    
    ocean_tide, load_tide = load_config()

    date_range = [pytz.utc.localize(datetime(2001, 1, 1)),
                  pytz.utc.localize(datetime(2025, 1, 1))]
    
    for site, centroid in sites.items():
        filename = os.path.join(os.getcwd(), 'tides_clipped', f'{site}_tides.csv')
        dates_ts, tides_ts = get_tide_data_at_point(ocean_tide, load_tide, date_range, centroid)
        write_to_csv(filename, dates_ts, tides_ts)