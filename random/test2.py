import numpy as np
from datetime import datetime
import pytz
import pyfes

# Load the updated configuration
print("Loading config...")
handlers = pyfes.load_config('C:\\Users\\psteeves\\coastal\\fes2022.yaml')
print(handlers)
ocean_tide = handlers['tide']
load_tide = handlers['radial']

# Coordinates (ensure longitude is in [0, 360])
centroid = [
          -123.46998890336721 + 360,
          48.65349688063148
        ]
test_date = [datetime(2022, 1, 1, tzinfo=pytz.utc)]
dates_np = np.array([np.datetime64(date) for date in test_date])
lons = np.array([centroid[0]])
lats = np.array([centroid[1]])

# Evaluate ocean tide
ocean_short, ocean_long, flags_ocean = pyfes.evaluate_tide(
    ocean_tide, dates_np, lons, lats, num_threads=1
)
print(f"Flags (Ocean): {flags_ocean}")

# Evaluate load tide
load_short, load_long, flags_load = pyfes.evaluate_tide(
    load_tide, dates_np, lons, lats, num_threads=1
)
print(f"Flags (Load): {flags_load}")

# Compute total tide level
tide_level = (ocean_short + ocean_long + load_short + load_long) / 100  # Convert cm to meters
print(f"Tide Level: {tide_level} meters")
print(centroid)