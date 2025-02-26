# %% [markdown]
# # Ordering Planet Imagery
# 

# %% [markdown]
# This tutorial uses [Planet](https://www.planet.com)'s Data and Orders API using the official [Python client](https://github.com/planetlabs/planet-client-python), the `planet` module. We will search for imagery while filtering for clouds based on a specific AOI.
# 
# ## Requirements
# 
# This tutorial assumes familiarity with the [Python](https://python.org) programming language throughout. Python modules used in this tutorial are:
# * [IPython](https://ipython.org/) and [Jupyter](https://jupyter.org/)
# * [planet](https://github.com/planetlabs/planet-client-python)
# * [geojsonio](https://pypi.python.org/pypi/geojsonio)
# * [rasterio](https://rasterio.readthedocs.io/en/latest/index.html)
# * [shapely](https://shapely.readthedocs.io/en/stable/index.html)
# * [asyncio](https://docs.python.org/3/library/asyncio.html)
# 
# You should also have an account on the Planet Platform and retrieve your API key from your [account page](https://www.planet.com/account/).
# 
# ## Useful links 
# * [Planet Client V2 Documentation](https://github.com/planetlabs/planet-client-python)
# * [Planet Data API reference](https://developers.planet.com/docs/apis/data/)

# %% [markdown]
# ## Set up
# 
# In order to interact with the Planet API using the client, we need to import the necessary packages & define helper functions.

# %%
#general packages
from dotenv import load_dotenv
import os
import shutil
import pytz

import json
import asyncio
import rasterio
import numpy as np
import nest_asyncio
from datetime import datetime
from collections import Counter
import datetime
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime
import pandas as pd
from datetime import datetime
from dateutil import parser
from collections import defaultdict
import requests
from planet import Session

#geospatial packages
from collections import defaultdict
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
import geopandas as gpd
from rasterio.mask import mask
from shapely.geometry import shape
from shapely.ops import unary_union
from shapely.geometry import mapping

#planet SDK
from planet import Auth
from planet import Session, data_filter
from planet import order_request, OrdersClient

load_dotenv()


# We will also create a small helper function to print out JSON with proper indentation.
def indent(data):
    print(json.dumps(data, indent=2))

# %% [markdown]
# We next need to create a `client` object registered with our API key. The API key will be automatically read from the `PL_API_KEY` environment variable if it exists. You can also authenticate via the CLI using [`auth init`](https://planet-sdk-for-python-v2.readthedocs.io/en/latest/cli/cli-reference/?h=auth#auth:~:text=message%20and%20exit.-,auth,-%C2%B6), this will store your API key as an environment variable.

# %%
API_KEY = os.getenv('PL_API_KEY')

print(f"API Key Loaded: {API_KEY}")

# %%

client = Auth.from_key(API_KEY)

# %% [markdown]
# ## Searching for available imagery
# 
# We can search for items that are interesting by using the `quick_search` member function. Searches, however, always require a proper request that includes a filter that selects the specific items to return as seach results.

# %% [markdown]
# Let's also read in a GeoJSON geometry into a variable so we can use it during testing. The geometry can only have one polygon to work with the data API

# %%
with open("tofino.geojson") as f:
    geom_all = json.loads(f.read())


# %% [markdown]
# ### Filters

# %% [markdown]
# The possible filters include `and_filter`, `date_range_filter`, `range_filter` and so on, mirroring the options supported by the Planet API. Additional filters are described [here](https://planet-sdk-for-python-v2.readthedocs.io/en/latest/python/sdk-guide/#filter:~:text=(main())-,Filter,-%C2%B6).

# %%
# Define the filters we'll use to find our data
item_types = ["PSScene"]

#Geometry filter
geom_filter = data_filter.geometry_filter(geom_all)

#Date range filter
date_range_filter = data_filter.date_range_filter("acquired", gt = datetime(month=1, day=1, year=2023), lt = datetime(month=12, day=31, year=2024))

#Cloud cover filter
clear_percent_filter = data_filter.range_filter('clear_percent', 60 ,None)

#String filter for available asset
asset_filter = data_filter.asset_filter(["ortho_analytic_4b"])

#Combine all of the filters
combined_filter = data_filter.and_filter([geom_filter, clear_percent_filter,asset_filter, date_range_filter])

# %%
combined_filter

# %% [markdown]
# Now let's build the request, it will return the ID needed to execute the search:

# %%
search_name = 'tofino_refined'

async with Session() as sess:
    cl = sess.client('data')
    request = await cl.create_search(name=search_name, search_filter=combined_filter, item_types=item_types)
    request['id']

# %% [markdown]
# The limit paramter allows us to limit the number of results from our search that are returned. Set the limit to 0 if you want all possible returns but be careful that you are not creating too large of a request that will time out. For a couple thousand items this will take about half a minute.

# %%
async with Session() as sess:
    cl = sess.client('data')
    items = cl.run_search(search_id=request['id'], limit=10000)
    item_list = [i async for i in items]

# %% [markdown]
# The number of available images within your AOI and TOI with greater then 60% clear pixels

# %%
print(len(item_list))

# %% [markdown]
# ## AOI coverage filtering
# 
# First stage is to group all of the imagery by day

# %%
grouped = defaultdict(list)

for item in item_list:
    # Extract the ISO 8601 date-time string
    acquired_time = item["properties"]["acquired"]
    # Parse out just the date portion
    date_str = acquired_time.split("T")[0]

    grouped[date_str].append(item)

print("Days of imagery:", len(grouped.keys()))


# %% [markdown]
# Next compare the daily image coverage with your AOI, filtering out any day with lass coverage then the desired threshold

# %%
aoi_coverage_threshold = 0.9


# Convert the GeoJSON geometry to a shapely geometry object
aoi_geom = unary_union([shape(feature['geometry']) for feature in geom_all['features']])
geom_area = aoi_geom.area
print("Geom Area: " + str(geom_area))

valid_days = {}
merged_features = []

for day, items in grouped.items():
    shapely_geoms = []
    for meta in items:
        if 'geometry' in meta:
            shapely_geoms.append(shape(meta['geometry']))
    if not shapely_geoms:
        print(f"No valid geometries for {day}")
        continue

    merged_geom = unary_union(shapely_geoms)
    intersection_area = merged_geom.intersection(aoi_geom).area
    overlap_ratio = intersection_area / geom_area

    if overlap_ratio >= aoi_coverage_threshold:
        valid_days[day] = items
        merged_features.append({
            "type": "Feature",
            "properties": {
                "day": day,
                "overlap_ratio": overlap_ratio
            },
            "geometry": mapping(merged_geom.intersection(aoi_geom))
        })

print("Days passing 50% overlap criterion:", len(valid_days.keys()))

merged_collection = {
    "type": "FeatureCollection",
    "features": merged_features
}

with open("merged_geoms.geojson", "w") as f:
    json.dump(merged_collection, f)

# %% [markdown]
# Aggregate all of the days together into a set of Image IDs

# %%
valid_IDs = []
for day, items in valid_days.items():
    for item in items:
        valid_IDs.append(item['id'])

print("Number of valid IDs: ", len(valid_IDs))

# %% [markdown]
# ## AOI based cloud filtering
# The data api searches based on scene wide cloud statistics which means that a scene that is occasionally 60% clear might have clouds covering most of your AOI. We can do AOI level cloud filtering but it takes a bit longer so its good to filter down first with a normal data API call.

# %%
def create_cloud_filter_request(item_id):
    url = f"https://api.planet.com/data/v1/item-types/PSScene/items/{item_id}/coverage"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "geometry": geom_all['features'][0]['geometry']
    }

    return requests.post(url, headers=headers, json=data, auth=(API_KEY, ""))
    

# %%
item_cloud = {}

# %% [markdown]
# Itterate over all of the images found using the data API, making a request for AOI based cloud cover.
# 
# Could cover requests take about 10 min to be processed on Planets side. The first time running will place the request and once activated the request will return a cloud cover value. Placing the request for 300 images takes approximately 10 min. 
# 
# Problematic files that do not get a 200 code will be stored in the problematic dictionary and can be delt with on a case by case basis.
# 
# Rerun this code again until there are no more files in the activated stage

# %%
# For 500 images this took 20 minutes
i = 1
total_files = len(valid_IDs)
problematic = {}
for item_id in valid_IDs:

    print(f"Processing file {i}/{total_files}", end="\r") 
    if item_id not in item_cloud:
        request = create_cloud_filter_request(item_id)
        if request.status_code == 200:
            item_cloud[item_id] = request
        else:
            problematic[item_id] = request

    elif item_cloud[item_id].status_code == 200:
        if item_cloud[item_id].json()["status"] == 'activating':
            request = create_cloud_filter_request(item_id)
            if request.status_code == 200:
                item_cloud[item_id] = request
            else:
                problematic[item_id] = item_cloud
    elif item_cloud[item_id].status_code == 500:
        request = create_cloud_filter_request(item_id)
        if request.status_code == 200:
            item_cloud[item_id] = request
        else:
            problematic[item_id] = request

    i += 1

for item in item_cloud:
    if item_cloud[item].status_code != 200:
        print(item_cloud[item].status_code)

count_complete = sum(resp.json()['status'] == 'complete' for resp in item_cloud.values())
print(f"\nItems completed: {count_complete}")
count_complete = sum(resp.json()['status'] == 'activating' for resp in item_cloud.values())
print(f"Items completed: {count_complete}")
print(f"Items with errors: {len(problematic)}")

# %% [markdown]
# ## Filter out the imagery based on the AOI level clear percent

# %%
order_imagery = {}
cloudy_imagery = {}
clear_percent_threshold = 80


for item in list(item_cloud.keys()):
    if item_cloud[item].status_code == 200:
        if item_cloud[item].json()["status"] == 'complete':
            if item_cloud[item].json()['clear_percent'] > clear_percent_threshold:
                order_imagery[item] = item_cloud[item].json()['clear_percent']
            else:
                cloudy_imagery[item] = item_cloud[item].json()['clear_percent']

print("Number of suitible images: " + str(len(order_imagery)))
print("Number of images exceding cloud cover: " + str(len(cloudy_imagery)))

# %% [markdown]
# # Visualize image distribution

# %%
# Extract year and month from image IDs
year_months = [img_id[:6] for img_id in order_imagery]

# Count the frequency of each year-month
year_month_counts = Counter(year_months)

# Convert year-month strings to datetime objects for plotting
year_month_objects = [datetime.datetime.strptime(year_month, '%Y%m') for year_month in year_month_counts.keys()]
counts = list(year_month_counts.values())

# Create a histogram
plt.figure(figsize=(12, 6))
plt.bar(year_month_objects, counts, width=20, edgecolor='black')
plt.xlabel('Year-Month')
plt.ylabel('Number of Images')
plt.title('Frequency of Images Captured in Each Month')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## Extract the clearest dates every month

# %%
# 1) Group images by year-month-day
monthly_groups = defaultdict(lambda: defaultdict(list))
for image_id, cloud_cover in order_imagery.items():
    # First 8 characters = YYYYMMDD
    date_str = image_id[:8]
    dt = datetime.strptime(date_str, "%Y%m%d")  # parse into a datetime
    year_month = dt.strftime("%Y-%m")           # e.g. "2024-01"
    day = dt.strftime("%d")
    
    monthly_groups[year_month][day].append((image_id, cloud_cover))


# 2) For each year-month, sort by cloud cover and pick up to 3
selected_images = {}
for ym, days in monthly_groups.items():
    # Sort by ascending cloud cover
    for day, images in days.items():
        average_cloud = sum(img[1] for img in images) / len(images)
        monthly_groups[ym][day] = [images,average_cloud]
# 3) Select up to 3 days with the clearest cover for each year-month
for ym, days in monthly_groups.items():
    # Sort days by average cloud cover
    sorted_days = sorted(days.items(), key=lambda x: x[1][1], reverse=True)
    # Select up to 3 days with the clearest cover
    for i in range(3):
        if i < len(sorted_days):
            date = ym + "-" + sorted_days[i][0]
            ids = []
            for id in sorted_days[i][1][0]:
                ids.append(id[0])
            selected_images[date] = ids
            print(date)
            print(selected_images[date])
    # for day, (images, avg_cloud) in sorted_days[:3]:
    #     selected_images.append()


print("Number of days selected: " + str(len(selected_images)))

# %% [markdown]
# ## Select the images with the most similar tides

# %%
tide_csv = "user_inputs/masset_tides.csv"
target_tide=2.0
max_images=3


# 1) Load tide data
tide_df = pd.read_csv(tide_csv)
# Ensure columns exist
required_cols = {"dates", "tide"}

# 2) Parse datetime from CSV's date & time
tide_df["datetime"] = tide_df["dates"].apply(parser.parse)

tide_df.set_index("datetime", inplace=True)

# Create a dictionary {datetime: tide_height} for fast lookup
tide_dict = tide_df["tide"].to_dict()


# %% [markdown]
# For 300 images this step will take about 10 min

# %%
image_ids = list(order_imagery.keys())

# 3) For each image ID, parse date/time, look up tide, store by month
monthly_groups = defaultdict(list)
i = 1
for image_id in image_ids:

    # Extract date/time from image ID
    # first 8 chars: YYYYMMDD, next 6 chars: HHMMSS
    date_str = image_id[:8]
    time_str = image_id[9:15]
    
    try:
        dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M%S")
        dt = dt.replace(tzinfo=pytz.UTC)  # Make datetime object timezone-aware

    except ValueError:
        # If the image ID doesn't match the expected format, skip or handle differently
        print("Wrong Format")
        continue
    if dt > datetime(2025, 1, 1, tzinfo=pytz.UTC):
        continue
    closest_time = min(tide_dict.keys(), key=lambda t: abs(t - dt))
    tide_height = tide_dict[closest_time]

    diff = abs(tide_height - target_tide)
    print(f" -{i}/{str(len(image_ids))}- {image_id}: time={dt}, closest time={closest_time}, tide={tide_height}, difference={diff}")
    i += 1
    # Group by month (YYYY-MM)
    year_month = dt.strftime("%Y-%m")
    monthly_groups[year_month].append((image_id, tide_height, diff))


print("selecting images")
# 4) Select up to `max_images` images with the smallest difference per month
selected = []
for ym, entries in monthly_groups.items():
    # Sort by difference from target
    entries_sorted = sorted(entries, key=lambda x: x[2])  # x[2] is diff
    # Take first N
    top_n = entries_sorted[:max_images]
    selected.extend(top_n)

print("Selected images closest to 2.0m tide (max 3 per month):")
for (img_id, tide_val, diff) in selected:
    print(f" - {img_id}: tide={tide_val}, difference={diff}")


# %% [markdown]
# ## Ordering
# 
# Create a function to place Planet orders given a json request and the function that creates the request given an array of IDs and an order name

# %%


async def do_order(order):
    async with Session() as sess:
        cl = OrdersClient(sess)
        order = await cl.create_order(order)
        return order

async def assemble_order(item_ids, name):
    products = [
        order_request.product(item_ids, 'analytic_udm2', 'PSScene')
    ]

    tools = [order_request.clip_tool(aoi=geom_all), order_request.composite_tool(group_by="strip_id")]

    request = order_request.build_request(
        name, products=products, tools=tools, order_type="partial")
    return request


# %% [markdown]
# Now we can order all the scenes at once, decide between the tide based selection or the cloud based selection

# %%
order_ids = []

# Iterate over each chunk and place the order
for date, ids in selected_images.items():
    order_name = f"masset_point_{date}"
    request = await assemble_order(ids, order_name)
    print(request)
    order = await do_order(request)
    print(order)
    order_ids.append(order['id'])

# Print all order IDs
print("Order IDs:", order_ids)

# %% [markdown]
# given the array of order IDs, download all of your orders to the specified folder

# %%
save_directory = './output/masset_point'

async def download_order(order):
    async with Session() as sess:
        cl = OrdersClient(sess)
        # if we get here that means the order completed. Yay! Download the files.
        await cl.download_order(order, directory=save_directory, progress_bar=True)

for id in order_ids:
    await download_order(id)

# %% [markdown]
# This goes through all of the order folders and moves the imagery to be in the base directory instead of in subfolders

# %%


# %%

order_url = 'https://api.planet.com/compute/ops/orders/v2/' 


# Setup the session
session = requests.Session()
# Authenticate
session.auth = (API_KEY, "")

r = requests.get(order_url, auth=session.auth)
response = r.json()
results = response['_links']['results']

# %%
import pathlib

def download_results(results, overwrite=False):
    results_urls = [r['location'] for r in results]
    results_names = [r['name'] for r in results]
    num_files = len(results_urls)
    print('{} items to download'.format(num_files))
    i = 1
    for url, name in zip(results_urls, results_names):
        path = pathlib.Path(os.path.join('./output', name))
        
        if overwrite or not path.exists():
            print('downloading {} / {}'.format(i, num_files))
            r = requests.get(url, allow_redirects=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            open(path, 'wb').write(r.content)
        else:
            print('{} already exists, skipping'.format(i))
        i += 1

# %%
download_results(results)


