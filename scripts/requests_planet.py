# %% [markdown]
# 
# # Planet Data API Introduction using Python
# 
# ---

# %% [markdown]
# ## Introduction
# 
# ---
# 
# This tutorial is an introduction to [Planet](https://www.planet.com)'s Data API. It provides code samples on how to write simple Python code to access data.
# 
# [`https://api.planet.com/data/v1`](https://api.planet.com/data/v1)
# 
# The focus of this tutorial will be the search portion of the Planet Data API. We will find and download imagery data using complex searches and save them for later use, as well as learn how to get stats on search results. After completing this tutorial, you should feel comfortable interacting with the Data API, and have a good foundation for leveraging the Planet Data API for your own applications.
# 

# %% [markdown]
# ## Table of Contents
# 
# ---
# 
# * **Introduction**
#     * Requirements
#        * Software & Modules
#        * Planet API Key  
#     * Useful Links
#     
# 
# * **Overview**
#      * The Basic Workflow
#       * Data API Endpoints
#       * Planet Explorer
#       
#       
# * **Basic Setup**
#     * Authentication
#     * Helper Modules and Functions
#     * Our First Request
#     
#     
# * **Search Mechanics & Filters**
#     * Statistics
#     * Using a Search Filter
#     * Filter types
#         * Field Filters
#         * Logical Filters
#         * (Field) Filter Type examples
#     * Complex Queries using Logical Filters
#     
#     
# * **Searching for Items**
#     * Quick Search
#         * Mapping Footprints Example
#         
#         
# * **Assets & Permissions**
#     * Activating Assets
#     * Downloading Assets
#     * Rate Limits
#     
#     
# * **Saved Searches**
# 
# 
# * **Conclusion**  

# %% [markdown]
# ### Requirements
# 
# ---
# 
# #### Software & Modules
# 
# This tutorial assumes familiarity with the [Python](https://python.org) programming language throughout. Familiarity with basic REST API concepts and usage is also assumed.
# 
# We'll be using a **"Jupyter Notebook"** (aka Python Notebook) to run through the examples.
# To learn more about and get started with using Jupyter, visit: [Jupyter](https://jupyter.org/) and [IPython](https://ipython.org/). 
# 
# For the best experience, download this notebook and run it on your system, and make sure to install the modules listed below first. You can also copy the examples' code to a separate Python files an run them directly with Python on your system if you prefer.
# 
# 
# Python modules used in this tutorial are:
# * [requests](http://docs.python-requests.org/)
# * [geojsonio](https://pypi.python.org/pypi/geojsonio)
# 
# 
# #### Planet API Key
# 
# You should have an account on the Planet Platform to access the Data API. You may retrieve your API key from your [account page](https://www.planet.com/account/), or from the "API Tab" in [Planet Explorer](https://www.planet.com/explorer).

# %% [markdown]
# ### Useful links 
# 
# ---
# 
# * [Planet Data API Reference & Documentation](https://developers.planet.com/docs/apis/data/)
# * [Planet Explorer](https://www.planet.com/explorer)

# %% [markdown]
# ## Overview
# 
# ---
# 
# ### The Basic Workflow:
# 
# 1. Search item types based on filters
# 1. Activate assets
# 1. Download assets

# %% [markdown]
# ### API Endpoints
# 
# This tutorial will cover the following API ***endpoints***:
# 
# * [`/item-types`](https://api.planet.com/data/v1/item-types/)
# * [`/asset-types`](https://api.planet.com/data/v1/asset-types/)
# * `/quick-search`
# * `/searches`
# * `/stats`
# 
# There is an additional [`/spec`](https://api.planet.com/data/v1/spec) endpoint that publishes a [Swagger](http://swagger.io/) specification for the Data API.
# 
# ### Exploring Endpoints
# 
# We can start by exploring two endpoints using our browser.
# 
# ##### What item types are available?
# 
# https://api.planet.com/data/v1/item-types/
# 
# ##### What assets are available?
# 
# https://api.planet.com/data/v1/asset-types/

# %% [markdown]
# 
# ***Note:*** *You may choose to install a helpful browser plugin called **JSONView** that formats JSON and makes it easier to read:*
# 
# * [Chrome Extension](https://chrome.google.com/webstore/detail/jsonview/chklaanhfefbnpoihckbnefhakgolnmc)
# * [Firefox Plugin](https://addons.mozilla.org/en-us/firefox/addon/jsonview/)

# %% [markdown]
# ### Planet Explorer
# We can also use Planet Explorer to expose relevant API information for search results. 
# 
# In Planet Explorer, while in *Daily Imagery Mode* click on the *"Api"* button in the bottom left corner of the results panel.
# 
# ![Planet Explorer Screenshot](images/planet-explorer.png)
# 
# Here, you may access your API Key, selected item id's, and `cURL` request information containing filters and options.

# %% [markdown]
# ## Basic Setup
# 
# ---
# 
# Before interacting with the Planet Data API using Python, we will set up our environment with some useful modules and helper functions.
# 
# * We'll configure *authentication* to the Planet Data API
# * We'll use the `requests` Python module to make HTTP communication easier. 
# * We'll use the `json` Python module to help us work with JSON responses from the API.
# * We'll create a function called `p` that will print Python dictionaries nicely.
# 
# Then we'll be ready to make our first call to the Planet API by hitting the base endpoint at `https://api.planet.com/data/v1`. 
# 
# Let's start by importing needed packages configuring authentication:

# %% [markdown]
# ### Import Packages & Define Helper Functions

# %%
import os
import json
import requests
import geojsonio
import time
from dotenv import load_dotenv

# Helper function to printformatted JSON using the json module
def p(data):
    print(json.dumps(data, indent=2))

# %% [markdown]
# ### Authentication
# 
# Authentication with the Planet Data API be achieved using a valid Planet **API key**.

# %% [markdown]
# You can *export* your API Key as an environment variable on your system:
# 
# `export API_KEY="YOUR API KEY HERE"`
# 
# Or add the variable to your path, etc. If you are using our Docker environement, it will already be set.
# 
# To start our Python code, we'll setup an API Key variable from an environment variable to use with our requests:
load_dotenv()
# %%
# if your Planet API Key is not set as an environment variable, you can paste it below
API_KEY = os.getenv('PL_API_KEY')

# construct auth tuple for use in the requests library
BASIC_AUTH = (API_KEY, '')

# %% [markdown]
# ### Our First Request

# %%
# Setup Planet Data API base URL
URL = "https://api.planet.com/data/v1"

# Setup the session
session = requests.Session()

# Authenticate
session.auth = (API_KEY, "")

# %%
# Make a GET request to the Planet Data API
res = session.get(URL)

# %% [markdown]
# Now we should get a response, hopefully it's a `200` code, saying everything is OK!

# %%
# Response status code
res.status_code

# %%
# Response Body
res.text

# %%
# Print formatted JSON response
p(res.json())

# %%
# Print the value of the item-types key from _links
print(res.json()["_links"]["item-types"])

# %% [markdown]
# **Congratulations!** You just made your first request to the Planet Data API using Python! 
# 
# Now, let's take a look at how to access all that high cadence satellite imagery!

# %% [markdown]
# ## Search Mechanics & Filters
# 
# ---
# 
# There's a lot of data available on the Planet Data API and in order to find what we're looking for, we'll have to perform various types of searches.
# 
# To construct a search, we use **filters** that let us limit results based on date, geography or other metadata properties. 
# 
# Using these filters, we are able to **search** for items or get **statistics** for search results.
# 
# Let's first make a request to the `/stats` endpoint to understand how the filters work:

# %% [markdown]
# ### Statistics `/stats`
# 
# ---
# 
# The `/stats` endpoint provides a summary of the available data based on some search criteria.
# 
# `https://api.planet.com/data/v1/stats`
# 
# We'll need to send a ***`POST`*** request to `/stats`, let's start by setting up the request url:

# %%
# Setup the stats URL
stats_url = "{}/stats".format(URL)

# Print the stats URL
print(stats_url)

# %% [markdown]
# ### Using a Search Filter
# 
# Search filters should have the following properties:
# 
# * **Type** (`type`) - The type of filter being used
# * **Configuration** (`config`) - The configuration for this filter
# * **Field Name** (`field_name`) - The field on which to filter 
# 
# For this example, let's use a filter to get some stats on what data is available for *Planet Scope (3 Band)* and *Rapid Eye (Ortho Tile)* products taken from 2013 until now:

# %%
# Specify the sensors/satellites or "item types" to include in our results
item_types = ["PSScene", "REOrthoTile"]

# %%
# Create filter object for all imagery captured between 2013-01-01 and present.
date_filter = {
    "type": "DateRangeFilter", # Type of filter -> Date Range
    "field_name": "acquired", # The field to filter on: "acquired" -> Date on which the "image was taken"
    "config": {
        "gte": "2013-01-01T00:00:00.000Z", # "gte" -> Greater than or equal to
    }
}

# %% [markdown]
# Depending on the filter type, some requests may need an `interval` field:
# 
# The following intervals are possible:
# * `year`
# * `month`
# * `week`
# * `day`
# * `hour`
# 
# An `interval` must be provided with the request so that the number of matching items can be aggregated. We'll use an interval with our date filter. 

# %% [markdown]
# Now let's perform our request using the `date_filter` filter we created above:

# %%
# Construct the request.
request = {
    "item_types" : item_types,
    "interval" : "year",
    "filter" : date_filter
}

# Send the POST request to the API stats endpoint
res = session.post(stats_url, json=request)

# Print response
p(res.json())

# %% [markdown]
# ***Good Job!*** You've received a response from the API that contains some statistics like item counts for the search criteria!

# %% [markdown]
# <div class="alert alert-info">
# 
# **Exercise:** Create a new date filter to find data from before 2013.
# 
# </div>

# %%
# Fill in this filter to complete the exercise! 
date_filter2 = {
}

request = {
    "item_types" : item_types,
    "interval" : "year",
    "filter" : date_filter2
}

res = session.post(stats_url, json=request)
p(res.json())

# %% [markdown]
# ---
# 
# Next, let's take a closer look at some of the available filters:

# %% [markdown]
# ### Filter types
# 
# The Planet Data API supports several filter types. The most common are the following:
# 
# ### Field Filters
# 
# * `DateRangeFilter`
# * `RangeFilter`
# * `StringInFilter`
# * `PermissionFilter`
# * `GeometryFilter`
# 
# ### Logical Filters
# 
# * `NotFilter`
# * `AndFilter`
# * `OrFilter`
# 
# More information and examples on filter types can be found at the [API Docs](https://developers.planet.com/docs/apis/). 
# 
# ---
# 
# ### (Field) Filter Type examples:

# %% [markdown]
# #### `DateRangeFilter`
# 
# Find imagery that was `acquired` or `published` before, after or between certain dates.
# 
# ```
# {
#   "type": "DateRangeFilter",
#   "field_name": "acquired",
#   "config": {
#     "gt": "2016-01-01T00:00:00Z",
#     "lte": "2016-03-01T00:00:00Z"
#   }
# }
# ```
# 
# The upper or lower bound may be omitted.

# %% [markdown]
# #### `RangeFilter`
# 
# Find imagery that has a metadata that matches a number within a range of numbers.
# 
# ```
# {
#   "type": "RangeFilter",
#   "field_name": "cloud_cover",
#   "config": {
#     "lt": 0.2,
#     "gt": 0.1
#   }
# }
# ```

# %% [markdown]
# The following **operators** are supported by the Data API's `DateRangeFilter` and `RangeFilter`:
# * `gt`: Greater Than
# * `gte`: Greater Than or Equal To
# * `lt`: Less Than
# * `lte`: Less Than or Equal To

# %% [markdown]
# #### `StringInFilter`
# 
# Find imagery that has a metadata that matches a string within the array of provided strings.
# 
# 
# ```
# {
#   "type": "StringInFilter",
#   "field_name": "instrument",
#   "config": ["PS2"]
# }
# ```

# %% [markdown]
# #### `PermissionFilter`
# 
# Find data which has assets that are accessible by the user.
# 
# ```
# {
#   "type": "PermissionFilter",
#   "config": ["assets.analytic:download"]
# }
# ```
# 
# ***Note:*** `assets:download` means *any* downloadable asset.

# %% [markdown]
# #### `GeometryFilter`
# 
# Find data contained within a given geometry. The filter's config value may be any valid GeoJSON object.
# 
# ```
# {
#   "type": "GeometryFilter",
#   "field_name": "geometry",
#   "config": {
#     "type": "Polygon",
#     "coordinates": [
#       [
#         [
#           -120.27282714843749,
#           38.348118547988065
#         ],
#         [
#           -120.27282714843749,
#           38.74337300148126
#         ],
#         [
#           -119.761962890625,
#           38.74337300148126
#         ],
#         [
#           -119.761962890625,
#           38.348118547988065
#         ],
#         [
#           -120.27282714843749,
#           38.348118547988065
#         ]
#       ]
#     ]
#   }
# }
# ```
# 
# A few quick ways to get a GeoJSON geometry to use in your search:
# 
# * Draw an Area of Interst (AOI) in [Planet Explorer](http://planet.com/explorer), and use the API button to see the geometry filter config.
# * Use your favorite GIS tools like [Quantum GIS (QGIS)](http://www.qgis.org) and export GeoJSON.
# * Draw a polygon in [GeoJSON.io](http://geojson.io).
# 
# Make sure the `config` property in the geometry filter is in the right format, which should be similar to a `feature` property in a GeoJSON object. 

# %% [markdown]
# ---
# 
# Let's try a few more requests and get some more stats, this time using different filters:

# %%
# Search for imagery only from PlanetScope satellites that have a PS2 telescope

# Setup item types
item_types = ["PSScene"]

# Setup a filter for instrument type
instrument_filter = {
    "type": "StringInFilter",
    "field_name": "instrument",
    "config": ["PS2"]
}

# %%
# Setup the request
request = {
    "item_types" : item_types,
    "interval" : "year",
    "filter" : instrument_filter
}

# Send the POST request to the API stats endpoint
res = session.post(stats_url, json=request)

# Print response
p(res.json())

# %% [markdown]
# <div class="alert alert-info">
# 
# **Exercise:** Create a new filter that finds all data from PS0 or PS1 telescopes.
# 
# </div>

# %%
# Fill in this filter to complete the exercise! 

instrument_filter2 = {
}
request = {
    "item_types" : item_types,
    "interval" : "year",
    "filter" : instrument_filter2
}

# Send the POST request to the API stats endpoint
res=session.post(stats_url, json=request)

# Print response
p(res.json())

# %% [markdown]
# Find a GeoJSON geometry from http://geojson.io and copy one `feature`.

# %%
# Search for imagery that only intersects with 40N, 90W

# Setup GeoJSON 
geom = {
    "type": "Point",
    "coordinates": [
        -90,
         40
    ]
}

# Setup Geometry Filter
geometry_filter = {
    "type": "GeometryFilter",
    "field_name": "geometry",
    "config": geom
}

# %%
# Setup the request
request = {
    "item_types" : item_types,
    "interval" : "year",
    "filter" : geometry_filter
}

# Send the POST request to the API stats endpoint
res=session.post(stats_url, json=request)

# Print response
p(res.json())

# %% [markdown]
# ### Complex Queries using Logical Filters
# 
# Complex queries may require combining field filters using **logical filters** (see Filter Types above).
# 
# Let's try a complex stats search:

# %%
# PS2 imagery; over 40N, 90W; captured between 2013 and present

# Setup an "AND" logical filter
and_filter = {
    "type": "AndFilter",
    "config": [instrument_filter, geometry_filter, date_filter]
}

# Print the logical filter
p(and_filter)

# %%
# Setup the request
request = {
    "item_types" : item_types,
    "interval" : "year",
    "filter" : and_filter
}

# Send the POST request to the API stats endpoint
res=session.post(stats_url, json=request)

# Print response
p(res.json())

# %% [markdown]
# Here's another complex search query example using a "Not" logical filter and "string in" field filter. Can you tell what it's requesting?

# %%
# Setup Item Types
item_types = ["PSScene"]

# Setup Instrument Filter
instrument_filter = {
    "type": "StringInFilter",
    "field_name": "instrument",
    "config": ["PS2"]
}

# Setup "not" Logical Filter
not_filter = {
    "type": "NotFilter",
    "config": instrument_filter
}

# Setup the request
request = {
    "item_types" : item_types,
    "interval" : "year",
    "filter" : not_filter
}

# Send the POST request to the API stats endpoint
res=session.post(stats_url, json=request)

# Print response
p(res.json())

# %% [markdown]
# Now that you're comfortable working with **filters**, let's take a look at how to perform searches!

# %% [markdown]
# ## Searching for Items
# 
# ---
# 
# There are two types of searches: 
# * **"Quick Search"** `/quick-search`
# * **"Saved Searches"** `/searches`
# 
# Saved searches are retained on the Planet Platform and may be performed again at any time in the future. You can use these to setup efficient workflows for repetitive tasks, for example, querying an area that is of interest to you, or getting data for specific sensors.
# 
# Quick searches are meant to be more fleeting, and are not guaranteed to be available on the API after they are executed.
# 
# Searches use the same request format as the `/stats` endpoint except without the `interval` field.

# %% [markdown]
# ### Quick Search
# 
# Let's dive right in and create our first `quick search`:

# %%
# Setup the quick search endpoint url
quick_url = "{}/quick-search".format(URL)

# %%
# Setup Item Types
item_types = ["PSScene"]

# Setup GeoJSON for only imagery that intersects with 40N, 90W
geom = {
    "type": "Point",
    "coordinates": [
        -90,
         40
    ]
}

# Setup a geometry filter
geometry_filter = {
    "type": "GeometryFilter",
    "field_name": "geometry",
    "config": geom
}

# Setup the request
request = {
    "item_types" : item_types,
    "filter" : geometry_filter
}

# %%
# Send the POST request to the API quick search endpoint
res = session.post(quick_url, json=request)

# Assign the response to a variable
geojson = res.json()

# Print the response
p(geojson)

# %% [markdown]
# Nice! The response gives us search results for Planet Scope (4 Band) for a specific area. 
# 
# Let's do another search, this time using the "not" filter we setup earlier:

# %%
# Setup Item Types
item_types = ["PSScene"]

# Setup the request
request = {
    "item_types" : item_types,
    "filter" : not_filter
}

# Send the POST request to the API quick search endpoint
res = session.post(quick_url, json=request)

# Assign the response to a variable
geojson = res.json()

# Print the response
p(list(geojson.items())[1][1][0])

# %% [markdown]
# Let's take a closer look at our response data:

# %%
# Assign a features variable 
features = geojson["features"]

# Get the number of features present in the response
len(features)

# %%
# Loop over all the features from the response
for f in features:
    # Print the ID for each feature
    p(f["id"])

# %%
# Print the first feature
p(features[0])

# %% [markdown]
# What happens when there are A LOT of results? 
# 
# When the number of matching items exceeds 250, the results are delivered in pages. Let's perform a search query that should return a large number of results:

# %%
# Setup a GeoJSON polygon for our geometry filter
geom = {
    "type": "Polygon",
    "coordinates": [
      [
        [
          -125.29632568359376,
          48.37084770238366
        ],
        [
          -125.29632568359376,
          49.335861591104106
        ],
        [
          -123.2391357421875,
          49.335861591104106
        ],
        [
          -123.2391357421875,
          48.37084770238366
        ],
        [
          -125.29632568359376,
          48.37084770238366
        ]
      ]
    ]
  }

# Setup the geometry filter
geometry_filter = {
    "type": "GeometryFilter",
    "field_name": "geometry",
    "config": geom
}

# Setup the request
request = {
    "item_types" : item_types,
    "filter" : geometry_filter
}

# %%
# Send the POST request to the API quick search endpoint
res = session.post(quick_url, json=request)

# Assign the response to a variable
geojson = res.json()

# %%
# Print the response "_links" property
p(geojson["_links"])

# %%
# Assign the "_links" -> "_next" property (link to next page of results) to a variable 
next_url = geojson["_links"]["_next"]

# Print the link to the next page of results
print(next_url)

# %% [markdown]
# The page size can be set with a `_page_size` parameter in the request:

# %%
# Send the POST request to the API quick search endpoint with a page size of 9
res = session.post(quick_url, json=request, params={"_page_size" : 9})

# Assign the response to a variable
geojson = res.json()

# Get the number of features present in the response
len(geojson["features"])

# %% [markdown]
# ### Mapping Footprints Example
# 
# The [`geojsonio`](https://pypi.python.org/pypi/geojsonio/) module can be used to map GeoJSON files and is therefore compatible with the output from Planet Data API searches.
# 
# Make sure you've got the module installed and then let's use it to show our resuls on a map:

# %%
# Assign the url variable to display the geojsonio map
url = geojsonio.display(res.text)

# %% [markdown]
# We can now iterate through the pages of results.

# %%
# Assign the next_url variable to the next page of results from the response (Setup the next page of results)
next_url = geojson["_links"]["_next"]

# Get the next page of results
res = session.get(next_url)

# Assign the response to a variable
geojson = res.json()

# Get the url see results on geojson.io
url = geojsonio.to_geojsonio(res.text)

# %%
# Setup the next page of results
next_url = geojson["_links"]["_next"]

# Get the next page of results
res = session.get(next_url)

# Assign the response to a variable
geojson = res.json()

# Get the url see results on geojson.io
url = geojsonio.to_geojsonio(res.text)

# %% [markdown]
# Cool huh? Now that we know how to search and find items we are interested in, let's try activating and downloading some assets.

# %% [markdown]
# ## Assets & Permissions
# 
# Assets may be imagery files, image masks, metadata files or some other file type that might be associated with the item.
# 
# The `_permissions` element in each feature contains a list of assets that the user has access to.

# %% [markdown]
# Let's pick the first item out of our search result to work with, and then see what the permissions are for that item:

# %%
# Assign a variable to the search result features (items)
features = geojson["features"]

# Get the first result's feature
feature = features[0]

# Print the ID
p(feature["id"])

# Print the permissions
p(feature["_permissions"])

# %% [markdown]
# Each item has an `assets` endpoint in the API that lists all of its assets. Let's get a list of available assets for our item:

# %%
# Get the assets link for the item
assets_url = feature["_links"]["assets"]

# Print the assets link
print(assets_url)

# %%
# Send a GET request to the assets url for the item (Get the list of available assets for the item)
res = session.get(assets_url)

# Assign a variable to the response
assets = res.json()

# %%
# Print the asset types that are available
print(assets.keys())

# %% [markdown]
# We can see that most items have several assets available. Let's check out the "basic_analytic_4b" asset for this item. This is the 4-band analytic asset for the PSScene item.

# %%
# Assign a variable to the visual asset from the item's assets
basic_analytic4b = assets["basic_analytic_4b"]

# Print the visual asset data
p(basic_analytic4b)

# %% [markdown]
# ### Activating Assets
# 
# Before an asset is available for download, it must be **activated**. 
# 
# You can activate an asset by sending a *POST* or *GET* request to the asset's **"activation link"**. 
# 
# The activation usually takes a little bit of time, but hey, I'm sure by now you have some ideas on how to automate your workflow. Go robots!

# %%
# Setup the activation url for a particular asset (in this case the basic_analytic_4b asset)
activation_url = basic_analytic4b["_links"]["activate"]

# Send a request to the activation url to activate the item
res = session.get(activation_url)

# Print the response from the activation request
p(res.status_code)

# %% [markdown]
# #### Activation Response Codes
# 
# After hitting an activation url, you should get a response code back from the API:
# 
# * **`202`** - The request has been accepted and the activation will begin shortly. 
# * **`204`** - The asset is already active and no further action is needed. 
# * **`401`** - The user does not have permissions to download this file.

# %% [markdown]
# After requesting to actiate an asset, let's do another request to see if the assset's status has changed. This may take some time.

# %%
asset_activated = False

while asset_activated == False:
    # Send a request to the item's assets url
    res = session.get(assets_url)

    # Assign a variable to the item's assets url response
    assets = res.json()

    # Assign a variable to the basic_analytic_4b asset from the response
    visual = assets["basic_analytic_4b"]

    asset_status = basic_analytic4b["status"]
    print(asset_status)
    
    # If asset is already active, we are done
    if asset_status == 'active':
        asset_activated = True
        print("Asset is active and ready to download")

# Print the ps3b_analytic asset data    
p(basic_analytic4b)

# %% [markdown]
# #### Activation Status
# 
# An asset may have a `status` of `inactive`, `activating` or `active`. 
# 
# Once an asset is active, the response will contain a `location`. We'll use the `location` to download the asset!
# 
# Below, we are polling the API until the item is activated. This may take awhile.

# %% [markdown]
# ### Downloading Assets
# 
# The asset's `location` property points asset file for use to download. Are you ready to download the asset? Let's do it!

# %%
# Assign a variable to the visual asset's location endpoint
location_url = basic_analytic4b["location"]

# Print the location endpoint
print(location_url)

# %% [markdown]
# There's the location url! You could click on it to open it in a browser and download the file... but why not let your code do it? 
# 
# Let's write a small helper function to download asset files:

# %%
# Create a function to download asset files
# Parameters: 
# - url (the location url)
# - filename (the filename to save it as. defaults to whatever the file is called originally)

def pl_download(url, filename=None):
    
    # Send a GET request to the provided location url, using your API Key for authentication
    res = requests.get(url, stream=True, auth=(API_KEY, ""))
    # If no filename argument is given
    if not filename:
        # Construct a filename from the API response
        if "content-disposition" in res.headers:
            filename = res.headers["content-disposition"].split("filename=")[-1].strip("'\"")
        # Construct a filename from the location url
        else:
            filename = url.split("=")[1][:10]
    # Save the file
    with open('output/' + filename, "wb") as f:
        for chunk in res.iter_content(chunk_size=1024):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
                f.flush()

    return filename

# %%
# Download the file from an activated asset's location url
pl_download(location_url)

# %% [markdown]
# **High Five!** You should now be downloading your very own GeoTIFF! We can't wait to see what you'll do with it!

# %% [markdown]
# ### Rate Limits
# 
# ---
# 
# Now that you know how to download assets, you should probably keep some of the API **rate limits** in mind: 
# 
# * For most endpoints, the rate limit is 10 requests per second, per API key.
# * For activation endpoints, the rate limit is 5 requests per second, per API key.
# * For download endpoints, the rate limit is 15 requests per second, per API key.
# 
# If you're writing to code to automate accessing the API, you should account for `429` responses and handle retries after a *backoff period*. 

# %% [markdown]
# ## Saved Searches
# 
# ---
# 
# The `/searches` endpoint lets you created saved searches that can be reused.
# 
# To view your saved searches, visit the [`searches/?search_type=saved`](https://api.planet.com/data/v1/searches/?search_type=saved) endpoint.
# 
# Finally, let's create a saved search:

# %%
# Setup the saved searches endpoint url
searches_url = "{}/searches".format(URL)

# %%
# Setup the request, specifying a name for the saved search, along with the usual search item_types and filter.
request = {
    "name" : "Vancouver Island",
    "item_types" : item_types,
    "filter" : geometry_filter
}

# Send a POST request to the saved searches endpoint (Create the saved search)
res = session.post(searches_url, json=request)

# Print the response
p(res.json())

# %% [markdown]
# Now that we created a saved search, let's get a list of our saved searches:

# %%
# Send a GET request to the saved searches endpoint with the saved search type parameter (Get saved searches)
res = session.get(searches_url, params={"search_type" : "saved"})

# Assign a variable to the searches property in the saved searches response
searches = res.json()["searches"]
print(searches[2])#['name'])
print(len(searches))

# Loop through the searches
for search in searches:
    # Print the ID, Created Time, and Name for each saved search
    print("id: {} created: {} name: {}".format(search["id"], search["created"], search['name']))

# %% [markdown]
# Ok, now let's check out the results from a particular saved search:

# %%
# Setup the saved search url, using the first saved search in the list
saved_url = "{}/{}".format(searches_url, searches[0]["id"])

# Print the saved search url
p(saved_url)

# Setup the saved search's results url
results_url = "{}/results".format(saved_url)

# Print the saved search's results url
p(results_url)

# %%
# Send a GET request to the saved search url (Get the saved search data)
res = session.get(saved_url)

# Print the response
p(res.json())

# %%
# Send a GET request to the saved search results url (Get the saved search results)
results = session.get(results_url).json()

# Print the number of features in the saved search
print(len(results["features"]))

# Print the first feature in the saved search
print(results["features"][0])

# %% [markdown]
# A saved search can also be updated by senidng a **`PUT`** request with a changed search definition back to the API.
# 
# Did you know that saved searches can also send you a daily email to alert you to when new items are added to the search resutlts? Oh yeah!

# %%
# Assign a variable to the saved search response
search = res.json()

# Print the saved search
p(search)

# %%
# Change the saved search name to "South Vancouver Island"
search["name"] = "South Vancouver Island"

# Set the daily email enabled to true (Get email alerts when new items show up in this search)
search["__daily_email_enabled"] = True

# Send a PUT request to the saved search endpoint (Update the saved search)
res = session.put(saved_url, json=search)

# The response status code
res.status_code

# %% [markdown]
# ## Conclusion
# 
# You made it! You should now be able to use the Planet API to find items and assets by searching, get stats and save searches, and activate and download assets using Python code! 
# 
# Don't forget to visit the [Planet API Reference Docs](https://developers.planet.com/docs/apis/).
# 


