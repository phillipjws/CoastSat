# ==========================================================#
# Shoreline extraction from satellite images               #
# ==========================================================#

# %% 1. Initial settings

# load modules
import os
import numpy as np
import pickle
import warnings

warnings.filterwarnings("ignore")
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec

try:
    # This should work in VS Code's interactive window
    matplotlib.use("Qt5Agg")
except ImportError:
    # Fallback to the default backend
    print("Qt5Agg not available, using default backend.")
plt.ion()
import pandas as pd
from scipy import interpolate
from scipy import stats
from datetime import datetime, timedelta
import pytz
from pyproj import CRS
from coastsat import (
    SDS_download,
    SDS_preprocess,
    SDS_shoreline,
    SDS_tools,
    SDS_transects,
    SDS_slope,
)

geojson_polygon = os.path.join(os.getcwd(), "pat_bay.geojson")
polygon = SDS_tools.polygon_from_geojson(geojson_polygon)
polygon = SDS_tools.smallest_rectangle(polygon)

# date range
dates = ["2014-01-01", "2025-01-01"]

# satellite missions
sat_list = ["L7", "L8", "L9", "S2"]
# name of the site
sitename = "PATRICIA_BAY"

# filepath where data will be stored
filepath_data = os.path.join(os.getcwd(), "data")

# put all the inputs into a dictionnary
inputs = {
    "polygon": polygon,
    "dates": dates,
    "sat_list": sat_list,
    "sitename": sitename,
    "filepath": filepath_data,
}

# before downloading the images, check how many images are available for your inputs
SDS_download.check_images_available(inputs)

# %% 2. Retrieve images

# retrieve satellite images from GEE
metadata = SDS_download.retrieve_images(inputs)

# if you have already downloaded the images, just load the metadata file
metadata = SDS_download.get_metadata(inputs)

# %% 3. Batch shoreline detection

# settings for the shoreline extraction
settings = {
    # general parameters:
    "cloud_thresh": 0.05,  # threshold on maximum cloud cover
    "dist_clouds": 300,  # ditance around clouds where shoreline can't be mapped
    "output_epsg": 4326,  # epsg code of spatial reference system desired for the output
    # quality control:
    "check_detection": False,  # if True, shows each shoreline detection to the user for validation
    "adjust_detection": False,  # if True, allows user to adjust the postion of each shoreline by changing the threhold
    "save_figure": True,  # if True, saves a figure showing the mapped shoreline for each image
    # [ONLY FOR ADVANCED USERS] shoreline detection parameters:
    "min_beach_area": 1000,  # minimum area (in metres^2) for an object to be labelled as a beach
    "min_length_sl": 500,  # minimum length (in metres) of shoreline perimeter to be valid
    "cloud_mask_issue": False,  # switch this parameter to True if sand pixels are masked (in black) on many images
    "sand_color": "default",  # 'default', 'latest', 'dark' (for grey/black sand beaches) or 'bright' (for white sand beaches)
    "pan_off": False,  # True to switch pansharpening off for Landsat 7/8/9 imagery
    "s2cloudless_prob": 40,  # threshold to identify cloud pixels in the s2cloudless probability mask
    # add the inputs defined previously
    "inputs": inputs,
}

# [OPTIONAL] preprocess images (cloud masking, pansharpening/down-sampling)
SDS_preprocess.save_jpg(metadata, settings, use_matplotlib=True)
# create MP4 timelapse animation
fn_animation = os.path.join(inputs['filepath'],inputs['sitename'], '%s_animation_RGB.mp4'%inputs['sitename'])
fp_images = os.path.join(inputs['filepath'], inputs['sitename'], 'jpg_files', 'preprocessed')
fps = 4 # frames per second in animation
SDS_tools.make_animation_mp4(fp_images, fps, fn_animation)

# [OPTIONAL] create a reference shoreline (helps to identify outliers and false detections)
settings["reference_shoreline"] = SDS_preprocess.get_reference_sl(metadata, settings)
# set the max distance (in meters) allowed from the reference shoreline for a detected shoreline to be valid
settings["max_dist_ref"] = 100

# extract shorelines from all images (also saves output.pkl and shorelines.kml)
output = SDS_shoreline.extract_shorelines(metadata, settings)

# remove duplicates (images taken on the same date by the same satellite)
output = SDS_tools.remove_duplicates(output)
# remove inaccurate georeferencing (set threshold to 10 m)
output = SDS_tools.remove_inaccurate_georef(output, 10)

# for GIS applications, save output into a GEOJSON layer
geomtype = "lines"  # choose 'points' or 'lines' for the layer geometry
gdf = SDS_tools.output_to_gdf(output, geomtype)
if gdf is None:
    raise Exception("output does not contain any mapped shorelines")
gdf.crs = CRS(settings["output_epsg"])  # set layer projection
# save GEOJSON layer to file
gdf.to_file(
    os.path.join(
        inputs["filepath"],
        inputs["sitename"],
        "%s_output_%s.geojson" % (sitename, geomtype),
    ),
    driver="GeoJSON",
    encoding="utf-8",
)

# create MP4 timelapse animation
fn_animation = os.path.join(inputs['filepath'],inputs['sitename'], '%s_animation_shorelines.mp4'%inputs['sitename'])
fp_images = os.path.join(inputs['filepath'], inputs['sitename'], 'jpg_files', 'detection')
fps = 4 # frames per second in animation
SDS_tools.make_animation_mp4(fp_images, fps, fn_animation)

# plot the mapped shorelines
plt.ion()
fig = plt.figure(figsize=[15,8], tight_layout=True)
plt.axis('equal')
plt.xlabel('Eastings')
plt.ylabel('Northings')
plt.grid(linestyle=':', color='0.5')
for i in range(len(output['shorelines'])):
    sl = output['shorelines'][i]
    date = output['dates'][i]
    plt.plot(sl[:,0], sl[:,1], '.', label=date.strftime('%d-%m-%Y'))
# plt.legend()
fig.savefig(os.path.join(inputs['filepath'], inputs['sitename'], 'mapped_shorelines.jpg'),dpi=200)

# %% 4. Shoreline analysis

# if you have already mapped the shorelines, load the output.pkl file
filepath = os.path.join(inputs["filepath"], sitename)
with open(os.path.join(filepath, sitename + "_output" + ".pkl"), "rb") as f:
    output = pickle.load(f)
# remove duplicates (images taken on the same date by the same satellite)
output = SDS_tools.remove_duplicates(output)
# remove inaccurate georeferencing (set threshold to 10 m)
output = SDS_tools.remove_inaccurate_georef(output, 10)

# now we have to define cross-shore transects over which to quantify the shoreline changes
# each transect is defined by two points, its origin and a second point that defines its orientation

# option 1: draw origin of transect first and then a second point to define the orientation
transects = SDS_transects.draw_transects(output, settings)

# Option 2: Retrieve transects from geojson file
geojson_transects = os.path.join(
    os.getcwd(), "data", sitename, f"{sitename}_transects.geojson"
)
transects = SDS_tools.transects_from_geojson(geojson_transects)

# plot the transects to make sure they are correct (origin landwards!)
fig = plt.figure(figsize=[15, 8], tight_layout=True)
plt.axis("equal")
plt.xlabel("Eastings")
plt.ylabel("Northings")
plt.grid(linestyle=":", color="0.5")
for i in range(len(output["shorelines"])):
    sl = output["shorelines"][i]
    date = output["dates"][i]
    plt.plot(sl[:, 0], sl[:, 1], ".", label=date.strftime("%d-%m-%Y"))
for i, key in enumerate(list(transects.keys())):
    plt.plot(transects[key][0, 0], transects[key][0, 1], "bo", ms=5)
    plt.plot(transects[key][:, 0], transects[key][:, 1], "k-", lw=1)
    plt.text(
        transects[key][0, 0] - 100,
        transects[key][0, 1] + 100,
        key,
        va="center",
        ha="right",
        bbox=dict(boxstyle="square", ec="k", fc="w"),
    )
fig.savefig(
    os.path.join(inputs["filepath"], inputs["sitename"], "mapped_shorelines.jpg"),
    dpi=200,
)

# %% Option 1: Compute intersections with quality-control parameters (recommended)

settings_transects = {  # parameters for computing intersections
    "along_dist": 25,  # along-shore distance to use for computing the intersection
    "min_points": 3,  # minimum number of shoreline points to calculate an intersection
    "max_std": 15,  # max std for points around transect
    "max_range": 30,  # max range for points around transect
    "min_chainage": -100,  # largest negative value along transect (landwards of transect origin)
    "multiple_inter": "auto",  # mode for removing outliers ('auto', 'nan', 'max')
    "auto_prc": 0.1,  # percentage to use in 'auto' mode to switch from 'nan' to 'max'
}
cross_distance = SDS_transects.compute_intersection_QC(
    output, transects, settings_transects
)

# %% Plot the time-series of cross-shore shoreline change

fig = plt.figure(figsize=[15, 8], tight_layout=True)
gs = gridspec.GridSpec(len(cross_distance), 1)
gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.05)
for i, key in enumerate(cross_distance.keys()):
    if np.all(np.isnan(cross_distance[key])):
        continue
    ax = fig.add_subplot(gs[i, 0])
    ax.grid(linestyle=":", color="0.5")
    ax.set_ylim([-50, 50])
    ax.plot(
        output["dates"],
        cross_distance[key] - np.nanmedian(cross_distance[key]),
        "-o",
        ms=4,
        mfc="w",
    )
    ax.set_ylabel("distance [m]", fontsize=12)
    ax.text(
        0.5,
        0.95,
        key,
        bbox=dict(boxstyle="square", ec="k", fc="w"),
        ha="center",
        va="top",
        transform=ax.transAxes,
        fontsize=14,
    )
fig.savefig(
    os.path.join(inputs["filepath"], inputs["sitename"], "time_series_raw.jpg"), dpi=200
)

# save time-series in a .csv file
out_dict = dict([])
out_dict["dates"] = output["dates"]
for key in transects.keys():
    out_dict["Transect " + key] = cross_distance[key]
df = pd.DataFrame(out_dict)
fn = os.path.join(
    settings["inputs"]["filepath"],
    settings["inputs"]["sitename"],
    "transect_time_series.csv",
)
df.to_csv(fn, sep=",")
print("Time-series of the shoreline change along the transects saved as:\n%s" % fn)

# %% 4. Tidal correction

# Option 1: Use fes2022
import pyfes

filepath = os.path.join(os.pardir)
config = os.path.join(filepath, "fes2022.yaml")
handlers = pyfes.load_config(config)
ocean_tide = handlers["tide"]
load_tide = handlers["radial"]

centroid = np.mean(polygon[0], axis=0)
print(centroid)
centroid[0] = centroid[0] + 360 if centroid[0] < 0 else centroid[0]
print(centroid)

date_range = [
    pytz.utc.localize(datetime(2014, 1, 1)),
    pytz.utc.localize(datetime(2025, 1, 1)),
]
timestep = 900  # seconds
dates_ts, tides_ts = SDS_slope.compute_tide(
    centroid, date_range, timestep, ocean_tide, load_tide
)

dates_sat = output["dates"]
tides_sat = SDS_slope.compute_tide_dates(
    centroid, output["dates"], ocean_tide, load_tide
)

fig, ax = plt.subplots(1, 1, figsize=(15, 4), tight_layout=True)
ax.grid(which="major", linestyle=":", color="0.5")
ax.plot(dates_ts, tides_ts, "-", color="0.6", label="all time-series")
ax.plot(
    dates_sat,
    tides_sat,
    "-o",
    color="k",
    ms=6,
    mfc="w",
    lw=1,
    label="image acquisition",
)
ax.set(
    ylabel="tide level [m]",
    xlim=[dates_sat[0], dates_sat[-1]],
    title="Tide levels at the time of image acquisition",
)
ax.legend()
fig.savefig(os.path.join(filepath, "%s_tide_timeseries.jpg" % sitename), dpi=200)


# Option 2: Use local data

# load the measured tide data
# filepath = os.path.join(os.getcwd(), 'patricia_bay_tides.csv')
# tide_data = pd.read_csv(filepath, parse_dates=['dates'])
# tide_data['dates'] = tide_data['dates'].dt.tz_localize('UTC')
# dates_ts = tide_data['dates'].tolist()
# tides_ts = np.array(tide_data['tide'])

# # get tide levels corresponding to the time of image acquisition
# dates_sat = output['dates']
# tides_sat = SDS_tools.get_closest_datapoint(dates_sat, dates_ts, tides_ts)

# # plot the subsampled tide data
# fig, ax = plt.subplots(1,1,figsize=(15,4), tight_layout=True)
# ax.grid(which='major', linestyle=':', color='0.5')
# ax.plot(dates_ts, tides_ts, '-', color='0.6', label='all time-series')
# ax.plot(dates_sat, tides_sat, '-o', color='k', ms=6, mfc='w',lw=1, label='image acquisition')
# ax.set(ylabel='tide level [m]',xlim=[dates_sat[0],dates_sat[-1]], title='Tide levels at the time of image acquisition');
# ax.legend();

# tidal correction along each transect
reference_elevation = 0
beach_slope = 0.1
cross_distance_tidally_corrected = {}
for key in cross_distance.keys():
    correction = (tides_sat - reference_elevation) / beach_slope
    cross_distance_tidally_corrected[key] = cross_distance[key] + correction

# store the tidally-corrected time-series in a .csv file
out_dict = dict([])
out_dict["dates"] = dates_sat
for key in cross_distance_tidally_corrected.keys():
    out_dict[key] = cross_distance_tidally_corrected[key]
df = pd.DataFrame(out_dict)
fn = os.path.join(
    settings["inputs"]["filepath"],
    settings["inputs"]["sitename"],
    "transect_time_series_tidally_corrected.csv",
)
df.to_csv(fn, sep=",")
print(
    "Tidally-corrected time-series of the shoreline change along the transects saved as:\n%s"
    % fn
)

# plot the time-series of shoreline change (both raw and tidally-corrected)
fig = plt.figure(figsize=[15, 8], tight_layout=True)
gs = gridspec.GridSpec(len(cross_distance), 1)
gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.05)
for i, key in enumerate(cross_distance.keys()):
    if np.all(np.isnan(cross_distance[key])):
        continue
    ax = fig.add_subplot(gs[i, 0])
    ax.grid(linestyle=":", color="0.5")
    ax.set_ylim([-50, 50])
    ax.plot(
        output["dates"],
        cross_distance[key] - np.nanmedian(cross_distance[key]),
        "-o",
        ms=6,
        mfc="w",
        label="raw",
    )
    ax.plot(
        output["dates"],
        cross_distance_tidally_corrected[key] - np.nanmedian(cross_distance[key]),
        "-o",
        ms=6,
        mfc="w",
        label="tidally-corrected",
    )
    ax.set_ylabel("distance [m]", fontsize=12)
    ax.text(
        0.5,
        0.95,
        key,
        bbox=dict(boxstyle="square", ec="k", fc="w"),
        ha="center",
        va="top",
        transform=ax.transAxes,
        fontsize=14,
    )
ax.legend()

# %% 5. Time-series post-processing

filename_output = os.path.join(os.getcwd(), "data", sitename, f"{sitename}_output.pkl")
with open(filename_output, "rb") as f:
    output = pickle.load(f)

# plot the mapped shorelines
fig = plt.figure(figsize=[15, 8], tight_layout=True)
plt.axis("equal")
plt.xlabel("Eastings")
plt.ylabel("Northings")
plt.grid(linestyle=":", color="0.5")
plt.title(f"%d shorelines mapped at {sitename} from 2014" % len(output["shorelines"]))
for i in range(len(output["shorelines"])):
    sl = output["shorelines"][i]
    date = output["dates"][i]
    plt.plot(sl[:, 0], sl[:, 1], ".", label=date.strftime("%d-%m-%Y"))
for i, key in enumerate(list(transects.keys())):
    plt.plot(transects[key][0, 0], transects[key][0, 1], "bo", ms=5)
    plt.plot(transects[key][:, 0], transects[key][:, 1], "k-", lw=1)
    plt.text(
        transects[key][0, 0] - 100,
        transects[key][0, 1] + 100,
        key,
        va="center",
        ha="right",
        bbox=dict(boxstyle="square", ec="k", fc="w"),
    )

filepath = os.path.join(
    os.getcwd(), "data", "PATRICIA_BAY", "transect_time_series_tidally_corrected.csv"
)
df = pd.read_csv(filepath, parse_dates=["dates"])
dates = [_.to_pydatetime() for _ in df["dates"]]
cross_distance = dict([])
for key in transects.keys():
    cross_distance[key] = np.array(df[key])

# %% 5.1 Remove outliers

# plot Otsu thresholds for the mapped shorelines
fig, ax = plt.subplots(1, 1, figsize=[12, 5], tight_layout=True)
ax.grid(which="major", ls=":", lw=0.5, c="0.5")
ax.plot(output["dates"], output["MNDWI_threshold"], "o-", mfc="w")
ax.axhline(y=-0.5, ls="--", c="r", label="otsu_threshold limits")
ax.axhline(y=0, ls="--", c="r")
ax.set(
    title="Otsu thresholds on MNDWI for the %d shorelines mapped"
    % len(output["shorelines"]),
    ylim=[-0.6, 0.2],
    ylabel="otsu threshold",
)
ax.legend(loc="upper left")
fig.savefig(os.path.join(inputs["filepath"], inputs["sitename"], "otsu_threhsolds.jpg"))

# remove outliers in the time-series (despiking)
settings_outliers = {
    "otsu_threshold": [
        -0.5,
        0,
    ],  # min and max intensity threshold use for contouring the shoreline
    "max_cross_change": 40,  # maximum cross-shore change observable between consecutive timesteps
    "plot_fig": True,  # whether to plot the intermediate steps
}
cross_distance = SDS_transects.reject_outliers(
    cross_distance, output, settings_outliers
)

# %% 5.2 Seasonal averaging

# compute seasonal averages along each transect
season_colors = {"DJF": "C3", "MAM": "C1", "JJA": "C2", "SON": "C0"}
for key in cross_distance.keys():
    chainage = cross_distance[key]
    # remove nans
    idx_nan = np.isnan(chainage)
    dates_nonan = [dates[_] for _ in np.where(~idx_nan)[0]]
    chainage = chainage[~idx_nan]

    # compute shoreline seasonal averages (DJF, MAM, JJA, SON)
    dict_seas, dates_seas, chainage_seas, list_seas = SDS_transects.seasonal_average(
        dates_nonan, chainage
    )

    # plot seasonal averages
    fig, ax = plt.subplots(1, 1, figsize=[14, 4], tight_layout=True)
    ax.grid(which="major", linestyle=":", color="0.5")
    ax.set_title("Time-series at %s" % key, x=0, ha="left")
    ax.set(ylabel="distance [m]")
    ax.plot(
        dates_nonan,
        chainage,
        "+",
        lw=1,
        color="k",
        mfc="w",
        ms=4,
        alpha=0.5,
        label="raw datapoints",
    )
    ax.plot(
        dates_seas,
        chainage_seas,
        "-",
        lw=1,
        color="k",
        mfc="w",
        ms=4,
        label="seasonally-averaged",
    )
    for k, seas in enumerate(dict_seas.keys()):
        ax.plot(
            dict_seas[seas]["dates"],
            dict_seas[seas]["chainages"],
            "o",
            mec="k",
            color=season_colors[seas],
            label=seas,
            ms=5,
        )
    ax.legend(
        loc="lower left",
        ncol=6,
        markerscale=1.5,
        frameon=True,
        edgecolor="k",
        columnspacing=1,
    )

# %% 5.3 Monthly averaging

# compute monthly averages along each transect
month_colors = plt.get_cmap("tab20")
for key in cross_distance.keys():
    chainage = cross_distance[key]
    # remove nans
    idx_nan = np.isnan(chainage)
    dates_nonan = [dates[_] for _ in np.where(~idx_nan)[0]]
    chainage = chainage[~idx_nan]

    # compute shoreline seasonal averages (DJF, MAM, JJA, SON)
    dict_month, dates_month, chainage_month, list_month = SDS_transects.monthly_average(
        dates_nonan, chainage
    )

    # plot seasonal averages
    fig, ax = plt.subplots(1, 1, figsize=[14, 4], tight_layout=True)
    ax.grid(which="major", linestyle=":", color="0.5")
    ax.set_title("Time-series at %s" % key, x=0, ha="left")
    ax.set(ylabel="distance [m]")
    ax.plot(
        dates_nonan,
        chainage,
        "+",
        lw=1,
        color="k",
        mfc="w",
        ms=4,
        alpha=0.5,
        label="raw datapoints",
    )
    ax.plot(
        dates_month,
        chainage_month,
        "-",
        lw=1,
        color="k",
        mfc="w",
        ms=4,
        label="monthly-averaged",
    )
    for k, month in enumerate(dict_month.keys()):
        ax.plot(
            dict_month[month]["dates"],
            dict_month[month]["chainages"],
            "o",
            mec="k",
            color=month_colors(k),
            label=month,
            ms=5,
        )
    ax.legend(
        loc="lower left",
        ncol=7,
        markerscale=1.5,
        frameon=True,
        edgecolor="k",
        columnspacing=1,
    )
