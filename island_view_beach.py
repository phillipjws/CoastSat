# ==========================================================#
# Shoreline extraction from satellite images                #
# ==========================================================#

import os
import numpy as np
import pickle
import warnings

warnings.filterwarnings('ignore')
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec

try:
    matplotlib.use('Qt5Agg')
except ImportError:
    print('Qt5Agg not available, using default backend.')
# plt.ion()
import geopandas as gpd
from shapely.geometry import MultiLineString
import pandas as pd
from scipy import interpolate
from scipy import stats
from datetime import datetime, timedelta
import pytz
import pyfes
from pyproj import CRS
from coastsat import (
    SDS_download,
    SDS_preprocess,
    SDS_shoreline,
    SDS_tools,
    SDS_transects,
    SDS_slope,
)

def initial_settings():
    """
    Initial settings for the shoreline extraction.
    Returns the inputs dictionary and settings dictionary.
    """
    # TODO: Ensure correct GeoJson
    # Load the polygon from a GeoJSON file
    geojson_polygon = os.path.join(os.pardir, 'planetscope_coastsat', 'user_inputs','island_view_beach.kml')
    polygon = SDS_tools.polygon_from_kml(geojson_polygon)
    polygon = SDS_tools.smallest_rectangle(polygon)

    # Date range
    dates = ['2020-07-30', '2024-10-23']

    # TODO: Set sat list and date range
    # Satellites
    sat_list = ['L5', 'L8', 'L9']
    # Name of the site
    sitename = 'ISLAND_VIEW_BEACH'

    # Filepath where data will be stored
    filepath_data = os.path.join(os.getcwd(), 'data')

    # Put all the inputs into a dictionary
    inputs = {
        'polygon': polygon,
        'dates': dates,
        'sat_list': sat_list,
        'sitename': sitename,
        'filepath': filepath_data,
    }

    # Settings for the shoreline extraction
    settings = {
        # General parameters:
        'cloud_thresh': 0.05,  # Threshold on maximum cloud cover
        'dist_clouds': 300,  # Distance around clouds where shoreline can't be mapped
        'output_epsg': 3157,  # EPSG code of spatial reference system desired for the output
        # Quality control:
        'check_detection': False,  # If True, shows each shoreline detection to the user for validation
        'adjust_detection': False,  # If True, allows user to adjust the position of each shoreline by changing the threshold
        'save_figure': True,  # If True, saves a figure showing the mapped shoreline for each image
        'skip_L7_slc': True,
        # [ONLY FOR ADVANCED USERS] Shoreline detection parameters:
        'min_beach_area': 1000,  # Minimum area (in metres^2) for an object to be labelled as a beach
        'min_length_sl': 100,  # Minimum length (in metres) of shoreline perimeter to be valid
        'cloud_mask_issue': False,  # Switch this parameter to True if sand pixels are masked (in black) on many images
        'sand_color': 'default',  # 'default', 'latest', 'dark' (for grey/black sand beaches) or 'bright' (for white sand beaches)
        'pan_off': False,  # True to switch pansharpening off for Landsat 7/8/9 imagery
        's2cloudless_prob': 40,  # Threshold to identify cloud pixels in the s2cloudless probability mask
        # Add the inputs defined previously
        'inputs': inputs,
    }

    return inputs, settings

def retrieve_images(inputs):
    """
    Retrieve images from GEE and load metadata.
    """
    # # Before downloading the images, check how many images are available for your inputs
    SDS_download.check_images_available(inputs)

    # # Retrieve satellite images from GEE
    metadata = SDS_download.retrieve_images(inputs)

    # If you have already downloaded the images, just load the metadata file
    metadata = SDS_download.get_metadata(inputs)

    return metadata

def batch_shoreline_detection(metadata, settings, inputs):
    """
    Perform batch shoreline detection.
    Returns the output dictionary.
    """
    # # Preprocess images (cloud masking, pansharpening/down-sampling)
    # SDS_preprocess.save_jpg(metadata, settings, use_matplotlib=True)
    # # create MP4 timelapse animation
    # fn_animation = os.path.join(inputs['filepath'], inputs['sitename'], '%s_animation_RGB.gif'%inputs['sitename'])
    # fp_images = os.path.join(inputs['filepath'], inputs['sitename'], 'jpg_files', 'preprocessed')
    # fps = 4 # frames per second in animation
    # SDS_tools.make_animation_mp4(fp_images, fps, fn_animation)

    # Create a reference shoreline (helps to identify outliers and false detections)
    # settings['reference_shoreline'] = SDS_preprocess.get_reference_sl(metadata, settings)
    settings['reference_shoreline'] = SDS_preprocess.get_reference_sl_from_geojson(settings['inputs']['sitename'], os.path.join(r'C:\Users\psteeves\coastal\planetscope_coastsat\user_inputs\reference_shorelines'), settings['output_epsg'])
    # print(settings['reference_shoreline'])
    # Set the max distance (in meters) allowed from the reference shoreline for a detected shoreline to be valid
    settings['max_dist_ref'] = 100

    # Extract shorelines from all images (also saves output.pkl and shorelines.kml)
    output = SDS_shoreline.extract_shorelines(metadata, settings)

    # Remove duplicates (images taken on the same date by the same satellite)
    output = SDS_tools.remove_duplicates(output)
    # Remove inaccurate georeferencing (set threshold to 10 m)
    output = SDS_tools.remove_inaccurate_georef(output, 10)

    # For GIS applications, save output into a GEOJSON layer
    geomtype = 'lines'  # Choose 'points' or 'lines' for the layer geometry
    gdf = SDS_tools.output_to_gdf(output, geomtype)
    if gdf is None:
        raise Exception('Output does not contain any mapped shorelines')
    gdf.crs = CRS(settings['output_epsg'])  # Set layer projection
    # Save GEOJSON layer to file
    gdf.to_file(
        os.path.join(
            settings['inputs']['filepath'],
            settings['inputs']['sitename'],
            '%s_output_%s.geojson' % (settings['inputs']['sitename'], geomtype),
        ),
        driver='GeoJSON',
        encoding='utf-8',
    )

    # # create MP4 timelapse animation
    # fn_animation = os.path.join(inputs['filepath'],inputs['sitename'], '%s_animation_shorelines.gif'%inputs['sitename'])
    # fp_images = os.path.join(inputs['filepath'], inputs['sitename'], 'jpg_files', 'detection')
    # fps = 4 # frames per second in animation
    # SDS_tools.make_animation_mp4(fp_images, fps, fn_animation)

    # Plot the mapped shorelines
    if settings.get('save_figure', False):
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
        fig.savefig(os.path.join(settings['inputs']['filepath'], settings['inputs']['sitename'], 'mapped_shorelines.jpg'), dpi=200)
        # Optionally, display the plot
        # plt.show()

    return output

def shoreline_analysis(output, settings):
    """
    Analyze the shorelines and compute cross-shore distances along transects.
    Returns the cross_distance dictionary and transects dictionary.
    """
    # If you have already mapped the shorelines, load the output.pkl file
    filepath = os.path.join(settings['inputs']['filepath'], settings['inputs']['sitename'])
    with open(os.path.join(filepath, settings['inputs']['sitename'] + '_output' + '.pkl'), 'rb') as f:
        output = pickle.load(f)
    # Remove duplicates (images taken on the same date by the same satellite)
    output = SDS_tools.remove_duplicates(output)
    # Remove inaccurate georeferencing (set threshold to 10 m)
    output = SDS_tools.remove_inaccurate_georef(output, 10)

    # Now we have to define cross-shore transects over which to quantify the shoreline changes
    # Each transect is defined by two points, its origin and a second point that defines its orientation

    # TODO: Set transects to be drawn
    # Option 1: Draw origin of transect first and then a second point to define the orientation
    # transects = SDS_transects.draw_transects(output, settings)

    # Option 2: Retrieve transects from geojson file
    geojson_transects = os.path.join(
        r'C:\Users\psteeves\coastal\planetscope_coastsat\user_inputs\transects', f"{settings['inputs']['sitename']}_transects.geojson"
    )
    # C:\Users\psteeves\coastal\planetscope_coastsat\user_inputs\transects
    # transects = SDS_tools.transects_from_geojson()
    transects = SDS_tools.transects_from_geojson(geojson_transects)

    # Plot the transects to make sure they are correct (origin landwards!)
    if settings.get('save_figure', False):
        fig = plt.figure(figsize=[15, 8], tight_layout=True)
        plt.axis('equal')
        plt.xlabel('Eastings')
        plt.ylabel('Northings')
        plt.grid(linestyle=':', color='0.5')
        for i in range(len(output['shorelines'])):
            sl = output['shorelines'][i]
            date = output['dates'][i]
            plt.plot(sl[:, 0], sl[:, 1], '.', label=date.strftime('%d-%m-%Y'))
        for i, key in enumerate(list(transects.keys())):
            plt.plot(transects[key][0, 0], transects[key][0, 1], 'bo', ms=5)
            plt.plot(transects[key][:, 0], transects[key][:, 1], 'k-', lw=1)
            plt.text(
                transects[key][0, 0] - 100,
                transects[key][0, 1] + 100,
                key,
                va='center',
                ha='right',
                bbox=dict(boxstyle='square', ec='k', fc='w'),
            )
        fig.savefig(
            os.path.join(settings['inputs']['filepath'], settings['inputs']['sitename'], 'mapped_shorelines_with_transects.jpg'),
            dpi=200,
        )
        # Optionally, display the plot
        # plt.show()

    # Compute intersections with quality-control parameters
    settings_transects = {  # Parameters for computing intersections
        'along_dist': 25,  # Along-shore distance to use for computing the intersection
        'min_points': 3,  # Minimum number of shoreline points to calculate an intersection
        'max_std': 15,  # Max std for points around transect
        'max_range': 30,  # Max range for points around transect
        'min_chainage': -100,  # Largest negative value along transect (landwards of transect origin)
        'multiple_inter': 'max',  # Mode for removing outliers ('auto', 'nan', 'max')
        'auto_prc': 0.1,  # Percentage to use in 'auto' mode to switch from 'nan' to 'max'
    }
    cross_distance = SDS_transects.compute_intersection_QC(
        output, transects, settings_transects
    )

    # Plot the time-series of cross-shore shoreline change
    if settings.get('save_figure', False):
        fig = plt.figure(figsize=[15, 8], tight_layout=True)
        gs = gridspec.GridSpec(len(cross_distance), 1)
        gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.05)
        for i, key in enumerate(cross_distance.keys()):
            if np.all(np.isnan(cross_distance[key])):
                continue
            ax = fig.add_subplot(gs[i, 0])
            ax.grid(linestyle=':', color='0.5')
            ax.set_ylim([-50, 50])
            ax.plot(
                output['dates'],
                cross_distance[key] - np.nanmedian(cross_distance[key]),
                '-o',
                ms=4,
                mfc='w',
            )
            ax.set_ylabel('distance [m]', fontsize=12)
            ax.text(
                0.5,
                0.95,
                key,
                bbox=dict(boxstyle='square', ec='k', fc='w'),
                ha='center',
                va='top',
                transform=ax.transAxes,
                fontsize=14,
            )
        fig.savefig(
            os.path.join(settings['inputs']['filepath'], settings['inputs']['sitename'], 'time_series_raw.jpg'), dpi=200
        )
        # Optionally, display the plot
        # plt.show()

    # Save time-series in a .csv file
    out_dict = dict([])
    out_dict['dates'] = output['dates']
    for key in transects.keys():
        out_dict['Transect ' + key] = cross_distance[key]
    df = pd.DataFrame(out_dict)
    fn = os.path.join(
        settings['inputs']['filepath'],
        settings['inputs']['sitename'],
        'transect_time_series.csv',
    )
    df.to_csv(fn, sep=',')
    print('Time-series of the shoreline change along the transects saved as:\n%s' % fn)

    return cross_distance, transects


def tidal_correction(output, cross_distance, transects, settings, slope_est, dates_sat, tides_sat):
    # Tidal correction along each transect
    reference_elevation = 0
    cross_distance_tidally_corrected = {}
    for key in cross_distance.keys():
        transect_slope = slope_est[key]  # Use the specific slope for each transect
        correction = (tides_sat - reference_elevation) / transect_slope
        cross_distance_tidally_corrected[key] = cross_distance[key] + correction

    # Store the tidally-corrected time-series in a .csv file
    out_dict = dict([])
    out_dict['dates'] = dates_sat
    for key in cross_distance_tidally_corrected.keys():
        out_dict[key] = cross_distance_tidally_corrected[key]
    df = pd.DataFrame(out_dict)
    fn = os.path.join(
        settings['inputs']['filepath'],
        settings['inputs']['sitename'],
        'transect_time_series_tidally_corrected.csv',
    )
    df.to_csv(fn, sep=',')
    print(
        'Tidally-corrected time-series of the shoreline change along the transects saved as:\n%s'
        % fn
    )

    # Plot the time-series of shoreline change (both raw and tidally-corrected)
    if settings.get('save_figure', False):
        fig = plt.figure(figsize=[15, 8], tight_layout=True)
        gs = gridspec.GridSpec(len(cross_distance), 1)
        gs.update(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.05)
        for i, key in enumerate(cross_distance.keys()):
            if np.all(np.isnan(cross_distance[key])):
                continue
            ax = fig.add_subplot(gs[i, 0])
            ax.grid(linestyle=':', color='0.5')
            ax.set_ylim([-50, 50])
            ax.plot(
                output['dates'],
                cross_distance[key] - np.nanmedian(cross_distance[key]),
                '-o',
                ms=6,
                mfc='w',
                label='raw',
            )
            ax.plot(
                output['dates'],
                cross_distance_tidally_corrected[key] - np.nanmedian(cross_distance[key]),
                '-o',
                ms=6,
                mfc='w',
                label='tidally-corrected',
            )
            ax.set_ylabel('distance [m]', fontsize=12)
            ax.text(
                0.5,
                0.95,
                key,
                bbox=dict(boxstyle='square', ec='k', fc='w'),
                ha='center',
                va='top',
                transform=ax.transAxes,
                fontsize=14,
            )
            ax.legend()
        fig.savefig(
            os.path.join(settings['inputs']['filepath'], settings['inputs']['sitename'], 'time_series_tidally_corrected.jpg'), dpi=200
        )
        # Optionally, display the plot
        # plt.show()

    return cross_distance_tidally_corrected


def time_series_post_processing(transects, settings, cross_distance_tidally_corrected):
    """
    Post-process the time series data.
    """
    sitename = settings['inputs']['sitename']
    filename_output = os.path.join(os.getcwd(), 'data', sitename, f'{sitename}_output.pkl')
    with open(filename_output, 'rb') as f:
        output = pickle.load(f)

    # Plot the mapped shorelines
    if settings.get('save_figure', False):
        fig = plt.figure(figsize=[15, 8], tight_layout=True)
        plt.axis('equal')
        plt.xlabel('Eastings')
        plt.ylabel('Northings')
        plt.grid(linestyle=':', color='0.5')
        plt.title(f"{len(output['shorelines'])} shorelines mapped at {sitename} from 2014")
        for i in range(len(output['shorelines'])):
            sl = output['shorelines'][i]
            date = output['dates'][i]
            plt.plot(sl[:, 0], sl[:, 1], '.', label=date.strftime('%d-%m-%Y'))
        for i, key in enumerate(list(transects.keys())):
            plt.plot(transects[key][0, 0], transects[key][0, 1], 'bo', ms=5)
            plt.plot(transects[key][:, 0], transects[key][:, 1], 'k-', lw=1)
            plt.text(
                transects[key][0, 0] - 100,
                transects[key][0, 1] + 100,
                key,
                va='center',
                ha='right',
                bbox=dict(boxstyle='square', ec='k', fc='w'),
            )

    # Load the tidally-corrected time-series
    filepath = os.path.join(
        settings['inputs']['filepath'],
        settings['inputs']['sitename'],
        'transect_time_series_tidally_corrected.csv'
    )
    df = pd.read_csv(filepath, parse_dates=['dates'])
    dates = [_.to_pydatetime() for _ in df['dates']]
    cross_distance = cross_distance_tidally_corrected

    # Remove outliers
    # Plot Otsu thresholds for the mapped shorelines
    if settings.get('save_figure', False):
        fig, ax = plt.subplots(1, 1, figsize=[12, 5], tight_layout=True)
        ax.grid(which='major', ls=':', lw=0.5, c='0.5')
        ax.plot(output['dates'], output['MNDWI_threshold'], 'o-', mfc='w')
        ax.axhline(y=-0.5, ls='--', c='r', label='otsu_threshold limits')
        ax.axhline(y=0, ls='--', c='r')
        ax.set(
            title='Otsu thresholds on MNDWI for the %d shorelines mapped'
            % len(output['shorelines']),
            ylim=[-0.6, 0.2],
            ylabel='otsu threshold',
        )
        ax.legend(loc='upper left')
        fig.savefig(os.path.join(settings['inputs']['filepath'], settings['inputs']['sitename'], 'otsu_thresholds.jpg'))
        # Optionally, display the plot
        # plt.show()

    # Remove outliers in the time-series (despiking)
    settings_outliers = {
        'otsu_threshold': [
            -0.5,
            0,
        ],  # Min and max intensity threshold used for contouring the shoreline
        'max_cross_change': 40,  # Maximum cross-shore change observable between consecutive timesteps
        'plot_fig': True,  # Whether to plot the intermediate steps
    }
    cross_distance = SDS_transects.reject_outliers(
        cross_distance, output, settings_outliers
    )

    # Seasonal averaging
    # Compute seasonal averages along each transect
    season_colors = {'DJF': 'C3', 'MAM': 'C1', 'JJA': 'C2', 'SON': 'C0'}
    for key in cross_distance.keys():
        chainage = cross_distance[key]
        # Remove NaNs
        idx_nan = np.isnan(chainage)
        dates_nonan = [dates[_] for _ in np.where(~idx_nan)[0]]
        chainage = chainage[~idx_nan]

        # Compute shoreline seasonal averages (DJF, MAM, JJA, SON)
        dict_seas, dates_seas, chainage_seas, list_seas = SDS_transects.seasonal_average(
            dates_nonan, chainage
        )

        trend, y = SDS_transects.calculate_trend(dates_seas, chainage_seas)

        # Plot seasonal averages
        if settings.get('save_figure', False):
            fig, ax = plt.subplots(1, 1, figsize=[14, 4], tight_layout=True)
            ax.grid(which='major', linestyle=':', color='0.5')
            ax.set_title('Time-series at %s' % key, x=0, ha='left')
            ax.set(ylabel='distance [m]')
            ax.plot(
                dates_nonan,
                chainage,
                '+',
                lw=1,
                color='k',
                mfc='w',
                ms=4,
                alpha=0.5,
                label='raw datapoints',
            )
            ax.plot(
                dates_seas,
                chainage_seas,
                '-',
                lw=1,
                color='k',
                mfc='w',
                ms=4,
                label='seasonally-averaged',
            )
            for k, seas in enumerate(dict_seas.keys()):
                ax.plot(
                    dict_seas[seas]['dates'],
                    dict_seas[seas]['chainages'],
                    'o',
                    mec='k',
                    color=season_colors[seas],
                    label=seas,
                    ms=5,
                )
            ax.plot(dates_seas,y,'--',color='b', label='trend %.1f m/year'%trend)
            ax.legend(
                loc='lower left',
                ncol=6,
                markerscale=1.5,
                frameon=True,
                edgecolor='k',
                columnspacing=1,
            )
            fig.savefig(os.path.join(settings['inputs']['filepath'], settings['inputs']['sitename'], f'{key}_seasonal_average.jpg'))
            # Optionally, display the plot
            # plt.show()

    # Monthly averaging
    # Compute monthly averages along each transect
    month_colors = plt.get_cmap('tab20')
    for key in cross_distance.keys():
        chainage = cross_distance[key]
        # Remove NaNs
        idx_nan = np.isnan(chainage)
        dates_nonan = [dates[_] for _ in np.where(~idx_nan)[0]]
        chainage = chainage[~idx_nan]

        # Compute shoreline monthly averages
        dict_month, dates_month, chainage_month, list_month = SDS_transects.monthly_average(
            dates_nonan, chainage
        )

        # Plot monthly averages
        if settings.get('save_figure', False):
            fig, ax = plt.subplots(1, 1, figsize=[14, 4], tight_layout=True)
            ax.grid(which='major', linestyle=':', color='0.5')
            ax.set_title('Time-series at %s' % key, x=0, ha='left')
            ax.set(ylabel='distance [m]')
            ax.plot(
                dates_nonan,
                chainage,
                '+',
                lw=1,
                color='k',
                mfc='w',
                ms=4,
                alpha=0.5,
                label='raw datapoints',
            )
            ax.plot(
                dates_month,
                chainage_month,
                '-',
                lw=1,
                color='k',
                mfc='w',
                ms=4,
                label='monthly-averaged',
            )
            for k, month in enumerate(dict_month.keys()):
                ax.plot(
                    dict_month[month]['dates'],
                    dict_month[month]['chainages'],
                    'o',
                    mec='k',
                    color=month_colors(k),
                    label=month,
                    ms=5,
                )
            ax.legend(
                loc='lower left',
                ncol=7,
                markerscale=1.5,
                frameon=True,
                edgecolor='k',
                columnspacing=1,
            )
            fig.savefig(os.path.join(settings['inputs']['filepath'], settings['inputs']['sitename'], f'{key}_monthly_average.jpg'))
            # Optionally, display the plot
            # plt.show()
    return cross_distance

def slope_estimation(settings, cross_distance, output):
    fp_slopes = os.path.join(settings['inputs']['filepath'], settings['inputs']['sitename'], 'slope_estimation')
    if not os.path.exists(fp_slopes):
        os.makedirs(fp_slopes)
    print(f'Outputs will be saved in {fp_slopes}')

    # Load FES2022 configuration for tide calculation
    print("Loading FES2022 config file...")
    config_filepath = os.pardir
    config = os.path.join(config_filepath, 'fes2022.yaml')
    handlers = pyfes.load_config(config)
    print("Config file loaded")
    ocean_tide = handlers['tide']
    load_tide = handlers['radial']

    # Calculate tides at centroid
    polygon = settings['inputs']['polygon']
    centroid = SDS_preprocess.get_point_from_geojson(settings['inputs']['sitename'], os.path.join(r'U:', 'tide_points'))
    centroid[0] = centroid[0] + 360 if centroid[0] < 0 else centroid[0]

    # Retrieve tide levels at image acquisition times
    # Compute tide levels '2020-07-30', '2024-10-23
    date_range = [
        pytz.utc.localize(datetime(2020, 7, 30)),
        pytz.utc.localize(datetime(2024, 10, 23)),
    ]
    timestep = 900  # seconds
    dates_ts, tides_ts = SDS_slope.compute_tide(
        centroid, date_range, timestep, ocean_tide, load_tide
    )

    # Compute tide levels at image acquisition times
    dates_sat = output['dates']
    tides_sat = SDS_slope.compute_tide_dates(
        centroid, output['dates'], ocean_tide, load_tide
    )

    # Plot tide levels at image acquisition times
    if settings.get('save_figure', False):
        fig, ax = plt.subplots(1, 1, figsize=(15, 4), tight_layout=True)
        ax.grid(which='major', linestyle=':', color='0.5')
        ax.plot(dates_ts, tides_ts, '-', color='0.6', label='all time-series')
        ax.plot(
            dates_sat,
            tides_sat,
            '-o',
            color='k',
            ms=6,
            mfc='w',
            lw=1,
            label='image acquisition',
        )
        ax.set(
            ylabel='tide level [m]',
            xlim=[dates_sat[0], dates_sat[-1]],
            title='Tide levels at the time of image acquisition',
        )
        ax.legend()
        fig.savefig(os.path.join(settings['inputs']['filepath'], settings['inputs']['sitename'], '%s_tide_timeseries.jpg' % settings['inputs']['sitename']), dpi=200)
        # Optionally, display the plot
        # plt.show()

    days_in_year = 365
    seconds_in_day = 24 * 3600
    settings_slope = {'slope_min':      0.035,                  # minimum slope to trial
                'slope_max':        0.2,                    # maximum slope to trial
                'delta_slope':      0.005,                  # slope increment
                'n0':               50,                     # parameter for Nyquist criterium in Lomb-Scargle transforms
                'freq_cutoff':     1./(seconds_in_day*30), # 1 month frequency
                'delta_f':          100*1e-10,              # deltaf for identifying peak tidal frequency band
                'prc_conf':         0.05,                   # percentage above minimum to define confidence bands in energy curve
                'plot_fig':         True,                   # whether to plot the intermediary products during analysis
                }
    # Set parameters for slope range and date range
    beach_slopes = SDS_slope.range_slopes(settings_slope['slope_min'], settings_slope['slope_max'], settings_slope['delta_slope'])

    settings_slope['date_range'] = [2020,2024]
    # re-write in datetime objects (same as shoreline in UTC)
    settings_slope['date_range'] = [pytz.utc.localize(datetime(settings_slope['date_range'][0],5,1)),
                                    pytz.utc.localize(datetime(settings_slope['date_range'][1],1,1))]
    # clip the time-series between 1999 and 2022 as we need at least 2 Landsat satellites 
    idx_dates = [np.logical_and(_>settings_slope['date_range'][0],_<settings_slope['date_range'][1]) for _ in output['dates']]
    dates_sat = [output['dates'][_] for _ in np.where(idx_dates)[0]]
    tides_sat = tides_sat[np.where(idx_dates)[0]]
    for key in cross_distance.keys():
        cross_distance[key] = cross_distance[key][idx_dates]

    SDS_slope.plot_timestep(dates_sat)

    plt.gcf().savefig(os.path.join(fp_slopes,'0_timestep_distribution.jpg'),dpi=200)

    settings_slope['n_days'] = 8

    settings_slope['freqs_max'] = SDS_slope.find_tide_peak(dates_sat,tides_sat,settings_slope)

    plt.gcf().savefig(os.path.join(fp_slopes,'1_tides_power_spectrum.jpg'),dpi=200)

    # Dictionary to store the slope estimates per transect
    slope_est, cis = dict([]), dict([])

    for key in cross_distance.keys():
        # Remove NaNs for the current transect
        idx_nan = np.isnan(cross_distance[key])
        dates = [dates_sat[_] for _ in np.where(~idx_nan)[0]]
        tide = tides_sat[~idx_nan]
        composite = cross_distance[key][~idx_nan]

        # Apply tidal correction and estimate slope for the current transect
        tsall = SDS_slope.tide_correct(composite, tide, beach_slopes)
        slope_est[key],cis[key] = SDS_slope.integrate_power_spectrum(dates,tsall,settings_slope, key)
        plt.gcf().savefig(os.path.join(fp_slopes,'2_energy_curve_%s.jpg'%key),dpi=200)
        # plot spectrums
        SDS_slope.plot_spectrum_all(dates,composite,tsall,settings_slope,slope_est[key])
        plt.gcf().savefig(os.path.join(fp_slopes,'3_slope_spectrum_%s.jpg'%key),dpi=200)
        print('Beach slope at transect %s: %.3f (%.4f - %.4f)'%(key, slope_est[key], cis[key][0], cis[key][1]))

    # Return slope estimates, dates, and tide levels at acquisition times
    return slope_est, dates_sat, tides_sat


def main():
    inputs, settings = initial_settings()
    metadata = retrieve_images(inputs)
    output = batch_shoreline_detection(metadata, settings, inputs)
    cross_distance, transects = shoreline_analysis(output, settings)

    # Estimate slopes and retrieve tide data for tidal correction
    slope_est, dates_sat, tides_sat = slope_estimation(settings, cross_distance, output)
    cross_distance_tidally_corrected = tidal_correction(output, cross_distance, transects, settings, slope_est, dates_sat, tides_sat)
    cross_distance = time_series_post_processing(transects, settings, cross_distance_tidally_corrected)

    # plt.show()


if __name__ == '__main__':
    main()
