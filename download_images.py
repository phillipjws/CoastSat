import os
import time
from coastsat import SDS_download, SDS_tools

def initial_settings(sitename):
    """
    Initial settings for the shoreline extraction.
    Returns the inputs dictionary and settings dictionary.
    """
    geojson_polygon = os.path.join(r'D:\Inputs', f'{sitename}.kml')
    polygon = SDS_tools.polygon_from_kml(geojson_polygon)
    polygon = SDS_tools.smallest_rectangle(polygon)

    # Date range
    dates = ['1984-01-01', '2025-01-01']

    # Satellites
    sat_list = ['L5', 'L7', 'L8', 'L9', 'S2']

    # Filepath where data will be stored
    filepath_data = os.path.join(r'D:\coastsat_data')

    # Put all the inputs into a dictionary
    inputs = {
        'polygon': polygon,
        'dates': dates,
        'sat_list': sat_list,
        'sitename': sitename,
        'filepath': filepath_data,
        # 'excluded_epsg_codes': ['32608'],
        # 'LandsatWRS': '054022',
        'S2tile': '09UVA',
        # 'months': [7, 8, 9, 10],
        # 'skip_L7_SLC': True
    }

    # Retrieve satellite images from GEE
    metadata = SDS_download.retrieve_images(inputs)
    metadata = SDS_download.get_metadata(inputs)


if __name__ == "__main__":
    dir = r'D:\Inputs'
    log_file = r'D:\Inputs\process_log.txt'

    with open(log_file, 'w') as log:
        log.write("Processing started\n")
        log.write(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write("-" * 40 + "\n")

    for file in os.listdir(dir):
        if file.endswith('.kml') and file.startswith(('METLAKATLA_2', 'METLAKATLA_1')):
            filename = os.path.splitext(file)[0]
            try:
                start_time = time.time()  # Start the timer
                initial_settings(filename)
                end_time = time.time()  # End the timer
                elapsed_time = end_time - start_time

                message = f"Sitename: {filename} completed in {elapsed_time:.2f} seconds\n"
                print(message)

                with open(log_file, 'a') as log:
                    log.write(message)

            except Exception as e:
                error_message = f"Sitename: {filename} failed with exception: {e}\n"
                print(error_message)

                with open(log_file, 'a') as log:
                    log.write(error_message)

    with open(log_file, 'a') as log:
        log.write("-" * 40 + "\n")
        log.write(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write("Processing finished\n")
